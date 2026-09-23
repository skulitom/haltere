"""Causal sparse depth from tracked image features and measured camera motion.

No course geometry, future frames or learned scale is used. Sparse matches are
surface hypotheses, not a map of free space. Callers must mask HUD/overlays and
evaluate moving objects, thin structures, pose timing and uncertainty before
using these measurements for control.
"""
from __future__ import annotations

import numpy as np

from .camera import Camera, quat_wxyz_to_mat


def triangulate_motion(camera: Camera, pixels_before, pixels_now, position_before,
                       quaternion_before, position_now, quaternion_now, *,
                       camera_offset_body=(0., 0., 0.), min_parallax_deg=.5,
                       max_range=30., pixel_sigma=1., max_ray_gap=.10,
                       min_baseline=.05, min_range=.25):
    """Intersect two observed rays using metric telemetry poses in simulator FLU.

    Return one validity flag per correspondence and first-order image-noise
    range uncertainty. Parallel, behind-camera, inconsistent or distant rays
    provide no depth. Current optical depth and Euclidean range are distinct.
    """
    before, now = np.asarray(pixels_before, float), np.asarray(pixels_now, float)
    if (before.ndim != 2 or before.shape[1] != 2 or now.shape != before.shape
            or not np.isfinite(before).all() or not np.isfinite(now).all()):
        raise ValueError('Expected matching finite Nx2 image coordinates')
    positions = [np.asarray(p, float) for p in (position_before, position_now)]
    quaternions = [np.asarray(q, float) for q in (quaternion_before, quaternion_now)]
    offset = np.asarray(camera_offset_body, float)
    if (any(p.shape != (3,) or not np.isfinite(p).all() for p in positions)
            or any(q.shape != (4,) or not np.isfinite(q).all()
                   or not np.isclose(np.linalg.norm(q), 1., atol=1e-5) for q in quaternions)
            or offset.shape != (3,) or not np.isfinite(offset).all()):
        raise ValueError('Use finite positions, body offset and unit wxyz quaternions')
    if (not np.isfinite([min_parallax_deg,max_range,pixel_sigma,max_ray_gap,camera.f,min_baseline,min_range]).all()
            or not 0 < min_parallax_deg < 90
            or min(max_range,pixel_sigma,max_ray_gap,camera.f,min_baseline,min_range) <= 0
            or min_range >= max_range):
        raise ValueError('Use positive finite triangulation limits and calibrated focal length')
    rotations = [quat_wxyz_to_mat(q) for q in quaternions]
    origins = [p+r@offset for p,r in zip(positions, rotations)]
    rays = [camera.unproject_body(px)@r.T for px,r in zip([before,now],rotations)]
    a,b = rays
    cosine = np.clip(np.sum(a*b,axis=1),-1.,1.)
    denominator = np.maximum(1-cosine*cosine,1e-15)
    baseline = origins[1]-origins[0]
    da,db = a@baseline,b@baseline
    distance_before = (da-cosine*db)/denominator
    distance_now = (cosine*da-db)/denominator
    point_before = origins[0]+distance_before[:,None]*a
    point_now = origins[1]+distance_now[:,None]*b
    gap = np.linalg.norm(point_before-point_now,axis=1)
    angle = np.arccos(cosine)
    valid = ((angle >= np.deg2rad(min_parallax_deg)) & (angle < np.pi/2)
             & (np.linalg.norm(baseline) >= min_baseline)
             & (distance_before >= min_range) & (distance_now >= min_range)
             & (distance_before <= max_range) & (distance_now <= max_range)
             & (gap <= max_ray_gap))
    sigma = np.sqrt(2)*pixel_sigma*np.abs(distance_now)/(camera.f*np.sqrt(denominator))
    camera_ray = camera.unproject_body(now)@camera.body_to_cam().T
    return dict(valid=valid, position_world=point_now, range_m=distance_now,
                optical_depth_m=distance_now*camera_ray[:,2], range_sigma_m=sigma,
                parallax_deg=np.rad2deg(angle), ray_gap_m=gap,
                source='causal image correspondences and metric telemetry',
                establishes_free_space=False)


class TemporalDepth:
    """Short-lived image keyframes; update only with increasing capture times.

    Valid masks are uint8 images: nonzero allows feature detection/matching.
    The caller supplies them so task overlays never silently become obstacles.
    """
    def __init__(self, camera: Camera, *, max_age=.7, max_corners=500,
                 require_third_view=True, reprojection_limit=2.,
                 refresh_sigma_fraction=None, refresh_sigma_m=None):
        if not np.isfinite(max_age) or max_age <= 0 or max_corners < 1:
            raise ValueError('Use a positive history window and feature count')
        self.camera,self.max_age,self.max_corners = camera,max_age,max_corners
        self.reference = None
        self.last_time = None
        self.previous_tracks = None
        if not np.isfinite(reprojection_limit) or reprojection_limit <= 0:
            raise ValueError('Use a positive reprojection limit')
        self.require_third_view = require_third_view
        self.reprojection_limit = reprojection_limit
        for limit in (refresh_sigma_fraction, refresh_sigma_m):
            if limit is not None and (not np.isfinite(limit) or limit <= 0):
                raise ValueError('Use positive finite keyframe precision limits')
        self.refresh_sigma_fraction = refresh_sigma_fraction
        self.refresh_sigma_m = refresh_sigma_m

    def update(self, gray, position, quaternion, timestamp, valid_mask):
        import cv2
        position, quaternion = np.asarray(position, float), np.asarray(quaternion, float)
        if (position.shape != (3,) or quaternion.shape != (4,)
                or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-5)):
            raise ValueError('Use a finite position and unit wxyz quaternion')
        if (gray.shape != (self.camera.height,self.camera.width) or gray.dtype != np.uint8
                or valid_mask.shape != gray.shape or valid_mask.dtype != np.uint8
                or not np.isfinite(timestamp)
                or self.last_time is not None and timestamp <= self.last_time):
            raise ValueError('Use chronological uint8 grayscale images and matching masks')
        self.last_time = timestamp
        result = None
        tracks = None
        if self.reference is not None:
            old,points,old_pos,old_q,old_time = self.reference
            if points is not None and timestamp-old_time <= self.max_age:
                current,status,_ = cv2.calcOpticalFlowPyrLK(old,gray,points,None,
                    winSize=(21,21),maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
                back,back_status,_ = cv2.calcOpticalFlowPyrLK(gray,old,current,None,
                    winSize=(21,21),maxLevel=3)
                p0,p1,pback = points.reshape(-1,2),current.reshape(-1,2),back.reshape(-1,2)
                finite = np.isfinite(p1).all(axis=1)&np.isfinite(pback).all(axis=1)
                xy = np.rint(np.nan_to_num(p1,nan=-1,posinf=-1,neginf=-1)).astype(int)
                inside = (xy[:,0]>=0)&(xy[:,0]<gray.shape[1])&(xy[:,1]>=0)&(xy[:,1]<gray.shape[0])
                allowed = np.zeros(len(points),bool)
                allowed[inside] = valid_mask[xy[inside,1],xy[inside,0]]>0
                good = finite&allowed&(status.ravel()>0)&(back_status.ravel()>0)&(np.linalg.norm(pback-p0,axis=1)<1.)
                tracks = (p1.copy(),good.copy(),np.array(position,copy=True),np.array(quaternion,copy=True))
                if good.any():
                    result = triangulate_motion(self.camera,p0[good],p1[good],old_pos,old_q,position,quaternion)
                    if self.require_third_view:
                        consistent = np.zeros(good.sum(),bool)
                        if self.previous_tracks is not None:
                            last_pixels,last_good,last_pos,last_q = self.previous_tracks
                            projected,front = self.camera.project_body(
                                (result['position_world']-last_pos)@quat_wxyz_to_mat(last_q))
                            residual = np.linalg.norm(projected-last_pixels[good],axis=1)
                            consistent = last_good[good]&front&(residual<=self.reprojection_limit)
                        result['valid'] &= consistent
                    result.update(pixels=p1[good],reference_time=old_time,capture_time=timestamp,
                                  baseline_m=float(np.linalg.norm(np.asarray(position)-old_pos)),
                                  third_view_required=self.require_third_view)
        # Refresh after a usable translation, a stale keyframe or failed tracks.
        # The returned measurements always use the old and current frame only.
        precise = np.zeros(0, bool) if result is None else result['valid'].copy()
        if result is not None and self.refresh_sigma_fraction is not None:
            precise &= result['range_sigma_m'] < self.refresh_sigma_fraction*result['range_m']
        if result is not None and self.refresh_sigma_m is not None:
            precise &= result['range_sigma_m'] < self.refresh_sigma_m
        refresh = (self.reference is None or timestamp-self.reference[4] >= self.max_age
                   or precise.sum() >= 12)
        if refresh:
            points = cv2.goodFeaturesToTrack(gray,maxCorners=self.max_corners,qualityLevel=.01,
                                             minDistance=5,mask=valid_mask,blockSize=5)
            self.reference = (gray.copy(),points,np.array(position,copy=True),np.array(quaternion,copy=True),timestamp)
            self.previous_tracks = None
        else:
            self.previous_tracks = tracks
        return result


class MultiBaselineDepth:
    """Keep the fast tracker while adding precise longer-baseline observations.

    The second tracker can retain a reference for two seconds while the drone
    slows down. Its additional points require three-view consistency, uncertainty
    below 10% of range and below 0.5 m. Neither tracker fills unobserved pixels.
    """
    def __init__(self, camera: Camera):
        self.fast = TemporalDepth(camera)
        self.precise = TemporalDepth(camera, max_age=2.,
                                     refresh_sigma_fraction=.1, refresh_sigma_m=.5)

    def update(self, gray, position, quaternion, timestamp, valid_mask):
        results = []
        for tracker, strict in ((self.fast, False), (self.precise, True)):
            result = tracker.update(gray, position, quaternion, timestamp, valid_mask)
            if result is None:
                continue
            good = result['valid'].copy()
            if strict:
                good &= ((result['range_sigma_m'] < .1*result['range_m'])
                         & (result['range_sigma_m'] < .5))
            results.append((result, good, strict))
        if not results:
            return None
        keys = ('position_world', 'range_m', 'optical_depth_m', 'range_sigma_m',
                'parallax_deg', 'ray_gap_m', 'pixels')
        merged = {key: np.concatenate([r[key][good] for r, good, _ in results]) for key in keys}
        merged['long_baseline'] = np.concatenate([
            np.full(good.sum(), strict, dtype=bool) for _, good, strict in results])
        # Both keyframes can track the same corner. Keep its more precise
        # measurement, rather than counting duplicates as independent support.
        selected = {}
        for i in np.argsort(merged['range_sigma_m'], kind='stable'):
            cell = tuple(np.floor(merged['pixels'][i]/2).astype(int))
            selected.setdefault(cell, i)
        indices = np.asarray(sorted(selected.values()), dtype=int)
        merged = {key: value[indices] for key, value in merged.items()}
        return dict(**merged, valid=np.ones(len(indices), bool), capture_time=timestamp,
                    third_view_required=True, establishes_free_space=False,
                    source='causal dual-keyframe image correspondences and metric telemetry')

    def metadata(self):
        return dict(mode='dual keyframes', fast_max_age_s=self.fast.max_age,
                    precise_max_age_s=self.precise.max_age,
                    precise_max_sigma_m=self.precise.refresh_sigma_m,
                    precise_max_sigma_fraction=self.precise.refresh_sigma_fraction,
                    third_view_required=True, runtime_course_geometry=False,
                    limits='Sparse surface observations only; no free-space certificate')
