"""Isolated camera/trajectory worker; passive unless explicitly given a proposal channel.

Only causal images and a bounded shared history of live motion/task goals cross
the process boundary. This worker never opens a pad or reads a route. The shadow
class has no feedback channel. Explicit experimental control uses a separate
class and a validated proposal boundary. Capture and JSONL writes stay off control.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from ..vision.camera import Camera
from ..vision.geometry_mask import liftoff_geometry_mask
from ..vision.local_trajectory import LocalTrajectoryPlanner
from ..vision.surface_memory import SurfaceMemory
from ..vision.temporal_depth import MultiBaselineDepth
from .camera_pose import CameraPoseHistory


class MotionBuffer:
    """One writer, one reader; neither can wait for the other's lock."""
    width = 16  # available time, pose receipt, game time, position, quaternion, velocity, task velocity

    def __init__(self, capacity=256):
        context = mp.get_context('spawn')
        self.capacity = capacity
        self.data = context.RawArray('d', capacity*self.width)
        self.count = context.RawValue('q', 0)
        self.lock = context.Lock()

    def publish(self, available_at, pose_time, game_time, position, quaternion, velocity, requested):
        row = np.asarray([available_at, pose_time, game_time, *position, *quaternion, *velocity, *requested], float)
        if row.shape != (self.width,) or not np.isfinite(row).all() or pose_time > available_at:
            raise ValueError('Use finite, already observed motion')
        if not self.lock.acquire(False):
            return False
        try:
            data = np.frombuffer(self.data).reshape(self.capacity,self.width)
            data[self.count.value % self.capacity] = row
            self.count.value += 1
        finally:
            self.lock.release()
        return True

    def read(self):
        if not self.lock.acquire(False):
            return None
        try:
            count = self.count.value
            indices = np.arange(max(0,count-self.capacity),count) % self.capacity
            return np.frombuffer(self.data).reshape(self.capacity,self.width)[indices].copy()
        finally:
            self.lock.release()


class ShadowGeometry:
    def __init__(self, camera):
        self.camera = camera
        self.tracker = MultiBaselineDepth(camera)
        # Liftoff scenery is stationary in the observed telemetry frame. Braking
        # removes the translation needed for fresh depth; it must not make a
        # nearby observed obstacle disappear merely because three seconds pass.
        self.memory = SurfaceMemory(lifetime=None,max_points=512)
        self.planner = LocalTrajectoryPlanner()
        self.last_game_time = None

    def observe(self, rgb, captured_at, motion, now):
        import cv2
        if motion is None or not len(motion):
            return dict(status='waiting_for_motion', capture_time=captured_at)
        motion = motion[motion[:,0] <= now]
        if not len(motion) or now-motion[-1,1] > .12:
            return dict(status='stale_motion', capture_time=captured_at)
        if captured_at < motion[0,1] or captured_at > motion[-1,1]+.04:
            return dict(status='image_pose_unaligned', capture_time=captured_at)
        latest = motion[-1]
        if self.last_game_time is not None and latest[2] < self.last_game_time-.1:
            raise ValueError('Game reset; stop this shadow attempt')
        self.last_game_time = latest[2]
        history = CameraPoseHistory()
        for row in motion:
            if not history.samples or row[1] > history.samples[-1][0]:
                history.append(row[1],row[3:6],row[6:10])
        position, quaternion = history.at(captured_at)
        rgb = cv2.resize(rgb,(self.camera.width,self.camera.height),interpolation=cv2.INTER_AREA)
        start = time.monotonic()
        result = self.tracker.update(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),position,quaternion,
                                     captured_at,liftoff_geometry_mask(rgb))
        points,sigma = np.empty((0,3)),np.empty(0)
        long_baseline_points = 0
        if result is not None:
            good = result['valid'] & (result['range_sigma_m'] < .25*result['range_m'])
            points,sigma = result['position_world'][good],result['range_sigma_m'][good]
            long_baseline_points = int(result['long_baseline'][good].sum())
        tracked = time.monotonic()
        surfaces = self.memory.update(latest[3:6],captured_at,points,sigma)
        # Compute against current motion and retain original depth receipt times.
        # The map query timestamp is current; it does not imply fresh depth.
        proposal = self.planner.propose(latest[3:6],latest[10:13],latest[13:16],surfaces,
                                        time.monotonic(),view=(self.camera,position,quaternion))
        finished = time.monotonic()
        finite = lambda value: float(value) if value is not None and np.isfinite(value) else None
        return dict(status=proposal['status'],capture_time=captured_at,game_time=float(latest[2]),
                    requested_goal_time=float(latest[0]),position=latest[3:6].tolist(),
                    camera_position=np.asarray(position).tolist(),camera_quaternion=np.asarray(quaternion).tolist(),
                    velocity=latest[10:13].tolist(),accepted_points=np.column_stack((points,sigma)).tolist(),
                    pose_extrapolation_s=max(0.,captured_at-motion[-1,1]),
                    input_age_s=now-latest[1],proposal_age_s=finished-captured_at,
                    tracking_ms=1000*(tracked-start),planning_and_surfaces_ms=1000*(finished-tracked),
                    valid_points=len(points),memory_points=len(surfaces['points']),triangles=len(surfaces['triangles']),
                    long_baseline_points=long_baseline_points,
                    oldest_observation_age_s=surfaces['oldest_observation_age_s'],
                    memory_capacity_evictions=surfaces['capacity_evictions'],
                    patch_support_points=surfaces['patch_support_points'],
                    requested_velocity=latest[13:16].tolist(),proposal_velocity=proposal['velocity'].tolist(),
                    nominal_margin_m=finite(proposal['nominal_margin_m']),
                    selected_margin_m=finite(proposal['selected_margin_m']),
                    changed=proposal['changed'],coverage_certified=False,live_authority=False)


def shadow_worker(buffer, done, path, camera_config, source_route_oracle, fps, proposals=None, archive_images=False):
    import cv2
    import mss
    from .recorder import _capture_game_frame
    cv2.setNumThreads(2)
    diagnostic = ShadowGeometry(Camera(**camera_config))
    counts, timings = Counter(), []
    error = None
    path = Path(path)
    try:
        image_folder=path.with_suffix('') if archive_images else None
        if image_folder is not None:
            image_folder.mkdir(exist_ok=False)
        with path.open('x') as file, getattr(mss,'MSS',mss.mss)() as screen:
            while not done.is_set():
                begin = time.monotonic()
                rgb = _capture_game_frame(screen,'Liftoff')
                captured = time.monotonic()
                if rgb is None:
                    row = dict(status='no_game_image',capture_time=begin)
                else:
                    rgb=cv2.resize(rgb,(camera_config['width'],camera_config['height']),interpolation=cv2.INTER_AREA)
                    row = diagnostic.observe(rgb,begin,buffer.read(),captured)
                if proposals is not None:
                    proposals.publish(row,time.monotonic())
                    row['live_authority']=True
                if image_folder is not None and rgb is not None:
                    filename=f'{sum(counts.values()):06d}.png'
                    if not cv2.imwrite(str(image_folder/filename),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_PNG_COMPRESSION,3]):
                        raise RuntimeError('Could not preserve geometry input image')
                    row['image_file']=filename
                row['capture_ms'] = 1000*(captured-begin)
                row['total_ms'] = 1000*(time.monotonic()-begin)
                counts[row['status']] += 1
                timings.append(row['total_ms'])
                file.write(json.dumps(row,allow_nan=False)+'\n')
                file.flush()
                done.wait(max(0.,1/fps-(time.monotonic()-begin)))
    except Exception as exc:
        error = repr(exc)
    finally:
        result = dict(mode='experimental visual geometry proposals' if proposals is not None else 'passive geometry and trajectory shadow',
                      live_authority=proposals is not None,
                      runtime_course_geometry=False,source_flight_runtime_route_oracle=source_route_oracle,
                      camera=camera_config,planner_config=asdict(diagnostic.planner.config),
                      depth_tracking=diagnostic.tracker.metadata(),
                      surface_memory=diagnostic.memory.metadata(),
                      fps=fps,frames=sum(counts.values()),status_counts=dict(counts),error=error,
                      images_preserved=archive_images,
                      total_ms=dict(p50=float(np.median(timings)),p95=float(np.percentile(timings,95)),
                                    maximum=float(max(timings))) if timings else {},
                      limits='Independent MSS receipt alignment; physical display delay unmeasured. Sparse geometry and point-mass model remain unqualified for steering.')
        path.with_suffix('.json').write_text(json.dumps(result,indent=2))


class _ProcessGeometry:
    def __init__(self, path, sensor, *, source_route_oracle=False, fps=5., proposals=None, archive_images=False):
        self.path = Path(path)
        if (self.path.exists() or self.path.with_suffix('.json').exists()
                or archive_images and self.path.with_suffix('').exists()):
            raise FileExistsError('Use a new geometry shadow log')
        if not 1 <= fps <= 10 or not sensor:
            raise ValueError('Geometry shadow needs a calibrated camera and 1-10 fps')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        context = mp.get_context('spawn')
        self.buffer = MotionBuffer()
        self.proposals = proposals
        self._last_proposal = None
        self.done = context.Event()
        camera = dict(width=640,height=360,f=2*sensor['focal_320'],tilt_deg=sensor['tilt_deg'])
        self.process = context.Process(target=shadow_worker,
            args=(self.buffer,self.done,str(self.path),camera,source_route_oracle,fps,proposals,archive_images),daemon=True)

    def start(self):
        self.process.start()
        return self

    def stop(self):
        self.done.set()
        self.process.join(timeout=4.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.)
        summary = self.path.with_suffix('.json')
        if summary.exists():
            return dict(json.loads(summary.read_text()),path=str(self.path),exit_code=self.process.exitcode)
        return dict(path=str(self.path),live_authority=self.proposals is not None,error='Worker exited without summary',
                    exit_code=self.process.exitcode)


class ProcessGeometryShadow(_ProcessGeometry):
    """Passive by construction: there is no proposal channel to read."""
    def __init__(self,path,sensor,*,source_route_oracle=False,fps=5.,archive_images=False):
        super().__init__(path,sensor,source_route_oracle=source_route_oracle,fps=fps,archive_images=archive_images)


class ProcessGeometryControl(_ProcessGeometry):
    def __init__(self,path,sensor,*,fps=5.):
        from .geometry_control import ProposalBuffer
        super().__init__(path,sensor,fps=fps,proposals=ProposalBuffer(),archive_images=True)

    def latest(self):
        row=self.proposals.read()
        if row is not None:
            self._last_proposal=row
        # Retain the original capture/query times across lock contention; the
        # control gate still rejects the cached row once it becomes stale.
        return self._last_proposal
