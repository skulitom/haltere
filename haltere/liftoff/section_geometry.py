"""Offline collision-geometry ray casting for generated-course validation.

This module is not a runtime sensor. Box dimensions come from inspected game
colliders; rendered depth still needs camera/surface alignment checks. Flags and
other unmodelled meshes remain explicit unknowns, never labelled free space.
"""
from __future__ import annotations

import numpy as np

from ..vision.camera import Camera, quat_wxyz_to_mat
from .frames import sim_vec_to_unity


def ray_box(origin, rays, center, size, yaw_deg=0.):
    """Unit world rays against an oriented box; return first occupied distance."""
    origin, rays = np.asarray(origin, dtype=float), np.asarray(rays, dtype=float)
    center, size = np.asarray(center, dtype=float), np.asarray(size, dtype=float)
    if origin.shape != (3,) or center.shape != (3,) or size.shape != (3,) or rays.shape[-1] != 3:
        raise ValueError('Expected 3-D origins, dimensions and rays')
    if (not all(np.isfinite(x).all() for x in [origin, rays, center, size])
            or not np.isfinite(yaw_deg) or (size <= 0).any()
            or not np.allclose(np.linalg.norm(rays, axis=-1), 1., atol=1e-5)):
        raise ValueError('Use finite geometry, positive dimensions and unit rays')
    yaw = np.deg2rad(yaw_deg)
    rotation = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    local_origin, local_rays = (origin-center)@rotation, rays@rotation
    parallel = abs(local_rays) < 1e-12
    lower = np.divide(-size/2-local_origin, local_rays, out=np.zeros_like(local_rays), where=~parallel)
    upper = np.divide(size/2-local_origin, local_rays, out=np.zeros_like(local_rays), where=~parallel)
    near, far = np.minimum(lower, upper), np.maximum(lower, upper)
    outside = abs(local_origin) > size/2
    near = np.where(parallel, np.where(outside, np.inf, -np.inf), near)
    far = np.where(parallel, np.where(outside, -np.inf, np.inf), far)
    enter, leave = np.maximum(near.max(axis=-1), 0.), far.min(axis=-1)
    return np.where(leave >= enter, enter, np.inf)


def collision_depth(geometry, origin, unit_rays, max_range=80.):
    """Compute a model-only range map; absence of a hit is not observed free space.

    Refuse incomplete scenes so a missing flag cannot become a false safe label.
    For box-only calibration, use an explicitly separate scene contract. Never
    remove unknown objects from a measured course merely to pass this check.
    """
    if geometry.get('runtime_geometry_allowed') is not False:
        raise ValueError('Expected an explicitly offline geometry contract')
    if geometry.get('unknown_geometry'):
        raise ValueError('Unmodelled meshes prevent complete depth labels for this course')
    if not np.isfinite(max_range) or max_range <= 0:
        raise ValueError('Use a finite positive ray range')
    rays = np.asarray(unit_rays, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if (origin.shape != (3,) or rays.ndim < 2 or rays.shape[-1] != 3
            or not np.isfinite(origin).all() or not np.isfinite(rays).all()
            or not np.allclose(np.linalg.norm(rays, axis=-1), 1., atol=1e-5)):
        raise ValueError('Use a finite origin and array of unit rays')
    distance = np.full(rays.shape[:-1], np.inf)
    instance = np.full(rays.shape[:-1], -1, dtype=int)
    for obj in geometry['primitives']:
        if obj['kind'] != 'box':
            raise ValueError('Only verified box collider geometry is currently supported')
        hit = ray_box(origin, rays, obj['center'], obj['size'], obj['yaw_deg'])
        closer = hit < distance
        instance[closer] = obj['instance_id']
        distance = np.minimum(distance, hit)
    if 'ground_plane_y' in geometry:
        down = rays[..., 1] < -1e-12
        ground = np.divide(geometry['ground_plane_y']-origin[1], rays[..., 1],
                           out=np.full_like(distance, np.inf), where=down)
        closer = (ground >= 0) & (ground < distance)
        instance[closer] = 0
        distance = np.where(closer, ground, distance)
    distance = np.where(distance <= max_range, distance, np.inf)
    instance[~np.isfinite(distance)] = -1
    return dict(range_m=distance, instance_id=instance, kind='offline collider range',
                rendered_depth_validated=bool(geometry.get('render_alignment_verified', False)),
                no_hit_means='outside the model range, not established free space')


def camera_collision_depth(geometry, camera: Camera, position_sim, quaternion_wxyz,
                           *, camera_offset_body=(0., 0., 0.), max_range=80.):
    """Project the offline collision model through a logged, calibrated camera.

    Position is absolute simulator-world FLU, including the log's origin offset.
    Pixel centers and the body's camera offset are explicit. Return both ray
    range and optical-axis depth: metric-depth models may predict the latter.
    This does not establish that collider boundaries match rendered surfaces.
    """
    position = np.asarray(position_sim, dtype=float)
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    offset = np.asarray(camera_offset_body, dtype=float)
    if (position.shape != (3,) or offset.shape != (3,) or quaternion.shape != (4,)
            or not all(np.isfinite(x).all() for x in [position, quaternion, offset])
            or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-5)):
        raise ValueError('Use finite absolute position, body offset and unit wxyz quaternion')
    if (camera.width < 1 or camera.height < 1 or not np.isfinite(camera.f)
            or camera.f <= 0 or not np.isfinite(camera.tilt_deg)):
        raise ValueError('Use a finite calibrated camera with positive size and focal length')
    yy, xx = np.mgrid[:camera.height, :camera.width]
    pixels = np.column_stack([xx.ravel()+.5, yy.ravel()+.5])
    body_rays = camera.unproject_body(pixels)
    rotation = quat_wxyz_to_mat(quaternion)
    origin = sim_vec_to_unity(position+rotation@offset)
    rays = sim_vec_to_unity(body_rays@rotation.T).reshape(camera.height, camera.width, 3)
    result = collision_depth(geometry, origin, rays, max_range)
    axis_cosine = (body_rays@camera.body_to_cam().T)[:, 2].reshape(camera.height, camera.width)
    result['optical_depth_m'] = result['range_m']*axis_cosine
    result['camera_origin_unity'] = origin
    result['pixel_convention'] = 'pixel centers at integer + 0.5'
    return result
