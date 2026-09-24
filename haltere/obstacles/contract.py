"""Image, camera, grid and fan geometry shared by runtime and offline code.

This module is runtime-safe: it imports only numpy and the camera model, and it
never reads labels. Label builders, the model, the evaluator and the governor all
use these definitions so that every stage agrees on pixels, rays and directions.

Frames of reference (the repository conventions, see haltere/vision/camera.py and
haltere/liftoff/frames.py):

- World: simulator FLU (x forward/north-ish, y left, z up), metres. The frame
  store keeps launch-relative positions as logged; absolute simulator
  positions add the run's ``origin_sim`` (store runs.json). Box colliders need
  absolute positions (haltere/liftoff/section_geometry.camera_collision_depth).
- Body: FLU (x forward, y left, z up). Attitude quaternions are wxyz and
  world-from-body: ``world = R(q) @ body``.
- Camera: computer-vision axes (x right, y down, z forward). The optical axis is
  body x tilted up by ``TILT_DEG`` about body y. The optical centre is the body
  origin (no lever arm, as in the geometry worker).
- Image: principal point at the image centre; pixel (row i, column j) has its
  centre at continuous coordinates (u, v) = (j + 0.5, i + 0.5).
- Heading frame (gravity-levelled): origin at the camera centre, z_h = world up,
  x_h = horizontal projection of the optical axis, y_h = z_h x x_h (left).
  Fallback when the optical axis is within ~11.5 deg of vertical: the horizontal
  projection of body x. If both are degenerate the frame is undefined (NaN) and
  the fan is invalid for that frame.

Store image: 448 x 252 RGB, f = FOCAL_320 * 448 / 320 = 140 px, cx = 224,
cy = 126, 30 deg uptilt: 116.0 deg horizontal and 84.0 deg vertical field of view.

Range grid: 18 x 32 cells of 14 x 14 px (one ViT-S/14 patch each). Cell (r, c)
covers u in [14c, 14c + 14) and v in [14r, 14r + 14); its centre ray passes
through (u, v) = (14c + 7, 14r + 7). Range values are Euclidean distances from
the camera centre in metres (not optical z-depth).

Clearance fan: 36 directions = 4 elevations x 9 yaws in the heading frame.
Array layout ``(..., 4, 9)``: axis -2 indexes ``FAN_ELEV_DEG`` (-10, 0, +10, +20;
positive = up), axis -1 indexes ``FAN_YAW_DEG`` (-40 ... +40 step 10; positive =
LEFT, the FLU yaw sense). Column 0 is therefore 40 deg to the RIGHT; a
horizontal image flip maps column j to 8 - j. Flat index k = e * 9 + j.
Direction (yaw psi, elevation theta) = cos(theta)cos(psi) x_h + cos(theta)sin(psi) y_h
+ sin(theta) z_h. A fan value is the along-ray distance s >= 0 from the camera
centre to the first occupied point whose perpendicular distance to the ray is
at most ``FAN_CORRIDOR_RADIUS_M``.
"""
from __future__ import annotations

import numpy as np

from ..vision.camera import Camera, body_to_cam_matrix

IMAGE_W = 448
IMAGE_H = 252
FRAME_SHAPE = (IMAGE_H, IMAGE_W, 3)       # HWC uint8 RGB
FOCAL_320 = 100.0                          # px at 320 px width (visual_brain gate sensor)
TILT_DEG = 30.0
FOCAL_PX = FOCAL_320 * IMAGE_W / 320.0     # 140.0 px at 448 x 252
PATCH_PX = 14
GRID_H = IMAGE_H // PATCH_PX               # 18
GRID_W = IMAGE_W // PATCH_PX               # 32
GRID_SHAPE = (GRID_H, GRID_W)

RANGE_MIN_M = 0.3      # ranges are clipped to [RANGE_MIN_M, RANGE_MAX_M] in labels and outputs
RANGE_MAX_M = 60.0     # beyond this a free observation is stored as a lower bound of RANGE_MAX_M

FAN_YAW_DEG = np.arange(-40.0, 40.1, 10.0)             # 9 values, positive = LEFT
FAN_ELEV_DEG = np.array([-10.0, 0.0, 10.0, 20.0])      # 4 values, positive = up
FAN_SHAPE = (len(FAN_ELEV_DEG), len(FAN_YAW_DEG))      # (4, 9)
FAN_N = FAN_SHAPE[0] * FAN_SHAPE[1]                    # 36
FAN_CORRIDOR_RADIUS_M = 0.5    # 0.35 m vehicle envelope + 0.15 m margin
FAN_MAX_M = 20.0               # a corridor observed free to this distance is a lower bound of FAN_MAX_M

QUANTILES = (0.2, 0.5)             # model range outputs are the q20 and q50 of ln(range)
BLOCKED_WITHIN_M = (4.0, 8.0)      # model fan probabilities P(first blocked <= d)

GRAVITY_WORLD = np.array([0.0, 0.0, -1.0])
_HEADING_MIN_HORIZONTAL = 0.2      # |horizontal component| below this = degenerate heading


def store_camera() -> Camera:
    """The pinhole camera of every frame in the store (448 x 252, f = 140 px, 30 deg uptilt)."""
    return Camera(IMAGE_W, IMAGE_H, FOCAL_PX, TILT_DEG)


def quat_wxyz_to_mats(q) -> np.ndarray:
    """World-from-body rotation matrices for wxyz quaternions of shape (..., 4) -> (..., 3, 3).

    Quaternions are normalised first; the formula matches haltere.vision.camera.quat_wxyz_to_mat.
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def cell_centres_px() -> np.ndarray:
    """(18, 32, 2) continuous (u, v) pixel coordinates of the grid cell centres at 448 x 252."""
    v, u = np.mgrid[:GRID_H, :GRID_W]
    return np.stack([u * PATCH_PX + PATCH_PX / 2.0, v * PATCH_PX + PATCH_PX / 2.0], axis=-1).astype(np.float64)


def cell_rays_body() -> np.ndarray:
    """(18, 32, 3) unit body-frame (FLU) rays through the grid cell centres."""
    px = cell_centres_px().reshape(-1, 2)
    return store_camera().unproject_body(px).reshape(GRID_H, GRID_W, 3)


def camera_to_world(quat_wb, tilt_deg: float = TILT_DEG) -> np.ndarray:
    """(..., 3, 3) matrices taking camera (right, down, forward) vectors to world FLU vectors."""
    R_wb = quat_wxyz_to_mats(quat_wb)
    R_cb = body_to_cam_matrix(tilt_deg)
    return R_wb @ R_cb.T


def gravity_camera(quat_wb, tilt_deg: float = TILT_DEG) -> np.ndarray:
    """(..., 3) unit gravity (down) vector in the camera frame: the model's attitude input."""
    R_wc = camera_to_world(quat_wb, tilt_deg)
    return np.einsum('...ji,j->...i', R_wc, GRAVITY_WORLD)


def heading_frame(quat_wb, tilt_deg: float = TILT_DEG) -> np.ndarray:
    """(..., 3, 3) world-from-heading matrices; columns are x_h (forward, level), y_h (left), z_h (up).

    NaN where both the optical axis and body x are within ~11.5 deg of vertical.
    """
    R_wb = quat_wxyz_to_mats(quat_wb)
    t = np.deg2rad(tilt_deg)
    optical = R_wb @ np.array([np.cos(t), 0.0, np.sin(t)])
    body_x = R_wb[..., :, 0]
    fwd = optical.copy()
    fwd[..., 2] = 0.0
    alt = body_x.copy()
    alt[..., 2] = 0.0
    n = np.linalg.norm(fwd, axis=-1, keepdims=True)
    n_alt = np.linalg.norm(alt, axis=-1, keepdims=True)
    use_alt = n < _HEADING_MIN_HORIZONTAL
    fwd = np.where(use_alt, alt, fwd)
    n = np.where(use_alt, n_alt, n)
    bad = n < _HEADING_MIN_HORIZONTAL
    x_h = fwd / np.where(bad, 1.0, n)
    z_h = np.broadcast_to(np.array([0.0, 0.0, 1.0]), x_h.shape)
    y_h = np.cross(z_h, x_h)
    H = np.stack([x_h, y_h, z_h], axis=-1)
    return np.where(bad[..., None], np.nan, H)


def fan_directions_heading() -> np.ndarray:
    """(4, 9, 3) unit fan directions in the heading frame, layout (elevation, yaw)."""
    th = np.deg2rad(FAN_ELEV_DEG)[:, None]
    ps = np.deg2rad(FAN_YAW_DEG)[None, :]
    d = np.stack([np.cos(th) * np.cos(ps), np.cos(th) * np.sin(ps), np.sin(th) * np.ones_like(ps)], axis=-1)
    return d


def fan_directions_world(quat_wb, tilt_deg: float = TILT_DEG) -> np.ndarray:
    """(..., 4, 9, 3) unit fan directions in world FLU for camera attitudes (..., 4)."""
    H = heading_frame(quat_wb, tilt_deg)
    return np.einsum('...ij,ekj->...eki', H, fan_directions_heading())


def fan_directions_camera(quat_wb, tilt_deg: float = TILT_DEG) -> np.ndarray:
    """(..., 4, 9, 3) fan directions in the camera frame (for projection and in-view checks)."""
    R_wc = camera_to_world(quat_wb, tilt_deg)
    d_w = fan_directions_world(quat_wb, tilt_deg)
    return np.einsum('...ji,...ekj->...eki', R_wc, d_w)


def project_camera(d_c) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame vectors (..., 3) -> store-image (u, v) (..., 2) and in-front mask (z > 0.05)."""
    d_c = np.asarray(d_c, dtype=np.float64)
    z = d_c[..., 2]
    ok = z > 0.05
    zs = np.where(ok, z, 1.0)
    uv = np.stack([IMAGE_W / 2.0 + FOCAL_PX * d_c[..., 0] / zs,
                   IMAGE_H / 2.0 + FOCAL_PX * d_c[..., 1] / zs], axis=-1)
    return uv, ok


def in_image(uv, ok=None, margin_px: float = 0.0) -> np.ndarray:
    """Boolean mask of (u, v) points inside the 448 x 252 image (minus a margin)."""
    uv = np.asarray(uv, dtype=np.float64)
    m = ((uv[..., 0] >= margin_px) & (uv[..., 0] < IMAGE_W - margin_px)
         & (uv[..., 1] >= margin_px) & (uv[..., 1] < IMAGE_H - margin_px))
    return m if ok is None else (m & ok)


def mirror_fan(a):
    """Horizontal image flip for fan arrays (..., 4, 9): yaw +psi <-> -psi."""
    return a[..., ::-1]


def mirror_grid(a):
    """Horizontal image flip for grid arrays (..., 18, 32)."""
    return a[..., ::-1]


def grid_to_fan(grid_range_m, quat_wb):
    """Geometric fan from a range grid: first-blocked distance per direction within the corridor.

    Contract (implemented by the evaluation agent for baselines B2-B4 and the governor
    diagnostic): unproject every cell centre at its range into the heading frame, then for
    each fan direction take the smallest along-ray distance s >= 0 among points within
    FAN_CORRIDOR_RADIUS_M of the ray; directions with no such point return +inf.
    Input (..., 18, 32) metres and (..., 4) quaternions; output (..., 4, 9) metres.
    Must stay runtime-safe (numpy only) and run in < 1 ms per frame.
    """
    raise NotImplementedError('grid_to_fan is delivered by the evaluation build agent')
