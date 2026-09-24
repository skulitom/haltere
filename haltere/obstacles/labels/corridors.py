"""Sub-ray and fan-corridor queries shared by the label sources. OFFLINE ONLY.

Sub-rays: every grid cell of contract.GRID_SHAPE is sampled by SUB x SUB rays through the
pixel positions (14 c + (i + 0.5) * 14 / SUB, 14 r + (k + 0.5) * 14 / SUB) of the 448 x 252 store
image, so the sub-ray image is (18 * SUB, 32 * SUB) = (54, 96). A source labels every sub-ray with a
(value, LabelKind) constraint on the Euclidean range of the first surface along that ray; cells are
the minimum over their sub-rays (``reduce_subrays`` -> labels.min_over) and are then clipped with
labels.clip_grid.

Fan corridors: ``first_blocked(points_w, observed, origin_w, quat_wb)``. The directions are
contract.fan_directions_world(quat_wb). For each, the value is the smallest along-ray distance
s >= 0 among occupied points whose perpendicular distance to the ray is <= FAN_CORRIDOR_RADIUS_M.

    hit at s and the corridor observed free to e >= s / (1 + EXACT_TOL)   -> EXACT s
    hit at s, corridor not observed free that far                         -> UPPER s
    no hit, corridor observed free to e > 0                               -> LOWER min(e, FAN_MAX_M)
    no hit, nothing observed                                              -> UNKNOWN
    undefined heading frame (camera near vertical)                        -> UNKNOWN

then labels.clip_fan. ``observed`` gives the along-axis distance e to which each corridor is known
free: None (nothing observed), an array broadcastable to (4, 9), or a callable
``observed(origin_w, dirs_w (36, 3), radius_m, s_max) -> (36,)``.
"""
from __future__ import annotations

import numpy as np

from .. import contract
from ..contract import FAN_CORRIDOR_RADIUS_M, FAN_MAX_M, FAN_SHAPE, GRID_H, GRID_W, PATCH_PX
from . import EXACT_TOL, LabelKind, clip_fan, clip_grid, min_over

SUB = 3                                   # sub-rays per cell side (>= 3 x 3 per the schema)
SUB_SHAPE = (GRID_H * SUB, GRID_W * SUB)  # (54, 96)


def subray_pixels(sub: int = SUB) -> np.ndarray:
    """(18 * sub, 32 * sub, 2) continuous (u, v) store-image coordinates of the sub-rays."""
    step = PATCH_PX / sub
    v, u = np.mgrid[:GRID_H * sub, :GRID_W * sub]
    return np.stack([(u + 0.5) * step, (v + 0.5) * step], axis=-1).astype(np.float64)


def pixel_dirs_camera(uv, f: float = contract.FOCAL_PX, cx: float = contract.IMAGE_W / 2.0,
                      cy: float = contract.IMAGE_H / 2.0) -> np.ndarray:
    """Unit camera-frame (right, down, forward) rays through continuous pixel coordinates (..., 2)."""
    uv = np.asarray(uv, dtype=np.float64)
    d = np.stack([(uv[..., 0] - cx) / f, (uv[..., 1] - cy) / f, np.ones(uv.shape[:-1])], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


_SUB_DIRS_C = pixel_dirs_camera(subray_pixels())


def subray_dirs_camera() -> np.ndarray:
    """(54, 96, 3) unit camera-frame sub-ray directions."""
    return _SUB_DIRS_C.copy()


def subray_dirs_world(quat_wb) -> np.ndarray:
    """(54, 96, 3) unit world FLU sub-ray directions for one world-from-body attitude (wxyz)."""
    R_wc = contract.camera_to_world(np.asarray(quat_wb, dtype=np.float64))
    return _SUB_DIRS_C @ R_wc.T


def world_to_pixels(points_w, origin_w, quat_wb):
    """World points (n, 3) -> store-image (u, v) (n, 2), range (n,) and in-front mask (n,)."""
    R_wc = contract.camera_to_world(np.asarray(quat_wb, dtype=np.float64))
    rel = np.asarray(points_w, dtype=np.float64) - np.asarray(origin_w, dtype=np.float64)
    d_c = rel @ R_wc                       # world -> camera: R_wc^T v
    uv, ok = contract.project_camera(d_c)
    return uv, np.linalg.norm(rel, axis=-1), ok


def reduce_subrays(values, kinds, *, min_known_frac: float = 1.0, sub: int = SUB):
    """Sub-ray constraints (..., 18 * sub, 32 * sub) -> cell constraints (..., 18, 32) via labels.min_over."""
    v = np.asarray(values, dtype=np.float64)
    k = np.asarray(kinds)
    lead = v.shape[:-2]
    v = v.reshape(lead + (GRID_H, sub, GRID_W, sub))
    k = k.reshape(lead + (GRID_H, sub, GRID_W, sub))
    n = len(lead)
    order = tuple(range(n)) + (n, n + 2, n + 1, n + 3)
    v = v.transpose(order).reshape(lead + (GRID_H, GRID_W, sub * sub))
    k = k.transpose(order).reshape(lead + (GRID_H, GRID_W, sub * sub))
    return min_over(v, k, axis=-1, min_known_frac=min_known_frac)


def grid_from_subrays(values, kinds, *, min_known_frac: float = 1.0):
    """Sub-ray constraints -> clipped grid (value float64 with NaN, kind uint8), each (18, 32)."""
    v, k = reduce_subrays(values, kinds, min_known_frac=min_known_frac)
    return clip_grid(v, k)


def unknown_grid():
    return np.full((GRID_H, GRID_W), np.nan), np.zeros((GRID_H, GRID_W), np.uint8)


def unknown_fan():
    return np.full(FAN_SHAPE, np.nan), np.zeros(FAN_SHAPE, np.uint8)


def unknown_subrays():
    return np.full(SUB_SHAPE, np.nan), np.zeros(SUB_SHAPE, np.uint8)


def corridor_hits(points_w, origin_w, dirs_w, radius_m: float = FAN_CORRIDOR_RADIUS_M,
                  s_max: float = np.inf) -> np.ndarray:
    """Smallest along-ray distance s in [0, s_max] of the points within ``radius_m`` of each ray.

    points (n, 3), origin (3,), dirs (m, 3) unit -> (m,) with +inf where no point qualifies.
    """
    dirs = np.asarray(dirs_w, dtype=np.float64).reshape(-1, 3)
    out = np.full(len(dirs), np.inf)
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return out
    rel = pts - np.asarray(origin_w, dtype=np.float64)
    dist = np.linalg.norm(rel, axis=1)
    keep = dist <= s_max + radius_m
    rel, dist = rel[keep], dist[keep]
    if len(rel) == 0:
        return out
    s = rel @ dirs.T                                               # (n, m)
    perp2 = np.maximum(dist[:, None] ** 2 - s ** 2, 0.0)
    inside = (s >= 0) & (perp2 <= radius_m ** 2) & (s <= s_max)
    s = np.where(inside, s, np.inf)
    return s.min(axis=0)


def _free_extent(observed, origin_w, dirs, radius_m, s_max):
    if observed is None:
        return np.zeros(len(dirs))
    if callable(observed):
        return np.asarray(observed(origin_w, dirs, radius_m, s_max), dtype=np.float64).reshape(len(dirs))
    return np.broadcast_to(np.asarray(observed, dtype=np.float64), FAN_SHAPE).reshape(-1).copy()


def fan_constraints(hit_s, free_e, *, tol: float = EXACT_TOL, s_max: float = FAN_MAX_M):
    """Per-direction hit distance and observed-free extent -> (value, kind) before clipping."""
    hit_s = np.asarray(hit_s, dtype=np.float64)
    free_e = np.nan_to_num(np.asarray(free_e, dtype=np.float64), nan=0.0)
    value = np.full(hit_s.shape, np.nan)
    kind = np.zeros(hit_s.shape, np.uint8)
    hit = np.isfinite(hit_s)
    exact = hit & (free_e * (1 + tol) >= hit_s)
    value[hit] = hit_s[hit]
    kind[hit] = LabelKind.UPPER
    kind[exact] = LabelKind.EXACT
    free = ~hit & (free_e > 0)
    value[free] = np.minimum(free_e[free], s_max)
    kind[free] = LabelKind.LOWER
    return value, kind


def first_blocked(points_w, observed, origin_w, quat_wb, *, radius_m: float = FAN_CORRIDOR_RADIUS_M,
                  s_max: float = FAN_MAX_M):
    """Fan (value (4, 9) float64 with NaN, kind (4, 9) uint8) for one frame, clipped with labels.clip_fan."""
    dirs = contract.fan_directions_world(np.asarray(quat_wb, dtype=np.float64)).reshape(-1, 3)
    if not np.isfinite(dirs).all():
        return unknown_fan()
    hit = corridor_hits(points_w, origin_w, dirs, radius_m, s_max)
    free = _free_extent(observed, origin_w, dirs, radius_m, s_max)
    v, k = fan_constraints(hit, free, s_max=s_max)
    return clip_fan(v.reshape(FAN_SHAPE), k.reshape(FAN_SHAPE))
