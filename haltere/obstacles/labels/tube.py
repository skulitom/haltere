"""L1: flown-tube free-space lower bounds. OFFLINE ONLY.

For a frame at t, the drone positions over [t, t + TUBE_HORIZON_S] (100 Hz telemetry, stopping
TUBE_STOP_MARGIN_S before the first contact, terminal impact or reset, and at the end of the
telemetry) are the centres of spheres of radius TUBE_RADIUS_M whose union is free space
(the vehicle envelope flew there without contact). LabelSource.TUBE.

Grid rule: every sub-ray starts at the camera (on the trajectory, inside the tube). It gets
LOWER s_exit, where s_exit is the distance at which it leaves the connected union of spheres,
when s_exit >= TUBE_MIN_LOWER_M; shorter bounds are dropped (UNKNOWN) because they carry no
information beyond the vehicle itself and would otherwise erase occupied evidence in the
combination. Cells take labels.min_over of their sub-rays (strict: a cell is bounded only when
all its sub-rays are).

Fan rule (``FAN_RULE``): the tube certifies the flown vehicle envelope, not the 0.5 m corridor.
A fan corridor counts as free from 0 to s while its AXIS stays within TUBE_FAN_AXIS_M
(= FAN_CORRIDOR_RADIUS_M - TUBE_RADIUS_M = 0.15 m) of the flown centreline; it then gets
LOWER s when s >= TUBE_MIN_LOWER_M. The 0.15 m margin ring outside the 0.35 m envelope is
assumed, not observed; occupied evidence from other sources overrides it through labels.intersect
(contradictions become UNKNOWN and are counted). Unreliable alignments get no tube labels.
This counters teacher smear inside gate openings.
"""
from __future__ import annotations

import numpy as np

from .. import contract
from ..contract import FAN_CORRIDOR_RADIUS_M, FAN_MAX_M, FAN_SHAPE, RANGE_MAX_M
from . import LabelKind, clip_fan
from .corridors import SUB_SHAPE, grid_from_subrays, subray_dirs_world, unknown_fan, unknown_grid

TUBE_HORIZON_S = 3.0
TUBE_RADIUS_M = 0.35
TUBE_STOP_MARGIN_S = 0.15          # stop the swept tube this long before a contact/impact/reset
TUBE_MIN_LOWER_M = 1.0             # shorter free-space bounds are not written
TUBE_FAN_AXIS_M = FAN_CORRIDOR_RADIUS_M - TUBE_RADIUS_M     # 0.15 m
TUBE_SAMPLE_M = 0.08               # trajectory resampling for the sphere union
FAN_RULE = ('fan corridor free from 0 to s while its axis stays within 0.15 m of the flown centreline '
            '(vehicle envelope 0.35 m certified by the flight; 0.15 m margin assumed); LOWER s if s >= 1 m')
GRID_RULE = ('sub-ray LOWER s_exit where it leaves the union of 0.35 m spheres on the trajectory over '
             '[t, t + 3 s] (stopping 0.15 s before a contact/impact/reset); dropped when s_exit < 1 m; '
             'cells = strict min over 3 x 3 sub-rays')


def trajectory_window(telemetry_t, telemetry_pos, t_frame: float, stop_time: float | None = None,
                      horizon_s: float = TUBE_HORIZON_S, margin_s: float = TUBE_STOP_MARGIN_S,
                      sample_m: float = TUBE_SAMPLE_M) -> np.ndarray:
    """Positions (k, 3) over [t_frame, t_end], resampled to ~sample_m spacing (always includes t_frame)."""
    t = np.asarray(telemetry_t, dtype=np.float64)
    p = np.asarray(telemetry_pos, dtype=np.float64)
    if len(t) < 2 or not np.isfinite(t_frame) or t_frame < t[0] or t_frame > t[-1]:
        return np.zeros((0, 3))
    t_end = min(t_frame + horizon_s, t[-1])
    if stop_time is not None and np.isfinite(stop_time):
        t_end = min(t_end, stop_time - margin_s)
    if t_end <= t_frame:
        return np.zeros((0, 3))
    i0, i1 = np.searchsorted(t, t_frame), np.searchsorted(t, t_end, side='right')
    ts = np.r_[t_frame, t[i0:i1], t_end]
    ps = np.stack([np.interp(ts, t, p[:, k]) for k in range(3)], axis=1)
    keep = np.isfinite(ps).all(axis=1)
    ts, ps = ts[keep], ps[keep]
    if len(ps) < 2:
        return ps
    seg = np.linalg.norm(np.diff(ps, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    if arc[-1] <= 0:
        return ps[:1]
    n = max(2, int(np.ceil(arc[-1] / sample_m)) + 1)
    a = np.linspace(0.0, arc[-1], n)
    return np.stack([np.interp(a, arc, ps[:, k]) for k in range(3)], axis=1)


def ray_tube_exit(origin, dirs, centres, radius: float = TUBE_RADIUS_M, s_max: float = RANGE_MAX_M) -> np.ndarray:
    """Distance at which rays from ``origin`` leave the connected union of spheres (0 if they start outside).

    origin (3,), dirs (m, 3) unit, centres (k, 3) -> (m,) in [0, s_max].
    """
    dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(len(dirs))
    if len(c) == 0 or len(dirs) == 0:
        return out
    rel = c - np.asarray(origin, dtype=np.float64)                  # (k, 3)
    proj = dirs @ rel.T                                             # (m, k) along-ray position of each centre
    d2 = np.sum(rel * rel, axis=1)[None, :] - proj ** 2             # squared perpendicular distance
    half = np.sqrt(np.maximum(radius ** 2 - d2, 0.0))
    hit = d2 <= radius ** 2
    s_in = np.where(hit, proj - half, np.inf)
    s_out = np.where(hit, proj + half, -np.inf)
    # Start: spheres that contain the origin (s_in <= 0 <= s_out).
    reach = np.where((s_in <= 0) & (s_out >= 0), s_out, -np.inf).max(axis=1)
    active = np.flatnonzero(reach > 0)
    while len(active):
        r = reach[active]
        cand = np.where(s_in[active] <= r[:, None] + 1e-9, s_out[active], -np.inf).max(axis=1)
        grew = cand > r + 1e-9
        reach[active[grew]] = cand[grew]
        active = active[grew]
    return np.clip(np.maximum(reach, 0.0), 0.0, s_max)


def axis_tube_extent(origin, dirs, centres, axis_tol: float = TUBE_FAN_AXIS_M, s_max: float = FAN_MAX_M,
) -> np.ndarray:
    """Along-axis distance to which each ray stays within ``axis_tol`` of the trajectory polyline samples."""
    dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    if len(c) < 1:
        return np.zeros(len(dirs))
    # Centres are resampled at <= TUBE_SAMPLE_M, so the union of axis_tol spheres approximates
    # "within axis_tol of the flown centreline" (to within ~5 mm at 0.08 m spacing).
    return ray_tube_exit(origin, dirs, c, radius=axis_tol, s_max=s_max)


def tube_labels(telemetry_t, telemetry_pos, t_frame: float, pos, quat_wb, stop_time: float | None = None, *,
                horizon_s: float = TUBE_HORIZON_S, return_subrays: bool = False):
    """-> (grid_value (18, 32), grid_kind, fan_value (4, 9), fan_kind) LOWER bounds for one frame.

    ``telemetry_pos`` and ``pos`` share one world frame (launch-relative or absolute).
    """
    centres = trajectory_window(telemetry_t, telemetry_pos, t_frame, stop_time, horizon_s)
    origin = np.asarray(pos, dtype=np.float64)
    sub_v = np.full(SUB_SHAPE, np.nan)
    sub_k = np.zeros(SUB_SHAPE, np.uint8)
    if len(centres) == 0 or not np.isfinite(origin).all():
        gv, gk = unknown_grid()
        fv, fk = unknown_fan()
        return (gv, gk, fv, fk, sub_v, sub_k) if return_subrays else (gv, gk, fv, fk)
    # The camera sits on the trajectory at t_frame; make sure the first sphere is centred on it.
    centres = np.vstack([origin[None], centres])
    dirs = subray_dirs_world(quat_wb).reshape(-1, 3)
    s = ray_tube_exit(origin, dirs, centres, TUBE_RADIUS_M, RANGE_MAX_M).reshape(SUB_SHAPE)
    ok = s >= TUBE_MIN_LOWER_M
    sub_v[ok] = s[ok]
    sub_k[ok] = LabelKind.LOWER
    gv, gk = grid_from_subrays(sub_v, sub_k)
    fdirs = contract.fan_directions_world(np.asarray(quat_wb, dtype=np.float64)).reshape(-1, 3)
    if np.isfinite(fdirs).all():
        e = axis_tube_extent(origin, fdirs, centres, TUBE_FAN_AXIS_M, FAN_MAX_M).reshape(FAN_SHAPE)
        fv = np.where(e >= TUBE_MIN_LOWER_M, e, np.nan)
        fk = np.where(e >= TUBE_MIN_LOWER_M, LabelKind.LOWER, LabelKind.UNKNOWN).astype(np.uint8)
    else:
        fv, fk = unknown_fan()
    fv, fk = clip_fan(fv, fk)
    if return_subrays:
        return gv, gk, fv, fk, sub_v, sub_k
    return gv, gk, fv, fk
