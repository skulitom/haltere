"""L3: impact/contact points as exact occupied points, and the labelled event list. OFFLINE ONLY.

Events: the 15 curated lateral-manifest impacts (plus its non-fatal contacts and near pass), and the
blind-labelled remaining Minus Two (6) and Pine Valley (2) terminal impacts
(configs/obstacles/events_f12.json), schema labels.EVENT_FIELDS. Blind protocol: every event in
events_f12.json is labelled (point_w, normal_w, obstacle, unique_obstacle, free sides) from frames and
telemetry BEFORE any model or baseline output on it is seen; blind=true, labeller and labelled_at
are set. Curated lateral-manifest events are not claimed blind (blind=false). Oracle-route flights are
marked and excluded from evaluation sets.

Contact point from telemetry (``contact_from_telemetry``): the contact tick is the first 100 Hz row
near the labelled time whose 3-tick mean acceleration exceeds CONTACT_ACCEL_MPS2 and whose force
cannot come from the propellers (as haltere.liftoff.flightlog.collisions: specific force off the
thrust axis > CONTACT_UNEXPLAINED_MPS2). drone_pos_w = position at the tick before; normal_w = the
unexplained (external) force direction when it is strong and faces against the motion, else minus
the pre-contact velocity direction; point_w = drone_pos_w - CONTACT_OFFSET_M * normal_w (the airframe
reaches the surface ~0.15 m from its centre; +-0.1 m uncertainty). Launch-relative FLU coordinates,
the same frame as the store's ``pos``.

Per frame in [T - 3 s, T - 0.15 s] where point_w projects into the 448 x 252 image: sub-rays that hit
a 0.2 m disc around point_w (normal normal_w, or facing the camera when unknown) are EXACT at their
range when the straight segment from the camera to the point stays within VISIBLE_TUBE_M (= the
0.35 m flown-tube radius) of the flown path (that segment is certified free, so nothing hides the
point), otherwise UPPER. Other sub-rays are untouched. The same disc points constrain fan corridors
(UPPER: the corridor before them is not observed by this source), also when the point is outside the
image. Writes <store>/labels/events.json (superset of <store>/events.json; keeps
store_event_id).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import contract
from . import EVENT_FIELDS, LabelKind, clip_fan, validate_event
from .corridors import (SUB_SHAPE, corridor_hits, grid_from_subrays, subray_dirs_world, unknown_fan,
                        world_to_pixels)

IMPACT_WINDOW_S = (0.15, 3.0)
IMPACT_DISC_RADIUS_M = 0.2
CONTACT_ACCEL_MPS2 = 20.0
CONTACT_UNEXPLAINED_MPS2 = 6.0
CONTACT_OFFSET_M = 0.15
VISIBLE_TUBE_M = 0.35                 # = tube radius: the line of sight lies in certified-free flown volume
DISC_SAMPLES = 61                     # points used for fan-corridor constraints
UNIQUE_RADIUS_M = 1.5                 # events this close (same environment) share a unique_obstacle key
NEAR_PASS_M = 1.0                     # near pass: closest approach to a labelled obstacle point (other flights)


# ----------------------------------------------------------------------------- telemetry contact

def _body_up(q):
    q = np.asarray(q, dtype=np.float64)
    return contract.quat_wxyz_to_mats(q)[..., :, 2]


def contact_from_telemetry(t, pos, vel, quat, t_label: float, *, search_s=(-0.4, 0.15)):
    """Contact geometry near ``t_label`` (same clock as ``t``) -> dict or None.

    Contact mode: the first tick in [t_label + search_s] whose 3-tick mean acceleration exceeds
    CONTACT_ACCEL_MPS2 with an unexplained (non-propeller) force above CONTACT_UNEXPLAINED_MPS2
    (as haltere.liftoff.flightlog.collisions). drone_pos_w = position one tick before; the normal is
    the external force direction when it is strong and opposes the motion, else minus the
    pre-contact velocity direction.

    Terminal mode (no such tick: the runner stops logging AT the impact, so the impulse is not in
    the CSV): drone_pos_w = the position at t_label (linear extrapolation past the last row with the
    last velocity, at most 0.05 s), normal = minus the pre-impact velocity direction (a frontal-hit
    assumption; blind labelling may override it from the frames).

    point_w = drone_pos_w - CONTACT_OFFSET_M * normal (airframe surface ~0.15 m from the centre).
    Returns t_contact, drone_pos_w, speed_mps, normal_w, normal_source, point_w, accel_mps2,
    unexplained_mps2, threshold_crossed, mode.
    """
    t = np.asarray(t, dtype=np.float64)
    P, V, Q = (np.asarray(a, dtype=np.float64) for a in (pos, vel, quat))
    if len(t) < 5 or not np.isfinite(t_label):
        return None
    dt = np.maximum(np.diff(t), 1e-3)
    A = np.diff(V, axis=0) / dt[:, None]
    k = np.ones(3) / 3
    A = np.stack([np.convolve(A[:, i], k, mode='same') for i in range(3)], axis=1)
    acc = np.linalg.norm(A, axis=1)
    spec = A + np.array([0.0, 0.0, 9.81])
    up = _body_up(Q[:len(A)])
    thrust = np.maximum(np.einsum('ij,ij->i', spec, up), 0.0)
    ext = spec - thrust[:, None] * up
    left = np.linalg.norm(ext, axis=1)
    win = np.flatnonzero((t[:-1] >= t_label + search_s[0]) & (t[:-1] <= t_label + search_s[1]))
    cand = win[(acc[win] > CONTACT_ACCEL_MPS2) & (left[win] > CONTACT_UNEXPLAINED_MPS2)] if len(win) else win
    if len(cand):
        e = int(cand[0])
        # The 3-tick smoothing spreads the impulse one tick early; the drone centre at contact is the
        # position just before the velocity change.
        e0 = max(e - 1, 0)
        v_pre = V[max(e0 - 5, 0)]
        speed = float(np.linalg.norm(v_pre))
        lo, hi = max(e - 3, 0), min(e + 4, len(ext))
        j = lo + int(np.argmax(left[lo:hi]))
        f = ext[j]
        n_force = f / max(np.linalg.norm(f), 1e-9)
        n_vel = -v_pre / max(speed, 1e-9)
        if left[j] > 2 * CONTACT_ACCEL_MPS2 and (speed < 0.5 or float(n_force @ n_vel) > np.cos(np.deg2rad(80))):
            normal, src = n_force, 'external force'
        elif speed >= 0.5:
            normal, src = n_vel, 'minus pre-contact velocity'
        else:
            normal, src = n_force, 'external force (weak)'
        drone, t_c, accv, un, mode = P[e0], float(t[e0]), float(acc[e]), float(left[j]), 'contact'
    else:
        if t_label < t[0] or t_label > t[-1] + 0.05:
            return None
        if t_label <= t[-1]:
            drone = np.array([np.interp(t_label, t, P[:, a]) for a in range(3)])
        else:
            drone = P[-1] + V[-1] * (t_label - t[-1])
        i_pre = int(np.clip(np.searchsorted(t, t_label - 0.05), 0, len(t) - 1))
        v_pre = V[i_pre]
        speed = float(np.linalg.norm(v_pre))
        if speed < 0.3:
            return None
        normal, src = -v_pre / speed, 'minus pre-impact velocity (impulse not in the CSV)'
        t_c, accv, un, mode = float(t_label), float('nan'), float('nan'), 'terminal'
    return dict(t_contact=t_c, drone_pos_w=np.asarray(drone, float).tolist(), speed_mps=speed,
                normal_w=np.asarray(normal, float).tolist(), normal_source=src,
                point_w=(np.asarray(drone, float) - CONTACT_OFFSET_M * np.asarray(normal, float)).tolist(),
                accel_mps2=accv, unexplained_mps2=un, threshold_crossed=mode == 'contact', mode=mode)


# ----------------------------------------------------------------------------- events

# E4 lateral scoring of the curated impacts, copied from the lateral study's evaluation
# (scratchpad/lateral/cand-split-looming/evalcore.py LAT / NOT_LATERAL): primary side, accepted sides.
LATERAL_SCORING = {
    'minus-brain03-01': ('right', ['right', 'left']),
    'minus-brain04-02': ('right', ['right', 'left']),
    'minus-brain05-01': ('either', ['right', 'left']),
    'minus-fast6-01': ('left', ['left']),
    'pine-fast6-ttc-01': ('right', ['right']),
    'pine-brain05-02': ('right', ['right']),
    'pine-fast6-01': ('right', ['right']),
    'pine-brain03-01': ('left', ['left', 'right']),
    'straw-brain05-01': ('right', ['right']),
    'straw-brain05-trim-02': ('left', ['left']),
    'straw-brain05-gov-01': ('right', ['right']),
}
NOT_LATERAL = {
    'pine-fast6-loom-01': ('up', ['up']),
    'pine-brain04-01': ('unknown', []),
    'straw-fast6-01': ('up', ['up']),
    'straw-brain05-trim-01': ('up', ['up']),
}


def _free_side(text: str) -> tuple[str, list]:
    s = (text or '').lower()
    order = [w for w in ('left', 'right', 'up', 'down') if w in s]
    if not order or 'unknown' in s.split('(')[0]:
        return 'unknown', []
    first = min(order, key=lambda w: s.index(w))
    return first, order


def lateral_manifest_events(manifest: dict, *, first_id: int = 0, labeller: str = 'lateral study') -> list[dict]:
    """Curated lateral-manifest impacts, non-fatal contacts and near passes as EVENT_FIELDS dicts.

    Telemetry-derived geometry (point_w, normal_w) is filled later by ``attach_contact_geometry``.
    """
    env_of = lambda s: ('Minus Two' if 'Minus' in s else 'Pine Valley' if 'Pine' in s else
                        'Straw Bale' if 'Straw' in s else s)
    out = []
    eid = first_id
    for e in manifest.get('impacts', []):
        if e['run'] in LATERAL_SCORING:
            (side, sides), lateral = LATERAL_SCORING[e['run']], True
        elif e['run'] in NOT_LATERAL:
            (side, sides), lateral = NOT_LATERAL[e['run']], False
        else:
            (side, sides), lateral = _free_side(e.get('free_side', '')), False
        out.append(dict(event_id=eid, store_event_id=-1, run=e['run'], flight=None, env=env_of(e['environment']),
                        kind='terminal_impact', t_phase=float(e['impact_phase_s']), t_wall=None,
                        point_w=None, normal_w=None, drone_pos_w=list(map(float, e['impact_pos'])),
                        speed_mps=e.get('speed_pre_mps'), obstacle=e.get('obstacle', 'unknown'),
                        unique_obstacle=None, obstacle_side=e.get('obstacle_side'),
                        primary_free_side=side, accepted_free_sides=sides, lateral=bool(lateral),
                        in_view_frac_T2_T1=None, oracle_route=False, source='lateral_manifest', blind=False,
                        labeller=labeller, labelled_at=manifest.get('created'), confidence=e.get('confidence'),
                        notes=e.get('notes'), evidence=[v for v in (e.get('evidence') or {}).values()
                                                        if isinstance(v, str)]))
        eid += 1
    for e in manifest.get('non_fatal_contacts', []):
        side = e.get('obstacle_side', '')
        free = {'left': 'right', 'right': 'left', 'below': 'up', 'below-left': 'right'}.get(side, 'unknown')
        run = e['run']
        env = ('Minus Two' if run.startswith('minus') else 'Pine Valley' if run.startswith('pine') else
               'Straw Bale' if run.startswith('straw') else 'unknown')
        out.append(dict(event_id=eid, store_event_id=-1, run=run, flight=None, env=env, kind='contact',
                        t_phase=float(e['phase_s']), t_wall=None, point_w=None, normal_w=None, drone_pos_w=None,
                        speed_mps=None, obstacle=e.get('obstacle', 'unknown'), unique_obstacle=None,
                        obstacle_side=side, primary_free_side=free, accepted_free_sides=[free] if free != 'unknown' else [],
                        lateral=False, in_view_frac_T2_T1=None, oracle_route=False, source='lateral_manifest',
                        blind=False, labeller=labeller, labelled_at=manifest.get('created'), confidence='medium',
                        notes=e.get('notes'), evidence=list(e.get('evidence') or [])))
        eid += 1
    for e in manifest.get('near_passes', []):
        a, b = e['phase_s']
        run = e['run']
        env = 'Minus Two' if run.startswith('minus') else 'Pine Valley' if run.startswith('pine') else 'Straw Bale'
        out.append(dict(event_id=eid, store_event_id=-1, run=run, flight=None, env=env, kind='near_pass',
                        t_phase=float(0.5 * (a + b)), t_wall=None, point_w=None, normal_w=None, drone_pos_w=None,
                        speed_mps=None, obstacle=e.get('obstacle', 'unknown'), unique_obstacle=None,
                        obstacle_side=e.get('obstacle_side'), primary_free_side='none', accepted_free_sides=[],
                        lateral=False, in_view_frac_T2_T1=None, oracle_route=False, source='lateral_manifest',
                        blind=False, labeller=labeller, labelled_at=manifest.get('created'), confidence='medium',
                        notes=e.get('notes'), evidence=list(e.get('evidence') or [])))
        eid += 1
    return out


def unique_obstacle_key(env: str, obstacle: str, point_w, cell_m: float = 0.1) -> str:
    """Grouping key for repeated hits on one physical obstacle (E7): env/obstacle-x-y (rounded point)."""
    slug = env.lower().replace(' ', '-')
    kind = (obstacle or 'unknown').split('(')[0].strip().lower() or 'unknown'
    if point_w is None:
        return f'{slug}/{kind}/unlocated'
    x, y = (round(float(point_w[i]) / cell_m) * cell_m for i in (0, 1))
    return f'{slug}/{kind}@{x:.1f},{y:.1f}'


def assign_unique_obstacles(events: list[dict], radius_m: float = None) -> list[dict]:
    """Fill missing ``unique_obstacle`` keys: an event within ``radius_m`` (same environment) of an event that
    already has a key takes that key; otherwise it starts a new key from its own point (events in list order)."""
    radius_m = UNIQUE_RADIUS_M if radius_m is None else radius_m
    keyed = [e for e in events if e.get('unique_obstacle') and e.get('point_w') is not None]
    out = []
    for e in events:
        e = dict(e)
        if not e.get('unique_obstacle'):
            best, dbest = None, radius_m
            if e.get('point_w') is not None:
                p = np.asarray(e['point_w'], float)
                for k in keyed:
                    if k['env'] != e['env']:
                        continue
                    d = float(np.linalg.norm(np.asarray(k['point_w'], float) - p))
                    if d <= dbest:
                        best, dbest = k, d
            e['unique_obstacle'] = (best['unique_obstacle'] if best else
                                    unique_obstacle_key(e['env'], e['obstacle'], e.get('point_w')))
            if e.get('point_w') is not None:
                keyed.append(e)
        out.append(e)
    return out


def attach_contact_geometry(event: dict, wall, phase, pos, vel, quat, *, search_s=(-0.4, 0.15)) -> dict:
    """Fill t_wall, point_w, normal_w, drone_pos_w and speed from 100 Hz telemetry (the event's t_phase is on
    the CSV phase clock). A point_w already present (blind label) is kept with its normal."""
    wall = np.asarray(wall, np.float64)
    phase = np.asarray(phase, np.float64)
    e = dict(event)
    ok = np.isfinite(phase) & np.isfinite(wall)
    if e.get('t_wall') is None:
        if ok.sum() < 2:
            return e
        e['t_wall'] = round(float(np.interp(e['t_phase'], phase[ok], wall[ok])), 3)
    if e.get('point_w') is not None:
        return e
    c = contact_from_telemetry(wall, pos, vel, quat, float(e['t_wall']), search_s=search_s)
    if c is None:
        return e
    e['point_w'] = [round(float(x), 3) for x in c['point_w']]
    e['normal_w'] = [round(float(x), 3) for x in c['normal_w']]
    if e.get('drone_pos_w') is None:
        e['drone_pos_w'] = [round(float(x), 3) for x in c['drone_pos_w']]
    if e.get('speed_mps') is None:
        e['speed_mps'] = round(c['speed_mps'], 2)
    note = (f"contact geometry from telemetry ({c['mode']} mode): normal from {c['normal_source']}, point = drone "
            f"centre - {CONTACT_OFFSET_M} m * normal")
    e['notes'] = (e.get('notes') or '') + (' | ' if e.get('notes') else '') + note
    return e


def write_events(path, events: list[dict], **meta) -> None:
    for e in events:
        validate_event(e)
    ids = [e['event_id'] for e in events]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate event_id')
    obj = dict(schema='haltere.obstacles.events.v1', fields=EVENT_FIELDS, **meta, events=events)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


def stamp_blind(event: dict, labeller: str) -> dict:
    return dict(event, blind=True, labeller=labeller, labelled_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))


# ----------------------------------------------------------------------------- per-frame labels

def disc_points(point_w, normal_w, radius: float = IMPACT_DISC_RADIUS_M, n: int = DISC_SAMPLES) -> np.ndarray:
    """(n, 3) points on the disc (sunflower pattern), centre first."""
    p = np.asarray(point_w, dtype=np.float64)
    nrm = np.asarray(normal_w, dtype=np.float64)
    nrm = nrm / np.linalg.norm(nrm)
    a = np.array([0.0, 0.0, 1.0]) if abs(nrm[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(nrm, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)
    k = np.arange(n)
    r = radius * np.sqrt(k / max(n - 1, 1))
    th = k * np.pi * (3 - np.sqrt(5))
    return p + r[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)


def segment_visible(origin, point_w, flown_path, tol: float = VISIBLE_TUBE_M, stop_short_m: float = 0.35) -> bool:
    """True when the straight segment camera -> point (minus stop_short_m) stays within tol of the flown path."""
    path = np.asarray(flown_path, dtype=np.float64).reshape(-1, 3)
    if len(path) == 0:
        return False
    o, p = np.asarray(origin, dtype=np.float64), np.asarray(point_w, dtype=np.float64)
    L = float(np.linalg.norm(p - o))
    if L <= stop_short_m:
        return True
    s = np.linspace(0.0, L - stop_short_m, max(2, int((L - stop_short_m) / 0.1) + 1))
    q = o + (p - o)[None] / L * s[:, None]
    d = np.sqrt(((q[:, None, :] - path[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return bool((d <= tol).all())


def impact_labels(event: dict, pos, quat_wb, *, flown_path=None, return_subrays: bool = False):
    """-> (grid_value, grid_kind, fan_value, fan_kind) for one frame, or None when the event has no point.

    ``pos`` is the camera position in the event's frame (launch-relative). ``flown_path``: trajectory
    samples from the frame to the contact, used for the EXACT visibility rule (None -> UPPER only).
    The grid is UNKNOWN when the point is out of view; the fan is computed in any case (a corridor can
    hold the point below or beside the image).
    """
    if event.get('point_w') is None:
        return None
    p = np.asarray(event['point_w'], dtype=np.float64)
    o = np.asarray(pos, dtype=np.float64)
    q = np.asarray(quat_wb, dtype=np.float64)
    nrm = event.get('normal_w')
    to_cam = o - p
    dist = float(np.linalg.norm(to_cam))
    if dist < 1e-3 or not np.isfinite(dist):
        return None
    nrm = to_cam / dist if nrm is None else np.asarray(nrm, dtype=np.float64)
    sub_v = np.full(SUB_SHAPE, np.nan)
    sub_k = np.zeros(SUB_SHAPE, np.uint8)
    uv, rng, ok = world_to_pixels(p[None], o, q)
    if contract.in_image(uv, ok)[0]:
        dirs = subray_dirs_world(q).reshape(-1, 3)
        denom = dirs @ nrm
        with np.errstate(divide='ignore', invalid='ignore'):
            s = ((p - o) @ nrm) / denom
        hitpt = o[None] + s[:, None] * dirs
        on = (np.abs(denom) > 1e-6) & (s > 0) & (np.linalg.norm(hitpt - p[None], axis=1) <= IMPACT_DISC_RADIUS_M)
        visible = flown_path is not None and segment_visible(o, p, flown_path)
        kind = LabelKind.EXACT if visible else LabelKind.UPPER
        sub_v.reshape(-1)[on] = s[on]
        sub_k.reshape(-1)[on] = kind
        if not on.any():
            # The disc is smaller than one sub-ray spacing (far away): constrain the sub-ray nearest to it.
            from .corridors import subray_pixels
            px = subray_pixels().reshape(-1, 2)
            j = int(np.argmin(np.linalg.norm(px - uv[0][None], axis=1)))
            sub_v.reshape(-1)[j] = rng[0]
            sub_k.reshape(-1)[j] = LabelKind.UPPER
        gv, gk = grid_from_subrays(sub_v, sub_k)
    else:
        from .corridors import unknown_grid
        gv, gk = unknown_grid()
    fdirs = contract.fan_directions_world(q).reshape(-1, 3)
    if np.isfinite(fdirs).all():
        h = corridor_hits(disc_points(p, nrm), o, fdirs, s_max=contract.FAN_MAX_M).reshape(contract.FAN_SHAPE)
        fv = np.where(np.isfinite(h), h, np.nan)
        fk = np.where(np.isfinite(h), LabelKind.UPPER, LabelKind.UNKNOWN).astype(np.uint8)
        fv, fk = clip_fan(fv, fk)
    else:
        fv, fk = unknown_fan()
    if return_subrays:
        return gv, gk, fv, fk, sub_v, sub_k
    return gv, gk, fv, fk


def in_view(event: dict, pos, quat_wb) -> bool:
    if event.get('point_w') is None:
        return False
    uv, _, ok = world_to_pixels(np.asarray(event['point_w'], dtype=np.float64)[None], pos, quat_wb)
    return bool(contract.in_image(uv, ok)[0])
