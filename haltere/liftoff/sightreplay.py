"""Offline replay of the rabbit pilot's tracker from a ``fly --log`` file (``haltere liftoff replay-sight``).

The log has every telemetry frame the pilot stepped on (100 Hz) and the detector's latest output at each of them.
The replay rebuilds a ``Detection`` for every new detector frame (a change of ``det_t``, its grab time) from ``det_u``,
``det_v``, ``det_w`` and ``det_p``, with the same geometry as the live detector (``vision.runtime.detection_geometry``
on ``--camera`` scaled to the network's input), rebuilds the pose history from the logged frames (wall time,
position, attitude) and steps a ``SightPilot`` row by row through a host that looks like ``TelemetryPilot`` to it:
the wall time of the row as its clock, the logged velocity and body rates, ``TelemetryPilot.pose_at`` over the last
second of logged poses, the game resets where the game clock jumps back, and a vision object returning the latest
detection available at that row. The drone's trajectory is the logged one (open loop): what the replay shows is the
tracker, the choices it makes (target, approach axis, passes) and the rabbit it would fly, under the logged motion.
Parameters are ``SightParams`` with the fly command's flags (``--sight-*``, ``--set name=value``), so a fix can be
A/B tested on a real flight.

When a detection became available. The logger reads the detector after the pilot's step, so a frame that shows up
first in row k reached the pilot in step k or k+1. The pilot marks the step that consumed a fresh frame: ``det_gap``
(seconds since the last fresh frame) is exactly 0 in that row. ``--sync log`` (default) uses that mark, ``first``
hands the frame over in the first row that shows it, ``next`` one row later.

Scoring against the true arches (``--gates``; an arch's visual centre is 1.5 m above its passage point): a track
belongs to the arch whose centre its estimate lies within ``--assoc-m`` (4 m) of, horizontally, for most of its
unpassed (confirmed, when it confirmed) life; a confirmed track near no arch is a phantom. Per approach (each crossing
of an arch's plane by the logged trajectory, as ``liftoff score`` counts them): the distinct confirmed tracks of the
arch alive since the previous crossing (fragmentation), the target's estimate error across and along the arch's
axis by the drone's range to the arch, the approach axis error against the arch's heading 3-12 m out, target
switches, and the passes the tracker declared around it.

The flights of 15 Sep 2026 (the flags each was flown with; ``--preset`` adds them):

  w16_rabbit_a1   --sight-speed 2.5 --sight-z-aim 0.3 --set up_bias=0.5 --set next_min_hits=0
                  (flown on e6b5461: z_aim 0.3 and up_bias 0.5 were the defaults, no next-gate hit minimum)
  w17_rabbit_a2   --sight-speed 2.5 --sight-z-aim 0 --set up_bias=0 --set next_min_hits=10 --set bisector_cap=35
  w18_rabbit_b1   --sight-speed 3.5 --sight-gate-speed 3.2 --sight-turn-gate-speed 2.8 --sight-flow-min 0.6
  w19_rabbit_b2_ground   the w18 flags and --sight-flow-alt ground
  w20_rabbit_rep1, w21_rabbit_rep2 (16 Sep)   the w19 flags (--preset w20 / w21 are the same list)

Every flight was flown on the defaults of its day; ``--set name=value`` puts a later default back, so
`--preset w19 --set ghost_keep_d=0 ...` replays a flight under the pilot it actually flew with.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from .flightlog import attempts, collisions, gate_crossings, load_log
from .frames import quat_wxyz_to_mat
from .sightpilot import CENTRE_UP_M, LOG_COLUMNS, SightParams, SightPilot

FLIGHT_FLAGS = {
    'w16': ['--sight-speed', '2.5', '--sight-z-aim', '0.3', '--set', 'up_bias=0.5', '--set', 'next_min_hits=0'],
    'w17': ['--sight-speed', '2.5', '--sight-z-aim', '0', '--set', 'up_bias=0', '--set', 'next_min_hits=10',
            '--set', 'bisector_cap=35'],
    'w18': ['--sight-speed', '3.5', '--sight-gate-speed', '3.2', '--sight-turn-gate-speed', '2.8',
            '--sight-flow-min', '0.6'],
    'w19': ['--sight-speed', '3.5', '--sight-gate-speed', '3.2', '--sight-turn-gate-speed', '2.8',
            '--sight-flow-min', '0.6', '--sight-flow-alt', 'ground'],
}
# w20 and w21 (16 Sep 2026) were flown with the w19 flags: w20 attempt 2 was the first clean 7/7 lap
FLIGHT_FLAGS['w20'] = FLIGHT_FLAGS['w21'] = FLIGHT_FLAGS['w19']
RANGE_BINS = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, math.inf))
COL = {c: i for i, c in enumerate(LOG_COLUMNS)}


def _wrap_deg(a):
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------------------------------- the replay

def detections_from_log(log: dict[str, np.ndarray], cam, sync: str = 'log') -> list[tuple[int, object]]:
    """(row the pilot gets it, Detection) for every new detector frame of the log, in order. ``cam`` is the camera at
    the network's input resolution (what GateVision.cam is)."""
    from ..vision.runtime import Detection, detection_geometry
    if sync not in ('log', 'first', 'next'):
        raise ValueError(f'sync {sync!r}: log, first or next')
    dt, gap, wall = log['det_t'], log['det_gap'], log['wall']
    n = len(dt)
    fin = np.isfinite(dt)
    if fin.any():
        # the log writes times to four significant digits, which is 0.1 ms on a real wall clock but whole seconds on
        # a small one (the simulator rehearsal starts its clock at 1000): then the grab times are unusable
        moved = int(np.sum((np.diff(log['det_u']) != 0) | (np.diff(log['det_v']) != 0)))
        if len(np.unique(dt[fin])) < 0.5 * moved:
            raise ValueError(f'det_t has only {len(np.unique(dt[fin]))} distinct values for {moved} changed '
                             'detections: this log was written with a small wall clock (a rehearsal), and the '
                             'detections cannot be placed in time')
    out = []
    prev = math.nan
    for i in range(n):
        t = dt[i]
        if not math.isfinite(t) or t == prev:
            prev = t if math.isfinite(t) else prev
            continue
        prev = t
        u, v, w, p = (float(log[k][i]) for k in ('det_u', 'det_v', 'det_w', 'det_p'))
        if not all(math.isfinite(x) for x in (u, v, w, p)):
            continue
        direction, dist = detection_geometry(cam, u, v, w)
        det = Detection(float(t), p, u, v, w, direction, dist, len(out) + 1)
        # the pilot cannot have had the frame before it was grabbed: the logger reads the detector after the step,
        # so the first row that shows a frame can be one whose step ran before the grab
        j = i
        while j < n and dt[j] == t and wall[j] < t:
            j += 1
        row = j if j < n and dt[j] == t else i
        if sync == 'next':
            row += 1
        elif sync == 'log':
            k = row
            while k < n and dt[k] == t and k < row + 50:
                if gap[k] == 0.0:
                    row = k
                    break
                k += 1
        out.append((row, det))
    return out


class ReplayVision:
    """What SightPilot reads from GateVision: ``cam`` and ``get()`` (the replay loop sets ``latest``)."""

    def __init__(self, cam):
        from ..vision.runtime import Detection
        self.cam = cam
        self.latest = Detection()

    def get(self):
        return self.latest


class ReplayHost:
    """The parts of TelemetryPilot the rabbit pilot uses, fed from the log (``pose_at`` is its very function)."""

    def __init__(self, vision, flow_gain: float = 1.0):
        self.vision = vision
        self.now = 0.0
        self.last_R = np.eye(3)
        self.last_vel = np.zeros(3)
        self.omega = np.zeros(3)
        self.flow_gain = flow_gain
        self.sightpilot = None
        self.vision_gate_w = None
        self.vision_passed_t = None
        self.vision_status = ''
        self._passed = []
        self._pose_hist = []           # (wall time, position, attitude) of the last second, as TelemetryPilot keeps it

    def clock(self) -> float:
        return self.now

    def pose_at(self, t: float, tol: float = 0.015):
        from .pilot import TelemetryPilot          # the pilot's own function, on this host's pose history
        return TelemetryPilot.pose_at(self, t, tol)


class RecordingSightPilot(SightPilot):
    """SightPilot that notes why tracks end and which passes it declares; its decisions are the parent's."""

    def __init__(self, host, params, rec: dict):
        self.rec = rec
        super().__init__(host, params)

    def _end(self, T, reason: str, into=None) -> None:
        self.rec['ends'].append({'row': self.rec['row'], 'epoch': self.rec['epoch'], 'id': T.id, 'reason': reason,
                                 'into': into, 'hits': T.hits, 'confirmed': T.confirmed, 'passed': T.passed,
                                 'm': [float(x) for x in T.m]})

    def _full_init(self, p, psi_n, now):
        old = list(getattr(self, 'tracks', []))
        super()._full_init(p, psi_n, now)
        for T in old:
            self._end(T, 'restart')

    def _maintain(self, now, dt, p, R):
        P = self.params
        before, target = list(self.tracks), self.target
        hits = {id(T): T.hits for T in before}
        super()._maintain(now, dt, p, R)
        kept = {id(T) for T in self.tracks}
        for T in before:
            if id(T) in kept:
                continue
            into = None
            life = P.conf_life if T is not target else (P.target_life or math.inf)
            if T.passed or not T.confirmed or now - max(T.t_last, self.t_stall_end) > life:
                reason = 'expired'
            elif T.unseen_in_view > self._ghost_s(T):
                reason = 'ghost'
            elif (math.sqrt(max(T.P[2, 2], 0.0)) < P.low_sigma
                  and T.m[2] < max(P.z_min, self.z_pass_last - P.z_drop_max)):   # same floor as the pilot
                reason = 'low'
            else:
                reason = 'merged'
                twin = next((o for o in self.tracks if id(o) in hits and o.hits - hits[id(o)] >= T.hits), None)
                into = twin.id if twin is not None else None
            self._end(T, reason, into)

    def _pass_check(self, T, n_vec, p, now, age):
        absorbed, ghosts = self.absorbed, self.ghosts
        out = super()._pass_check(T, n_vec, p, now, age)
        if T not in self.tracks and not T.passed:
            if self.absorbed > absorbed and self.target is not None:
                self._end(T, 'fragment', self.target.id)
            elif self.ghosts > ghosts:
                self._end(T, 'ghost_at_pass')
        return out

    def _pass(self, T, now, kind, n_vec, clear_target: bool = True):
        before = list(self.tracks)
        n = self.n_passes
        super()._pass(T, now, kind, n_vec, clear_target=clear_target)
        self.rec['events'].append({'row': self.rec['row'], 'epoch': self.rec['epoch'], 'id': T.id, 'kind': kind,
                                   'm': [float(x) for x in T.m], 'hits': T.hits, 'counted': self.n_passes > n,
                                   'axis_deg': math.degrees(math.atan2(T.n_pass[1], T.n_pass[0]))})
        for o in before:
            if o is not T and o not in self.tracks:
                self._end(o, 'pass_absorbed', T.id)

    def _unpass(self, T, now):
        super()._unpass(T, now)
        self.rec['events'].append({'row': self.rec['row'], 'epoch': self.rec['epoch'], 'id': T.id, 'kind': 'unpass',
                                   'm': [float(x) for x in T.m], 'axis_deg': math.nan, 'hits': T.hits})

    def _intake(self, now, det, pose, p):
        # every sighting that reached the filters, with the pose it was placed by: which arch it really saw is
        # read off the true gate list afterwards (_sighting_arches)
        self._sight = {'row': self.rec['row'], 'epoch': self.rec['epoch'], 'u': float(det.u), 'v': float(det.v),
                       'width_px': float(det.width_px), 't': float(det.t), 'pos': np.asarray(pose[0], dtype=float),
                       'R': np.asarray(pose[1], dtype=float), 'id': None, 'kind': 'elevation'}
        self.rec['sightings'].append(self._sight)
        super()._intake(now, det, pose, p)

    def _update(self, now, z, Rm, r, rho, pg, crop):
        absorbed, behind = self.absorbed, self.behind
        T = super()._update(now, z, Rm, r, rho, pg, crop)
        s = getattr(self, '_sight', None)
        if s is not None:
            s['id'] = None if T is None else T.id
            s['kind'] = ('track' if T is not None else 'absorbed' if self.absorbed > absorbed
                         else 'behind' if self.behind > behind else 'dropped')
            s['m'] = [float(x) for x in z]
        return T


def replay(log: dict[str, np.ndarray], params: SightParams, cam, sync: str = 'log', flow_gain: float = 1.0,
           clock_offset: float = 0.0) -> dict:
    """Step a SightPilot through every row of a loaded fly --log. Returns the replayed LOG_COLUMNS per row
    (``cols``), the status lines, per-row snapshots of every track (``tracks``: row, epoch, id, x, y, z, confirmed,
    passed, hits), the declared passes and un-passes (``events``), why tracks ended (``ends``) and ``epoch`` per row
    (it counts the game resets, which restart the track ids)."""
    dets = detections_from_log(log, cam, sync)
    vision = ReplayVision(cam)
    host = ReplayHost(vision, flow_gain)
    rec = {'row': -1, 'epoch': 0, 'events': [], 'ends': [], 'sightings': []}
    sp = RecordingSightPilot(host, params, rec)
    host.sightpilot = sp
    n = len(log['wall'])
    cols = np.full((n, len(LOG_COLUMNS)), np.nan)
    status = [''] * n
    epoch = np.zeros(n, dtype=np.int64)
    snap = {k: [] for k in ('row', 'epoch', 'id', 'x', 'y', 'z', 'confirmed', 'passed', 'hits')}
    wall, ts = log['wall'], log['ts']
    P = np.c_[log['px'], log['py'], log['pz']]
    V = np.c_[log['vx'], log['vy'], log['vz']]
    Q = np.c_[log['qw'], log['qx'], log['qy'], log['qz']]
    W = np.c_[log['wx'], log['wy'], log['wz']]
    k = 0
    last_ts = -1.0
    for r in range(n):
        rec['row'] = r
        if ts[r] < last_ts - 0.5:              # the drone was reset in the game: TelemetryPilot.reset
            rec['epoch'] += 1
            host._pose_hist = []
            host.vision_gate_w = None
            host.vision_passed_t = None
            host._passed = []
            sp.reset()
        last_ts = ts[r]
        now = float(wall[r]) + clock_offset
        host.now = now
        R = quat_wxyz_to_mat(Q[r])
        host.last_R = R
        host.last_vel = V[r].copy()
        host.omega = W[r].copy()
        host._pose_hist.append((now, P[r].copy(), R.copy()))
        while host._pose_hist and now - host._pose_hist[0][0] > 1.0:
            host._pose_hist.pop(0)
        while k < len(dets) and dets[k][0] <= r:
            vision.latest = dets[k][1]
            k += 1
        sp.goal(P[r].copy())
        cols[r] = sp.log_values(vision.latest)
        status[r] = host.vision_status
        epoch[r] = rec['epoch']
        for T in sp.tracks:
            snap['row'].append(r)
            snap['epoch'].append(rec['epoch'])
            snap['id'].append(T.id)
            snap['x'].append(float(T.m[0]))
            snap['y'].append(float(T.m[1]))
            snap['z'].append(float(T.m[2]))
            snap['confirmed'].append(T.confirmed)
            snap['passed'].append(T.passed)
            snap['hits'].append(T.hits)
    tracks = {k2: np.asarray(v) for k2, v in snap.items()}
    return {'cols': cols, 'status': status, 'epoch': epoch, 'tracks': tracks, 'events': rec['events'],
            'ends': rec['ends'], 'sightings': rec['sightings'], 'n_detections': len(dets), 'sync': sync,
            'errors': sp.errors, 'cam': cam}


# ---------------------------------------------------------------------------------------------------- validation

def _logged(log, name):
    return log[name] if name in log else np.full(len(log['wall']), np.nan)


def validate(log: dict[str, np.ndarray], res: dict) -> dict:
    """How closely the replay reproduces the logged rabbit columns."""
    C = res['cols']
    n = len(C)
    out = {'rows': n}
    lid = np.nan_to_num(_logged(log, 'tgt_id'), nan=-1.0)
    rid = np.nan_to_num(C[:, COL['tgt_id']], nan=-1.0)
    same = lid == rid
    has = lid >= 0
    out['tgt_id_match'] = float(same.mean())
    out['tgt_id_match_when_logged_target'] = float(same[has].mean()) if has.any() else math.nan
    both = same & has
    if both.any():
        d = np.linalg.norm(np.c_[[_logged(log, c)[both] - C[both, COL[c]] for c in ('tgt_x', 'tgt_y', 'tgt_z')]].T,
                           axis=1)
        out['tgt_xyz_err_median'] = float(np.median(d))
        out['tgt_xyz_err_p99'] = float(np.percentile(d, 99))
        out['tgt_xyz_within_5cm'] = float((d < 0.05).mean())
        ax = np.abs(_wrap_deg(_logged(log, 'axis_deg')[both] - C[both, COL['axis_deg']]))
        ax = ax[np.isfinite(ax)]
        out['axis_within_1deg'] = float((ax < 1.0).mean()) if len(ax) else math.nan
    # the target's estimate, whatever its id: is the replay flying at the same arch?
    pair = has & (rid >= 0)
    if pair.any():
        dp = np.hypot(_logged(log, 'tgt_x')[pair] - C[pair, COL['tgt_x']],
                      _logged(log, 'tgt_y')[pair] - C[pair, COL['tgt_y']])
        out['tgt_pos_median_any_id'] = float(np.median(dp))
        out['tgt_within_1m_any_id'] = float((dp < 1.0).mean())
    for c in ('n_passes', 'mode', 'next_id', 'n_conf', 'n_tent', 'pass_kind'):
        lv = np.nan_to_num(_logged(log, c), nan=-1.0)
        rv = np.nan_to_num(C[:, COL[c]], nan=-1.0)
        out[f'{c}_match'] = float((lv == rv).mean())
    # the tracker's counters at the end (a log's can start above zero: the pilot flew before its first row)
    out['counters'] = {}
    for c in ('rej_elev', 'rej_stale', 'rej_offaxis', 'absorbed', 'low', 'reseeds', 'ghosts', 'orphans', 'unpasses',
              'behind', 'goal_clips'):
        lv, rv = _logged(log, c), C[:, COL[c]]
        lv, rv = lv[np.isfinite(lv)], rv[np.isfinite(rv)]
        if len(lv) and len(rv):
            out['counters'][c] = [float(lv[-1]), float(rv[-1])]
    rb = np.hypot(_logged(log, 'rb_x') - C[:, COL['rb_x']], _logged(log, 'rb_y') - C[:, COL['rb_y']])
    rb = rb[np.isfinite(rb)]
    out['rabbit_xy_err_median'] = float(np.median(rb)) if len(rb) else math.nan
    out['rabbit_xy_within_10cm'] = float((rb < 0.1).mean()) if len(rb) else math.nan
    out['rabbit_xy_err_p99'] = float(np.percentile(rb, 99)) if len(rb) else math.nan
    # pass events (pass_kind 1-3) in the log and in the replay, matched within 5 rows
    lp = np.where(np.isin(_logged(log, 'pass_kind'), (1, 2, 3)))[0]
    rp = np.where(np.isin(C[:, COL['pass_kind']], (1, 2, 3)))[0]
    matched = sum(1 for i in lp if len(rp) and np.min(np.abs(rp - i)) <= 5)
    out['passes_logged'], out['passes_replayed'], out['passes_matched'] = int(len(lp)), int(len(rp)), int(matched)
    # the target mismatches: a few rows early or late (timing), or a different choice
    bad = np.where(~same)[0]
    shifted = sum(1 for i in bad if rid[i] in lid[max(i - 5, 0):i + 6])
    out['tgt_id_mismatch_rows'] = int(len(bad))
    out['tgt_id_mismatch_within_5_rows'] = int(shifted)
    runs = []
    if len(bad):
        start = prev = bad[0]
        for i in list(bad[1:]) + [None]:
            if i is not None and i == prev + 1:
                prev = i
                continue
            runs.append((int(start), int(prev)))
            if i is not None:
                start = prev = i
    runs.sort(key=lambda ab: ab[0] - ab[1])
    wall = log['wall']
    out['tgt_id_mismatch_runs'] = [{'row': a, 'rows': b - a + 1, 's': float(wall[b] - wall[a]),
                                    't_wall_rel': float(wall[a] - wall[0]),
                                    'logged': sorted({int(x) for x in lid[a:b + 1]}),
                                    'replayed': sorted({int(x) for x in rid[a:b + 1]})} for a, b in runs[:10]]
    return out


# ---------------------------------------------------------------------------------------------------- scoring

def _sighting_arches(res: dict, gates: list[dict], px_tol: float = 25.0) -> None:
    """Which arch each sighting really saw: the true arch whose visual centre projects nearest the detected centre
    in the frame the sighting was grabbed in (``arch`` None = it saw nothing that is an arch)."""
    cam = res.get('cam')
    cen = np.array([np.asarray(g['pos'], dtype=float) + [0.0, 0.0, CENTRE_UP_M] for g in gates])
    for s in res['sightings']:
        s['arch'], s['arch_px'] = None, math.inf
        if cam is None:
            continue
        rel = (cen - s['pos']) @ s['R']                      # world -> body (R is body->world)
        px, ok = cam.project_body(rel)
        d = np.hypot(px[:, 0] - s['u'], px[:, 1] - s['v'])
        d = np.where(ok & (np.linalg.norm(rel, axis=1) < 60.0), d, np.inf)
        g = int(np.argmin(d))
        if math.isfinite(d[g]) and d[g] < max(px_tol, 0.7 * s['width_px']):
            s['arch'], s['arch_px'] = g, float(d[g])


def _track_table(log, res, gates, assoc_m, ray_deg=6.0):
    """Per track (epoch, id): life, hits, the arch it belongs to (or None), where it was.

    An estimate belongs to the arch it sits within ``assoc_m`` of, horizontally, for most of its unpassed life
    (``assoc`` 'near'). Failing that, to the arch whose bearing from the drone it shares to within ``ray_deg``
    (``assoc`` 'ray'): the bearing is the accurate part of a sighting, so a track on an arch's bearing at the wrong
    range is that arch badly placed, not a phantom. Failing both, to the arch most of its sightings actually saw
    (``assoc`` 'seen'; the detected centre is matched to the arches projected into that frame). Nothing: a phantom."""
    tr = res['tracks']
    wall = log['wall']
    cen = np.array([np.asarray(g['pos'], dtype=float) + [0.0, 0.0, CENTRE_UP_M] for g in gates])
    P = np.c_[log['px'], log['py'], log['pz']]
    C = res['cols']
    tgt_rows = {}
    rid = np.nan_to_num(C[:, COL['tgt_id']], nan=-1.0).astype(int)
    for r in np.where(rid >= 0)[0]:
        key = (int(res['epoch'][r]), int(rid[r]))
        tgt_rows.setdefault(key, []).append(int(r))
    ends = {}
    for e in res['ends']:
        ends.setdefault((e['epoch'], e['id']), e)
    seen: dict = {}                       # (epoch, id) -> {arch or None: sightings}
    for s in res['sightings']:
        if s.get('id') is None:
            continue
        seen.setdefault((s['epoch'], s['id']), {}).setdefault(s.get('arch'), 0)
        seen[(s['epoch'], s['id'])][s.get('arch')] += 1
    keys = np.c_[tr['epoch'], tr['id']] if len(tr['id']) else np.zeros((0, 2), dtype=int)
    table = {}
    if not len(keys):
        return table
    order = np.lexsort((tr['row'], tr['id'], tr['epoch']))
    ks = keys[order]
    cuts = np.r_[0, np.where(np.any(np.diff(ks, axis=0) != 0, axis=1))[0] + 1, len(ks)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        sel = order[a:b]
        key = (int(tr['epoch'][sel[0]]), int(tr['id'][sel[0]]))
        rows = tr['row'][sel]
        conf = tr['confirmed'][sel]
        passed = tr['passed'][sel]
        live = ~passed
        use = live & conf if (live & conf).any() else (live if live.any() else np.ones(len(sel), bool))
        X = np.c_[tr['x'][sel], tr['y'][sel], tr['z'][sel]]
        D = np.hypot(X[use, 0, None] - cen[None, :, 0], X[use, 1, None] - cen[None, :, 1])
        frac = (D < assoc_m).mean(axis=0)
        g = int(np.argmax(frac))
        med = np.median(X[use], axis=0)
        dmed = np.hypot(med[0] - cen[:, 0], med[1] - cen[:, 1])
        near = int(np.argmin(dmed))
        dr = P[rows[use]]
        drone = np.median(dr, axis=0)
        # the bearing of the estimate from the drone against each arch's (the accurate part of a sighting)
        b_est = np.arctan2(X[use, 1] - dr[:, 1], X[use, 0] - dr[:, 0])
        b_arch = np.arctan2(cen[None, :, 1] - dr[:, 1, None], cen[None, :, 0] - dr[:, 0, None])
        b_err = np.abs(_wrap_deg(np.degrees(b_est[:, None] - b_arch)))
        b_med = np.median(b_err, axis=0)
        gb = int(np.argmin(b_med))
        saw = seen.get(key, {})
        n_saw = sum(saw.values())
        n_id = sum(v for k2, v in saw.items() if k2 is not None)
        gs, n_gs = (max(((k2, v) for k2, v in saw.items() if k2 is not None), key=lambda kv: kv[1], default=(None, 0)))
        ok_seen = gs is not None and n_id >= 3 and n_gs >= 0.5 * n_id and n_id >= 0.25 * max(n_saw, 1)
        assoc = ('near' if frac[g] >= 0.5 else 'seen' if ok_seen else 'ray' if b_med[gb] <= ray_deg else None)
        arch = {'near': g, 'ray': gb, 'seen': gs}.get(assoc)
        ga = arch if arch is not None else (gb if assoc is None and np.isfinite(b_med[gb]) else near)
        rr = np.hypot(X[use, 0] - dr[:, 0], X[use, 1] - dr[:, 1]) / np.maximum(
            np.hypot(cen[ga, 0] - dr[:, 0], cen[ga, 1] - dr[:, 1]), 1e-6)
        arch_d = float(np.median(np.hypot(X[use, 0] - cen[ga, 0], X[use, 1] - cen[ga, 1])))
        others = sorted(((v, k2) for k2, v in saw.items() if k2 is not None and k2 != arch), reverse=True)
        e = ends.get(key)
        table[key] = {
            'epoch': key[0], 'id': key[1], 'arch': arch, 'assoc': assoc, 'arch_frac': float(frac[g]),
            'ray_arch': gb, 'ray_bearing_deg': float(b_med[gb]), 'range_ratio': float(np.median(rr)),
            'arch_d_median': arch_d, 'mixed_arch': (int(others[0][1]) if others and others[0][0] >= 0.2 * max(n_saw, 1)
                                                    else None),
            'saw': {('none' if k2 is None else int(k2)): int(v) for k2, v in sorted(saw.items(), key=lambda kv: -kv[1])},
            'row_first': int(rows[0]), 'row_last': int(rows[-1]),
            'row_confirmed': int(rows[np.argmax(conf)]) if conf.any() else None,
            'row_passed': int(rows[np.argmax(passed)]) if passed.any() else None,
            'life_s': float(wall[rows[-1]] - wall[rows[0]]), 'hits': int(tr['hits'][sel].max()),
            'confirmed': bool(conf.any()), 'passed': bool(passed.any()),
            'median_pos': [round(float(x), 2) for x in med], 'nearest_arch': near, 'nearest_arch_d': float(dmed[near]),
            'drone_median_pos': [round(float(x), 2) for x in drone],
            'drone_range_median': float(np.median(np.hypot(X[use, 0] - P[rows[use], 0], X[use, 1] - P[rows[use], 1]))),
            'target_rows': len(tgt_rows.get(key, [])),
            'end': e['reason'] if e else ('alive' if rows[-1] == len(wall) - 1 else 'removed'),
            'end_into': e['into'] if e else None,
        }
    return table


def _range_bias(sights, cen, bins=((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, math.inf))) -> list:
    """The range a sighting of a known arch was placed at, against the true range to it (and the bearing error):
    what the width-to-range model and the detector's box width are worth on this flight."""
    rows = []
    for s in sights:
        if s.get('arch') is None or 'm' not in s:
            continue
        rel = np.asarray(s['m'], dtype=float) - s['pos']
        true = cen[s['arch']] - s['pos']
        rng, tr = float(np.linalg.norm(rel)), float(np.linalg.norm(true))
        if rng < 1e-6 or tr < 1e-6:
            continue
        cosb = float(rel @ true) / (rng * tr)
        rows.append((tr, rng / tr, math.degrees(math.acos(min(max(cosb, -1.0), 1.0)))))
    out = []
    for lo, hi in bins:
        b = [r for r in rows if lo <= r[0] < hi]
        out.append(None if not b else {'range': [lo, hi], 'n': len(b),
                                       'range_ratio_median': float(np.median([r[1] for r in b])),
                                       'bearing_err_median': float(np.median([r[2] for r in b]))})
    return out


def _bin_stats(rng, across, along, dz):
    out = []
    for lo, hi in RANGE_BINS:
        m = (rng >= lo) & (rng < hi)
        if not m.any():
            out.append(None)
            continue
        a, b, c = across[m], along[m], dz[m]
        out.append({'range': [lo, hi], 'n': int(m.sum()), 'across_median': float(np.median(a)),
                    'across_abs_median': float(np.median(np.abs(a))), 'across_abs_p90': float(np.percentile(np.abs(a), 90)),
                    'along_median': float(np.median(b)), 'along_abs_median': float(np.median(np.abs(b))),
                    'dz_median': float(np.median(c))})
    return out


def score(log: dict[str, np.ndarray], res: dict, gates: list[dict], assoc_m: float = 4.0,
          ray_deg: float = 6.0) -> dict:
    """The replayed tracker against the true arches: per attempt, per approach of an arch, and its phantoms."""
    wall, ts = log['wall'], log['ts']
    C = res['cols']
    P = np.c_[log['px'], log['py'], log['pz']]
    V = np.c_[log['vx'], log['vy'], log['vz']]
    cen = np.array([np.asarray(g['pos'], dtype=float) + [0.0, 0.0, CENTRE_UP_M] for g in gates])
    head = np.array([float(g['heading']) for g in gates])
    _sighting_arches(res, gates)
    table = _track_table(log, res, gates, assoc_m, ray_deg)
    rid = np.nan_to_num(C[:, COL['tgt_id']], nan=-1.0).astype(int)
    ep = res['epoch']

    def arch_of(r):
        if rid[r] < 0:
            return None, None
        t = table.get((int(ep[r]), int(rid[r])))
        return (t['arch'] if t else None), t

    out = {'assoc_m': assoc_m, 'ray_deg': ray_deg, 'attempts': [], 'tracks': list(table.values())}
    for ai, idx in enumerate(attempts(log)):
        r0, r1 = int(idx[0]), int(idx[-1])
        t_rel = lambda r: float(ts[r] - ts[r0])                                     # noqa: E731
        Pa, Va = P[idx], V[idx]
        ts_a = ts[idx] - ts[r0]
        air = (Pa[:, 2] > 0.5) & (log['phase'][idx] > 3.0)
        hits = collisions(Pa, Va, ts_a, air)
        cr = gate_crossings(Pa, idx.astype(float), gates)
        for c in cr:
            c['row'] = int(c['t'])
            c['t'] = t_rel(c['row'])
            g = np.asarray(gates[c['gate']]['pos'], dtype=float)
            c['hit'] = any(abs(h['t'] - c['t']) < 1.0 and np.linalg.norm(np.asarray(h['pos'])[:2] - g[:2]) < 5.0
                           for h in hits)
            c['through'] = c['through'] and not c['hit']
        cr.sort(key=lambda c: c['row'])
        delim = [c for c in cr if abs(c['lateral_m']) < 6.0]
        tracks_here = [t for t in table.values() if r0 <= t['row_first'] <= r1]
        # declared passes of this attempt
        passes = []
        for e in res['events']:
            if not r0 <= e['row'] <= r1:
                continue
            t = table.get((e['epoch'], e['id']))
            ent = {'row': e['row'], 't': t_rel(e['row']), 'kind': e['kind'], 'id': e['id'],
                   'arch': t['arch'] if t else None, 'm': [round(x, 2) for x in e['m']], 'crossing': None,
                   'counted': e.get('counted', True)}   # False: the arch already on the books, not a second gate
            passes.append(ent)
        approaches = []
        windows = []
        for c in cr:
            g = c['gate']
            prev = [d['row'] for d in delim if d['row'] < c['row']] + \
                   [d['row'] for d in cr if d['gate'] == g and d['row'] < c['row']]
            windows.append((g, max(prev + [r0]), c['row'], c))
        for g in range(len(gates)):
            if any(w[0] == g for w in windows):
                continue
            mine = [t for t in tracks_here if t['arch'] == g and t['confirmed']]
            if mine:
                windows.append((g, min(t['row_first'] for t in mine), min(max(t['row_last'] for t in mine), r1), None))
        windows.sort(key=lambda w: (w[2], w[0]))
        for g, s, e, c in windows:
            n_vec = np.array([math.cos(head[g]), math.sin(head[g])])
            l_vec = np.array([-n_vec[1], n_vec[0]])
            tr = res['tracks']
            m = (tr['row'] >= s) & (tr['row'] <= e) & tr['confirmed'] & ~tr['passed']
            ids_conf = sorted({int(i) for i, epo in zip(tr['id'][m], tr['epoch'][m])
                               if table.get((int(epo), int(i)), {}).get('arch') == g})
            m2 = (tr['row'] >= s) & (tr['row'] <= e) & ~tr['passed']
            ids_all = sorted({int(i) for i, epo in zip(tr['id'][m2], tr['epoch'][m2])
                              if table.get((int(epo), int(i)), {}).get('arch') == g})
            ids_ray = sorted({int(i) for i, epo in zip(tr['id'][m], tr['epoch'][m])
                              if table.get((int(epo), int(i)), {}).get('assoc') == 'ray'
                              and table.get((int(epo), int(i)), {}).get('arch') == g})
            rows = np.arange(s, e + 1)
            keys = [(int(ep[r]), int(rid[r])) if rid[r] >= 0 else None for r in rows]
            ta = [None if k is None else table.get(k, {}).get('arch') for k in keys]
            on = np.array([x == g for x in ta], dtype=bool)
            rr = rows[on]
            est = np.c_[C[rr, COL['tgt_x']], C[rr, COL['tgt_y']], C[rr, COL['tgt_z']]]
            err = est - cen[g]
            rng = np.hypot(cen[g, 0] - P[rr, 0], cen[g, 1] - P[rr, 1])
            across, along, dz = err[:, :2] @ l_vec, err[:, :2] @ n_vec, err[:, 2]
            before = (P[rr, :2] - cen[g, :2]) @ n_vec < 0.0
            ax_m = before & (rng >= 3.0) & (rng <= 12.0)
            ax_err = _wrap_deg(C[rr[ax_m], COL['axis_deg']] - math.degrees(head[g]))
            ax_err = ax_err[np.isfinite(ax_err)]
            # target switches in the window (rows without a target skipped)
            seq = [k for k in keys if k is not None]
            sw = {'same_arch': 0, 'other_arch': 0, 'to_phantom': 0, 'from_phantom': 0}
            for a, b in zip(seq, seq[1:]):
                if a == b:
                    continue
                ga = table.get(a, {}).get('arch')
                gb = table.get(b, {}).get('arch')
                if gb is None:
                    sw['to_phantom'] += 1
                elif ga is None:
                    sw['from_phantom'] += 1
                elif ga == gb:
                    sw['same_arch'] += 1
                else:
                    sw['other_arch'] += 1
            ap = {'gate': g, 'row_start': int(s), 'row_end': int(e), 't_start': t_rel(s), 't_end': t_rel(e),
                  'crossing': None, 'tracks_confirmed': ids_conf, 'tracks_all': ids_all, 'tracks_ray': ids_ray,
                  'fragments': len(ids_conf), 'target_rows': int(on.sum()),
                  'target_ids': sorted({k[1] for k, x in zip(keys, ta) if k is not None and x == g}),
                  'err_by_range': _bin_stats(rng, across, along, dz),
                  'axis_err_3_12': ({'n': int(len(ax_err)), 'median': float(np.median(ax_err)),
                                     'abs_p90': float(np.percentile(np.abs(ax_err), 90)),
                                     'abs_max': float(np.abs(ax_err).max())} if len(ax_err) else None),
                  'switches': sw, 'target_distinct': len(set(seq)),
                  'passes': []}
            if c is not None:
                ga, t = arch_of(c['row'])
                ap['crossing'] = {'t': c['t'], 'row': c['row'], 'lateral_m': c['lateral_m'], 'dz_m': c['dz_m'],
                                  'through': c['through'], 'hit': c['hit'],
                                  'target_at_crossing': (t['id'] if t else -1), 'target_arch': ga}
                if ga == g:
                    e_x = np.array([C[c['row'], COL['tgt_x']], C[c['row'], COL['tgt_y']]]) - cen[g, :2]
                    ap['crossing']['target_across_err'] = float(e_x @ l_vec)
            approaches.append(ap)
        # match declared passes to crossings of their arch (within 3 s), each crossing once
        used = set()
        for p in passes:
            if p['kind'] == 'unpass':
                continue
            best = None
            for k, ap in enumerate(approaches):
                c = ap['crossing']
                if c is None or ap['gate'] != p['arch'] or k in used:
                    continue
                dtc = float(wall[p['row']] - wall[c['row']])
                if abs(dtc) <= 3.0 and (best is None or abs(dtc) < abs(best[1])):
                    best = (k, dtc)
            if best is not None:
                used.add(best[0])
                p['crossing'] = {'gate': approaches[best[0]]['gate'], 'dt': best[1]}
                approaches[best[0]]['passes'].append({'kind': p['kind'], 'id': p['id'], 'dt': best[1],
                                                      'counted': p['counted']})
        for p in passes:            # passes of an arch that was not crossed then: attach to the nearest approach of it
            if p['crossing'] is None and p['kind'] != 'unpass':
                cand = [ap for ap in approaches if ap['gate'] == p['arch']]
                if cand:
                    ap = min(cand, key=lambda a: abs(a['row_end'] - p['row']))
                    ap['passes'].append({'kind': p['kind'], 'id': p['id'], 'counted': p['counted'],
                                         'dt': float(wall[p['row']] - wall[ap['row_end']]), 'unmatched': True})
        phantoms = []
        for t in tracks_here:
            if t['arch'] is None and t['confirmed']:
                phantoms.append({k: t[k] for k in ('id', 'hits', 'life_s', 'median_pos', 'nearest_arch', 'nearest_arch_d',
                                                   'drone_median_pos', 'drone_range_median', 'target_rows', 'end',
                                                   'ray_arch', 'ray_bearing_deg', 'range_ratio', 'saw')}
                                | {'t_first': t_rel(t['row_first']), 't_last': t_rel(t['row_last'])})
        allsw = [(int(ep[r]), int(rid[r])) for r in range(r0, r1 + 1) if rid[r] >= 0]
        n_sw = sum(1 for a, b in zip(allsw, allsw[1:]) if a != b)
        sights = [s for s in res['sightings'] if r0 <= s['row'] <= r1]
        sight_stats = {'total': len(sights), 'saw_nothing': sum(1 for s in sights if s.get('arch') is None),
                       'per_arch': {int(g): sum(1 for s in sights if s.get('arch') == g) for g in range(len(gates))},
                       'kinds': {k: sum(1 for s in sights if s['kind'] == k)
                                 for k in ('track', 'absorbed', 'behind', 'elevation', 'dropped')},
                       'range': _range_bias(sights, cen)}
        out['attempts'].append({
            'attempt': ai + 1, 'rows': [r0, r1], 'seconds': float(ts[r1] - ts[r0]),
            'crossings': [{k: c[k] for k in ('gate', 't', 'row', 'lateral_m', 'dz_m', 'through', 'hit')} for c in cr],
            'gates_through': sorted({c['gate'] for c in cr if c['through']}),
            'approaches': approaches, 'passes': passes,
            'false_passes': [p for p in passes if p['kind'] != 'unpass' and p['crossing'] is None],
            'missed_passes': [{'gate': ap['gate'], 't': ap['crossing']['t'], 'lateral_m': ap['crossing']['lateral_m']}
                              for ap in approaches if ap['crossing'] is not None and not ap['passes']],
            'phantoms': phantoms, 'target_switches': n_sw,
            'tracks': len(tracks_here), 'tracks_confirmed': sum(1 for t in tracks_here if t['confirmed']),
            'tracks_by_arch': {g: [t for t in tracks_here if t['confirmed'] and t['arch'] == g]
                               for g in range(len(gates)) if any(t['confirmed'] and t['arch'] == g for t in tracks_here)},
            'tracks_on_a_ray': sum(1 for t in tracks_here if t['confirmed'] and t['assoc'] in ('ray', 'seen')),
            'sightings': sight_stats,
            'target_err_all': _pooled(approaches),
        })
    return out


def _pooled(approaches) -> list:
    """Across/along error bins pooled over the approaches (median of per-approach medians weighted by rows)."""
    out = []
    for k, (lo, hi) in enumerate(RANGE_BINS):
        bs = [ap['err_by_range'][k] for ap in approaches if ap['err_by_range'][k]]
        if not bs:
            out.append(None)
            continue
        w = np.array([b['n'] for b in bs], dtype=float)
        f = lambda key: float(np.average([b[key] for b in bs], weights=w))    # noqa: E731
        out.append({'range': [lo, hi], 'n': int(w.sum()), 'across_abs_median': f('across_abs_median'),
                    'across_median': f('across_median'), 'along_median': f('along_median'),
                    'along_abs_median': f('along_abs_median')})
    return out


# ---------------------------------------------------------------------------------------------------- report

def _fmt_bins(bins, key_a='across_abs_median', key_b='along_median') -> str:
    s = []
    for b in bins:
        s.append('   -    ' if b is None else f'{b[key_a]:4.2f}/{b[key_b]:+4.1f}')
    return ' '.join(f'{x:>9s}' for x in s)


def describe(name: str, val: dict | None, sc: dict) -> str:
    lines = []
    if val is not None:
        lines.append(f'{name}: replay vs log over {val["rows"]} rows: tgt_id {100 * val["tgt_id_match"]:.1f}% '
                     f'({100 * val["tgt_id_match_when_logged_target"]:.1f}% of rows with a target), target estimate '
                     f'{val.get("tgt_pos_median_any_id", math.nan):.3f} m apart (median, whatever the id; within 1 m '
                     f'{100 * val.get("tgt_within_1m_any_id", math.nan):.1f}%), same id and place to 5 cm '
                     f'{100 * val.get("tgt_xyz_within_5cm", math.nan):.1f}%, n_passes {100 * val["n_passes_match"]:.1f}%, '
                     f'mode {100 * val["mode_match"]:.1f}%, next_id {100 * val["next_id_match"]:.1f}%, '
                     f'n_conf {100 * val["n_conf_match"]:.1f}%, axis within 1 deg '
                     f'{100 * val.get("axis_within_1deg", math.nan):.1f}%, rabbit median '
                     f'{val["rabbit_xy_err_median"]:.2f} m, passes {val["passes_matched"]}/{val["passes_logged"]} '
                     f'matched (replay {val["passes_replayed"]})')
        lines.append('  counters at the end (log / replay): '
                     + ', '.join(f'{k} {a:.0f}/{b:.0f}' for k, (a, b) in val['counters'].items()))
    for a in sc['attempts']:
        lines.append(f'{name} attempt {a["attempt"]} ({a["seconds"]:.0f} s): gates through {a["gates_through"]}, '
                     f'{a["tracks_confirmed"]} confirmed tracks of {a["tracks"]} ({a["tracks_on_a_ray"]} of them an '
                     f'arch badly ranged along its bearing), {len(a["phantoms"])} confirmed '
                     f'phantoms, {a["target_switches"]} target switches, passes declared '
                     f'{sum(1 for p in a["passes"] if p["kind"] != "unpass")} ({len(a["false_passes"])} not at a '
                     f'crossing of their arch), crossings without a declared pass {len(a["missed_passes"])}')
        lines.append('  gate    t(s)  crossing       | frag (confirmed tracks) | target estimate vs the arch, '
                     '|across|/along m, by range 0-5, 5-10, 10-20, 20+ m      | axis err 3-12 m | switch s/o/p | '
                     'declared pass')
        for ap in a['approaches']:
            c = ap['crossing']
            if c is None:
                cs = f'{"not crossed":<14s}'
                t = ap['t_end']
            else:
                res = 'through' if c['through'] else ('hit' if c['hit'] else 'miss')
                cs = f'{c["lateral_m"]:+5.1f} {res:<8s}'
                t = c['t']
            ids = ','.join(f'{i}~' if i in ap['tracks_ray'] else str(i) for i in ap['tracks_confirmed'])
            ax = ap['axis_err_3_12']
            axs = f'{ax["median"]:+5.0f} (max {ax["abs_max"]:3.0f})' if ax else '       -       '
            sw = ap['switches']
            sws = f'{sw["same_arch"]}/{sw["other_arch"]}/{sw["to_phantom"] + sw["from_phantom"]}'
            ps = ', '.join(f'{p["kind"]} #{p["id"]} {p["dt"]:+.1f}s' + ('' if p.get('counted', True) else ' [same arch]')
                           + (' (no crossing)' if p.get('unmatched') else '')
                           for p in ap['passes']) or '-'
            lines.append(f'  g{ap["gate"]}    {t:6.1f}  {cs} | {ap["fragments"]:2d} ({ids:<18s}) | '
                         f'{_fmt_bins(ap["err_by_range"])} | {axs} | {sws:>8s}     | {ps}')
        for g, ts_ in a['tracks_by_arch'].items():
            lines.append(f'  arch {g} tracks: ' + ', '.join(
                f'#{t["id"]}({t["assoc"]},{t["hits"]}h,{t["life_s"]:.1f}s,{t["arch_d_median"]:.1f}m,'
                f'range x{t["range_ratio"]:.2f},tgt {t["target_rows"]}r,{t["end"]}'
                + (f',mixed with {t["mixed_arch"]}' if t['mixed_arch'] is not None else '') + ')' for t in ts_))
        for p in a['false_passes']:
            lines.append(f'  false pass: {p["kind"]} of track {p["id"]} (arch {p["arch"]}) at {p["t"]:.1f} s, '
                         f'estimate {p["m"]}')
        for ph in a['phantoms']:
            lines.append(f'  phantom #{ph["id"]}: {ph["t_first"]:.1f}-{ph["t_last"]:.1f} s, {ph["hits"]} hits, at '
                         f'{ph["median_pos"]} ({ph["nearest_arch_d"]:.1f} m from arch {ph["nearest_arch"]}, '
                         f'{ph["ray_bearing_deg"]:.0f} deg off arch {ph["ray_arch"]}\'s bearing), sightings saw '
                         f'{ph["saw"]}, drone at {ph["drone_median_pos"]} {ph["drone_range_median"]:.0f} m away, '
                         f'target {ph["target_rows"]} rows, ended {ph["end"]}')
        sg = a['sightings']
        lines.append(f'  sightings: {sg["total"]} ({sg["saw_nothing"]} of nothing that is an arch), per arch '
                     + ' '.join(f'g{g}:{n}' for g, n in sg['per_arch'].items())
                     + '; range placed / true range (bearing error) by true range: '
                     + ' '.join('-' if b is None else f'{b["range"][0]:.0f}-{b["range"][1]:.0f} m '
                                f'x{b["range_ratio_median"]:.2f} ({b["bearing_err_median"]:.1f} deg, n {b["n"]})'
                                for b in sg['range']))
        pe = a['target_err_all']
        lines.append('  all approaches, target |across| median / along median by range: '
                     + ' '.join('-' if b is None else f'{b["range"][0]:.0f}-{b["range"][1]:.0f} m '
                                f'{b["across_abs_median"]:.2f}/{b["along_median"]:+.2f} (n {b["n"]})' for b in pe))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------------- CLI

def replay_epilog() -> str:
    """The per-flight flags, for the command's help."""
    lines = ['the flags each game flight of 15 Sep 2026 was flown with (--preset adds them):']
    for k, v in FLIGHT_FLAGS.items():
        lines.append(f'  --preset {k:<4s} = ' + ' '.join(v))
    lines.append('e.g.  haltere liftoff replay-sight data/liftoff/logs/w17_rabbit_a2.csv --preset w17 '
                 '--json data/liftoff/replay/w17.json')
    return '\n'.join(lines)


def add_cli_args(q) -> None:
    from .sightpilot import add_cli_args as add_sight_args
    q.add_argument('log', help='a `liftoff fly --sight rabbit --log` CSV')
    q.add_argument('--camera', default='configs/camera_seat.yaml', help='camera calibration the flight used')
    q.add_argument('--gates', default='configs/gates_strawbale.json', help='true gate list to score against ("" = none)')
    q.add_argument('--preset', choices=sorted(FLIGHT_FLAGS), default=None,
                   help='the flags of one of the 15 Sep 2026 flights (flags given here win; --set is applied last)')
    q.add_argument('--flow-gain', type=float, default=1.0,
                   help="the fly command's --flow-gain (the speed sense's highest value)")
    q.add_argument('--set', dest='sight_set', action='append', default=[], metavar='NAME=VALUE',
                   help='override a SightParams field (same as --sight-set), e.g. --set next_min_hits=10')
    q.add_argument('--sync', choices=['log', 'first', 'next'], default='log',
                   help='when a detection reaches the pilot: the step the log marks (det_gap 0), the first row that '
                        'shows it, or the row after')
    q.add_argument('--clock-offset', type=float, default=0.0, help='s added to the logged wall time as the pilot clock')
    q.add_argument('--assoc-m', type=float, default=4.0, help='a track belongs to an arch within this, horizontally (m)')
    q.add_argument('--ray-deg', type=float, default=6.0,
                   help='... or, failing that, to the arch whose bearing it shares to within this (an arch at the '
                        'wrong range, not a phantom); 0 = off')
    q.add_argument('--out', default='', help='write the replayed log (fly --log format, rabbit columns replayed)')
    q.add_argument('--json', default='', help='write validation, parameters and the full score to this JSON file')
    add_sight_args(q, yaw_rate_default=2.3)


def params_for(a) -> SightParams:
    """SightParams from the parsed flags (after the preset), validated like the fly command's."""
    from .sightpilot import add_cli_args as add_sight_args
    from .sightpilot import params_from_args
    if getattr(a, 'preset', None):
        p = argparse.ArgumentParser(add_help=False)
        add_sight_args(p)
        p.add_argument('--set', dest='sight_set', action='append', default=[])
        p.add_argument('--flow-gain', type=float, default=1.0)
        pre = p.parse_args(FLIGHT_FLAGS[a.preset])
        for k, v in vars(pre).items():
            if k == 'sight_set':
                a.sight_set = list(v) + list(a.sight_set)
            elif getattr(a, k, None) == p.get_default(k):
                setattr(a, k, v)
    return params_from_args(a, flow_gain=a.flow_gain)


def write_replayed_log(src: str, dst: str, res: dict) -> None:
    rows = list(csv.reader(open(src, newline='', encoding='utf-8')))
    head = rows[0]
    pos = {c: head.index(c) for c in LOG_COLUMNS if c in head and c not in ('det_u', 'det_v', 'det_t')}
    st = head.index('status') if 'status' in head else None
    n = len(res['cols'])
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(head)
        k = 0
        for row in rows[1:]:
            if k >= n or len(row) != len(head):
                continue
            for c, i in pos.items():
                v = float(res['cols'][k, COL[c]])
                row[i] = '' if not math.isfinite(v) else (f'{v:.4f}' if abs(v) >= 1e5 else f'{v:.4g}' if v != int(v)
                                                          else str(int(v)))
            if st is not None:
                row[st] = res['status'][k].replace(',', ';')
            w.writerow(row)
            k += 1


def run_from_args(a) -> dict:
    import yaml
    from ..vision.camera import Camera
    from ..vision.model import IN_H, IN_W
    params = params_for(a)
    c = yaml.safe_load(Path(a.camera).read_text(encoding='utf-8'))
    cam = Camera(int(c['width']), int(c['height']), float(c['f']), float(c['tilt_deg'])).scaled(IN_W, IN_H)
    log = load_log(a.log)
    res = replay(log, params, cam, sync=a.sync, flow_gain=a.flow_gain, clock_offset=a.clock_offset)
    val = validate(log, res) if 'tgt_id' in log else None
    gates = json.load(open(a.gates, encoding='utf-8'))['gates'] if a.gates else None
    sc = score(log, res, gates, a.assoc_m, a.ray_deg) if gates else {'attempts': []}
    print(describe(Path(a.log).stem, val, sc))
    if a.out:
        write_replayed_log(a.log, a.out, res)
        print(f'replayed log written to {a.out}')
    changed = {f.name: getattr(params, f.name) for f in fields(SightParams)
               if getattr(params, f.name) != f.default}
    out = {'log': a.log, 'sync': a.sync, 'preset': a.preset, 'params_changed': changed, 'params': asdict(params),
           'detections': res['n_detections'], 'sight_errors': res['errors'], 'validation': val, 'score': sc,
           'events': res['events'], 'ends': res['ends']}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1, default=_json_default)
        print(f'written to {a.json}')
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
