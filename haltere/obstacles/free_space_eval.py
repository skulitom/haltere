"""FreeSpace corridor planner offline evaluation (OFFLINE ONLY): frames, depth, plan streams, gates S1-S3, A1, B1,
T1, Q1, L1.

The runtime planner is haltere.vision.free_space; this module feeds it recorded frames (the gap-cue evaluation's
aligned flight frames and the obstacle store) in time order and scores it. Hindsight quantities (impact times,
time to impact, the pillar boxes of the m2design impact anatomy, the Minus Two floor height, checkpoint-switch
times) select and score frames only; none reaches the planner. Configs: configs/obstacles/free_space.json (the
planner) and configs/obstacles/free_space_gates.json (gate definitions), both frozen with their sha256 before any
held-out frame is scored; ``score`` refuses unfrozen or edited files and records both hashes.

Confirmation (A1, B1, T1, Q1) uses the section-6 CorridorAim. `SpecCorridorAim` is its offline replica (the pilot
half lives on another branch); the results name the implementation used.

Commands (``python -m haltere.obstacles.free_space_eval <cmd> --out DIR [--cache GAPCUE_DIR]``):
  frames   decode the gate flights that are not cached yet (gap_cue_eval.extract_flight; CPU)
  depth    DA-V2-Small 336 x 602 fp16 block disparity of those frames and their leak frames (GPU, one chunk)
  freeze   freeze free_space.json and free_space_gates.json
  run      plan streams (``streams/<name>.plan.npz``: one array per PLAN_FIELD + pub_time) and leak results
  score    gates -> DIR/results.json
  dev      report-only dev numbers (store runs, dev flights), allowed before the freeze
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..vision import free_space as fs
from ..vision.camera import quat_wxyz_to_mat
from . import gap_cue_eval as gce
from .thermal import ChunkGuard, limit_threads, require_flight_lock_path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATES_PATH = REPO_ROOT / 'configs' / 'obstacles' / 'free_space_gates.json'
MASK_LAYERS = gce.MASK_LAYERS
GRID, FRAME_HW = gce.GRID, gce.FRAME_HW
POST_EVENT = gce.POST_EVENT
VALID_KINDS = fs.VALID_KINDS
ACTING_KINDS = ('aperture', 'shift', 'blocked')


def _log(*a):
    print(*a, flush=True)


def load_gates(path: str | Path = GATES_PATH, *, require_frozen: bool = False) -> tuple[dict, str]:
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    sha = fs.config_sha256(obj)
    if obj.get('frozen') and obj.get('sha256') != sha:
        raise RuntimeError(f'{path} changed after freezing (sha256 mismatch)')
    if require_frozen and not obj.get('frozen'):
        raise RuntimeError(f'{path} is not frozen')
    return obj, sha


def validity(frac, layers=MASK_LAYERS, max_fraction=0.0) -> np.ndarray:
    return gce.validity_from_fractions(frac, layers, max_fraction)


# ----------------------------------------------------------------------------- data

def _dirs(args) -> list[Path]:
    out = [Path(args.out)]
    if getattr(args, 'cache', None):
        out.append(Path(args.cache))
    return out


def flight_data(name: str, dirs) -> dict:
    """Aligned frames, masks, pose, cue and 336 x 602 disparity of a decoded flight (first directory holding it)."""
    for d in dirs:
        fp, dp = Path(d) / 'flights' / f'{name}.npz', Path(d) / 'depth' / f'{name}.npz'
        if fp.exists() and dp.exists():
            rec = gce._npz(fp)
            dep = gce._npz(dp)
            n = len(rec['frame_idx'])
            mm = np.memmap(Path(d) / 'flights' / f'{name}.u8', dtype=np.uint8, mode='r', shape=(n,) + FRAME_HW + (3,))
            nl = len(rec['leak_src'])
            lm = (np.memmap(Path(d) / 'flights' / f'{name}.leaks.u8', dtype=np.uint8, mode='r',
                            shape=(nl,) + FRAME_HW + (3,)) if nl else None)
            keep = rec['keep'] & rec['tel_valid'] if 'tel_valid' in rec else rec['keep']
            sel = np.flatnonzero(rec['keep'])
            sel = sel[np.argsort(rec['t_wall'][sel], kind='stable')]
            imp = float(rec['impact_wall'])
            return dict(name=name, kind='flight', dir=str(d), rec=rec, dep=dep, mm=mm, leaks=lm, sel=sel,
                        t=rec['t_wall'][sel], pos=rec['pos'][sel], vel=rec['vel'][sel], quat=rec['quat'][sel],
                        cue=rec['cue_uv'][sel], tel=rec['tel_valid'][sel], frac=rec['maskfrac'][sel],
                        disp=dep['teacher'], impact=imp, keep_pose=keep)
    raise FileNotFoundError(f'{name}: no decoded frames + depth in {[str(d) for d in dirs]}')


def store_data(rid: int, cache: Path) -> dict:
    from .store import FrameStore
    root = gce.data_root() / 'runs' / 'obstacle-store-v1'
    st = FrameStore(root)
    ix = st.index
    masks = gce._npz(Path(cache) / 'store_masks.npz')
    rows = masks[f'r{rid}_rows']
    fr = masks[f'r{rid}']
    flags = np.asarray(ix['flags'][rows])
    sel = np.flatnonzero((flags & POST_EVENT) == 0)
    t = np.asarray(ix['t_wall'][rows[sel]], np.float64)
    o = np.argsort(t, kind='stable')
    sel = sel[o]
    rs = rows[sel]
    teacher = np.load(root / 'labels' / 'teacher' / 'disparity.npy', mmap_mode='r')
    return dict(name=f'store:{rid}', kind='store', store=st, rows=rs, t=np.asarray(ix['t_wall'][rs], np.float64),
                pos=np.asarray(ix['pos'][rs], np.float64), vel=np.asarray(ix['vel'][rs], np.float64),
                quat=np.asarray(ix['quat'][rs], np.float64), cue=np.asarray(ix['cue_uv'][rs], np.float64),
                tel=np.ones(len(rs), bool), frac=fr[sel], disp_rows=teacher, tti=np.asarray(ix['tti_s'][rs], float),
                impact=np.nan)


def _frame(seq, i):
    if seq['kind'] == 'flight':
        return np.asarray(seq['mm'][seq['sel'][i]])
    return np.asarray(seq['store'].frame(int(seq['rows'][i])))


def _disp(seq, i):
    if seq['kind'] == 'flight':
        return np.asarray(seq['disp'][seq['sel'][i]], np.float64)
    return np.asarray(seq['disp_rows'][int(seq['rows'][i])], np.float64)


def _cue(uv):
    uv = np.asarray(uv, np.float64)
    return (float(uv[0]), float(uv[1])) if np.isfinite(uv).all() else None


# ----------------------------------------------------------------------------- plan streams

def run_sequence(seq, config, motor, *, offline_age=0.1, leak_tests=False, hud_scale=None, log=None) -> dict:
    """Feed one sequence (time order) through a FreeSpacePlanner; returns the plan stream (one array per
    PLAN_FIELD, pub_time = time + offline_age) plus leak / HUD-only results when asked."""
    planner = fs.FreeSpacePlanner(config, motor, enabled=True)
    n = len(seq['t'])
    stream = {k: np.full(n, np.nan) for k in fs.PLAN_FIELDS}
    leak_map = {}
    if leak_tests and seq['kind'] == 'flight' and len(seq['rec']['leak_src']):
        pos = {int(j): i for i, j in enumerate(seq['sel'])}
        for li, j in enumerate(seq['rec']['leak_src']):
            if int(j) in pos:
                leak_map.setdefault(pos[int(j)], []).append(li)
    leaks, hud = [], []
    t0 = time.time()
    for i in range(n):
        rgb = _frame(seq, i)
        d = _disp(seq, i)
        valid = validity(seq['frac'][i])
        pose = (seq['quat'][i], seq['vel'][i]) if seq['tel'][i] else (None, None)
        cue = _cue(seq['cue'][i])
        ti = float(seq['t'][i])
        if i in leak_map:
            snap = planner.snapshot()
            for li in leak_map[i]:
                planner.restore(snap)
                lrgb = np.asarray(seq['leaks'][li])
                ld = np.asarray(seq['dep']['leak_teacher'][li], np.float64)
                lv = validity(seq['rec']['leak_maskfrac'][li])
                lo = planner.process(lrgb, ld, lv, *pose, cue, ti, seq=i)
                leaks.append(dict(i=i, leak=li, kind=str(seq['rec']['leak_kind'][li]), p_kind=fs.kind_name(lo['kind']),
                                  p_az=lo['az'], p_el=lo['el']))
            planner.restore(snap)
        out = planner.process(rgb, d, valid, *pose, cue, ti, seq=float(i))
        for k in fs.PLAN_FIELDS:
            stream[k][i] = out[k]
        if i in leak_map:
            for r in leaks[-len(leak_map[i]):]:
                r.update(o_kind=fs.kind_name(out['kind']), o_az=out['az'], o_el=out['el'])
        if hud_scale is not None:
            f = seq['frac'][i]
            dh = 1.0 + 9.0 * f[MASK_LAYERS.index('hud')].astype(np.float64) / 100.0
            ho = planner.decide_with_scale(dh, valid, *pose, cue, ti, hud_scale[0], hud_scale[1])
            hud.append(fs.kind_name(ho['kind']))
        if log and i and i % 1000 == 0:
            log(f'  {seq["name"]}: {i}/{n} frames, {time.time() - t0:.0f} s')
    stream['age'][:] = offline_age
    stream['pub_time'] = stream['time'] + offline_age
    return dict(stream=stream, leaks=leaks, hud=hud, counts=dict(planner.counts), seconds=time.time() - t0)


def save_stream(path: Path, stream: dict, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in stream.items()}, **(extra or {}))


# ----------------------------------------------------------------------------- CorridorAim (section 6 replica)

def wrap_deg(a):
    return (np.asarray(a, np.float64) + 180.0) % 360.0 - 180.0


class SpecCorridorAim:
    """Offline replica of the section-6 CorridorAim (confirmation, latch, flip, slew, release, decay, ring
    conflicts). Classes: +1 left, -1 right, +2 vertical; ``confirmed`` 9 = blocked."""
    BLOCKED = 9

    def __init__(self, pilot: dict):
        self.c = dict(pilot)
        self.reset()
        self.counts = dict(samples=0, stale=0, invalid=0, repeated=0, episodes_l=0, episodes_r=0, episodes_v=0,
                           blocked_episodes=0, flips=0, conflicts=0)
        self.engaged_s = 0.

    def reset(self):
        self.hist = deque(maxlen=int(self.c['window']))
        self.last_time = None
        self.last_fresh = -np.inf
        self.state, self.cls, self.since = 'idle', 0, -np.inf
        self.target = np.zeros(2)
        self.applied = np.zeros(2)
        self.decay_from = None
        self.no_option = 0
        self.clear_run = 0
        self.hold_until = -np.inf
        self.confirm_sample = None

    def _option(self, s, cls):
        if cls == 1:
            a, e = s.get('l_az'), s.get('l_el')
        elif cls == -1:
            a, e = s.get('r_az'), s.get('r_el')
        else:
            a, e = 0., s.get('v_el')
        if a is None or e is None or not (np.isfinite(a) and np.isfinite(e)):
            return None
        return np.array([float(a), float(e)])

    def ingest(self, s: dict, now: float) -> bool:
        stamp = s.get('time')
        if stamp is None or not np.isfinite(stamp) or (self.last_time is not None and stamp <= self.last_time):
            self.counts['repeated'] += 1
            return False
        self.last_time = stamp
        if not 0 <= now - stamp <= self.c['max_age_s']:
            self.counts['stale'] += 1
            return False
        if not s.get('valid'):
            self.counts['invalid'] += 1
            return False
        self.hist.append(s)
        self.counts['samples'] += 1
        self.last_fresh = now
        self.clear_run = self.clear_run + 1 if s['kind'] in ('clear', 'aperture') else 0
        if self.state == 'shift':
            self.no_option = 0 if self._option(s, self.cls) is not None else self.no_option + 1
        return True

    def _conflict(self, now):
        self.hist.clear()
        self.state, self.cls, self.since = 'idle', 0, -np.inf
        self.target[:] = 0.
        self.applied[:] = 0.
        self.decay_from = None
        self.hold_until = now + self.c['conflict_hold_s']
        self.counts['conflicts'] += 1

    def _release(self):
        self.state, self.cls, self.since = 'idle', 0, -np.inf
        self.target[:] = 0.
        self.decay_from = np.abs(self.applied).copy()
        self.no_option = 0

    def step(self, now: float, dt: float, ring_az_cue=None) -> dict:
        c = self.c
        engaged = self.state != 'idle' or np.any(np.abs(self.applied) > 1e-9)
        newest = self.hist[-1] if self.hist else None
        if ring_az_cue is not None and newest is not None and np.isfinite(newest.get('ring_az') or np.nan):
            if abs(float(wrap_deg(newest['ring_az'] - ring_az_cue))) > c['conflict_deg']:
                if engaged:
                    self._conflict(now)
                else:
                    self.hist.clear()
                newest = None
        confirmed_new = None
        if now < self.hold_until:
            pass
        else:
            if self.state != 'idle' and now - self.last_fresh > c['decay_s']:
                self._release()
            recent = [s for s in self.hist if newest is not None and newest['time'] - s['time'] <= c['confirm_window_s']]
            votes = {}
            blocked = 0
            for s in recent:
                if s['kind'] == 'shift' and s.get('cls') in (1., -1., 2.):
                    votes[int(s['cls'])] = votes.get(int(s['cls']), 0) + 1
                blocked += s['kind'] == 'blocked'
            cand = [k for k, v in votes.items() if v >= c['confirm']]
            cand = cand[0] if cand else None
            cand_blocked = blocked >= c['confirm']
            if self.state == 'idle':
                if cand is not None:
                    self.state, self.cls, self.since, self.no_option = 'shift', cand, now, 0
                    self.counts[{1: 'episodes_l', -1: 'episodes_r', 2: 'episodes_v'}[cand]] += 1
                    confirmed_new = cand
                elif cand_blocked:
                    self.state, self.cls, self.since = 'blocked', 0, now
                    self.counts['blocked_episodes'] += 1
                    confirmed_new = self.BLOCKED
            elif self.state == 'shift':
                if now - self.since >= c['side_latch_s'] and self.clear_run >= c['release_after']:
                    self._release()
                elif self.no_option >= c['flip_after'] and cand is not None and cand != self.cls:
                    self.cls, self.since, self.no_option = cand, now, 0
                    self.counts['flips'] += 1
                    confirmed_new = cand
            elif self.state == 'blocked':
                if cand is not None:
                    self.state, self.cls, self.since, self.no_option = 'shift', cand, now, 0
                    self.counts[{1: 'episodes_l', -1: 'episodes_r', 2: 'episodes_v'}[cand]] += 1
                    confirmed_new = cand
                elif now - self.since >= c['side_latch_s'] and self.clear_run >= c['release_after']:
                    self._release()
            if confirmed_new is not None:
                self.confirm_sample = newest
            if self.state == 'shift' and newest is not None:
                opt = self._option(newest, self.cls)
                if opt is not None:
                    self.target = opt
            elif self.state == 'blocked':
                self.target = np.zeros(2)
            self.blocked_confirmed = cand_blocked
        if self.state != 'idle':
            self.decay_from = None
            room = np.array([c['slew_az_deg_s'], c['slew_el_deg_s']]) * dt
            self.applied += np.clip(self.target - self.applied, -room, room)
        elif self.decay_from is not None:
            room = np.minimum(np.array([c['slew_az_deg_s'], c['slew_el_deg_s']]), self.decay_from / c['decay_s']) * dt
            self.applied -= np.clip(self.applied, -room, room)
        self.applied = np.array([np.clip(self.applied[0], -20., 20.), np.clip(self.applied[1], 0., 12.)])
        v_cap = None
        if newest is not None and newest.get('v_cap') is not None and np.isfinite(newest['v_cap']):
            urgent = self.state == 'blocked' or (self.state == 'shift' and newest['kind'] == 'shift'
                                                 and newest.get('feasible') == 0.)
            if urgent:
                v_cap = float(newest['v_cap'])
        if self.state != 'idle' or np.any(np.abs(self.applied) > 1e-9):
            self.engaged_s += dt
        confirmed = self.BLOCKED if self.state == 'blocked' else (self.cls if self.state == 'shift' else 0)
        intended = self.target.copy() if self.state == 'shift' else np.zeros(2)
        return dict(confirmed=confirmed, intended=intended, applied=self.applied.copy(), v_cap=v_cap,
                    new=confirmed_new, confirm_time=None if self.confirm_sample is None else self.confirm_sample['time'],
                    newest=newest)


def stream_samples(stream: dict) -> list:
    """Per-frame sample dicts (as camera_process.plan_sample) with pub_time."""
    out = []
    n = len(stream['time'])
    for i in range(n):
        s = {}
        for k in fs.PLAN_FIELDS:
            v = float(stream[k][i])
            s[k] = v if np.isfinite(v) else None
        s['time'] = float(stream['time'][i])
        s['kind'] = fs.kind_name(stream['kind'][i])
        s['valid'] = bool(stream['valid'][i] == 1.)
        s['pub_time'] = float(stream['pub_time'][i])
        out.append(s)
    return out


def replay_aim(stream: dict, pilot: dict, ar: dict) -> dict:
    """Step the section-6 CorridorAim over a plan stream on a fixed tick grid (see the gates' aim_replay)."""
    samples = stream_samples(stream)
    t_frames = np.asarray(stream['time'], np.float64)
    ring = np.asarray(stream['ring_az'], np.float64)
    aim = SpecCorridorAim(pilot)
    dt = 1.0 / float(ar['tick_hz'])
    if not len(samples):
        return dict(t=np.zeros(0))
    ticks = np.arange(samples[0]['pub_time'], samples[-1]['pub_time'] + 0.5, dt)
    rec = {k: [] for k in ('t', 'confirmed', 'int_az', 'int_el', 'app_az', 'app_el', 'v_cap', 'new', 'conf_time',
                           'ring_az', 'newest_time', 'feasible', 'kind_newest')}
    k = 0
    lat = float(ar['cue_latency_s'])
    for now in ticks:
        while k < len(samples) and samples[k]['pub_time'] <= now + 1e-9:
            aim.ingest(samples[k], now)
            k += 1
        j = int(np.searchsorted(t_frames + lat, now, side='right')) - 1
        rc = ring[j] if j >= 0 and np.isfinite(ring[j]) else None
        o = aim.step(now, dt, rc)
        nw = o['newest']
        rec['t'].append(now)
        rec['confirmed'].append(o['confirmed'])
        rec['int_az'].append(o['intended'][0])
        rec['int_el'].append(o['intended'][1])
        rec['app_az'].append(o['applied'][0])
        rec['app_el'].append(o['applied'][1])
        rec['v_cap'].append(np.nan if o['v_cap'] is None else o['v_cap'])
        rec['new'].append(0 if o['new'] is None else o['new'])
        rec['conf_time'].append(np.nan if o['confirm_time'] is None else o['confirm_time'])
        rec['ring_az'].append(np.nan if rc is None else rc)
        rec['newest_time'].append(np.nan if nw is None else nw['time'])
        rec['feasible'].append(np.nan if nw is None or nw.get('feasible') is None else nw['feasible'])
        rec['kind_newest'].append('' if nw is None else nw['kind'])
    out = {k: np.asarray(v) for k, v in rec.items()}
    out['counts'] = dict(aim.counts, engaged_s=round(aim.engaged_s, 2))
    return out


def episodes(active: np.ndarray, t: np.ndarray, merge_s: float = 0.3) -> list:
    """(start, end) tick index ranges of active runs, runs less than merge_s apart merged."""
    a = np.asarray(active, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, a.astype(np.int8), 0]))
    runs = list(zip(edges[::2], edges[1::2]))
    out = []
    for s, e in runs:
        if out and t[s] - t[out[-1][1] - 1] < merge_s:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _interp_pos(seq_t, seq_pos, t):
    return np.stack([np.interp(t, seq_t, seq_pos[:, k]) for k in range(3)], -1)


# ----------------------------------------------------------------------------- gate scoring helpers

def pillar_extent(pos_xy, box, inflate=0.0):
    xs = [box['x'][0] - inflate, box['x'][1] + inflate]
    ys = [box['y'][0] - inflate, box['y'][1] + inflate]
    az = np.stack([np.degrees(np.arctan2(y - pos_xy[:, 1], x - pos_xy[:, 0])) for x in xs for y in ys], 1)
    ref = az[:, :1]
    rel = wrap_deg(az - ref)
    return ref[:, 0] + rel.min(1), ref[:, 0] + rel.max(1)


def _inside(a, lo, hi):
    return (wrap_deg(a - lo) >= 0) & (wrap_deg(hi - a) >= 0)


def pillar_blocks(geo: fs.Geometry, R, pos, box, sub=7):
    """(36, 64) coverage and mean true optical depth of rays hitting an axis-aligned box from a camera at pos."""
    H, W = geo.grid
    s = (np.arange(sub) + .5) / sub * geo.bpx
    ys, xs = np.meshgrid(s, s, indexing='ij')
    cc, rr = np.meshgrid(np.arange(W), np.arange(H))
    px = (cc[..., None, None] * geo.bpx + xs[None, None]).reshape(-1)
    py = (rr[..., None, None] * geo.bpx + ys[None, None]).reshape(-1)
    rays_c = np.stack([(px - geo.cx) / geo.f, (py - geo.cy) / geo.f, np.ones(len(px))], 1)
    rw = rays_c @ geo.M @ R.T
    lo = np.array([box['x'][0], box['y'][0], box['z'][0]]) - pos
    hi = np.array([box['x'][1], box['y'][1], box['z'][1]]) - pos
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = lo[None] / rw, hi[None] / rw
        tmin = np.nanmax(np.minimum(t1, t2), 1)
        tmax = np.nanmin(np.maximum(t1, t2), 1)
    hit = (tmax >= tmin) & (tmin > 0)
    hit = hit.reshape(H, W, sub * sub)
    depth = np.where(hit, tmin.reshape(H, W, sub * sub), np.nan)
    cov = hit.mean(-1)
    with np.errstate(invalid='ignore'):
        dmean = np.nanmean(np.where(hit, depth, np.nan), -1)
    return cov, dmean


def sim_clearance(p0, v_h, course_deg, aim_deg, model, box, t_max=3.0, dt=0.01):
    """Minimum 2-D distance to a box of a kinematic path: straight along the course for the model's delay, then
    turning toward the aim at a_lat / v (report-only pillar_sim)."""
    x, y = float(p0[0]), float(p0[1])
    h = math.radians(course_deg)
    aim = math.radians(aim_deg)
    om = model.a_lat_mps2 / max(v_h, .1)
    best = np.inf
    t = 0.
    while t < t_max and x < box['x'][1] + 0.5:
        if t >= model.delay_s:
            err = (aim - h + math.pi) % (2 * math.pi) - math.pi
            h += max(-om * dt, min(om * dt, err))
        x += v_h * dt * math.cos(h)
        y += v_h * dt * math.sin(h)
        dx = max(box['x'][0] - x, 0., x - box['x'][1])
        dy = max(box['y'][0] - y, 0., y - box['y'][1])
        best = min(best, math.hypot(dx, dy))
        t += dt
    return round(float(best), 3)


def _dist(pos, point):
    return np.hypot(point[0] - pos[..., 0], point[1] - pos[..., 1])


def d_switch_of(seq_t, stream, pos, pa):
    reg = pa['approach_region']
    sp = np.hypot(stream_speed(stream), 0)
    sw = gce.checkpoint_switches(seq_t, np.asarray(stream['ring_az'], float), sp)
    d = _dist(pos, pa['distance_to'])
    inreg = lambda i: reg['x'][0] <= pos[i, 0] <= reg['x'][1] and reg['y'][0] <= pos[i, 1] <= reg['y'][1]
    i_close = int(np.argmin(np.where((pos[:, 0] <= reg['x'][1]), d, np.inf)))
    cands = [i for i in sw if i <= i_close and inreg(i)]
    if cands:
        return float(d[cands[-1]]), 'switch', float(seq_t[cands[-1]])
    return 5.9, 'anatomy default (no switch detected)', None


def stream_speed(stream):
    return np.asarray(stream['speed'], float)


def a1_approach(name, seq, stream, aim, gates, models):
    g, pa = gates['A1'], gates['pillar_a']
    reg = pa['approach_region']
    t = aim['t']
    pos_tick = _interp_pos(seq['t'], seq['pos'], t)
    conf_pos = _interp_pos(seq['t'], seq['pos'], np.where(np.isfinite(aim['conf_time']), aim['conf_time'], t))
    d_conf = _dist(conf_pos, pa['distance_to'])
    inreg = ((pos_tick[:, 0] >= reg['x'][0]) & (pos_tick[:, 0] <= reg['x'][1]) & (pos_tick[:, 1] >= reg['y'][0])
             & (pos_tick[:, 1] <= reg['y'][1]))
    d_sw, how, t_sw = d_switch_of(seq['t'], stream, seq['pos'], pa)
    need = min(g['min_distance_cap_m'], d_sw - g['switch_margin_m'])
    samples = stream_samples(stream)
    by_time = {s['time']: s for s in samples}
    first = None
    for i in np.flatnonzero(inreg):
        c = aim['confirmed'][i]
        ok = c == 1
        if c == SpecCorridorAim.BLOCKED:
            s = by_time.get(float(aim['newest_time'][i]))
            ok = bool(s is not None and s.get('l_ok') == 0. and s.get('l_az') is not None)
        if ok:
            first = i
            break
    right = [round(float(d_conf[i]), 2) for i in np.flatnonzero(inreg & (aim['confirmed'] == -1))
             if d_conf[i] > g['right_max_distance_m']]
    lo, hi = pillar_extent(pos_tick[:, :2], pa['box'], g['aim_inflate_m'])
    ring = aim['ring_az']
    aim_az = ring + aim['app_az']
    bad = inreg & np.isfinite(ring) & _inside(aim_az, lo, hi) & ~_inside(ring, lo, hi) & (np.abs(aim['app_az']) > 1e-9)
    res = dict(d_switch_m=round(d_sw, 2), d_switch_source=how, required_m=round(need, 2),
               first_left_d_m=None if first is None else round(float(d_conf[first]), 2),
               first_left_kind='L' if first is not None and aim['confirmed'][first] == 1 else (
                   'blocked+urgent L' if first is not None else None),
               passes=bool(first is not None and d_conf[first] >= need),
               confirmed_right_beyond_2m_d_m=sorted(set(right))[:20], confirmed_right_ticks=len(right),
               aim_into_pillar_ticks=int(bad.sum()),
               aim_into_pillar_d_m=sorted({round(float(_dist(pos_tick[i], pa['distance_to'])), 2)
                                           for i in np.flatnonzero(bad)})[:20],
               ticks=int(inreg.sum()),
               closest_d_m=round(float(_dist(seq['pos'], pa['distance_to']).min()), 2))
    # raw per-sample classes in the region (diagnostic)
    sreg = ((seq['pos'][:, 0] >= reg['x'][0]) & (seq['pos'][:, 0] <= reg['x'][1]) & (seq['pos'][:, 1] >= reg['y'][0])
            & (seq['pos'][:, 1] <= reg['y'][1]))
    kinds = np.array([fs.kind_name(k) for k in stream['kind']])
    res['sample_kinds_in_region'] = {k: int((sreg & (kinds == k)).sum()) for k in fs.PLAN_KINDS if (sreg & (kinds == k)).any()}
    raw_l = np.flatnonzero(sreg & (kinds == 'shift') & (np.asarray(stream['cls']) == 1))
    res['first_raw_left_d_m'] = round(float(_dist(seq['pos'][raw_l[0]], pa['distance_to'])), 2) if len(raw_l) else None
    if first is not None:
        i = first
        tt = float(aim['conf_time'][i]) if np.isfinite(aim['conf_time'][i]) else float(t[i])
        k = int(np.clip(np.searchsorted(seq['t'], tt), 0, len(seq['t']) - 1))
        v = seq['vel'][k]
        course = math.degrees(math.atan2(v[1], v[0]))
        az = float(aim['ring_az'][i]) + float(aim['int_az'][i]) if np.isfinite(aim['ring_az'][i]) else course
        vh = float(math.hypot(v[0], v[1]))
        p_now = _interp_pos(seq['t'], seq['pos'], np.array([t[i]]))[0]
        res['pillar_sim_clearance_m'] = {m: sim_clearance(p_now, vh, course, az, models[m], pa['box'])
                                         for m in ('brain08', 'fast_pd')}
        res['pillar_sim_note'] = f'from the confirmation tick (x {p_now[0]:.2f}, y {p_now[1]:.2f}), {vh:.2f} m/s, aim {az:.1f} deg'
    return res


def s1_s2(name, seq, stream, gates, config):
    g1, g2, pa = gates['S1'], gates['S2'], gates['pillar_a']
    reg = g1['region']
    p = seq['pos']
    sp = np.hypot(seq['vel'][:, 0], seq['vel'][:, 1])
    h_tel = p[:, 2] - gates['minus_floor_z_m']
    hf = np.asarray(stream['h_floor'], float)
    m = (seq['tel'] & (sp > g1['min_speed_mps']) & np.isfinite(hf) & (p[:, 0] >= reg['x'][0]) & (p[:, 0] <= reg['x'][1])
         & (p[:, 1] >= reg['y'][0]) & (p[:, 1] <= reg['y'][1]) & (h_tel > 0.05))
    ratio = hf[m] / h_tel[m]
    kinds = np.array([fs.kind_name(k) for k in stream['kind']])
    s1 = dict(samples=int(m.sum()), ratios=ratio.tolist(),
              valid_kind_samples=int((m & np.isin(kinds, VALID_KINDS)).sum()),
              frames_in_region_fast=int((seq['tel'] & (sp > g1['min_speed_mps']) & (p[:, 0] >= reg['x'][0])
                                         & (p[:, 0] <= reg['x'][1]) & (p[:, 1] >= reg['y'][0])
                                         & (p[:, 1] <= reg['y'][1])).sum()))
    geo = fs.Geometry(config.camera)
    frame_meds, pooled = [], []
    for i in range(len(seq['t'])):
        a, b = stream['scale_a'][i], stream['scale_b'][i]
        if not (np.isfinite(a) and np.isfinite(b)) or not seq['tel'][i]:
            continue
        if _dist(p[i], pa['distance_to']) > 12:
            continue
        R = quat_wxyz_to_mat(seq['quat'][i] / np.linalg.norm(seq['quat'][i]))
        cov, dtrue = pillar_blocks(geo, R, p[i], pa['box'])
        valid = validity(seq['frac'][i])
        sel = valid & (cov >= g2['coverage_min']) & (dtrue >= g2['true_depth_m'][0]) & (dtrue <= g2['true_depth_m'][1])
        if not sel.any():
            continue
        d = _disp(seq, i)
        Z = 1.0 / np.maximum(a * d + b, 1.0 / config.scale.max_range_m)
        r = (Z / dtrue)[sel]
        frame_meds.append(float(np.median(r)))
        pooled.extend(r.tolist())
    s2 = dict(frames=len(frame_meds), blocks=len(pooled), frame_medians=frame_meds, pooled=pooled)
    return s1, s2


def _summ_ratio(r, rng, band, minfrac):
    r = np.asarray(r, float)
    if not len(r):
        return dict(n=0, median=None, band_fraction=None, passes=False)
    med = float(np.median(r))
    frac = float(((r >= band[0]) & (r <= band[1])).mean())
    return dict(n=int(len(r)), median=round(med, 3), p10=round(float(np.percentile(r, 10)), 3),
                p90=round(float(np.percentile(r, 90)), 3), band_fraction=round(frac, 3),
                passes=bool(rng[0] <= med <= rng[1] and frac >= minfrac))


def q1_flight(name, seq, stream, aim, gates, csv_cmd=None):
    g = gates['Q1']
    t = seq['t']
    dts = np.diff(t)
    minutes = float(np.sum(dts[dts < 0.3])) / 60
    tt = aim['t']
    conf = aim['confirmed']
    intended = np.hypot(aim['int_az'], aim['int_el']) > 1e-9
    active = ((conf != 0) & (conf != SpecCorridorAim.BLOCKED) & intended) | (conf == SpecCorridorAim.BLOCKED)
    eps = episodes(active, tt)
    urgent_tick = (conf == SpecCorridorAim.BLOCKED) | ((conf != 0) & (aim['feasible'] == 0.) &
                                                       (aim['kind_newest'] == 'shift'))
    n_v = n_urg = 0
    for s, e in eps:
        first_cls = next((c for c in conf[s:e] if c != 0), 0)
        n_v += first_cls == 2
        n_urg += bool(urgent_tick[s:e].any())
    app_abs = np.abs(aim['app_az'][active])
    # terrain climbs: VerticalGuard rule 3 trigger replica
    samples = stream_samples(stream)
    vz = seq['vel'][:, 2]
    climb_t = []
    hist = deque(maxlen=3)
    for i, s in enumerate(samples):
        if not s['valid']:
            continue
        hist.append(s['rise'] is not None)
        if sum(hist) >= 2 and vz[i] >= -0.3:
            climb_t.append(s['pub_time'])
    climbs = 0
    last_end = -np.inf
    for tc in climb_t:
        if tc > last_end + 0.3:
            climbs += 1
        last_end = tc + 0.5
    # checkpoint switches
    sp = np.hypot(seq['vel'][:, 0], seq['vel'][:, 1])
    sw = gce.checkpoint_switches(t, np.asarray(stream['ring_az'], float), sp)
    offs = np.maximum(np.abs(aim['app_az']), np.abs(aim['app_el']))
    quiet = []
    for i in sw:
        m = (tt >= t[i] - g['switch_window_s']) & (tt < t[i])
        quiet.append(bool(not m.any() or offs[m].max() <= g['switch_max_offset_deg']))
    # floor-bound activations (report): rule 1 lo above the logged cmd_vz
    floor_s = None
    if csv_cmd is not None:
        ok = np.array([s['valid'] and s['h_floor'] is not None for s in samples])
        idx = np.flatnonzero(ok)
        act = 0.
        if len(idx):
            k = np.clip(np.searchsorted(np.asarray([samples[i]['pub_time'] for i in idx]), tt, side='right') - 1, 0, None)
            for j, now in enumerate(tt):
                if k[j] >= len(idx):
                    continue
                s = samples[idx[k[j]]]
                if s['pub_time'] > now or now - s['time'] > 0.2:
                    continue
                vzm = float(np.interp(now, t, vz))
                hn = s['h_floor'] + vzm * (now - s['time'])
                lo = -(hn - 0.5) / 0.5
                if lo > 0:
                    lo = min(lo, 1.0)
                cmd = float(np.interp(now, csv_cmd[0], csv_cmd[1]))
                act += (lo > cmd) / float(gates['aim_replay']['tick_hz'])
        floor_s = act
    kinds = np.array([fs.kind_name(k) for k in stream['kind']])
    return dict(minutes=round(minutes, 3), episodes=len(eps), vertical_episodes=int(n_v), urgent_episodes=int(n_urg),
                terrain_climbs=int(climbs), app_abs_az=app_abs.tolist(), switches=len(sw), switches_quiet=int(sum(quiet)),
                floor_bound_active_s=None if floor_s is None else round(floor_s, 2),
                kinds={k: int((kinds == k).sum()) for k in fs.PLAN_KINDS if (kinds == k).any()},
                aim_counts=aim['counts'])


def _csv_cmd(name):
    import pandas as pd
    p = gce.data_root() / gce.FLIGHT_DIR / f'{name}.csv'
    if not p.exists():
        return None
    d = pd.read_csv(p, usecols=['wall', 'cmd_vz'])
    d = d[np.isfinite(d.cmd_vz)]
    return d.wall.values.astype(float), d.cmd_vz.values.astype(float)


# ----------------------------------------------------------------------------- commands

def cmd_frames(args):
    limit_threads(2, cv2=True)
    gates, _ = load_gates()
    guard = ChunkGuard(require_flight_lock_path(args.flight_lock), gpu=False)
    out = Path(args.out) / 'flights'
    out.mkdir(parents=True, exist_ok=True)
    names = args.flights.split(',') if args.flights else list(gates['new_flights'])
    for name in names:
        if (out / f'{name}.npz').exists():
            continue
        gce.extract_flight(name, gce.data_root(), out, guard)


def cmd_depth(args):
    """336 x 602 fp16 block disparity (the runtime input) of the new flights' frames and leak frames; one GPU chunk
    of at most args.max_s, resumable per flight."""
    limit_threads(2, torch=True, cv2=True)
    from ..vision.relative_depth import RelativeDepth, TEACHER_INPUT_HW
    gates, _ = load_gates()
    guard = ChunkGuard(require_flight_lock_path(args.flight_lock), gpu=True, chunk_max_s=float(args.max_s))
    fdir, ddir = Path(args.out) / 'flights', Path(args.out) / 'depth'
    ddir.mkdir(parents=True, exist_ok=True)
    guard.before_chunk()
    t0 = time.time()
    te = RelativeDepth(input_hw=TEACHER_INPUT_HW)
    prov = dict(teacher=te.provenance, teacher_latency_ms=round(te.latency_ms(), 2))
    (ddir / 'provenance.json').write_text(json.dumps(prov, indent=1) + '\n', encoding='utf-8')
    bs = int(args.batch)

    def run(mm):
        a = np.empty((len(mm),) + GRID, np.float16)
        for s in range(0, len(mm), bs):
            if guard.chunk_expired():
                raise TimeoutError('GPU chunk expired')
            a[s:s + bs] = te(np.asarray(mm[s:s + bs]))
        return a
    names = args.flights.split(',') if args.flights else list(gates['new_flights'])
    done = []
    try:
        for name in names:
            path = ddir / f'{name}.npz'
            if path.exists():
                continue
            rec = np.load(fdir / f'{name}.npz')
            n = len(rec['frame_idx'])
            mm = np.memmap(fdir / f'{name}.u8', dtype=np.uint8, mode='r', shape=(n,) + FRAME_HW + (3,))
            d = run(mm)
            nl = len(rec['leak_src'])
            ld = (run(np.memmap(fdir / f'{name}.leaks.u8', dtype=np.uint8, mode='r', shape=(nl,) + FRAME_HW + (3,)))
                  if nl else np.zeros((0,) + GRID, np.float16))
            np.savez_compressed(path, teacher=d, leak_teacher=ld)
            done.append(name)
            _log(f'{name}: {n} frames + {nl} leak frames, {time.time() - t0:.0f} s')
    finally:
        _log(f'depth chunk: {time.time() - t0:.0f} s, flights done {done}, guard {guard.summary()}')


def cmd_freeze(args):
    for p in (fs.CONFIG_PATH, GATES_PATH):
        o = gce.freeze_file(p)
        _log(f'{p.name}: frozen {o["frozen_at"]} sha256 {o["sha256"]}')


def _motor(gates, key):
    return gates['motors'][key]


def _sequences(gates, dirs, cache, which):
    for name in which:
        if name.startswith('store:'):
            rid = name.split(':')[1]
            yield name, store_data(int(rid), cache), _motor(gates, gates['store_runs'][rid]['motor'])
        else:
            yield name, flight_data(name, dirs), _motor(gates, gates['flights'][name]['motor'])


def cmd_run(args):
    """Plan streams (and leak / HUD-only results for the L1 flights) of the named sequences with the frozen v1."""
    limit_threads(2, cv2=True)
    config, _, cfg_sha = fs.load_config(require_frozen=not args.draft)
    gates, gates_sha = load_gates(require_frozen=not args.draft)
    out = Path(args.out)
    names = args.names.split(',')
    l1 = set(gates['L1']['flights'])
    for name, seq, motor in _sequences(gates, _dirs(args), Path(args.cache), names):
        path = out / 'streams' / f'{name.replace(":", "_")}.plan.npz'
        if path.exists() and not args.force:
            continue
        leak = name in l1 and not args.no_leaks
        r = run_sequence(seq, config, motor, offline_age=float(gates['offline_age_s']), leak_tests=leak,
                         hud_scale=tuple(gates['L1']['hud_scale']) if leak else None, log=_log)
        extra = dict(pos=seq['pos'], vel=seq['vel'], quat=seq['quat'], cue_uv=seq['cue'], tel_valid=seq['tel'],
                     config_sha256=np.asarray(cfg_sha), gates_sha256=np.asarray(gates_sha), motor=np.asarray(motor))
        if 'tti' in seq:
            extra['tti'] = seq['tti']
        if np.isfinite(seq['impact']):
            extra['impact_wall'] = np.asarray(seq['impact'])
        save_stream(path, r['stream'], extra)
        if leak:
            (out / 'streams' / f'{name}.leaks.json').write_text(json.dumps(dict(leaks=r['leaks'], hud=r['hud']),
                                                                          default=float) + '\n', encoding='utf-8')
        kinds = {k: v for k, v in r['counts'].items() if v}
        _log(f'{name}: {len(seq["t"])} frames in {r["seconds"]:.0f} s, kinds {kinds}')


def _load_stream(out, name):
    z = np.load(Path(out) / 'streams' / f'{name.replace(":", "_")}.plan.npz')
    return {k: z[k] for k in z.files}


def cmd_score(args):
    limit_threads(2, cv2=True)
    config, raw, cfg_sha = fs.load_config(require_frozen=True)
    gates, gates_sha = load_gates(require_frozen=True)
    models = fs.load_response_models(config.response_models)
    out = Path(args.out)
    dirs = _dirs(args)
    cache = Path(args.cache)
    pilot = config.pilot
    ar = gates['aim_replay']
    try:
        import importlib
        importlib.import_module('haltere.liftoff.corridor_aim')
        real = True
    except ImportError:
        real = False
    res = dict(schema='haltere.obstacles.free_space_results.v1', scored_at=_dt.datetime.now().isoformat(timespec='seconds'),
               free_space_sha256=cfg_sha, free_space_version=raw.get('version'), gates_sha256=gates_sha,
               gates_version=gates.get('version'),
               aim_implementation='SpecCorridorAim (section-6 replica)' + (
                   '; haltere.liftoff.corridor_aim exists on this branch but its API was not wired' if real else
                   '; haltere.liftoff.corridor_aim is not on this branch'))
    seqs, streams, aims = {}, {}, {}

    def get(name):
        if name not in seqs:
            if name.startswith('store:'):
                seqs[name] = store_data(int(name.split(':')[1]), cache)
            else:
                seqs[name] = flight_data(name, dirs)
            streams[name] = _load_stream(out, name)
            assert len(streams[name]['time']) == len(seqs[name]['t']), name
            aims[name] = replay_aim(streams[name], pilot, ar)
        return seqs[name], streams[name], aims[name]
    # S1 / S2 / A1
    s1_all, s2_frames, s2_pooled, s1_per, s2_per, a1 = [], [], [], {}, {}, {}
    for name in gates['S1']['approaches']:
        seq, st, aim = get(name)
        s1, s2 = s1_s2(name, seq, st, gates, config)
        s1_all.extend(s1['ratios'])
        s2_frames.extend(s2['frame_medians'])
        s2_pooled.extend(s2['pooled'])
        s1_per[name] = dict(_summ_ratio(s1['ratios'], gates['S1']['median_range'], gates['S1']['band'],
                                        gates['S1']['min_band_fraction']), valid_kind_samples=s1['valid_kind_samples'],
                            frames_in_region_fast=s1['frames_in_region_fast'])
        s2_per[name] = dict(frames=s2['frames'], blocks=s2['blocks'],
                            pooled_median=round(float(np.median(s2['pooled'])), 3) if s2['pooled'] else None)
        a1[name] = a1_approach(name, seq, st, aim, gates, models)
    g = gates['S1']
    res['S1'] = dict(total=_summ_ratio(s1_all, g['median_range'], g['band'], g['min_band_fraction']), per_flight=s1_per)
    res['S1']['passes'] = res['S1']['total']['passes']
    g = gates['S2']
    fm = np.asarray(s2_frames)
    pm = float(np.median(s2_pooled)) if s2_pooled else None
    ff = float(((fm >= g['frame_band'][0]) & (fm <= g['frame_band'][1])).mean()) if len(fm) else None
    res['S2'] = dict(frames=int(len(fm)), blocks=len(s2_pooled), pooled_median=None if pm is None else round(pm, 3),
                     frame_band_fraction=None if ff is None else round(ff, 3),
                     frame_median_p10_p50_p90=[round(float(x), 3) for x in np.percentile(fm, [10, 50, 90])] if len(fm) else None,
                     per_flight=s2_per,
                     passes=bool(pm is not None and g['pooled_median_range'][0] <= pm <= g['pooled_median_range'][1]
                                 and ff is not None and ff >= g['min_frame_fraction']))
    g = gates['A1']
    n_pass = sum(v['passes'] for v in a1.values())
    n_right = sum(v['confirmed_right_ticks'] for v in a1.values())
    n_into = sum(v['aim_into_pillar_ticks'] for v in a1.values())
    res['A1'] = dict(approaches=a1, passing=n_pass, required=g['min_passing'], confirmed_right_beyond_2m_ticks=n_right,
                     aim_into_pillar_ticks=n_into,
                     passes=bool(n_pass >= g['min_passing'] and n_right == 0 and n_into == 0))
    res['A1_reported_not_gated'] = {}
    for name in gates['S1']['reported_not_gated']:
        seq, st, aim = get(name)
        res['A1_reported_not_gated'][name] = a1_approach(name, seq, st, aim, gates, models)
    # S3
    g = gates['S3']
    s3 = {}
    for env, names in g['environments'].items():
        num = den = 0
        per = {}
        for name in names:
            seq, st, aim = get(name)
            sp = np.hypot(seq['vel'][:, 0], seq['vel'][:, 1])
            kinds = np.array([fs.kind_name(k) for k in st['kind']])
            m = seq['tel'] & (sp > g['min_speed_mps']) & ~np.isin(kinds, ('no_ring', 'no_pose'))
            v = m & np.isin(kinds, VALID_KINDS)
            num += int(v.sum())
            den += int(m.sum())
            per[name] = dict(frames=int(m.sum()), valid=int(v.sum()),
                             kinds={k: int((m & (kinds == k)).sum()) for k in fs.PLAN_KINDS if (m & (kinds == k)).any()})
        s3[env] = dict(frames=den, valid=num, valid_fraction=round(num / max(den, 1), 4), per_flight=per,
                       passes=bool(den and num / den >= g['min_valid_fraction']))
    res['S3'] = dict(environments=s3, passes=bool(all(v['passes'] for v in s3.values())))
    # B1
    g = gates['B1']
    b1 = {}
    for rid in g['store_runs']:
        name = f'store:{rid}'
        seq, st, aim = get(name)
        tti_tick = np.interp(aim['t'], seq['t'], seq['tti'])
        tti_conf = np.interp(np.where(np.isfinite(aim['conf_time']), aim['conf_time'], aim['t']), seq['t'], seq['tti'])
        w = (tti_tick >= g['window_tti_s'][0]) & (tti_tick <= g['window_tti_s'][1])
        xdir = aim['int_az'] * (-np.sin(np.radians(np.nan_to_num(aim['ring_az']))))
        conf_lr = np.isin(aim['confirmed'], (1, -1))
        mx = np.flatnonzero(w & conf_lr & (xdir < 0))
        px = np.flatnonzero(w & conf_lr & (xdir > 0))
        first = float(tti_conf[mx[0]]) if len(mx) else None
        b1[rid] = dict(first_minus_x_tti_s=None if first is None else round(first, 2),
                       passes=bool(first is not None and first >= g['min_lead_s']),
                       plus_x_before_tti_s=sorted({round(float(tti_conf[k]), 2) for k in px if not len(mx) or k < mx[0]}),
                       kinds_in_window={k: int(v) for k, v in zip(*np.unique(
                           [fs.kind_name(x) for x, tt in zip(st['kind'], seq['tti']) if g['window_tti_s'][0] <= tt <= g['window_tti_s'][1]],
                           return_counts=True))})
    res['B1'] = dict(runs=b1, passes=bool(all(v['passes'] for v in b1.values())))
    # T1
    g = gates['T1']
    seq, st, aim = get(g['flight'])
    imp = seq['impact']
    tti_conf = imp - np.where(np.isfinite(aim['conf_time']), aim['conf_time'], aim['t'])
    tti_tick = imp - aim['t']
    right = np.flatnonzero(aim['confirmed'] == -1)
    left_late = np.flatnonzero((aim['confirmed'] == 1) & (tti_tick <= g['no_left_last_s']) & (tti_tick >= 0))
    first_r = float(tti_conf[right[0]]) if len(right) else None
    raw_r = [float(imp - st['time'][i]) for i in range(len(st['time']))
             if fs.kind_name(st['kind'][i]) == 'shift' and st['cls'][i] == -1 and imp - st['time'][i] >= 0]
    mdl = models['brain08']
    res['T1'] = dict(first_confirmed_right_tti_s=None if first_r is None else round(first_r, 2),
                     confirmed_left_last_2s_ticks=int(len(left_late)),
                     first_raw_right_tti_s=round(max(raw_r), 2) if raw_r else None,
                     brain08_kinematic_clearance_m=None if first_r is None else round(
                         0.5 * mdl.a_lat_mps2 * max(first_r - mdl.delay_s, 0.) ** 2, 3),
                     last_2s_samples=[dict(tti_s=round(float(imp - st['time'][i]), 2), kind=fs.kind_name(st['kind'][i]),
                                           cls=None if not np.isfinite(st['cls'][i]) else int(st['cls'][i]),
                                           az=None if not np.isfinite(st['az'][i]) else round(float(st['az'][i]), 1),
                                           l_az=None if not np.isfinite(st['l_az'][i]) else round(float(st['l_az'][i]), 1),
                                           r_az=None if not np.isfinite(st['r_az'][i]) else round(float(st['r_az'][i]), 1),
                                           d_free_ring=None if not np.isfinite(st['d_free_ring'][i]) else round(float(st['d_free_ring'][i]), 2))
                                      for i in range(len(st['time'])) if 0 <= imp - st['time'][i] <= 2.0],
                     reference=g['reference'],
                     passes=bool(first_r is not None and first_r >= g['min_lead_s'] and len(left_late) == 0))
    # Q1
    g = gates['Q1']
    q = {}
    for name in g['flights']:
        seq, st, aim = get(name)
        q[name] = q1_flight(name, seq, st, aim, gates, _csv_cmd(name))
    mins = sum(v['minutes'] for v in q.values())
    eps = sum(v['episodes'] for v in q.values())
    veps = sum(v['vertical_episodes'] for v in q.values())
    ueps = sum(v['urgent_episodes'] for v in q.values())
    climbs = sum(v['terrain_climbs'] for v in q.values())
    allabs = np.concatenate([np.asarray(v.pop('app_abs_az')) for v in q.values()])
    sw = sum(v['switches'] for v in q.values())
    swq = sum(v['switches_quiet'] for v in q.values())
    fb = [v['floor_bound_active_s'] for v in q.values() if v['floor_bound_active_s'] is not None]
    tot = dict(minutes=round(mins, 2), episodes_per_min=round(eps / mins, 2), vertical_per_min=round(veps / mins, 2),
               urgent_per_min=round(ueps / mins, 2), terrain_climbs_per_min=round(climbs / mins, 2),
               applied_abs_az_p90=round(float(np.percentile(allabs, 90)), 2) if len(allabs) else 0.,
               switches=sw, switch_quiet_fraction=round(swq / max(sw, 1), 4),
               floor_bound_active_s_per_min=round(sum(fb) / mins, 2) if fb else None)
    checks = dict(episodes=tot['episodes_per_min'] <= g['max_episodes_per_min'],
                  vertical=tot['vertical_per_min'] <= g['max_vertical_episodes_per_min'],
                  urgent=tot['urgent_per_min'] <= g['max_urgent_episodes_per_min'],
                  terrain=tot['terrain_climbs_per_min'] <= g['max_terrain_climbs_per_min'],
                  p90=tot['applied_abs_az_p90'] <= g['p90_abs_az_max_deg'],
                  switches=tot['switch_quiet_fraction'] >= g['switch_min_fraction'])
    res['Q1'] = dict(flights=q, total=tot, checks=checks, passes=bool(all(checks.values())))
    # L1
    g = gates['L1']
    rows, hud_n, hud_bad = [], 0, 0
    per = {}
    for name in g['flights']:
        p = out / 'streams' / f'{name}.leaks.json'
        obj = json.loads(p.read_text(encoding='utf-8'))
        lk = obj['leaks']
        same = [(r['o_kind'] == r['p_kind']) and _close(r['o_az'], r['p_az'], g['max_dangle_deg'])
                and _close(r['o_el'], r['p_el'], g['max_dangle_deg']) for r in lk]
        nc = [(r['o_kind'] in ACTING_KINDS or r['p_kind'] in ACTING_KINDS) for r in lk]
        rows.extend(zip(same, nc, [r['kind'] for r in lk]))
        hk = obj['hud']
        hud_n += len(hk)
        bad = sum(k in ('blocked', 'shift') for k in hk)
        hud_bad += bad
        per[name] = dict(n=len(lk), same=int(sum(same)), non_clear=int(sum(nc)),
                         same_non_clear=int(sum(s for s, c in zip(same, nc) if c)), hud_frames=len(hk), hud_bad=bad)
    same = np.array([r[0] for r in rows])
    nc = np.array([r[1] for r in rows])
    kinds = np.array([r[2] for r in rows])
    l1 = dict(n=int(len(rows)), same_fraction=round(float(same.mean()), 4) if len(rows) else None,
              non_clear=int(nc.sum()), same_fraction_non_clear=round(float(same[nc].mean()), 4) if nc.any() else None,
              by_perturbation={k: round(float(same[kinds == k].mean()), 4) for k in np.unique(kinds)},
              hud_only=dict(frames=hud_n, blocked_or_shift=hud_bad), per_flight=per)
    l1['passes'] = bool(l1['same_fraction'] is not None and l1['same_fraction'] >= g['min_same_fraction']
                        and (l1['same_fraction_non_clear'] is None or l1['same_fraction_non_clear'] >= g['min_same_fraction_non_clear'])
                        and hud_bad == 0)
    res['L1'] = l1
    # performance of the offline planner (CPU, 2 threads; not the R1 runtime bench)
    lk_ms = np.concatenate([np.asarray(s['lk_ms'], float) for s in streams.values()])
    pl_ms = np.concatenate([np.asarray(s['plan_ms'], float) for s in streams.values()])
    res['offline_timing_ms'] = dict(lk_p50=_p(lk_ms, 50), lk_p95=_p(lk_ms, 95), plan_p50=_p(pl_ms, 50), plan_p95=_p(pl_ms, 95))
    res['aim_counts'] = {k: v['counts'] for k, v in aims.items()}
    res['verdict'] = {k: res[k]['passes'] for k in ('S1', 'S2', 'S3', 'A1', 'B1', 'T1', 'Q1', 'L1')}
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=_json_default) + '\n', encoding='utf-8')
    _log(json.dumps(res['verdict']))
    return res


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _p(x, q):
    x = x[np.isfinite(x)]
    return round(float(np.percentile(x, q)), 2) if len(x) else None


def _close(a, b, tol):
    a, b = float(a), float(b)
    if not np.isfinite(a) and not np.isfinite(b):
        return True
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) <= tol


def cmd_dev(args):
    """Report-only dev numbers (before or after the freeze): A1-style pillar-A numbers of the dev Minus runs and Q1-style
    episode rates of the dev Straw flight, with the current (draft or frozen) configs."""
    limit_threads(2, cv2=True)
    config, raw, cfg_sha = fs.load_config()
    gates, gates_sha = load_gates()
    models = fs.load_response_models(config.response_models)
    dirs = _dirs(args)
    out = Path(args.out) / 'dev'
    out.mkdir(parents=True, exist_ok=True)
    res = dict(free_space_sha256=cfg_sha, gates_sha256=gates_sha, note='dev data: report only')
    for name in args.names.split(','):
        if name.startswith('store:'):
            rid = name.split(':')[1]
            seq = store_data(int(rid), Path(args.cache))
            motor = 'brain08' if int(rid) != 122 and int(rid) != 123 else 'fast_pd'
        else:
            seq = flight_data(name, dirs)
            motor = _motor(gates, gates['flights'][name]['motor'])
        r = run_sequence(seq, config, motor, offline_age=float(gates['offline_age_s']), log=_log)
        st = r['stream']
        save_stream(out / f'{name.replace(":", "_")}.plan.npz', st, dict(pos=seq['pos'], vel=seq['vel']))
        aim = replay_aim(st, config.pilot, gates['aim_replay'])
        entry = dict(kinds={k: v for k, v in r['counts'].items() if v}, seconds=round(r['seconds'], 1),
                     timing_ms=dict(lk_p50=_p(np.asarray(st['lk_ms']), 50), lk_p95=_p(np.asarray(st['lk_ms']), 95),
                                    plan_p50=_p(np.asarray(st['plan_ms']), 50), plan_p95=_p(np.asarray(st['plan_ms']), 95)),
                     aim=aim['counts'])
        if 'minus' in name or name in ('store:117', 'store:118', 'store:119', 'store:120', 'store:121', 'store:122'):
            entry['A1'] = a1_approach(name, seq, st, aim, gates, models)
            s1, s2 = s1_s2(name, seq, st, gates, config)
            entry['S1'] = _summ_ratio(s1['ratios'], gates['S1']['median_range'], gates['S1']['band'], gates['S1']['min_band_fraction'])
            entry['S2'] = dict(frames=s2['frames'], blocks=s2['blocks'],
                               pooled_median=round(float(np.median(s2['pooled'])), 3) if s2['pooled'] else None)
        if 'straw' in name:
            qq = q1_flight(name, seq, st, aim, gates, None)
            qq.pop('app_abs_az')
            qq['episodes_per_min'] = round(qq['episodes'] / max(qq['minutes'], 1e-9), 2)
            entry['Q1'] = qq
        res[name] = entry
        _log(name, json.dumps(entry, default=_json_default)[:1500])
    (out / f'dev_{args.tag}.json').write_text(json.dumps(res, indent=1, default=_json_default) + '\n', encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m haltere.obstacles.free_space_eval')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('frames', 'depth', 'run', 'score', 'dev'):
        a = sub.add_parser(name)
        a.add_argument('--out', required=True)
        a.add_argument('--cache', default=None, help='the gap-cue evaluation directory (cached frames and depth)')
        a.add_argument('--flight-lock', default=None)
        a.add_argument('--flights', default=None)
        a.add_argument('--names', default='')
        a.add_argument('--batch', type=int, default=32)
        a.add_argument('--max-s', type=float, default=300.0)
        a.add_argument('--force', action='store_true')
        a.add_argument('--draft', action='store_true', help='run: allow unfrozen configs (dev data only)')
        a.add_argument('--no-leaks', action='store_true')
        a.add_argument('--tag', default='draft')
    sub.add_parser('freeze')
    args = ap.parse_args(argv)
    {'frames': cmd_frames, 'depth': cmd_depth, 'freeze': cmd_freeze, 'run': cmd_run, 'score': cmd_score,
     'dev': cmd_dev}[args.cmd](args)


if __name__ == '__main__':
    main()
