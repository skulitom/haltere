"""Frame-store builder and timing re-pose (offline only).

``python -m haltere.obstacles.store build`` calls ``build(args)``; ``repose`` calls ``repose(args)``.

Data root. Recordings live in the main checkout (``runs/``, ``data/vision``), which is not where
this code may run from (git worktrees). Every source path is resolved against the inventory
``root`` (override: ``--data-root``); the resolved root is written to plan.json/manifest.json
(``build.data_root``) so ``repose`` and ``timing`` find the same files.

build(args):

1. Refuses to start without a flight lock path (thermal.require_flight_lock_path), limits OpenMP,
   BLAS and OpenCV to two threads and calls ChunkGuard.before_chunk() before every run and at
   least every CHUNK_FRAMES decoded frames (ffmpeg pipes simply block while paused).
2. Reads the environment of every source from configs/obstacles/folds.json (never re-derived).
   Sealed sources are skipped unless --sealed-final, which writes them only to the 'sealed' part.
3. Plans one source per physical flight, best first: geometry PNG (exact), capture dataset,
   good/fair aligned run video (unreliable videos only with --include-unreliable). Derived
   data/vision subsets are skipped. Flights named in --secondary also store their next source,
   flagged SECONDARY_SOURCE. Plan order: environment code, then flight key (run_id = position).
4. Per run: pass 1 looks at every frame at 320 x 180 (repeat test, pause menu, HUD presence);
   frames are dropped for these reasons (counted in runs.json ``dropped``):

   repeat             recorder repeat (grey mean |diff| < 0.1 at 160 x 90) or unchanged game clock
   no_pose            geometry-worker frame without a pose (waiting/stale motion)
   outside_telemetry  capture time outside the telemetry span, no pose, or across a reset
   telemetry_gap      interpolation across a telemetry gap > 0.25 s
   menu               pause-menu detector (+-3 frames) or a stalled game clock (paused game)
   countdown          before the launch of each attempt (race-start countdown / drone on the ground;
                      also covers post-reset countdowns)
   finish             trailing HUD-absent results screen of a run that ended without an impact
   post_event         after the terminal impact, within POST_CONTACT_DROP_S after a contact,
                      and from a contact to a reset that follows it within 10 s (crash)

   Of the remaining (valid) frames every s-th is kept (the stride grid) plus every valid frame
   with tti_s <= PRE_EVENT_DENSE_S and every valid clean-window frame (DENSE_EXTRA when off the
   grid). The grid step is s = clip(floor(r / 2.5 Hz), 1, stride) where r is the source's valid
   new-frame rate: 13-18 Hz run videos and manual captures keep every 3rd frame (the plan's
   stride), 7-9 Hz capture sets every 2nd or 3rd, and the ~5 Hz exact geometry PNGs and the
   3 Hz capture sets are all kept, so no source falls below ~2.5 Hz. Pass 2 resizes the kept
   frames to 448 x 252 with cv2.INTER_AREA from the 1280 x 720 gameplay crop (x >= 648) or the
   640 x 360 image.
5. Index rows follow store.py. Clocks and poses per source:

   run video          t_wall = csv.wall[0] + t_video + offset (offset = alignment.used_offset_s,
                      else offset_after_first_row_s; t_video = last repeated copy / fps; wall[0] =
                      the CSV's first data row, as in the aligner). Pose TELEMETRY_INTERP from the
                      100 Hz CSV on its ``wall`` (row write) clock, the clock the offsets were fitted on.
   geometry PNG       t_wall = worker capture_time (monotonic grab start) + C, C = the monotonic->epoch
                      offset (1st percentile of csv.wall - csv.frame_time). Pose WORKER_INTERP (the
                      worker's camera_position/quaternion: the CSV pose interpolated on the receipt
                      clock at the grab start); vel/omega/phase/ts from the CSV on its receipt clock
                      (frame_time + C) at t_wall.
   capture dataset    DatasetWriter sets (raw telemetry.csv): t_wall = grab start (epoch), pose
                      TELEMETRY_INTERP from the raw UDP log (receive clock, Unity -> simulator FLU,
                      launch-relative to runs.json origin_sim). Older recorder sets (no raw log, no
                      logged age): SOURCE_RAW (the shared-state pose logged with the image; t_wall =
                      the index wall_time, written after the grab); vel/omega/phase from the flight
                      CSV (receipt clock) where one exists, else finite differences of the logged
                      poses; pose_lag_s = t_wall - CSV time of the logged game timestamp where a CSV
                      exists, else NaN.

   cue_uv is LOGGED from the flight CSV cue columns: the pilot detection whose capture time
   (capture_time + C) is nearest to t_wall within 50 ms; NaN when absent (-1), off the image or
   clamped at the edge.
6. events.json: {store_event_id, run_id, source_id, flight, env, kind, t_wall, t_phase, t_game,
   drone_pos_w, speed_mps, accel_mps2, unexplained_mps2, source, clock}; ``source`` is sidecar
   (terminal impact), csv_contact (haltere.liftoff.flightlog.collisions per attempt, as the survey),
   capture_contact (the same detector on a capture set's ~35 Hz raw UDP log) or lateral_manifest
   (curated non-fatal contacts the detector missed). ``t_wall`` is on the run's own frame clock.
   clean_windows.json: the curated lateral-study windows ({run_id, source_id, alias, start_phase_s,
   end_phase_s, criterion, origin}); every valid frame inside them is stored.
7. finalize(); writes build_report.json with counts per environment and fold, disk size, the
   pass criteria and consistency checks. Video runs also keep parts/rNNNNN.frames.npz (every
   decoded new frame: index, last copy, drop reason, kept flag) for haltere.obstacles.timing.

repose(args) applies timing/refined.json (per-run delta_s from timing.py, accepted runs only). Run
videos: t_wall += delta, pose, vel, omega, t_phase, t_game and cue_uv re-interpolated from the run CSV
at the corrected time, tti_s and the PRE_EVENT / IN_CLEAN / HIGH_YAW_RATE flags recomputed,
Flag.TIMING_REFINED and timing_delta_s set. Older recorder capture sets (``pose_lag``): pos/quat from
the logged pose series at t_wall + delta, pose_method SOURCE_COMPENSATED (repose_capture_rows). Pose
gate: every run whose timing residual after the applied delta exceeds ``--pose-gate-px`` (2 px at
640, twice the K0a threshold) is regraded UNRELIABLE (runs.json pose_check records every scored run). The previous index is kept as index.v<k>.npy,
runs.json records alignment.refine_delta_s, manifest.json gets the new index_sha256. Labels
built on an older index become invalid (LabelSet checks the hash).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from . import contract, thermal
from .splits import ENV_BY_CODE, ENV_CODE, FOLDS, SEALED_ENVS, env_side, load_folds_config
from .store import (FRAME_SHAPE, HIGH_YAW_RATE, INDEX_DTYPE, POST_CONTACT_DROP_S, PRE_EVENT_DENSE_S, REPO_ROOT,
                    CueSource, Flag, Grade, PoseMethod, Source, StoreWriter, _json_dump, empty_index, index_sha256,
                    run_stem)

BUILD_SCHEMA = 'haltere.obstacles.store_build.v1'
CHUNK_FRAMES = 500            # ChunkGuard.before_chunk() at least this often (decoded frames)
REPEAT_GREY = 0.1             # recorder repeat: mean |grey diff| at 160 x 90 below this
MIN_KEPT_RATE_HZ = 2.5        # stride grid step s = clip(floor(rate / 2.5), 1, stride)
MENU_DILATE = 3               # frames around a detected pause menu
TELEMETRY_GAP_S = 0.25
PAUSE_RATE = 0.5              # game clock advancing slower than this fraction of the wall clock = paused
LAUNCH_SPEED = 0.5            # m/s: an attempt has launched once it moves this fast ...
LAUNCH_DISP = 0.3             # ... or this far from its start
CRASH_RESET_S = 10.0          # a contact followed by a reset within this is a crash (dropped to the reset)
FINISH_TAIL_S = 3.0           # finish screen: trailing HUD-absent frames within the last 3 s of telemetry
FINISH_MIN_S = 0.3
CUE_MATCH_S = 0.05
EPOCH_PERCENTILE = 1.0        # C = this percentile of (csv.wall - csv.frame_time)
GAMEPLAY_W, GAMEPLAY_H = 1280, 720
VIDEO_DECODE = ('ffmpeg rgb24 over a pipe (-fps_mode passthrough, 2 decoder threads, gameplay crop x >= width-1280); '
                'pass 1 scale=320:180:flags=area')
DROP_REASONS = ('', 'repeat', 'no_pose', 'outside_telemetry', 'telemetry_gap', 'menu', 'countdown', 'finish',
                'post_event')
DROP_CODE = {r: i for i, r in enumerate(DROP_REASONS)}
LATERAL_RUN_PREFIX = 'fast-stack-20260923/'   # lateral-study run names are fast-stack short names
_M_UNITY = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])   # sim = M @ unity (frames.py)


def _log(*a):
    print(*a, flush=True)


def _posix(path) -> str | None:
    return None if path is None else str(path).replace('\\', '/')


def data_path(root, rel) -> Path | None:
    """Absolute path of an inventory-relative source path (the data root is the main checkout)."""
    if rel is None:
        return None
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


# ----------------------------------------------------------------------------- quaternions

def _qmul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _qnorm(q):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def omega_from_quat_series(q, t):
    """(n, 3) body-frame rates (rad/s) of world-from-body wxyz quaternions by central differences."""
    q = _qnorm(np.asarray(q, np.float64))
    t = np.asarray(t, np.float64)
    n = len(q)
    out = np.full((n, 3), np.nan)
    if n < 2:
        return out
    i0 = np.clip(np.arange(n) - 1, 0, n - 1)
    i1 = np.clip(np.arange(n) + 1, 0, n - 1)
    dt = t[i1] - t[i0]
    dq = _qmul(_qconj(q[i0]), q[i1])
    dq = np.where(dq[:, :1] < 0, -dq, dq)
    v = dq[:, 1:]
    nv = np.linalg.norm(v, axis=1)
    ang = 2.0 * np.arctan2(nv, dq[:, 0])
    ok = (dt > 1e-4) & (dt < 0.5)
    axis = v / np.maximum(nv, 1e-12)[:, None]
    out[ok] = axis[ok] * (ang[ok] / dt[ok])[:, None]
    out[ok & (nv < 1e-12)] = 0.0
    return out


def _mat_to_quat(R):
    """(n, 3, 3) rotation matrices -> (n, 4) wxyz (w >= 0)."""
    R = np.asarray(R, np.float64)
    n = len(R)
    q = np.empty((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for k in range(n):
        m = R[k]
        if tr[k] > 0:
            s = math.sqrt(tr[k] + 1.0) * 2
            q[k] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[k] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[k] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[k] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = _qnorm(q)
    return np.where(q[:, :1] < 0, -q, q)


def unity_quats_to_sim(q_xyzw):
    """Vectorised haltere.liftoff.frames.unity_quat_to_sim: Unity [x,y,z,w] -> simulator wxyz."""
    x, y, z, w = np.asarray(q_xyzw, np.float64).T
    Ru = contract.quat_wxyz_to_mats(np.stack([w, x, y, z], axis=1))
    Rs = _M_UNITY @ Ru @ _M_UNITY.T
    return _mat_to_quat(Rs)


# ----------------------------------------------------------------------------- telemetry

class Telemetry:
    """A pose time series on an epoch clock: the sampling side of every pose method.

    ``clock`` names the time base of ``wall``: 'wall' (CSV row write time, the video alignment
    clock), 'receipt' (CSV frame_time + C, the UDP receipt time) or 'recv' (raw capture UDP log).
    """

    def __init__(self, wall, ts, pos, quat, vel, omega=None, phase=None, *, epoch_offset=None, cue=None,
                 kind='run_csv', path=None, clock='wall', wall0=None, notes=None):
        wall = np.asarray(wall, np.float64)
        order = np.argsort(wall, kind='stable')
        if not np.all(order == np.arange(len(wall))):
            wall = wall[order]
            ts, pos, quat, vel = (np.asarray(a)[order] for a in (ts, pos, quat, vel))
            omega = None if omega is None else np.asarray(omega)[order]
            phase = None if phase is None else np.asarray(phase)[order]
        self.wall = wall
        self.ts = np.asarray(ts, np.float64)
        self.pos = np.asarray(pos, np.float64)
        q = _qnorm(np.asarray(quat, np.float64))
        flip = np.r_[False, np.einsum('ij,ij->i', q[1:], q[:-1]) < 0]   # hemisphere continuity for interpolation
        q = np.where((np.cumsum(flip) % 2 == 1)[:, None], -q, q)
        self.quat = q
        self.vel = np.asarray(vel, np.float64)
        self.omega_logged = omega is not None
        self.omega = np.asarray(omega, np.float64) if omega is not None else omega_from_quat_series(q, self.ts)
        self.phase = None if phase is None else np.asarray(phase, np.float64)
        self.epoch_offset = epoch_offset
        self.cue = cue
        self.kind, self.path, self.clock = kind, path, clock
        self.wall0 = float(wall[0]) if wall0 is None else float(wall0)
        self.notes = dict(notes or {})
        resets = np.r_[False, np.diff(self.ts) < -0.5]
        self.attempt = np.cumsum(resets)
        self.attempt_starts = np.r_[0, np.flatnonzero(resets)]
        self.attempt_ends = np.r_[self.attempt_starts[1:], len(self.ts)]
        self.launch_wall = self._launches()
        self.reset_walls = self.wall[self.attempt_starts[1:]]

    def __len__(self):
        return len(self.wall)

    def _launches(self):
        out = []
        for a, b in zip(self.attempt_starts, self.attempt_ends):
            sp = np.linalg.norm(self.vel[a:b], axis=1)
            disp = np.linalg.norm(self.pos[a:b] - self.pos[a], axis=1)
            moving = np.flatnonzero((sp > LAUNCH_SPEED) | (disp > LAUNCH_DISP))
            out.append(self.wall[a + moving[0]] if len(moving) else np.inf)
        return np.array(out)

    def sample(self, t):
        """Interpolate at epoch times t (n,) -> dict of arrays plus validity reasons."""
        t = np.asarray(t, np.float64)
        n = len(self.wall)
        i1 = np.clip(np.searchsorted(self.wall, t, side='right'), 1, n - 1)
        i0 = i1 - 1
        span = self.wall[i1] - self.wall[i0]
        a = np.clip((t - self.wall[i0]) / np.maximum(span, 1e-9), 0.0, 1.0)
        inside = np.isfinite(t) & (t >= self.wall[0]) & (t <= self.wall[-1])
        same = self.attempt[i0] == self.attempt[i1]
        gap = span > TELEMETRY_GAP_S

        def lerp(arr):
            return arr[i0] * (1 - a)[:, None] + arr[i1] * a[:, None]
        out = dict(pos=lerp(self.pos), quat=_qnorm(lerp(self.quat)), vel=lerp(self.vel), omega=lerp(self.omega),
                   ts=self.ts[i0] * (1 - a) + self.ts[i1] * a,
                   attempt=np.where(a < 0.5, self.attempt[i0], self.attempt[i1]))
        if self.phase is not None:
            ph = self.phase[i0] * (1 - a) + self.phase[i1] * a
            # continue the control clock linearly outside the span (clean windows use it)
            ph = np.where(t > self.wall[-1], self.phase[-1] + (t - self.wall[-1]), ph)
            ph = np.where(t < self.wall[0], self.phase[0] - (self.wall[0] - t), ph)
        else:
            ph = t - self.wall0
        out['phase'] = ph
        out['valid'] = inside & same & ~gap
        out['outside'] = ~(inside & same)
        out['gap'] = inside & same & gap
        return out

    def game_rate(self, t, half=0.15):
        """Game-clock rate d(ts)/d(wall) around t (1 = running, 0 = paused)."""
        t = np.asarray(t, np.float64)
        lo = np.clip(t - half, self.wall[0], self.wall[-1])
        hi = np.clip(t + half, self.wall[0], self.wall[-1])
        dt = hi - lo
        rate = np.where(dt > 0.05, (np.interp(hi, self.wall, self.ts) - np.interp(lo, self.wall, self.ts))
                        / np.maximum(dt, 1e-9), 1.0)
        # a reset inside the window makes the rate negative: not a pause
        return np.where(rate < -0.5, 1.0, rate)

    def wall_at_phase(self, phase):
        if self.phase is None:
            return self.wall0 + phase
        k = int(np.argmin(np.abs(self.phase - phase)))
        return float(self.wall[k] + (phase - self.phase[k]))

    def wall_at_ts(self, ts, near_wall):
        """Epoch time of game times ``ts`` (n,), each looked up in the attempt that contains its ``near_wall``
        epoch time (game time restarts at every reset); NaN when that attempt does not contain it."""
        ts = np.atleast_1d(np.asarray(ts, np.float64))
        near_wall = np.atleast_1d(np.asarray(near_wall, np.float64))
        out = np.full(len(ts), np.nan)
        k = np.clip(np.searchsorted(self.wall, near_wall), 0, len(self.wall) - 1)
        att = self.attempt[k]
        for a_id, (a, b) in enumerate(zip(self.attempt_starts, self.attempt_ends)):
            seg_ts, seg_w = self.ts[a:b], self.wall[a:b]
            m = (att == a_id) & (ts >= seg_ts[0]) & (ts <= seg_ts[-1]) if b - a >= 2 else None
            if m is not None and m.any():
                u, iu = np.unique(seg_ts, return_index=True)
                out[m] = np.interp(ts[m], u, seg_w[iu])
        return out

    def row_at_ts(self, ts, last_attempt=True):
        """Row index of game time ``ts`` (nearest), searching the last attempt first."""
        order = range(len(self.attempt_starts) - 1, -1, -1) if last_attempt else range(len(self.attempt_starts))
        best = None
        for k in order:
            a, b = self.attempt_starts[k], self.attempt_ends[k]
            seg = self.ts[a:b]
            if seg[0] - 0.5 <= ts <= seg[-1] + 0.5:
                return int(a + np.argmin(np.abs(seg - ts)))
            j = int(a + np.argmin(np.abs(seg - ts)))
            if best is None or abs(self.ts[j] - ts) < abs(self.ts[best] - ts):
                best = j
        return best

    def cue_at(self, t):
        """LOGGED ring cue (u, v) nearest in capture time to t (epoch), NaN if none / clamped / too far."""
        t = np.asarray(t, np.float64)
        out = np.full((len(t), 2), np.nan)
        if self.cue is None or len(self.cue['t']) < 2:
            return out
        ct = self.cue['t']
        j = np.clip(np.searchsorted(ct, t), 1, len(ct) - 1)
        j = np.where(np.abs(ct[j - 1] - t) < np.abs(ct[j] - t), j - 1, j)
        ok = (np.abs(ct[j] - t) <= CUE_MATCH_S) & self.cue['ok'][j]
        out[ok] = self.cue['uv'][j[ok]]
        return out


def _num(d, col):
    import pandas as pd
    return pd.to_numeric(d[col], errors='coerce').to_numpy(np.float64)


def load_run_csv(path, clock='wall') -> Telemetry:
    """A runner flight CSV (~100 Hz). ``clock='receipt'`` puts rows on frame_time + C (UDP receipt, epoch)."""
    import pandas as pd
    path = Path(path)
    with open(path, encoding='utf-8', errors='replace') as f:
        head = f.readline().strip().split(',')
    pos = ['x', 'y', 'z'] if 'x' in head else ['px', 'py', 'pz']
    om = (['omega_x', 'omega_y', 'omega_z'] if 'omega_x' in head else
          (['wx', 'wy', 'wz'] if 'wx' in head else None))
    want = ['wall', 'ts'] + pos + ['vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz'] + (om or [])
    extra = [c for c in ('phase', 'cue_u', 'cue_v', 'cue_edge', 'capture_time', 'frame_time') if c in head]
    d = pd.read_csv(path, usecols=want + extra, low_memory=False)
    wall_raw = _num(d, 'wall')
    wall0 = float(wall_raw[0])        # the aligner's csv.wall[0]: its first data row (prep.py reads every row)
    core = np.stack([_num(d, c) for c in want], 1)
    ok = np.isfinite(core).all(1)
    notes = dict(rows=int(len(d)), rows_nonfinite=int((~ok).sum()))
    ft = _num(d, 'frame_time') if 'frame_time' in d else None
    C = None
    if ft is not None:
        m = ok & np.isfinite(ft)
        if m.sum() > 10:
            lat = wall_raw[m] - ft[m]
            C = float(np.percentile(lat, EPOCH_PERCENTILE))
            notes.update(epoch_offset_s=C, write_latency_median_s=round(float(np.median(lat) - C), 5),
                         write_latency_p95_s=round(float(np.percentile(lat, 95) - C), 5))
    t = wall_raw
    used_clock = 'wall'
    if clock == 'receipt' and C is not None:
        t = ft + C
        ok &= np.isfinite(t)
        used_clock = 'receipt'
    idx = np.flatnonzero(ok)
    if used_clock == 'receipt':
        # consecutive controller ticks between two packets log the same receipt time and pose: keep the first
        tt = t[idx]
        first = np.r_[True, np.diff(tt) > 0]
        idx = idx[first]
    core = core[idx]
    t = t[idx]
    P, V, Q = core[:, 2:5], core[:, 5:8], core[:, 8:12]
    W = core[:, 12:15] if om else None
    phase = None
    if 'phase' in d:
        phase = _num(d, 'phase')[idx]
        fin = np.isfinite(phase)
        if not fin.all():
            phase = np.interp(np.arange(len(phase)), np.flatnonzero(fin), phase[fin]) if fin.any() else None
    cue = None
    if 'cue_u' in d and 'capture_time' in d and C is not None:
        ct, cu, cv = _num(d, 'capture_time'), _num(d, 'cue_u'), _num(d, 'cue_v')
        edge = (d['cue_edge'].astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy() if 'cue_edge' in d
                else np.zeros(len(d), bool))
        m = np.isfinite(ct)
        ct, cu, cv, edge = ct[m], cu[m], cv[m], edge[m]
        first = np.r_[True, np.diff(ct) != 0]
        ct, cu, cv, edge = ct[first], cu[first], cv[first], edge[first]
        order = np.argsort(ct, kind='stable')
        ct, cu, cv, edge = ct[order], cu[order], cv[order], edge[order]
        okc = np.isfinite(cu) & np.isfinite(cv) & (cu > 0) & (cu < 1) & (cv > 0) & (cv < 1) & ~edge
        cue = dict(t=ct + C, uv=np.stack([cu, cv], 1), ok=okc)
    return Telemetry(t, core[:, 1], P, Q, V, W, phase, epoch_offset=C, cue=cue, kind='run_csv',
                     path=_posix(path), clock=used_clock, wall0=wall0 if used_clock == 'wall' else None, notes=notes)


def load_capture_telemetry(dsdir, origin_sim) -> Telemetry | None:
    """DatasetWriter raw UDP log (Unity frame, receive clock) -> launch-relative simulator FLU."""
    import pandas as pd
    p = Path(dsdir) / 'telemetry.csv'
    if not p.exists() or origin_sim is None:
        return None
    d = pd.read_csv(p, usecols=['recv_time', 'timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz'])
    a = d.to_numpy(np.float64)
    a = a[np.isfinite(a).all(1)]
    # results/menu telemetry has an all-zero position despite an advancing clock (liftoff.manual_recording.live_pose)
    a = a[np.any(a[:, 2:5] != 0, axis=1) & (np.abs(np.linalg.norm(a[:, 5:9], axis=1) - 1) < 0.01)]
    if len(a) < 20:
        return None
    wall, ts = a[:, 0], a[:, 1]
    pos = a[:, 2:5] @ _M_UNITY.T - np.asarray(origin_sim, np.float64)
    quat = unity_quats_to_sim(a[:, 5:9])
    vel = a[:, 9:12] @ _M_UNITY.T
    return Telemetry(wall, ts, pos, quat, vel, None, None, kind='capture_telemetry', path=_posix(p), clock='recv')


def capture_index_telemetry(dsdir):
    """(Telemetry, index DataFrame) of the poses logged next to each capture-set image, keyed by the index
    wall_time (rows with a repeated wall_time dropped). For older recorder sets this is the only pose source."""
    import pandas as pd
    idx = pd.read_csv(Path(dsdir) / 'index.csv').sort_values('wall_time', kind='stable').reset_index(drop=True)
    t = idx['wall_time'].to_numpy(np.float64)
    keep = np.r_[True, np.diff(t) > 1e-4]
    idx, t = idx[keep].reset_index(drop=True), t[keep]
    P = idx[['px', 'py', 'pz']].to_numpy(np.float64)
    Q = _qnorm(idx[['qw', 'qx', 'qy', 'qz']].to_numpy(np.float64))
    ts = idx['ts'].to_numpy(np.float64) if 'ts' in idx else t
    tel = Telemetry(t, ts, P, Q, np.nan_to_num(_gradient(P, t)), None, None, kind='capture_index', clock='index')
    return tel, idx


# ----------------------------------------------------------------------------- image tests (320 x 180 RGB)

def pause_menu_320(s):
    """Survey decode_scan rule: white panel with a red selected button."""
    c = s[46:134, 129:191].astype(np.int16)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    return bool((c.min(-1) > 225).mean() > 0.45 and ((r > 170) & (g < 110) & (b < 110)).mean() > 0.004)


def hud_present_320(s):
    """The two dashed white framing columns at x = 435 / 844 of the 1280 px gameplay view."""
    lo = s[20:170].min(-1)

    def bar(x0, x1):
        col = lo[:, x0:x1].max(1) > 150
        side = (lo[:, x0 - 4] > 150) | (lo[:, x1 + 3] > 150)
        return int((col & ~side).sum())
    return bar(107, 111) >= 8 and bar(209, 213) >= 8


def grey160(s):
    g = s[..., 0] * 0.299 + s[..., 1] * 0.587 + s[..., 2] * 0.114
    return g.reshape(90, 2, 160, 2).mean((1, 3))


def to_store(rgb):
    import cv2
    if rgb.shape[:2] == FRAME_SHAPE[:2]:
        return np.ascontiguousarray(rgb)
    return cv2.resize(rgb, (FRAME_SHAPE[1], FRAME_SHAPE[0]), interpolation=cv2.INTER_AREA)


def luma(frames):
    f = frames.astype(np.float32)
    g = (f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114).mean((1, 2))
    return np.clip(np.rint(g), 0, 255).astype(np.uint8)


def read_rgb(path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f'cannot read image {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def small320(rgb):
    import cv2
    return cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- planning

def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8', errors='replace'))
    except (OSError, ValueError):
        return None


def _inventory_index(inv):
    src = {}
    for r in inv['run_videos']:
        src[r['id']] = ('run_video', r)
    for r in inv['capture_datasets']:
        if 'id' in r:
            src[r['id']] = ('capture_dataset', r)
    for r in inv['geometry_archives']:
        src[r['id']] = ('geometry_png', r)
    return src


def _is_human(kind, rec):
    if kind == 'capture_dataset':
        ds = rec.get('dataset', '')
        return '/manual_sessions/' in ds or '/human_' in ds
    mode = str((rec.get('controller') or {}).get('control_mode') or '').lower()
    return 'human' in mode or 'manual' in mode


def video_offset(al: dict):
    """(offset_s, basis): alignment.used_offset_s when set, else offset_after_first_row_s."""
    if al.get('used_offset_s') is not None:
        return float(al['used_offset_s']), 'used_offset_s'
    if al.get('offset_after_first_row_s') is not None:
        return float(al['offset_after_first_row_s']), 'offset_after_first_row_s'
    return None, None


def plan_records(inv, cfg, root, *, include_unreliable=False, secondary=(), only=None, part='main'):
    """Ordered plan records (one best source per flight; --secondary flights also their next)."""
    src = _inventory_index(inv)
    videos = {r['id']: r for r in inv['run_videos']}
    recs, skipped = [], []
    flights = sorted(inv['flights'], key=lambda f: (ENV_CODE.get(cfg['flights'].get(f['flight']), 255), f['flight']))
    for f in flights:
        key = f['flight']
        env = cfg['flights'][key]
        if (env in SEALED_ENVS) != (part == 'sealed'):
            continue
        cands = []
        for s in f['sources']:
            kind, r = src[s['id']]
            c = cfg['sources'].get(s['id'])
            if c is None or not c['usable'] or c.get('derived_from'):
                continue
            if c['env'] != env:
                raise ValueError(f'{s["id"]}: folds.json environment {c["env"]} != flight {env}')
            if kind == 'geometry_png':
                if r.get('png_frames', 0) > 0 and r.get('png_dir') and r.get('archive'):
                    cands.append((0, kind, r, Grade.EXACT))
            elif kind == 'capture_dataset':
                cands.append((1, kind, r, Grade.CAPTURE))
            else:
                al = r.get('alignment') or {}
                if not (r.get('files') or {}).get('csv') or video_offset(al)[0] is None:
                    continue
                q = al.get('quality')
                if q == 'good':
                    cands.append((2, kind, r, Grade.GOOD))
                elif q == 'fair':
                    cands.append((2, kind, r, Grade.FAIR))
                elif include_unreliable:
                    cands.append((3, kind, r, Grade.UNRELIABLE))
        cands.sort(key=lambda c: (c[0], int(c[3])))
        if not cands:
            skipped.append(dict(flight=key, env=env, reason='no usable source (unreliable/unaligned video, no '
                                                            'telemetry, or no images)'))
            continue
        chosen = [cands[0]] + ([cands[1]] if key in secondary and len(cands) > 1 else [])
        for k, (_, kind, r, grade) in enumerate(chosen):
            sid = r['id']
            if only and sid not in only and key not in only:
                continue
            linked = sid if kind == 'run_video' else r.get('same_flight_as')
            vid = videos.get(linked) if linked else None
            files = (vid or {}).get('files') or {}
            sidecar = files.get('json')
            csv = files.get('csv')
            meta = _load_json(data_path(root, sidecar)) if sidecar else None
            meta = meta if isinstance(meta, dict) else {}
            origin = meta.get('origin_sim')
            cap_json = None
            if kind == 'capture_dataset':
                cap_json = _load_json(data_path(root, r['dataset']) / 'capture.json') or {}
                if origin is None:
                    origin = cap_json.get('origin_sim')
            if kind == 'run_video':
                path = r.get('video')
            elif kind == 'geometry_png':
                path = r.get('png_dir')
            else:
                path = r.get('dataset')
            al = r.get('alignment') or {}
            if kind == 'run_video':
                off, basis = video_offset(al)
                alignment = dict(grade=grade.name.lower(), offset_s=off, basis=basis, method=al.get('method'),
                                 source=al.get('source'), quality_basis=al.get('quality_basis'), refine_delta_s=0.0)
            else:
                alignment = dict(grade=grade.name.lower(), offset_s=0.0,
                                 basis='exact worker capture time' if kind == 'geometry_png' else 'capture wall_time',
                                 refine_delta_s=0.0)
            controller = dict(vid.get('controller') or {}) if vid else {}
            oracle = (bool(controller.get('runtime_route_oracle')) or bool(r.get('oracle_route'))
                      or bool((cap_json or {}).get('oracle_route')))
            if kind == 'capture_dataset':
                controller.setdefault('capture_source', r.get('source') or (cap_json or {}).get('source'))
                if (cap_json or {}).get('pilot'):
                    controller.setdefault('pilot', cap_json.get('pilot'))
            names = {sid, sid.split(':')[0]}
            if linked:
                names.add(linked)
            aliases = sorted(names | {n.rsplit('/', 1)[-1] for n in names if '/' in n})
            recs.append(dict(
                source_id=sid, flight=key, aliases=aliases,
                source={'geometry_png': 'geometry_png', 'capture_dataset': 'capture_dataset', 'run_video': 'run_video'}[kind],
                env=env, path=path, archive=r.get('archive') if kind == 'geometry_png' else None,
                telemetry_csv=csv, sidecar_json=sidecar,
                origin_sim=[float(x) for x in origin] if origin is not None else None,
                alignment=alignment, controller=controller, oracle_route=oracle,
                human=_is_human(kind, vid if kind == 'run_video' else r)
                or str((cap_json or {}).get('pilot') or '').lower() == 'human', secondary=k > 0,
                linked_video=linked, fps=r.get('fps'), resolution=r.get('resolution'),
                stop_reason=(vid or r).get('stop_reason')))
    return recs, skipped


# ----------------------------------------------------------------------------- events and clean windows

def _speed_before(tel, k):
    return float(np.linalg.norm(tel.vel[max(k - 10, 0)]))


def detect_contacts(tel: Telemetry):
    """haltere.liftoff.flightlog.collisions per attempt, exactly as the survey applied it."""
    from ..liftoff import flightlog
    out = []
    P, V, Q, ts = tel.pos, tel.vel, tel.quat, tel.ts
    for a, b in zip(tel.attempt_starts, tel.attempt_ends):
        if b - a <= 200 * (1 if tel.kind == 'run_csv' else 0.35):
            continue
        ix = np.arange(a, b)
        tt = ts[ix] - ts[a]
        ph = tel.phase[ix] if tel.phase is not None else tt
        air = np.maximum.accumulate(np.abs(P[ix, 2] - P[a, 2]) > 0.5) & (ph > 3.0)
        for c in flightlog.collisions(P[ix], V[ix], Q[ix], tt, air):
            row = a + int(np.argmin(np.abs(tt - c['t'])))
            out.append((row, c))
    return out


def run_events(rec, tel: Telemetry | None, root, lateral=None):
    """Store events of one run from its flight telemetry (terminal impact, contacts)."""
    if tel is None:
        return [], {}
    meta = _load_json(data_path(root, rec['sidecar_json'])) if rec.get('sidecar_json') else None
    meta = meta if isinstance(meta, dict) else {}
    events, notes = [], {}
    imp = meta.get('impact') if isinstance(meta.get('impact'), dict) else None
    t_imp = None
    if imp and imp.get('timestamp') is not None and tel.kind == 'run_csv':
        k = tel.row_at_ts(float(imp['timestamp']))
        if k is not None:
            t_imp = float(tel.wall[k])
            events.append(dict(kind='terminal_impact', t_wall=t_imp, t_phase=float(tel.sample([t_imp])['phase'][0]),
                               t_game=float(tel.ts[k]), drone_pos_w=[round(float(x), 3) for x in tel.pos[k]],
                               speed_mps=round(_speed_before(tel, k), 3), accel_mps2=imp.get('acceleration_mps2'),
                               unexplained_mps2=imp.get('unexplained_mps2'), source='sidecar'))
    for row, c in detect_contacts(tel):
        tw = float(tel.wall[row])
        if t_imp is not None and (abs(tw - t_imp) < 1.0 or tw > t_imp):
            continue                      # the terminal impact itself, or its aftermath
        events.append(dict(kind='contact', t_wall=tw, t_phase=float(tel.sample([tw])['phase'][0]),
                           t_game=float(tel.ts[row]), drone_pos_w=[round(float(x), 3) for x in tel.pos[row]],
                           speed_mps=round(float(c['speed_before']), 3), accel_mps2=round(float(c['peak']), 2),
                           unexplained_mps2=round(float(c['unexplained']), 2),
                           source='csv_contact' if tel.kind == 'run_csv' else 'capture_contact'))
    if lateral:
        name = rec.get('lateral_name')
        for c in lateral.get('contacts', []):
            if c['run'] != name:
                continue
            tw = tel.wall_at_phase(float(c['phase_s']))
            if any(abs(e['t_wall'] - tw) < 0.5 for e in events):
                continue
            k = int(np.argmin(np.abs(tel.wall - tw)))
            events.append(dict(kind='contact', t_wall=float(tw), t_phase=float(c['phase_s']), t_game=float(tel.ts[k]),
                               drone_pos_w=[round(float(x), 3) for x in tel.pos[k]], speed_mps=round(_speed_before(tel, k), 3),
                               accel_mps2=None, unexplained_mps2=None, source='lateral_manifest', note=c.get('obstacle')))
        for c in lateral.get('impacts', []):
            if c['run'] != name:
                continue
            match = [e for e in events if e['kind'] == 'terminal_impact']
            dt = min((abs(e['t_phase'] - c['impact_phase_s']) for e in match), default=None)
            notes.setdefault('lateral_impacts', []).append(dict(
                run=c['run'], source_id=rec['source_id'], impact_phase_s=c['impact_phase_s'],
                matched_dt_s=None if dt is None else round(float(dt), 3)))
    for e in events:
        e['clock'] = tel.clock
    events.sort(key=lambda e: e['t_wall'])
    return events, notes


def load_lateral(path):
    if not path:
        return None
    m = json.loads(Path(path).read_text(encoding='utf-8'))
    return dict(path=_posix(path), impacts=m.get('impacts', []), contacts=m.get('non_fatal_contacts', []),
                clean=m.get('clean_windows', []), near=m.get('near_passes', []))


def attach_lateral_names(records, lateral):
    """Map lateral-study run names (fast-stack short names) to planned records; report misses/ambiguities."""
    if not lateral:
        return {}
    names = sorted({x['run'] for k in ('impacts', 'contacts', 'clean', 'near') for x in lateral[k]})
    report = {}
    for name in names:
        hits = [r for r in records if r.get('linked_video') == LATERAL_RUN_PREFIX + name
                or r['source_id'] == LATERAL_RUN_PREFIX + name]
        if len(hits) == 1:
            hits[0]['lateral_name'] = name
            report[name] = hits[0]['source_id']
        else:
            report[name] = None if not hits else [h['source_id'] for h in hits]
    return report


def clean_windows_for(records, lateral):
    """Curated lateral-study clean windows mapped onto planned runs (phase clock of the flight CSV)."""
    by_name = {r['lateral_name']: r for r in records if r.get('lateral_name')}
    out, missing = [], []
    for w in (lateral or {}).get('clean', []):
        r = by_name.get(w['run'])
        if r is None:
            missing.append(dict(run=w['run'], origin='lateral'))
            continue
        out.append(dict(run_id=r['run_id'], source_id=r['source_id'], alias=w['run'], start_phase_s=float(w['start_s']),
                        end_phase_s=float(w['end_s']), criterion=w.get('criterion'),
                        origin=f'lateral manifest ({lateral["path"]})'))
    out.sort(key=lambda w: (w['run_id'], w['start_phase_s']))
    return out, missing


# ----------------------------------------------------------------------------- frame selection

def tti_for(t, events):
    """Seconds to the next event at or after t, and that event's store id."""
    tti = np.full(len(t), np.inf, np.float32)
    eid = np.full(len(t), -1, np.int32)
    for e in sorted(events, key=lambda e: e['t_wall'], reverse=True):
        m = t <= e['t_wall']
        tti[m] = (e['t_wall'] - t[m]).astype(np.float32)
        eid[m] = e['store_event_id']
    return tti, eid


def in_clean(phase, windows):
    m = np.zeros(len(phase), bool)
    for w in windows:
        m |= (phase >= w['start_phase_s']) & (phase <= w['end_phase_s'])
    return m


def drop_reasons(t, tel: Telemetry | None, events, *, repeat, menu, hud, finished=False):
    """Per-frame drop reason (object array of str or '') for capture times t (sorted)."""
    n = len(t)
    why = np.full(n, '', object)
    why[repeat] = 'repeat'
    if tel is not None:
        s = tel.sample(t)
        why[(why == '') & s['outside']] = 'outside_telemetry'
        why[(why == '') & s['gap']] = 'telemetry_gap'
        paused = tel.game_rate(t) < PAUSE_RATE
        why[(why == '') & (menu | paused)] = 'menu'
        att = s['attempt']
        launch = tel.launch_wall[np.clip(att, 0, len(tel.launch_wall) - 1)]
        why[(why == '') & (t < launch - 0.1)] = 'countdown'
        span_end = tel.wall[-1]
    else:
        why[(why == '') & menu] = 'menu'
        span_end = t[-1] if n else 0
    if finished and n:
        k = n - 1
        while k >= 0 and not hud[k]:
            k -= 1
        start = k + 1
        if start < n and t[-1] - t[start] >= FINISH_MIN_S and t[start] >= span_end - FINISH_TAIL_S:
            tail = why[start:]
            tail[tail == ''] = 'finish'
            why[start:] = tail
    for e in events:
        if e['kind'] == 'terminal_impact':
            why[(why == '') & (t > e['t_wall'])] = 'post_event'
        else:
            end = e['t_wall'] + POST_CONTACT_DROP_S
            if tel is not None:
                later = tel.reset_walls[tel.reset_walls > e['t_wall']]
                if len(later) and later[0] - e['t_wall'] <= CRASH_RESET_S:
                    end = max(end, later[0])
            why[(why == '') & (t > e['t_wall']) & (t <= end)] = 'post_event'
    return why


def grid_step(t_valid, stride):
    if len(t_valid) < 3:
        return 1, None
    dt = np.diff(t_valid)
    dt = dt[dt > 1e-4]
    if not len(dt):
        return 1, None
    rate = 1.0 / float(np.median(dt))
    return int(np.clip(math.floor(rate / MIN_KEPT_RATE_HZ), 1, stride)), rate


def select(why, t, phase, events, windows, stride):
    """Keep mask, DENSE_EXTRA mask, tti, event ids, IN_CLEAN/PRE_EVENT, grid step and rate."""
    valid = why == ''
    s, rate = grid_step(t[valid], stride)
    tti, eid = tti_for(t, events)
    clean = in_clean(phase, windows) if windows else np.zeros(len(t), bool)
    pre = tti <= PRE_EVENT_DENSE_S
    dense = valid & (pre | clean)
    grid = np.zeros(len(t), bool)
    grid[np.flatnonzero(valid)[::s]] = True
    keep = grid | dense
    return dict(keep=keep, extra=keep & ~grid, tti=tti, eid=eid, clean=clean, pre=pre, step=s, rate=rate,
                n_valid=int(valid.sum()), n_dense=int(dense.sum()))


def count_drops(why, keep):
    out = {}
    for r in np.unique(why):
        if r:
            out[str(r)] = int((why == r).sum())
    out['stride'] = int(((why == '') & ~keep).sum())
    return out


def why_codes(why):
    return np.array([DROP_CODE[str(w)] for w in why], np.uint8)


# ----------------------------------------------------------------------------- per-source runs

class RunContext:
    def __init__(self, rec, run_id, guard, events, windows, stride, root, parts_dir):
        self.rec, self.run_id, self.guard = rec, run_id, guard
        self.events, self.windows, self.stride = events, windows, stride
        self.root, self.parts_dir = Path(root), Path(parts_dir)
        self.env_code = ENV_CODE[rec['env']]
        self.flags = Flag.NONE
        if rec.get('oracle_route'):
            self.flags |= Flag.ORACLE_ROUTE
        if rec.get('human'):
            self.flags |= Flag.HUMAN
        if rec.get('secondary'):
            self.flags |= Flag.SECONDARY_SOURCE


def fill_rows(ctx, n, *, t, sample, pose_method, source, grade, pos=None, quat=None, vel=None, omega=None,
              pose_lag=None, align_offset=0.0, cue=None, sel=None, phase=None, has_csv=False, t_game=None):
    rows = empty_index(n)
    rows['env'] = ctx.env_code
    rows['source'] = int(source)
    rows['grade'] = int(grade)
    rows['pose_method'] = int(pose_method)
    rows['t_wall'] = t
    rows['t_phase'] = phase if phase is not None else np.nan
    rows['t_game'] = t_game if t_game is not None else (sample['ts'] if sample is not None else np.nan)
    rows['pose_lag_s'] = pose_lag if pose_lag is not None else 0.0
    rows['align_offset_s'] = align_offset
    rows['timing_delta_s'] = 0.0
    rows['pos'] = pos if pos is not None else sample['pos']
    rows['quat'] = _qnorm(np.asarray(quat if quat is not None else sample['quat'], np.float64))
    rows['vel'] = vel if vel is not None else sample['vel']
    rows['omega'] = omega if omega is not None else sample['omega']
    if cue is not None:
        rows['cue_uv'] = cue
        rows['cue_src'] = np.where(np.isfinite(cue).all(1), int(CueSource.LOGGED), int(CueSource.NONE))
    rows['tti_s'] = sel['tti']
    rows['event_id'] = sel['eid']
    flags = np.full(n, int(ctx.flags), np.uint16)
    flags |= np.where(sel['clean'], int(Flag.IN_CLEAN), 0).astype(np.uint16)
    flags |= np.where(sel['pre'], int(Flag.PRE_EVENT), 0).astype(np.uint16)
    flags |= np.where(sel['extra'], int(Flag.DENSE_EXTRA), 0).astype(np.uint16)
    wz = np.abs(rows['omega'][:, 2])
    flags |= np.where(np.isfinite(wz) & (wz > HIGH_YAW_RATE), int(Flag.HIGH_YAW_RATE), 0).astype(np.uint16)
    if has_csv:
        flags |= np.uint16(int(Flag.HAS_CSV))
    rows['flags'] = flags
    return rows


def _append(writer, rows, frames, ridx):
    f = np.stack(frames)
    r = rows[np.asarray(ridx)].copy()
    r['luma'] = luma(f)
    writer.append(f, r)


def _write_kept(writer, ctx, rows_all, keep_idx, load_frame, batch=64):
    """Load, resize and append kept frames in batches (with a chunk guard every CHUNK_FRAMES)."""
    since = 0
    for b0 in range(0, len(keep_idx), batch):
        ids = keep_idx[b0:b0 + batch]
        _append(writer, rows_all, [to_store(load_frame(int(i))) for i in ids], ids)
        since += len(ids)
        if since >= CHUNK_FRAMES:
            ctx.guard.before_chunk()
            since = 0


def _image_pass(ctx, paths, ts_logged=None):
    """Pass 1 over still images: repeat (grey diff or unchanged logged game time), pause menu, HUD."""
    n = len(paths)
    repeat, menu, hud = np.zeros(n, bool), np.zeros(n, bool), np.ones(n, bool)
    prev = None
    for i, p in enumerate(paths):
        if i and i % CHUNK_FRAMES == 0:
            ctx.guard.before_chunk()
        s = small320(read_rgb(p))
        g = grey160(s)
        rep = prev is not None and float(np.abs(g - prev).mean()) < REPEAT_GREY
        if ts_logged is not None and i and ts_logged[i] == ts_logged[i - 1]:
            rep = True
        repeat[i] = rep
        prev = g
        menu[i], hud[i] = pause_menu_320(s), hud_present_320(s)
    return repeat, _dilate(menu, MENU_DILATE), hud


def run_geometry(writer, ctx, tel):
    rec = ctx.rec
    if tel is None:
        raise RuntimeError('geometry run without telemetry CSV')
    recs = [json.loads(line) for line in open(data_path(ctx.root, rec['archive']), encoding='utf-8') if line.strip()]
    png_dir = data_path(ctx.root, rec['path'])
    with_image = [r for r in recs if r.get('image_file') and (png_dir / r['image_file']).exists()]
    items = [r for r in with_image if r.get('camera_position') is not None and r.get('camera_quaternion') is not None
             and r.get('capture_time') is not None]
    if tel.epoch_offset is None:
        raise RuntimeError('geometry run CSV has no frame_time column: cannot place the worker clock')
    C = tel.epoch_offset
    t = np.array([r['capture_time'] for r in items], np.float64) + C
    order = np.argsort(t, kind='stable')
    items = [items[i] for i in order]
    t = t[order]
    n = len(items)
    repeat, menu, hud = _image_pass(ctx, [png_dir / r['image_file'] for r in items])
    sample = tel.sample(t)
    finished = not any(e['kind'] == 'terminal_impact' for e in ctx.events)
    why = drop_reasons(t, tel, ctx.events, repeat=repeat, menu=menu, hud=hud, finished=finished)
    sel = select(why, t, sample['phase'], ctx.events, ctx.windows, ctx.stride)
    pos = np.array([r['camera_position'] for r in items], np.float64)
    quat = _qnorm(np.array([r['camera_quaternion'] for r in items], np.float64))
    lag = np.array([max(0.0, float(r.get('pose_extrapolation_s') or 0.0)) for r in items])
    rows = fill_rows(ctx, n, t=t, sample=sample, pose_method=PoseMethod.WORKER_INTERP, source=Source.GEOMETRY_PNG,
                     grade=Grade.EXACT, pos=pos, quat=quat, pose_lag=lag, cue=tel.cue_at(t), sel=sel,
                     phase=sample['phase'], has_csv=True)
    keep_idx = np.flatnonzero(sel['keep'])
    _write_kept(writer, ctx, rows, keep_idx, lambda i: read_rgb(png_dir / items[i]['image_file']))
    ok = sample['valid'] & (why == '')
    dp = np.linalg.norm(pos - sample['pos'], axis=1)
    dq = 2 * np.degrees(np.arccos(np.clip(np.abs(np.einsum('ij,ij->i', quat, sample['quat'])), 0, 1)))
    drops = count_drops(why, sel['keep'])
    drops['no_pose'] = len(with_image) - n
    extra = dict(grid_step=sel['step'], source_rate_hz=_r(sel['rate']), n_source_frames=len(with_image),
                 n_valid=sel['n_valid'], n_dense_candidates=sel['n_dense'], epoch_offset_s=C, clock=tel.clock,
                 check_worker_vs_csv_pos_m_median=_r(float(np.median(dp[ok]))) if ok.any() else None,
                 check_worker_vs_csv_pos_m_p95=_r(float(np.percentile(dp[ok], 95))) if ok.any() else None,
                 check_worker_vs_csv_deg_p95=_r(float(np.percentile(dq[ok], 95))) if ok.any() else None)
    return drops, extra


def run_capture(writer, ctx, tel_flight):
    import pandas as pd
    rec = ctx.rec
    ds = data_path(ctx.root, rec['path'])
    idx = pd.read_csv(ds / 'index.csv')
    exists = np.array([(ds / 'frames' / f).exists() for f in idx['file']])
    missing = int((~exists).sum())
    idx = idx[exists].reset_index(drop=True)
    t = idx['wall_time'].to_numpy(np.float64)
    order = np.argsort(t, kind='stable')
    idx = idx.iloc[order].reset_index(drop=True)
    t = t[order]
    n = len(idx)
    tel_raw = load_capture_telemetry(ds, rec.get('origin_sim')) if (ds / 'telemetry.csv').exists() else None
    logged_pos = idx[['px', 'py', 'pz']].to_numpy(np.float64)
    logged_q = _qnorm(idx[['qw', 'qx', 'qy', 'qz']].to_numpy(np.float64))
    ts_logged = idx['ts'].to_numpy(np.float64) if 'ts' in idx else None
    repeat, menu, hud = _image_pass(ctx, [ds / 'frames' / f for f in idx['file']], ts_logged)
    extra = dict(n_source_frames=n, missing_images=missing)
    lag = None
    if tel_raw is not None:
        tel = tel_raw
        sample = tel.sample(t)
        # self-check: the logged pose is the latest packet received before the grab start
        recv = idx['capture_end'].to_numpy(np.float64) - idx['telemetry_age_s'].to_numpy(np.float64)
        chk = tel.sample(recv)
        extra['check_logged_vs_telemetry_pos_m_median'] = _r(float(np.median(np.linalg.norm(chk['pos'] - logged_pos, axis=1))))
        pose_method, pos, quat, vel, omega = PoseMethod.TELEMETRY_INTERP, None, None, None, None
        has_csv = True
        phase = sample['phase'] if tel_flight is None else tel_flight.sample(t)['phase']
        t_game = None
    else:
        tel = tel_flight
        pose_method, pos, quat = PoseMethod.SOURCE_RAW, logged_pos, logged_q
        lag = np.full(n, np.nan)
        tt = ts_logged if ts_logged is not None else t
        if tel is not None:
            sample = tel.sample(t)
            vel, omega = sample['vel'], sample['omega']
            phase = sample['phase']
            has_csv = True
            if ts_logged is not None:
                lag = t - tel.wall_at_ts(ts_logged, t)
                fin = np.isfinite(lag)
                extra['pose_lag_s_median'] = _r(float(np.median(lag[fin]))) if fin.any() else None
        else:
            sample = None
            vel = _gradient(logged_pos, tt)
            omega = omega_from_quat_series(logged_q, tt)
            phase = np.full(n, np.nan)
            has_csv = False
        t_game = ts_logged
    if tel is not None:
        why = drop_reasons(t, tel, ctx.events, repeat=repeat, menu=menu, hud=hud,
                           finished=not any(e['kind'] == 'terminal_impact' for e in ctx.events))
    else:
        why = _drops_without_telemetry(t, logged_pos, ts_logged, repeat, menu)
    sel = select(why, t, phase, ctx.events, ctx.windows, ctx.stride)
    cue = tel_flight.cue_at(t) if tel_flight is not None else None
    rows = fill_rows(ctx, n, t=t, sample=sample, pose_method=pose_method, source=Source.CAPTURE_DATASET,
                     grade=Grade.CAPTURE, pos=pos, quat=quat, vel=vel, omega=omega, pose_lag=lag, cue=cue, sel=sel,
                     phase=phase, has_csv=has_csv, t_game=t_game)
    keep_idx = np.flatnonzero(sel['keep'])
    _write_kept(writer, ctx, rows, keep_idx, lambda i: read_rgb(ds / 'frames' / idx['file'][i]))
    extra.update(grid_step=sel['step'], source_rate_hz=_r(sel['rate']), n_valid=sel['n_valid'],
                 n_dense_candidates=sel['n_dense'], pose_method=PoseMethod(pose_method).name.lower(),
                 capture_telemetry=_posix(ds / 'telemetry.csv') if tel_raw is not None else None,
                 clock=tel.clock if tel is not None else 'index wall_time')
    return count_drops(why, sel['keep']), extra


def _gradient(pos, tt):
    n = len(pos)
    if n < 3:
        return np.zeros((n, 3))
    tt = np.asarray(tt, np.float64)
    ok = np.r_[True, np.diff(tt) > 1e-6]
    out = np.zeros((n, 3))
    if ok.sum() >= 3:
        g = np.stack([np.gradient(pos[ok, k], tt[ok]) for k in range(3)], 1)
        out[ok] = g
        out[~ok] = np.nan
    return out


def _drops_without_telemetry(t, pos, ts, repeat, menu):
    """Recorder sets without any CSV: repeats, menus, stalled game clock, resets and pre-launch frames."""
    n = len(t)
    why = np.full(n, '', object)
    why[repeat] = 'repeat'
    why[(why == '') & menu] = 'menu'
    starts = [0]
    if ts is not None and n > 2:
        dts = np.diff(ts)
        first_after = np.flatnonzero(dts < -0.5) + 1     # first frame of every new attempt (game reset)
        for k in first_after:     # the frames on both sides of a reset: velocity by differences is meaningless
            seg = why[k - 1:k + 1]
            seg[seg == ''] = 'outside_telemetry'
            why[k - 1:k + 1] = seg
        rate = np.r_[1.0, dts / np.maximum(np.diff(t), 1e-3)]
        why[(why == '') & (rate < PAUSE_RATE) & (rate > -0.5)] = 'menu'
        starts += first_after.tolist()
    for a, b in zip(starts, starts[1:] + [n]):          # every attempt: pre-launch frames are countdown
        moved = np.linalg.norm(pos[a:b] - pos[a], axis=1) > LAUNCH_DISP
        launch = a + (int(np.argmax(moved)) if moved.any() else b - a)
        seg = why[a:launch]
        seg[seg == ''] = 'countdown'
        why[a:launch] = seg
    return why


def _dilate(m, k):
    out = m.copy()
    for d in range(1, k + 1):
        out[d:] |= m[:-d]
        out[:-d] |= m[d:]
    return out


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


class VideoReader:
    """ffmpeg raw-video pipe of the gameplay crop (readinto on an unbuffered pipe)."""

    def __init__(self, path, width, height, scale=None, threads=2, pix_fmt='rgb24'):
        x0 = max(0, width - GAMEPLAY_W)
        w = min(width, GAMEPLAY_W)
        vf = f'crop={w}:{height}:{x0}:0'
        self.w, self.h = (w, height) if scale is None else scale
        if scale is not None:
            vf += f',scale={scale[0]}:{scale[1]}:flags=area'
        self.channels = 3 if pix_fmt == 'rgb24' else 1
        self.cmd = ['ffmpeg', '-v', 'error', '-threads', str(threads), '-i', str(path), '-fps_mode', 'passthrough',
                    '-filter_threads', '1', '-vf', vf, '-pix_fmt', pix_fmt, '-f', 'rawvideo', '-']
        self.fb = self.w * self.h * self.channels
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.buf = bytearray(self.fb)
        self.mv = memoryview(self.buf)

    def read(self):
        got = 0
        while got < self.fb:
            k = self.proc.stdout.readinto(self.mv[got:])
            if not k:
                return None
            got += k
        a = np.frombuffer(self.buf, np.uint8)
        return a.reshape(self.h, self.w, 3) if self.channels == 3 else a.reshape(self.h, self.w)

    def close(self):
        try:
            self.proc.stdout.close()
        except OSError:
            pass
        try:
            self.proc.kill()
        except OSError:
            pass
        self.proc.wait()


def video_props(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 18.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return float(fps), w, h


def video_new_frames(dg, fps):
    """New (non-repeated) frames and their times: t_video = last repeated copy / fps (the aligner's rule)."""
    n = len(dg)
    new = np.asarray(dg) >= REPEAT_GREY
    if n:
        new[0] = True
    idx = np.flatnonzero(new)
    last = np.r_[idx[1:] - 1, n - 1] if n else idx
    return idx, last, last / fps


def run_video(writer, ctx, tel):
    rec = ctx.rec
    path = data_path(ctx.root, rec['path'])
    fps, W, H = video_props(path)
    if H != GAMEPLAY_H:
        raise RuntimeError(f'{path}: unexpected height {H}')
    # pass 1: every frame at 320 x 180
    rd = VideoReader(path, W, H, scale=(320, 180))
    dg, menu, hud = [], [], []
    prev = None
    try:
        while True:
            s = rd.read()
            if s is None:
                break
            k = len(dg)
            if k and k % CHUNK_FRAMES == 0:
                ctx.guard.before_chunk()
            g = grey160(s)
            dg.append(99.0 if prev is None else float(np.abs(g - prev).mean()))
            prev = g
            menu.append(pause_menu_320(s))
            hud.append(hud_present_320(s))
    finally:
        rd.close()
    n = len(dg)
    idx, last, t_video = video_new_frames(dg, fps)
    off = float(rec['alignment']['offset_s'])
    t = tel.wall0 + t_video + off
    menu_all = _dilate(np.array(menu, bool), MENU_DILATE)
    m_new, h_new = menu_all[idx], np.array(hud, bool)[idx]
    sample = tel.sample(t)
    finished = not any(e['kind'] == 'terminal_impact' for e in ctx.events)
    why = drop_reasons(t, tel, ctx.events, repeat=np.zeros(len(idx), bool), menu=m_new, hud=h_new, finished=finished)
    sel = select(why, t, sample['phase'], ctx.events, ctx.windows, ctx.stride)
    rows = fill_rows(ctx, len(idx), t=t, sample=sample, pose_method=PoseMethod.TELEMETRY_INTERP, source=Source.RUN_VIDEO,
                     grade=Grade[rec['alignment']['grade'].upper()], align_offset=off, cue=tel.cue_at(t), sel=sel,
                     phase=sample['phase'], has_csv=True)
    keep_new = np.flatnonzero(sel['keep'])
    keep_frames = idx[keep_new]                       # decoded-frame index of each kept new frame
    np.savez_compressed(ctx.parts_dir / f'{run_stem(ctx.run_id)}.frames.npz', frame_idx=idx.astype(np.int32),
                        last_copy=last.astype(np.int32), t_video=t_video, t_wall=t, why=why_codes(why),
                        keep=sel['keep'], dg=np.asarray(dg, np.float32), fps=fps, wall0=tel.wall0, offset_s=off,
                        drop_reasons=np.array(DROP_REASONS))
    # pass 2: full-resolution gameplay crop, kept frames only
    ctx.guard.before_chunk()
    rd = VideoReader(path, W, H)
    batch_f, batch_r = [], []
    k = since = 0
    want = iter(zip(keep_frames.tolist(), keep_new.tolist()))
    nxt = next(want, None)
    try:
        while nxt is not None:
            fr = rd.read()
            if fr is None:
                raise RuntimeError(f'{path}: video ended at frame {k} before kept frame {nxt[0]}')
            if k == nxt[0]:
                batch_f.append(to_store(fr))
                batch_r.append(nxt[1])
                nxt = next(want, None)
                if len(batch_f) >= 64:
                    _append(writer, rows, batch_f, batch_r)
                    batch_f, batch_r = [], []
            k += 1
            since += 1
            if since >= CHUNK_FRAMES:
                ctx.guard.before_chunk()
                since = 0
        if batch_f:
            _append(writer, rows, batch_f, batch_r)
    finally:
        rd.close()
    drops = count_drops(why, sel['keep'])
    drops['repeat'] = int(n - len(idx))
    extra = dict(grid_step=sel['step'], source_rate_hz=_r(sel['rate']), n_source_frames=n, n_new_frames=int(len(idx)),
                 n_valid=sel['n_valid'], n_dense_candidates=sel['n_dense'], fps=fps, csv_wall0=tel.wall0,
                 video_size=[W, H], frame_time='csv.wall[0] + last repeated copy / fps + offset', clock=tel.clock,
                 frames_table=f'parts/{run_stem(ctx.run_id)}.frames.npz')
    return drops, extra


# ----------------------------------------------------------------------------- build

def telemetry_for(rec, root):
    """The flight telemetry that defines events and the phase clock of a planned run."""
    if rec.get('telemetry_csv'):
        clock = 'wall' if rec['source'] == 'run_video' else 'receipt'
        return load_run_csv(data_path(root, rec['telemetry_csv']), clock=clock)
    if rec['source'] == 'capture_dataset':
        return load_capture_telemetry(data_path(root, rec['path']), rec.get('origin_sim'))
    return None


def _prepare(records, lateral, root, log=_log):
    """Events and clean windows for every planned run (telemetry only; no frames)."""
    events, notes = [], {}
    names = attach_lateral_names(records, lateral)
    windows, missing = clean_windows_for(records, lateral)
    notes['lateral_names'] = names
    for r in records:
        try:
            tel = telemetry_for(r, root)
        except Exception as ex:   # noqa: BLE001 - recorded, the run then has no events
            notes.setdefault('telemetry_errors', []).append(dict(source_id=r['source_id'], error=repr(ex)[:300]))
            tel = None
        ev, nt = run_events(r, tel, root, lateral)
        for e in ev:
            e.update(store_event_id=len(events), run_id=r['run_id'], source_id=r['source_id'], flight=r['flight'],
                     env=r['env'])
            events.append(e)
        for k, v in nt.items():
            notes.setdefault(k, []).extend(v)
    log(f'prepared {len(events)} events and {len(windows)} clean windows ({len(missing)} unmatched windows)')
    return events, windows, missing, notes


def _events_json(events):
    return dict(schema='haltere.obstacles.store.events.v1',
                note='Store events: terminal impacts (runner sidecar), contacts (flightlog.collisions per attempt on the '
                     'flight CSV, or on a capture set\'s ~35 Hz raw UDP log) and curated lateral-study non-fatal contacts '
                     'the detector missed. t_wall is on the run\'s frame clock (see runs.json clock); drone_pos_w is '
                     'launch-relative simulator FLU like the index pos (add runs.json origin_sim for absolute). '
                     'capture_contact events come from ~35 Hz UDP logs and are less certain than csv_contact. '
                     'HINDSIGHT data: never a model input.',
                events=events)


def build(args):
    lock = thermal.require_flight_lock_path(args.flight_lock)
    thermal.limit_threads(2, cv2=True)
    guard = thermal.ChunkGuard(lock)
    cfg = load_folds_config(args.folds)
    inv = json.loads(Path(args.inventory).read_text(encoding='utf-8'))
    root = Path(getattr(args, 'data_root', None) or inv.get('root') or REPO_ROOT)
    if not (root / 'runs').is_dir():
        raise SystemExit(f'data root {root} has no runs/ directory; pass --data-root')
    lateral = load_lateral(args.lateral_manifest)
    parts = ['main'] + (['sealed'] if args.sealed_final else [])
    reports = [_build_part(args, part, cfg, inv, lateral, guard, root) for part in parts]
    return reports[0]


def _build_part(args, part, cfg, inv, lateral, guard, root):
    writer = StoreWriter(args.out, part=part, sealed_final=args.sealed_final)
    records, skipped = plan_records(inv, cfg, root, include_unreliable=args.include_unreliable,
                                    secondary=set(args.secondary or ()), only=set(args.runs) if args.runs else None,
                                    part=part)
    build_info = dict(schema=BUILD_SCHEMA, data_root=_posix(root), stride=args.stride,
                      min_kept_rate_hz=MIN_KEPT_RATE_HZ, pre_event_dense_s=PRE_EVENT_DENSE_S,
                      post_contact_drop_s=POST_CONTACT_DROP_S, repeat_grey=REPEAT_GREY,
                      include_unreliable=bool(args.include_unreliable), secondary=sorted(args.secondary or ()),
                      runs_filter=args.runs, inventory=_posix(args.inventory), inventory_sha256=_sha256(args.inventory),
                      folds=_posix(args.folds), folds_sha256=_sha256(args.folds),
                      lateral_manifest=_posix(args.lateral_manifest) if args.lateral_manifest else None,
                      lateral_manifest_sha256=_sha256(args.lateral_manifest) if args.lateral_manifest else None,
                      code_commit=_git_commit(), video=VIDEO_DECODE, drop_reasons=list(DROP_REASONS),
                      skipped_flights=skipped)
    planned = writer.write_plan(records, build_info)
    _log(f'[{part}] plan: {len(planned)} runs ({len(skipped)} flights without a usable source); data root {root}')
    side = Path(writer.root) / 'build_events.json'
    attach_lateral_names(planned, lateral)      # runs.json records carry lateral_name (also on resume)
    if side.exists():
        prep = json.loads(side.read_text(encoding='utf-8'))
    else:
        guard.before_chunk()
        events, windows, missing, notes = _prepare(planned, lateral, root)
        prep = dict(events=events, windows=windows, missing_windows=missing, notes=notes)
        _json_dump(side, prep)
    events, windows = prep['events'], prep['windows']
    writer.write_json('events.json', _events_json(events))
    writer.write_json('clean_windows.json', dict(schema='haltere.obstacles.store.clean_windows.v1',
                                                 note='Every valid frame inside these windows is stored (IN_CLEAN).',
                                                 windows=windows))
    failures = []
    t_start = time.monotonic()
    for r in planned:
        rid = r['run_id']
        if writer.done(rid):
            continue
        guard.before_chunk()
        t0 = time.monotonic()
        ctx = RunContext(r, rid, guard, [e for e in events if e['run_id'] == rid],
                         [w for w in windows if w['run_id'] == rid], args.stride, root, Path(writer.root) / 'parts')
        try:
            clock = 'wall' if r['source'] == 'run_video' else 'receipt'
            tel = load_run_csv(data_path(root, r['telemetry_csv']), clock=clock) if r.get('telemetry_csv') else None
            with writer.begin_run(rid) as w:
                if r['source'] == 'geometry_png':
                    dropped, extra = run_geometry(w, ctx, tel)
                elif r['source'] == 'capture_dataset':
                    dropped, extra = run_capture(w, ctx, tel)
                else:
                    if tel is None:
                        raise RuntimeError('run video without telemetry CSV')
                    dropped, extra = run_video(w, ctx, tel)
                if tel is not None:
                    extra['flight_csv'] = tel.notes
                extra.update(seconds=round(time.monotonic() - t0, 1), n_events=len(ctx.events),
                             n_clean_windows=len(ctx.windows))
                rec = w.close(dropped=dropped, extra=extra)
            _log(f'[{part}] run {rid:4d} {r["source"]:15s} {r["env"][:14]:14s} {r["source_id"][:58]:58s} '
                 f'kept {rec["n_frames"]:6d} step {extra.get("grid_step")} {extra["seconds"]:5.0f}s '
                 f'(total {time.monotonic() - t_start:.0f}s)')
        except Exception as ex:   # noqa: BLE001 - one bad source must not stop the build; reported below
            failures.append(dict(run_id=rid, source_id=r['source_id'], error=repr(ex)[:500]))
            _log(f'[{part}] run {rid} {r["source_id"]} FAILED: {ex!r}')
            traceback.print_exc()
    m = writer.finalize(allow_missing=bool(failures))
    report = build_report(writer.root, m, failures, prep)
    _json_dump(Path(writer.root) / 'build_report.json', report)
    _log(json.dumps(report['summary'], indent=1))
    return report


def build_report(root, manifest, failures, prep):
    root = Path(root)
    index = np.load(root / 'index.npy')
    runs = json.loads((root / 'runs.json').read_text(encoding='utf-8'))
    per_env = {}
    for code in np.unique(index['env']):
        m = index['env'] == code
        name = ENV_BY_CODE[int(code)].name
        fl = index['flags'][m]
        per_env[name] = dict(frames=int(m.sum()),
                             by_source={s.name.lower(): int((index['source'][m] == int(s)).sum()) for s in Source
                                        if (index['source'][m] == int(s)).any()},
                             by_grade={g.name.lower(): int((index['grade'][m] == int(g)).sum()) for g in Grade
                                       if (index['grade'][m] == int(g)).any()},
                             dense_extra=int(((fl & int(Flag.DENSE_EXTRA)) != 0).sum()),
                             pre_event=int(((fl & int(Flag.PRE_EVENT)) != 0).sum()),
                             in_clean=int(((fl & int(Flag.IN_CLEAN)) != 0).sum()),
                             human=int(((fl & int(Flag.HUMAN)) != 0).sum()),
                             oracle_route=int(((fl & int(Flag.ORACLE_ROUTE)) != 0).sum()),
                             runs=int(len(np.unique(index['run_id'][m]))),
                             gb=round(float(m.sum()) * np.prod(FRAME_SHAPE) / 1e9, 2))
    folds = {}
    for f in FOLDS:
        folds[f] = {}
        for env, v in per_env.items():
            side = env_side(f, env)
            folds[f][side] = folds[f].get(side, 0) + v['frames']
    events, windows = prep['events'], prep['windows']
    # completeness: every valid frame in a clean window / pre-event window is stored
    dense_bad = []
    for r in runs:
        if not r.get('n_frames'):
            continue
        stored = int(((index['run_id'] == r['run_id'])
                      & ((index['flags'] & int(Flag.IN_CLEAN | Flag.PRE_EVENT)) != 0)).sum())
        if stored != r.get('n_dense_candidates', 0):
            dense_bad.append(dict(run_id=r['run_id'], source_id=r['source_id'], candidates=r.get('n_dense_candidates'),
                                  stored=stored))
    by_grade = {g.name.lower(): int((index['grade'] == int(g)).sum()) for g in Grade}
    minus, pine = per_env.get('Minus Two', {}).get('frames', 0), per_env.get('Pine Valley', {}).get('frames', 0)
    lat = prep.get('notes', {}).get('lateral_impacts', [])
    names = prep.get('notes', {}).get('lateral_names', {})
    ev_runs = {e['run_id'] for e in events}
    ev_stored = {int(x) for x in np.unique(index['event_id'][index['event_id'] >= 0])}
    summary = dict(
        n_frames=int(len(index)), n_runs=len(runs), n_failed=len(failures),
        disk_gb=round(sum((root / r['frames_file']).stat().st_size for r in runs if r.get('frames_file')) / 1e9, 2),
        per_env={k: v['frames'] for k, v in per_env.items()}, folds=folds, by_grade=by_grade,
        by_source={s.name.lower(): int((index['source'] == int(s)).sum()) for s in Source},
        events=dict(total=len(events), terminal=sum(e['kind'] == 'terminal_impact' for e in events),
                    contacts=sum(e['kind'] == 'contact' for e in events),
                    by_source={s: sum(e['source'] == s for e in events) for s in sorted({e['source'] for e in events})},
                    runs_with_events=len(ev_runs), events_with_stored_pre_frames=len(ev_stored)),
        clean_windows=len(windows), missing_clean_windows=prep.get('missing_windows'),
        lateral_names_unresolved=sorted(k for k, v in names.items() if not isinstance(v, str)),
        lateral_impacts=len(lat), lateral_impacts_matched=sum(1 for x in lat if x['matched_dt_s'] is not None
                                                              and x['matched_dt_s'] <= 0.1),
        dense_incomplete_runs=dense_bad,
        pass_criteria=dict(frames_ge_100k=len(index) >= 100_000, minus_two_ge_8k=minus >= 8000,
                           pine_valley_ge_8k=pine >= 8000,
                           all_clean_and_pre_event_frames=not dense_bad and not failures
                           and not prep.get('missing_windows')))
    return dict(summary=summary, per_env=per_env, failures=failures, manifest_index_sha256=manifest.get('index_sha256'),
                lateral_impacts=lat, telemetry_errors=prep.get('notes', {}).get('telemetry_errors', []))


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ----------------------------------------------------------------------------- repose

def store_data_root(manifest) -> Path:
    root = (manifest.get('build') or {}).get('data_root')
    return Path(root) if root else REPO_ROOT


def repose_rows(rows, tel: Telemetry, delta, events, windows):
    """Rows of one run video re-posed at t_wall(original alignment) + delta (pure; used by repose and tests)."""
    rows = rows.copy()
    prev = rows['timing_delta_s'].astype(np.float64)
    t = rows['t_wall'] - prev + float(delta)            # relative to the original alignment
    s = tel.sample(t)
    rows['t_wall'] = t
    rows['pos'], rows['quat'], rows['vel'], rows['omega'] = s['pos'], s['quat'], s['vel'], s['omega']
    rows['t_phase'], rows['t_game'] = s['phase'], s['ts']
    rows['timing_delta_s'] = delta
    cue = tel.cue_at(t)
    rows['cue_uv'] = cue
    rows['cue_src'] = np.where(np.isfinite(cue).all(1), int(CueSource.LOGGED), int(CueSource.NONE))
    tti, eid = tti_for(t, events)
    rows['tti_s'], rows['event_id'] = tti, eid
    flags = rows['flags'].astype(np.uint32)
    flags &= ~np.uint32(int(Flag.PRE_EVENT | Flag.IN_CLEAN | Flag.HIGH_YAW_RATE))
    flags |= np.where(tti <= PRE_EVENT_DENSE_S, int(Flag.PRE_EVENT), 0).astype(np.uint32)
    flags |= np.where(in_clean(s['phase'], windows), int(Flag.IN_CLEAN), 0).astype(np.uint32)
    wz = np.abs(s['omega'][:, 2])
    flags |= np.where(wz > HIGH_YAW_RATE, int(Flag.HIGH_YAW_RATE), 0).astype(np.uint32)
    flags |= np.uint32(int(Flag.TIMING_REFINED))
    rows['flags'] = flags.astype(np.uint16)
    return rows


def repose_capture_rows(rows, index_tel: Telemetry, delta):
    """Older recorder capture-set rows: pose from the logged pose series at t_wall + delta (pure).

    The logged (shared-state) pose was read before the grab and the index wall_time written after it, so
    the image matches the logged series ``delta`` later. t_wall (the logged time) and vel/omega/t_phase are
    kept; pos/quat/t_game become the compensated pose, pose_method SOURCE_COMPENSATED, pose_lag_s = delta.
    """
    rows = rows.copy()
    s = index_tel.sample(rows['t_wall'].astype(np.float64) + float(delta))
    rows['pos'], rows['quat'], rows['t_game'] = s['pos'], s['quat'], s['ts']
    rows['pose_method'] = int(PoseMethod.SOURCE_COMPENSATED)
    rows['pose_lag_s'] = delta
    rows['timing_delta_s'] = delta
    rows['flags'] = (rows['flags'].astype(np.uint32) | np.uint32(int(Flag.TIMING_REFINED))).astype(np.uint16)
    return rows


POSE_GATE_PX = 2.0     # twice the K0a threshold, ~4x the exact-time floor (~0.45 px at 640)


def pose_checks(refined, gate_px=POSE_GATE_PX):
    """{run_id: pose_check} for every run scored by timing refine (videos, exact-time controls, pose-lag sets)."""
    out = {}
    for section in ('runs', 'controls', 'pose_lag'):
        for rid_s, r in (refined.get(section) or {}).items():
            res = r.get('residual_px_after')
            status = 'unscored' if res is None else ('failed' if res > gate_px else 'passed')
            out[int(rid_s)] = dict(section=section, residual_px=res, gate_px=gate_px, status=status,
                                   delta_s=r.get('delta_s'), accepted=bool(r.get('accepted')), reason=r.get('reason'))
    return out


def repose(args):
    """Apply timing/refined.json: shift t_wall by delta, re-interpolate pose fields from the run CSV
    (run videos, ``runs``); older recorder capture sets (``pose_lag``) get compensated logged poses."""
    lock = thermal.require_flight_lock_path(args.flight_lock)
    thermal.limit_threads(2)
    guard = thermal.ChunkGuard(lock)
    root = Path(args.store)
    timing = Path(args.timing) if args.timing else root / 'timing' / 'refined.json'
    refined = json.loads(timing.read_text(encoding='utf-8'))
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise SystemExit('store is not complete')
    data_root = store_data_root(manifest)
    index = np.load(root / 'index.npy')
    if refined.get('store_index_sha256') and refined['store_index_sha256'] != index_sha256(index):
        raise SystemExit('timing/refined.json was computed on a different index; rerun timing refine')
    runs = json.loads((root / 'runs.json').read_text(encoding='utf-8'))
    events = json.loads((root / 'events.json').read_text(encoding='utf-8'))['events']
    windows = json.loads((root / 'clean_windows.json').read_text(encoding='utf-8'))['windows']
    new = index.copy()
    applied = {}
    for rid_s, r in sorted(refined['runs'].items(), key=lambda kv: int(kv[0])):
        rid = int(rid_s)
        if not r.get('accepted'):
            continue
        rec = runs[rid]
        if rec['source'] != 'run_video' or not rec.get('telemetry_csv'):
            continue
        m = np.flatnonzero(new['run_id'] == rid)
        if not len(m):
            continue
        guard.before_chunk()
        tel = load_run_csv(data_path(data_root, rec['telemetry_csv']), clock='wall')
        delta = float(r['delta_s'])
        new[m] = repose_rows(new[m], tel, delta, [e for e in events if e['run_id'] == rid],
                             [w for w in windows if w['run_id'] == rid])
        rec.setdefault('alignment', {})['refine_delta_s'] = delta
        applied[rid] = delta
    compensated = {}
    for rid_s, r in sorted((refined.get('pose_lag') or {}).items(), key=lambda kv: int(kv[0])):
        rid = int(rid_s)
        rec = runs[rid]
        if not r.get('accepted') or rec['source'] != 'capture_dataset':
            continue
        m = np.flatnonzero(new['run_id'] == rid)
        if not len(m) or not (new['pose_method'][m] == int(PoseMethod.SOURCE_RAW)).all():
            continue
        guard.before_chunk()
        index_tel, _ = capture_index_telemetry(data_path(data_root, rec['path']))
        delta = float(r['delta_s'])
        new[m] = repose_capture_rows(new[m], index_tel, delta)
        rec.setdefault('alignment', {})['pose_lag_compensation_s'] = delta
        compensated[rid] = delta
    gate = float(getattr(args, 'pose_gate_px', None) or POSE_GATE_PX)
    checks = pose_checks(refined, gate)
    regraded = {}
    for rid, chk in checks.items():
        runs[rid]['pose_check'] = chk
        if chk['status'] == 'failed':
            m = new['run_id'] == rid
            new['grade'][m] = int(Grade.UNRELIABLE)
            runs[rid]['alignment']['grade_before_pose_gate'] = runs[rid]['alignment'].get('grade')
            runs[rid]['alignment']['grade'] = 'unreliable'
            regraded[rid] = int(m.sum())
    k = 1
    while (root / f'index.v{k}.npy').exists():
        k += 1
    os.replace(root / 'index.npy', root / f'index.v{k}.npy')
    np.save(root / 'index.npy', new)
    _json_dump(root / 'runs.json', runs)
    manifest.update(index_sha256=index_sha256(new), n_frames=int(len(new)),
                    timing=dict(refined=_posix(timing), applied_runs=len(applied),
                                compensated_capture_sets=len(compensated), previous_index=f'index.v{k}.npy',
                                pose_gate=dict(gate_px=gate, failed_runs=len(regraded),
                                               regraded_frames=int(sum(regraded.values())),
                                               rule='timing residual after the applied delta (px at 640) > gate: grade '
                                                    'UNRELIABLE (excluded by default, no geometry labels)'),
                                previous_index_sha256=index_sha256(index),
                                refined_sha256=_sha256(timing)))
    _json_dump(root / 'manifest.json', manifest)
    _log(f'repose: {len(applied)} run videos re-posed, {len(compensated)} capture sets pose-compensated, '
         f'{len(regraded)} runs ({sum(regraded.values())} frames) regraded UNRELIABLE by the {gate:g} px pose gate; '
         f'previous index kept as index.v{k}.npy; new index sha256 {manifest["index_sha256"][:16]}')
    return dict(applied=applied, compensated=compensated, regraded=regraded)
