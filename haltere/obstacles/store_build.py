"""Frame-store builder and timing re-pose (offline only).

``python -m haltere.obstacles.store build`` calls ``build(args)``; ``repose`` calls ``repose(args)``.

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
4. Per run: pass 1 looks at every frame at 320 x 180 (repeat test, pause menu, HUD presence,
   luminance); frames are dropped for these reasons (counted in runs.json ``dropped``):

   repeat             recorder repeat (grey mean |diff| < 0.1 at 160 x 90) or unchanged game clock
   outside_telemetry  capture time outside the telemetry span, no pose, or across a reset
   telemetry_gap      interpolation across a telemetry gap > 0.25 s
   menu               pause-menu detector (+-3 frames) or a stalled game clock (paused game)
   countdown          before the launch of each attempt (race-start countdown / LIFTOFF text,
                      the drone still on the ground; also covers post-reset countdowns)
   finish             trailing HUD-absent results screen of a run that ended without an impact
   post_event         after the terminal impact, within POST_CONTACT_DROP_S after a contact,
                      and from a contact to a reset that follows it within 10 s (crash)

   Of the remaining frames every s-th is kept (the stride grid) plus every frame with
   tti_s <= PRE_EVENT_DENSE_S and every clean-window frame (DENSE_EXTRA when off the grid).
   The grid step is s = clip(floor(r / 2.5 Hz), 1, stride) where r is the source's new-frame
   rate: 13-18 Hz run videos and manual captures keep every 3rd frame (the plan's stride), while
   slower sources are thinned less so no source falls below ~2.5 Hz (the ~5 Hz geometry PNGs are
   all kept, 7 Hz recorder sets keep every 2nd frame). Pass 2 resizes the kept frames to 448 x 252
   with cv2.INTER_AREA from the 1280 x 720 gameplay crop (x >= 648) or the 640 x 360 image.
5. Index rows follow store.py exactly. Poses: run videos TELEMETRY_INTERP from the 100 Hz CSV at
   t_wall = csv.wall[0] + t_video + offset (offset = alignment.used_offset_s, else
   offset_after_first_row_s; t_video = last repeated copy / fps). Geometry PNGs WORKER_INTERP
   (worker camera_position/quaternion at the grab start; monotonic capture_time converted to epoch
   with the CSV's wall - frame_time); vel/omega/t_phase/t_game from the flight CSV at t_wall.
   Capture datasets with a raw telemetry.csv (DatasetWriter): TELEMETRY_INTERP at the grab start
   from the UDP receive clock (Unity -> simulator FLU, launch-relative to capture.json origin_sim).
   Older recorder sets (no telemetry.csv, no logged age): SOURCE_RAW, the logged shared-state pose,
   with vel/omega from the flight CSV where one exists, else finite differences of the logged poses.
   cue_uv is LOGGED from the CSV cue columns: the pilot detection whose capture time is nearest to
   t_wall within 50 ms, NaN when absent (-1) or clamped at the image edge.
6. events.json: {store_event_id, run_id, source_id, flight, kind, t_wall, t_phase, t_game,
   drone_pos_w, speed_mps, accel_mps2, unexplained_mps2, source} with source sidecar (terminal
   impact), csv_contact (haltere.liftoff.flightlog.collisions per attempt, as the survey) or
   lateral_manifest (curated non-fatal contacts the detector missed). clean_windows.json: the
   lateral manifest windows plus the looming catalogue windows of runs the lateral manifest does not
   cover ({run_id, source_id, alias, start_phase_s, end_phase_s, criterion, origin}).
7. finalize(); writes build_report.json with counts per environment and fold and the pass criteria.

repose(args) applies timing/refined.json (per-run delta_s from timing.py): t_wall += delta, pose,
vel, omega, t_phase and t_game re-interpolated from the run CSV at the corrected time, tti_s and
the PRE_EVENT / IN_CLEAN flags recomputed, Flag.TIMING_REFINED and timing_delta_s set. The previous
index is kept as index.v<k>.npy, runs.json records refine_delta_s, manifest.json gets the new
index_sha256. Labels built on an older index become invalid (LabelSet checks the hash).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import contract, thermal
from .splits import ENV_CODE, FOLDS, SEALED_ENVS, env_side, load_folds_config
from .store import (DEFAULT_STRIDE, FRAME_SHAPE, INDEX_DTYPE, POST_CONTACT_DROP_S, PRE_EVENT_DENSE_S,
                    HIGH_YAW_RATE, REPO_ROOT, CueSource, Flag, Grade, PoseMethod, Source, StoreWriter, _json_dump,
                    empty_index, index_sha256, run_stem)

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
GAMEPLAY_W, GAMEPLAY_H = 1280, 720
VIDEO_ENCODE = 'raw rgb24 over a pipe (ffmpeg -fps_mode passthrough); pass 1 scale=320:180:flags=area'
_M_UNITY = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])   # sim = M @ unity (frames.py)


def _log(*a):
    print(*a, flush=True)


def _rel(path) -> str | None:
    if path is None:
        return None
    p = str(path).replace('\\', '/')
    root = str(REPO_ROOT).replace('\\', '/') + '/'
    return p[len(root):] if p.startswith(root) else p


def _abs(rel) -> Path | None:
    if rel is None:
        return None
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


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
    """A pose time series on the wall clock (epoch s): the sampling side of every pose method."""

    def __init__(self, wall, ts, pos, quat, vel, omega=None, phase=None, *, mono_offset=None, cue=None,
                 kind='run_csv', path=None):
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
        # hemisphere continuity for interpolation
        for i in range(1, len(q)):
            if q[i] @ q[i - 1] < 0:
                q[i] = -q[i]
        self.quat = q
        self.vel = np.asarray(vel, np.float64)
        self.omega_logged = omega is not None
        self.omega = np.asarray(omega, np.float64) if omega is not None else omega_from_quat_series(q, self.ts)
        self.phase = None if phase is None else np.asarray(phase, np.float64)
        self.mono_offset = mono_offset
        self.cue = cue
        self.kind, self.path = kind, path
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
        lerp = lambda arr: arr[i0] * (1 - a)[:, None] + arr[i1] * a[:, None]   # noqa: E731
        q = _qnorm(lerp(self.quat))
        out = dict(pos=lerp(self.pos), quat=q, vel=lerp(self.vel), omega=lerp(self.omega),
                   ts=self.ts[i0] * (1 - a) + self.ts[i1] * a,
                   attempt=np.where(a < 0.5, self.attempt[i0], self.attempt[i1]))
        if self.phase is not None:
            ph = self.phase[i0] * (1 - a) + self.phase[i1] * a
            # continue the control clock linearly outside the span (clean windows use it)
            ph = np.where(t > self.wall[-1], self.phase[-1] + (t - self.wall[-1]), ph)
            ph = np.where(t < self.wall[0], self.phase[0] - (self.wall[0] - t), ph)
        else:
            ph = t - self.wall[0]
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
        g = lambda x: np.interp(x, self.wall, self.ts)   # noqa: E731
        dt = hi - lo
        rate = np.where(dt > 0.05, (g(hi) - g(lo)) / np.maximum(dt, 1e-9), 1.0)
        # a reset inside the window makes the rate negative: not a pause
        return np.where(rate < -0.5, 1.0, rate)

    def wall_at_phase(self, phase):
        if self.phase is None:
            return self.wall[0] + phase
        k = int(np.argmin(np.abs(self.phase - phase)))
        return float(self.wall[k] + (phase - self.phase[k]))

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
        if self.cue is None or not len(self.cue['t']):
            return out
        ct = self.cue['t']
        j = np.clip(np.searchsorted(ct, t), 1, len(ct) - 1)
        j = np.where(np.abs(ct[j - 1] - t) < np.abs(ct[j] - t), j - 1, j)
        ok = (np.abs(ct[j] - t) <= CUE_MATCH_S) & self.cue['ok'][j]
        out[ok] = self.cue['uv'][j[ok]]
        return out


def load_run_csv(path) -> Telemetry:
    import pandas as pd
    path = _abs(path)
    head = open(path, encoding='utf-8', errors='replace').readline().strip().split(',')
    pos = ['x', 'y', 'z'] if 'x' in head else ['px', 'py', 'pz']
    om = (['omega_x', 'omega_y', 'omega_z'] if 'omega_x' in head else
          (['wx', 'wy', 'wz'] if 'wx' in head else None))
    want = ['wall', 'ts'] + pos + ['vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz'] + (om or [])
    extra = [c for c in ('phase', 'cue_u', 'cue_v', 'cue_edge', 'capture_time', 'frame_time') if c in head]
    d = pd.read_csv(path, usecols=want + extra, low_memory=False)
    core = d[want].apply(pd.to_numeric, errors='coerce').to_numpy(np.float64)
    ok = np.isfinite(core).all(1)
    d = d[ok].reset_index(drop=True)
    core = core[ok]
    wall, ts = core[:, 0], core[:, 1]
    P, V, Q = core[:, 2:5], core[:, 5:8], core[:, 8:12]
    W = core[:, 12:15] if om else None
    phase = pd.to_numeric(d['phase'], errors='coerce').to_numpy(np.float64) if 'phase' in d else None
    if phase is not None and not np.isfinite(phase).all():
        phase = np.where(np.isfinite(phase), phase, np.interp(np.arange(len(phase)), np.flatnonzero(np.isfinite(phase)),
                                                              phase[np.isfinite(phase)]) if np.isfinite(phase).any() else 0.0)
    mono = None
    if 'frame_time' in d:
        ft = pd.to_numeric(d['frame_time'], errors='coerce').to_numpy(np.float64)
        m = np.isfinite(ft)
        if m.sum() > 10:
            mono = float(np.median(wall[m] - ft[m]))
    cue = None
    if 'cue_u' in d and 'capture_time' in d and mono is not None:
        ct = pd.to_numeric(d['capture_time'], errors='coerce').to_numpy(np.float64)
        cu = pd.to_numeric(d['cue_u'], errors='coerce').to_numpy(np.float64)
        cv = pd.to_numeric(d['cue_v'], errors='coerce').to_numpy(np.float64)
        edge = d['cue_edge'].astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy() if 'cue_edge' in d else \
            np.zeros(len(d), bool)
        m = np.isfinite(ct)
        ct, cu, cv, edge = ct[m], cu[m], cv[m], edge[m]
        first = np.r_[True, np.diff(ct) != 0]
        ct, cu, cv, edge = ct[first], cu[first], cv[first], edge[first]
        order = np.argsort(ct, kind='stable')
        ct, cu, cv, edge = ct[order], cu[order], cv[order], edge[order]
        okc = np.isfinite(cu) & np.isfinite(cv) & (cu >= 0) & (cu <= 1) & (cv >= 0) & (cv <= 1) & ~edge
        cue = dict(t=ct + mono, uv=np.stack([cu, cv], 1), ok=okc)
    return Telemetry(wall, ts, P, Q, V, W, phase, mono_offset=mono, cue=cue, kind='run_csv', path=_rel(path))


def load_capture_telemetry(dsdir, origin_sim) -> Telemetry | None:
    """DatasetWriter raw UDP log (Unity frame, receive clock) -> launch-relative simulator FLU."""
    import pandas as pd
    p = Path(dsdir) / 'telemetry.csv'
    if not p.exists() or origin_sim is None:
        return None
    d = pd.read_csv(p, usecols=['recv_time', 'timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz'])
    a = d.to_numpy(np.float64)
    a = a[np.isfinite(a).all(1)]
    if len(a) < 20:
        return None
    wall, ts = a[:, 0], a[:, 1]
    pos = a[:, 2:5] @ _M_UNITY.T - np.asarray(origin_sim, np.float64)
    quat = unity_quats_to_sim(a[:, 5:9])
    vel = a[:, 9:12] @ _M_UNITY.T
    return Telemetry(wall, ts, pos, quat, vel, None, None, kind='capture_telemetry', path=_rel(p))


# ----------------------------------------------------------------------------- image tests (320 x 180 RGB)

def pause_menu_320(s):
    """haltere scratch decode_scan rule: white panel with a red selected button."""
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
    return np.clip(np.rint((f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114).mean((1, 2))), 0, 255).astype(np.uint8)


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
    except Exception:
        return None


def _sidecar(rel):
    if not rel:
        return {}
    m = _load_json(_abs(rel))
    return m if isinstance(m, dict) else {}


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


def plan_records(inv, cfg, *, include_unreliable=False, secondary=(), only=None, part='main'):
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
                if r.get('png_frames', 0) > 0 and r.get('png_dir'):
                    cands.append((0, kind, r, Grade.EXACT))
            elif kind == 'capture_dataset':
                cands.append((1, kind, r, Grade.CAPTURE))
            else:
                q = (r.get('alignment') or {}).get('quality')
                if not (r.get('files') or {}).get('csv'):
                    continue
                if q == 'good':
                    cands.append((2, kind, r, Grade.GOOD))
                elif q == 'fair':
                    cands.append((2, kind, r, Grade.FAIR))
                elif include_unreliable:
                    cands.append((3, kind, r, Grade.UNRELIABLE))
        cands.sort(key=lambda c: (c[0], int(c[3])))
        if not cands:
            skipped.append(dict(flight=key, env=env, reason='no usable source'))
            continue
        chosen = [cands[0]] + ([cands[1]] if key in secondary and len(cands) > 1 else [])
        for k, (_, kind, r, grade) in enumerate(chosen):
            sid = r['id']
            if only and sid not in only and key not in only:
                continue
            linked = r.get('same_flight_as') if kind != 'run_video' else sid
            vid = videos.get(linked) if linked else None
            files = (vid or {}).get('files') or {}
            sidecar = files.get('json')
            csv = files.get('csv')
            meta = _sidecar(sidecar)
            origin = meta.get('origin_sim')
            path = r.get('video') if kind == 'run_video' else (r.get('png_dir') if kind == 'geometry_png' else r.get('dataset'))
            archive = r.get('archive') if kind == 'geometry_png' else None
            cap_json = None
            if kind == 'capture_dataset':
                cap_json = _load_json(_abs(r['dataset']) / 'capture.json') or {}
                if origin is None:
                    origin = cap_json.get('origin_sim')
            al = r.get('alignment') or {}
            if kind == 'run_video':
                off = al.get('used_offset_s', al.get('offset_after_first_row_s'))
                alignment = dict(grade=grade.name.lower(), offset_s=off,
                                 basis='used_offset_s' if 'used_offset_s' in al else 'offset_after_first_row_s',
                                 method=al.get('method'), source=al.get('source'), refine_delta_s=0.0)
            else:
                alignment = dict(grade=grade.name.lower(), offset_s=0.0, basis='exact capture time' if kind == 'geometry_png'
                                 else 'capture wall_time', refine_delta_s=0.0)
            controller = dict(vid.get('controller') or {}) if vid else {}
            oracle = bool(controller.get('runtime_route_oracle')) or bool(r.get('oracle_route')) or \
                bool((cap_json or {}).get('oracle_route'))
            if kind == 'capture_dataset':
                controller.setdefault('capture_source', r.get('source'))
            base = sid.split(':')[0]
            aliases = sorted({sid, base, base.rsplit('/', 1)[-1] if '/' in base else base,
                              linked or sid, (linked or sid).rsplit('/', 1)[-1]})
            recs.append(dict(source_id=sid, flight=key, aliases=aliases, source={0: 'geometry_png', 1: 'capture_dataset',
                                                                                 2: 'run_video'}[
                {'geometry_png': 0, 'capture_dataset': 1, 'run_video': 2}[kind]],
                env=env, path=path, archive=archive, telemetry_csv=csv, sidecar_json=sidecar,
                origin_sim=[float(x) for x in origin] if origin is not None else None,
                alignment=alignment, controller=controller, oracle_route=oracle,
                human=_is_human(kind, vid if kind == 'run_video' else r), secondary=k > 0,
                linked_video=linked, fps=r.get('fps'), resolution=r.get('resolution')))
    return recs, skipped


# ----------------------------------------------------------------------------- events and clean windows

def _sim_speed(tel, k):
    return float(np.linalg.norm(tel.vel[max(k - 10, 0)]))


def run_events(rec, tel: Telemetry | None, lateral=None):
    """Store events of one run from its flight telemetry (terminal impact, contacts)."""
    from ..liftoff import flightlog
    if tel is None:
        return [], {}
    meta = _sidecar(rec.get('sidecar_json'))
    events, notes = [], {}
    imp = meta.get('impact') if isinstance(meta.get('impact'), dict) else None
    t_imp = None
    if imp and imp.get('timestamp') is not None:
        k = tel.row_at_ts(float(imp['timestamp']))
        if k is not None:
            t_imp = float(tel.wall[k])
            events.append(dict(kind='terminal_impact', t_wall=t_imp, t_phase=float(tel.sample([t_imp])['phase'][0]),
                               t_game=float(tel.ts[k]), drone_pos_w=[round(float(x), 3) for x in tel.pos[k]],
                               speed_mps=round(_sim_speed(tel, k), 3), accel_mps2=imp.get('acceleration_mps2'),
                               unexplained_mps2=imp.get('unexplained_mps2'), source='sidecar'))
    if tel.kind == 'run_csv':
        P, V, Q, ts = tel.pos, tel.vel, tel.quat, tel.ts
        for a, b in zip(tel.attempt_starts, tel.attempt_ends):
            if b - a <= 200:
                continue
            ix = np.arange(a, b)
            tt = ts[ix] - ts[a]
            ph = tel.phase[ix] if tel.phase is not None else tt
            air = np.maximum.accumulate(np.abs(P[ix, 2] - P[a, 2]) > 0.5) & (ph > 3.0)
            for c in flightlog.collisions(P[ix], V[ix], Q[ix], tt, air):
                row = a + int(np.argmin(np.abs(tt - c['t'])))
                tw = float(tel.wall[row])
                if t_imp is not None and abs(tw - t_imp) < 1.0:
                    continue                      # the terminal impact itself
                if t_imp is not None and tw > t_imp:
                    continue
                events.append(dict(kind='contact', t_wall=tw, t_phase=float(tel.sample([tw])['phase'][0]),
                                   t_game=float(ts[row]), drone_pos_w=[round(float(x), 3) for x in P[row]],
                                   speed_mps=round(float(c['speed_before']), 3), accel_mps2=round(float(c['peak']), 2),
                                   unexplained_mps2=round(float(c['unexplained']), 2), source='csv_contact'))
    if lateral:
        for c in lateral.get('contacts', []):
            if not set(rec['aliases']) & {c['run']}:
                continue
            tw = tel.wall_at_phase(float(c['phase_s']))
            if any(abs(e['t_wall'] - tw) < 0.5 for e in events):
                continue
            k = int(np.argmin(np.abs(tel.wall - tw)))
            events.append(dict(kind='contact', t_wall=float(tw), t_phase=float(c['phase_s']), t_game=float(tel.ts[k]),
                               drone_pos_w=[round(float(x), 3) for x in tel.pos[k]], speed_mps=round(_sim_speed(tel, k), 3),
                               accel_mps2=None, unexplained_mps2=None, source='lateral_manifest',
                               note=c.get('obstacle')))
        for c in lateral.get('impacts', []):
            if not set(rec['aliases']) & {c['run']}:
                continue
            match = [e for e in events if e['kind'] == 'terminal_impact']
            notes.setdefault('lateral_impacts', []).append(dict(
                run=c['run'], impact_phase_s=c['impact_phase_s'],
                matched_dt_s=round(min((abs(e['t_phase'] - c['impact_phase_s']) for e in match), default=np.inf), 3)))
    events.sort(key=lambda e: e['t_wall'])
    return events, notes


def load_lateral(path):
    if not path:
        return None
    m = json.loads(Path(path).read_text(encoding='utf-8'))
    return dict(path=str(path).replace('\\', '/'), impacts=m.get('impacts', []), contacts=m.get('non_fatal_contacts', []),
                clean=m.get('clean_windows', []), near=m.get('near_passes', []))


def clean_windows_for(records, lateral, catalog_path):
    """Curated clean windows mapped onto planned runs (phase clock of the flight CSV)."""
    by_alias = {}
    for r in records:
        for a in r['aliases']:
            by_alias.setdefault(a, r)
    out, missing = [], []
    covered = set()
    if lateral:
        for w in lateral['clean']:
            r = by_alias.get(w['run']) or by_alias.get('fast-stack-20260923/' + w['run'])
            if r is None:
                missing.append(dict(run=w['run'], origin='lateral'))
                continue
            covered.add(r['run_id'])
            out.append(dict(run_id=r['run_id'], source_id=r['source_id'], alias=w['run'], start_phase_s=float(w['start_s']),
                            end_phase_s=float(w['end_s']), criterion=w.get('criterion'),
                            origin=f'lateral manifest ({lateral["path"]})'))
    if catalog_path and Path(catalog_path).exists():
        cat = json.loads(Path(catalog_path).read_text(encoding='utf-8'))
        for w in cat.get('clean', []):
            vid = w['video'].replace('\\', '/').split('/runs/', 1)[-1][:-4]
            r = by_alias.get(vid)
            if r is None:
                missing.append(dict(run=vid, origin='looming catalogue'))
                continue
            if r['run_id'] in covered:
                continue
            a, b = w['clean_phase']
            out.append(dict(run_id=r['run_id'], source_id=r['source_id'], alias=vid, start_phase_s=float(a),
                            end_phase_s=float(b),
                            criterion=f'looming catalogue clean window ({w.get("kind")}: {w.get("result")}); CSV contacts '
                                      f'{w.get("csv_contacts_phase")}',
                            origin=f'looming catalogue ({str(catalog_path).replace(chr(92), "/")})'))
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


def drop_reasons(t, tel: Telemetry | None, events, *, repeat, menu, hud, pos_series=None, finished=False):
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
        # countdown: before the launch of the attempt the frame belongs to
        att = s['attempt']
        launch = tel.launch_wall[np.clip(att, 0, len(tel.launch_wall) - 1)]
        why[(why == '') & (t < launch - 0.1)] = 'countdown'
        span_end = tel.wall[-1]
    else:
        why[(why == '') & menu] = 'menu'
        span_end = t[-1] if n else 0
    if finished and n:
        valid_hud = hud.copy()
        k = n - 1
        while k >= 0 and not valid_hud[k]:
            k -= 1
        start = k + 1
        if start < n and t[-1] - t[start] >= FINISH_MIN_S and t[start] >= span_end - FINISH_TAIL_S:
            why[start:][why[start:] == ''] = 'finish'
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
    clean = in_clean(phase, windows)
    dense = valid & ((tti <= PRE_EVENT_DENSE_S) | clean)
    grid = np.zeros(len(t), bool)
    vi = np.flatnonzero(valid)
    grid[vi[::s]] = True
    keep = grid | dense
    extra = keep & ~grid
    return dict(keep=keep, extra=extra, tti=tti, eid=eid, clean=clean, pre=valid & (tti <= PRE_EVENT_DENSE_S),
                step=s, rate=rate, n_valid=int(valid.sum()), n_dense=int(dense.sum()))


def count_drops(why, keep):
    out = {}
    for r in np.unique(why):
        if r:
            out[str(r)] = int((why == r).sum())
    out['stride'] = int(((why == '') & ~keep).sum())
    return out


# ----------------------------------------------------------------------------- per-source runs

class RunContext:
    def __init__(self, rec, run_id, guard, events, windows, stride):
        self.rec, self.run_id, self.guard = rec, run_id, guard
        self.events, self.windows, self.stride = events, windows, stride
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
    rows['quat'] = quat if quat is not None else sample['quat']
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


def _write_kept(writer, ctx, rows_all, keep_idx, load_frame, batch=64):
    """Load, resize and append kept frames in batches (with a chunk guard every CHUNK_FRAMES)."""
    since = 0
    for b0 in range(0, len(keep_idx), batch):
        ids = keep_idx[b0:b0 + batch]
        frames = np.stack([to_store(load_frame(int(i))) for i in ids])
        rows = rows_all[ids].copy()
        rows['luma'] = luma(frames)
        writer.append(frames, rows)
        since += len(ids)
        if since >= CHUNK_FRAMES:
            ctx.guard.before_chunk()
            since = 0


def run_geometry(writer, ctx, tel):
    rec = ctx.rec
    recs = [json.loads(line) for line in open(_abs(rec['archive']), encoding='utf-8') if line.strip()]
    png_dir = _abs(rec['path'])
    items = [r for r in recs if r.get('image_file') and r.get('camera_position') is not None
             and r.get('camera_quaternion') is not None and r.get('capture_time') is not None
             and (png_dir / r['image_file']).exists()]
    if tel is None or tel.mono_offset is None:
        # fall back to the game clock: publish time (monotonic) <-> game time -> CSV wall
        if tel is None:
            raise RuntimeError('geometry run without telemetry CSV')
        offs = [float(np.interp(r['game_time'], tel.ts, tel.wall)) - r['requested_goal_time'] for r in recs
                if r.get('game_time') is not None and r.get('requested_goal_time') is not None]
        mono = float(np.median(offs))
    else:
        mono = tel.mono_offset
    t = np.array([r['capture_time'] for r in items], np.float64) + mono
    order = np.argsort(t, kind='stable')
    items = [items[i] for i in order]
    t = t[order]
    n = len(items)
    # pass 1: images at 320 x 180
    repeat, menu, hud = np.zeros(n, bool), np.zeros(n, bool), np.ones(n, bool)
    prev = None
    for i, r in enumerate(items):
        if i and i % CHUNK_FRAMES == 0:
            ctx.guard.before_chunk()
        s = small320(read_rgb(png_dir / r['image_file']))
        g = grey160(s)
        repeat[i] = prev is not None and float(np.abs(g - prev).mean()) < REPEAT_GREY
        prev = g
        menu[i], hud[i] = pause_menu_320(s), hud_present_320(s)
    menu = _dilate(menu, MENU_DILATE)
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
    dp = np.linalg.norm(pos - sample['pos'], axis=1)
    ok = sample['valid']
    extra = dict(grid_step=sel['step'], source_rate_hz=_r(sel['rate']), n_source_frames=n, n_valid=sel['n_valid'],
                 n_dense_candidates=sel['n_dense'], mono_to_epoch_s=mono,
                 worker_vs_csv_pos_m_median=_r(float(np.median(dp[ok]))) if ok.any() else None)
    return count_drops(why, sel['keep']), extra


def run_capture(writer, ctx, tel_flight):
    import pandas as pd
    rec = ctx.rec
    ds = _abs(rec['path'])
    idx = pd.read_csv(ds / 'index.csv')
    idx = idx[[(ds / 'frames' / f).exists() for f in idx['file']]].reset_index(drop=True)
    t = idx['wall_time'].to_numpy(np.float64)
    order = np.argsort(t, kind='stable')
    idx = idx.iloc[order].reset_index(drop=True)
    t = t[order]
    n = len(idx)
    tel_raw = load_capture_telemetry(ds, rec.get('origin_sim')) if (ds / 'telemetry.csv').exists() else None
    logged_pos = idx[['px', 'py', 'pz']].to_numpy(np.float64)
    logged_q = _qnorm(idx[['qw', 'qx', 'qy', 'qz']].to_numpy(np.float64))
    ts_logged = idx['ts'].to_numpy(np.float64) if 'ts' in idx else None
    # pass 1
    repeat, menu, hud = np.zeros(n, bool), np.zeros(n, bool), np.ones(n, bool)
    prev = None
    for i in range(n):
        if i and i % CHUNK_FRAMES == 0:
            ctx.guard.before_chunk()
        s = small320(read_rgb(ds / 'frames' / idx['file'][i]))
        g = grey160(s)
        rep = prev is not None and float(np.abs(g - prev).mean()) < REPEAT_GREY
        if ts_logged is not None and i and ts_logged[i] == ts_logged[i - 1]:
            rep = True
        repeat[i] = rep
        prev = g
        menu[i], hud[i] = pause_menu_320(s), hud_present_320(s)
    menu = _dilate(menu, MENU_DILATE)
    extra = dict(n_source_frames=n)
    if tel_raw is not None:
        tel = tel_raw
        sample = tel.sample(t)
        # self-check: the logged pose is the latest packet received before the grab start
        recv = idx['capture_end'].to_numpy(np.float64) - idx['telemetry_age_s'].to_numpy(np.float64)
        chk = tel.sample(recv)
        extra['logged_vs_telemetry_pos_m_median'] = _r(float(np.median(np.linalg.norm(chk['pos'] - logged_pos, axis=1))))
        pose_method, pos, quat, lag = PoseMethod.TELEMETRY_INTERP, None, None, None
        vel = omega = None
        has_csv = True
        events_tel = tel_flight or tel
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
        else:
            sample = None
            vel = np.stack([np.gradient(logged_pos[:, k], tt) for k in range(3)], 1) if n > 2 else np.zeros((n, 3))
            omega = omega_from_quat_series(logged_q, tt)
            phase = np.full(n, np.nan)
            has_csv = False
        events_tel = tel
        t_game = ts_logged
    if tel is not None:
        why = drop_reasons(t, tel, ctx.events, repeat=repeat, menu=menu, hud=hud,
                           finished=not any(e['kind'] == 'terminal_impact' for e in ctx.events))
    else:
        why = _drops_without_telemetry(t, logged_pos, ts_logged, repeat, menu)
    sel = select(why, t, phase if phase is not None else np.full(n, np.nan), ctx.events, ctx.windows, ctx.stride)
    cue = events_tel.cue_at(t) if events_tel is not None else None
    rows = fill_rows(ctx, n, t=t, sample=sample, pose_method=pose_method, source=Source.CAPTURE_DATASET,
                     grade=Grade.CAPTURE, pos=pos, quat=quat, vel=vel, omega=omega, pose_lag=lag, cue=cue, sel=sel,
                     phase=phase, has_csv=has_csv, t_game=t_game)
    keep_idx = np.flatnonzero(sel['keep'])
    _write_kept(writer, ctx, rows, keep_idx, lambda i: read_rgb(ds / 'frames' / idx['file'][i]))
    extra.update(grid_step=sel['step'], source_rate_hz=_r(sel['rate']), n_valid=sel['n_valid'],
                 n_dense_candidates=sel['n_dense'], pose_method=PoseMethod(pose_method).name.lower(),
                 capture_telemetry=_rel(ds / 'telemetry.csv') if tel_raw is not None else None)
    return count_drops(why, sel['keep']), extra


def _drops_without_telemetry(t, pos, ts, repeat, menu):
    """Recorder sets without any CSV: repeats, menus, stalled game clock and pre-launch frames."""
    n = len(t)
    why = np.full(n, '', object)
    why[repeat] = 'repeat'
    why[(why == '') & menu] = 'menu'
    if ts is not None and n > 2:
        dts = np.gradient(ts, t)
        why[(why == '') & (dts < PAUSE_RATE)] = 'menu'
    moved = np.linalg.norm(pos - pos[0], axis=1) > LAUNCH_DISP
    first = int(np.argmax(moved)) if moved.any() else n
    why[:first][why[:first] == ''] = 'countdown'
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
    """ffmpeg raw-video pipe of the gameplay crop (readinto on an unbuffered pipe: 500+ fps)."""

    def __init__(self, path, width, height, scale=None, threads=2):
        x0 = max(0, width - GAMEPLAY_W)
        w = min(width, GAMEPLAY_W)
        vf = f'crop={w}:{height}:{x0}:0'
        self.w, self.h = (w, height) if scale is None else scale
        if scale is not None:
            vf += f',scale={scale[0]}:{scale[1]}:flags=area'
        self.cmd = ['ffmpeg', '-v', 'error', '-threads', str(threads), '-i', str(path), '-fps_mode', 'passthrough',
                    '-filter_threads', '1', '-vf', vf, '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
        self.fb = self.w * self.h * 3
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.buf = bytearray(self.fb)
        self.mv = memoryview(self.buf)

    def read(self):
        got = 0
        while got < self.fb:
            k = self.proc.stdout.readinto(self.mv[got:])
            if not k:
                return None
            got += k
        return np.frombuffer(self.buf, np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass
        self.proc.wait()


def video_props(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 18.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return float(fps), w, h


def run_video(writer, ctx, tel):
    rec = ctx.rec
    path = _abs(rec['path'])
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
    dg = np.array(dg)
    new = dg >= REPEAT_GREY
    new[0] = True
    idx = np.flatnonzero(new)
    last = np.r_[idx[1:] - 1, n - 1]
    off = float(rec['alignment']['offset_s'])
    t_video = last / fps
    t = tel.wall[0] + t_video + off
    menu_all = _dilate(np.array(menu), MENU_DILATE)
    m_new, h_new = menu_all[idx], np.array(hud)[idx]
    sample = tel.sample(t)
    finished = not any(e['kind'] == 'terminal_impact' for e in ctx.events)
    why = drop_reasons(t, tel, ctx.events, repeat=np.zeros(len(idx), bool), menu=m_new, hud=h_new, finished=finished)
    sel = select(why, t, sample['phase'], ctx.events, ctx.windows, ctx.stride)
    rows = fill_rows(ctx, len(idx), t=t, sample=sample, pose_method=PoseMethod.TELEMETRY_INTERP, source=Source.RUN_VIDEO,
                     grade=Grade[rec['alignment']['grade'].upper()], align_offset=off, cue=tel.cue_at(t), sel=sel,
                     phase=sample['phase'], has_csv=True)
    keep_new = np.flatnonzero(sel['keep'])
    keep_frames = idx[keep_new]                       # decoded-frame index of each kept new frame
    # pass 2: full-resolution gameplay crop, kept frames only
    ctx.guard.before_chunk()
    rd = VideoReader(path, W, H)
    batch_f, batch_r = [], []
    k = 0
    since = 0
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
                    _flush(writer, rows, batch_f, batch_r)
                    batch_f, batch_r = [], []
            k += 1
            since += 1
            if since >= CHUNK_FRAMES:
                ctx.guard.before_chunk()
                since = 0
        if batch_f:
            _flush(writer, rows, batch_f, batch_r)
    finally:
        rd.close()
    drops = count_drops(why, sel['keep'])
    drops['repeat'] = int(n - len(idx))
    extra = dict(grid_step=sel['step'], source_rate_hz=_r(sel['rate']), n_source_frames=n, n_new_frames=int(len(idx)),
                 n_valid=sel['n_valid'], n_dense_candidates=sel['n_dense'], fps=fps, csv_wall0=float(tel.wall[0]),
                 video_size=[W, H], frame_time='last repeated copy / fps')
    return drops, extra


def _flush(writer, rows, frames, ridx):
    f = np.stack(frames)
    r = rows[np.asarray(ridx)].copy()
    r['luma'] = luma(f)
    writer.append(f, r)


# ----------------------------------------------------------------------------- build

def _telemetry_for(rec):
    if rec.get('telemetry_csv'):
        return load_run_csv(rec['telemetry_csv'])
    if rec['source'] == 'capture_dataset':
        return load_capture_telemetry(_abs(rec['path']), rec.get('origin_sim'))
    return None


def _prepare(records, lateral, catalog, stride, log=_log):
    """Events and clean windows for every planned run (telemetry only; no frames)."""
    events, notes = [], {}
    windows, missing = clean_windows_for(records, lateral, catalog)
    for r in records:
        try:
            tel = _telemetry_for(r)
        except Exception as ex:   # noqa: BLE001
            notes.setdefault('telemetry_errors', []).append(dict(source_id=r['source_id'], error=repr(ex)[:300]))
            tel = None
        ev, nt = run_events(r, tel, lateral)
        for e in ev:
            e.update(store_event_id=len(events), run_id=r['run_id'], source_id=r['source_id'], flight=r['flight'],
                     env=r['env'])
            events.append(e)
        for k, v in nt.items():
            notes.setdefault(k, []).extend(v)
    return events, windows, missing, notes


def _events_json(events):
    return dict(schema='haltere.obstacles.store.events.v1',
                note='store events: terminal impacts (runner sidecar) and contacts (flightlog.collisions per attempt; '
                     'lateral manifest non-fatal contacts the detector missed). HINDSIGHT data: never a model input.',
                events=events)


def build(args):
    lock = thermal.require_flight_lock_path(args.flight_lock)
    thermal.limit_threads(2, cv2=True)
    os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'threads;2')
    guard = thermal.ChunkGuard(lock)
    cfg = load_folds_config(args.folds)
    inv = json.loads(Path(args.inventory).read_text(encoding='utf-8'))
    lateral = load_lateral(args.lateral_manifest)
    catalog = getattr(args, 'looming_catalog', None)
    parts = ['main'] + (['sealed'] if args.sealed_final else [])
    for part in parts:
        _build_part(args, part, cfg, inv, lateral, catalog, guard)


def _build_part(args, part, cfg, inv, lateral, catalog, guard):
    writer = StoreWriter(args.out, part=part, sealed_final=args.sealed_final)
    records, skipped = plan_records(inv, cfg, include_unreliable=args.include_unreliable,
                                    secondary=set(args.secondary or ()), only=set(args.runs) if args.runs else None,
                                    part=part)
    build_info = dict(schema=BUILD_SCHEMA, stride=args.stride, min_kept_rate_hz=MIN_KEPT_RATE_HZ,
                      pre_event_dense_s=PRE_EVENT_DENSE_S, post_contact_drop_s=POST_CONTACT_DROP_S,
                      repeat_grey=REPEAT_GREY, include_unreliable=bool(args.include_unreliable),
                      secondary=sorted(args.secondary or ()), runs_filter=args.runs,
                      inventory=str(args.inventory).replace('\\', '/'), inventory_sha256=_sha256(args.inventory),
                      folds=_rel(args.folds), folds_sha256=_sha256(args.folds),
                      lateral_manifest=str(args.lateral_manifest).replace('\\', '/') if args.lateral_manifest else None,
                      looming_catalog=str(catalog).replace('\\', '/') if catalog else None,
                      code_commit=_git_commit(), video=VIDEO_ENCODE, skipped_flights=skipped)
    planned = writer.write_plan(records, build_info)
    _log(f'[{part}] plan: {len(planned)} runs ({len(skipped)} flights without a usable source)')
    side = Path(writer.root) / 'build_events.json'
    if side.exists():
        prep = json.loads(side.read_text(encoding='utf-8'))
        events, windows = prep['events'], prep['windows']
    else:
        events, windows, missing, notes = _prepare(planned, lateral, catalog, args.stride)
        prep = dict(events=events, windows=windows, missing_windows=missing, notes=notes)
        _json_dump(side, prep)
    writer.write_json('events.json', _events_json(events))
    writer.write_json('clean_windows.json', dict(schema='haltere.obstacles.store.clean_windows.v1', windows=windows))
    failures = []
    t_start = time.monotonic()
    for r in planned:
        rid = r['run_id']
        if writer.done(rid):
            continue
        guard.before_chunk()
        t0 = time.monotonic()
        ctx = RunContext(r, rid, guard, [e for e in events if e['run_id'] == rid],
                         [w for w in windows if w['run_id'] == rid], args.stride)
        try:
            tel = load_run_csv(r['telemetry_csv']) if r.get('telemetry_csv') else None
            with writer.begin_run(rid) as w:
                if r['source'] == 'geometry_png':
                    dropped, extra = run_geometry(w, ctx, tel)
                elif r['source'] == 'capture_dataset':
                    dropped, extra = run_capture(w, ctx, tel)
                else:
                    if tel is None:
                        raise RuntimeError('run video without telemetry CSV')
                    dropped, extra = run_video(w, ctx, tel)
                extra.update(seconds=round(time.monotonic() - t0, 1), n_events=len(ctx.events),
                             n_clean_windows=len(ctx.windows))
                rec = w.close(dropped=dropped, extra=extra)
            _log(f'[{part}] run {rid:4d} {r["source"]:15s} {r["env"]:26s} {r["source_id"][:60]:60s} '
                 f'kept {rec["n_frames"]:6d} step {extra.get("grid_step")} {extra["seconds"]:.0f}s '
                 f'(total {time.monotonic() - t_start:.0f}s)')
        except Exception as ex:   # noqa: BLE001
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
    from .splits import ENV_BY_CODE
    per_env = {}
    for code in np.unique(index['env']):
        m = index['env'] == code
        name = ENV_BY_CODE[int(code)].name
        per_env[name] = dict(frames=int(m.sum()),
                             by_source={s.name.lower(): int((index['source'][m] == int(s)).sum()) for s in Source
                                        if (index['source'][m] == int(s)).any()},
                             dense_extra=int(((index['flags'][m] & int(Flag.DENSE_EXTRA)) != 0).sum()),
                             pre_event=int(((index['flags'][m] & int(Flag.PRE_EVENT)) != 0).sum()),
                             in_clean=int(((index['flags'][m] & int(Flag.IN_CLEAN)) != 0).sum()),
                             runs=int(len(np.unique(index['run_id'][m]))),
                             gb=round(float(m.sum()) * np.prod(FRAME_SHAPE) / 1e9, 2))
    folds = {}
    for f in FOLDS:
        folds[f] = {}
        for env, v in per_env.items():
            folds[f].setdefault(env_side(f, env), 0)
            folds[f][env_side(f, env)] += v['frames']
    events = prep['events']
    windows = prep['windows']
    # completeness: every valid frame in a clean window / pre-event window is stored
    dense_ok = []
    for r in runs:
        if not r.get('n_frames'):
            continue
        dense_ok.append(r.get('n_dense_candidates', 0) <= int(((index['run_id'] == r['run_id'])
                                                               & ((index['flags'] & int(Flag.IN_CLEAN | Flag.PRE_EVENT)) != 0)).sum()))
    by_grade = {g.name.lower(): int((index['grade'] == int(g)).sum()) for g in Grade}
    minus, pine = per_env.get('Minus Two', {}).get('frames', 0), per_env.get('Pine Valley', {}).get('frames', 0)
    lat = prep.get('notes', {}).get('lateral_impacts', [])
    summary = dict(n_frames=int(len(index)), n_runs=len(runs), n_failed=len(failures),
                   disk_gb=round(sum((root / r['frames_file']).stat().st_size for r in runs if r.get('frames_file')) / 1e9, 2),
                   per_env={k: v['frames'] for k, v in per_env.items()}, folds=folds, by_grade=by_grade,
                   events=dict(total=len(events), terminal=sum(e['kind'] == 'terminal_impact' for e in events),
                               contacts=sum(e['kind'] == 'contact' for e in events)),
                   clean_windows=len(windows), missing_clean_windows=prep.get('missing_windows'),
                   lateral_impacts_matched=sum(1 for x in lat if x['matched_dt_s'] <= 0.1), lateral_impacts=len(lat),
                   pass_criteria=dict(frames_ge_100k=len(index) >= 100_000, minus_two_ge_8k=minus >= 8000,
                                      pine_valley_ge_8k=pine >= 8000,
                                      all_clean_and_pre_event_frames=bool(all(dense_ok)) and not failures))
    return dict(summary=summary, per_env=per_env, failures=failures, manifest_index_sha256=manifest.get('index_sha256'),
                lateral_impacts=lat)


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
    except Exception:
        return None


# ----------------------------------------------------------------------------- repose

def repose(args):
    """Apply timing/refined.json: shift t_wall by delta, re-interpolate pose fields from the run CSV."""
    lock = thermal.require_flight_lock_path(args.flight_lock)
    thermal.limit_threads(2)
    guard = thermal.ChunkGuard(lock)
    root = Path(args.store)
    timing = Path(args.timing) if args.timing else root / 'timing' / 'refined.json'
    refined = json.loads(timing.read_text(encoding='utf-8'))
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise SystemExit('store is not complete')
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
        guard.before_chunk()
        rec = runs[rid]
        if rec['source'] != 'run_video' or not rec.get('telemetry_csv'):
            continue
        m = np.flatnonzero(new['run_id'] == rid)
        if not len(m):
            continue
        tel = load_run_csv(rec['telemetry_csv'])
        prev_delta = new['timing_delta_s'][m].astype(np.float64)
        delta = float(r['delta_s'])
        t = new['t_wall'][m] - prev_delta + delta            # relative to the original alignment
        s = tel.sample(t)
        rows = new[m]
        rows['t_wall'] = t
        rows['pos'], rows['quat'], rows['vel'], rows['omega'] = s['pos'], s['quat'], s['vel'], s['omega']
        rows['t_phase'], rows['t_game'] = s['phase'], s['ts']
        rows['timing_delta_s'] = delta
        ev = [e for e in events if e['run_id'] == rid]
        tti, eid = tti_for(t, ev)
        rows['tti_s'], rows['event_id'] = tti, eid
        flags = rows['flags'].astype(np.uint32)
        flags &= ~np.uint32(int(Flag.PRE_EVENT | Flag.IN_CLEAN | Flag.HIGH_YAW_RATE))
        flags |= np.where(tti <= PRE_EVENT_DENSE_S, int(Flag.PRE_EVENT), 0).astype(np.uint32)
        flags |= np.where(in_clean(s['phase'], [w for w in windows if w['run_id'] == rid]), int(Flag.IN_CLEAN), 0).astype(np.uint32)
        wz = np.abs(s['omega'][:, 2])
        flags |= np.where(wz > HIGH_YAW_RATE, int(Flag.HIGH_YAW_RATE), 0).astype(np.uint32)
        flags |= np.uint32(int(Flag.TIMING_REFINED))
        rows['flags'] = flags.astype(np.uint16)
        new[m] = rows
        rec.setdefault('alignment', {})['refine_delta_s'] = delta
        applied[rid] = delta
    k = 1
    while (root / f'index.v{k}.npy').exists():
        k += 1
    os.replace(root / 'index.npy', root / f'index.v{k}.npy')
    np.save(root / 'index.npy', new)
    _json_dump(root / 'runs.json', runs)
    manifest.update(index_sha256=index_sha256(new), n_frames=int(len(new)),
                    timing=dict(refined=_rel(timing), applied_runs=len(applied), previous_index=f'index.v{k}.npy',
                                previous_index_sha256=index_sha256(index)))
    _json_dump(root / 'manifest.json', manifest)
    _log(f'repose: {len(applied)} runs re-posed; previous index kept as index.v{k}.npy; '
         f'new index sha256 {manifest["index_sha256"][:16]}')
    return applied
