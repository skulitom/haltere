"""Gap cue offline evaluation (OFFLINE ONLY): overlay-mask cache, new-flight extraction, gates G1-G4.

The runtime rule is haltere.vision.gap_cue; this module feeds it recorded frames and scores it. It reads
hindsight quantities (impact times, time to impact, hand-entered pillar boxes from the impact anatomy) ONLY to
select and score frames; none of them reaches the cue. Configs: configs/obstacles/gap_cue.json (runtime
parameters) and configs/obstacles/gap_cue_gates.json (gate definitions), both frozen with their sha256
(``freeze``) before any gate is scored; ``score`` refuses unfrozen or changed files and records both hashes.

Disparity sources (``basis``):
- ``teacher``: the obstacle store's teacher cache (labels/teacher/disparity.npy: DA-V2-Small, input 602 x 336
  bicubic, (36, 64) block means) for store frames; for new flights the same preprocessing through
  haltere.vision.relative_depth.RelativeDepth(input_hw=(336, 602)).
- ``runtime``: RelativeDepth() at 448 x 252 fp16 (what a flight would run) for new flights and for the store
  frames of the Minus Two / Pine Valley gate runs.

Commands (``python -m haltere.obstacles.gap_cue_eval <cmd> --out DIR``):
  masks     overlay block fractions (hud, ring, ghost, propeller) for store runs (CPU)
  frames    decode new flight videos, align them to telemetry (HUD ring marker vs logged cue), keep every new
            frame at 448 x 252 in a scratch memmap with its masks and pose (CPU)
  depth     relative disparity for new-flight frames (both bases), G4 perturbations and the store runtime
            cross-check (GPU, one ChunkGuard chunk)
  freeze    freeze gap_cue.json and gap_cue_gates.json
  score     gates G1-G4 and the Pine Valley trunk case -> DIR/results.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..vision import gap_cue as gc
from .thermal import ChunkGuard, limit_threads, require_flight_lock_path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATES_PATH = REPO_ROOT / 'configs' / 'obstacles' / 'gap_cue_gates.json'
MASK_LAYERS = ('hud', 'ring', 'ghost', 'propeller')
GRID = (36, 64)
FRAME_HW = (252, 448)
IN_CLEAN, POST_EVENT = 1, 4


def _log(*a):
    print(*a, flush=True)


def data_root() -> Path:
    env = os.environ.get('HALTERE_DATA_ROOT')
    if env:
        return Path(env)
    if (REPO_ROOT / 'runs').exists():
        return REPO_ROOT
    return Path('C:/DEV/Haltere')


def load_gates(path: str | Path = GATES_PATH, *, require_frozen: bool = False) -> tuple[dict, str]:
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    sha = gc.config_sha256(obj)
    if obj.get('frozen') and obj.get('sha256') != sha:
        raise RuntimeError(f'{path} changed after freezing (sha256 mismatch)')
    if require_frozen and not obj.get('frozen'):
        raise RuntimeError(f'{path} is not frozen')
    return obj, sha


def freeze_file(path: str | Path) -> dict:
    """Mark a config frozen with its content sha256; a frozen file that changed must bump ``version`` first."""
    path = Path(path)
    obj = json.loads(path.read_text(encoding='utf-8'))
    sha = gc.config_sha256(obj)
    if obj.get('frozen') and obj.get('sha256') != sha:
        raise ValueError(f'{path} was frozen and has since changed: bump "version" and freeze again')
    if not obj.get('frozen'):
        obj.update(frozen=True, frozen_at=_dt.datetime.now().isoformat(timespec='seconds'), sha256=sha)
        path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return obj


# ----------------------------------------------------------------------------- masks

def mask_fractions(rgb) -> tuple[np.ndarray, tuple]:
    """(4, 36, 64) uint8 percent of masked pixels per block for MASK_LAYERS, and the detected ring markers."""
    from . import overlays as ov
    m = ov.overlay_masks(rgb)
    out = np.empty((len(MASK_LAYERS),) + GRID, np.uint8)
    for k, name in enumerate(MASK_LAYERS):
        a = np.asarray(getattr(m, name), bool)
        frac = a.reshape(GRID[0], FRAME_HW[0] // GRID[0], GRID[1], FRAME_HW[1] // GRID[1]).mean(axis=(1, 3))
        out[k] = np.rint(frac * 100).astype(np.uint8)
    return out, m.rings_uv


def validity_from_fractions(frac, layers, max_fraction) -> np.ndarray:
    """(36, 64) bool: blocks whose union-of-layers masked fraction is <= max_fraction (union ~ max of layers:
    the layers rarely overlap; the sum is used, capped at 100)."""
    f = np.zeros(GRID, np.int32)
    for name in layers:
        f += frac[MASK_LAYERS.index(name)].astype(np.int32)
    return np.minimum(f, 100) <= max_fraction * 100 + 1e-9


def cmd_masks(args):
    from .store import FrameStore
    limit_threads(2, cv2=True)
    st = FrameStore(Path(data_root()) / 'runs' / 'obstacle-store-v1')
    runs = [int(r) for r in args.runs.split(',')]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'store_masks.npz'
    have = dict(np.load(path)) if path.exists() else {}
    rid_all = np.asarray(st.index['run_id'])
    for rid in runs:
        key = f'r{rid}'
        if key in have:
            continue
        rows = np.flatnonzero(rid_all == rid)
        fr = np.empty((len(rows), len(MASK_LAYERS)) + GRID, np.uint8)
        t0 = time.time()
        for k, i in enumerate(rows):
            fr[k] = mask_fractions(np.asarray(st.frame(i)))[0]
        have[key] = fr
        have[key + '_rows'] = rows
        np.savez_compressed(path, **have)
        _log(f'run {rid}: {len(rows)} frames masked in {time.time() - t0:.0f} s')


# ----------------------------------------------------------------------------- sequences

def run_sequence(t, disp, quat, cue, vel, valid, params, response, latency_s):
    """Feed one flight's frames (time order) through a GapCue; returns per-frame arrays."""
    cue_obj = gc.GapCue(replace(params, enabled=True), gc.DEFAULT_CAMERA, response)
    n = len(t)
    out = dict(shift=np.zeros(n), raw=np.full(n, np.nan), confirmed=np.zeros(n, bool), valid=np.zeros(n, bool),
               kind=np.empty(n, object), ring=np.full(n, np.nan), r_ring=np.full(n, np.nan),
               r_peak=np.full(n, np.nan), lr=np.full(n, np.nan), near_on_path=np.zeros(n, bool),
               episode=np.zeros(n, np.int32), interval_lo=np.full(n, np.nan), interval_hi=np.full(n, np.nan))
    for k in range(n):
        o = cue_obj.update(disp[k], quat[k], cue[k], float(t[k]), float(t[k]) + latency_s, velocity=vel[k],
                           valid=None if valid is None else valid[k])
        d = o.decision
        out['shift'][k] = o.shift_deg
        out['confirmed'][k] = o.confirmed
        out['kind'][k] = o.kind
        out['episode'][k] = o.episode
        if d is not None:
            out['valid'][k] = d.valid
            out['raw'][k] = d.shift_deg if d.valid else np.nan
            out['ring'][k] = np.nan if d.ring_bearing_deg is None else d.ring_bearing_deg
            out['r_ring'][k], out['r_peak'][k], out['lr'][k] = d.r_ring, d.r_peak, d.lr
            out['near_on_path'][k] = bool(d.near_on_path)
            if d.interval_deg is not None:
                out['interval_lo'][k], out['interval_hi'][k] = d.interval_deg
    return out


def episodes_of(confirmed) -> list:
    """(start, end) index ranges of confirmed runs."""
    c = np.asarray(confirmed, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, c.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def checkpoint_switches(t, ring_deg, speed, *, jump_deg=8.0, settle_deg=(4.0, 6.0), max_dt=0.25, min_sep_s=1.0,
                        min_speed=3.0):
    """Frames where the ring bearing jumps (a new checkpoint became the target): the m2design anatomy rule
    (bearing jump > 8 deg within 0.25 s, next two frames within 4 / 6 deg of the new bearing, >= 1 s apart,
    speed >= 3 m/s). Uses the cue bearing only (as the pilot sees it)."""
    ok = np.flatnonzero(np.isfinite(ring_deg))
    ev = []
    for a in range(1, len(ok) - 2):
        i0, i, i1, i2 = ok[a - 1], ok[a], ok[a + 1], ok[a + 2]
        if t[i] - t[i0] >= max_dt:
            continue
        if abs(gc.wrap_deg(ring_deg[i] - ring_deg[i0])) <= jump_deg:
            continue
        if abs(gc.wrap_deg(ring_deg[i1] - ring_deg[i])) >= settle_deg[0] or \
                abs(gc.wrap_deg(ring_deg[i2] - ring_deg[i])) >= settle_deg[1]:
            continue
        if speed[i] < min_speed:
            continue
        if ev and t[i] - t[ev[-1]] < min_sep_s:
            continue
        ev.append(i)
    return ev


# ----------------------------------------------------------------------------- new flights: frames (CPU)

FLIGHT_DIR = Path('runs') / 'fast-stack-20260923'
ALIGN_OFFSETS = np.arange(-0.6, 0.4001, 0.01)
ALIGN_DEFAULT = -0.03
ALIGN_MIN_MARKERS = 12
LEAK_EVERY = 10
LEAK_SEED = 20260925


def _first_marker(rings_uv):
    return (np.asarray(rings_uv[0], np.float64) if len(rings_uv) else np.array([np.nan, np.nan]))


def align_offset(t_video, markers, tel, wall0) -> tuple[float, dict]:
    """Video offset (s) minimising the median distance (1280 x 720 px) between the overlay-detected marker and
    the logged cue interpolated at the candidate capture time; ``ALIGN_DEFAULT`` with too few markers."""
    have = np.isfinite(markers).all(1)
    if tel.cue is None or have.sum() < ALIGN_MIN_MARKERS:
        return ALIGN_DEFAULT, dict(method='default', markers=int(have.sum()))
    ct, uv, ok = tel.cue['t'], tel.cue['uv'], tel.cue['ok']
    tv, mk = t_video[have], markers[have]
    curve = []
    for off in ALIGN_OFFSETS:
        t = wall0 + off + tv
        j = np.clip(np.searchsorted(ct, t) - 1, 0, len(ct) - 2)
        j2 = j + 1
        a = np.clip((t - ct[j]) / np.maximum(ct[j2] - ct[j], 1e-3), 0, 1)
        both = ok[j] & ok[j2] & (np.abs(ct[j] - t) < 0.12) & (np.abs(ct[j2] - t) < 0.12)
        pu = (uv[j, 0] + a * (uv[j2, 0] - uv[j, 0])) * 1280
        pv = (uv[j, 1] + a * (uv[j2, 1] - uv[j, 1])) * 720
        e = np.hypot(pu - mk[:, 0] * 1280, pv - mk[:, 1] * 720)[both]
        curve.append((float(off), float(np.median(e)) if len(e) > 8 else np.inf, int(len(e))))
    best = min(curve, key=lambda c: c[1])
    if not np.isfinite(best[1]):
        return ALIGN_DEFAULT, dict(method='default (no matches)', markers=int(have.sum()))
    at_default = min(curve, key=lambda c: abs(c[0] - ALIGN_DEFAULT))
    return best[0], dict(method='hud marker', markers=int(have.sum()), matched=best[2],
                         median_err_px=round(best[1], 2), err_px_at_default=round(at_default[1], 2))


def _impact_wall(tel, sidecar) -> float | None:
    imp = (sidecar or {}).get('impact')
    if not imp:
        return None
    a = tel.attempt_starts[-1]
    k = a + int(np.argmin(np.abs(tel.ts[a:] - float(imp['timestamp']))))
    return float(tel.wall[k])


def extract_flight(name: str, root: Path, out: Path, guard, log=_log) -> dict:
    """Decode one flight video: every new frame at 448 x 252 (to ``<name>.u8``), overlay block fractions,
    markers, alignment, pose, drop reasons and the G4 perturbed frames (``<name>.leaks.u8``)."""
    import cv2
    from . import leaks
    from . import overlays as ov
    from .store_build import (REPEAT_GREY, VideoReader, drop_reasons, grey160, hud_present_320, load_run_csv,
                              pause_menu_320, to_store, video_props, MENU_DILATE, _dilate)
    base = root / FLIGHT_DIR
    vpath, cpath, spath = base / f'{name}.mp4', base / f'{name}.csv', base / f'{name}.json'
    fps, W, H = video_props(vpath)
    tel = load_run_csv(cpath)
    sidecar = json.loads(spath.read_text(encoding='utf-8')) if spath.exists() else {}
    fpath = out / f'{name}.u8'
    t0 = time.time()
    rd = VideoReader(vpath, W, H)
    dg, menu, hud, new_idx = [], [], [], []
    fracs, markers = [], []
    prev = None
    k = 0
    try:
        with open(fpath, 'wb') as fw:
            while True:
                if k % 500 == 0:
                    guard.before_chunk()
                fr = rd.read()
                if fr is None:
                    break
                s320 = cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA)
                g = grey160(s320.astype(np.float32))
                d = 99.0 if prev is None else float(np.abs(g - prev).mean())
                prev = g
                dg.append(d)
                menu.append(pause_menu_320(s320))
                hud.append(hud_present_320(s320))
                if k == 0 or d >= REPEAT_GREY:
                    rgb = to_store(fr)
                    f, rings = mask_fractions(rgb)
                    fracs.append(f)
                    markers.append(_first_marker(rings))
                    new_idx.append(k)
                    fw.write(np.ascontiguousarray(rgb).tobytes())
                k += 1
    finally:
        rd.close()
    n = k
    new_idx = np.asarray(new_idx, np.int64)
    last = np.r_[new_idx[1:] - 1, n - 1]
    t_video = last / fps
    markers = np.asarray(markers, np.float64).reshape(-1, 2)
    off, info = align_offset(t_video, markers, tel, tel.wall0)
    t = tel.wall0 + t_video + off
    s = tel.sample(t)
    imp = _impact_wall(tel, sidecar)
    events = [dict(kind='terminal_impact', t_wall=imp)] if imp is not None else []
    menu_new = _dilate(np.array(menu, bool), MENU_DILATE)[new_idx]
    hud_new = np.array(hud, bool)[new_idx]
    why = drop_reasons(t, tel, events, repeat=np.zeros(len(t), bool), menu=menu_new, hud=hud_new,
                       finished=imp is None)
    keep = why == ''
    cue = tel.cue_at(t)
    # G4 perturbations on every LEAK_EVERY-th kept frame with a detected marker
    rng = np.random.default_rng(LEAK_SEED + sum(map(ord, name)))
    cand = np.flatnonzero(keep & np.isfinite(markers).all(1))[::LEAK_EVERY]
    mm = np.memmap(fpath, dtype=np.uint8, mode='r', shape=(len(new_idx),) + FRAME_HW + (3,))
    leak_src, leak_kind, leak_frac = [], [], []
    with open(out / f'{name}.leaks.u8', 'wb') as fl:
        for j in cand:
            rgb = np.array(mm[j])
            mu, mv = markers[j] * np.array([FRAME_HW[1], FRAME_HW[0]])
            side = rng.choice([-1, 1])
            cx = int(np.clip(mu + side * rng.uniform(25, 90), 10, FRAME_HW[1] - 10))
            cy = int(np.clip(mv + rng.uniform(-6, 6), 10, FRAME_HW[0] - 10))
            painted = leaks.paint_ring(rgb, (cx, cy), int(rng.uniform(8, 16)), 3)[0]
            removed = leaks.remove_ring(rgb, ov.overlay_masks(rgb).ring)
            for kind, img in (('ring_painted', painted), ('ring_removed', removed)):
                leak_src.append(j)
                leak_kind.append(kind)
                leak_frac.append(mask_fractions(img)[0])
                fl.write(np.ascontiguousarray(img).tobytes())
    del mm
    rec = dict(name=name, fps=fps, n_decoded=n, frame_idx=new_idx, t_video=t_video, t_wall=t, phase=s['phase'],
               pos=s['pos'], quat=s['quat'], vel=s['vel'], tel_valid=s['valid'], cue_uv=cue, markers=markers,
               maskfrac=np.asarray(fracs, np.uint8), why=np.asarray(why, dtype='U24'), keep=keep,
               impact_wall=np.nan if imp is None else imp, offset_s=off, align=json.dumps(info),
               leak_src=np.asarray(leak_src, np.int64), leak_kind=np.asarray(leak_kind, dtype='U16'),
               leak_maskfrac=np.asarray(leak_frac, np.uint8).reshape(-1, len(MASK_LAYERS), *GRID))
    np.savez_compressed(out / f'{name}.npz', **rec)
    log(f'{name}: {n} decoded, {len(new_idx)} new, {int(keep.sum())} kept, offset {off:+.2f} s {info}, '
        f'{len(leak_src)} leak frames, {time.time() - t0:.0f} s')
    return rec


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
        extract_flight(name, data_root(), out, guard)


# ----------------------------------------------------------------------------- relative depth (GPU)

def cmd_depth(args):
    """Relative disparity for every new-flight frame and leak frame (teacher and runtime preprocessing) and for
    the store frames of the store gate runs (runtime preprocessing). One GPU chunk of at most args.max_s;
    resumable per flight."""
    limit_threads(2, torch=True, cv2=True)
    from ..vision.relative_depth import RelativeDepth, TEACHER_INPUT_HW
    gates, _ = load_gates()
    guard = ChunkGuard(require_flight_lock_path(args.flight_lock), gpu=True, chunk_max_s=float(args.max_s))
    fdir = Path(args.out) / 'flights'
    ddir = Path(args.out) / 'depth'
    ddir.mkdir(parents=True, exist_ok=True)
    guard.before_chunk()
    t0 = time.time()
    rt = RelativeDepth()
    te = RelativeDepth(input_hw=TEACHER_INPUT_HW)
    prov = dict(runtime=rt.provenance, teacher=te.provenance, runtime_latency_ms=round(rt.latency_ms(), 2),
                teacher_latency_ms=round(te.latency_ms(), 2))
    (ddir / 'provenance.json').write_text(json.dumps(prov, indent=1) + '\n', encoding='utf-8')
    bs = int(args.batch)

    def run(mm):
        a = np.empty((len(mm),) + GRID, np.float16)
        b = np.empty((len(mm),) + GRID, np.float16)
        for s in range(0, len(mm), bs):
            if guard.chunk_expired():
                raise TimeoutError('GPU chunk expired')
            x = np.asarray(mm[s:s + bs])
            a[s:s + bs] = rt(x)
            b[s:s + bs] = te(x)
        return a, b

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
            d_rt, d_te = run(mm)
            nl = len(rec['leak_src'])
            if nl:
                ml = np.memmap(fdir / f'{name}.leaks.u8', dtype=np.uint8, mode='r', shape=(nl,) + FRAME_HW + (3,))
                l_rt, l_te = run(ml)
            else:
                l_rt = l_te = np.zeros((0,) + GRID, np.float16)
            np.savez_compressed(path, runtime=d_rt, teacher=d_te, leak_runtime=l_rt, leak_teacher=l_te)
            done.append(name)
            _log(f'{name}: {n} frames + {nl} leak frames, {time.time() - t0:.0f} s')
        path = ddir / 'store_runtime.npz'
        if not path.exists():
            from .store import FrameStore
            st = FrameStore(data_root() / 'runs' / 'obstacle-store-v1')
            rid = np.asarray(st.index['run_id'])
            outd = {}
            for key in gates['store_runs']:
                rows = np.flatnonzero(rid == int(key))
                frames = st.frames(rows)
                a = np.empty((len(rows),) + GRID, np.float16)
                for s in range(0, len(rows), bs):
                    if guard.chunk_expired():
                        raise TimeoutError('GPU chunk expired')
                    a[s:s + bs] = rt(frames[s:s + bs])
                outd[f'r{key}'] = a
                outd[f'r{key}_rows'] = rows
            np.savez_compressed(path, **outd)
            _log(f'store runtime cross-check: {sum(len(v) for k, v in outd.items() if k.endswith("_rows"))} frames')
    finally:
        _log(f'depth chunk: {time.time() - t0:.0f} s, flights done {done}, guard {guard.summary()}')


# ----------------------------------------------------------------------------- scoring

class Seq(dict):
    """One flight's frames in time order with the cue outputs (dict of arrays)."""


def _motor(gates, key):
    return gates['motors'][key]


def store_sequence(rid, basis, params, layers, response, latency, cache):
    ix, disp_te, masks, rt = cache['index'], cache['teacher'], cache['masks'], cache.get('runtime')
    rows = masks[f'r{rid}_rows']
    fr = masks[f'r{rid}']
    flags = np.asarray(ix['flags'][rows])
    keep = (flags & POST_EVENT) == 0
    sel = np.flatnonzero(keep)
    t = np.asarray(ix['t_wall'][rows[sel]], np.float64)
    o = np.argsort(t, kind='stable')
    sel = sel[o]
    rows_s = rows[sel]
    if basis == 'teacher':
        d = np.asarray(disp_te[rows_s], np.float32)
    else:
        rr = rt[f'r{rid}_rows']
        pos = {int(r): k for k, r in enumerate(rr)}
        d = np.asarray(rt[f'r{rid}'][[pos[int(r)] for r in rows_s]], np.float32)
    valid = np.stack([validity_from_fractions(f, layers, params.block_mask_max_fraction) for f in fr[sel]])
    q = np.asarray(ix['quat'][rows_s], np.float64)
    cue = np.asarray(ix['cue_uv'][rows_s], np.float64)
    vel = np.asarray(ix['vel'][rows_s], np.float64)
    t = np.asarray(ix['t_wall'][rows_s], np.float64)
    res = Seq(run_sequence(t, d, q, cue, vel, valid, params, response, latency))
    res.update(t=t, pos=np.asarray(ix['pos'][rows_s], np.float64), vel=vel, tti=np.asarray(ix['tti_s'][rows_s], float),
               phase=np.asarray(ix['t_phase'][rows_s], float), flags=np.asarray(ix['flags'][rows_s]), rows=rows_s)
    return res


def flight_sequence(name, basis, params, layers, response, latency, out):
    rec = np.load(Path(out) / 'flights' / f'{name}.npz')
    dep = np.load(Path(out) / 'depth' / f'{name}.npz')
    keep = rec['keep']
    sel = np.flatnonzero(keep)
    t = rec['t_wall'][sel]
    o = np.argsort(t, kind='stable')
    sel = sel[o]
    d = np.asarray(dep[basis][sel], np.float32)
    valid = np.stack([validity_from_fractions(f, layers, params.block_mask_max_fraction) for f in rec['maskfrac'][sel]])
    res = Seq(run_sequence(rec['t_wall'][sel], d, rec['quat'][sel], rec['cue_uv'][sel], rec['vel'][sel], valid,
                           params, response, latency))
    imp = float(rec['impact_wall'])
    res.update(t=rec['t_wall'][sel], pos=rec['pos'][sel], vel=rec['vel'][sel], phase=rec['phase'][sel],
               tti=(imp - rec['t_wall'][sel]) if np.isfinite(imp) else np.full(len(sel), np.inf), sel=sel)
    return res


def _pillar_extent(pos, box):
    xs, ys = box['x'], box['y']
    az = [np.degrees(np.arctan2(y - pos[:, 1], x - pos[:, 0])) for x in xs for y in ys]
    az = np.stack(az, 1)
    ref = az[:, :1]
    rel = gc.wrap_deg(az - ref)
    return ref[:, 0] + rel.min(1), ref[:, 0] + rel.max(1)


def g1_approach(seq, g):
    p = seq['pos']
    dx, dy = g['distance_to']
    d = np.hypot(dx - p[:, 0], dy - p[:, 1])
    reg = ((p[:, 0] > g['approach_region']['x'][0]) & (p[:, 0] < g['approach_region']['x'][1])
           & (p[:, 1] > g['approach_region']['y'][0]) & (p[:, 1] < g['approach_region']['y'][1]))
    conf = seq['confirmed']
    left = np.flatnonzero(reg & conf & (seq['shift'] > 0))
    right = np.flatnonzero(reg & conf & (seq['shift'] < 0))
    raw_left = np.flatnonzero(reg & (seq['raw'] >= 2.0))
    lo, hi = _pillar_extent(p, g['pillar_box_xy'])
    aim = seq['ring'] + seq['shift']
    inside = lambda a: (gc.wrap_deg(a - lo) >= 0) & (gc.wrap_deg(hi - a) >= 0)
    aim_in = reg & conf & np.isfinite(aim) & inside(aim) & ~inside(seq['ring'])
    ts = seq['t'][reg]
    first = float(d[left[0]]) if len(left) else None
    return dict(frames=int(reg.sum()), rate_hz=round(float(1 / np.median(np.diff(ts))), 1) if len(ts) > 2 else None,
                first_left_confirmed_d_m=None if first is None else round(first, 2),
                first_raw_left_d_m=round(float(d[raw_left[0]]), 2) if len(raw_left) else None,
                passes=bool(first is not None and first >= g['left_confirm_distance_m']),
                confirmed_right_d_m=[round(float(d[k]), 2) for k in right],
                aim_into_pillar_d_m=[round(float(d[k]), 2) for k in np.flatnonzero(aim_in)],
                closest_d_m=round(float(d[reg].min()), 2) if reg.any() else None)


def g2_run(seq, g):
    w = (seq['tti'] >= g['window_tti_s'][0]) & (seq['tti'] <= g['window_tti_s'][1])
    xdir = seq['shift'] * (-np.sin(np.radians(seq['ring'])))
    conf = seq['confirmed']
    mx = np.flatnonzero(w & conf & (xdir < 0))
    px = np.flatnonzero(w & conf & (xdir > 0))
    first = float(seq['tti'][mx[0]]) if len(mx) else None
    before = [round(float(seq['tti'][k]), 2) for k in px if not len(mx) or k < mx[0]]
    return dict(first_minus_x_tti_s=None if first is None else round(first, 2),
                passes=bool(first is not None and first >= g['min_lead_s']), plus_x_before_tti_s=before,
                frames=int(w.sum()))


def g3_flight(seq, g):
    t = seq['t']
    dt = np.diff(t)
    minutes = float(np.sum(dt[dt < 0.3])) / 60
    eps = episodes_of(seq['confirmed'])
    conf_abs = np.abs(seq['shift'][seq['confirmed']])
    sp = np.hypot(seq['vel'][:, 0], seq['vel'][:, 1])
    sw = checkpoint_switches(t, seq['ring'], sp)
    ok = []
    worst = []
    for i in sw:
        m = (t >= t[i] - g['switch_window_s']) & (t < t[i])
        mx = float(np.max(np.abs(seq['shift'][m]))) if m.any() else 0.0
        ok.append(mx <= g['switch_abs_shift_max_deg'])
        worst.append(mx)
    return dict(minutes=round(minutes, 2), frames=int(len(t)), episodes=len(eps),
                episodes_per_min=round(len(eps) / max(minutes, 1e-9), 2), confirmed_frames=int(len(conf_abs)),
                confirmed_share=round(float(seq['confirmed'].mean()), 4),
                abs_shift_p50=round(float(np.percentile(conf_abs, 50)), 2) if len(conf_abs) else 0.0,
                abs_shift_p90=round(float(np.percentile(conf_abs, 90)), 2) if len(conf_abs) else 0.0,
                switches=len(sw), switches_ok=int(sum(ok)),
                switch_max_abs_shift_deg=[round(x, 1) for x in worst],
                kinds={k: int((seq['kind'] == k).sum()) for k in gc.KINDS if (seq['kind'] == k).any()},
                near_on_path_share=round(float(seq['near_on_path'].mean()), 4))


def g4_flight(name, basis, params, layers, out):
    rec = np.load(Path(out) / 'flights' / f'{name}.npz')
    dep = np.load(Path(out) / 'depth' / f'{name}.npz')
    res = {}
    src, kinds = rec['leak_src'], rec['leak_kind']
    for kind in ('ring_painted', 'ring_removed'):
        ds = []
        for k in np.flatnonzero(kinds == kind):
            j = int(src[k])
            v0 = validity_from_fractions(rec['maskfrac'][j], layers, params.block_mask_max_fraction)
            v1 = validity_from_fractions(rec['leak_maskfrac'][k], layers, params.block_mask_max_fraction)
            a = gc.decide(np.asarray(dep[basis][j], np.float32), rec['quat'][j], rec['cue_uv'][j], valid=v0, params=params)
            b = gc.decide(np.asarray(dep['leak_' + basis][k], np.float32), rec['quat'][j], rec['cue_uv'][j], valid=v1,
                          params=params)
            if a.valid and b.valid:
                ds.append(abs(a.shift_deg - b.shift_deg))
        res[kind] = np.asarray(ds)
    # HUD-only synthetic disparity on every kept frame
    bad = n = 0
    for j in np.flatnonzero(rec['keep']):
        f = rec['maskfrac'][j]
        d = 1.0 + 9.0 * f[MASK_LAYERS.index('hud')].astype(np.float32) / 100.0
        v = validity_from_fractions(f, layers, params.block_mask_max_fraction)
        dec = gc.decide(d, rec['quat'][j], rec['cue_uv'][j], valid=v, params=params, keep_profile=True)
        if not dec.valid:
            continue
        n += 1
        bad += int((np.isfinite(dec.ratio) & (dec.ratio > params.kappa)).any())
    res['hud_only'] = (n, bad)
    return res


def cmd_freeze(args):
    for p in (gc.CONFIG_PATH, GATES_PATH):
        o = freeze_file(p)
        _log(f'{p.name}: frozen {o["frozen_at"]} sha256 {o["sha256"]}')


def cmd_score(args):
    limit_threads(2, cv2=True)
    params, cfg, cfg_sha = gc.load_config(require_frozen=True)
    gates, gates_sha = load_gates(require_frozen=True)
    params = replace(params, enabled=True)
    layers = tuple(cfg['mask_layers'])
    models = gc.load_response_models()
    lat = float(gates['latency_s'])
    out = Path(args.out)
    S = data_root() / 'runs' / 'obstacle-store-v1'
    cache = dict(index=np.load(S / 'index.npy', mmap_mode='r'),
                 teacher=np.load(S / 'labels' / 'teacher' / 'disparity.npy', mmap_mode='r'),
                 masks=np.load(out / 'store_masks.npz'))
    rtp = out / 'depth' / 'store_runtime.npz'
    if rtp.exists():
        cache['runtime'] = np.load(rtp)
    results = dict(schema='haltere.obstacles.gap_cue_results.v1',
                   scored_at=_dt.datetime.now().isoformat(timespec='seconds'),
                   gap_cue_config_sha256=cfg_sha, gates_sha256=gates_sha,
                   gap_cue_config_version=cfg.get('version'), gates_version=gates.get('version'),
                   depth_provenance=json.loads((out / 'depth' / 'provenance.json').read_text())
                   if (out / 'depth' / 'provenance.json').exists() else None, bases={})
    for basis in ('teacher', 'runtime'):
        R = {}
        seqs = {}

        def seq_for(spec):
            key = spec.get('flight') or f"store:{spec['store_run_id']}"
            if key not in seqs:
                if 'flight' in spec:
                    mot = _motor(gates, gates['new_flights'][spec['flight']]['motor'])
                    seqs[key] = flight_sequence(spec['flight'], basis, params, layers, models[mot], lat, out)
                else:
                    rid = str(spec['store_run_id'])
                    mot = _motor(gates, gates['store_runs'][rid]['motor'])
                    seqs[key] = store_sequence(int(rid), basis, params, layers, models[mot], lat, cache)
            return seqs[key]
        g = gates['G1']
        app = {a['name']: g1_approach(seq_for(a), g) for a in g['approaches']}
        extra = {a['name']: g1_approach(seq_for(a), g) for a in g['reported_not_gated']}
        n_pass = sum(a['passes'] for a in app.values())
        into = sum(len(a['aim_into_pillar_d_m']) for a in app.values())
        R['G1'] = dict(approaches=app, reported_not_gated=extra, passing=n_pass, required=g['min_passing_approaches'],
                       aim_into_pillar_frames=into,
                       passes=bool(n_pass >= g['min_passing_approaches'] and into == 0))
        g = gates['G2']
        runs = {r['name']: g2_run(seq_for(r), g) for r in g['runs']}
        R['G2'] = dict(runs=runs, passes=bool(all(r['passes'] for r in runs.values())))
        g = gates['G3']
        fl = {f: g3_flight(seq_for({'flight': f}), g) for f in g['flights']}
        mins = sum(v['minutes'] for v in fl.values())
        eps = sum(v['episodes'] for v in fl.values())
        allabs = np.concatenate([np.abs(seq_for({'flight': f})['shift'][seq_for({'flight': f})['confirmed']])
                                 for f in g['flights']])
        sw = sum(v['switches'] for v in fl.values())
        swok = sum(v['switches_ok'] for v in fl.values())
        tot = dict(minutes=round(mins, 2), episodes_per_min=round(eps / max(mins, 1e-9), 2),
                   abs_shift_p90=round(float(np.percentile(allabs, 90)), 2) if len(allabs) else 0.0,
                   switches=sw, switch_ok_fraction=round(swok / max(sw, 1), 4))
        R['G3'] = dict(flights=fl, total=tot, passes=bool(tot['episodes_per_min'] <= g['max_episodes_per_min']
                                                           and tot['abs_shift_p90'] <= g['p90_abs_shift_max_deg']
                                                           and tot['switch_ok_fraction'] >= g['switch_min_fraction']),
                       checks=dict(episodes=tot['episodes_per_min'] <= g['max_episodes_per_min'],
                                   p90=tot['abs_shift_p90'] <= g['p90_abs_shift_max_deg'],
                                   switches=tot['switch_ok_fraction'] >= g['switch_min_fraction']))
        g = gates['G4']
        pert = {'ring_painted': [], 'ring_removed': []}
        hud_n = hud_bad = 0
        per = {}
        for f in gates['new_flights']:
            r = g4_flight(f, basis, params, layers, out)
            for k in pert:
                pert[k].append(r[k])
            hud_n += r['hud_only'][0]
            hud_bad += r['hud_only'][1]
            per[f] = {k: dict(n=int(len(r[k])), within=round(float((r[k] <= g['max_dshift_deg']).mean()), 4)
                              if len(r[k]) else None) for k in pert}
        g4 = {}
        for k, v in pert.items():
            v = np.concatenate(v) if v else np.zeros(0)
            g4[k] = dict(n=int(len(v)), within_fraction=round(float((v <= g['max_dshift_deg']).mean()), 4) if len(v) else None,
                         dshift_p95=round(float(np.percentile(v, 95)), 2) if len(v) else None,
                         dshift_max=round(float(v.max()), 2) if len(v) else None)
        g4['hud_only'] = dict(frames=hud_n, frames_with_near_column=hud_bad)
        g4['per_flight'] = per
        g4['passes'] = bool(all(g4[k]['within_fraction'] is not None and g4[k]['within_fraction'] >= g['min_fraction']
                                for k in pert) and hud_bad == 0)
        R['G4'] = g4
        # Pine Valley trunk (report only)
        g = gates['pine_trunk']
        sq = seq_for({'flight': g['flight']})
        w = sq['tti'] <= g['window_before_impact_s']
        rows = []
        for k in np.flatnonzero(w):
            rows.append(dict(tti_s=round(float(sq['tti'][k]), 2), kind=str(sq['kind'][k]),
                             raw_shift=None if not np.isfinite(sq['raw'][k]) else round(float(sq['raw'][k]), 1),
                             shift=round(float(sq['shift'][k]), 1), confirmed=bool(sq['confirmed'][k]),
                             r_ring=None if not np.isfinite(sq['r_ring'][k]) else round(float(sq['r_ring'][k]), 2),
                             near_on_path=bool(sq['near_on_path'][k])))
        conf = [r for r in rows if r['confirmed']]
        side = sorted({'left' if r['shift'] > 0 else 'right' for r in conf})
        R['pine_trunk'] = dict(frames=rows, first_confirmed_tti_s=conf[0]['tti_s'] if conf else None,
                               confirmed_sides=side, reference=g['reference'])
        results['bases'][basis] = R
    prim = results['bases'][gates['basis']['primary']]
    results['verdict'] = {k: prim[k]['passes'] for k in ('G1', 'G2', 'G3', 'G4')}
    (out / 'results.json').write_text(json.dumps(results, indent=1, default=str) + '\n', encoding='utf-8')
    _log(json.dumps(results['verdict']))
    return results


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m haltere.obstacles.gap_cue_eval')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('masks')
    a.add_argument('--out', required=True)
    a.add_argument('--runs', required=True, help='comma-separated store run ids')
    a = sub.add_parser('frames')
    a.add_argument('--out', required=True)
    a.add_argument('--flight-lock', default=None)
    a.add_argument('--flights', default=None, help='comma-separated (default: every new flight of the gates)')
    a = sub.add_parser('depth')
    a.add_argument('--out', required=True)
    a.add_argument('--flight-lock', default=None)
    a.add_argument('--flights', default=None)
    a.add_argument('--batch', type=int, default=32)
    a.add_argument('--max-s', type=float, default=300.0, help='GPU chunk length limit (s)')
    sub.add_parser('freeze')
    a = sub.add_parser('score')
    a.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    {'masks': cmd_masks, 'frames': cmd_frames, 'depth': cmd_depth, 'freeze': cmd_freeze,
     'score': cmd_score}[args.cmd](args)


if __name__ == '__main__':
    main()
