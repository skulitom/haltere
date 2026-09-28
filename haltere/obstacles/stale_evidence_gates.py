"""Offline gates of round 5 (arches): the stale-evidence rule (configs/obstacles/stale_evidence.json: the looming
governor's cap follows the ray of its evidence) and the ring-marker reader rule (configs/pilot/ring_marker.json: a
marker candidate needs a continuous white annulus). The gates are declared in configs/obstacles/stale_evidence_gates.json
and scored only when that file is frozen and names these declarations. Development and held-out data are as the gates
file declares. Nothing here is flight evidence: replays are open loop (the recorded motion does not respond), windows are
semi-closed loop in the IdentifiedSim surrogate (the logged state, the replayed requests), the reader runs on recorded
h264 frames (not the live captures) and the hairpins are the synthetic motor-assist harness scenarios.

Hindsight quantities (impact times, contact points, the Straw Bale start-arch box and the Minus Two arch points) select
and score windows only; none of them reaches a pilot or a reader.

usage (one process, CPU, two threads; resumable: finished outputs are kept):
  python -m haltere.obstacles.stale_evidence_gates frames  --out DIR [--flights a,b]   recorded frames: readers, overlay
  python -m haltere.obstacles.stale_evidence_gates replays --out DIR [--flights a,b]   open-loop replays of the variants
  python -m haltere.obstacles.stale_evidence_gates windows --out DIR                   semi-closed-loop windows
  python -m haltere.obstacles.stale_evidence_gates hairpin --out DIR                   closed-loop harness hairpins
  python -m haltere.obstacles.stale_evidence_gates score   --out DIR --json SCORES.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GATES_PATH = REPO/'configs'/'obstacles'/'stale_evidence_gates.json'
GATES_V1_PATH = REPO/'configs'/'obstacles'/'stale_evidence_gates_v1.json'
META = ('frozen', 'frozen_at', 'sha256')
PY = sys.executable
COMMAND_KEYS = ('cvx', 'cvy', 'cvz', 'yaw_cmd', 'state', 'cap', 'climb', 'descent_scale')


def content_sha256(obj):
    body = {k: v for k, v in obj.items() if k not in META}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
                          .encode('utf-8')).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gates(path=GATES_PATH):
    """The frozen gates and their hash; refuses an unfrozen or edited file and declarations other than the ones it
    names (version and content hash)."""
    gates = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = content_sha256(gates)
    if gates.get('frozen') is not True or gates.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    for spec in gates['declarations'].values():
        declaration = json.loads(declaration_path(spec).read_text(encoding='utf-8'))
        if (declaration.get('frozen') is not True or declaration.get('version') != spec['version']
                or declaration.get('sha256') != spec['sha256'] or content_sha256(declaration) != spec['sha256']):
            raise ValueError(f'{spec["file"]} is not the frozen declaration these gates score')
    return gates, digest


def declaration_path(spec):
    """The declaration file a gates entry names, or, once a later version replaced it, the kept copy beside it
    (<stem>_v<version>.json)."""
    path = REPO/spec['file']
    declared = json.loads(path.read_text(encoding='utf-8'))
    if declared.get('version') != spec['version']:
        kept = path.with_name(f"{path.stem}_v{spec['version']}.json")
        if kept.exists():
            return kept
    return path


def _limit_threads():
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(k, '2')
    import cv2
    import torch
    cv2.setNumThreads(2)
    torch.set_num_threads(2)


def _rule(gates):
    from ..vision.race_cues import ring_marker_rule
    return ring_marker_rule(json.loads(declaration_path(gates['declarations']['ring_marker']).read_text(encoding='utf-8')))


# ---------------------------------------------------------------------------------------------------------------
# Recorded frames: the earlier reader, the reader rule, the overlay (template) detector, alignment to the log
# ---------------------------------------------------------------------------------------------------------------
def read_flight(name, runs, out, rule, i4_every=10, log=print):
    """Every new recorded frame of one flight (repeats of the recorder skipped): the reader candidates with the rule's
    verdict, the earlier reader's and the rule's reading, the overlay detector's markers, and on every i4_every-th frame
    the reader without the rule against the m4b reader (identity). Aligned to the log with the HUD marker. Saves
    OUT/frames/<name>.npz."""
    import cv2
    from .gap_cue_eval import align_offset, mask_fractions
    from .store_build import REPEAT_GREY, VideoReader, grey160, load_run_csv, to_store, video_props
    from ..vision.race_cues import checkpoint_ring, ring_candidates
    m4b = _m4b_reader()
    base = Path(runs)
    vpath = base/f'{name}.mp4'
    fps, W, H = video_props(vpath)
    tel = load_run_csv(base/f'{name}.csv')
    begin = time.time()
    rd = VideoReader(vpath, W, H)
    prev = None
    k = 0
    rows = dict(frame=[], old=[], new=[], overlay=[], overlay_n=[], i4=[])
    cand = dict(frame=[], u=[], v=[], edge=[], ok=[])
    try:
        while True:
            fr = rd.read()
            if fr is None:
                break
            s320 = cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA)
            g = grey160(s320.astype(np.float32))
            diff = 99. if prev is None else float(np.abs(g-prev).mean())
            prev = g
            if k == 0 or diff >= REPEAT_GREY:
                j = len(rows['frame'])
                hits = ring_candidates(fr, rule)
                for h in hits:
                    cand['frame'].append(j)
                    cand['u'].append(h['u'])
                    cand['v'].append(h['v'])
                    cand['edge'].append(h['edge'])
                    cand['ok'].append(h['annulus_ok'])
                old = [h for h in hits]
                new = [h for h in hits if h['annulus_ok']]
                rows['frame'].append(k)
                rows['old'].append((old[0]['u'], old[0]['v'], float(old[0]['edge'])) if len(old) == 1 else (np.nan,)*3)
                rows['new'].append((new[0]['u'], new[0]['v'], float(new[0]['edge'])) if len(new) == 1 else (np.nan,)*3)
                _, rings = mask_fractions(to_store(fr))
                rows['overlay'].append(tuple(rings[0]) if len(rings) else (np.nan, np.nan))
                rows['overlay_n'].append(len(rings))
                if j % i4_every == 0:
                    a, b = checkpoint_ring(fr), m4b(fr)
                    rows['i4'].append(1 if a == b else 0)
                else:
                    rows['i4'].append(-1)
            k += 1
    finally:
        rd.close()
    frame = np.asarray(rows['frame'], np.int64)
    last = np.r_[frame[1:]-1, k-1]
    t_video = last/fps
    overlay = np.asarray(rows['overlay'], float).reshape(-1, 2)
    offset, info = align_offset(t_video, overlay, tel, tel.wall0)
    rec = dict(name=name, fps=fps, n_decoded=k, frame=frame, t_video=t_video, t_wall=tel.wall0+t_video+offset,
               offset_s=offset, align=json.dumps(info), old=np.asarray(rows['old'], float).reshape(-1, 3),
               new=np.asarray(rows['new'], float).reshape(-1, 3), overlay=overlay,
               overlay_n=np.asarray(rows['overlay_n'], np.int16), i4=np.asarray(rows['i4'], np.int8),
               cand_frame=np.asarray(cand['frame'], np.int64), cand_u=np.asarray(cand['u'], float),
               cand_v=np.asarray(cand['v'], float), cand_edge=np.asarray(cand['edge'], bool),
               cand_ok=np.asarray(cand['ok'], bool))
    (Path(out)/'frames').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(out)/'frames'/f'{name}.npz', **rec)
    log(json.dumps(dict(flight=name, decoded=k, new=len(frame), candidates=len(cand['u']), offset_s=round(offset, 3),
                        align=info, seconds=round(time.time()-begin, 1))), flush=True)
    return rec


def _m4b_reader():
    """The m4b tree's checkpoint_ring (the reader before the rule), from this repository's history."""
    source = subprocess.run(['git', '-C', str(REPO), 'show', 'm4b:haltere/vision/race_cues.py'], capture_output=True,
                            text=True, check=True).stdout
    namespace = {}
    exec(compile(source, 'm4b:haltere/vision/race_cues.py', 'exec'), namespace)
    return namespace['checkpoint_ring']


def epoch_offset(d):
    """The runner log's epoch offset C (wall - frame_time percentile), as haltere.obstacles.store_build.load_run_csv."""
    from .store_build import EPOCH_PERCENTILE
    wall, ft = d['wall'].to_numpy(float), d['frame_time'].to_numpy(float)
    m = np.isfinite(wall) & np.isfinite(ft)
    return float(np.percentile(wall[m]-ft[m], EPOCH_PERCENTILE))


def logged_verdicts(name, runs, frames, reader):
    """Per logged detection (unique capture with an in-image marker): the aligned recorded frame (|dt| <= match_dt_s),
    whether a reader candidate lies within tolerance_px of it there and the rule's verdict on the nearest one
    ('kept', 'rejected' or 'unmatched': no frame or no candidate), and whether the overlay detector confirms it on that
    frame (a marker within tolerance_px; None without a frame)."""
    import pandas as pd
    d = pd.read_csv(Path(runs)/f'{name}.csv', low_memory=False,
                    usecols=['wall', 'frame_time', 'capture_time', 'cue_u', 'cue_v', 'cue_edge', 'phase'])
    C = epoch_offset(d)
    ct = d.capture_time.to_numpy(float)
    first = np.r_[True, np.diff(ct) != 0] & np.isfinite(ct)
    rows = d[first]
    u, v = rows.cue_u.to_numpy(float), rows.cue_v.to_numpy(float)
    seen = np.isfinite(u) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    rows, u, v = rows[seen], u[seen], v[seen]
    t = rows.capture_time.to_numpy(float)+C
    tw = frames['t_wall']
    tol, dt_max = reader['tolerance_px'], reader['match_dt_s']
    j = np.clip(np.searchsorted(tw, t), 1, len(tw)-1)
    j = np.where(np.abs(tw[j-1]-t) < np.abs(tw[j]-t), j-1, j)
    have = np.abs(tw[j]-t) <= dt_max
    cf, cu, cv, cok = frames['cand_frame'], frames['cand_u'], frames['cand_v'], frames['cand_ok']
    starts = np.searchsorted(cf, np.arange(len(tw)+1))
    verdict, confirmed = [], []
    for i in range(len(t)):
        if not have[i]:
            verdict.append('unmatched')
            confirmed.append(None)
            continue
        a, b = starts[j[i]], starts[j[i]+1]
        dist = np.hypot((cu[a:b]-u[i])*1280, (cv[a:b]-v[i])*720)
        if len(dist) and dist.min() <= tol:
            verdict.append('kept' if cok[a+int(np.argmin(dist))] else 'rejected')
        else:
            verdict.append('unmatched')
        ov = frames['overlay'][j[i]]
        confirmed.append(bool(np.isfinite(ov).all() and np.hypot((ov[0]-u[i])*1280, (ov[1]-v[i])*720) <= tol))
    return dict(capture_time=rows.capture_time.to_numpy(float), phase=rows.phase.to_numpy(float), u=u, v=v,
                edge=rows.cue_edge.astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy(),
                verdict=np.asarray(verdict), confirmed=np.asarray(confirmed, dtype=object))


def cue_drop(out, name):
    """The logged capture times the rule rejects on the aligned recorded frame (the replays' --cue-drop list)."""
    path = Path(out)/'verdicts'/f'{name}.json'
    return json.loads(path.read_text())['rejected_capture_times']


def main_frames(args, gates):
    _limit_threads()
    rule = _rule(gates)
    names = args.flights.split(',') if args.flights else gates['reader']['flights']+gates['development']
    for name in names:
        path = Path(args.out)/'frames'/f'{name}.npz'
        if not path.exists():
            read_flight(name, gates['runs'], args.out, rule)
        vpath = Path(args.out)/'verdicts'/f'{name}.json'
        if name in gates['replay_flights'] and not vpath.exists():
            frames = dict(np.load(path))
            v = logged_verdicts(name, gates['runs'], frames, gates['reader'])
            vpath.parent.mkdir(parents=True, exist_ok=True)
            vpath.write_text(json.dumps(dict(
                flight=name, n=int(len(v['verdict'])),
                counts={k: int((v['verdict'] == k).sum()) for k in ('kept', 'rejected', 'unmatched')},
                rejected_capture_times=[float(c) for c in v['capture_time'][v['verdict'] == 'rejected']],
                rows=[dict(capture_time=float(c), phase=float(p), u=float(a), v=float(b), edge=bool(e), verdict=str(x),
                           confirmed=y) for c, p, a, b, e, x, y in zip(v['capture_time'], v['phase'], v['u'], v['v'],
                                                                       v['edge'], v['verdict'], v['confirmed'])]),
                indent=0))
            print(json.dumps(dict(verdicts=name, counts=json.loads(vpath.read_text())['counts'])), flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Open-loop replays
# ---------------------------------------------------------------------------------------------------------------
def variant_args(gates, variant, out):
    """(tree, harness arguments) of a replay variant (see the gates file's 'replays')."""
    spec = gates['replays']['variants'][variant]
    tree = gates['baseline_tree_path'] if spec['tree'] == 'm4b' else str(REPO)
    args = list(spec['args'])
    args = [a.replace('{tree}', tree).replace('{repo}', str(REPO)) for a in args]
    if spec.get('cue_drop'):
        args += ['--cue-drop', str(Path(out)/'verdicts'/'{flight}.json')]
    return tree, args


def replay_path(out, variant, flight):
    return Path(out)/'replays'/f'{variant}_{flight}.npz'


def main_replays(args, gates):
    out = Path(args.out)
    (out/'replays').mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    flights = args.flights.split(',') if args.flights else gates['replay_flights']
    streamed = set(gates['replays']['streamed'])
    variants = args.variants.split(',') if args.variants else list(gates['replays']['variants'])
    for variant in variants:
        tree, extra = variant_args(gates, variant, out)
        only = gates['replays']['variants'][variant].get('flights')
        for flight in flights:
            if only is not None and flight not in only:
                continue
            target = replay_path(out, variant, flight)
            if target.exists():
                continue
            tmp = out/'replays'/'tmp'/variant
            tmp.mkdir(parents=True, exist_ok=True)
            cmd = [PY, '-W', 'ignore', str(Path(tree)/'haltere'/'obstacles'/'vertical_replay.py'), '--tree', tree,
                   '--out', str(tmp/'r')]+extra
            if flight in streamed:
                cmd += ['--looming-stream', gates['replays']['stream']]
            cmd.append(flight)
            begin = time.time()
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f'{variant} {flight} failed: {r.stderr[-3000:]}')
            made = [p for p in tmp.glob(f'r_*_{flight}.npz')]
            if len(made) != 1:
                raise RuntimeError(f'{variant} {flight}: expected one replay file, found {made}')
            made[0].replace(target)
            print(json.dumps(dict(variant=variant, flight=flight, seconds=round(time.time()-begin, 1))), flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Semi-closed-loop windows (IdentifiedSim from the logged state, the replayed requests and yaw)
# ---------------------------------------------------------------------------------------------------------------
class Surrogate:
    def __init__(self, gates):
        import torch
        from ..train import brake_gates as bg
        self.bg = bg
        torch.set_num_threads(2)
        spec = gates['surrogate']
        if file_sha256(spec['profile']) != spec['profile_sha256']:
            raise ValueError('The dynamics profile is not the one the gates name')
        self.profile = json.loads(Path(spec['profile']).read_text())
        self.gates = gates
        self.controllers, self.flights = {}, {}

    def controller(self, motor):
        if motor not in self.controllers:
            spec = self.gates['surrogate']['motors'][motor]
            ref = self.gates['surrogate']['motors'][self.gates['surrogate']['reference']]
            for s in (spec, ref):
                if s['checkpoint'] != 'pd' and file_sha256(s['checkpoint']) != s['sha256']:
                    raise ValueError(f'{s["checkpoint"]} is not the checkpoint the gates name')
            self.controllers[motor] = self.bg.Controller('pd' if spec['checkpoint'] == 'pd' else spec['checkpoint'],
                                                         ref['checkpoint'])
        return self.controllers[motor]

    def flight(self, name):
        if name not in self.flights:
            try:
                self.flights[name] = self.bg.Flight(self.gates['runs'], name)
            except FileNotFoundError:
                # a fast-PD flight without recorded brain inputs: the PD needs only the logged states and requests
                self.flights[name] = _pd_flight(self.bg, self.gates['runs'], name)
        return self.flights[name]

    @staticmethod
    def aligned(flight, arrays):
        """(request (N, 3), yaw stick (N,)) of a replay on the CSV rows (rows the replay skipped keep the log)."""
        req, yaw = flight.req.copy(), flight.cmds[:, 3].copy()
        t = np.asarray(arrays['t'], float)
        idx = np.clip(np.searchsorted(flight.t, t), 0, len(flight.t)-1)
        ok = np.abs(flight.t[idx]-t) < 1e-6
        req[idx[ok]] = np.stack([arrays['cvx'], arrays['cvy'], arrays['cvz']], 1)[ok]
        yaw[idx[ok]] = np.clip(np.asarray(arrays['yaw_cmd'], float)[ok], -1., 1.)
        return req, yaw

    def fly(self, name, motor, t0, horizon_s, req, yaw):
        """Positions and velocities (H, 3) from the logged state at t0 with the request and yaw stick given per CSV row."""
        f = self.flight(name)
        ctl = self.controller(motor)
        saved = f.req, f.ff, f.cmds
        try:
            f.req = req
            f.ff = _feedforward(req, f.t)
            f.cmds = f.cmds.copy()
            f.cmds[:, 3] = yaw
            k0 = f.index(t0)
            H = int(round(horizon_s/self.bg.DT))
            pos, vel = self.bg.replay_window(ctl, self.profile, f, k0, H, 'controller',
                                             warm_s=self.gates['surrogate']['warm_s'])
        finally:
            f.req, f.ff, f.cmds = saved
        return pos, vel


def _pd_flight(bg, runs, name):
    """brake_gates.Flight without the recorded brain inputs (its replay npz), for fast-PD windows only."""
    import pandas as pd
    f = bg.Flight.__new__(bg.Flight)
    f.name = name
    f.d = pd.read_csv(Path(runs)/f'{name}.csv', low_memory=False)
    f.z = None
    f.t = f.d.phase.to_numpy(float)
    f.vel = f.d[['vx', 'vy', 'vz']].to_numpy(float)
    f.req = f.d[['cmd_vx', 'cmd_vy', 'cmd_vz']].to_numpy(float)
    f.cmds = f.d[['command_thr', 'command_roll', 'command_pitch', 'command_yaw']].to_numpy(float)
    f.jumps = np.flatnonzero(np.linalg.norm(np.diff(f.vel, axis=0), axis=1) > .6)
    f.ff = _feedforward(f.req, f.t)
    return f


def _feedforward(req, t):
    ff = np.zeros_like(req)
    for k in range(1, len(req)):
        dt = float(np.clip(t[k]-t[k-1], 0., .1))
        ok = np.isfinite(req[k]).all() and np.isfinite(req[k-1]).all()
        raw = (req[k]-req[k-1])/max(dt, 1e-3) if ok else 0.
        ff[k] = ff[k-1]+(1-np.exp(-dt/.05))*(raw-ff[k-1])
    return ff


def _load(out, variant, flight):
    return dict(np.load(replay_path(out, variant, flight), allow_pickle=False))


def changed_windows(a, b, *, tol=.05, merge_s=.5, pre_s=.5, post_s=1., t_min=None, t_max=None):
    """[t0, t1] windows (phase) where two replays' requests differ by more than tol m/s (horizontal or vertical),
    merged within merge_s and padded pre_s before and post_s after."""
    t = np.asarray(a['t'], float)
    diff = (np.hypot(a['cvx']-b['cvx'], a['cvy']-b['cvy']) > tol) | (np.abs(a['cvz']-b['cvz']) > tol)
    if t_min is not None:
        diff &= t >= t_min
    if t_max is not None:
        diff &= t <= t_max
    idx = np.flatnonzero(diff)
    windows = []
    for i in idx:
        if windows and t[i]-windows[-1][1] <= merge_s:
            windows[-1][1] = t[i]
        else:
            windows.append([t[i], t[i]])
    return [[max(float(t[0]), w0-pre_s), min(float(t[-1]), w1+post_s)] for w0, w1 in windows]


def path_compare(pos_ref, vel_ref, pos_new):
    """Lateral deviation (m, max over the window) of pos_new from pos_ref, and the along-track progress lost at the
    window end in seconds at the reference speed (positive = behind)."""
    lat = []
    for i in range(len(pos_ref)):
        v = vel_ref[i, :2]
        s = float(np.linalg.norm(v))
        if s < .5:
            continue
        n = np.array([-v[1], v[0]])/s
        lat.append(abs(float((pos_new[i, :2]-pos_ref[i, :2]) @ n)))
    v = vel_ref[-1, :2]
    s = max(float(np.linalg.norm(v)), .5)
    lost = -float((pos_new[-1, :2]-pos_ref[-1, :2]) @ (v/s))/s
    return (max(lat) if lat else 0.), lost


def main_windows(args, gates):
    _limit_threads()
    out = Path(args.out)
    (out/'windows').mkdir(parents=True, exist_ok=True)
    sur = Surrogate(gates)
    g = gates['gates']
    jobs = []
    ds = g['DS2_straw_path']
    jobs.append(('DS2', ds['flight'], ds['motor'], ds['t0'], ds['horizon_s'], ['m4b_on', 'new_full', 'logged']))
    for flight in g['Q_clean']['flights']:
        a, b = _load(out, 'm4b_on', flight), _load(out, 'new_full', flight)
        for w0, w1 in changed_windows(a, b, **g['Q_clean']['windows']):
            jobs.append(('Q', flight, g['Q_clean']['motor'], w0, w1-w0, ['m4b_on', 'new_full']))
    for flight, w0, w1 in pass_windows(out, gates):
        a, b = _load(out, 'm4b_on', flight), _load(out, 'new_full', flight)
        changed = changed_windows(a, b, t_min=w0, t_max=w1, pre_s=0., post_s=0.)
        if changed:
            motor = gates['surrogate']['flight_motor'].get(flight)
            if motor is not None:
                jobs.append(('HP', flight, motor, w0, w1-w0+1., ['m4b_on', 'new_full']))
    for kind, flight, motor, t0, horizon, variants in jobs:
        key = f'{kind}_{flight}_{t0:.2f}'
        path = out/'windows'/f'{key}.json'
        if path.exists():
            continue
        f = sur.flight(flight)
        res = dict(kind=kind, flight=flight, motor=motor, t0=round(t0, 3), horizon_s=round(horizon, 3), paths={})
        paths = {}
        for variant in variants:
            if variant == 'logged':
                req, yaw = f.req.copy(), f.cmds[:, 3].copy()
            else:
                req, yaw = Surrogate.aligned(f, _load(out, variant, flight))
            pos, vel = sur.fly(flight, motor, t0, horizon, req, yaw)
            paths[variant] = (pos, vel)
            res['paths'][variant] = dict(pos=np.round(pos[::5], 3).tolist(), vel=np.round(vel[::5], 3).tolist())
        if 'm4b_on' in paths and 'new_full' in paths:
            lat, lost = path_compare(*paths['m4b_on'], paths['new_full'][0])
            res.update(lateral_max_m=round(lat, 3), progress_lost_s=round(lost, 3))
        if kind == 'DS2':
            point = np.asarray(ds['contact_point'], float)
            res['min_distance_m'] = {k: round(float(np.hypot(*(p[:, :2]-point).T).min()), 3) for k, (p, _) in paths.items()}
            res['side'] = {k: side_of(p, v, point) for k, (p, v) in paths.items()}
        path.write_text(json.dumps(res))
        print(json.dumps({k: v for k, v in res.items() if k != 'paths'}), flush=True)


def side_of(pos, vel, point):
    """'left' or 'right' of the travel direction at the closest approach to `point`."""
    i = int(np.argmin(np.hypot(*(pos[:, :2]-point).T)))
    v = vel[i, :2]
    cross = float(v[0]*(point[1]-pos[i, 1])-v[1]*(point[0]-pos[i, 0]))
    return 'point on the right (path left of it)' if cross < 0 else 'point on the left (path right of it)'


def pass_windows(out, gates):
    """(flight, t0, t1) of every held-out start-arch / arch pass (hindsight positions select them)."""
    import pandas as pd
    spec = gates['gates']['HP_passes']
    res = []
    for flight in gates['replay_flights']:
        if flight in gates['development']:
            continue
        d = pd.read_csv(Path(gates['runs'])/f'{flight}.csv', low_memory=False, usecols=['phase', 'x', 'y'])
        t, x, y = d.phase.to_numpy(float), d.x.to_numpy(float), d.y.to_numpy(float)
        if flight.startswith('straw'):
            box = spec['straw_box']
            inside = (x >= box['x'][0]) & (x <= box['x'][1]) & (np.abs(y) <= box['abs_y'])
        elif flight.startswith('minus'):
            inside = np.zeros(len(t), bool)
            for px, py in spec['minus_arches']:
                inside |= np.hypot(x-px, y-py) <= spec['minus_radius_m']
        else:
            continue
        on = np.flatnonzero(inside & ~np.r_[False, inside[:-1]])
        for i in on:
            j = i
            while j+1 < len(t) and inside[j+1]:
                j += 1
            res.append((flight, float(t[i])-spec['pre_s'], float(t[j])))
    return res


# ---------------------------------------------------------------------------------------------------------------
# Closed-loop harness hairpins (the motor-assist gates' hairpin sets)
# ---------------------------------------------------------------------------------------------------------------
def main_hairpin(args, gates):
    _limit_threads()
    from ..liftoff import motor_assist_eval as mae
    from ..liftoff.descent_rehearsal import load_controller
    from ..liftoff.motor_assist_gates import pilot_kwargs
    from ..liftoff.fast_race_cue import stale_evidence_configs
    out = Path(args.out)
    (out/'hairpin').mkdir(parents=True, exist_ok=True)
    spec = gates['gates']['HA_hairpin']
    sur = gates['surrogate']
    if file_sha256(sur['profile']) != sur['profile_sha256']:
        raise ValueError('The dynamics profile is not the one the gates name')
    profile = json.loads(Path(sur['profile']).read_text())
    stale = stale_evidence_configs(json.loads(declaration_path(gates['declarations']['stale_evidence'])
                                              .read_text(encoding='utf-8')))
    for motor in spec['motors']:
        m = sur['motors'][motor]
        controller = None
        for set_name, s in spec['sets'].items():
            for variant in ('baseline', 'rule'):
                path = out/'hairpin'/f'{motor}_{set_name}_{variant}.json'
                if path.exists():
                    continue
                if controller is None:
                    ckpt = sur['motors'][sur['reference']]['checkpoint'] if m['checkpoint'] == 'pd' else m['checkpoint']
                    if file_sha256(ckpt) != (sur['motors'][sur['reference']]['sha256'] if m['checkpoint'] == 'pd'
                                             else m['sha256']):
                        raise ValueError(f'{ckpt} is not the checkpoint the gates name')
                    controller = load_controller('pd' if m['checkpoint'] == 'pd' else 'brain', ckpt)
                kw = pilot_kwargs(m['contract'])
                if variant == 'rule':
                    kw.update(stale)
                begin = time.time()
                rows, _ = mae.run_scenarios(controller, profile, mae.hairpin_set(s['set']), pilot_kwargs=kw,
                                            seconds=s['seconds'], seed=s['sim_seed'])
                path.write_text(json.dumps(dict(motor=motor, set=set_name, variant=variant, rows=rows,
                                                seconds=round(time.time()-begin, 1)), default=float))
                print(json.dumps(dict(motor=motor, set=set_name, variant=variant,
                                      wall=sum(r['wall_contact'] for r in rows),
                                      clean=sum(r['finished'] and not (r['wall_contact'] or r['floor_contact']
                                                                       or r['ceiling_contact'] or r['crashed'])
                                                for r in rows), seconds=round(time.time()-begin, 1))), flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------------
def _identical(a, b, keys=COMMAND_KEYS):
    for k in keys:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            return False
        if x.dtype.kind in 'fc' and y.dtype.kind in 'fc':
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not bool(np.all(x == y)):
            return False
    return True


def travel_speedup(a, b, min_speed=1.):
    """Per tick: (b - a) request component along the measured horizontal velocity (NaN below min_speed)."""
    v = np.stack([a['vx'], a['vy']], 1).astype(float)
    s = np.hypot(v[:, 0], v[:, 1])
    u = v/np.maximum(s, 1e-9)[:, None]
    d = (np.asarray(b['cvx'])-np.asarray(a['cvx']))*u[:, 0]+(np.asarray(b['cvy'])-np.asarray(a['cvy']))*u[:, 1]
    return np.where(s >= min_speed, d, np.nan)


def score(out, gates, digest):
    if gates['version'] >= 2:
        return score_v2(out, gates, digest)
    out = Path(out)
    g = gates['gates']
    result = dict(gates_sha256=digest, gates_version=gates['version'], declarations=gates['declarations'],
                  baseline_tree=gates['baseline_tree'])
    flights = gates['replay_flights']
    # ---- identity
    ident = {}
    for name, (left, right) in dict(I1_default=('m4b_none', 'new_none'), I2_stack_without_rule=('m4b_on', 'new_on'),
                                    I3_shadow=('m4b_shadow', 'new_shadow')).items():
        rows = {f: _identical(_load(out, left, f), _load(out, right, f)) for f in flights}
        ident[name] = dict(identical=sum(rows.values()), of=len(rows),
                           differ=[f for f, ok in rows.items() if not ok], passed=all(rows.values()))
    checked = equal = 0
    for f in gates['reader']['flights']:
        fr = np.load(out/'frames'/f'{f}.npz')
        checked += int((fr['i4'] >= 0).sum())
        equal += int((fr['i4'] == 1).sum())
    ident['I4_reader_off'] = dict(frames_checked=checked, equal=equal, passed=checked > 0 and equal == checked)
    result['identity'] = ident
    # ---- D-M: Minus clip, open loop
    dm = g['DM_minus']
    a, b = _load(out, 'm4b_on', dm['flight']), _load(out, 'new_on_se', dm['flight'])
    t = np.asarray(b['t'], float)
    sel = (t >= dm['window'][0]) & (t <= dm['impact_t'])
    su = travel_speedup(a, b)
    brake_b = sel & (np.asarray(b['braking']) > 0) if 'braking' in b else sel & False
    slower = sel & (su <= -dm['slower_mps'])
    onset = np.flatnonzero(brake_b | slower)
    base_brake = sel & (np.asarray(a['braking']) > 0)
    first = None if not len(onset) else float(t[onset[0]])
    pos = np.stack([b['x'], b['y']], 1).astype(float)
    impact = np.asarray(dm['impact_point'], float)
    dist = None if first is None else float(np.hypot(*(pos[onset[0]]-impact)))
    # the first hairpin of the same log: the flown governor's first brake and the flown drone's rest point
    ta = np.asarray(a['t'], float)
    fh = dm['first_hairpin']
    w = (ta >= fh['brake_window'][0]) & (ta <= fh['brake_window'][1]) & (np.asarray(a['braking']) > 0)
    k_brake = int(np.flatnonzero(w)[0]) if w.any() else None
    rest_w = (ta >= fh['rest_window'][0]) & (ta <= fh['rest_window'][1])
    hs = np.hypot(np.asarray(a['vx'], float), np.asarray(a['vy'], float))
    k_rest = int(np.flatnonzero(rest_w)[np.argmin(hs[rest_w])])
    pa = np.stack([a['x'], a['y']], 1).astype(float)
    ref = None if k_brake is None else float(np.hypot(*(pa[k_brake]-pa[k_rest])))
    lead = None if first is None else dm['impact_t']-first
    result['DM_minus'] = dict(
        brake_onset_t=first, lead_s=None if lead is None else round(lead, 3),
        baseline_brake_ticks_in_window=int(base_brake.sum()),
        onset_distance_to_impact_m=None if dist is None else round(dist, 3),
        first_hairpin=dict(brake_t=None if k_brake is None else float(ta[k_brake]), rest_t=float(ta[k_rest]),
                           rest_speed=round(float(hs[k_rest]), 3), brake_to_rest_m=None if ref is None else round(ref, 3)),
        min_request_along_travel_before_impact=round(float(np.nanmin(
            (np.asarray(b['cvx'])*np.asarray(b['vx'])+np.asarray(b['cvy'])*np.asarray(b['vy']))[sel]
            / np.maximum(hs[sel], 1e-9))), 3),
        reseats=int(np.nanmax(np.asarray(b['cap_reseat'], float)[sel])) if 'cap_reseat' in b else None,
        DM1_passed=bool(first is not None and lead >= dm['brake_lead_s'] and base_brake.sum() == 0),
        DM2_passed=bool(dist is not None and ref is not None and dist >= ref))
    # ---- D-S: Straw clip
    ds1 = g['DS1_straw_reader']
    ver = json.loads((out/'verdicts'/f'{ds1["flight"]}.json').read_text())
    false_rows = [r for r in ver['rows'] if ds1['phase'][0] <= r['phase'] <= ds1['phase'][1]
                  and ds1['u'][0] <= r['u'] <= ds1['u'][1] and ds1['v'][0] <= r['v'] <= ds1['v'][1]]
    result['DS1_straw_reader'] = dict(false_detections=len(false_rows),
                                      verdicts=[r['verdict'] for r in false_rows],
                                      confirmed_by_overlay=[r['confirmed'] for r in false_rows],
                                      passed=bool(false_rows) and all(r['verdict'] == 'rejected' for r in false_rows))
    ds2 = g['DS2_straw_path']
    wpath = next((out/'windows').glob(f'DS2_{ds2["flight"]}_*.json'))
    w = json.loads(wpath.read_text())
    md = w['min_distance_m']
    result['DS2_straw_path'] = dict(min_distance_m=md, side=w['side'],
                                    validation_passed=md['m4b_on'] <= ds2['validation_max_m'],
                                    passed=bool(md['m4b_on'] <= ds2['validation_max_m']
                                                and md['new_full'] >= ds2['clear_min_m']
                                                and 'path left' in w['side']['new_full']))
    # ---- H-R: reader on recorded frames
    hr = g['HR_reader']
    tol = gates['reader']['tolerance_px']
    per, pooled = {}, dict(differ=0, old_agree=0, new_agree=0)
    for f in gates['reader']['flights']:
        fr = dict(np.load(out/'frames'/f'{f}.npz'))
        old, new, ov = fr['old'], fr['new'], fr['overlay']
        has_old, has_new, has_ov = np.isfinite(old[:, 0]), np.isfinite(new[:, 0]), np.isfinite(ov[:, 0])
        near = lambda p, q: np.hypot((p[:, 0]-q[:, 0])*1280, (p[:, 1]-q[:, 1])*720) <= tol
        confirmed = has_old & has_ov & np.where(has_old & has_ov, near(old[:, :2], ov), False)
        kept = confirmed & has_new & np.where(has_new & has_old, np.all(np.nan_to_num(new) == np.nan_to_num(old), 1),
                                              False)
        differ = (has_old != has_new) | (has_old & has_new & ~np.all(np.nan_to_num(new) == np.nan_to_num(old), 1))
        agree = lambda r, has: (~has & ~has_ov) | (has & has_ov & np.where(has & has_ov, near(r[:, :2], ov), False))
        pooled['differ'] += int(differ.sum())
        pooled['old_agree'] += int((differ & agree(old, has_old)).sum())
        pooled['new_agree'] += int((differ & agree(new, has_new)).sum())
        retention = float(kept.sum()/confirmed.sum()) if confirmed.sum() else None
        per[f] = dict(frames=int(len(old)), old_markers=int(has_old.sum()), new_markers=int(has_new.sum()),
                      overlay_confirmed_old=int(confirmed.sum()), kept=int(kept.sum()),
                      retention=None if retention is None else round(retention, 5),
                      removed_unconfirmed=int((has_old & ~confirmed & ~has_new).sum()),
                      gained=int((~has_old & has_new).sum()),
                      gained_confirmed=int((~has_old & has_new & has_ov
                                            & np.where(has_new & has_ov, near(new[:, :2], ov), False)).sum()),
                      differ=int(differ.sum()),
                      passed=retention is None or retention >= hr['retention_min'])
    result['HR_reader'] = dict(per_flight=per, pooled=pooled,
                               retention_min=min((v['retention'] for v in per.values() if v['retention'] is not None),
                                                 default=None),
                               HR1_passed=all(v['passed'] for v in per.values()),
                               HR2_passed=pooled['new_agree'] >= pooled['old_agree'])
    # ---- H-A: the stale-evidence rule on every held-out log (open loop) and in the harness hairpins
    ha = g['HA_logs']
    rows = {}
    for f in flights:
        if f in gates['development']:
            continue
        a, b = _load(out, 'm4b_on', f), _load(out, 'new_on_se', f)
        su = travel_speedup(a, b)
        changed = (np.hypot(a['cvx']-b['cvx'], a['cvy']-b['cvy']) > .05) | (np.abs(a['cvz']-b['cvz']) > .05)
        rows[f] = dict(changed_ticks=int(changed.sum()), max_speedup=round(float(np.nanmax(su)), 3) if np.isfinite(
            su).any() else 0., max_slowdown=round(float(-np.nanmin(su)), 3) if np.isfinite(su).any() else 0.,
            reseats=int(np.nanmax(np.asarray(b['cap_reseat'], float))) if 'cap_reseat' in b else None,
            passed=not np.isfinite(su).any() or float(np.nanmax(su)) <= ha['max_speedup'])
    result['HA_logs'] = dict(per_flight=rows, passed=all(r['passed'] for r in rows.values()))
    hp = g['HA_hairpin']
    table = {}
    ok = True
    for motor in hp['motors']:
        for set_name in hp['sets']:
            counts = {}
            for variant in ('baseline', 'rule'):
                rs = json.loads((out/'hairpin'/f'{motor}_{set_name}_{variant}.json').read_text())['rows']
                counts[variant] = dict(wall=sum(r['wall_contact'] for r in rs), floor=sum(r['floor_contact'] for r in rs),
                                       ceiling=sum(r['ceiling_contact'] for r in rs), crashed=sum(r['crashed'] for r in rs),
                                       clean=sum(r['finished'] and not (r['wall_contact'] or r['floor_contact']
                                                                        or r['ceiling_contact'] or r['crashed'])
                                                 for r in rs),
                                       reseat_scenarios=None)
            passed = (counts['rule']['wall'] <= counts['baseline']['wall']
                      and counts['rule']['clean'] >= counts['baseline']['clean'])
            ok &= passed
            table[f'{motor}/{set_name}'] = dict(counts, passed=passed)
    result['HA_hairpin'] = dict(table=table, passed=ok)
    # ---- H-P: held-out passes
    hpp = g['HP_passes']
    passes = []
    for flight, w0, w1 in pass_windows(out, gates):
        ver = json.loads((out/'verdicts'/f'{flight}.json').read_text())
        rows_w = [r for r in ver['rows'] if w0 <= r['phase'] <= w1]
        dropped = [r for r in rows_w if r['verdict'] == 'rejected']
        a, b, c = _load(out, 'm4b_on', flight), _load(out, 'new_on_se', flight), _load(out, 'new_full', flight)
        tt = np.asarray(a['t'], float)
        sel = (tt >= w0) & (tt <= w1)
        su = travel_speedup(a, b)[sel]
        same = all(np.array_equal(np.asarray(a[k])[sel], np.asarray(c[k])[sel], equal_nan=True)
                   for k in ('cvx', 'cvy', 'cvz'))
        win = [p for p in (out/'windows').glob(f'HP_{flight}_*.json')
               if abs(json.loads(p.read_text())['t0']-round(w0, 3)) < 1e-3]
        entry = dict(flight=flight, window=[round(w0, 2), round(w1, 2)], requests_unchanged=bool(same),
                     detections=len(rows_w), dropped=len(dropped),
                     dropped_confirmed=sum(bool(r['confirmed']) for r in dropped),
                     max_speedup_rule_A=round(float(np.nanmax(su)), 3) if np.isfinite(su).any() else 0.)
        if win:
            wj = json.loads(win[0].read_text())
            entry.update(surrogate_lateral_max_m=wj.get('lateral_max_m'), surrogate_progress_lost_s=wj.get(
                'progress_lost_s'))
        entry['passed'] = entry['dropped_confirmed'] == 0 and entry['max_speedup_rule_A'] <= ha['max_speedup']
        passes.append(entry)
    result['HP_passes'] = dict(passes=passes, n=len(passes), unchanged=sum(p['requests_unchanged'] for p in passes),
                               passed=all(p['passed'] for p in passes))
    # ---- Q: clean Straw Bale laps
    q = g['Q_clean']
    qrows = {}
    for f in q['flights']:
        ver = json.loads((out/'verdicts'/f'{f}.json').read_text())
        n = len(ver['rows'])
        dropped = [r for r in ver['rows'] if r['verdict'] == 'rejected']
        confirmed = [r for r in ver['rows'] if r['confirmed']]
        wins = [json.loads(p.read_text()) for p in (out/'windows').glob(f'Q_{f}_*.json')]
        lost = sum(max(0., w['progress_lost_s']) for w in wins)
        lat = max((w['lateral_max_m'] for w in wins), default=0.)
        entry = dict(detections=n, dropped=len(dropped), dropped_fraction=round(len(dropped)/max(n, 1), 5),
                     confirmed=len(confirmed), dropped_confirmed=sum(bool(r['confirmed']) for r in dropped),
                     windows=len(wins), progress_lost_s=round(lost, 3), lateral_max_m=round(lat, 3))
        entry['passed'] = (entry['dropped_fraction'] <= q['max_dropped_fraction']
                           and entry['dropped_confirmed'] <= q['max_dropped_confirmed_fraction']*max(len(confirmed), 1)
                           and lost <= q['max_progress_lost_s'] and lat <= q['max_lateral_m'])
        qrows[f] = entry
    result['Q_clean'] = dict(per_flight=qrows, passed=all(r['passed'] for r in qrows.values()))
    return result


def _hairpin_counts(rows):
    return dict(wall=sum(r['wall_contact'] for r in rows), floor=sum(r['floor_contact'] for r in rows),
                ceiling=sum(r['ceiling_contact'] for r in rows), crashed=sum(r['crashed'] for r in rows),
                clean=sum(r['finished'] and not (r['wall_contact'] or r['floor_contact'] or r['ceiling_contact']
                                                 or r['crashed']) for r in rows),
                finish_s_mean=round(float(np.mean([r['finish_s'] for r in rows if r['finished']])), 3)
                if any(r['finished'] for r in rows) else None)


def score_v2(out, gates, digest):
    """Gates version 2 (the stale-evidence rule version 2): identity, the kept version 1 reproduced, the Minus
    development case, the development logs and hairpin sets, and the fresh held-out hairpin set."""
    out = Path(out)
    g = gates['gates']
    result = dict(gates_sha256=digest, gates_version=gates['version'], declarations=gates['declarations'],
                  baseline_tree=gates['baseline_tree'])
    flights = gates['replay_flights']
    ident = {}
    for name, (left, right) in dict(I1_default=('m4b_none', 'new_none'), I2_stack_without_rule=('m4b_on', 'new_on'),
                                    I3_shadow=('m4b_shadow', 'new_shadow2')).items():
        rows = {f: _identical(_load(out, left, f), _load(out, right, f)) for f in flights}
        ident[name] = dict(identical=sum(rows.values()), of=len(rows),
                           differ=[f for f, ok in rows.items() if not ok], passed=all(rows.values()))
    result['identity'] = ident
    v1 = g['V1_reproduced']
    rows = {}
    for f in v1['flights']:
        a = dict(np.load(Path(v1['v1_replays'])/f'new_on_se_{f}.npz'))
        b = _load(out, 'new_on_se1', f)
        rows[f] = _identical(a, b, tuple(COMMAND_KEYS)+('cap_reseat',))
    result['V1_reproduced'] = dict(rows=rows, passed=all(rows.values()))
    dm = g['DM_minus']
    a, b = _load(out, 'm4b_on', dm['flight']), _load(out, 'new_on_se2', dm['flight'])
    t = np.asarray(b['t'], float)
    sel = (t >= dm['window'][0]) & (t <= dm['impact_t'])
    su = travel_speedup(a, b)
    onset = np.flatnonzero(sel & ((np.asarray(b['braking']) > 0) | (su <= -dm['slower_mps'])))
    base_brake = sel & (np.asarray(a['braking']) > 0)
    first = None if not len(onset) else float(t[onset[0]])
    pos = np.stack([b['x'], b['y']], 1).astype(float)
    dist = None if first is None else float(np.hypot(*(pos[onset[0]]-np.asarray(dm['impact_point'], float))))
    ta = np.asarray(a['t'], float)
    fh = dm['first_hairpin']
    w = (ta >= fh['brake_window'][0]) & (ta <= fh['brake_window'][1]) & (np.asarray(a['braking']) > 0)
    rest_w = (ta >= fh['rest_window'][0]) & (ta <= fh['rest_window'][1])
    hs = np.hypot(np.asarray(a['vx'], float), np.asarray(a['vy'], float))
    k_rest = int(np.flatnonzero(rest_w)[np.argmin(hs[rest_w])])
    pa = np.stack([a['x'], a['y']], 1).astype(float)
    k_brake = int(np.flatnonzero(w)[0]) if w.any() else None
    ref = None if k_brake is None else float(np.hypot(*(pa[k_brake]-pa[k_rest])))
    lead = None if first is None else dm['impact_t']-first
    along = (np.asarray(b['cvx'])*np.asarray(b['vx'])+np.asarray(b['cvy'])*np.asarray(b['vy']))/np.maximum(hs, 1e-9)
    result['DM_minus'] = dict(
        brake_onset_t=first, lead_s=None if lead is None else round(lead, 3),
        baseline_brake_ticks_in_window=int(base_brake.sum()),
        onset_distance_to_impact_m=None if dist is None else round(dist, 3), first_hairpin_brake_to_rest_m=None
        if ref is None else round(ref, 3),
        request_along_travel_at_impact=round(float(along[sel][-1]), 3),
        min_request_along_travel_before_impact=round(float(np.nanmin(along[sel])), 3),
        reseats=int(np.nanmax(np.asarray(b['cap_reseat'], float)[sel])),
        DM1_passed=bool(first is not None and lead >= dm['brake_lead_s'] and base_brake.sum() == 0),
        DM2_passed=bool(dist is not None and ref is not None and dist >= ref))
    ha = g['HA_logs']
    rows = {}
    for f in ha['flights']:
        a, b = _load(out, 'm4b_on', f), _load(out, 'new_on_se2', f)
        su = travel_speedup(a, b)
        changed = (np.hypot(a['cvx']-b['cvx'], a['cvy']-b['cvy']) > .05) | (np.abs(a['cvz']-b['cvz']) > .05)
        finite = np.isfinite(su).any()
        rows[f] = dict(changed_ticks=int(changed.sum()), max_speedup=round(float(np.nanmax(su)), 3) if finite else 0.,
                       max_slowdown=round(float(-np.nanmin(su)), 3) if finite else 0.,
                       reseats=int(np.nanmax(np.asarray(b['cap_reseat'], float))),
                       passed=not finite or float(np.nanmax(su)) <= ha['max_speedup'])
    result['HA_logs'] = dict(per_flight=rows, passed=all(r['passed'] for r in rows.values()))
    hp = g['HA_hairpin']
    table = {}
    for motor in hp['motors']:
        for set_name in hp['sets']:
            counts = {v: _hairpin_counts(json.loads((out/'hairpin'/f'{motor}_{set_name}_{v}.json').read_text())['rows'])
                      for v in ('baseline', 'rule')}
            table[f'{motor}/{set_name}'] = dict(counts, passed=(counts['rule']['wall'] <= counts['baseline']['wall']
                                                                and counts['rule']['clean'] >= counts['baseline']['clean']))
    gated = [k for k in table if k.split('/')[1] in hp['gated_sets']]
    result['HA_hairpin'] = dict(table=table, gated=gated, passed=all(table[k]['passed'] for k in gated),
                                development_passed=all(v['passed'] for k, v in table.items() if k not in gated))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['frames', 'replays', 'windows', 'hairpin', 'score'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--gates', default=str(GATES_PATH))
    parser.add_argument('--flights', default=None)
    parser.add_argument('--variants', default=None)
    parser.add_argument('--json', default=None)
    args = parser.parse_args(argv)
    gates, digest = load_gates(args.gates)
    if args.command == 'frames':
        main_frames(args, gates)
    elif args.command == 'replays':
        main_replays(args, gates)
    elif args.command == 'windows':
        main_windows(args, gates)
    elif args.command == 'hairpin':
        main_hairpin(args, gates)
    else:
        result = score(args.out, gates, digest)
        text = json.dumps(result, indent=1, default=float)
        if args.json:
            Path(args.json).write_text(text+'\n', encoding='utf-8')
        print(text)


if __name__ == '__main__':
    main()
