"""Offline gates of the near-ring lead (round 7, in-gate turns): configs/pilot/ring_lead.json, the lag-aware turn leads
the ring bearing while the in-view ring's line of sight swings. The gates are declared in configs/pilot/ring_lead_gates.json
and scored only when that file is frozen and names the frozen declaration. Development and held-out data are as the gates
file declares. Nothing here is flight evidence: replays are open loop (the recorded motion does not respond), windows are
semi-closed loop in the IdentifiedSim surrogate (the logged state, the replayed requests), and the harness scenarios are
synthetic.

Hindsight quantities (impact times, contact points, the ring centres triangulated from a log's own in-view marker rays,
the pillar box) select and score windows only; none of them reaches the pilot.

usage (one process, CPU, two threads; resumable: finished outputs are kept):
  python -m haltere.obstacles.ring_lead_gates replays --out DIR [--flights a,b]   open-loop replays (both trees)
  python -m haltere.obstacles.ring_lead_gates windows --out DIR                   semi-closed-loop windows
  python -m haltere.obstacles.ring_lead_gates harness --out DIR                   closed-loop harness sets
  python -m haltere.obstacles.ring_lead_gates score   --out DIR --json SCORES.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .stale_evidence_gates import Surrogate, changed_windows, content_sha256, file_sha256, path_compare, side_of

REPO = Path(__file__).resolve().parents[2]
GATES_PATH = REPO/'configs'/'pilot'/'ring_lead_gates.json'
PY = sys.executable
COMMAND_KEYS = ('cvx', 'cvy', 'cvz', 'yaw_cmd', 'state', 'cap', 'climb', 'descent_scale')
TREES = ('m6', 'repo')


def load_gates(path=GATES_PATH):
    """The frozen gates and their hash; refuses an unfrozen or edited file and a declaration other than the one it
    names (version and content hash)."""
    gates = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = content_sha256(gates)
    if gates.get('frozen') is not True or gates.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    for spec in gates['declarations'].values():
        declaration = json.loads((REPO/spec['file']).read_text(encoding='utf-8'))
        if (declaration.get('frozen') is not True or declaration.get('version') != spec['version']
                or declaration.get('sha256') != spec['sha256'] or content_sha256(declaration) != spec['sha256']):
            raise ValueError(f'{spec["file"]} is not the frozen declaration these gates score')
    return gates, digest


def _limit_threads():
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(k, '2')
    import torch
    torch.set_num_threads(2)


def replayable_ticks(runs, flight):
    """Rows the replay harness feeds to the pilot (finite capture time and image age)."""
    import pandas as pd
    d = pd.read_csv(Path(runs)/f'{flight}.csv', low_memory=False, usecols=['capture_time', 'image_age'])
    return int((np.isfinite(d.capture_time.to_numpy(float)) & np.isfinite(d.image_age.to_numpy(float))).sum())


# ---------------------------------------------------------------------------------------------------------------
# Open-loop replays
# ---------------------------------------------------------------------------------------------------------------
def replay_path(out, tree, base, variant, flight):
    return Path(out)/'replays'/f'{tree}_{base}_{variant}_{flight}.npz'


def variant_jobs(gates):
    """(tree name, base, variant) of every replay: the m6 tree runs the bases only (it has no rule)."""
    jobs = []
    for base in gates['replays']['bases']:
        jobs.append(('m6', base, 'base'))
        for variant in gates['replays']['variants']:
            jobs.append(('repo', base, variant))
    return jobs


def main_replays(args, gates):
    out = Path(args.out)
    (out/'replays').mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    flights = args.flights.split(',') if args.flights else gates['flights']
    streamed = set(gates['replays']['streamed'])
    skipped_path = out/'replays'/'skipped.json'
    skipped = json.loads(skipped_path.read_text(encoding='utf-8')) if skipped_path.exists() else {}
    jobs = variant_jobs(gates)
    for tree_name, base, variant in jobs:
        tree = gates['baseline_tree_path'] if tree_name == 'm6' else str(REPO)
        extra = [a.replace('{tree}', tree) for a in gates['replays']['bases'][base]+gates['replays']['variants'][variant]]
        for flight in flights:
            target = replay_path(out, tree_name, base, variant, flight)
            if target.exists() or flight in skipped:
                continue
            if replayable_ticks(gates['runs'], flight) == 0:
                skipped[flight] = 'no replayable tick'
                skipped_path.write_text(json.dumps(skipped, indent=1), encoding='utf-8')
                continue
            tmp = out/'replays'/'tmp'/f'{tree_name}_{base}_{variant}'
            tmp.mkdir(parents=True, exist_ok=True)
            cmd = [PY, '-W', 'ignore', str(Path(tree)/'haltere'/'obstacles'/'vertical_replay.py'), '--tree', tree,
                   '--runs', gates['runs'], '--out', str(tmp/'r')]+extra
            if base == 'stack' and flight in streamed:
                cmd += ['--looming-stream', gates['replays']['stream']]
            cmd.append(flight)
            begin = time.time()
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if r.returncode and (tree_name, base, variant) == jobs[0]:
                # the m6 tree's own harness cannot replay this log: reported, scores nothing, like a log without ticks
                skipped[flight] = 'the m6 harness cannot replay it: '+r.stderr.strip().splitlines()[-1][:300]
                skipped_path.write_text(json.dumps(skipped, indent=1), encoding='utf-8')
                print(json.dumps(dict(skipped=flight, reason=skipped[flight])), flush=True)
                continue
            if r.returncode:
                raise RuntimeError(f'{tree_name} {base} {variant} {flight} failed: {r.stderr[-3000:]}')
            made = [p for p in tmp.glob(f'r_*_{flight}.npz')]
            if len(made) != 1:
                raise RuntimeError(f'{tree_name} {base} {variant} {flight}: expected one replay file, found {made}')
            made[0].replace(target)
            for s in tmp.glob('r_*_summary.json'):
                meta = json.loads(s.read_text(encoding='utf-8')).get(flight, {}).get('ring_lead_metadata')
                if meta is not None:
                    target.with_suffix('.ring.json').write_text(json.dumps(meta), encoding='utf-8')
                s.unlink()
            print(json.dumps(dict(tree=tree_name, base=base, variant=variant, flight=flight,
                                  seconds=round(time.time()-begin, 1))), flush=True)


def _load(out, tree, base, variant, flight):
    return dict(np.load(replay_path(out, tree, base, variant, flight), allow_pickle=False))


def identical(a, b, keys=COMMAND_KEYS):
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


def windows_of(gates, out, base, flight):
    """Windows (padded, phase) where the rule's requests differ from the base's."""
    s = gates['surrogate']
    a, b = _load(out, 'repo', base, 'base', flight), _load(out, 'repo', base, 'rule', flight)
    return changed_windows(a, b, pre_s=s['pre_s'], post_s=s['post_s'])


def motor_of(gates, flight):
    side = json.loads((Path(gates['runs'])/f'{flight}.json').read_text(encoding='utf-8'))
    if (side.get('motor_controller') or {}).get('contract') == 'fast_velocity_pd_v1':
        return 'pd'
    sha = str(side.get('checkpoint_sha256', ''))[:8]
    for name, spec in gates['surrogate']['motors'].items():
        if spec['checkpoint'] != 'pd' and spec['sha256'][:8] == sha:
            return name
    return None


# ---------------------------------------------------------------------------------------------------------------
# Hindsight ring centres (scoring only): the log's own in-view marker rays, triangulated per ring
# ---------------------------------------------------------------------------------------------------------------
def capture_rays(runs, flight, sensor=None):
    """In-view (not edge-clamped) ring-centre rays of every logged capture: arrays t (phase of the capture: the first
    row carrying it minus its image age), p (N, 3) the drone position at the capture and d (N, 3) the world unit ray
    (camera calibration of the sidecar, the pose at capture from the logged telemetry as the pilot's pose history)."""
    import pandas as pd
    from ..liftoff.camera_pose import CameraPoseHistory
    from ..vision.camera import Camera, quat_wxyz_to_mat
    side = json.loads((Path(runs)/f'{flight}.json').read_text(encoding='utf-8'))
    sensor = sensor or side['gate_sensor']
    camera = Camera(320, 180, sensor['focal_320'], sensor['tilt_deg'])
    cols = ['ts', 'phase', 'frame_time', 'capture_time', 'image_age', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'cue_u',
            'cue_v', 'cue_edge']
    d = pd.read_csv(Path(runs)/f'{flight}.csv', low_memory=False, usecols=cols)
    history = CameraPoseHistory()
    last_ts = last_capture = None
    t, p, rays = [], [], []
    for r in d.itertuples(index=False):
        if r.ts != last_ts and np.isfinite(r.frame_time):
            history.append(float(r.frame_time), np.array([r.x, r.y, r.z], float), np.array([r.qw, r.qx, r.qy, r.qz]))
            last_ts = r.ts
        if not (np.isfinite(r.capture_time) and np.isfinite(r.image_age)) or r.capture_time == last_capture:
            continue
        last_capture = r.capture_time
        if not (np.isfinite(r.cue_u) and 0 <= r.cue_u <= 1) or str(r.cue_edge).strip().lower() in ('true', '1', '1.0'):
            continue
        try:
            position, q = history.at(float(r.capture_time))
        except Exception:
            continue
        if position is None or q is None:
            continue
        ray = quat_wxyz_to_mat(np.asarray(q, float)) @ camera.unproject_body(np.array([[r.cue_u*320, r.cue_v*180]]))[0]
        t.append(float(r.phase)-float(r.image_age))
        p.append(np.asarray(position, float))
        rays.append(ray/max(np.linalg.norm(ray), 1e-9))
    return np.asarray(t), np.asarray(p).reshape(-1, 3), np.asarray(rays).reshape(-1, 3)


def triangulate(p, d):
    """Least-squares point nearest to the lines (p_i, d_i): (point, smallest eigenvalue of sum(I - d d^T), metric RMS
    residual)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for pi, di in zip(p, d):
        M = np.eye(3)-np.outer(di, di)
        A += M
        b += M @ pi
    x = np.linalg.solve(A, b)
    res = [np.linalg.norm((x-pi)-((x-pi) @ di)*di) for pi, di in zip(p, d)]
    return x, float(np.linalg.eigvalsh(A)[0]), float(np.sqrt(np.mean(np.square(res))))


def ring_segments(t, d, spec):
    """Index ranges [a, b) of a log's in-view readings that belong to one ring (hindsight): consecutive readings split
    where the ring-centre azimuth jumps by more than jump_deg or the readings pause for more than gap_s. A switch to a
    ring at a similar bearing does not split a segment; the fit of a window (ring_fit) then fails its residual test."""
    if len(t) == 0:
        return []
    az = np.arctan2(d[:, 1], d[:, 0])
    cuts = [0]
    for i in range(1, len(t)):
        jump = abs((az[i]-az[i-1]+np.pi) % (2*np.pi)-np.pi)
        if jump > np.radians(spec['jump_deg']) or t[i]-t[i-1] > spec['gap_s']:
            cuts.append(i)
    cuts.append(len(t))
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def ring_fit(t, p, d, keep, spec):
    """The ring of the readings `keep` (indices): triangulated when there are at least min_rays; valid when the smallest
    eigenvalue is at least min_eig, the RMS residual at most max_rms_m and the point lies ahead of the last reading
    along its ray."""
    row = dict(t0=round(float(t[keep[0]]), 2), t1=round(float(t[keep[-1]]), 2), rays=int(len(keep)), valid=False)
    if len(keep) >= spec['min_rays']:
        x, eig, rms = triangulate(p[keep], d[keep])
        ahead = float((x-p[keep[-1]]) @ d[keep[-1]])
        row.update(centre=[round(float(v), 3) for v in x], eig=round(eig, 4), rms=round(rms, 4), ahead_m=round(ahead, 3),
                   valid=bool(eig >= spec['min_eig'] and rms <= spec['max_rms_m'] and ahead > 0))
    return row


# ---------------------------------------------------------------------------------------------------------------
# Semi-closed-loop windows
# ---------------------------------------------------------------------------------------------------------------
def _fly_capped(sur, flight_name, motor, t0, horizon, req, yaw):
    """Surrogate.fly with the horizon cut at the flight's first velocity jump after t0 (the impact)."""
    f = sur.flight(flight_name)
    k0 = f.index(t0)
    cut = f.impact_after(k0)
    horizon = min(horizon, max(.02, float(f.t[min(cut, len(f.t)-1)]-f.t[k0])))
    return sur.fly(flight_name, motor, t0, horizon, req, yaw)


def held_out(gates, flight):
    return flight not in gates['development_logs']


def window_jobs(gates, out):
    jobs = []
    for key in ('D1_floor_arch', 'D2_fat_shark', 'D4_hairpin_approach'):
        g = gates['gates'][key]
        jobs.append((key, g['flight'], g['motor'], g['t0'], g['horizon_s']))
    for flight in gates['flights']:
        if not held_out(gates, flight) or not replay_path(out, 'repo', 'stack', 'rule', flight).exists():
            continue
        motor = motor_of(gates, flight)
        if motor is None:
            continue
        for w0, w1 in windows_of(gates, out, 'stack', flight):
            jobs.append(('H', flight, motor, w0, w1-w0))
    return jobs


def main_windows(args, gates):
    _limit_threads()
    out = Path(args.out)
    (out/'windows').mkdir(parents=True, exist_ok=True)
    sur = Surrogate(gates)
    for kind, flight, motor, t0, horizon in window_jobs(gates, out):
        key = f'{kind}_{flight}_{t0:.2f}'
        path = out/'windows'/f'{key}.json'
        if path.exists():
            continue
        begin = time.time()
        f = sur.flight(flight)
        paths = {}
        res = dict(kind=kind, flight=flight, motor=motor, t0=round(t0, 3), horizon_s=round(horizon, 3))
        for variant in ('base', 'rule'):
            req, yaw = Surrogate.aligned(f, _load(out, 'repo', 'stack', variant, flight))
            paths[variant] = _fly_capped(sur, flight, motor, t0, horizon, req, yaw)
        if kind != 'H':
            g = gates['gates'][kind]
            req, yaw = f.req.copy(), f.cmds[:, 3].copy()
            paths['logged'] = _fly_capped(sur, flight, motor, t0, horizon, req, yaw)
            point = np.asarray(g['contact_point'], float)
            res['min_distance_m'] = {k: round(float(np.hypot(*(q[:, :2]-point).T).min()), 3)
                                     for k, (q, _) in paths.items()}
            res['side'] = {k: side_of(q, v, point) for k, (q, v) in paths.items()}
            if g.get('ring_centre') is not None:
                ring = np.asarray(g['ring_centre'], float)[:2]
                res['ring_distance_m'] = {k: round(float(np.hypot(*(q[:, :2]-ring).T).min()), 3)
                                          for k, (q, _) in paths.items()}
        lat, lost = path_compare(*paths['base'], paths['rule'][0])
        res.update(lateral_max_m=round(lat, 3), progress_lost_s=round(lost, 3), ticks=int(len(paths['base'][0])),
                   paths={k: dict(pos=np.round(q, 3).tolist()) for k, (q, _) in paths.items()},
                   seconds=round(time.time()-begin, 1))
        path.write_text(json.dumps(res), encoding='utf-8')
        print(json.dumps({k: v for k, v in res.items() if k != 'paths'}), flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Closed-loop harness sets
# ---------------------------------------------------------------------------------------------------------------
def main_harness(args, gates):
    _limit_threads()
    from ..liftoff import motor_assist_eval as mae
    from ..liftoff.descent_rehearsal import load_controller
    from ..train.deployed_pilot import deployed_pilot_kwargs
    out = Path(args.out)
    (out/'harness').mkdir(parents=True, exist_ok=True)
    h = gates['harness']
    sur = gates['surrogate']
    if file_sha256(sur['profile']) != sur['profile_sha256']:
        raise ValueError('The dynamics profile is not the one the gates name')
    profile = json.loads(Path(sur['profile']).read_text(encoding='utf-8'))
    builders = dict(gate=mae.gate_set, hairpin=mae.hairpin_set, passthrough=mae.passthrough_set)
    for motor, m in h['motors'].items():
        controller = None
        for set_name, spec in h['sets'].items():
            for variant in ('base', 'rule'):
                path = out/'harness'/f'{motor}_{set_name}_{variant}.json'
                if path.exists():
                    continue
                if controller is None:
                    if file_sha256(m['checkpoint']) != m['sha256']:
                        raise ValueError(f'{m["checkpoint"]} is not the checkpoint the gates name')
                    controller = load_controller(m['kind'], m['checkpoint'])
                kw, record = deployed_pilot_kwargs(m['contract'], **h['pilot'],
                                                   **(dict(ring_lead='on') if variant == 'rule' else {}))
                begin = time.time()
                rows, _ = mae.run_scenarios(controller, profile, builders[spec['builder']](spec['spec']),
                                            pilot_kwargs=kw, seconds=spec['seconds'], seed=spec['sim_seed'],
                                            live_wall_samples=spec.get('live_wall_samples', False))
                path.write_text(json.dumps(dict(motor=motor, set=set_name, variant=variant, rows=rows,
                                                declarations=record, seconds=round(time.time()-begin, 1)),
                                           default=float), encoding='utf-8')
                print(json.dumps(dict(motor=motor, set=set_name, variant=variant, seconds=round(time.time()-begin, 1))),
                      flush=True)


def _clean(r):
    return bool(r['finished'] and not (r['wall_contact'] or r['floor_contact'] or r['ceiling_contact'] or r['crashed']
                                      or r.get('post_contact', False)))


# ---------------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------------
def _request_travel(a, t):
    dt = np.clip(np.diff(t, prepend=t[0]), 0., .1)
    return float(np.sum(np.hypot(a['cvx'], a['cvy'])*dt))


def _box_distance(xy, box):
    """Horizontal distance (m) of points (N, 2) from an axis-aligned box {x: [lo, hi], y: [lo, hi]} (0 inside)."""
    dx = np.maximum(0., np.maximum(box['x'][0]-xy[:, 0], xy[:, 0]-box['x'][1]))
    dy = np.maximum(0., np.maximum(box['y'][0]-xy[:, 1], xy[:, 1]-box['y'][1]))
    return np.hypot(dx, dy)


def ring_clearance(gates, flight, window, paths, rings_cache):
    """H2 rows of one held-out window: every ring segment with readings from lookback_s before the window to its end,
    triangulated from those readings (ring_fit); for a valid ring, both surrogate paths' closest horizontal approach to
    its centre over the window."""
    spec = gates['gates']['H2_ring_clearance']
    if flight not in rings_cache:
        t, p, d = capture_rays(gates['runs'], flight)
        rings_cache[flight] = (t, p, d, ring_segments(t, d, spec['segments']))
    t, p, d, segments = rings_cache[flight]
    w0, w1 = window
    rows = []
    for a, b in segments:
        idx = np.arange(a, b)
        keep = idx[(t[idx] >= w0-spec['lookback_s']) & (t[idx] <= w1)]
        if len(keep) == 0:
            continue
        row = ring_fit(t, p, d, keep, spec['segments'])
        if row['valid']:
            c = np.asarray(row['centre'][:2], float)
            for k in ('base', 'rule'):
                q = np.asarray(paths[k]['pos'], float)
                dist = np.hypot(*(q[:, :2]-c).T)
                i = int(np.argmin(dist))
                row[f'{k}_m'] = round(float(dist[i]), 3)
                row[f'{k}_reached'] = bool(i < len(q)-1)
            row['passed'] = bool(row['rule_m'] <= row['base_m']+spec['tolerance_m'])
        rows.append(row)
    return rows


def score(out, gates, digest):
    out = Path(out)
    g = gates['gates']
    result = dict(gates_sha256=digest, gates_version=gates['version'],
                  declaration=gates['declarations']['ring_lead'], table={})
    table = result['table']
    skipped = {}
    sk = out/'replays'/'skipped.json'
    if sk.exists():
        skipped = json.loads(sk.read_text(encoding='utf-8'))
    flights = [f for f in gates['flights'] if f not in skipped]
    held = [f for f in flights if held_out(gates, f)]
    result['flights'] = dict(scored=len(flights), held_out=len(held), skipped=skipped)
    # identities
    for name, (ta, va), (tb, vb) in (('I1_default_off', ('m6', 'base'), ('repo', 'base')),
                                     ('I2_shadow', ('repo', 'base'), ('repo', 'shadow'))):
        rows, same = {}, 0
        for base in gates['replays']['bases']:
            for flight in flights:
                ok = identical(_load(out, ta, base, va, flight), _load(out, tb, base, vb, flight))
                same += ok
                if not ok:
                    rows.setdefault(base, []).append(flight)
        n = len(flights)*len(gates['replays']['bases'])
        table[name] = dict(identical=f'{same}/{n}', different=rows, passed=same == n, held_out=True)
    # activity (report): rule seconds, episodes and changed windows per flight
    activity = {}
    for flight in flights:
        meta_path = replay_path(out, 'repo', 'stack', 'rule', flight).with_suffix('.ring.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else None
        wins = windows_of(gates, out, 'stack', flight)
        if (meta and meta.get('seconds')) or wins:
            activity[flight] = dict(seconds=None if meta is None else meta['seconds'],
                                    episodes=None if meta is None else meta['counts']['episodes'],
                                    windows=[[round(a, 2), round(b, 2)] for a, b in wins],
                                    held_out=held_out(gates, flight))
    result['activity'] = activity

    def window(kind, flight, t0):
        return json.loads((out/'windows'/f'{kind}_{flight}_{t0:.2f}.json').read_text(encoding='utf-8'))
    # D1: the Minus floor arch
    d1 = g['D1_floor_arch']
    w = window('D1_floor_arch', d1['flight'], d1['t0'])
    md, rd = w['min_distance_m'], w['ring_distance_m']
    table['D1_floor_arch'] = dict(min_distance_m=md, ring_distance_m=rd, side=w['side'],
                                  passed=bool(md['base'] <= d1['validation_max_m'] and md['rule'] >= d1['clear_min_m']
                                              and rd['rule'] < rd['base']), held_out=False)
    # D2: FAT SHARK (not worse)
    d2 = g['D2_fat_shark']
    w = window('D2_fat_shark', d2['flight'], d2['t0'])
    md = w['min_distance_m']
    table['D2_fat_shark'] = dict(min_distance_m=md, ring_distance_m=w.get('ring_distance_m'), side=w['side'],
                                 passed=bool(md['rule'] >= md['base']-1e-9), held_out=False)
    # D3: the fast PD's arch bump (no entry: its requests unchanged)
    d3 = g['D3_pd_arch_bump']
    same = identical(_load(out, 'repo', 'stack', 'base', d3['flight']), _load(out, 'repo', 'stack', 'rule', d3['flight']))
    table['D3_pd_arch_bump'] = dict(identical=bool(same), passed=bool(same), held_out=False)
    # D4: the hairpin approach of the Minus development log (the arch before the hairpin not moved off its centre)
    d4 = g['D4_hairpin_approach']
    w = window('D4_hairpin_approach', d4['flight'], d4['t0'])
    md, rd, tol = w['min_distance_m'], w['ring_distance_m'], d4['tolerance_m']
    table['D4_hairpin_approach'] = dict(min_distance_m=md, ring_distance_m=rd,
                                        passed=bool(rd['rule'] <= rd['base']+tol and md['rule'] >= md['base']-tol),
                                        held_out=False)
    # H1: no new stop (held-out logs, both bases)
    stop = g['H1_no_new_stop']['stop_mps']
    new_stops, n_windows = [], {}
    for base in gates['replays']['bases']:
        n = 0
        for flight in held:
            wins = windows_of(gates, out, base, flight)
            if not wins:
                continue
            a, b = _load(out, 'repo', base, 'base', flight), _load(out, 'repo', base, 'rule', flight)
            t = np.asarray(a['t'], float)
            for w0, w1 in wins:
                n += 1
                m = (t >= w0) & (t <= w1)
                hb, hr = np.hypot(a['cvx'], a['cvy'])[m], np.hypot(b['cvx'], b['cvy'])[m]
                bad = (hr < stop) & (hb >= stop)
                if bad.any():
                    new_stops.append(dict(base=base, flight=flight, window=[round(w0, 2), round(w1, 2)],
                                          ticks=int(bad.sum()), lowest_rule=round(float(hr.min()), 3),
                                          lowest_base=round(float(hb.min()), 3)))
        n_windows[base] = n
    table['H1_no_new_stop'] = dict(held_out_windows=n_windows, new_stops=new_stops, passed=not new_stops, held_out=True)
    # H2 ring clearance, H3 course time, H4 pillar A (stack base, held-out logs, surrogate windows)
    h2 = g['H2_ring_clearance']
    rings_cache, h2_rows, missing, no_motor, progress = {}, [], [], [], {}
    lateral = []
    windows_total = windows_scored = 0
    h4 = g['H4_pillar_a']
    pillar_rows = []
    for flight in held:
        wins = windows_of(gates, out, 'stack', flight)
        if not wins:
            continue
        motor = motor_of(gates, flight)
        if motor is None:
            no_motor.append(flight)
            continue
        base_arr = _load(out, 'repo', 'stack', 'base', flight)
        rule_arr = _load(out, 'repo', 'stack', 'rule', flight)
        for w0, w1 in wins:
            p = out/'windows'/f'H_{flight}_{w0:.2f}.json'
            if not p.exists():
                missing.append(f'{flight} {w0:.2f}')
                continue
            r = json.loads(p.read_text(encoding='utf-8'))
            windows_total += 1
            rows = ring_clearance(gates, flight, (w0, w1), r['paths'], rings_cache)
            scored = [x for x in rows if x['valid']]
            windows_scored += bool(scored)
            # a window without a valid ring (little parallax: the ring nearly dead ahead) falls back on the lateral
            # deviation of the rule's path from the base's
            ok = (all(x['passed'] for x in scored) if scored
                  else r['lateral_max_m'] <= h2['fallback_lateral_m'])
            h2_rows.append(dict(flight=flight, window=[round(w0, 2), round(w1, 2)], motor=motor, rings=rows,
                                passed=bool(ok), scored=bool(scored),
                                lateral_max_m=r['lateral_max_m'], progress_lost_s=r['progress_lost_s']))
            lateral.append(r['lateral_max_m'])
            progress[flight] = progress.get(flight, 0.)+r['progress_lost_s']
            # H4: windows over the pillar-A approach region
            t = np.asarray(base_arr['t'], float)
            m = (t >= w0) & (t <= w1)
            xy = np.stack([base_arr['x'], base_arr['y']], 1)[m]
            reg = h4['approach_region']
            inside = ((xy[:, 0] >= reg['x'][0]) & (xy[:, 0] <= reg['x'][1]) & (xy[:, 1] >= reg['y'][0])
                      & (xy[:, 1] <= reg['y'][1]))
            if inside.any():
                qb = np.asarray(r['paths']['base']['pos'], float)[:, :2]
                qr = np.asarray(r['paths']['rule']['pos'], float)[:, :2]
                db, dr = float(_box_distance(qb, h4['pillar_box_xy']).min()), float(_box_distance(qr, h4['pillar_box_xy']).min())
                shifted = m & (np.abs(np.nan_to_num(base_arr['gap_offset'])) > 1e-9)
                changed = (np.hypot(rule_arr['cvx']-base_arr['cvx'], rule_arr['cvy']-base_arr['cvy']) > 1e-9) & shifted
                pillar_rows.append(dict(flight=flight, window=[round(w0, 2), round(w1, 2)], base_m=round(db, 3),
                                        rule_m=round(dr, 3), changed_while_shifted=int(changed.sum()),
                                        passed=bool(dr >= db-h4['tolerance_m'] and not changed.any())))
    fraction = windows_scored/max(windows_total, 1)
    failed = [x for x in h2_rows if not x['passed']]
    table['H2_ring_clearance'] = dict(windows=windows_total, scored=windows_scored, scored_fraction=round(fraction, 3),
                                      failed=failed, missing=missing, open_loop_only=no_motor,
                                      worst_lateral_m=max(lateral, default=0.),
                                      worst_lateral_unscored_m=max((x['lateral_max_m'] for x in h2_rows
                                                                    if not x['scored']), default=0.),
                                      passed=bool(not missing and not failed), held_out=True, rows=h2_rows)
    frac = g['H3_course_time']['max_fraction']
    h3 = {}
    for flight in held:
        a = _load(out, 'repo', 'stack', 'base', flight)
        t = np.asarray(a['t'], float)
        duration = float(t[-1]-t[0]) if len(t) > 1 else 0.
        if flight in progress:
            h3[flight] = dict(kind='surrogate', progress_lost_s=round(progress[flight], 3),
                              fraction=round(progress[flight]/max(duration, 1e-9), 5))
        elif flight in no_motor:
            b = _load(out, 'repo', 'stack', 'rule', flight)
            m = np.zeros(len(t), bool)
            for w0, w1 in windows_of(gates, out, 'stack', flight):
                m |= (t >= w0) & (t <= w1)
            dt = np.clip(np.diff(t, prepend=t[0]), 0., .1)
            removed = float(np.sum(np.clip(np.hypot(a['cvx'], a['cvy'])-np.hypot(b['cvx'], b['cvy']), 0., None)[m]
                                   * dt[m]))
            h3[flight] = dict(kind='open loop', removed_m=round(removed, 3),
                              fraction=round(removed/max(_request_travel(a, t), 1e-9), 5))
    over = {k: v for k, v in h3.items() if v['fraction'] > frac}
    table['H3_course_time'] = dict(flights=h3, over=over, passed=not over, held_out=True)
    table['H4_pillar_a'] = dict(windows=pillar_rows, passed=all(x['passed'] for x in pillar_rows), held_out=True)
    # harness
    hdir = out/'harness'

    def rows_of(motor, set_name, variant):
        return json.loads((hdir/f'{motor}_{set_name}_{variant}.json').read_text(encoding='utf-8'))['rows']
    tf, sm = g['HG_gates']['max_time_fraction'], g['HG_gates']['stop_mps']
    for name, set_name in (('HG1_gate_posts', 'gate_posts'), ('HG2_gate_near', 'gate_near')):
        per = {}
        for motor in gates['harness']['motors']:
            base, rule = rows_of(motor, set_name, 'base'), rows_of(motor, set_name, 'rule')
            pb, pr = sum(r['post_contact'] for r in base), sum(r['post_contact'] for r in rule)
            fb, fr = sum(r['finished'] for r in base), sum(r['finished'] for r in rule)
            both = [(x['finish_s'], y['finish_s']) for x, y in zip(base, rule) if x['finished'] and y['finished']]
            tb = float(np.mean([a for a, _ in both])) if both else float('nan')
            tr = float(np.mean([b for _, b in both])) if both else float('nan')
            slower = (tr-tb)/tb if both else 0.
            stops = [x['params'] for x, y in zip(base, rule)
                     if (x['gate_min_speed'] or 0.) >= sm and (y['gate_min_speed'] or 0.) < sm]
            gaps = [(x['min_post_gap_m'], y['min_post_gap_m']) for x, y in zip(base, rule)]
            per[motor] = dict(posts_base=pb, posts_rule=pr, finished_base=fb, finished_rule=fr,
                              mean_finish_base=round(tb, 3), mean_finish_rule=round(tr, 3),
                              slower_fraction=round(slower, 5), new_stops=stops,
                              smallest_post_gap_base=min((a for a, _ in gaps if a is not None), default=None),
                              smallest_post_gap_rule=min((b for _, b in gaps if b is not None), default=None),
                              passed=bool(pr <= pb and fr >= fb and slower <= tf and not stops))
        table[name] = dict(per, passed=all(v['passed'] for v in per.values()), held_out=True)
    per = {}
    for motor in gates['harness']['motors']:
        hb, hr = rows_of(motor, 'hairpin', 'base'), rows_of(motor, 'hairpin', 'rule')
        wb, wr = sum(r['wall_contact'] for r in hb), sum(r['wall_contact'] for r in hr)
        cb, cr = sum(_clean(r) for r in hb), sum(_clean(r) for r in hr)
        per[motor] = dict(wall_base=wb, wall_rule=wr, clean_base=cb, clean_rule=cr, passed=wr <= wb and cr >= cb)
    table['HG3_hairpin'] = dict(per, passed=all(v['passed'] for v in per.values()), held_out=True)
    per = {}
    for motor in gates['harness']['motors']:
        pb, pr = rows_of(motor, 'passthrough', 'base'), rows_of(motor, 'passthrough', 'rule')
        fb, fr = sum(r['finished'] for r in pb), sum(r['finished'] for r in pr)
        cb, cr = sum(_clean(r) for r in pb), sum(_clean(r) for r in pr)
        stops = [x['params'] for x, y in zip(pb, pr)
                 if (x['arch_min_speed'] or 0.) >= sm and (y['arch_min_speed'] or 0.) < sm]
        both = [(x['finish_s'], y['finish_s']) for x, y in zip(pb, pr) if x['finished'] and y['finished']]
        tb = float(np.mean([a for a, _ in both])) if both else float('nan')
        tr = float(np.mean([b for _, b in both])) if both else float('nan')
        slower = (tr-tb)/tb if both else 0.
        per[motor] = dict(finished_base=fb, finished_rule=fr, clean_base=cb, clean_rule=cr, new_stops=stops,
                          slower_fraction=round(slower, 5),
                          passed=bool(fr >= fb and cr >= cb and not stops and slower <= tf))
    table['HG4_passthrough'] = dict(per, passed=all(v['passed'] for v in per.values()), held_out=True)
    gated = [k for k, v in table.items() if v.get('passed') is not None]
    result.update(passed=all(table[k]['passed'] for k in gated),
                  held_out_passed=all(table[k]['passed'] for k in gated if table[k]['held_out']),
                  development_passed=all(table[k]['passed'] for k in gated if not table[k]['held_out']),
                  gates_passed=f'{sum(bool(table[k]["passed"]) for k in gated)} of {len(gated)}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['replays', 'windows', 'harness', 'score'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--gates', default=str(GATES_PATH))
    parser.add_argument('--flights', default=None)
    parser.add_argument('--json', default=None)
    args = parser.parse_args(argv)
    gates, digest = load_gates(args.gates)
    if args.command == 'replays':
        main_replays(args, gates)
    elif args.command == 'windows':
        main_windows(args, gates)
    elif args.command == 'harness':
        main_harness(args, gates)
    else:
        result = score(args.out, gates, digest)
        text = json.dumps(result, indent=1, default=float)
        if args.json:
            Path(args.json).write_text(text+'\n', encoding='utf-8')
        print(text)


if __name__ == '__main__':
    main()
