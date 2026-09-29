"""Offline gates of the marker-jump rule (round 6, gate clearance): configs/pilot/marker_jump.json, a checkpoint marker
that jumps after a gap in the readings is held until confirmed. The gates are declared in
configs/pilot/marker_jump_gates.json and scored only when that file is frozen and names the frozen declaration.
Development and held-out data are as the gates file declares. Nothing here is flight evidence: replays are open loop
(the recorded motion does not respond), windows are semi-closed loop in the IdentifiedSim surrogate (the logged state,
the replayed requests), and the harness scenarios are synthetic.

Hindsight quantities (impact times, contact points, development windows) select and score windows only; none of them
reaches the pilot.

usage (one process, CPU, two threads; resumable: finished outputs are kept):
  python -m haltere.obstacles.marker_jump_gates replays --out DIR [--flights a,b]   open-loop replays (both trees)
  python -m haltere.obstacles.marker_jump_gates windows --out DIR                   semi-closed-loop windows
  python -m haltere.obstacles.marker_jump_gates harness --out DIR                   closed-loop harness sets
  python -m haltere.obstacles.marker_jump_gates score   --out DIR --json SCORES.json
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
GATES_PATH = REPO/'configs'/'pilot'/'marker_jump_gates.json'
PY = sys.executable
COMMAND_KEYS = ('cvx', 'cvy', 'cvz', 'yaw_cmd', 'state', 'cap', 'climb', 'descent_scale')


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
TREES = ('m5', 'repo')


def replay_path(out, tree, base, variant, flight):
    return Path(out)/'replays'/f'{tree}_{base}_{variant}_{flight}.npz'


def variant_jobs(gates):
    """(tree name, base, variant) of every replay: the m5 tree runs the bases only (it has no rule)."""
    jobs = []
    for base in gates['replays']['bases']:
        jobs.append(('m5', base, 'base'))
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
        tree = gates['baseline_tree_path'] if tree_name == 'm5' else str(REPO)
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
                # the m5 tree's own harness cannot replay this log (e.g. a log without a column it reads): reported,
                # scores nothing, like a log without replayable ticks
                skipped[flight] = 'the m5 harness cannot replay it: '+r.stderr.strip().splitlines()[-1][:300]
                skipped_path.write_text(json.dumps(skipped, indent=1), encoding='utf-8')
                print(json.dumps(dict(skipped=flight, reason=skipped[flight])), flush=True)
                continue
            if r.returncode:
                raise RuntimeError(f'{tree_name} {base} {variant} {flight} failed: {r.stderr[-3000:]}')
            made = [p for p in tmp.glob(f'r_*_{flight}.npz')]
            if len(made) != 1:
                raise RuntimeError(f'{tree_name} {base} {variant} {flight}: expected one replay file, found {made}')
            made[0].replace(target)
            summaries = [p for p in tmp.glob('r_*_summary.json')]
            for s in summaries:
                meta = json.loads(s.read_text(encoding='utf-8')).get(flight, {}).get('marker_jump_metadata')
                if meta is not None:
                    target.with_suffix('.marker.json').write_text(json.dumps(meta), encoding='utf-8')
                s.unlink()
            print(json.dumps(dict(tree=tree_name, base=base, variant=variant, flight=flight,
                                  seconds=round(time.time()-begin, 1))), flush=True)
    if skipped:
        (out/'replays'/'skipped.json').write_text(json.dumps(skipped, indent=1), encoding='utf-8')


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


def _overlaps(window, span):
    return span is not None and window[0] <= span[1] and window[1] >= span[0]


def windows_of(gates, out, base, flight):
    """(held-out windows, development windows) where the rule's requests differ from the base's (padded)."""
    s = gates['surrogate']
    a, b = _load(out, 'repo', base, 'base', flight), _load(out, 'repo', base, 'rule', flight)
    found = changed_windows(a, b, pre_s=s['pre_s'], post_s=s['post_s'])
    dev = gates['development_windows'].get(flight)
    held = [w for w in found if not _overlaps(w, dev)]
    return held, [w for w in found if _overlaps(w, dev)]


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
# Semi-closed-loop windows
# ---------------------------------------------------------------------------------------------------------------
def _fly_capped(sur, flight_name, motor, t0, horizon, req, yaw):
    """Surrogate.fly with the horizon cut at the flight's first velocity jump after t0 (the impact)."""
    f = sur.flight(flight_name)
    k0 = f.index(t0)
    cut = f.impact_after(k0)
    horizon = min(horizon, max(.02, float(f.t[min(cut, len(f.t)-1)]-f.t[k0])))
    return sur.fly(flight_name, motor, t0, horizon, req, yaw)


def main_windows(args, gates):
    _limit_threads()
    out = Path(args.out)
    (out/'windows').mkdir(parents=True, exist_ok=True)
    sur = Surrogate(gates)
    d1 = gates['gates']['D1_start_arch']
    jobs = [('D1', d1['flight'], d1['motor'], d1['t0'], d1['horizon_s'])]
    for flight in gates['flights']:
        if not replay_path(out, 'repo', 'stack', 'rule', flight).exists():
            continue
        motor = motor_of(gates, flight)
        if motor is None:
            continue
        held, _ = windows_of(gates, out, 'stack', flight)
        for w0, w1 in held:
            jobs.append(('H', flight, motor, w0, w1-w0))
    for kind, flight, motor, t0, horizon in jobs:
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
        if kind == 'D1':
            req, yaw = f.req.copy(), f.cmds[:, 3].copy()
            paths['logged'] = _fly_capped(sur, flight, motor, t0, horizon, req, yaw)
            point = np.asarray(d1['contact_point'], float)
            res['min_distance_m'] = {k: round(float(np.hypot(*(p[:, :2]-point).T).min()), 3)
                                     for k, (p, _) in paths.items()}
            res['side'] = {k: side_of(p, v, point) for k, (p, v) in paths.items()}
        lat, lost = path_compare(*paths['base'], paths['rule'][0])
        res.update(lateral_max_m=round(lat, 3), progress_lost_s=round(lost, 3), ticks=int(len(paths['base'][0])),
                   paths={k: dict(pos=np.round(p[::5], 3).tolist()) for k, (p, _) in paths.items()},
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
    sets = dict(gate=(mae.gate_set, h['gate_set']), hairpin=(mae.hairpin_set, h['hairpin_set']))
    for motor, m in h['motors'].items():
        controller = None
        for set_name, (build, spec) in sets.items():
            for variant in ('base', 'rule'):
                path = out/'harness'/f'{motor}_{set_name}_{variant}.json'
                if path.exists():
                    continue
                if controller is None:
                    controller = load_controller(m['kind'], m['checkpoint'])
                kw, record = deployed_pilot_kwargs(m['contract'], stale_evidence=True,
                                                   marker_jump='on' if variant == 'rule' else None)
                begin = time.time()
                rows, _ = mae.run_scenarios(controller, profile, build(spec['spec']), pilot_kwargs=kw,
                                            seconds=spec['seconds'], seed=spec['sim_seed'])
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
def _held_at_captures(gates, out, spec):
    """marker_held of the rule replay at the first tick of each listed logged capture (None: capture not replayed)."""
    import pandas as pd
    d = pd.read_csv(Path(gates['runs'])/f'{spec["flight"]}.csv', low_memory=False, usecols=['capture_time', 'image_age'])
    ok = np.isfinite(d.capture_time.to_numpy(float)) & np.isfinite(d.image_age.to_numpy(float))
    captures = d.capture_time.to_numpy(float)[ok]
    rule = _load(out, 'repo', spec['base'], 'rule', spec['flight'])
    res = {}
    for c in spec['held_captures']:
        idx = np.flatnonzero(np.abs(captures-c) < 5e-4)
        res[str(c)] = None if len(idx) == 0 else bool(rule['marker_held'][idx[0]] > 0)
    return res


def _request_travel(a, t):
    dt = np.clip(np.diff(t, prepend=t[0]), 0., .1)
    return float(np.sum(np.hypot(a['cvx'], a['cvy'])*dt))


def score(out, gates, digest):
    out = Path(out)
    g = gates['gates']
    result = dict(gates_sha256=digest, gates_version=gates['version'],
                  declaration=gates['declarations']['marker_jump'], table={})
    table = result['table']
    skipped = {}
    sk = out/'replays'/'skipped.json'
    if sk.exists():
        skipped = json.loads(sk.read_text(encoding='utf-8'))
    flights = [f for f in gates['flights'] if f not in skipped]
    result['flights'] = dict(scored=len(flights), skipped=skipped)
    # identities
    for name, pairs in (('I1_default_off', [(('m5', 'base'), ('repo', 'base'))]),
                        ('I2_shadow', [(('repo', 'base'), ('repo', 'shadow'))])):
        rows, same = {}, 0
        for base in gates['replays']['bases']:
            for flight in flights:
                (ta, va), (tb, vb) = pairs[0]
                ok = identical(_load(out, ta, base, va, flight), _load(out, tb, base, vb, flight))
                same += ok
                if not ok:
                    rows.setdefault(base, []).append(flight)
        n = len(flights)*len(gates['replays']['bases'])
        table[name] = dict(identical=f'{same}/{n}', different=rows, passed=same == n, held_out=True)
    # rule activity (report)
    activity = {}
    for base in gates['replays']['bases']:
        per = {}
        for flight in flights:
            meta_path = replay_path(out, 'repo', base, 'rule', flight).with_suffix('.marker.json')
            counts = json.loads(meta_path.read_text(encoding='utf-8'))['counts'] if meta_path.exists() else None
            held, dev = windows_of(gates, out, base, flight)
            if counts and (counts.get('candidates') or held or dev):
                per[flight] = dict(counts=counts, held_out_windows=[[round(a, 2), round(b, 2)] for a, b in held],
                                   development_windows=[[round(a, 2), round(b, 2)] for a, b in dev])
        activity[base] = per
    result['activity'] = activity
    # D1
    d1 = g['D1_start_arch']
    held_caps = _held_at_captures(gates, out, d1)
    w = json.loads((out/'windows'/f'D1_{d1["flight"]}_{d1["t0"]:.2f}.json').read_text(encoding='utf-8'))
    val = w['min_distance_m']['base']
    new = w['min_distance_m']['rule']
    side_ok = w['side']['rule'].startswith('point on the right')
    table['D1_start_arch'] = dict(held_captures=held_caps, surrogate_min_distance_m=w['min_distance_m'],
                                  side=w['side'], passed=bool(all(held_caps.values()) and val <= d1['validation_max_m']
                                                              and new >= d1['clear_min_m'] and side_ok),
                                  held_out=False)
    # D2 (report)
    d2 = {}
    for flight in gates['development_windows']:
        if flight == d1['flight'] or flight not in flights:
            continue
        _, dev = windows_of(gates, out, 'stack', flight)
        d2[flight] = [[round(a, 2), round(b, 2)] for a, b in dev]
    table['D2_other_crashes'] = dict(changed_development_windows=d2, passed=None, held_out=False, report=True)
    # H1 no new stop
    stop = g['H1_no_new_stop']['stop_mps']
    new_stops = []
    n_windows = {}
    for base in gates['replays']['bases']:
        n = 0
        for flight in flights:
            held, _ = windows_of(gates, out, base, flight)
            if not held:
                continue
            a, b = _load(out, 'repo', base, 'base', flight), _load(out, 'repo', base, 'rule', flight)
            t = np.asarray(a['t'], float)
            for w0, w1 in held:
                n += 1
                m = (t >= w0) & (t <= w1)
                hb, hr = np.hypot(a['cvx'], a['cvy'])[m], np.hypot(b['cvx'], b['cvy'])[m]
                bad = (hr < stop) & (hb >= stop)
                if bad.any():
                    new_stops.append(dict(base=base, flight=flight, window=[round(w0, 2), round(w1, 2)],
                                          ticks=int(bad.sum()), lowest_rule=round(float(hr.min()), 3),
                                          lowest_base=round(float(hb.min()), 3)))
        n_windows[base] = n
    table['H1_no_new_stop'] = dict(held_out_windows=n_windows, new_stops=new_stops, passed=not new_stops,
                                   held_out=True)
    # H2 / H3
    lateral, progress, missing = [], {}, []
    no_motor = []
    for flight in flights:
        held, _ = windows_of(gates, out, 'stack', flight)
        if not held:
            continue
        motor = motor_of(gates, flight)
        if motor is None:
            no_motor.append(flight)
            continue
        for w0, w1 in held:
            p = out/'windows'/f'H_{flight}_{w0:.2f}.json'
            if not p.exists():
                missing.append(f'{flight} {w0:.2f}')
                continue
            r = json.loads(p.read_text(encoding='utf-8'))
            lateral.append(dict(flight=flight, window=[round(w0, 2), round(w1, 2)], motor=motor,
                                lateral_max_m=r['lateral_max_m'], progress_lost_s=r['progress_lost_s'],
                                read=flight in gates['read_by_measurement']))
            progress[flight] = progress.get(flight, 0.)+r['progress_lost_s']
    worst = max([r['lateral_max_m'] for r in lateral], default=0.)
    table['H2_clearance'] = dict(windows=len(lateral), worst_lateral_m=worst,
                                 over=[r for r in lateral if r['lateral_max_m'] > g['H2_clearance']['max_lateral_m']],
                                 missing=missing, open_loop_only=no_motor,
                                 passed=bool(not missing and worst <= g['H2_clearance']['max_lateral_m']),
                                 held_out=True, rows=lateral)
    frac = g['H3_course_time']['max_fraction']
    h3 = {}
    for flight in flights:
        a = _load(out, 'repo', 'stack', 'base', flight)
        t = np.asarray(a['t'], float)
        duration = float(t[-1]-t[0]) if len(t) > 1 else 0.
        if flight in progress:
            h3[flight] = dict(kind='surrogate', progress_lost_s=round(progress[flight], 3),
                              fraction=round(progress[flight]/max(duration, 1e-9), 5))
        elif flight in no_motor:
            b = _load(out, 'repo', 'stack', 'rule', flight)
            held, _ = windows_of(gates, out, 'stack', flight)
            m = np.zeros(len(t), bool)
            for w0, w1 in held:
                m |= (t >= w0) & (t <= w1)
            dt = np.clip(np.diff(t, prepend=t[0]), 0., .1)
            removed = float(np.sum(np.clip(np.hypot(a['cvx'], a['cvy'])-np.hypot(b['cvx'], b['cvy']), 0., None)[m]
                                   * dt[m]))
            total = _request_travel(a, t)
            h3[flight] = dict(kind='open loop', removed_m=round(removed, 3), fraction=round(removed/max(total, 1e-9), 5))
    over = {k: v for k, v in h3.items() if v['fraction'] > frac}
    table['H3_course_time'] = dict(flights=h3, over=over, passed=not over, held_out=True)
    # harness
    h = out/'harness'

    def rows_of(motor, set_name, variant):
        return json.loads((h/f'{motor}_{set_name}_{variant}.json').read_text(encoding='utf-8'))['rows']
    hg = dict(HG1_gate_posts={}, HG2_gate_finishes={}, HG3_gate_quiet={}, HG4_hairpin={})
    tf, sm = g['HG3_gate_quiet']['max_time_fraction'], g['HG3_gate_quiet']['stop_mps']
    for motor in gates['harness']['motors']:
        base, rule = rows_of(motor, 'gate', 'base'), rows_of(motor, 'gate', 'rule')
        pb, pr = sum(r['post_contact'] for r in base), sum(r['post_contact'] for r in rule)
        hg['HG1_gate_posts'][motor] = dict(base=pb, rule=pr, passed=pr <= pb,
                                           false_marker=dict(
                                               base=sum(r['post_contact'] for r in base if r['params']['false_deg']),
                                               rule=sum(r['post_contact'] for r in rule if r['params']['false_deg'])))
        fb, fr = sum(r['finished'] for r in base), sum(r['finished'] for r in rule)
        hg['HG2_gate_finishes'][motor] = dict(base=fb, rule=fr, passed=fr >= fb)
        quiet = [(x, y) for x, y in zip(base, rule) if not x['params']['false_deg']]
        both = [(x['finish_s'], y['finish_s']) for x, y in quiet if x['finished'] and y['finished']]
        tb = float(np.mean([a for a, _ in both])) if both else float('nan')
        tr = float(np.mean([b for _, b in both])) if both else float('nan')
        stops = [x['params'] for x, y in quiet if (x['gate_min_speed'] or 0.) >= sm and (y['gate_min_speed'] or 0.) < sm]
        slower = (tr-tb)/tb if both else 0.
        hg['HG3_gate_quiet'][motor] = dict(both_finished=len(both), mean_finish_base=round(tb, 3),
                                           mean_finish_rule=round(tr, 3), slower_fraction=round(slower, 5),
                                           new_stops=stops, passed=bool(slower <= tf and not stops))
        hb, hr = rows_of(motor, 'hairpin', 'base'), rows_of(motor, 'hairpin', 'rule')
        wb, wr = sum(r['wall_contact'] for r in hb), sum(r['wall_contact'] for r in hr)
        cb, cr = sum(_clean(r) for r in hb), sum(_clean(r) for r in hr)
        hg['HG4_hairpin'][motor] = dict(wall_base=wb, wall_rule=wr, clean_base=cb, clean_rule=cr,
                                        passed=wr <= wb and cr >= cb)
    for name, per in hg.items():
        table[name] = dict(per, passed=all(v['passed'] for v in per.values()), held_out=True)
    gated = [k for k, v in table.items() if v.get('passed') is not None]
    result.update(passed=all(table[k]['passed'] for k in gated),
                  held_out_passed=all(table[k]['passed'] for k in gated if table[k]['held_out']),
                  development_passed=all(table[k]['passed'] for k in gated if not table[k]['held_out']))
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
