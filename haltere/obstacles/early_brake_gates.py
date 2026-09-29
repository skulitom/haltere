"""Frozen gates of the round-6 brake rules (configs/obstacles/early_brake_gates.json): the looming governor's early brake
(configs/obstacles/early_brake.json version 1) and motor assist version 4 (configs/pilot/motor_assist.json; version 3
kept as motor_assist_v3.json). Development and held-out evidence as each gate says; none of it is flight evidence.

Commands (one process, CPU, two threads; resumable: existing outputs are kept):
  harness  the declared surrogate parts (motor_assist_eval hairpin and pass-through scenarios with live-like wall
           samples, a report with the gate-style samples, and descent_rehearsal hill courses) for each declared brain and
           variant (base: the m5 flight stack as haltere.train.deployed_pilot builds it with the stale-evidence rule;
           ma3: + motor assist v3; eb: + the early brake; eb_ma4: + the early brake and motor assist v4);
  replays  open-loop replays of every declared log through this tree and the baseline tree (git archive of m5) with
           haltere/obstacles/vertical_replay.py, one variant per call;
  windows  the development hairpin window replayed in the surrogate from the logged state (stale_evidence_gates.Surrogate);
  score    every gate, written as JSON.

usage:
  python -m haltere.obstacles.early_brake_gates harness --dir OUT
  python -m haltere.obstacles.early_brake_gates replays --dir OUT --baseline-tree TREE
  python -m haltere.obstacles.early_brake_gates windows --dir OUT
  python -m haltere.obstacles.early_brake_gates score --dir OUT --json SCORES
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

REPO = Path(__file__).resolve().parents[2]
GATES_PATH = REPO/'configs'/'obstacles'/'early_brake_gates.json'
PY = sys.executable
EPS = 1e-9
RING_STATES = ('cue', 'below', 'below_weak', 'above')


def content_sha256(obj):
    from ..liftoff.descent_gates import content_sha256 as sha
    return sha(obj)


def file_sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gates(path=GATES_PATH):
    """(gates, content hash). Refuses unfrozen or edited gates and declarations other than the frozen ones they name."""
    gates = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = content_sha256(gates)
    if gates.get('frozen') is not True or gates.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    for key in ('early_brake', 'motor_assist', 'motor_assist_v3'):
        spec = gates['declarations'][key]
        declaration = json.loads((REPO/spec['file']).read_text(encoding='utf-8'))
        if (declaration.get('version') != spec['version'] or declaration.get('sha256') != spec['sha256']
                or content_sha256(declaration) != spec['sha256'] or declaration.get('frozen') is not True):
            raise ValueError(f'{spec["file"]} is not the frozen declaration these gates score')
    return gates, digest


def declaration(gates, key):
    return json.loads((REPO/gates['declarations'][key]['file']).read_text(encoding='utf-8'))


def _limit_threads():
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    import torch
    torch.set_num_threads(2)


# ---------------------------------------------------------------------------------------------------------------
# Surrogate parts
# ---------------------------------------------------------------------------------------------------------------
def pilot_kwargs(gates, variant, contract):
    """FastRaceCue kwargs of a declared variant for a motor contract."""
    from ..liftoff import fast_race_cue as frc
    from ..train.deployed_pilot import deployed_pilot_kwargs
    kw, _ = deployed_pilot_kwargs(contract, stale_evidence=True)
    if variant in ('ma3', 'eb_ma4'):
        key = 'motor_assist_v3' if variant == 'ma3' else 'motor_assist'
        config = frc.motor_assist_for_contract(declaration(gates, key), contract)
        if config is not None:
            kw['motor_assist'] = config
    if variant in ('eb', 'eb_ma4'):
        config = frc.early_brake_for_contract(declaration(gates, 'early_brake'), contract)
        if config is not None:
            kw['early_brake'] = config
    elif variant not in ('base', 'ma3'):
        raise ValueError(variant)
    return kw


def main_harness(args, gates):
    _limit_threads()
    from ..liftoff import motor_assist_eval as mae
    from ..liftoff.descent_rehearsal import course_set, load_controller, parse_set, run_batch, _summary
    from ..liftoff.motor_assist_gates import scenario_counts
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    s = gates['surrogate']
    if file_sha256(s['profile']) != s['profile_sha256']:
        raise ValueError('The dynamics profile is not the one the gates name')
    profile = json.loads(Path(s['profile']).read_text())
    h = gates['harness']
    for name, spec in h['controllers'].items():
        if args.only and name not in args.only:
            continue
        if file_sha256(spec['checkpoint']) != spec['sha256']:
            raise ValueError(f'{spec["checkpoint"]} is not the checkpoint the gates name')
        controller = None
        for part, pspec in h['parts'].items():
            for variant in pspec['variants']:
                path = out/f'{name}_{part}_{variant}.json'
                if path.exists():
                    continue
                if controller is None:
                    controller = load_controller(spec['kind'], spec['checkpoint'])
                kw = pilot_kwargs(gates, variant, spec['contract'])
                begin = time.time()
                summary = None
                if pspec['kind'] in ('hairpin', 'passthrough'):
                    maker = mae.hairpin_set if pspec['kind'] == 'hairpin' else mae.passthrough_set
                    rows, _ = mae.run_scenarios(controller, profile, maker(pspec['set']), pilot_kwargs=kw,
                                                seconds=pspec['seconds'], seed=pspec['sim_seed'],
                                                live_wall_samples=pspec['live_wall_samples'])
                    summary = scenario_counts(rows)
                else:
                    kind, seeds = parse_set(pspec['set'])
                    courses, terrains = course_set(kind, seeds)
                    rows, _ = run_batch(controller, profile, courses, terrains, pilot_kwargs=kw,
                                        seconds=pspec['seconds'], seed=pspec['sim_seed'])
                    for seed, row in zip(seeds, rows):
                        row['seed'] = seed
                    summary = _summary(rows)
                path.write_text(json.dumps(dict(controller=name, part=part, variant=variant, gates_sha256=gates['sha256'],
                                                elapsed_s=round(time.time()-begin, 1), summary=summary, rows=rows),
                                           indent=1, default=float))
                print(json.dumps(dict(out=path.name, elapsed_s=round(time.time()-begin, 1))), flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Replays
# ---------------------------------------------------------------------------------------------------------------
def replay_path(out, variant, flight):
    return Path(out)/'replays'/variant/f'{flight}.npz'


def main_replays(args, gates):
    r = gates['replays']
    flights = gates['logs']['development']+gates['logs']['held_out']
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    for variant, spec in r['variants'].items():
        if args.only and variant not in args.only:
            continue
        tree = Path(args.baseline_tree) if spec['tree'] == 'baseline' else REPO
        harness = REPO/'haltere'/'obstacles'/'vertical_replay.py'
        for streamed in (False, True):
            todo = [f for f in flights if (f in r['stream_flights']) == streamed
                    and not replay_path(args.dir, variant, f).exists()]
            if not todo:
                continue
            tmp = Path(args.dir)/'replays'/variant/'tmp'
            tmp.mkdir(parents=True, exist_ok=True)
            cmd = ([PY, '-W', 'ignore', str(harness), '--tree', str(tree), '--runs', r['runs'], '--out', str(tmp/'r')]
                   + spec['args']+(['--looming-stream', r['stream']] if streamed else [])+todo)
            begin = time.time()
            proc = subprocess.run(cmd, cwd=str(tree), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  text=True)
            if proc.returncode:
                raise RuntimeError(f'{variant} failed: {proc.stderr[-3000:]}')
            for flight in todo:
                made = list(tmp.glob(f'r_*_{flight}.npz'))
                if len(made) != 1:
                    raise RuntimeError(f'{variant} {flight}: expected one replay file, found {made}')
                made[0].replace(replay_path(args.dir, variant, flight))
            print(json.dumps(dict(variant=variant, flights=len(todo), stream=streamed,
                                  seconds=round(time.time()-begin, 1))), flush=True)


def load_replay(out, variant, flight):
    return dict(np.load(replay_path(out, variant, flight), allow_pickle=True))


# ---------------------------------------------------------------------------------------------------------------
# Surrogate windows
# ---------------------------------------------------------------------------------------------------------------
def main_windows(args, gates):
    _limit_threads()
    from .stale_evidence_gates import Surrogate
    out = Path(args.dir)/'windows'
    out.mkdir(parents=True, exist_ok=True)
    sur = Surrogate(dict(surrogate=gates['surrogate'], runs=gates['replays']['runs']))
    for w in gates['windows']:
        path = out/f'{w["flight"]}_{w["t0"]:.2f}.json'
        if path.exists():
            continue
        f = sur.flight(w['flight'])
        res = {}
        for variant in w['variants']:
            if variant == 'logged':
                req, yaw = f.req.copy(), f.cmds[:, 3].copy()
            else:
                req, yaw = Surrogate.aligned(f, load_replay(args.dir, variant, w['flight']))
            pos, vel = sur.fly(w['flight'], w['motor'], w['t0'], w['horizon_s'], req, yaw)
            res[variant] = dict(end_speed=round(float(np.hypot(vel[-1, 0], vel[-1, 1])), 3),
                                end_position=[round(float(v), 3) for v in pos[-1]],
                                min_height=round(float(pos[:, 2].min()), 3))
        path.write_text(json.dumps(dict(window=w, results=res), indent=1))
        print(json.dumps(dict(window=path.name, results=res)), flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------
def durations(t):
    t = np.asarray(t, float)
    return np.diff(t, append=t[-1])


def horizontal(a, prefix='c'):
    return np.hypot(np.asarray(a[f'{prefix}vx'], float), np.asarray(a[f'{prefix}vy'], float))


def own_horizontal(a):
    return (np.hypot(np.asarray(a['assist_pilot_vx'], float), np.asarray(a['assist_pilot_vy'], float))
            if 'assist_pilot_vx' in a else horizontal(a))


def identical(a, b):
    from .vertical_replay import COMMAND_KEYS, identical as same
    return same(a, b, COMMAND_KEYS)[0]


def brake_warning_s(a, window_s=3.):
    """Seconds before the log's last tick (the impact) at which the governor's cap first bound the request within the
    last window_s (braking == 1), or None."""
    t = np.asarray(a['t'], float)
    idx = np.flatnonzero((t >= t[-1]-window_s) & (np.asarray(a['braking'], float) > 0))
    return None if not len(idx) else float(t[-1]-t[idx[0]])


def final_cut_s(base, var, cut=1., gap_s=.1):
    """Seconds before the last tick at which the final continuous run of ticks whose horizontal request lies at least
    `cut` below the base replay's (gaps up to gap_s bridged) begins, or None."""
    t = np.asarray(base['t'], float)
    ok = horizontal(base)-horizontal(var) >= cut
    if not ok[-1]:
        return None
    i = len(t)-1
    while i > 0:
        if ok[i-1]:
            i -= 1
            continue
        j = i-1
        while j >= 0 and not ok[j] and t[i]-t[j] <= gap_s:
            j -= 1
        if j >= 0 and ok[j] and t[i]-t[j] <= gap_s:
            i = j
            continue
        break
    return float(t[-1]-t[i])


def crawl_s(base, var):
    """Seconds with the ring ahead (pilot states cue, below, below_weak, above) where the variant's horizontal request is
    below 2 m/s while the base replay asks for at least 3 m/s: a crawl the variant plans that the base did not."""
    ring = np.isin(np.asarray(var['state']).astype(str), RING_STATES)
    bad = ring & (horizontal(var) < 2.) & (horizontal(base) >= 3.)
    return float(durations(base['t'])[bad].sum())


def removed_share(base, var):
    dt = durations(base['t'])
    hb, hv = horizontal(base), horizontal(var)
    travel = float((hb*dt).sum())
    return float((np.maximum(hb-hv, 0.)*dt).sum())/travel if travel > 0 else 0.


def floor_violation_s(a, floor, settle_s=.3):
    """Seconds without a wall-ahead condition (and none in the preceding settle_s: the slewed release) where the
    assisted horizontal request lies more than 0.01 m/s below min(the pilot's own, floor)."""
    if 'assist_wall_ahead' not in a:
        return 0.
    t = np.asarray(a['t'], float)
    wall = np.nan_to_num(np.asarray(a['assist_wall_ahead'], float)) > 0
    recent = np.zeros(len(t), bool)
    last = -np.inf
    for i in range(len(t)):
        if wall[i]:
            last = t[i]
        recent[i] = t[i]-last <= settle_s
    bad = ~recent & (horizontal(a) < np.minimum(own_horizontal(a), floor)-.01)
    return float(durations(t)[bad].sum())


def share_loss_s(stack, var, tol=.05, t0=None, t1=None):
    """Seconds in pilot state cue while the assist adds a climb where the pilot's own vertical request lies more than
    tol below the unassisted stack's (the ceiling cut taking the pilot's own climb)."""
    t = np.asarray(stack['t'], float)
    sel = np.ones(len(t), bool) if t0 is None else (t >= t0) & (t <= t1)
    cue = np.asarray(var['state']).astype(str) == 'cue'
    adds = np.nan_to_num(np.asarray(var['assist_vertical'], float)) > 0
    lower = np.asarray(var['assist_pilot_vz'], float) < np.asarray(stack['cvz'], float)-tol
    return float(durations(t)[sel & cue & adds & lower].sum())


def gate(threshold, result, passed, **detail):
    return dict(threshold=threshold, result=result, **detail, **{'pass': bool(passed)})


# ---------------------------------------------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------------------------------------------
def main_score(args, gates):
    from ..liftoff.motor_assist_gates import _paired_finish, course_totals, scenario_counts
    out = Path(args.dir)
    g = gates['gates']
    logs = gates['logs']
    flights = logs['development']+logs['held_out']
    motors = gates['replays']['flight_motor']
    brains = [f for f in flights if motors[f] == 'brain']
    pds = [f for f in flights if motors[f] == 'pd']
    result = dict(gates_sha256=gates['sha256'], declarations=gates['declarations'], gates={}, reports={})
    R = lambda v, f: load_replay(out, v, f)

    # identity
    for key, (a, b, which) in dict(I1_default=('none', 'base_none', flights), I2_stack=('stack', 'base_stack', flights),
                                   I3_assist_v3=('ma3', 'base_ma3', flights),
                                   I4_shadow=('shadow_eb', 'base_shadow', flights)).items():
        same = [f for f in which if identical(R(a, f), R(b, f))]
        result['gates'][key] = gate(g['identity']['threshold'], f'{len(same)}/{len(which)}', len(same) == len(which),
                                    differ=[f for f in which if f not in same])
    same = [f for f in pds if identical(R('eb_ma4', f), R('stack', f)) and identical(R('eb', f), R('stack', f))]
    result['gates']['I5_fast_pd'] = gate(g['identity']['threshold'], f'{len(same)}/{len(pds)}', len(same) == len(pds),
                                         differ=[f for f in pds if f not in same])

    # development cases
    d = g['D1_hairpin_warning']
    base, eb = R('stack', d['flight']), R('eb', d['flight'])
    w_base, w_eb = brake_warning_s(base), brake_warning_s(eb)
    result['gates']['D1_hairpin_warning'] = gate(d['threshold'], round(w_eb, 3) if w_eb is not None else None,
                                                 w_eb is not None and w_eb >= d['min_s'],
                                                 base=None if w_base is None else round(w_base, 3))
    d = g['D2_hairpin_cut']
    cut = final_cut_s(R('stack', d['flight']), R('eb_ma4', d['flight']), d['cut'])
    cut3 = final_cut_s(R('stack', d['flight']), R('ma3', d['flight']), d['cut'])
    result['gates']['D2_hairpin_cut'] = gate(d['threshold'], None if cut is None else round(cut, 3),
                                             cut is not None and cut >= d['min_s'],
                                             assist_v3=None if cut3 is None else round(cut3, 3))
    d = g['D3_hairpin_window']
    path = out/'windows'/f'{d["flight"]}_{d["t0"]:.2f}.json'
    res = json.loads(path.read_text())['results']
    result['gates']['D3_hairpin_window'] = gate(d['threshold'], res['eb_ma4']['end_speed'],
                                                res['eb_ma4']['end_speed'] <= res['stack']['end_speed']-d['min_drop'],
                                                stack=res['stack']['end_speed'], logged=res['logged']['end_speed'],
                                                assist_v3=res.get('ma3', {}).get('end_speed'))
    d = g['D4_first_arch']
    a = R('eb_ma4', d['flight'])
    x = np.asarray(a['x'], float)
    sel = x < d['x_before']
    viol = floor_violation_s({k: (v[sel] if isinstance(v, np.ndarray) and v.shape == x.shape else v)
                              for k, v in a.items()}, d['floor'])
    a3 = R('ma3', d['flight'])
    result['gates']['D4_first_arch'] = gate(d['threshold'], round(viol, 3), viol <= EPS,
                                            min_request_before_arch=round(float(horizontal(a)[sel].min()), 3),
                                            assist_v3_min_request=round(float(horizontal(a3)[sel].min()), 3))
    d = g['D5_ceiling_share']
    loss4 = share_loss_s(R('stack', d['flight']), R('ma4', d['flight']), t0=d['t0'], t1=d['t1'])
    loss3 = share_loss_s(R('stack', d['flight']), R('ma3', d['flight']), t0=d['t0'], t1=d['t1'])
    result['gates']['D5_ceiling_share'] = gate(d['threshold'], round(loss4, 3), loss4 <= d['max_s'],
                                               assist_v3=round(loss3, 3))

    # held-out surrogate
    h = gates['harness']
    load = lambda name, part, variant: json.loads((out/f'{name}_{part}_{variant}.json').read_text())['rows']
    gh = g['H1_hairpin']
    gated = h['gated']
    for name in h['controllers']:
        base = scenario_counts(load(name, 'hairpin', 'base'))
        new = scenario_counts(load(name, 'hairpin', 'eb_ma4'))
        v3 = scenario_counts(load(name, 'hairpin', 'ma3'))
        ok = (new['clean'] >= base['clean']+gh['clean_gain_min'] and new['wall'] <= int(gh['wall_fraction_max']*base['wall'])
              and new['floor'] <= base['floor']+gh['floor_extra_max'] and new['ceiling'] <= base['ceiling']+gh['ceiling_extra_max']
              and new['wall'] <= v3['wall']+gh['wall_vs_v3_extra_max'])
        keys = ('clean', 'finished', 'wall', 'floor', 'ceiling', 'crashed')
        (result['gates'] if name in gated else result['reports'])[f'H1_hairpin_{name}'] = gate(gh['threshold'], {k: new[k] for k in keys}, ok,
                                                     base={k: base[k] for k in keys}, assist_v3={k: v3[k] for k in keys},
                                                     n=base['n'])
        report = {v: {k: scenario_counts(load(name, 'hairpin_gate', v))[k] for k in keys}
                  for v in h['parts']['hairpin_gate']['variants']}
        result['reports'][f'hairpin_gate_samples_{name}'] = report
    gp = g['H2_passthrough']
    for name in h['controllers']:
        rows = {v: load(name, 'passthrough', v) for v in h['parts']['passthrough']['variants']}
        lowest = {v: [r.get('arch_min_speed') for r in rs] for v, rs in rows.items()}
        finished = {v: sum(r['finished'] for r in rs) for v, rs in rows.items()}
        contacts = {v: scenario_counts(rs)['contacts'] for v, rs in rows.items()}
        ok = True
        for v in gp['variants']:
            speeds = [s for s in lowest[v]]
            ok &= finished[v] >= finished['base'] and contacts[v] <= contacts['base']
            ok &= all(s is not None and s >= gp['min_speed'] for s in speeds)
            ok &= float(np.median([s for s in speeds if s is not None])) >= gp['median_min_speed'] if speeds else False
        (result['gates'] if name in gated else result['reports'])[f'H2_passthrough_{name}'] = gate(gp['threshold'], {v: dict(finished=finished[v], contacts=contacts[v],
                                                                                   lowest=[None if s is None else round(s, 2) for s in lowest[v]])
                                                                           for v in rows}, ok)
    gl = g['H3_hill']
    for name in h['controllers']:
        rb, ra = load(name, 'hill', 'base'), load(name, 'hill', 'eb_ma4')
        b, a = course_totals(rb), course_totals(ra)
        mb, ma, n = _paired_finish(rb, ra)
        ratio = None if mb is None else ma/mb
        ok = (a['finished'] >= b['finished'] and a['crashed'] <= b['crashed']
              and a['chatter'] <= gl['chatter_ratio_max']*b['chatter']
              and ratio is not None and ratio <= 1+gl['time_slower_max']
              and a['contacts'] <= b['contacts']+gl['contacts_extra_max']
              and a['high_passes'] <= b['high_passes']+gl['high_passes_extra_max'])
        r3 = course_totals(load(name, 'hill', 'ma3'))
        (result['gates'] if name in gated else result['reports'])[f'H3_hill_{name}'] = gate(gl['threshold'], a, ok, base=b, assist_v3=r3, paired=n,
                                                  ratio=None if ratio is None else round(ratio, 4))

    # held-out replays
    held_brains = [f for f in logs['held_out'] if motors[f] == 'brain']
    quiet = [f for f in held_brains if not f.startswith('minus')]
    gq = g['H4_replays']
    crawl = {f: round(crawl_s(R('stack', f), R('eb', f)), 3) for f in held_brains}
    crawl4 = {f: round(crawl_s(R('stack', f), R('eb_ma4', f)), 3) for f in held_brains}
    result['gates']['H4a_no_new_crawl'] = gate(gq['crawl_threshold'], dict(eb=crawl, eb_ma4=crawl4),
                                               all(v <= gq['crawl_max_s'] for v in list(crawl.values())+list(crawl4.values())))
    share = {f: round(100*removed_share(R('stack', f), R('eb', f)), 3) for f in quiet}
    result['gates']['H4b_quiet_early_brake'] = gate(gq['eb_quiet_threshold'], share,
                                                    all(v <= 100*gq['eb_removed_share_max'] for v in share.values()))
    viol = {f: round(floor_violation_s(R('eb_ma4', f), gq['floor']), 3) for f in brains}
    result['gates']['H4c_ring_in_view_floor'] = gate(gq['floor_threshold'], viol, all(v <= EPS for v in viol.values()))
    share4 = {f: round(100*removed_share(R('stack', f), R('eb_ma4', f)), 3) for f in quiet}
    share3 = {f: round(100*removed_share(R('stack', f), R('ma3', f)), 3) for f in quiet}
    result['gates']['H4d_quiet_assist'] = gate(gq['assist_quiet_threshold'], share4,
                                               all(v <= 100*gq['assist_removed_share_max'] for v in share4.values()),
                                               assist_v3=share3)
    loss = {f: round(share_loss_s(R('stack', f), R('ma4', f)), 3) for f in held_brains}
    result['gates']['H5_ceiling_share'] = gate(g['H5_ceiling_share']['threshold'], loss,
                                               all(v <= g['H5_ceiling_share']['max_s'] for v in loss.values()),
                                               assist_v3={f: round(share_loss_s(R('stack', f), R('ma3', f)), 3)
                                                          for f in held_brains})

    # reports: every log, what the rules change (development logs marked)
    rep = {}
    for f in flights:
        base = R('stack', f)
        row = dict(development=f in logs['development'], motor=motors[f])
        for v in ('eb', 'eb_ma4', 'ma3'):
            a = R(v, f)
            dt = durations(base['t'])
            changed = np.abs(horizontal(a)-horizontal(base)) > .05
            row[v] = dict(changed_s=round(float(dt[changed].sum()), 2), removed_pct=round(100*removed_share(base, a), 3),
                          min_request=round(float(horizontal(a).min()), 3),
                          early_s=round(float(dt[np.nan_to_num(np.asarray(a.get('early_brake', np.zeros(len(dt))), float)) > 0].sum()), 2))
        rep[f] = row
    result['reports']['replays'] = rep
    passed = sum(v['pass'] for v in result['gates'].values())
    result['summary'] = dict(passed=passed, total=len(result['gates']),
                             failed=[k for k, v in result['gates'].items() if not v['pass']])
    Path(args.json).write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result['summary']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['harness', 'replays', 'windows', 'score'])
    p.add_argument('--dir', required=True)
    p.add_argument('--gates', default=str(GATES_PATH))
    p.add_argument('--baseline-tree', default=None)
    p.add_argument('--json', default=None)
    p.add_argument('--only', nargs='*', default=None)
    args = p.parse_args(argv)
    gates, digest = load_gates(args.gates)
    gates['sha256'] = digest
    if args.command == 'harness':
        main_harness(args, gates)
    elif args.command == 'replays':
        if not args.baseline_tree:
            raise SystemExit('--baseline-tree (a git archive of m5) is required')
        main_replays(args, gates)
    elif args.command == 'windows':
        main_windows(args, gates)
    else:
        if not args.json:
            raise SystemExit('--json is required')
        main_score(args, gates)


if __name__ == '__main__':
    main()
