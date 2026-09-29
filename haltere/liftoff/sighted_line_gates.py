"""Round-7 gates of the sighted descent version 2, the sighted line (configs/pilot/sighted_descent_v2_gates.json):
offline development and held-out evidence, never flight evidence.

Kinds of evidence (the gates file says which are development and which held out):
- closed-loop windows from logged states (``windows``; haltere.liftoff.gate_top.window): the rule's development cases,
  the brain-11 Straw Bale laps at the two downhill arches;
- the two-ring Straw downhill (``straw``; gate_top.StrawTwoRings: the logged geometry with fresh per-drone
  randomisation, and seeded variations on seeds nothing had run before), with the scoring-only arch bars;
- the descent surrogate on course seeds nothing had run before (``surrogate``; descent_rehearsal.run_batch with the arch
  bars at every checkpoint);
- open-loop replays of the logged flights (``replays``; haltere/obstacles/vertical_replay.py, run from this tree and
  from a git archive of m6): identity with the rule off, as flown (version 1) and in shadow, and what the rule changes
  with it on. The three brain-11 Straw laps are the rule's development logs; every other log is held out.
The baseline is the m6 live stack (deployed_pilot_kwargs(contract, stale_evidence=True, early_brake=True,
motor_assist=True, contact_support='shadow', marker_jump='shadow', sighted_descent='on', sighted_version=1): the round-6
plan's flags, with the kept sighted descent version 1); the rule replaces version 1 by version 2 in it.

usage:
  python -m haltere.liftoff.sighted_line_gates windows|straw|surrogate --out DIR [--controllers pd brain11cw13]
  python -m haltere.liftoff.sighted_line_gates replays --out DIR --m6-tree M6TREE
  python -m haltere.liftoff.sighted_line_gates score --out DIR --json SCORES
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GATES_PATH = REPO/'configs'/'pilot'/'sighted_descent_v2_gates.json'
RUNS = Path('C:/DEV/Haltere/runs/fast-stack-20260923')
REPLAY_KEYS = ('cvx', 'cvy', 'cvz', 'yaw_cmd', 'state', 'cap', 'climb', 'descent_scale')
VARIANTS = ('base', 'rule')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gates(path=GATES_PATH):
    """The frozen gates and their content hash; refuses an unfrozen or edited file, and declarations other than the
    frozen ones the gates name (the rule's and the baseline's, version and content hash)."""
    from .gap_stack import config_sha256
    gates = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = config_sha256(gates)
    if gates.get('frozen') is not True or gates.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    for key in ('declaration', 'baseline'):
        spec = gates[key]
        declared = json.loads((REPO/spec['file']).read_text(encoding='utf-8'))
        if (declared.get('frozen') is not True or declared.get('version') != spec['version']
                or declared.get('sha256') != spec['sha256'] or config_sha256(declared) != spec['sha256']):
            raise ValueError(f'{spec["file"]} is not the frozen declaration these gates name ({key})')
    return gates, digest


def _controller(gates, name):
    from .descent_rehearsal import load_controller
    spec = gates['controllers'][name]
    if file_sha256(spec['checkpoint']) != spec['sha256']:
        raise SystemExit(f'{spec["checkpoint"]} is not the declared checkpoint of {name}')
    return spec, load_controller(spec['kind'], spec['checkpoint'])


def _profile(gates):
    path = Path(gates['surrogate']['profile'])
    if file_sha256(path) != gates['surrogate']['profile_sha256']:
        raise SystemExit('the surrogate profile changed')
    return json.loads(path.read_text())


def pilot_kwargs(contract, variant):
    """The FastRaceCue kwargs of a variant: 'base' (the m6 live stack, sighted descent version 1) or 'rule' (the same
    with version 2)."""
    from ..train.deployed_pilot import deployed_pilot_kwargs
    return deployed_pilot_kwargs(contract, stale_evidence=True, early_brake=True, motor_assist=True,
                                 contact_support='shadow', marker_jump='shadow', sighted_descent='on',
                                 sighted_version=1 if variant == 'base' else 2)


def replay_kwargs(variant):
    """The replay harness's `replay` kwargs of a variant for a closed-loop window (the m6 live stack as the runner
    builds it: --obstacle-stack on --stale-evidence on --early-brake on --descent-view on --contact-support shadow
    --sighted-descent on --motor-assist on --marker-jump shadow)."""
    from . import fast_race_cue as frc
    from .gap_stack import config_sha256

    def load(rel):
        d = json.loads((REPO/rel).read_text(encoding='utf-8'))
        if d.get('frozen') is not True or d.get('sha256') != config_sha256(d):
            raise SystemExit(f'{rel} is not frozen')
        return d
    view = load('configs/pilot/descent_view.json')
    sighted = load('configs/pilot/sighted_descent_v1.json' if variant == 'base' else 'configs/pilot/sighted_descent.json')
    return dict(stack='on', near_on_path=True, throttle_column='command_thr',
                descent_view=frc.descent_view_config(view), contact_support=frc.contact_support_config(view),
                contact_apply=False, stale_evidence=load('configs/obstacles/stale_evidence.json'),
                early_brake=load('configs/obstacles/early_brake.json'),
                motor_assist=load('configs/pilot/motor_assist.json'), marker_jump=load('configs/pilot/marker_jump.json'),
                marker_jump_apply=False, sighted_descent=frc.sighted_descent_config(sighted))


def _write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str), encoding='utf-8')


def _seeds(spec):
    lo, _, hi = str(spec).partition('-')
    return list(range(int(lo), int(hi or lo)+1))


def run_windows(gates, out):
    from ..train import brake_gates as bg
    from .gate_top import GateTopConfig, STRAW_ARCHES, StrawTwoRings, window
    w = gates['windows']
    profile = _profile(gates)
    spec = gates['controllers'][w['controller']]
    if file_sha256(spec['checkpoint']) != spec['sha256']:
        raise SystemExit('the window controller changed')
    ctl = bg.Controller(spec['checkpoint'], spec['checkpoint'])
    ground = StrawTwoRings(None).ground_logged
    config = GateTopConfig(**gates['gate_top'])
    for name, case in w['cases'].items():
        arches = [(STRAW_ARCHES[k]['centre'], STRAW_ARCHES[k]['approach']) for k in case['arches']]
        inject = [(t, c) for t, c in case.get('inject', [])]
        for variant in VARIANTS:
            path = Path(out)/f'window_{name}_{variant}.json'
            if path.exists():
                continue
            rows = []
            begin = time.time()
            for seed in _seeds(w['seeds']):
                rows.append(window(case['flight'], case['t0'], arches, replay_kwargs(variant), controller=ctl,
                                   profile=profile, seconds=w['seconds'], seed=seed, ground=ground, inject=inject,
                                   config=config))
            _write(path, dict(case=name, variant=variant, elapsed_s=round(time.time()-begin, 1), rows=rows))
            print(path.name, [[c and c['above_m'] for c in r['crossings']] for r in rows], flush=True)


def run_straw(gates, out, controllers=None):
    from .descent_rehearsal import _summary, run_batch
    from .gate_top import GateTopConfig, StrawTwoRings, _TwoRingTerrain
    s = gates['straw']
    profile = _profile(gates)
    config = GateTopConfig(**gates['gate_top'])
    for name in controllers or gates['controllers']:
        spec, controller = _controller(gates, name)
        for variant in VARIANTS:
            kw, record = pilot_kwargs(spec['contract'], variant)
            for label, courses in (('logged', [StrawTwoRings(None) for _ in range(s['logged_n'])]),
                                   ('family', [StrawTwoRings(k) for k in _seeds(s['family_seeds'])])):
                path = Path(out)/f'straw_{name}_{label}_{variant}.json'
                if path.exists():
                    continue
                begin = time.time()
                rows, _ = run_batch(controller, profile, [v.course for v in courses],
                                    [_TwoRingTerrain(v) for v in courses], pilot_kwargs=kw, seconds=s['seconds'],
                                    seed=s['sim_seed'], gate_top=config)
                for v, r in zip(courses, rows):
                    r['variant'] = v.describe()
                _write(path, dict(controller=name, set=label, variant=variant, sim_seed=s['sim_seed'],
                                  declarations=record, elapsed_s=round(time.time()-begin, 1), summary=_summary(rows),
                                  courses=rows))
                print(path.name, json.dumps(_stats(rows)), flush=True)
                time.sleep(3.)


def run_surrogate(gates, out, controllers=None):
    from .descent_rehearsal import _summary, course_set, parse_set, run_batch
    from .gate_top import GateTopConfig
    s = gates['surrogate']
    profile = _profile(gates)
    config = GateTopConfig(**gates['gate_top'])
    for name in controllers or gates['controllers']:
        spec, controller = _controller(gates, name)
        for variant in VARIANTS:
            kw, record = pilot_kwargs(spec['contract'], variant)
            for set_spec in s['sets']:
                path = Path(out)/f"surrogate_{name}_{set_spec.replace(':', '_')}_{variant}.json"
                if path.exists():
                    continue
                kind, seeds = parse_set(set_spec)
                courses, terrains = course_set(kind, seeds)
                begin = time.time()
                rows, _ = run_batch(controller, profile, courses, terrains, pilot_kwargs=kw, seconds=s['seconds'],
                                    seed=s['sim_seed'], gate_top=config)
                for seed, row in zip(seeds, rows):
                    row['seed'] = seed
                _write(path, dict(controller=name, set=set_spec, variant=variant, sim_seed=s['sim_seed'],
                                  declarations=record, elapsed_s=round(time.time()-begin, 1), summary=_summary(rows),
                                  courses=rows))
                print(path.name, json.dumps(_stats(rows)), flush=True)
                time.sleep(3.)


def _replay_args(gates, variant, tree):
    return [a.replace('{tree}', str(tree)) for a in gates['replays']['variants'][variant]['args']]


def run_replays(gates, out, m6_tree):
    """Every replay variant of the gates file, from this tree and the m6 archive, flight by flight (the streamed
    flights with the offline looming stream). Outputs OUT/replay_{variant}_{tag}_{flight}.npz."""
    r = gates['replays']
    for variant, spec in r['variants'].items():
        tree = Path(m6_tree) if spec['tree'] == 'm6' else REPO
        for streamed in (False, True):
            todo = [f for f in r['flights'] if (f in r['streamed']) == streamed
                    and not list(Path(out).glob(f'replay_{variant}_*_{f}.npz'))]
            if not todo:
                continue
            cmd = [sys.executable, str(tree/'haltere'/'obstacles'/'vertical_replay.py'), *todo, '--tree', str(tree),
                   '--out', str(Path(out)/f'replay_{variant}'), *_replay_args(gates, variant, tree),
                   *(['--looming-stream', r['looming_stream']] if streamed else [])]
            begin = time.time()
            done = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tree))
            if done.returncode:
                raise SystemExit(f'replay {variant} failed: {done.stderr[-3000:]}')
            print(variant, 'streamed' if streamed else 'plain', len(todo), round(time.time()-begin, 1), 's', flush=True)


def _replay(out, variant, flight):
    paths = sorted(Path(out).glob(f'replay_{variant}_*_{flight}.npz'))
    if len(paths) != 1:
        raise SystemExit(f'expected one replay of {variant} for {flight}, found {len(paths)}')
    return dict(np.load(paths[0], allow_pickle=False))


def _identical(a, b, keys=REPLAY_KEYS):
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


def score_replays(gates, out):
    r, g = gates['replays'], gates['gates']
    identity, causal, quiet = {}, {}, {}
    for flight in r['flights']:
        identity[flight] = {f'{a} = {b}': _identical(_replay(out, a, flight), _replay(out, b, flight))
                            for a, b in g['TB_Identity']['pairs']}
        if flight in r['development']:
            continue
        base, rule = _replay(out, 'stack_v1', flight), _replay(out, 'stack_v2', flight)
        diff = np.zeros(len(base['t']), bool)
        for k in ('cvx', 'cvy', 'cvz'):
            diff |= ~np.isclose(base[k], rule[k], rtol=0., atol=0., equal_nan=True)
        # where either version acts: v1 withholds sink (sighted_withheld > 0), v2 sets the sink (sighted_bound finite)
        acting = (np.nan_to_num(base['sighted_withheld']) > 0) | np.isfinite(rule['sighted_bound'])
        first = int(np.flatnonzero(diff)[0]) if diff.any() else None
        causal[flight] = dict(changed_ticks=int(diff.sum()), first_change_t=None if first is None
                              else round(float(rule['t'][first]), 3),
                              ok=first is None or bool(acting[first]))
        vertical = np.abs(np.asarray(base['cvz'], float)-np.asarray(rule['cvz'], float)) > g['TB_Replay_Quiet']['tol_mps']
        share = float(vertical.mean()) if len(vertical) else 0.
        t = np.asarray(rule['t'], float)
        h = np.hypot(base['cvx'], base['cvy'])-np.hypot(rule['cvx'], rule['cvy'])
        straw = flight.startswith('straw-')
        quiet[flight] = dict(vertical_changed_share=round(share, 5), straw=straw,
                             rule_acting_s=round(float(np.isfinite(rule['sighted_bound']).sum()*.01), 2),
                             max_lower_horizontal_mps=round(float(h[acting].max()), 4) if acting.any() else 0.,
                             ok=straw or share <= g['TB_Replay_Quiet']['max_share'])
    return identity, causal, quiet


def _load(out, pattern):
    return json.loads((Path(out)/pattern).read_text(encoding='utf-8'))


def _stats(rows):
    finished = [r for r in rows if r['finished']]
    tops = [r.get('gate_top') or {} for r in rows]
    return dict(courses=len(rows), finished=len(finished), crashed=sum(r['crashed'] for r in rows),
                contacts=sum(r.get('contacts') or 0 for r in rows),
                contact_s=round(sum(r.get('contact_s') or 0 for r in rows), 2),
                high_passes=sum(r['high_passes'] for r in rows),
                top_bar_hits=sum(t.get('top_bar_hits', 0) for t in tops),
                over_arch=sum(t.get('over_arch', 0) for t in tops),
                support_climbs=sum(r.get('support_climbs', 0) for r in rows))


def _paired_time(base, rule):
    both = [(a['finish_s'], b['finish_s']) for a, b in zip(base, rule) if a['finished'] and b['finished']]
    if not both:
        return None
    a, b = np.mean([x for x, _ in both]), np.mean([y for _, y in both])
    return dict(pairs=len(both), base=round(float(a), 3), rule=round(float(b), 3), change=round(float(b/a-1), 5))


def _no_harm(base, rule, g):
    """The held-out checks of one set (the gate's thresholds): counts not more than the base's plus the gate's
    allowance, finishes not fewer, crashes not more, paired finish time within the gate's bound either way."""
    a, b = _stats(base), _stats(rule)
    time_ = _paired_time(base, rule)
    allow = g.get('allow', {})
    checks = {k: b[k] <= a[k]+allow.get(k, 0) for k in ('contacts', 'high_passes', 'top_bar_hits', 'over_arch')}
    checks.update(contact_s=b['contact_s'] <= a['contact_s']*g.get('contact_s_ratio', 1.)+g.get('contact_s_add', 0.),
                  finished=b['finished'] >= a['finished'], crashed=b['crashed'] <= a['crashed'],
                  time=time_ is None or abs(time_['change']) <= g['max_time_change'])
    return dict(base=a, rule=b, paired_time=time_, checks=checks, passed=all(checks.values()))


def _window_rows(out, name, variant):
    return _load(out, f'window_{name}_{variant}.json')['rows']


def score(gates, out):
    g = gates['gates']
    result = {}
    # development windows
    for gate_name in ('TB_Dev_B', 'TB_Dev_Downhill', 'TB_Dev_R5'):
        spec = g[gate_name]
        rows = {v: _window_rows(out, spec['case'], v) for v in VARIANTS}
        above = {v: [[None if c is None else c['above_m'] for c in r['crossings']] for r in rows[v]] for v in VARIANTS}
        rule = rows['rule']
        ok = all(all(c is not None and spec['min_above_m'] <= c['above_m'] <= spec['max_above_m']
                     and c['verdict'] == 'through' for c in r['crossings']) for r in rule)
        if spec.get('no_ground_contact'):
            ok = ok and all(r['ground_contact_s'] == 0 for r in rule)
        result[gate_name] = dict(development=True, case=spec['case'], above_m=above,
                                 top_bar_hits={v: sum(r['top_bar_hits'] for r in rows[v]) for v in VARIANTS},
                                 over_arch={v: sum(r['over_arch'] for r in rows[v]) for v in VARIANTS},
                                 ground_contact_s={v: round(sum(r['ground_contact_s'] for r in rows[v]), 3)
                                                   for v in VARIANTS},
                                 min_ground_clearance_m={v: min(r['min_ground_clearance_m'] for r in rows[v])
                                                         for v in VARIANTS},
                                 support_climbs={v: sum(r['support_climbs'] for r in rows[v]) for v in VARIANTS},
                                 passed=bool(ok))
    for name, case in gates['windows']['cases'].items():
        if case.get('report'):
            rows = {v: _window_rows(out, name, v) for v in VARIANTS}
            result[f'TB_Report_{name}'] = dict(report=True, above_m={
                v: [[None if c is None else c['above_m'] for c in r['crossings']] for r in rows[v]] for v in VARIANTS},
                support_climbs={v: sum(r['support_climbs'] for r in rows[v]) for v in VARIANTS})
    # held-out surrogate sets
    s = gates['surrogate']
    hills = [x for x in s['sets'] if not x.startswith('flat')]
    flats = [x for x in s['sets'] if x.startswith('flat')]
    for name in gates['controllers']:
        for label, gate in (('family', 'TB_Straw_Family'), ('logged', 'TB_Straw_Logged')):
            base = _load(out, f'straw_{name}_{label}_base.json')['courses']
            rule = _load(out, f'straw_{name}_{label}_rule.json')['courses']
            entry = _no_harm(base, rule, g[gate])
            if name in g[gate].get('benefit_controllers', []) and entry['base']['top_bar_hits'] > 0:
                entry['checks']['fewer_top_bar_hits'] = entry['rule']['top_bar_hits'] < entry['base']['top_bar_hits']
                entry['passed'] = all(entry['checks'].values())
            result[f'{gate}/{name}'] = entry
        pool = lambda sets, v: sum((_load(out, f"surrogate_{name}_{x.replace(':', '_')}_{v}.json")['courses']  # noqa
                                    for x in sets), [])
        result[f'TB_Hills/{name}'] = _no_harm(pool(hills, 'base'), pool(hills, 'rule'), g['TB_Hills'])
        flat = _no_harm(pool(flats, 'base'), pool(flats, 'rule'), g['TB_Flat'])
        flat['checks']['finished'] = flat['rule']['finished'] == flat['base']['finished']
        flat['checks']['crashed'] = flat['rule']['crashed'] == flat['base']['crashed']
        flat['passed'] = all(flat['checks'].values())
        result[f'TB_Flat/{name}'] = flat
    identity, causal, quiet = score_replays(gates, out)
    result['TB_Identity'] = dict(pairs=sum(len(v) for v in identity.values()),
                                 identical=sum(sum(v.values()) for v in identity.values()), per_flight=identity)
    result['TB_Identity']['passed'] = result['TB_Identity']['pairs'] == result['TB_Identity']['identical']
    result['TB_Replay_Causal'] = dict(per_flight=causal, passed=all(v['ok'] for v in causal.values()))
    result['TB_Replay_Quiet'] = dict(per_flight=quiet, passed=all(v['ok'] for v in quiet.values()))
    passed = {k: v['passed'] for k, v in result.items() if 'passed' in v}
    return dict(gates_sha256=gates['sha256'], passed=passed, n_passed=sum(passed.values()), n_gates=len(passed),
                results=result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('windows', 'straw', 'surrogate', 'replays', 'score'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--controllers', nargs='*', default=None)
    parser.add_argument('--m6-tree', default=None)
    parser.add_argument('--json', default=None)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(2)
    from ..obstacles.thermal import flight_lock_path
    if flight_lock_path() is None and args.command != 'score':
        raise SystemExit('set HALTERE_FLIGHT_LOCK (heavy offline runs honour the flight lock)')
    gates, _ = load_gates()
    if args.command == 'windows':
        run_windows(gates, args.out)
    elif args.command == 'straw':
        run_straw(gates, args.out, args.controllers)
    elif args.command == 'surrogate':
        run_surrogate(gates, args.out, args.controllers)
    elif args.command == 'replays':
        if not args.m6_tree:
            raise SystemExit('--m6-tree is the git archive of m6 the identity gate compares with')
        run_replays(gates, args.out, args.m6_tree)
    else:
        scores = score(gates, args.out)
        if args.json:
            _write(args.json, scores)
        print(json.dumps(dict(passed=scores['passed'], n_passed=scores['n_passed'], n_gates=scores['n_gates']),
                         indent=1))


if __name__ == '__main__':
    main()
