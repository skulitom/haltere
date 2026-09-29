"""Round-6 gates of the sighted descent (configs/pilot/sighted_descent_gates.json): offline development and held-out
evidence, never flight evidence.

Kinds of evidence (the gates file says which are development and which held out):
- the descent surrogate on course seeds nothing had run before this branch (``surrogate``;
  descent_rehearsal.run_batch, the round-5 Straw stack of haltere.train.deployed_pilot with the rule off and on);
- Straw-like downhills (``straw``; haltere.liftoff.straw_downhill: the logged Straw Bale hilltop and downhill with fresh
  per-drone randomisation, and seeded variations of it on seeds nothing had run before);
- open-loop replays of the logged flights (haltere/obstacles/vertical_replay.py, run from this tree and from a git
  archive of m5; the variants the gates file lists): identity with the rule off and in shadow, and what the rule
  changes with it on. The two brain-11 Straw laps are the rule's development logs; every other log is held out.

usage:
  python -m haltere.liftoff.sighted_descent_gates surrogate|straw --out DIR [--controllers pd brain11cw13]
  python -m haltere.liftoff.sighted_descent_gates replays --out DIR --m5-tree M5TREE
  python -m haltere.liftoff.sighted_descent_gates score --out DIR --json SCORES
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
GATES_PATH = REPO/'configs'/'pilot'/'sighted_descent_gates.json'
RUNS = Path('C:/DEV/Haltere/runs/fast-stack-20260923')
REPLAY_KEYS = ('cvx', 'cvy', 'cvz', 'yaw_cmd', 'state', 'cap', 'climb', 'descent_scale')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gates(path=GATES_PATH):
    """The frozen gates and their content hash; refuses an unfrozen or edited file and gates written for another
    declaration than this tree's configs/pilot/sighted_descent.json."""
    from .gap_stack import config_sha256
    gates = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = config_sha256(gates)
    if gates.get('frozen') is not True or gates.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    spec = gates['declaration']
    declared = json.loads((REPO/spec['file']).read_text(encoding='utf-8'))
    if declared.get('version') != spec['version'] or declared.get('sha256') != spec['sha256']:
        raise ValueError(f'{path} scores sighted descent version {spec["version"]} ({spec["sha256"][:12]}), not the '
                         f'declaration {spec["file"]} in this tree')
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


def _kwargs(spec, variant):
    from ..train.deployed_pilot import deployed_pilot_kwargs
    return deployed_pilot_kwargs(spec['contract'], stale_evidence=True, sighted_descent=variant)


def _write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str), encoding='utf-8')


def run_surrogate(gates, out, controllers=None):
    from .descent_rehearsal import _summary, course_set, parse_set, run_batch
    s = gates['surrogate']
    profile = _profile(gates)
    for name in controllers or gates['controllers']:
        spec, controller = _controller(gates, name)
        for variant in ('off', 'on'):
            kw, record = _kwargs(spec, variant)
            for set_spec in s['sets']:
                path = Path(out)/f"surrogate_{name}_{set_spec.replace(':', '_')}_{variant}.json"
                if path.exists():
                    continue
                kind, seeds = parse_set(set_spec)
                courses, terrains = course_set(kind, seeds)
                begin = time.time()
                rows, _ = run_batch(controller, profile, courses, terrains, pilot_kwargs=kw, seconds=s['seconds'],
                                    seed=s['sim_seed'])
                for seed, row in zip(seeds, rows):
                    row['seed'] = seed
                _write(path, dict(controller=name, set=set_spec, variant=variant, sim_seed=s['sim_seed'],
                                  declarations=record, elapsed_s=round(time.time()-begin, 1), summary=_summary(rows),
                                  courses=rows))
                print(path.name, json.dumps(_summary(rows)), flush=True)
                time.sleep(3.)


def run_straw(gates, out, controllers=None):
    from .descent_rehearsal import _summary
    from .straw_downhill import StrawVariant, run, variants
    s = gates['straw']
    profile = _profile(gates)
    for name in controllers or gates['controllers']:
        spec, controller = _controller(gates, name)
        for variant in ('off', 'on'):
            kw, record = _kwargs(spec, variant)
            for label, courses in (('logged', [StrawVariant(None) for _ in range(s['logged_n'])]),
                                   ('family', variants(s['family_seeds']))):
                path = Path(out)/f'straw_{name}_{label}_{variant}.json'
                if path.exists():
                    continue
                begin = time.time()
                rows = run(controller, profile, courses, pilot_kwargs=kw, seconds=s['seconds'], sim_seed=s['sim_seed'])
                _write(path, dict(controller=name, set=label, variant=variant, sim_seed=s['sim_seed'],
                                  declarations=record, elapsed_s=round(time.time()-begin, 1), summary=_summary(rows),
                                  courses=rows))
                print(path.name, json.dumps(_summary(rows)), flush=True)
                time.sleep(3.)


def _replay_args(gates, variant, tree):
    r = gates['replays']
    args = list(r['variants'][variant]['args'])
    return [a.replace('{tree}', str(tree)) for a in args]


def run_replays(gates, out, m5_tree):
    """Every replay variant of the gates file, from this tree and the m5 archive, flight by flight (the streamed flights
    with the offline looming stream). Outputs OUT/replay_{variant}_{tag}_{flight}.npz."""
    r = gates['replays']
    for variant, spec in r['variants'].items():
        tree = Path(m5_tree) if spec['tree'] == 'm5' else REPO
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


def _audit(flight, runs=RUNS):
    """(t_start, t_end, kind) of the contact audit's contacts of a log: its own .contact-audit.json, else the frozen
    audit v1 contacts of docs/experiments."""
    own = Path(runs)/f'{flight}.contact-audit.json'
    if own.exists():
        results = json.loads(own.read_text())['results']
    else:
        results = json.loads((REPO/'docs'/'experiments'/'contact_audit_v1_contacts.json').read_text())['results']
    return [(c['t_start'], c['t_end'], c['kind'], c['pos']) for r in results if r['log'] == flight
            for c in r['contacts']]


def _path_deg(a):
    return np.degrees(np.arctan2(-np.asarray(a['cvz'], float), np.hypot(a['cvx'], a['cvy'])))


def score_replays(gates, out):
    r = gates['replays']
    g = gates['gates']
    identity, causal, speed, report = {}, {}, {}, {}
    for flight in r['flights']:
        pairs = {pair: _identical(_replay(out, pair[0], flight), _replay(out, pair[1], flight))
                 for pair in (('m5_none', 'none'), ('m5_stack', 'stack_off'), ('stack_off', 'stack_shadow'))}
        identity[flight] = {f'{a} = {b}': v for (a, b), v in pairs.items()}
        off, on = _replay(out, 'stack_off', flight), _replay(out, 'stack_on', flight)
        diff = np.zeros(len(off['t']), bool)
        for k in ('cvx', 'cvy', 'cvz'):
            diff |= ~np.isclose(off[k], on[k], rtol=0., atol=0., equal_nan=True)
        withheld = np.nan_to_num(on['sighted_withheld']) > 0
        first = int(np.flatnonzero(diff)[0]) if diff.any() else None
        causal[flight] = dict(changed_ticks=int(diff.sum()), first_change_t=None if first is None else
                              round(float(on['t'][first]), 3), first_change_withheld=None if first is None
                              else bool(withheld[first]), ok=first is None or bool(withheld[first]))
        h_off, h_on = np.hypot(off['cvx'], off['cvy']), np.hypot(on['cvx'], on['cvy'])
        lower = (h_off-h_on)[withheld] if withheld.any() else np.zeros(0)
        speed[flight] = dict(withheld_ticks=int(withheld.sum()),
                             max_lower_horizontal=round(float(lower.max()), 4) if len(lower) else 0.,
                             ok=not len(lower) or float(lower.max()) <= g['SD_Replay_KeepSpeed']['max_lower_mps'])
        t = np.asarray(on['t'], float)
        dt = np.diff(t, append=t[-1])
        minutes = max((t[-1]-t[0])/60., 1e-9)
        extra = np.asarray(off['cvz'], float)-np.asarray(on['cvz'], float)       # > 0: the rule asks more sink
        contacts = []
        for t0, t1, kind, pos in _audit(flight):
            w = (t >= t0-g['SD_Straw_Development']['lookback_s']) & (t <= t0)
            if not w.any():
                continue
            contacts.append(dict(t_start=t0, kind=kind, pos=pos,
                                 steepest_request_deg_m5=round(float(_path_deg(off)[w].max()), 2),
                                 steepest_request_deg_rule=round(float(_path_deg(on)[w].max()), 2),
                                 rule_withheld_s=round(float((dt*withheld)[w].sum()), 3)))
        report[flight] = dict(withheld_s_per_min=round(float((dt*withheld).sum())/minutes, 3),
                              max_withheld_mps=round(float(np.nan_to_num(on['sighted_withheld']).max()), 3),
                              more_sink_ticks=int((extra < -0.05).sum()),
                              max_more_sink_mps=round(float(max(0., -extra.min())), 3), contacts=contacts)
    return identity, causal, speed, report


def _load(out, pattern):
    return json.loads((Path(out)/pattern).read_text(encoding='utf-8'))


def _pool(out, prefix, name, labels, variant):
    rows = []
    for label in labels:
        rows += _load(out, f"{prefix}_{name}_{label.replace(':', '_')}_{variant}.json")['courses']
    return rows


def _stats(rows):
    finished = [r for r in rows if r['finished']]
    return dict(courses=len(rows), finished=len(finished), crashed=sum(r['crashed'] for r in rows),
                contacts=sum(r.get('contacts') or 0 for r in rows),
                contact_s=round(sum(r.get('contact_s') or 0 for r in rows), 2),
                high_passes=sum(r['high_passes'] for r in rows))


def _paired_time(off, on):
    both = [(a['finish_s'], b['finish_s']) for a, b in zip(off, on) if a['finished'] and b['finished']]
    if not both:
        return None
    a, b = np.mean([x for x, _ in both]), np.mean([y for _, y in both])
    return dict(pairs=len(both), off=round(float(a), 3), on=round(float(b), 3), change=round(float(b/a-1), 5))


def _no_harm(off, on, g):
    a, b = _stats(off), _stats(on)
    time_ = _paired_time(off, on)
    checks = dict(contacts=b['contacts'] <= a['contacts'], high_passes=b['high_passes'] <= a['high_passes'],
                  finished=b['finished'] >= a['finished'], crashed=b['crashed'] <= a['crashed'],
                  time=time_ is None or time_['change'] <= g['max_time_change'])
    return dict(off=a, on=b, paired_time=time_, checks=checks, passed=all(checks.values()))


def score(gates, out):
    g = gates['gates']
    result = {}
    s = gates['surrogate']
    hills = [x for x in s['sets'] if not x.startswith('flat')]
    flats = [x for x in s['sets'] if x.startswith('flat')]
    for name in gates['controllers']:
        off, on = (_pool(out, 'surrogate', name, hills, v) for v in ('off', 'on'))
        result[f'SD_Surrogate_Hills/{name}'] = _no_harm(off, on, g['SD_Surrogate_Hills'])
        off, on = (_pool(out, 'surrogate', name, flats, v) for v in ('off', 'on'))
        flat = _no_harm(off, on, g['SD_Flat'])
        time_ = flat['paired_time']
        flat['checks']['time'] = time_ is None or abs(time_['change']) <= g['SD_Flat']['max_time_change']
        flat['checks']['finished'] = flat['on']['finished'] == flat['off']['finished']
        flat['checks']['crashed'] = flat['on']['crashed'] == flat['off']['crashed']
        flat['passed'] = all(flat['checks'].values())
        result[f'SD_Flat/{name}'] = flat
        for label, gate in (('family', 'SD_Straw_Family'), ('logged', 'SD_Straw_Logged')):
            off = _load(out, f'straw_{name}_{label}_off.json')['courses']
            on = _load(out, f'straw_{name}_{label}_on.json')['courses']
            entry = _no_harm(off, on, g[gate])
            if name in g[gate].get('benefit_controllers', []):
                a, b = entry['off'], entry['on']
                entry['checks']['contact_s_benefit'] = b['contact_s'] <= g[gate]['max_contact_s_ratio']*a['contact_s']
                entry['passed'] = all(entry['checks'].values())
            result[f'{gate}/{name}'] = entry
    identity, causal, speed, report = score_replays(gates, out)
    result['SD_Identity'] = dict(pairs=sum(len(v) for v in identity.values()),
                                 identical=sum(sum(v.values()) for v in identity.values()), per_flight=identity)
    result['SD_Identity']['passed'] = result['SD_Identity']['pairs'] == result['SD_Identity']['identical']
    result['SD_Replay_Causal'] = dict(per_flight=causal, passed=all(v['ok'] for v in causal.values()))
    result['SD_Replay_KeepSpeed'] = dict(per_flight=speed, passed=all(v['ok'] for v in speed.values()))
    dev = g['SD_Straw_Development']
    entries = []
    for flight in dev['flights']:
        for c in report[flight]['contacts']:
            if c['kind'] == 'terminal':
                continue
            rule, m5 = c['steepest_request_deg_rule'], c['steepest_request_deg_m5']
            entries.append(dict(flight=flight, **c, ok=rule <= m5-dev['min_shallower_deg']
                                and rule <= dev['ring_los_deg'][flight]+dev['max_above_los_deg']))
    result['SD_Straw_Development'] = dict(development=True, contacts=entries,
                                          passed=bool(entries) and all(e['ok'] for e in entries))
    result['SD_Replay_Report'] = dict(report=True, per_flight=report)
    passed = {k: v['passed'] for k, v in result.items() if 'passed' in v}
    return dict(gates_sha256=gates['sha256'], passed=passed, n_passed=sum(passed.values()), n_gates=len(passed),
                results=result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('surrogate', 'straw', 'replays', 'score'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--controllers', nargs='*', default=None)
    parser.add_argument('--m5-tree', default=None)
    parser.add_argument('--json', default=None)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(2)
    gates, _ = load_gates()
    if args.command == 'surrogate':
        run_surrogate(gates, args.out, args.controllers)
    elif args.command == 'straw':
        run_straw(gates, args.out, args.controllers)
    elif args.command == 'replays':
        if not args.m5_tree:
            raise SystemExit('--m5-tree is the git archive of m5 the identity gate compares with')
        run_replays(gates, args.out, args.m5_tree)
    else:
        scores = score(gates, args.out)
        if args.json:
            _write(args.json, scores)
        print(json.dumps(dict(passed=scores['passed'], n_passed=scores['n_passed'], n_gates=scores['n_gates']),
                         indent=1))


if __name__ == '__main__':
    main()
