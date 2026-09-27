"""Frozen surrogate gates of the view-keeping descent (configs/pilot/descent_view_gates.json); development evidence only.

``runall`` flies every (controller, course set, variant) of the gates file through
`haltere.liftoff.descent_rehearsal.run_batch` (one process, CPU, two threads, resumable: existing outputs are kept),
``score`` compares the descent-view variant with the baseline (the current pilot) on the same seeds and writes
every gate with its threshold, baseline, result and verdict. Both refuse a gates file that is not frozen or changed
after its freeze, and a descent-view declaration other than the one the gates name.

usage:
  python -m haltere.liftoff.descent_gates runall --gates configs/pilot/descent_view_gates.json --dir OUT
  python -m haltere.liftoff.descent_gates score --gates configs/pilot/descent_view_gates.json --dir OUT --json SCORES
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GATES_PATH = REPO/'configs'/'pilot'/'descent_view_gates.json'
META_KEYS = ('frozen', 'frozen_at', 'sha256')


def content_sha256(obj):
    """Hash of a declaration without its freeze keys (canonical sorted, compact JSON; as configs/obstacles)."""
    body = {k: v for k, v in obj.items() if k not in META_KEYS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode('utf-8')).hexdigest()


def load_gates(path=GATES_PATH):
    gates = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = content_sha256(gates)
    if gates.get('frozen') is not True or gates.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    declaration_path = REPO/gates['descent_view']['file']
    declaration = json.loads(declaration_path.read_text(encoding='utf-8'))
    if declaration.get('version') != gates['descent_view']['version']:
        # a later version replaced the declaration; the scored one is kept beside it (descent_view_v<version>.json)
        kept = declaration_path.with_name(f"{declaration_path.stem}_v{gates['descent_view']['version']}.json")
        if kept.exists():
            declaration_path = kept
            declaration = json.loads(kept.read_text(encoding='utf-8'))
    if (declaration.get('version') != gates['descent_view']['version']
            or declaration.get('sha256') != gates['descent_view']['sha256']
            or content_sha256(declaration) != declaration['sha256'] or declaration.get('frozen') is not True):
        raise ValueError('The descent-view declaration is not the frozen one these gates score')
    from .fast_rehearsal import VIEWABLE_SLOPE_DEG
    if gates['terrain']['viewable_slope_deg'] != VIEWABLE_SLOPE_DEG:
        raise ValueError('The scoring code uses another viewable slope than the gates declare')
    return gates, digest, declaration_path


def output_name(controller, course_set, variant):
    return f"{controller}_{course_set.replace(':', '_')}_{variant}.json"


def main_runall(args):
    import torch
    from .descent_rehearsal import course_set, load_controller, parse_set, run_batch, sha256, _summary
    from .fast_race_cue import descent_view_config
    torch.set_num_threads(2)
    gates, digest, declaration_path = load_gates(args.gates)
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_path = REPO/gates['surrogate']['profile'] if not Path(gates['surrogate']['profile']).is_absolute() \
        else Path(gates['surrogate']['profile'])
    if sha256(profile_path) != gates['surrogate']['profile_sha256']:
        raise ValueError(f'{profile_path} is not the dynamics profile the gates name')
    profile = json.loads(profile_path.read_text())
    declaration = json.loads(declaration_path.read_text(encoding='utf-8'))
    for name, spec in gates['controllers'].items():
        if args.only and name not in args.only:
            continue
        controller = None
        for set_spec in gates['course_sets']['gate']:
            kind, seeds = parse_set(set_spec)
            courses, terrains = course_set(kind, seeds)
            for variant in ('baseline', 'descent_view'):
                path = out/output_name(name, set_spec, variant)
                if path.exists():
                    continue
                if controller is None:
                    if sha256(spec['checkpoint']) != spec['sha256']:
                        raise ValueError(f'{spec["checkpoint"]} is not the checkpoint the gates name')
                    controller = load_controller(spec['kind'], spec['checkpoint'])
                kw = {} if variant == 'baseline' else dict(descent_view=descent_view_config(declaration))
                begin = time.time()
                rows, _ = run_batch(controller, profile, courses, terrains, pilot_kwargs=kw,
                                    seconds=gates['surrogate']['seconds'], seed=gates['surrogate']['sim_seed'])
                for seed, row in zip(seeds, rows):
                    row['seed'] = seed
                result = dict(controller=name, kind=spec['kind'], checkpoint=spec['checkpoint'],
                              checkpoint_sha256=spec['sha256'], set=set_spec, variant=variant,
                              gates_sha256=digest, declaration_sha256=declaration['sha256'],
                              profile_sha256=sha256(profile_path), elapsed_s=round(time.time()-begin, 1),
                              summary=_summary(rows), courses=rows)
                path.write_text(json.dumps(result, indent=1))
                print(json.dumps(dict(out=path.name, elapsed_s=result['elapsed_s'], **result['summary'])), flush=True)
                if args.rest > 0:
                    time.sleep(args.rest)


def _load(out, controller, set_spec, variant):
    return json.loads((Path(out)/output_name(controller, set_spec, variant)).read_text())


def _totals(runs, key):
    return sum((r.get(key) or 0) for run in runs for r in run['courses'])


def score_controller(out, name, gates):
    """Every gate of one controller: dict(gate -> dict(threshold, baseline, result, pass, detail))."""
    terrain_sets = gates['course_sets']['terrain_gate']
    flat_sets = gates['course_sets']['flat_gate']
    load = lambda sets, variant: [_load(out, name, s, variant) for s in sets]
    base_t, view_t = load(terrain_sets, 'baseline'), load(terrain_sets, 'descent_view')
    base_f, view_f = load(flat_sets, 'baseline'), load(flat_sets, 'descent_view')
    base_all, view_all = base_t+base_f, view_t+view_f
    g = gates['gates']
    out_gates = {}
    # contacts
    b, v = _totals(base_t, 'contacts'), _totals(view_t, 'contacts')
    new = [f"{run['set']}:{r['seed']}" for brun, run in zip(base_t, view_t)
           for rb, r in zip(brun['courses'], run['courses']) if (rb.get('contacts') or 0) == 0 and (r.get('contacts') or 0) > 0]
    reduction = None if b == 0 else 1-v/b
    out_gates['contacts_reduced'] = dict(
        threshold=f">= {g['contacts_reduction_min']:.0%} fewer than the baseline, and no course with contacts where the "
                  f"baseline had none", baseline=b, result=v, reduction=None if reduction is None else round(reduction, 3),
        new_contact_courses=new,
        contact_s=dict(baseline=round(_totals(base_t, 'contact_s'), 2), result=round(_totals(view_t, 'contact_s'), 2)),
        **{'pass': bool(b > 0 and reduction >= g['contacts_reduction_min'] and not new)})
    # crashes and finishes (every set)
    b, v = _totals(base_all, 'crashed'), _totals(view_all, 'crashed')
    out_gates['crashes_not_more'] = dict(threshold='<= baseline', baseline=b, result=v, **{'pass': bool(v <= b)})
    b, v = _totals(base_all, 'finished'), _totals(view_all, 'finished')
    out_gates['finishes_not_fewer'] = dict(threshold='>= baseline', baseline=b, result=v, **{'pass': bool(v >= b)})
    # path below view while descending (terrain sets)
    def frac(runs):
        return _totals(runs, 'below_view_s')/max(_totals(runs, 'descent_s'), 1e-9)
    out_gates['path_below_view'] = dict(
        threshold=f"<= {g['below_view_max']:.0%} of the descent time", baseline=round(frac(base_t), 4),
        result=round(frac(view_t), 4), descent_s=dict(baseline=round(_totals(base_t, 'descent_s'), 1),
                                                     result=round(_totals(view_t, 'descent_s'), 1)),
        **{'pass': bool(frac(view_t) <= g['below_view_max'])})

    def viewable(runs):
        return _totals(runs, 'viewable_hill_below_view_s')/max(_totals(runs, 'viewable_hill_descent_s'), 1e-9)
    out_gates['path_below_view_viewable'] = dict(
        threshold=f"<= {g['below_view_viewable_max']:.0%} of the descent time on hill legs no steeper than "
                  f"{gates['terrain']['viewable_slope_deg']:g} deg", baseline=round(viewable(base_t), 4),
        result=round(viewable(view_t), 4),
        descent_s=dict(baseline=round(_totals(base_t, 'viewable_hill_descent_s'), 1),
                       result=round(_totals(view_t, 'viewable_hill_descent_s'), 1)),
        **{'pass': bool(viewable(view_t) <= g['below_view_viewable_max'])})
    # high passes (every set)
    b, v = _totals(base_all, 'high_passes'), _totals(view_all, 'high_passes')
    out_gates['high_passes'] = dict(threshold=f"<= baseline + {g['high_passes_extra']}", baseline=b, result=v,
                                    **{'pass': bool(v <= b+g['high_passes_extra'])})
    # course time, paired over courses both finish (terrain sets)
    def paired(bruns, vruns):
        pairs = [(rb['finish_s'], r['finish_s']) for brun, run in zip(bruns, vruns)
                 for rb, r in zip(brun['courses'], run['courses']) if rb['finished'] and r['finished']]
        if not pairs:
            return None, None, 0
        pairs = np.asarray(pairs, float)
        return float(pairs[:, 0].mean()), float(pairs[:, 1].mean()), len(pairs)
    mb, mv, n = paired(base_t, view_t)
    ratio = None if mb is None else mv/mb
    out_gates['course_time'] = dict(threshold=f"paired mean <= baseline x {1+g['time_slower_max']:.2f}",
                                    baseline=None if mb is None else round(mb, 2),
                                    result=None if mv is None else round(mv, 2), paired=n,
                                    ratio=None if ratio is None else round(ratio, 4),
                                    **{'pass': bool(ratio is not None and ratio <= 1+g['time_slower_max'])})
    # flat courses unchanged
    mb, mv, n = paired(base_f, view_f)
    ratio = None if mb is None else mv/mb
    fb, fv = _totals(base_f, 'finished'), _totals(view_f, 'finished')
    cb, cv = _totals(base_f, 'crashed'), _totals(view_f, 'crashed')
    ok = (ratio is not None and abs(ratio-1) <= g['flat_change_max'] and fv == fb and cv == cb)
    out_gates['flat_unchanged'] = dict(
        threshold=f"same finishes and crashes, paired mean time within +-{g['flat_change_max']:.0%}",
        baseline=dict(finished=fb, crashed=cb, mean_s=None if mb is None else round(mb, 2)),
        result=dict(finished=fv, crashed=cv, mean_s=None if mv is None else round(mv, 2)), paired=n,
        ratio=None if ratio is None else round(ratio, 4), **{'pass': bool(ok)})
    # reported, not gated
    chatter = lambda runs: float(np.mean([r['stick_chatter'] for run in runs for r in run['courses']]))
    report = dict(stick_chatter=dict(baseline=round(chatter(base_all), 5), result=round(chatter(view_all), 5)),
                  support_climbs=dict(baseline=_totals(base_t, 'support_climbs'), result=_totals(view_t, 'support_climbs')),
                  per_set={run['set']: dict(baseline=brun['summary'], result=run['summary'])
                           for brun, run in zip(base_all, view_all)})
    return out_gates, report


def main_score(args):
    gates, digest, _ = load_gates(args.gates)
    result = dict(gates_file=str(args.gates), gates_version=gates['version'], gates_sha256=digest,
                  descent_view=gates['descent_view'], scope=gates['scope'], controllers={})
    everything = True
    for name in gates['controllers']:
        scored, report = score_controller(args.dir, name, gates)
        result['controllers'][name] = dict(gates=scored, report=report,
                                           passed=all(v['pass'] for v in scored.values()))
        everything &= result['controllers'][name]['passed']
        print(name, {k: v['pass'] for k, v in scored.items()}, flush=True)
    result['all_passed'] = everything
    Path(args.json).write_text(json.dumps(result, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('runall')
    run.add_argument('--gates', default=str(GATES_PATH))
    run.add_argument('--dir', required=True)
    run.add_argument('--only', nargs='*', default=None, help='controller names to run')
    run.add_argument('--rest', type=float, default=5., help='seconds of rest between batches (thermal duty cycle)')
    score = sub.add_parser('score')
    score.add_argument('--gates', default=str(GATES_PATH))
    score.add_argument('--dir', required=True)
    score.add_argument('--json', required=True)
    args = parser.parse_args()
    (main_runall if args.command == 'runall' else main_score)(args)


if __name__ == '__main__':
    main()
