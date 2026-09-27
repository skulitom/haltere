"""Frozen gates of the motor assist (configs/pilot/motor_assist_gates.json); development evidence only.

``run`` flies every declared part for every declared brain, with and without the motor assist, through the round-4
pilot as the runner builds it for the motor contract (`--obstacle-stack on --descent-view on`); one process, CPU, two
threads, resumable (existing outputs are kept):
- ``hairpin`` / ``accelerate``: the held-out scenario sets of `haltere.liftoff.motor_assist_eval` (parameters and sim
  seed disjoint from the development sets the rule was designed on);
- ``dev16`` / ``hill``: `haltere.liftoff.descent_rehearsal.run_batch` on the declared course sets;
- ``windows``: the live r4 windows (development cases) replayed in the surrogate from the logged state;
- ``hairpin_flatray`` (report only, never a gate): the hairpin set with the looming governor's cap along the
  horizontal part of its ray (`motor_assist_eval.horizontal_governor_ray`, a stand-in for another branch's fix).
``score`` compares the assist with the baseline on the same scenarios, courses and drones, and scores the open-loop
replay gates (identity against the m4 tree, and how much the assist changes the requests of clean Straw Bale and Pine
Valley brain laps) from replay files written by `haltere/obstacles/vertical_replay.py`. Both refuse a gates file that
is not frozen or changed after its freeze, and a motor-assist declaration other than the one the gates name.

usage:
  python -m haltere.liftoff.motor_assist_gates run --dir OUT [--parts ...] [--only CONTROLLER ...]
  python -m haltere.liftoff.motor_assist_gates score --dir OUT --replays PREFIX --baseline-replays PREFIX --json SCORES
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .descent_gates import content_sha256

REPO = Path(__file__).resolve().parents[2]
GATES_PATH = REPO/'configs'/'pilot'/'motor_assist_gates.json'
PARTS = ('hairpin', 'accelerate', 'dev16', 'hill', 'windows', 'hairpin_flatray')
VARIANTS = ('baseline', 'assist')


def load_gates(path=GATES_PATH):
    gates = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = content_sha256(gates)
    if gates.get('frozen') is not True or gates.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    declaration_path = REPO/gates['motor_assist']['file']
    declaration = json.loads(declaration_path.read_text(encoding='utf-8'))
    if (declaration.get('version') != gates['motor_assist']['version']
            or declaration.get('sha256') != gates['motor_assist']['sha256']
            or content_sha256(declaration) != declaration['sha256'] or declaration.get('frozen') is not True):
        raise ValueError('The motor-assist declaration is not the frozen one these gates score')
    return gates, digest, declaration


def pilot_kwargs(contract, declaration=None):
    """FastRaceCue kwargs of `--obstacle-stack on --descent-view on` for a motor contract as the runner builds them, plus
    the motor assist of `declaration` for that contract when given."""
    from . import visual_brain as vb
    from .fast_race_cue import (descent_view_config, lag_turn_for_contract, motor_assist_for_contract,
                                vertical_guard_config, wall_pilot_configs)
    from .gap_aim import GapAimConfig
    from .gap_stack import load_gap_pilot
    lag, _ = vb.load_lag_turn_declaration(vb.LAG_TURN_DECLARATION)
    wall, _ = vb.load_wall_pilot()
    vertical, _ = vb.load_vertical_guard()
    gap, _ = load_gap_pilot()
    view, _ = vb.load_descent_view()
    kw = dict(lag_turn=lag_turn_for_contract(lag, contract), gap_aim=GapAimConfig.from_dict(gap['pilot']),
              vertical_guard=vertical_guard_config(vertical), descent_view=descent_view_config(view),
              **wall_pilot_configs(wall, contract))
    if declaration is not None:
        assist = motor_assist_for_contract(declaration, contract)
        if assist is not None:
            kw['motor_assist'] = assist
    return kw


def output_name(controller, part, variant, extra=''):
    return f"{controller}_{part}{extra}_{variant}.json"


def _jobs(gates, spec, parts):
    jobs = []
    for part in parts:
        if part in ('hairpin', 'accelerate', 'hairpin_flatray'):
            jobs.append((part, ''))
        elif part in ('dev16', 'hill'):
            jobs += [(part, '_'+s.replace(':', '_')) for s in gates['course_sets'][part]]
        elif part == 'windows' and any(w['controller'] == spec['name'] for w in gates['windows']['windows']):
            jobs.append((part, ''))
    return jobs


def main_run(args):
    import torch
    from .descent_rehearsal import course_set, load_controller, parse_set, run_batch, sha256, _summary
    from . import motor_assist_eval as mae
    torch.set_num_threads(2)
    gates, digest, declaration = load_gates(args.gates)
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_path = Path(gates['surrogate']['profile'])
    if sha256(profile_path) != gates['surrogate']['profile_sha256']:
        raise ValueError(f'{profile_path} is not the dynamics profile the gates name')
    profile = json.loads(profile_path.read_text())
    parts = args.parts or list(PARTS)
    for name, spec in gates['controllers'].items():
        if args.only and name not in args.only:
            continue
        controller = None
        for part, extra in _jobs(gates, dict(spec, name=name), parts):
            for variant in VARIANTS:
                path = out/output_name(name, part, variant, extra)
                if path.exists():
                    continue
                if sha256(spec['checkpoint']) != spec['sha256']:
                    raise ValueError(f'{spec["checkpoint"]} is not the checkpoint the gates name')
                if controller is None and part != 'windows':
                    controller = load_controller(spec['kind'], spec['checkpoint'])
                kw = pilot_kwargs(spec['contract'], declaration if variant == 'assist' else None)
                if variant == 'assist' and 'motor_assist' not in kw:
                    raise ValueError(f'The declaration assigns no motor assist to {spec["contract"]}')
                begin = time.time()
                summary = None
                if part in ('hairpin', 'accelerate', 'hairpin_flatray'):
                    sc = gates['accelerate' if part == 'accelerate' else 'hairpin']
                    scenarios = (mae.accelerate_set if part == 'accelerate' else mae.hairpin_set)(sc['set'])
                    rows, _ = mae.run_scenarios(controller, profile, scenarios, pilot_kwargs=kw, seconds=sc['seconds'],
                                                seed=sc['sim_seed'], flat_governor_ray=part == 'hairpin_flatray')
                elif part in ('dev16', 'hill'):
                    kind, seeds = parse_set(extra[1:].replace('_', ':', 1))
                    courses, terrains = course_set(kind, seeds)
                    rows, _ = run_batch(controller, profile, courses, terrains, pilot_kwargs=kw,
                                        seconds=gates['surrogate']['seconds'], seed=gates['surrogate']['sim_seed'])
                    for seed, row in zip(seeds, rows):
                        row['seed'] = seed
                    summary = _summary(rows)
                else:
                    rows = run_windows(gates, name, spec, profile, declaration if variant == 'assist' else None)
                result = dict(controller=name, kind=spec['kind'], checkpoint_sha256=spec['sha256'],
                              contract=spec['contract'], part=part, set=extra[1:] or None, variant=variant,
                              gates_sha256=digest, declaration_sha256=declaration['sha256'],
                              elapsed_s=round(time.time()-begin, 1), summary=summary, rows=rows)
                path.write_text(json.dumps(result, indent=1, default=float))
                print(json.dumps(dict(out=path.name, elapsed_s=result['elapsed_s'])), flush=True)
                if args.rest > 0:
                    time.sleep(args.rest)


def run_windows(gates, name, spec, profile, declaration):
    """The declared live windows of this brain (the brain that flew them), with or without the assist."""
    from . import motor_assist_eval as mae
    from .fast_race_cue import motor_assist_for_contract
    from ..train.brake_gates import Controller, Flight
    w = gates['windows']
    ctl = Controller(spec['checkpoint'], gates['controllers'][w['reference_controller']]['checkpoint'])
    assist = None if declaration is None else motor_assist_for_contract(declaration, spec['contract'])
    rows = []
    for window in w['windows']:
        if window['controller'] != name:
            continue
        flight = Flight(w['runs'], window['flight'])
        arrays = sources = None
        if assist is not None:
            arrays, sources, _ = mae.window_sources(window['flight'], str(REPO), declaration, w['runs'])
        res = mae.live_window(ctl, profile, flight, window['t0'], window['horizon_s'], assist=assist, arrays=arrays,
                              sources=sources, warm_s=w['warm_s'])
        k0, H = res['k0'], res['H']
        live_v = flight.vel[k0+1:k0+H+1]
        live_z = flight.d.z.to_numpy(float)[k0+1:k0+H+1]
        hs = np.hypot(res['vel'][:, 0], res['vel'][:, 1])
        rows.append(dict(flight=window['flight'], t0=window['t0'], horizon_s=round(H*.01, 2),
                         end_speed=round(float(hs[-1]), 3), live_end_speed=round(float(np.hypot(*live_v[-1, :2])), 3),
                         min_height=round(float(res['pos'][:, 2].min()), 3),
                         live_min_height=round(float(live_z.min()), 3),
                         request_end=np.round(res['req'][-1], 3).tolist(),
                         speed_trace=np.round(hs[::10], 2).tolist(), height_trace=np.round(res['pos'][::10, 2], 2).tolist(),
                         request_vz_trace=np.round(res['req'][::10, 2], 2).tolist(),
                         request_h_trace=np.round(np.hypot(res['req'][::10, 0], res['req'][::10, 1]), 2).tolist()))
    return rows


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------
def _load(out, controller, part, variant, extra=''):
    return json.loads((Path(out)/output_name(controller, part, variant, extra)).read_text())


def scenario_counts(rows):
    """Per-set counts of a scenario part. Contacts: wall (the drone centre within 0.3 m of the wall plane), floor (below
    0.05 m after the first ring), ceiling (above the scoring ceiling - 0.1 m) or a surrogate crash."""
    loss = [r['height_loss_m'] for r in rows if r['height_loss_m'] is not None]
    finished = [r['finish_s'] for r in rows if r['finished']]
    contact = lambda r: bool(r['wall_contact'] or r['floor_contact'] or r['ceiling_contact'] or r['crashed'])
    return dict(n=len(rows), finished=len(finished), clean=sum(r['finished'] and not contact(r) for r in rows),
                crashed=sum(r['crashed'] for r in rows), wall=sum(r['wall_contact'] for r in rows),
                floor=sum(r['floor_contact'] for r in rows), ceiling=sum(r['ceiling_contact'] for r in rows),
                contacts=sum(contact(r) for r in rows),
                height_loss_mean=None if not loss else round(float(np.mean(loss)), 4),
                height_loss_max=None if not loss else round(float(np.max(loss)), 4),
                max_height=round(float(max(r['max_height_m'] for r in rows)), 3),
                chatter=round(float(np.mean([r['stick_chatter'] for r in rows])), 6),
                finish_mean=None if not finished else round(float(np.mean(finished)), 3))


def _paired_finish(base_rows, rows, key='finish_s'):
    pairs = [(b[key], r[key]) for b, r in zip(base_rows, rows) if b['finished'] and r['finished']]
    if not pairs:
        return None, None, 0
    pairs = np.asarray(pairs, float)
    return float(pairs[:, 0].mean()), float(pairs[:, 1].mean()), len(pairs)


def gate(threshold, baseline, result, passed, **detail):
    return dict(threshold=threshold, baseline=baseline, result=result, **detail, **{'pass': bool(passed)})


def course_totals(rows):
    total = lambda k: sum((r.get(k) or 0) for r in rows)
    return dict(courses=len(rows), finished=total('finished'), crashed=total('crashed'), contacts=total('contacts'),
                contact_s=round(total('contact_s'), 2), high_passes=total('high_passes'),
                support_climbs=total('support_climbs'),
                chatter=round(float(np.mean([r['stick_chatter'] for r in rows])), 6),
                below_view_fraction=round(total('below_view_s')/max(total('descent_s'), 1e-9), 4))


def score(args):
    gates, digest, declaration = load_gates(args.gates)
    g = gates['gates']
    result = dict(gates_file=str(args.gates), gates_version=gates['version'], gates_sha256=digest,
                  motor_assist=gates['motor_assist'], scope=gates['scope'], gates={}, reports={})
    out = args.dir
    brains = list(gates['controllers'])

    # hairpin (held-out scenario set)
    gh = g['hairpin']
    rep = {}
    for name in brains:
        b = scenario_counts(_load(out, name, 'hairpin', 'baseline')['rows'])
        a = scenario_counts(_load(out, name, 'hairpin', 'assist')['rows'])
        rep[name] = dict(baseline=b, assist=a)
        braking = name in gh['braking_brains']
        need_clean = b['clean']+int(np.ceil(gh['clean_gain_fraction']*b['n'])) if braking else b['clean']
        wall_max = int(np.floor(gh['wall_fraction_max']*b['wall'])) if braking else b['wall']
        ok = (a['clean'] >= need_clean and a['wall'] <= wall_max and a['contacts'] <= b['contacts']
              and a['ceiling'] <= b['ceiling']+gh['ceiling_extra_max'])
        result['gates'][f'hairpin_{name}'] = gate(
            f"clean passes (finished, no contact) >= {need_clean}; wall contacts <= {wall_max}; scenarios with any "
            f"contact <= baseline; ceiling contacts <= baseline + {gh['ceiling_extra_max']}",
            {k: b[k] for k in ('clean', 'wall', 'floor', 'ceiling', 'crashed', 'contacts')},
            {k: a[k] for k in ('clean', 'wall', 'floor', 'ceiling', 'crashed', 'contacts')}, ok, n=b['n'],
            braking_brain=braking)
    result['reports']['hairpin'] = rep
    rep = {}
    for name in brains:
        if (Path(out)/output_name(name, 'hairpin_flatray', 'assist')).exists():
            rep[name] = {v: scenario_counts(_load(out, name, 'hairpin_flatray', v)['rows']) for v in VARIANTS}
    result['reports']['hairpin_flatray'] = dict(
        note='report only: the governor cap along the horizontal part of its ray (a stand-in for the round-4b contact '
             'fix, which another branch owns); not a gate', **rep)

    # stop-then-accelerate (held-out scenario set)
    ga = g['accelerate']
    rep = {}
    for name in brains:
        base, assist = _load(out, name, 'accelerate', 'baseline')['rows'], _load(out, name, 'accelerate', 'assist')['rows']
        b, a = scenario_counts(base), scenario_counts(assist)
        mb, ma, n = _paired_finish(base, assist)
        ratio = None if mb is None else ma/mb
        rep[name] = dict(baseline=b, assist=a, paired_finish=[mb, ma, n])
        lb, la = b['height_loss_mean'] or 0., a['height_loss_mean'] or 0.
        if lb >= ga['sagging_loss_min']:
            loss_ok, loss_text = la <= ga['loss_fraction_max']*lb, f"mean height loss <= {ga['loss_fraction_max']:.0%} of the baseline's"
        else:
            loss_ok, loss_text = la <= lb+ga['loss_extra_max'], f"mean height loss <= baseline + {ga['loss_extra_max']} m"
        ok = (loss_ok and a['floor'] <= b['floor'] and a['ceiling'] <= b['ceiling'] and a['finished'] >= b['finished']
              and a['crashed'] <= b['crashed'] and ratio is not None and ratio <= 1+ga['time_slower_max'])
        result['gates'][f'accelerate_{name}'] = gate(
            f"{loss_text}; floor and ceiling contacts, crashes <= baseline; finishes >= baseline; paired finish time <= "
            f"baseline x {1+ga['time_slower_max']:.2f}",
            dict(loss=b['height_loss_mean'], floor=b['floor'], ceiling=b['ceiling'], finished=b['finished'], time=mb),
            dict(loss=a['height_loss_mean'], floor=a['floor'], ceiling=a['ceiling'], finished=a['finished'], time=ma),
            ok, paired=n, ratio=None if ratio is None else round(ratio, 4))
    result['reports']['accelerate'] = rep

    # course sets: the 16-course development gate and the descent hills
    for part in ('dev16', 'hill'):
        gp = g[part]
        rep = {}
        for name in brains:
            sets = gates['course_sets'][part]
            rows_b = [r for s in sets for r in _load(out, name, part, 'baseline', '_'+s.replace(':', '_'))['rows']]
            rows_a = [r for s in sets for r in _load(out, name, part, 'assist', '_'+s.replace(':', '_'))['rows']]
            b, a = course_totals(rows_b), course_totals(rows_a)
            mb, ma, n = _paired_finish(rows_b, rows_a)
            ratio = None if mb is None else ma/mb
            rep[name] = dict(baseline=b, assist=a, paired_finish=[mb, ma, n],
                             per_course=[dict(seed=rb.get('seed'), baseline=rb['finish_s'], assist=ra['finish_s'],
                                              contacts=[rb.get('contacts'), ra.get('contacts')])
                                         for rb, ra in zip(rows_b, rows_a)])
            ok = (a['finished'] >= b['finished'] and a['crashed'] <= b['crashed']
                  and a['chatter'] <= gp['chatter_ratio_max']*b['chatter']
                  and ratio is not None and ratio <= 1+gp['time_slower_max']
                  and a['contacts'] <= b['contacts']+gp['contacts_extra_max']
                  and a['high_passes'] <= b['high_passes']+gp['high_passes_extra_max'])
            result['gates'][f'{part}_{name}'] = gate(
                f"finishes >= baseline; crashes <= baseline; chatter <= {gp['chatter_ratio_max']:.2f} x baseline; "
                f"paired finish time <= baseline x {1+gp['time_slower_max']:.2f}; terrain contacts <= baseline + "
                f"{gp['contacts_extra_max']}; passes > 1.5 m high <= baseline + {gp['high_passes_extra_max']}",
                b, a, ok, paired=n, ratio=None if ratio is None else round(ratio, 4))
        result['reports'][part] = rep

    # live windows (development cases): the surrogate reproduces the live failure, and the assist changes it
    gw = g['windows']
    for name in brains:
        if not (Path(out)/output_name(name, 'windows', 'assist')).exists():
            continue
        base, assist = _load(out, name, 'windows', 'baseline'), _load(out, name, 'windows', 'assist')
        for rb, ra in zip(base['rows'], assist['rows']):
            key = f"window_{rb['flight']}_{rb['t0']:g}"
            spec = next(w for w in gates['windows']['windows'] if w['flight'] == rb['flight'] and w['t0'] == rb['t0'])
            if spec['metric'] == 'end_speed':
                repro = abs(rb['end_speed']-rb['live_end_speed']) <= gw['repro_speed_max']
                ok = ra['end_speed'] <= rb['end_speed']-gw['end_speed_drop_min']
                thr = (f"the unassisted surrogate reproduces the live speed at the window end within "
                       f"{gw['repro_speed_max']} m/s; assisted <= unassisted - {gw['end_speed_drop_min']} m/s")
            else:
                repro = abs(rb['min_height']-rb['live_min_height']) <= gw['repro_height_max']
                ok = ra['min_height'] >= rb['min_height']+gw['min_height_gain_min']
                thr = (f"the unassisted surrogate reproduces the live lowest height within {gw['repro_height_max']} m; "
                       f"assisted lowest height >= unassisted + {gw['min_height_gain_min']} m")
            result['gates'][key] = gate(thr, dict(end_speed=rb['end_speed'], min_height=rb['min_height'],
                                                  live_end_speed=rb['live_end_speed'],
                                                  live_min_height=rb['live_min_height']),
                                        dict(end_speed=ra['end_speed'], min_height=ra['min_height']), repro and ok,
                                        reproduced=bool(repro), development_case=True)

    # open-loop replays: identity and the assist's activity on clean laps
    if args.replays:
        result['gates'].update(score_replays(gates, args.replays, args.baseline_replays, args.runs, result['reports']))
    result['passed'] = {k: v['pass'] for k, v in result['gates'].items()}
    result['all_passed'] = all(result['passed'].values())
    Path(args.json).write_text(json.dumps(result, indent=1, default=float), encoding='utf-8')
    for k, v in result['passed'].items():
        print(f'{k:56s} {"pass" if v else "FAIL"}', flush=True)


def _tag(r, kind, flight):
    """The vertical_replay file tag of a variant ('full': the round-4 stack, 'default': the default pilot) for a flight
    (with '_stream' for the flights replayed with the offline looming stream)."""
    return r['tags'][kind+('_stream' if flight in r['stream_flights'] else '')]


def score_replays(gates, prefix, baseline_prefix, runs, reports):
    """Identity (this tree without the assist, and the fast PD with it, against the m4 tree's replays; the default
    pilot too) and the assist's activity on clean brain laps of Straw Bale and Pine Valley, from vertical_replay
    files. The recorded motion does not respond to the requests: removals are upper bounds of what a brain that
    responds would see."""
    from ..obstacles.vertical_replay import durations, identical, onsets
    r = gates['replays']
    ma = f'-ma{gates["motor_assist"]["version"]}'
    out = {}
    pairs = []
    for flight in r['identity_flights']:
        side = json.loads((Path(runs)/f'{flight}.json').read_text(encoding='utf-8'))
        contract = side['motor_controller'].get('contract')
        for kind in ('full', 'default'):
            tag = _tag(r, kind, flight)
            same, keys = identical(np.load(f'{prefix}_{tag}_{flight}.npz'), np.load(f'{baseline_prefix}_{tag}_{flight}.npz'))
            pairs.append(dict(flight=flight, variant=f'{tag}, assist off', identical=same, keys=keys))
        if contract == 'fast_velocity_pd_v1':
            tag = _tag(r, 'full', flight)
            same, keys = identical(np.load(f'{prefix}_{tag}{ma}_{flight}.npz'),
                                   np.load(f'{baseline_prefix}_{tag}_{flight}.npz'))
            pairs.append(dict(flight=flight, variant=f'{tag}, fast PD with --motor-assist on', identical=same, keys=keys))
    out['identity'] = gate('every pair bit-identical (per-tick request, yaw, state, cap, climb, descent scale)', None,
                           dict(pairs=len(pairs), identical=sum(bool(p['identical']) for p in pairs)),
                           all(p['identical'] for p in pairs), pairs=pairs)
    q = r['quiet']
    rows = []
    for flight in q['flights']:
        tag = _tag(r, 'full', flight)
        base = np.load(f'{prefix}_{tag}_{flight}.npz', allow_pickle=True)
        on = np.load(f'{prefix}_{tag}{ma}_{flight}.npz', allow_pickle=True)
        t = np.asarray(on['t'], float)
        dt = durations(t)
        minutes = float(t[-1]-t[0])/60
        flown = float((dt*np.hypot(on['vx'], on['vy'])).sum())
        own = np.stack([on['assist_pilot_vx'], on['assist_pilot_vy'], on['assist_pilot_vz']], -1)
        plain = np.stack([base['cvx'], base['cvy'], base['cvz']], -1)
        hb, ha = np.hypot(own[:, 0], own[:, 1]), np.hypot(on['cvx'], on['cvy'])
        src = np.asarray(on['assist_source']).astype(str)
        vertical = np.nan_to_num(np.asarray(on['assist_vertical'], float))
        rows.append(dict(flight=flight, minutes=round(minutes, 2), flown_m=round(flown, 1),
                         turn_first_episodes=[len(onsets(np.nan_to_num(base['turn_first']) > 0)),
                                              len(onsets(np.nan_to_num(on['turn_first']) > 0))],
                         removed_m=round(float((dt*np.maximum(hb-ha, 0)).sum()), 2),
                         removed_share=round(float((dt*np.maximum(hb-ha, 0)).sum())/max(flown, 1e-9), 4),
                         active_s_per_min={s: round(float(dt[src == s].sum())/minutes, 3)
                                           for s in ('request', 'governor', 'turn_first', 'stopping')},
                         climb_added_m_per_min=round(float((dt*vertical).sum())/minutes, 3),
                         sag_s_per_min=round(float(dt[vertical > 0].sum())/minutes, 3),
                         own_request_changed_s=round(float(dt[np.any(np.abs(own-plain) > 1e-9, -1)].sum()), 2)))
    ok = all(x['turn_first_episodes'][1] <= x['turn_first_episodes'][0]
             and x['removed_share'] <= q['removed_share_max']
             and x['active_s_per_min']['stopping'] <= q['stopping_s_per_min_max'] for x in rows)
    out['quiet'] = gate(f"per clean lap log: no more turn-first episodes than without the assist; horizontal request "
                        f"removed <= {q['removed_share_max']:.0%} of the flown distance; the stopping bound active <= "
                        f"{q['stopping_s_per_min_max']} s per minute (open loop: upper bounds)", None, rows, ok)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--gates', default=str(GATES_PATH))
    run.add_argument('--dir', required=True)
    run.add_argument('--parts', nargs='*', default=None, choices=PARTS)
    run.add_argument('--only', nargs='*', default=None, help='controller names to run')
    run.add_argument('--rest', type=float, default=3., help='seconds of rest between batches (thermal duty cycle)')
    sc = sub.add_parser('score')
    sc.add_argument('--gates', default=str(GATES_PATH))
    sc.add_argument('--dir', required=True)
    sc.add_argument('--replays', default=None, help="this tree's replay prefix (vertical_replay --out)")
    sc.add_argument('--baseline-replays', default=None, help="the m4 tree's replay prefix")
    sc.add_argument('--runs', default='C:/DEV/Haltere/runs/fast-stack-20260923')
    sc.add_argument('--json', required=True)
    args = parser.parse_args()
    (main_run if args.command == 'run' else score)(args)


if __name__ == '__main__':
    main()
