"""Score the frozen wall-pilot (turn-first version 4) gates from open-loop replay files (offline; never flight evidence).

The replays are made with haltere/obstacles/vertical_replay.py (one npz per flight and variant, named
{prefix}_{variant_tag}_{flight}.npz) for this tree (``--out``) and for the baseline tree named in the gates file
(``--baseline``). The gates are declared in configs/obstacles/wall_pilot_gates.json and are scored only when that file
is frozen and names the wall-pilot declaration this tree carries. Hindsight (the logged end of a flight at an impact,
the wall plane found from the logs) is used for scoring only; the rules never see it.

usage: python -m haltere.obstacles.wall_pilot_gates --out PREFIX --baseline PREFIX --json SCORES.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from haltere.obstacles.vertical_replay import COMMAND_KEYS, RUNS, durations, identical, onsets, variant_tag

ROOT = Path(__file__).resolve().parents[2]
GATES_PATH = ROOT/'configs'/'obstacles'/'wall_pilot_gates.json'
WALL_PILOT_PATH = ROOT/'configs'/'obstacles'/'wall_pilot.json'
EPS = 1e-9


def load_gates(path=GATES_PATH, wall_pilot=WALL_PILOT_PATH):
    """The frozen gates and their content hash; refuses an unfrozen or edited file and gates written for another
    wall-pilot declaration than the one in this tree."""
    from haltere.liftoff.gap_stack import config_sha256
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = config_sha256(obj)
    if obj.get('frozen') is not True or obj.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    declared = json.loads(Path(wall_pilot).read_text(encoding='utf-8'))
    if obj['wall_pilot']['sha256'] != declared.get('sha256') or obj['wall_pilot']['version'] != declared.get('version'):
        raise ValueError(f'{path} scores wall-pilot version {obj["wall_pilot"]["version"]} '
                         f'({obj["wall_pilot"]["sha256"][:12]}), not the declaration in this tree')
    return obj, digest


def load(prefix, tag, flight):
    return dict(np.load(f'{prefix}_{tag}_{flight}.npz', allow_pickle=False))


def episodes(arrays, key='turn_first'):
    """(start t, end t, duration s) of every run of ticks with arrays[key] == 1."""
    t = np.asarray(arrays['t'], float)
    on = np.nan_to_num(np.asarray(arrays.get(key, np.zeros(len(t))), float)) > .5
    dt = durations(t)
    out = []
    for i in onsets(on):
        j = i
        while j+1 < len(t) and on[j+1]:
            j += 1
        out.append((round(float(t[i]), 2), round(float(t[j]), 2), round(float(dt[i:j+1].sum()), 3)))
    return out


def first_engagement(arrays, t0, t1):
    """The first tick in [t0, t1) at which a turn-first episode is active, or None."""
    t = np.asarray(arrays['t'], float)
    on = np.nan_to_num(np.asarray(arrays.get('turn_first', np.zeros(len(t))), float)) > .5
    idx = np.flatnonzero(on & (t >= t0) & (t < t1))
    return None if not len(idx) else float(t[idx[0]])


def toward(arrays, normal):
    """The requested horizontal velocity along the (hindsight) wall normal at every tick."""
    n = np.asarray(normal, float)
    n = n/np.linalg.norm(n)
    return np.asarray(arrays['cvx'], float)*n[0]+np.asarray(arrays['cvy'], float)*n[1]


def score_lead(arrays, gate, wall):
    """Engagement before the end of an impact flight (the graze/impact is the end of the log) with a lead of at least
    lead_s, and (when max_toward_mps is declared) no request toward the wall from engagement + settle_s to the end."""
    t = np.asarray(arrays['t'], float)
    end = float(t[-1])
    t_e = first_engagement(arrays, end-gate['window_s'], end)
    result = dict(end_t=round(end, 2), window=[round(end-gate['window_s'], 2), round(end, 2)],
                  first_engagement_t=None if t_e is None else round(t_e, 2),
                  lead_s=None if t_e is None else round(end-t_e, 3), episodes=episodes(arrays))
    passed = t_e is not None and end-t_e >= gate['lead_s']-EPS
    if gate.get('max_toward_mps') is not None and t_e is not None:
        span = (t >= t_e+gate['settle_s']) & (t <= end)
        worst = float(np.max(toward(arrays, wall['normal'])[span])) if span.any() else None
        result['max_toward_after_settle'] = None if worst is None else round(worst, 4)
        passed = passed and worst is not None and worst <= gate['max_toward_mps']+EPS
    result['passed'] = bool(passed)
    return result


def score_pd(mine, base, flight_log_speed, gate, wall):
    """The fast PD's hairpin pass: never pushed toward the wall more than as flown (version 3), bounded engaged time
    and a bounded open-loop delay estimate (the path deficit of the request over the corridor exit speed)."""
    t = np.asarray(mine['t'], float)
    w = (t >= gate['window'][0]) & (t <= gate['window'][1])
    # only a request toward the wall counts: max(along the normal, 0) against the baseline's
    push = np.maximum(toward(mine, wall['normal']), 0.)-np.maximum(toward(base, wall['normal']), 0.)
    dt = durations(t)
    engaged = np.nan_to_num(np.asarray(mine['turn_first'], float)) > .5
    h_mine = np.hypot(mine['cvx'], mine['cvy'])
    h_base = np.hypot(base['cvx'], base['cvy'])
    deficit = float(np.sum(np.maximum(0., h_base-h_mine)[w]*dt[w]))
    exit_speed = float(np.mean(flight_log_speed[(t >= gate['exit_window'][0]) & (t <= gate['exit_window'][1])]))
    result = dict(window=gate['window'], max_extra_toward_wall=round(float(np.max(push[w])), 4),
                  engaged_s=round(float(dt[w & engaged].sum()), 3), path_deficit_m=round(deficit, 3),
                  exit_speed=round(exit_speed, 3), delay_estimate_s=round(deficit/max(exit_speed, 1e-6), 3),
                  episodes=[e for e in episodes(mine) if gate['window'][0] <= e[0] <= gate['window'][1]],
                  base_episodes=[e for e in episodes(base) if gate['window'][0] <= e[0] <= gate['window'][1]])
    result['passed'] = bool(result['max_extra_toward_wall'] <= gate['max_extra_toward_mps']+EPS
                            and result['engaged_s'] <= gate['max_engaged_s']+EPS
                            and result['delay_estimate_s'] <= gate['max_delay_s']+EPS)
    return result


def score_v3case(arrays, gate, wall):
    """The version-3 design case: turn-first active at the crash hairpin and no request toward the wall after
    from_t."""
    t = np.asarray(arrays['t'], float)
    end = float(t[-1])
    t_e = first_engagement(arrays, gate['engaged_from_t'], end+1.)
    span = (t >= gate['from_t']) & (t <= end)
    worst = float(np.max(toward(arrays, wall['normal'])[span]))
    result = dict(first_engagement_t=None if t_e is None else round(t_e, 2), max_toward_after=round(worst, 4),
                  episodes=episodes(arrays))
    result['passed'] = bool(t_e is not None and worst <= gate['max_toward_mps']+EPS)
    return result


def score_quiet(arrays):
    t = np.asarray(arrays['t'], float)
    dt = durations(t)
    tf = np.nan_to_num(np.asarray(arrays.get('turn_first', np.zeros(len(t))), float)) > .5
    sg = np.nan_to_num(np.asarray(arrays.get('side_guard', np.zeros(len(t))), float)) > .5
    return dict(turn_first_episodes=len(onsets(tf)), turn_first_s=round(float(dt[tf].sum()), 3),
                side_guard_s=round(float(dt[sg].sum()), 3), minutes=round(float(t[-1]-t[0])/60., 2),
                passed=bool(not tf.any() and not sg.any()))


# ---------------------------------------------------------------------------------------------
# Kinematic check (report only): a delayed, rate-limited first-order motor driven by the replayed request
# ---------------------------------------------------------------------------------------------
def motor_rollout(v0, request_t, request, t0, t1, delay, tau, accel, dt=.01, p0=None):
    """Horizontal velocity (and position from p0) of a motor whose velocity follows request(t - delay) with time
    constant tau, the acceleration bounded to `accel` (norm), from v0 at t0 to t1."""
    v = np.array(v0, float)
    p = np.zeros(2) if p0 is None else np.array(p0, float)
    out_t, out_v, out_p = [], [], []
    req = np.asarray(request, float)
    for t in np.arange(t0, t1, dt):
        k = int(np.searchsorted(request_t, t-delay, side='right'))-1
        r = req[max(k, 0)]
        a = (r-v)/tau
        n = float(np.linalg.norm(a))
        if n > accel:
            a *= accel/n
        v = v+a*dt
        p = p+v*dt
        out_t.append(t+dt)
        out_v.append(v.copy())
        out_p.append(p.copy())
    return np.array(out_t), np.array(out_v), np.array(out_p)


def fit_motor(arrays, windows, grid):
    """(delay, tau, accel, rms) minimizing the horizontal velocity error of 1 s rollouts from logged states driven by
    the logged request, over the declared fitting windows (motor property; no v4 request is used)."""
    t = np.asarray(arrays['t'], float)
    req = np.stack([arrays['log_cvx'], arrays['log_cvy']], 1)
    vel = np.stack([arrays['vx'], arrays['vy']], 1)
    best = None
    starts = [s for a, b in windows for s in np.arange(a, b-1., .5)]
    for delay in grid['delay_s']:
        for tau in grid['tau_s']:
            for accel in grid['accel_mps2']:
                errors = []
                for s in starts:
                    i = int(np.searchsorted(t, s))
                    rt, rv, _ = motor_rollout(vel[i], t, req, t[i], t[i]+1., delay, tau, accel)
                    logged = np.stack([np.interp(rt, t, vel[:, 0]), np.interp(rt, t, vel[:, 1])], 1)
                    errors.append(np.mean(np.sum((rv-logged)**2, 1)))
                rms = float(np.sqrt(np.mean(errors)))
                if best is None or rms < best[3]:
                    best = (float(delay), float(tau), float(accel), rms)
    return best


def kinematic_report(mine, base, spec, wall):
    """Rollouts of the fitted motor from the logged state at spec['from_t'] driven by the version-4 request (this
    tree) and by the as-flown request (baseline), with the logged position; the crossing of the wall plane."""
    fit = fit_motor(base, spec['fit_windows'], spec['grid'])
    delay, tau, accel, rms = fit
    t = np.asarray(base['t'], float)
    i = int(np.searchsorted(t, spec['from_t']))
    v0 = [base['vx'][i], base['vy'][i]]
    p0 = [base['x'][i], base['y'][i]]
    n = np.asarray(wall['normal'], float)
    n = n/np.linalg.norm(n)
    out = dict(fit=dict(delay_s=delay, tau_s=tau, accel_mps2=accel, rms_mps=round(rms, 4)),
               from_t=spec['from_t'], wall_plane=wall['plane_offset_m'])
    logged_t = t
    logged_along = np.asarray(base['x'], float)*n[0]+np.asarray(base['y'], float)*n[1]
    for name, arrays in (('as_flown', base), ('version_4', mine)):
        req = np.stack([arrays['cvx'], arrays['cvy']], 1)
        rt, rv, rp = motor_rollout(v0, np.asarray(arrays['t'], float), req, t[i], spec['to_t'], delay, tau, accel,
                                   p0=p0)
        along = rp @ n
        speed_toward = rv @ n
        cross = np.flatnonzero(along >= wall['plane_offset_m'])
        out[name] = dict(max_along=round(float(along.max()), 3),
                         along_at_end_of_log=round(float(np.interp(t[-1], rt, along)), 3),
                         crosses_plane_t=None if not len(cross) else round(float(rt[cross[0]]), 2),
                         speed_toward_at_crossing=None if not len(cross) else round(float(speed_toward[cross[0]]), 3),
                         speed_toward_at_end_of_log=round(float(np.interp(t[-1], rt, speed_toward)), 3))
    out['logged_along_at_end'] = round(float(logged_along[-1]), 3)
    out['logged_t_end'] = round(float(logged_t[-1]), 2)
    return out


# ---------------------------------------------------------------------------------------------
def score_all(out, baseline, gates_path=GATES_PATH, runs=RUNS):
    gates, digest = load_gates(gates_path)
    g = gates['gates']
    wall = gates['hairpin_wall']
    result = dict(gates_sha256=digest, wall_pilot=gates['wall_pilot'], baseline_tree=gates['baseline_tree'])
    minus = gates['flights']['logged']
    straw = gates['flights']['straw_stream']
    vertical = {f: ('on' if f in gates['flights']['flown_with_vertical_guard'] else 'off') for f in minus}

    def tag(stack, wall_mode, vert, stream=False):
        return variant_tag(stack, wall_mode, vert, stream)

    # Identity with the baseline tree (commands bit for bit)
    rows = []
    for f in minus:
        for t_ in (tag('none', 'off', 'off'), tag('flown', 'off', vertical[f]),
                   tag('shadow', 'shadow', 'shadow')):
            rows.append((f, t_, 'gate'))
        rows.append((f, tag('flown', 'on', vertical[f]), 'report'))
    for f in straw:
        for t_ in (tag('none', 'off', 'off'), tag('none', 'off', 'off', True), tag('shadow', 'shadow', 'shadow', True)):
            rows.append((f, t_, 'gate'))
    identity = []
    for f, t_, kind in rows:
        try:
            same, keys = identical(load(out, t_, f), load(baseline, t_, f))
        except FileNotFoundError as exc:
            same, keys = None, str(exc)
        identity.append(dict(flight=f, variant=t_, kind=kind, identical=same, keys=keys))
    gate_rows = [r for r in identity if r['kind'] == 'gate']
    result['W_Identity'] = dict(pairs=identity, passed=bool(gate_rows and all(r['identical'] for r in gate_rows)))
    # Shadow: the wall rules in shadow leave the command unchanged (this tree)
    shadow = []
    for f in minus:
        on_shadow = load(out, tag('flown', 'shadow', vertical[f]), f)
        same, keys = identical(on_shadow, load(out, tag('flown', 'off', vertical[f]), f))
        shadow.append(dict(flight=f, identical=same, keys=keys, shadow_episodes=episodes(on_shadow)))
    result['W_Shadow'] = dict(flights=shadow, passed=all(r['identical'] for r in shadow))
    # Brain-09b hairpin graze and brain-08 approach impact
    for name in ('W_B09', 'W_B08'):
        spec = g[name]
        f = spec['flight']
        mine = load(out, tag('flown', 'on', vertical[f]), f)
        base = load(baseline, tag('flown', 'on', vertical[f]), f)
        result[name] = dict(score_lead(mine, spec, wall), flight=f,
                            baseline_report_only=score_lead(base, dict(spec, max_toward_mps=None), wall))
    # Fast PD hairpin pass
    spec = g['W_PD']
    f = spec['flight']
    mine = load(out, tag('flown', 'on', vertical[f]), f)
    base = load(baseline, tag('flown', 'on', vertical[f]), f)
    speed = np.hypot(np.asarray(mine['vx'], float), np.asarray(mine['vy'], float))
    result['W_PD'] = dict(score_pd(mine, base, speed, spec, wall), flight=f)
    # Version-3 design case (fast PD side push, minus-fast6-gapon-01)
    spec = g['W_V3case']
    f = spec['flight']
    result['W_V3case'] = dict(score_v3case(load(out, tag('flown', 'on', vertical[f]), f), spec, wall), flight=f,
                              baseline_report_only=score_v3case(load(baseline, tag('flown', 'on', vertical[f]), f),
                                                                spec, wall))
    # Quiet on clean Straw Bale laps and on Pine Valley
    quiet = {}
    for f in g['W_Quiet']['pine']:
        quiet[f] = dict(full_stack=score_quiet(load(out, tag('on', 'on', 'on'), f)),
                        as_flown_with_wall=score_quiet(load(out, tag('flown', 'on', vertical.get(f, 'off')), f)))
    for f in g['W_Quiet']['straw']:
        quiet[f] = dict(full_stack=score_quiet(load(out, tag('on', 'on', 'on', True), f)))
    result['W_Quiet'] = dict(flights=quiet, passed=all(v['passed'] for q in quiet.values() for v in q.values()))
    # Report only: every episode of every logged-looming flight, this tree and the baseline, as flown + wall rules
    report = {}
    for f in minus:
        mine = load(out, tag('flown', 'on', vertical[f]), f)
        base = load(baseline, tag('flown', 'on', vertical[f]), f)
        diff = np.hypot(mine['cvx']-base['cvx'], mine['cvy']-base['cvy'])
        dt = durations(mine['t'])
        report[f] = dict(version_4=episodes(mine), version_3=episodes(base),
                         side_guard=episodes(mine, 'side_guard'),
                         request_changed_s=round(float(dt[diff > 1e-9].sum()), 3),
                         max_horizontal_change=round(float(diff.max()), 3))
    result['episodes_report_only'] = report
    spec = gates['kinematic_report_only']
    f = spec['flight']
    result['kinematic_report_only'] = kinematic_report(load(out, tag('flown', 'on', vertical[f]), f),
                                                       load(baseline, tag('flown', 'on', vertical[f]), f), spec, wall)
    result['passed'] = {k: result[k]['passed'] for k in ('W_Identity', 'W_Shadow', 'W_B09', 'W_B08', 'W_PD',
                                                         'W_V3case', 'W_Quiet')}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help="this tree's replay prefix")
    parser.add_argument('--baseline', required=True, help="the baseline tree's replay prefix")
    parser.add_argument('--gates', default=str(GATES_PATH))
    parser.add_argument('--runs', default=str(RUNS))
    parser.add_argument('--json', required=True)
    args = parser.parse_args(argv)
    result = score_all(args.out, args.baseline, args.gates, args.runs)
    Path(args.json).write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    print(json.dumps(result['passed']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
