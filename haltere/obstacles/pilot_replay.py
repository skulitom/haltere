"""Open-loop replay of live fast-stack logs through FastRaceCue (offline; report only, never flight evidence).

A committed port of the M2 round-2 harness (scratchpad m2r2/pilot/replay_rules.py) with the free-space planner.
Each logged controller tick is fed to the FastRaceCue of a code tree (``--tree``; default: this checkout) as the
runner fed it: the recorded pose/velocity/attitude/rates, the ring cue of that tick (u, v, edge, aim_u) with its
capture time, the logged looming sample (capture time = now - looming_age; inf -> None, NaN -> no sample), the
logged gap sample of obstacle-stack flights (capture time = now - gap_age) and, with ``--plan-stream``, planner
samples. ``now`` is the controller clock at the tick (capture_time + image_age). The camera pose history gets
(frame_time, pose) whenever the telemetry timestamp changes, as in the runner. The recorded motion does not respond
to the replayed requests: the output is what each pilot variant would have REQUESTED at each recorded state.

Variants:
  --stack none     the default fast pilot (the logged looming samples still reach the TTC governor)
  --stack flown    as flown: gap aim + lag turn when the flight used --obstacle-stack on (m2r2 'flown'), with
                   --wall off|on|shadow (m2r2 'flown', 'wall', 'shadow')
  --stack on       the full stack declared for the flight's motor contract (gap aim, lag turn, wall rules)
  --stack shadow   the full stack computed and logged, nothing applied (--obstacle-stack shadow)
  --planner off|shadow|on   the free-space corridor planner (needs --stack flown/on/shadow; shadow and on leave the
                   gap aim unapplied); ``--gap-apply off`` builds the planner-off control of the planner shadow.
Streams: ``--plan-stream`` an npz per flight (``{flight}`` in the path) with one array per camera_process.PLAN_FIELDS
entry and ``pub_time`` (capture + the declared offline age 0.10 s); a sample is published at the first tick whose
now >= pub_time. ``source`` in the npz labels its origin; anything but 'perception' is reported as a plumbing check,
never as gate evidence. ``--looming-stream`` replaces the logged looming samples (flights flown without looming):
t_wall, ttc, distance, below_fraction, ttc_lower, received DELAY (0.085 s) after capture.

Gates V1-V3 (``--score-v``, configs/obstacles/free_space_gates.json, frozen before scoring) read the replay of
variant ``--stack on --planner on``; hindsight (the logged impact, the logged governor climb, telemetry height) is
used for scoring only.

usage: python haltere/obstacles/pilot_replay.py --out PREFIX [--tree TREE] [--stack ...] [--planner ...] flight ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

RUNS = Path('C:/DEV/Haltere/runs/fast-stack-20260923')
DELAY = .085
PLAN_OFFLINE_AGE = .10
# The runner's pad calibration of the original drone (identical in every fast-stack sidecar), used by the support rule.
CALIBRATION = dict(hover_processed=0.13566076755523682, throttle_scale=0.8, hover_stick_sim=-0.43019758448357537)
STATE_CODES = ('', 'launch', 'wait', 'search', 'coast', 'side', 'below', 'below_weak', 'above', 'cue', 'support_climb')
GATES_PATH = Path(__file__).resolve().parents[2]/'configs'/'obstacles'/'free_space_gates.json'


def finite(v):
    return None if v is None or not np.isfinite(v) else float(v)


def edge_flag(v):
    return str(v).strip().lower() in ('true', '1', '1.0')


def _modules():
    from haltere.liftoff import fast_race_cue as frc
    from haltere.liftoff.camera_pose import CameraPoseHistory
    from haltere.liftoff.gap_aim import GapAimConfig
    return frc, CameraPoseHistory, GapAimConfig


def planner_configs(tree):
    """(CorridorAimConfig, VerticalGuardConfig) from the tree's frozen free-space declaration when it exists, else
    the declared defaults (which equal the v1 specification)."""
    from haltere.liftoff import corridor_aim as ca
    path = Path(tree)/'configs'/'obstacles'/'free_space.json'
    if path.exists():
        parsed = ca.planner_pilot_configs(json.loads(path.read_text(encoding='utf-8')))
        return parsed['corridor'], parsed['vertical'], str(path)
    return ca.CorridorAimConfig(), ca.VerticalGuardConfig(), 'defaults (free_space.json not in this tree)'


def build(side, tree, stack='flown', wall='off', planner='off', gap_apply='auto'):
    """The FastRaceCue variant for a flight's sidecar and its description."""
    frc, CameraPoseHistory, GapAimConfig = _modules()
    ob = Path(tree)/'configs'/'obstacles'
    contract = side['motor_controller'].get('contract') or 'fast_velocity_brain_v1'
    flown_on = (side.get('obstacle_stack') or {}).get('mode') == 'on'
    kw, applied = {}, True
    components = stack in ('on', 'shadow') or (stack == 'flown' and flown_on)
    if stack not in ('none', 'flown', 'on', 'shadow'):
        raise ValueError(stack)
    if components:
        kw['lag_turn'] = frc.lag_turn_for_contract(json.loads((ob/'lag_turn.json').read_text(encoding='utf-8')),
                                                   contract)
        kw['gap_aim'] = GapAimConfig.from_dict(json.loads((ob/'gap_pilot.json').read_text(encoding='utf-8'))['pilot'])
    walls = 'on' if stack == 'on' else 'shadow' if stack == 'shadow' else wall if stack == 'flown' else 'off'
    if walls in ('on', 'shadow'):
        kw.update(frc.wall_pilot_configs(json.loads((ob/'wall_pilot.json').read_text(encoding='utf-8'))))
    if stack == 'shadow':
        applied = False
        kw.update(lag_turn_apply=False, gap_apply=False)
    if walls != 'off':
        kw['wall_apply'] = walls == 'on' and applied
    source = None
    if planner != 'off':
        if stack == 'none':
            raise ValueError('The planner is part of the obstacle stack: use --stack flown, on or shadow')
        corridor, vertical, source = planner_configs(tree)
        kw.update(corridor_aim=corridor, vertical_guard=vertical, planner_apply=planner == 'on' and applied)
        if 'gap_aim' in kw:
            kw['gap_apply'] = False
    if gap_apply == 'off' and 'gap_aim' in kw:
        kw['gap_apply'] = False
    pa = side['pilot_assistance']
    speed = float(pa.get('nominal_speed_mps') or 6.)
    reference = float(pa.get('trained_motor_reference_mps') or speed)
    yaw_curve = tuple(pa.get('yaw_curve') or frc.DEFAULT_YAW_CURVE)
    pilot = frc.FastRaceCue(side['gate_sensor'], CameraPoseHistory(), speed, reference_speed=reference,
                            yaw_curve=yaw_curve, calibration=CALIBRATION, **kw)
    return pilot, dict(contract=contract, flown_stack=flown_on, stack=stack, wall=walls, planner=planner,
                       gap_apply=kw.get('gap_apply', 'gap_aim' in kw), planner_config=source)


class PlanStream:
    """Planner samples of one flight published at pub_time (camera_process.PLAN_FIELDS arrays + pub_time)."""

    def __init__(self, path):
        from haltere.liftoff.camera_process import PLAN_FIELDS, PLAN_KINDS
        z = np.load(path, allow_pickle=False)
        self.path = str(path)
        self.source = str(z['source']) if 'source' in z.files else 'unlabelled'
        self.fields = PLAN_FIELDS
        self.columns = {k: z[k] for k in PLAN_FIELDS if k in z.files}
        missing = [k for k in ('time', 'kind', 'valid') if k not in self.columns]
        if missing:
            raise ValueError(f'{path} lacks {missing}')
        self.pub = z['pub_time'] if 'pub_time' in z.files else self.columns['time']+PLAN_OFFLINE_AGE
        order = np.argsort(self.pub, kind='stable')
        self.pub = self.pub[order]
        self.columns = {k: v[order] for k, v in self.columns.items()}
        kinds = self.columns['kind']
        self.kinds = ([PLAN_KINDS[int(k)] if np.isfinite(k) and 0 <= int(k) < len(PLAN_KINDS) else '' for k in kinds]
                      if np.issubdtype(kinds.dtype, np.number) else [str(k) for k in kinds])
        self.pointer = 0
        self.latest = None
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def at(self, now):
        """The newest sample published by `now` (a dict as camera_process.plan_sample returns), or None."""
        index = None
        while self.pointer < len(self.pub) and self.pub[self.pointer] <= now:
            index, self.pointer = self.pointer, self.pointer+1
        if index is not None:
            sample = {}
            for k in self.fields:
                v = self.columns[k][index] if k in self.columns else np.nan
                sample[k] = None if not np.isfinite(v) else float(v)
            sample.update(time=float(self.columns['time'][index]), kind=self.kinds[index],
                          valid=bool(self.columns['valid'][index] == 1))
            self.latest = sample
        return self.latest


def replay(flight, tree, runs=RUNS, *, stack='flown', wall='off', planner='off', gap_apply='auto', plan_stream=None,
           looming_stream=None):
    """Per-tick arrays of one flight replayed through one variant, and the pilot and description."""
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    side = json.loads((Path(runs)/f'{flight}.json').read_text(encoding='utf-8'))
    pilot, info = build(side, tree, stack, wall, planner, gap_apply)
    history = pilot.pose_history
    d = pd.read_csv(Path(runs)/f'{flight}.csv', low_memory=False)
    have_gap = 'gap_age' in d and 'gap_aim' in _pilot_components(pilot)
    have_loom = 'looming_age' in d
    stream = None
    if looming_stream:
        z = np.load(looming_stream.replace('{flight}', flight))
        stream = dict(t=z['t_wall'], ttc=z['ttc'], distance=z['distance'], below=z['below_fraction'],
                      lower=z['ttc_lower'])
        have_loom, pointer = False, 0
    plans = PlanStream(plan_stream.replace('{flight}', flight)) if plan_stream else None
    has_plan = hasattr(pilot, 'plan_log')
    rows = {k: [] for k in ('t', 'now', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'cvx', 'cvy', 'cvz', 'yaw_cmd', 'state',
                            'log_cvx', 'log_cvy', 'log_cvz', 'log_state', 'log_climb', 'cap', 'climb')}
    plan_rows = {}
    last_ts = None
    for r in d.itertuples(index=False):
        if not (np.isfinite(r.capture_time) and np.isfinite(r.image_age)):
            continue
        now = float(r.capture_time)+float(r.image_age)
        pos, q, vel = np.array([r.x, r.y, r.z]), np.array([r.qw, r.qx, r.qy, r.qz]), np.array([r.vx, r.vy, r.vz])
        if r.ts != last_ts:
            history.append(float(r.frame_time), pos, q)
            last_ts = r.ts
        senses = dict(pos=torch.tensor(pos[None]), vel_world=torch.tensor(vel[None]), quat=torch.tensor(q[None]))
        cue = None
        if np.isfinite(r.cue_u) and 0 <= r.cue_u <= 1:
            cue = dict(u=float(r.cue_u), v=float(r.cue_v), edge=edge_flag(r.cue_edge), aim_u=float(r.cue_aim_u))
        clearance = None
        if stream is not None:
            fresh = None
            while pointer < len(stream['t']) and stream['t'][pointer]+DELAY <= float(r.wall):
                fresh, pointer = pointer, pointer+1
            if fresh is not None:
                k = fresh
                clearance = dict(time=float(stream['t'][k])+(now-float(r.wall)), ttc=finite(stream['ttc'][k]),
                                 distance=finite(stream['distance'][k]), below_fraction=finite(stream['below'][k]),
                                 ttc_lower=finite(stream['lower'][k]))
        if have_loom and np.isfinite(r.looming_age):
            clearance = dict(time=now-float(r.looming_age), ttc=finite(r.looming_ttc),
                             distance=finite(r.looming_distance), below_fraction=finite(r.looming_below_fraction),
                             ttc_lower=finite(r.looming_ttc_lower))
        gap = None
        if have_gap and np.isfinite(r.gap_age):
            kind = r.gap_kind if isinstance(r.gap_kind, str) else ''
            gap = dict(time=now-float(r.gap_age), shift=finite(r.gap_shift) or 0., kind=kind,
                       valid=bool(r.gap_valid == 1 or r.gap_valid == '1'), ring_deg=finite(r.gap_ring_deg),
                       lr=finite(r.gap_lr), near_on_path=None, confirmed=False)
        omega = np.array([r.omega_x, r.omega_y, r.omega_z])
        extra = dict(clearance=clearance) if clearance is not None else {}
        if gap is not None:
            extra['gap'] = gap
        if plans is not None:
            sample = plans.at(now)
            if sample is not None:
                extra['plan'] = sample
        pilot.update(senses, omega, dict(race_cue=cue) if cue else None, float(r.capture_time), now, **extra)
        pilot.issued_throttle = float(r.thr)
        gov = pilot.clearance
        values = dict(t=float(r.phase), now=now, x=r.x, y=r.y, z=r.z, vx=r.vx, vy=r.vy, vz=r.vz,
                      cvx=pilot.velocity_command[0], cvy=pilot.velocity_command[1], cvz=pilot.velocity_command[2],
                      yaw_cmd=pilot.pilot.sight_yaw, state=pilot.state, log_cvx=r.cmd_vx, log_cvy=r.cmd_vy,
                      log_cvz=r.cmd_vz, log_state=r.pilot_state if isinstance(r.pilot_state, str) else '',
                      log_climb=getattr(r, 'clearance_climb', np.nan),
                      cap=np.nan if gov is None or gov.cap is None else gov.cap,
                      climb=np.nan if gov is None else gov.climb)
        for k, v in values.items():
            rows[k].append(v)
        if has_plan and pilot.corridor is not None:
            log = pilot.plan_log()
            latest = pilot.corridor.latest or {}
            log.update(plan_h_floor=finite(latest.get('h_floor')) or np.nan,
                       plan_h_ceil=finite(latest.get('h_ceil')) or np.nan,
                       plan_d_h=finite(latest.get('d_h')) or np.nan,
                       plan_time=finite(latest.get('time')) or np.nan,
                       plan_rise_confirmed=float(pilot.vertical.rise_confirmed),
                       plan_vertical_applied=float(pilot.plan_vertical_applied))
            for k, v in log.items():
                plan_rows.setdefault(k, []).append(v)
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays.update({k: np.asarray(v) for k, v in plan_rows.items()})
    info.update(flight=flight, ticks=len(arrays['t']), stop_reason=side.get('stop_reason'), impact=side.get('impact'),
                plan_stream=None if plans is None else dict(path=plans.path, sha256=plans.sha256, source=plans.source),
                haltere=_haltere_file())
    return arrays, pilot, info


def _pilot_components(pilot):
    return {name for name in ('gap_aim', 'lag_turn', 'turn_first', 'ceiling_guard', 'corridor')
            if getattr(pilot, name, None) is not None}


def _haltere_file():
    import haltere
    return str(Path(haltere.__file__).resolve().parent)


def summary(arrays, info):
    horizontal = np.hypot(arrays['cvx']-arrays['log_cvx'], arrays['cvy']-arrays['log_cvy'])
    return dict(info, end_t=round(float(arrays['t'][-1]), 2),
                cmd_vs_log_horizontal_p50=round(float(np.nanpercentile(horizontal, 50)), 4),
                cmd_vs_log_horizontal_p99=round(float(np.nanpercentile(horizontal, 99)), 4),
                cmd_vs_log_vertical_p99=round(float(np.nanpercentile(abs(arrays['cvz']-arrays['log_cvz']), 99)), 4),
                state_match=round(float((arrays['state'] == arrays['log_state']).mean()), 4))


COMMAND_KEYS = ('cvx', 'cvy', 'cvz', 'yaw_cmd', 'state', 'cap', 'climb')


def identical(a, b, keys=COMMAND_KEYS):
    """Bitwise equality of the per-tick command arrays of two replays (NaN equal to NaN)."""
    out = {}
    for k in keys:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.dtype.kind in 'fc' and y.dtype.kind in 'fc':
            same = x.shape == y.shape and np.array_equal(x, y, equal_nan=True)
        else:
            same = x.shape == y.shape and bool(np.all(x == y))
        out[k] = bool(same)
    return all(out.values()), out


# ---------------------------------------------------------------------------------------------
# Gates V1-V3 (frozen in configs/obstacles/free_space_gates.json before scoring)
# ---------------------------------------------------------------------------------------------
def load_gates(path=GATES_PATH):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    from haltere.liftoff.gap_stack import config_sha256
    digest = config_sha256(obj)
    if obj.get('frozen') is not True or obj.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    return obj, digest


def _impact_time(arrays, info):
    """Phase time of the logged impact (the end of a flight stopped by an impact), or None."""
    reason = str(info.get('stop_reason') or '')
    return float(arrays['t'][-1]) if 'Impact' in reason else None


def _episodes(mask, t):
    """Seconds covered by a boolean tick mask (at the replay's own tick spacing)."""
    if not mask.any():
        return 0.
    dt = np.diff(t, append=t[-1])
    return float(dt[mask].sum())


def score_v1(arrays, info, gate):
    """V1 ceiling: from 0.5 s before the logged governor climb onset to the ceiling impact."""
    t = arrays['t']
    impact = _impact_time(arrays, info)
    climb = np.nan_to_num(np.asarray(arrays['log_climb'], float)) > 0
    before = climb & (t <= (impact if impact is not None else t[-1]))
    onsets = np.flatnonzero(before & ~np.r_[False, before[:-1]])
    if impact is None or not len(onsets):
        return dict(scored=False, reason='no logged impact or governor climb onset')
    start = float(t[onsets[-1]])-float(gate['before_onset_s'])
    window = (t >= start) & (t <= impact)
    vz = arrays['cvz'][window]
    age = arrays['now'][window]-arrays['plan_time'][window]
    h_ceil_now = arrays['plan_h_ceil'][window]-arrays['vz'][window]*age
    low = np.isfinite(h_ceil_now) & (h_ceil_now < gate['h_ceil_low_m'])
    v_class = arrays['plan_cls_confirmed'][window] == 2
    el = np.radians(arrays['plan_intended_el'][window])
    reach = np.nan_to_num(arrays['plan_d_h'][window], nan=0.)*np.tan(el)+gate['r_vert_m']
    into = v_class & np.isfinite(arrays['plan_h_ceil'][window]) & (reach >= arrays['plan_h_ceil'][window])
    result = dict(scored=True, window=[round(start, 2), round(impact, 2)], ticks=int(window.sum()),
                  max_vz=round(float(np.nanmax(vz)), 3), low_ceiling_ticks=int(low.sum()),
                  max_vz_low_ceiling=round(float(np.nanmax(vz[low])), 3) if low.any() else None,
                  v_class_into_ceiling_ticks=int(into.sum()))
    result['passed'] = bool(result['max_vz'] <= gate['max_vz'] and (not low.any() or result['max_vz_low_ceiling']
                                                                     <= gate['max_vz_low_ceiling'])
                            and result['v_class_into_ceiling_ticks'] == 0)
    return result


def score_v2(arrays, info, gate, window):
    """V2 floor sink: no hard climb from a flat floor; no descent below 0.5 m; the governor climb displaced."""
    t = arrays['t']
    if window == 'last_3s':
        mask = t >= t[-1]-3.
    else:
        mask = (t >= window[0]) & (t <= window[1])
    vz = arrays['cvz'][mask]
    height = arrays['z'][mask]
    rise = arrays['plan_rise_confirmed'][mask] > 0
    descending = arrays['vz'][mask] < -gate['descending_mps']
    flat_high = (height >= gate['floor_clear_m']) & ~rise
    low_descending = (height < gate['floor_clear_m']) & descending
    logged = np.nan_to_num(np.asarray(arrays['log_climb'], float)[mask]) > 0
    displaced = arrays['plan_vertical_applied'][mask] > 0
    result = dict(ticks=int(mask.sum()), max_vz=round(float(np.nanmax(vz)), 3),
                  max_vz_flat_high=round(float(np.nanmax(vz[flat_high])), 3) if flat_high.any() else None,
                  min_vz_low_descending=round(float(np.nanmin(vz[low_descending])), 3) if low_descending.any() else None,
                  logged_climb_ticks=int(logged.sum()),
                  displaced_fraction=round(float(displaced[logged].mean()), 3) if logged.any() else None)
    result['passed'] = bool(result['max_vz'] <= gate['max_vz']
                            and (result['max_vz_flat_high'] is None or result['max_vz_flat_high'] <= gate['max_vz_flat'])
                            and (result['min_vz_low_descending'] is None or result['min_vz_low_descending'] >= 0.)
                            and (result['displaced_fraction'] is None
                                 or result['displaced_fraction'] >= gate['displaced_fraction']))
    return result


def score_v3(arrays, info, gate):
    """V3 mound (dev, no-regression): climb or side-step while the logged governor climbed; no late descent."""
    t = arrays['t']
    logged = np.nan_to_num(np.asarray(arrays['log_climb'], float)) > 0
    answered = (arrays['cvz'] >= gate['min_vz']) | ((np.abs(arrays['plan_applied_az']) >= gate['min_az_deg'])
                                                   & np.isin(arrays['plan_cls_confirmed'], (1, -1)))
    covered, total = _episodes(logged & answered, t), _episodes(logged, t)
    impact = _impact_time(arrays, info)
    end = impact if impact is not None else float(t[-1])
    late = t >= end-gate['last_s']
    result = dict(logged_climb_s=round(total, 2), answered_s=round(covered, 2),
                  answered_fraction=round(covered/total, 3) if total else None,
                  late_min_vz=round(float(np.nanmin(arrays['cvz'][late])), 3))
    result['passed'] = bool(result['answered_fraction'] is not None
                            and result['answered_fraction'] >= gate['answered_fraction']
                            and result['late_min_vz'] >= 0.)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('flights', nargs='+')
    parser.add_argument('--out', required=True, help='output prefix (npz per flight and a summary json)')
    parser.add_argument('--tree', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--runs', default=str(RUNS))
    parser.add_argument('--stack', choices=['none', 'flown', 'on', 'shadow'], default='flown')
    parser.add_argument('--wall', choices=['off', 'on', 'shadow'], default='off')
    parser.add_argument('--planner', choices=['off', 'shadow', 'on'], default='off')
    parser.add_argument('--gap-apply', choices=['auto', 'off'], default='auto')
    parser.add_argument('--plan-stream', default=None)
    parser.add_argument('--looming-stream', default=None)
    parser.add_argument('--score-v', action='store_true')
    args = parser.parse_args(argv)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    sys.path.insert(0, str(Path(args.tree).resolve()))
    tag = f'{args.stack}-{args.wall}-{args.planner}' + ('-gapoff' if args.gap_apply == 'off' else '')
    results = {}
    for flight in args.flights:
        arrays, pilot, info = replay(flight, args.tree, args.runs, stack=args.stack, wall=args.wall,
                                     planner=args.planner, gap_apply=args.gap_apply, plan_stream=args.plan_stream,
                                     looming_stream=args.looming_stream)
        np.savez_compressed(f'{args.out}_{tag}_{flight}.npz', **arrays)
        results[flight] = summary(arrays, info)
        planner_meta = pilot.metadata().get('planner')
        results[flight]['planner_metadata'] = planner_meta
        print(flight, json.dumps({k: v for k, v in results[flight].items() if k != 'planner_metadata'},
                                 default=str), flush=True)
    if args.score_v:
        gates, digest = load_gates()
        v = gates['gates']
        scores = {}
        for flight in args.flights:
            arrays = dict(np.load(f'{args.out}_{tag}_{flight}.npz', allow_pickle=False))
            info = results[flight]
            label = ('GATE' if (info.get('plan_stream') or {}).get('source') == 'perception'
                     else 'PLUMBING CHECK (synthetic stream): not a gate result, not evidence')
            s = {}
            if flight in v['V1']['flights']:
                s['V1'] = score_v1(arrays, info, v['V1'])
            if flight in v['V2']['windows']:
                s['V2'] = score_v2(arrays, info, v['V2'], v['V2']['windows'][flight])
            if flight in v['V3']['flights']:
                s['V3'] = score_v3(arrays, info, v['V3'])
            scores[flight] = dict(label=label, gates_sha256=digest, **s)
            print(flight, label, json.dumps(s), flush=True)
        results['v_scores'] = scores
    Path(f'{args.out}_{tag}_summary.json').write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
