"""Open-loop replay of live fast-stack logs through FastRaceCue, with the vertical guard (offline; report only).

A port of the M2 round-2 harness (session scratchpad m2r2/pilot/replay_rules.py) as committed on m3-pilot
(haltere/obstacles/pilot_replay.py), without the free-space planner and with the vertical guard
(configs/obstacles/vertical_guard.json) and its gates (configs/obstacles/vertical_guard_gates.json). Nothing here is
flight evidence.

Each logged controller tick is fed to the FastRaceCue of a code tree (``--tree``; default: this checkout) as the
runner fed it: the recorded pose/velocity/attitude/rates, the ring cue of that tick (u, v, edge, aim_u) with its
capture time, the logged looming sample (capture time = now - looming_age; inf -> None, NaN -> no sample) and the
logged gap sample of obstacle-stack flights (capture time = now - gap_age). ``now`` is the controller clock at the
tick (capture_time + image_age). The camera pose history gets (frame_time, pose) whenever the telemetry timestamp
changes, as in the runner. The recorded motion does not respond to the replayed requests: the output is what each
pilot variant would have REQUESTED at each recorded state.

Variants:
  --stack none     the default fast pilot (the logged looming samples still reach the TTC governor)
  --stack flown    as flown: gap aim + lag turn when the flight used --obstacle-stack on, with --wall off|on|shadow
                   and --vertical off|on|shadow
  --stack on       the full stack declared for the flight's motor contract (gap aim, lag turn, wall rules and, when
                   the tree has it, the vertical guard; --vertical off gives the matched control without it)
  --stack shadow   the full stack computed and logged, nothing applied (--obstacle-stack shadow)
``--looming-stream`` replaces the logged looming samples (flights flown without looming): an npz per flight
(``{flight}`` in the path) with t_wall, ttc, distance, below_fraction, ttc_lower (NaN = None), received DELAY
(0.085 s) after capture.
``--gap-pilot DECLARATION`` gives the gap aim another gap pilot declaration than the tree's (file tag -gp<stem>);
``--near-on-path`` lets the gap samples carry the logged near_on_path as the live camera's samples did (tag -nop;
the side commitment of gap pilot version 4 reads it, version 2 never did). The per-tick arrays also carry the gap
aim's target, applied shift, flown offset, mode, committed side, conflict and the sample it saw
(haltere.obstacles.gap_commit_eval scores them). ``--descent-view DECLARATION`` adds the view-keeping descent of a
frozen descent-view declaration to any variant (tag -dv<version>; the arrays gain view_sink_bound, view_withheld and
view_boost): with ``--stack on --near-on-path`` it is the full round-4 stack as `--obstacle-stack on --descent-view on`
would fly it. The gates (``score``) are versioned: version 4 scores guard v4 against the m4 tree (guard v3), with the
round-4 live flights replayed as flown (their stack variants tagged -nop-dv1). ``--motor-assist DECLARATION`` adds the
motor assist of a frozen motor-assist declaration for the flight's motor contract (tag -ma<version>; nothing for a
contract without an entry, the fast PD): the arrays gain the pilot's own request (assist_pilot_vx/vy/vz),
assist_horizontal, assist_vertical and assist_source, while cvx/cvy/cvz are the assisted request. The recorded motion
does not respond to it either.

usage: python -m haltere.obstacles.vertical_replay --out PREFIX [--tree TREE] [--stack ...] flight ...
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np

RUNS = Path('C:/DEV/Haltere/runs/fast-stack-20260923')
DELAY = .085
# The runner's pad calibration of the original drone (identical in every fast-stack sidecar), used by the support rule.
CALIBRATION = dict(hover_processed=0.13566076755523682, throttle_scale=0.8, hover_stick_sim=-0.43019758448357537)
GATES_PATH = Path(__file__).resolve().parents[2]/'configs'/'obstacles'/'vertical_guard_gates.json'
COMMAND_KEYS = ('cvx', 'cvy', 'cvz', 'yaw_cmd', 'state', 'cap', 'climb', 'descent_scale')
EPS = 1e-9


def finite(v):
    return None if v is None or not np.isfinite(v) else float(v)


def edge_flag(v):
    return str(v).strip().lower() in ('true', '1', '1.0')


def _modules():
    from haltere.liftoff import fast_race_cue as frc
    from haltere.liftoff.camera_pose import CameraPoseHistory
    from haltere.liftoff.gap_aim import GapAimConfig
    return frc, CameraPoseHistory, GapAimConfig


def resolve_vertical(stack, vertical, has_vertical=True):
    """The vertical-guard mode of a variant: off without the stack; as asked (default off) for the stack as flown;
    with the full stack on/shadow it follows the stack unless asked off (or the tree has no vertical guard)."""
    if stack == 'none':
        return 'off'
    if stack == 'flown':
        return vertical or 'off'
    if vertical == 'off' or not has_vertical:
        return 'off'
    return 'on' if stack == 'on' else 'shadow'


def variant_tag(stack, wall, vertical, stream):
    """File tag of a replay variant (the wall mode as resolved for the stack)."""
    walls = 'on' if stack == 'on' else 'shadow' if stack == 'shadow' else wall if stack == 'flown' else 'off'
    return f'{stack}-w{walls}-v{vertical}'+('-stream' if stream else '')


def build(side, tree, stack='flown', wall='off', vertical=None, gap_pilot=None, descent_view=None, contact_support=None,
          wall_pilot=None, motor_assist=None, stale_evidence=None):
    """The FastRaceCue variant for a flight's sidecar and its description. `gap_pilot`: the gap pilot declaration
    whose pilot values the gap aim uses (default: the tree's configs/obstacles/gap_pilot.json). ``descent_view``: an
    optional DescentViewConfig added to any variant; ``contact_support``: an optional ContactSupportConfig (descent
    view version 2). ``wall_pilot``: the wall-pilot declaration of the wall rules (default: the tree's
    configs/obstacles/wall_pilot.json; e.g. the kept version 4 to replay the round-4 stack as flown).
    ``motor_assist``: an optional motor-assist declaration (parsed and hash-checked) whose entry for the flight's motor
    contract is added to any variant. ``stale_evidence``: an optional stale-evidence declaration (parsed and
    hash-checked) whose rules are added to a stack variant (applied with --stack on, computed in shadow; ignored by
    --stack none and by --stack flown without the stack)."""
    frc, CameraPoseHistory, GapAimConfig = _modules()
    ob = Path(tree)/'configs'/'obstacles'
    contract = side['motor_controller'].get('contract') or 'fast_velocity_brain_v1'
    flown_on = (side.get('obstacle_stack') or {}).get('mode') == 'on'
    has_vertical = hasattr(frc, 'VerticalGuardConfig')
    if stack not in ('none', 'flown', 'on', 'shadow'):
        raise ValueError(stack)
    kw, applied = {}, stack != 'shadow'
    if stack in ('on', 'shadow') or (stack == 'flown' and flown_on):
        kw['lag_turn'] = frc.lag_turn_for_contract(json.loads((ob/'lag_turn.json').read_text(encoding='utf-8')),
                                                   contract)
        declaration = Path(gap_pilot) if gap_pilot else ob/'gap_pilot.json'
        kw['gap_aim'] = GapAimConfig.from_dict(json.loads(declaration.read_text(encoding='utf-8'))['pilot'])
    walls = 'on' if stack == 'on' else 'shadow' if stack == 'shadow' else wall if stack == 'flown' else 'off'
    verticals = resolve_vertical(stack, vertical, has_vertical)
    if verticals != 'off' and not has_vertical:
        raise SystemExit('this tree has no vertical guard')
    if walls in ('on', 'shadow'):
        declaration = json.loads((Path(wall_pilot) if wall_pilot else ob/'wall_pilot.json').read_text(encoding='utf-8'))
        # wall-pilot version 4 declares turn-first's stopping model per motor contract (the runner passes it)
        takes_contract = 'contract' in inspect.signature(frc.wall_pilot_configs).parameters
        kw.update(frc.wall_pilot_configs(declaration, contract) if takes_contract
                  else frc.wall_pilot_configs(declaration))
        kw['wall_apply'] = walls == 'on' and applied
    if verticals in ('on', 'shadow'):
        kw['vertical_guard'] = frc.vertical_guard_config(
            json.loads((ob/'vertical_guard.json').read_text(encoding='utf-8')))
        kw['vertical_apply'] = verticals == 'on' and applied
    if stale_evidence is not None and (stack in ('on', 'shadow') or (stack == 'flown' and flown_on)):
        kw.update(frc.stale_evidence_configs(stale_evidence), stale_apply=applied)
    if not applied:
        kw.update(lag_turn_apply=False, gap_apply=False)
    if descent_view is not None:
        kw['descent_view'] = descent_view
    if contact_support is not None:
        kw['contact_support'] = contact_support
    assist = None
    if motor_assist is not None:
        assist = frc.motor_assist_for_contract(motor_assist, contract)
        if assist is not None:
            kw['motor_assist'] = assist
    pa = side['pilot_assistance']
    speed = float(pa.get('nominal_speed_mps') or 6.)
    reference = float(pa.get('trained_motor_reference_mps') or speed)
    yaw_curve = tuple(pa.get('yaw_curve') or frc.DEFAULT_YAW_CURVE)
    pilot = frc.FastRaceCue(side['gate_sensor'], CameraPoseHistory(), speed, reference_speed=reference,
                            yaw_curve=yaw_curve, calibration=CALIBRATION, **kw)
    return pilot, dict(contract=contract, flown_stack=flown_on, stack=stack, wall=walls, vertical=verticals,
                       descent_view=descent_view is not None, contact_support=contact_support is not None,
                       wall_pilot=None if wall_pilot is None else str(wall_pilot), motor_assist=assist is not None,
                       stale_evidence='stale_apply' in kw)


def _near_on_path(value):
    """The logged near_on_path column (1/0/empty) as the camera's gap sample carries it (True/False/None)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(value) else bool(value)


def replay(flight, tree, runs=RUNS, *, stack='flown', wall='off', vertical=None, looming_stream=None,
           gap_pilot=None, near_on_path=False, descent_view=None, contact_support=None, throttle_column='thr',
           wall_pilot=None, motor_assist=None, sources=None, stale_evidence=None, cue_drop=None):
    """Per-tick arrays of one flight replayed through one variant, and the pilot and description.

    `gap_pilot`: the gap pilot declaration of the gap aim (default: the tree's). `near_on_path`: the gap samples
    carry the logged near_on_path, as the live camera's samples did (the default None keeps the earlier replays;
    the version 2 gap aim never reads it). ``descent_view``: an optional DescentViewConfig added to any variant;
    ``contact_support``: an optional ContactSupportConfig (descent view version 2). ``throttle_column``: the logged
    column the pilot's issued throttle is taken from after each tick (the default 'thr' keeps the earlier replays; the
    runner's pilot sees the motor's command, 'command_thr', which differs from 'thr' on fast-PD flights, where 'thr'
    is the brain in shadow). ``wall_pilot``: the wall-pilot declaration of the wall rules (default: the tree's). With
    a tree whose pilot measures the clearance brake's sink (brake_log), the arrays also carry brake_added_sink,
    brake_sink_left, brake_sink_withheld, brake_reference (the vertical request before the brake), the cap's ray
    (brake_ray_x/y/z) and braking (1 while the cap bound the request). ``motor_assist``: an optional motor-assist
    declaration (see build); with it, a list passed as ``sources`` receives each tick's binding caps
    (FastRaceCue.assist_sources) and pilot state (not saved). ``stale_evidence``: an optional stale-evidence declaration
    (see build); with it the arrays also carry cap_ray_deg and cap_reseat (FastRaceCue.stale_log). ``cue_drop``: logged
    capture times (the CSV's capture_time values) whose ring cue is replaced by no detection (a frame the reader read no
    marker in; e.g. the logged detections a reader rule rejects on the aligned recorded frame)."""
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    side = json.loads((Path(runs)/f'{flight}.json').read_text(encoding='utf-8'))
    pilot, info = build(side, tree, stack, wall, vertical, gap_pilot, descent_view, contact_support,
                        **({} if wall_pilot is None else dict(wall_pilot=wall_pilot)),
                        **({} if motor_assist is None else dict(motor_assist=motor_assist)),
                        **({} if stale_evidence is None else dict(stale_evidence=stale_evidence)))
    info['gap_pilot'] = None if gap_pilot is None else str(gap_pilot)
    info['throttle_column'] = throttle_column
    info['near_on_path'] = bool(near_on_path)
    history = pilot.pose_history
    d = pd.read_csv(Path(runs)/f'{flight}.csv', low_memory=False)
    have_gap = 'gap_age' in d and getattr(pilot, 'gap_aim', None) is not None
    have_loom = 'looming_age' in d
    stream = None
    if looming_stream:
        z = np.load(looming_stream.replace('{flight}', flight))
        stream = dict(t=z['t_wall'], ttc=z['ttc'], distance=z['distance'], below=z['below_fraction'],
                      lower=z['ttc_lower'])
        have_loom, pointer = False, 0
    has_vertical = hasattr(pilot, 'vertical_log')
    keys = ('t', 'now', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'cvx', 'cvy', 'cvz', 'yaw_cmd', 'state', 'log_cvx', 'log_cvy',
            'log_cvz', 'log_state', 'log_climb', 'cap', 'climb', 'descent_scale', 'support_since',
            'slope_support_since', 'new_sample', 'ttc', 'below', 'lower', 'vertical_pilot', 'vertical_target',
            'vertical_factor', 'vertical_arrest', 'vertical_stage', 'vertical_climb', 'turn_first', 'side_guard',
            'gap_target', 'gap_applied', 'gap_offset', 'gap_mode', 'gap_commit', 'gap_conflict', 'gap_ring',
            'gap_sample_time', 'gap_shift', 'gap_kind', 'gap_near_on_path')
    if descent_view is not None:
        keys += ('view_sink_bound', 'view_withheld', 'view_boost')
    if contact_support is not None:
        keys += ('contact_unexplained', 'contact_gain', 'contact_fire')
    has_brake_log = hasattr(pilot, 'brake_log')
    if has_brake_log:
        keys += ('brake_added_sink', 'brake_sink_left', 'brake_sink_withheld', 'brake_reference', 'brake_ray_x',
                 'brake_ray_y', 'brake_ray_z', 'braking')
    if motor_assist is not None:
        keys += ('assist_pilot_vx', 'assist_pilot_vy', 'assist_pilot_vz', 'assist_horizontal', 'assist_vertical',
                 'assist_source', 'assist_plan', 'assist_wall_ahead')
    has_stale = info.get('stale_evidence', False)
    if has_stale:
        keys += ('cap_ray_deg', 'cap_reseat')
    rows = {k: [] for k in keys}
    drop = None if cue_drop is None else {round(float(c), 6) for c in cue_drop}
    last_ts = None
    nan = float('nan')
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
        if drop is not None and cue is not None and round(float(r.capture_time), 6) in drop:
            cue = None
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
                       lr=finite(r.gap_lr), near_on_path=_near_on_path(r.near_on_path) if near_on_path else None,
                       confirmed=False)
        omega = np.array([r.omega_x, r.omega_y, r.omega_z])
        extra = dict(clearance=clearance) if clearance is not None else {}
        if gap is not None:
            extra['gap'] = gap
        pilot.update(senses, omega, dict(race_cue=cue) if cue else None, float(r.capture_time), now, **extra)
        pilot.issued_throttle = float(getattr(r, throttle_column))
        gov = pilot.clearance
        values = dict(t=float(r.phase), now=now, x=r.x, y=r.y, z=r.z, vx=r.vx, vy=r.vy, vz=r.vz,
                      cvx=pilot.velocity_command[0], cvy=pilot.velocity_command[1], cvz=pilot.velocity_command[2],
                      yaw_cmd=pilot.pilot.sight_yaw, state=pilot.state, log_cvx=r.cmd_vx, log_cvy=r.cmd_vy,
                      log_cvz=r.cmd_vz, log_state=r.pilot_state if isinstance(r.pilot_state, str) else '',
                      log_climb=getattr(r, 'clearance_climb', nan),
                      cap=nan if gov is None or gov.cap is None else gov.cap,
                      climb=nan if gov is None else gov.climb, descent_scale=pilot.descent_scale,
                      support_since=nan if pilot.support_since is None else pilot.support_since,
                      slope_support_since=nan if pilot.slope_support_since is None else pilot.slope_support_since,
                      new_sample=clearance is not None,
                      ttc=nan if clearance is None or clearance['ttc'] is None else clearance['ttc'],
                      below=nan if clearance is None or clearance['below_fraction'] is None
                      else clearance['below_fraction'],
                      lower=nan if clearance is None or clearance['ttc_lower'] is None else clearance['ttc_lower'])
        if has_vertical:
            values.update(pilot.vertical_log())
        if getattr(pilot, 'turn_first', None) is not None:
            # 1 while a turn-first episode is active (also in shadow); the side guard (wall-pilot version 4)
            values.update(turn_first=float(pilot.turn_first_active),
                          side_guard=float(getattr(pilot, 'side_guard_active', False)))
        aim = getattr(pilot, 'gap_aim', None)
        if aim is not None:
            values.update(gap_target=aim.target, gap_applied=aim.applied, gap_offset=pilot.gap_offset_deg,
                          gap_mode=aim.mode or '', gap_commit=float(getattr(aim, 'commit_side', 0)),
                          gap_conflict=pilot.gap_conflict, gap_ring=pilot.gap_ring_deg,
                          gap_sample_time=nan if gap is None else gap['time'],
                          gap_shift=nan if gap is None else gap['shift'],
                          gap_kind='' if gap is None else gap['kind'],
                          gap_near_on_path=nan if gap is None or gap['near_on_path'] is None
                          else float(gap['near_on_path']))
        else:
            values.update(gap_mode='', gap_conflict='', gap_kind='')
        if descent_view is not None:
            values.update(pilot.descent_view_log())
        if contact_support is not None:
            values.update(pilot.contact_log())
        if has_brake_log:
            # brake_reference: the vertical request before the clearance brake (after the guard and the ceiling cap);
            # braking: 1 while the clearance cap bound the request
            values.update(pilot.brake_log(), braking=float(pilot.clearance_braking))
        if motor_assist is not None:
            if hasattr(pilot, 'motor_assist_log'):
                values.update(pilot.motor_assist_log())
            else:
                values['assist_source'] = ''
            if sources is not None:
                sources.append((list(getattr(pilot, 'assist_sources', [])), pilot.state,
                                bool(getattr(pilot, 'assist_wall_ahead', False))))
        if has_stale:
            values.update(pilot.stale_log())
        for k in keys:
            rows[k].append(values.get(k, nan))
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    info.update(flight=flight, ticks=len(arrays['t']), stop_reason=side.get('stop_reason'), impact=side.get('impact'),
                looming_stream=looming_stream, haltere=_haltere_file())
    return arrays, pilot, info


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
# Gates (frozen in configs/obstacles/vertical_guard_gates.json before scoring)
# ---------------------------------------------------------------------------------------------
def load_gates(path=GATES_PATH):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    from haltere.liftoff.gap_stack import config_sha256
    digest = config_sha256(obj)
    if obj.get('frozen') is not True or obj.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    return obj, digest


def durations(t):
    """Per-tick durations (to the next tick; the last tick gets 0)."""
    t = np.asarray(t, float)
    return np.diff(t, append=t[-1])


def onsets(mask):
    mask = np.asarray(mask, bool)
    return np.flatnonzero(mask & ~np.r_[False, mask[:-1]])


def support_contacts(flown):
    """(onset time, contact time) of every support-climb episode of the logged pilot: `flown` is the replay of the
    logged pilot without looming (it reproduces the logged states); contact is the start of the support timer
    (support_since or slope_support_since) that produced the onset."""
    out = []
    state = np.asarray(flown['state'])
    for i in onsets(state == 'support_climb'):
        # the timer is reset on the tick that starts the climb; the state shows it from the next tick
        start = []
        for j in range(i-1, max(i-6, -1), -1):
            start = [flown[k][j] for k in ('support_since', 'slope_support_since') if np.isfinite(flown[k][j])]
            if start:
                break
        out.append((float(flown['t'][i]), float(flown['now'][i]), float(min(start)) if start else float('nan')))
    return out


def score_straw(guard, control, flown, gate):
    """V-Straw on one lap. guard: stack on with the vertical guard; control: the same stack without it (both with the
    offline looming stream); flown: the logged pilot replayed without looming (support-climb episodes, contacts)."""
    t, now = guard['t'], guard['now']
    level = gate['level_band']
    episodes = []
    windows = np.zeros(len(t), bool)
    for onset_t, onset_now, contact in support_contacts(flown):
        before = (now >= contact-gate['lookback_s']) & (now < contact)
        limited = before & (guard['vertical_pilot'] < 0) & (guard['vertical_target'] > guard['vertical_pilot']+EPS)
        spot = (now >= contact-gate['lookback_s']) & (now <= onset_now+gate['after_onset_s'])
        windows |= spot
        first = float(t[np.flatnonzero(limited)[0]]) if limited.any() else None
        ds_g, ds_c = guard['descent_scale'][spot], control['descent_scale'][spot]
        episodes.append(dict(onset_t=round(onset_t, 2), contact_t=round(float(np.interp(contact, now, t)), 2),
                             limited_before_contact=bool(limited.any()),
                             first_limit_t=None if first is None else round(first, 2),
                             samples_with_ttc_lower_before=int((before & np.isfinite(guard['lower'])).sum()),
                             samples_before=int((before & guard['new_sample'].astype(bool)).sum()),
                             descent_scale_guard_min=round(float(np.nanmin(ds_g)), 3),
                             descent_scale_control_min=round(float(np.nanmin(ds_c)), 3),
                             descent_scale_guard_mean=round(float(np.nanmean(ds_g)), 3),
                             descent_scale_control_mean=round(float(np.nanmean(ds_c)), 3)))
    descending = windows & (guard['vz'] < -level)
    ceiling = np.maximum(guard['vertical_pilot'], 0.)
    raised = descending & (guard['vertical_target'] > ceiling+EPS)
    h_guard = np.hypot(guard['cvx'], guard['cvy'])
    h_control = np.hypot(control['cvx'], control['cvy'])
    reduced = windows & (h_guard < h_control-EPS)
    minutes = float(t[-1]-t[0])/60.
    dt = durations(t)
    limiting = (guard['vertical_pilot'] < 0) & (guard['vertical_target'] > guard['vertical_pilot']+EPS)
    arrest = guard['vertical_arrest'] > 0
    climb = np.nan_to_num(guard['vertical_climb']) > 0
    control_climb = np.nan_to_num(control['climb'])
    result = dict(
        episodes=episodes, n_episodes=len(episodes),
        limited_fraction=round(sum(e['limited_before_contact'] for e in episodes)/len(episodes), 3)
        if episodes else None,
        ticks_descending_in_windows=int(descending.sum()), ticks_raised_above_pilot=int(raised.sum()),
        ticks_horizontal_reduced=int(reduced.sum()),
        max_horizontal_reduction=round(float(np.max(h_control[windows]-h_guard[windows])), 4) if windows.any() else None,
        whole_lap=dict(minutes=round(minutes, 2),
                       max_guard_climb=round(float(np.nanmax(np.nan_to_num(guard['vertical_climb']))), 3),
                       ticks_guard_climb_above_limit=int((np.nan_to_num(guard['vertical_climb'])
                                                          > gate['max_new_climb']+EPS).sum()),
                       control_max_climb=round(float(control_climb.max()), 3),
                       control_climb_s=round(float(dt[control_climb > 0].sum()), 2),
                       limiting_s=round(float(dt[limiting].sum()), 2),
                       limiting_per_min=round(len(onsets(limiting))/minutes, 2),
                       arrest_s=round(float(dt[arrest].sum()), 2), arrest_per_min=round(len(onsets(arrest))/minutes, 2),
                       climb_s=round(float(dt[climb].sum()), 2), climb_per_min=round(len(onsets(climb))/minutes, 2),
                       escalated_s=round(float(dt[guard['vertical_stage'] == 2].sum()), 2),
                       descent_scale_guard_mean=round(float(np.nanmean(guard['descent_scale'])), 4),
                       descent_scale_control_mean=round(float(np.nanmean(control['descent_scale'])), 4)))
    result['passed'] = bool(episodes and result['limited_fraction'] >= gate['limited_fraction']
                            and result['ticks_raised_above_pilot'] == 0 and result['ticks_horizontal_reduced'] == 0
                            and result['whole_lap']['ticks_guard_climb_above_limit'] == 0)
    return result


def _impact_end(arrays, info):
    reason = str(info.get('stop_reason') or '')
    return float(arrays['t'][-1]) if 'Impact' in reason else None


def score_minus(guard, info, gate, floor_sink=False):
    """V-Minus on one flight: no requested climb above max_climb from before_onset_s before the last logged
    governor-climb onset to the ceiling impact; with floor_sink, the sink before that onset is met with a request of
    at least level before max_height_loss_m of height is lost."""
    t = guard['t']
    end = _impact_end(guard, info)
    logged = np.nan_to_num(np.asarray(guard['log_climb'], float)) > 0
    starts = [i for i in onsets(logged) if end is None or t[i] <= end]
    if end is None or not starts:
        return dict(scored=False, reason='no logged impact or governor climb onset')
    onset = float(t[starts[-1]])
    window = (t >= onset-gate['before_onset_s']) & (t <= end)
    cvz = guard['cvz'][window]
    result = dict(scored=True, window=[round(onset-gate['before_onset_s'], 2), round(end, 2)],
                  logged_climb_onset=round(onset, 2), ticks=int(window.sum()),
                  max_requested_vz=round(float(np.max(cvz)), 3),
                  logged_max_requested_vz=round(float(np.nanmax(guard['log_cvz'][window])), 3))
    passed = result['max_requested_vz'] <= gate['max_climb']+EPS
    if floor_sink:
        idx = np.flatnonzero(window)
        vz, z = guard['vz'][idx], guard['z'][idx]
        descending = vz < -gate['level_band']
        before = t[idx] <= onset
        first = np.flatnonzero(descending & before)
        if not len(first):
            result['floor_sink'] = dict(scored=False, reason='no descent before the logged climb onset')
            passed = False
        else:
            # the sink that was under way at the logged climb onset
            j = first[-1]
            while j > 0 and descending[j-1]:
                j -= 1
            k = j
            while k > 0 and vz[k-1] < 0:
                k -= 1
            z_ref = float(z[k])
            e = j
            while e+1 < len(idx) and descending[e+1]:
                e += 1
            sink = np.arange(j, e+1)
            ok = guard['cvz'][idx[sink]] >= -EPS
            # first tick from which the request stays at least level for the rest of the sink
            tail = np.flatnonzero(~ok)
            start = sink[0] if not len(tail) else (tail[-1]+1+sink[0] if tail[-1]+1 < len(sink) else None)
            loss = None if start is None else round(z_ref-float(z[start]), 3)
            result['floor_sink'] = dict(sink_start_t=round(float(t[idx[k]]), 2), z_ref=round(z_ref, 3),
                                        descent_t=[round(float(t[idx[j]]), 2), round(float(t[idx[e]]), 2)],
                                        level_request_from_t=None if start is None else round(float(t[idx[start]]), 2),
                                        height_loss_at_level_request=loss,
                                        max_height_loss_in_sink=round(z_ref-float(z[sink].min()), 3),
                                        min_requested_vz_in_sink=round(float(guard['cvz'][idx[sink]].min()), 3))
            passed = passed and loss is not None and loss < gate['max_height_loss_m']
    result['passed'] = bool(passed)
    return result


def score_pine(guard, info, gate):
    """V-Pine: requested vz >= min_vz for >= climb_fraction of the seconds the logged governor climbed; no requested
    descent in the last last_s before the contact (the logged impact)."""
    t = guard['t']
    dt = durations(t)
    logged = np.nan_to_num(np.asarray(guard['log_climb'], float)) > 0
    answered = logged & (guard['cvz'] >= gate['min_vz']-EPS)
    total, covered = float(dt[logged].sum()), float(dt[answered].sum())
    end = _impact_end(guard, info)
    end = float(t[-1]) if end is None else end
    late = t >= end-gate['last_s']
    per_episode = []
    for i in onsets(logged):
        j = i
        while j+1 < len(t) and logged[j+1]:
            j += 1
        span = slice(i, j+1)
        per_episode.append(dict(t=[round(float(t[i]), 2), round(float(t[j]), 2)],
                                logged_s=round(float(dt[span][logged[span]].sum()), 2),
                                answered_s=round(float(dt[span][answered[span]].sum()), 2),
                                max_requested_vz=round(float(guard['cvz'][span].max()), 3)))
    result = dict(logged_climb_s=round(total, 3), answered_s=round(covered, 3),
                  answered_fraction=round(covered/total, 4) if total else None, episodes=per_episode,
                  late_window=[round(end-gate['last_s'], 2), round(end, 2)],
                  late_min_requested_vz=round(float(np.min(guard['cvz'][late])), 4),
                  logged_late_min_requested_vz=round(float(np.nanmin(guard['log_cvz'][late])), 4))
    result['passed'] = bool(result['answered_fraction'] is not None
                            and result['answered_fraction'] >= gate['climb_fraction']
                            and result['late_min_requested_vz'] >= -EPS)
    return result


def score_straw_uphill(laps, gate):
    """V-Straw uphill (gates v3): the guard's escalated climb (vertical_stage 2: beyond the gentle bound) pooled over
    the Straw Bale laps replayed with the offline looming stream, in seconds per minute of flight; `laps` maps the flight
    to its guard replay. Also reported per lap: escalated and climbing seconds, escalations, and the ticks whose guard
    climb request exceeds the gentle bound (max_new_climb)."""
    rows, total_s, total_min = {}, 0., 0.
    for flight, guard in laps.items():
        t = np.asarray(guard['t'], float)
        dt = durations(t)
        stage = np.nan_to_num(np.asarray(guard['vertical_stage'], float))
        climb = np.nan_to_num(np.asarray(guard['vertical_climb'], float))
        minutes = float(t[-1]-t[0])/60.
        escalated = float(dt[stage == 2].sum())
        rows[flight] = dict(minutes=round(minutes, 2), escalated_s=round(escalated, 2),
                            escalated_per_min=round(escalated/minutes, 3) if minutes else None,
                            escalations=int(len(onsets(stage == 2))), climb_s=round(float(dt[climb > 0].sum()), 2),
                            climb_onsets=int(len(onsets(climb > 0))), max_guard_climb=round(float(climb.max()), 3),
                            ticks_guard_climb_above_limit=int((climb > gate['max_new_climb']+EPS).sum()))
        total_s += escalated
        total_min += minutes
    pooled = total_s/total_min if total_min else None
    return dict(laps=rows, escalated_s=round(total_s, 2), minutes=round(total_min, 2),
                escalated_per_min=None if pooled is None else round(pooled, 3),
                passed=bool(pooled is not None and pooled <= gate['max_escalated_per_min']+EPS))


def score_minus_escalation(guard, gate):
    """V-Minus no escalation (gates v3): the guard's terrain climb request never exceeds max_climb on a Minus Two
    flight (a flat garage floor under a ~2.2 m ceiling: rising ground is never there)."""
    climb = np.nan_to_num(np.asarray(guard['vertical_climb'], float))
    stage = np.nan_to_num(np.asarray(guard['vertical_stage'], float))
    t = np.asarray(guard['t'], float)
    return dict(max_guard_climb=round(float(climb.max()), 3), escalated_s=round(float(durations(t)[stage == 2].sum()), 3),
                max_requested_vz=round(float(np.max(guard['cvz'])), 3),
                passed=bool(climb.max() <= gate['max_climb']+EPS))


def score_pine_v3(guard, info, gate):
    """V-Pine (gates v3): the issued vertical request is at least min_vz for at least climb_fraction of the seconds in
    which the logged governor itself requested at least min_vz (its release tail and weaker climbs excluded); no
    requested descent in the last last_s before the contact (v2's criterion); on the mound (the first logged climb
    episode) the issued height request (the integral of the issued vertical request over the episode) is at least
    mound_fraction of the flown one (the logged issued request)."""
    t = np.asarray(guard['t'], float)
    dt = durations(t)
    logged = np.nan_to_num(np.asarray(guard['log_climb'], float))
    strong = logged >= gate['min_vz']-EPS
    answered = strong & (guard['cvz'] >= gate['min_vz']-EPS)
    total, covered = float(dt[strong].sum()), float(dt[answered].sum())
    end = _impact_end(guard, info)
    end = float(t[-1]) if end is None else end
    late = t >= end-gate['last_s']
    episodes = []
    climbing = logged > 0
    for i in onsets(climbing):
        j = i
        while j+1 < len(t) and climbing[j+1]:
            j += 1
        span = slice(i, j+1)
        episodes.append(dict(t=[round(float(t[i]), 2), round(float(t[j]), 2)],
                             strong_s=round(float(dt[span][strong[span]].sum()), 2),
                             answered_s=round(float(dt[span][answered[span]].sum()), 2),
                             height_request_m=round(float((guard['cvz'][span]*dt[span]).sum()), 3),
                             flown_height_request_m=round(float((guard['log_cvz'][span]*dt[span]).sum()), 3),
                             max_requested_vz=round(float(guard['cvz'][span].max()), 3)))
    mound = episodes[0] if episodes else None
    result = dict(strong_logged_s=round(total, 3), answered_s=round(covered, 3),
                  answered_fraction=round(covered/total, 4) if total else None, episodes=episodes,
                  late_window=[round(end-gate['last_s'], 2), round(end, 2)],
                  late_min_requested_vz=round(float(np.min(guard['cvz'][late])), 4),
                  logged_late_min_requested_vz=round(float(np.nanmin(guard['log_cvz'][late])), 4),
                  mound_height_fraction=None if mound is None or mound['flown_height_request_m'] <= 0 else
                  round(mound['height_request_m']/mound['flown_height_request_m'], 4))
    result['climb_passed'] = bool(result['answered_fraction'] is not None
                                  and result['answered_fraction'] >= gate['climb_fraction'])
    result['no_descent_passed'] = bool(result['late_min_requested_vz'] >= -EPS)
    result['mound_passed'] = bool(result['mound_height_fraction'] is not None
                                  and result['mound_height_fraction'] >= gate['mound_fraction'])
    result['passed'] = result['climb_passed'] and result['no_descent_passed'] and result['mound_passed']
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('flights', nargs='+')
    parser.add_argument('--out', required=True, help='output prefix (npz per flight and a summary json)')
    parser.add_argument('--tree', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--runs', default=str(RUNS))
    parser.add_argument('--stack', choices=['none', 'flown', 'on', 'shadow'], default='flown')
    parser.add_argument('--wall', choices=['off', 'on', 'shadow'], default='off')
    parser.add_argument('--vertical', choices=['off', 'on', 'shadow'], default=None,
                        help='with --stack flown: off (default), on or shadow; with --stack on/shadow: off removes it')
    parser.add_argument('--looming-stream', default=None)
    parser.add_argument('--gap-pilot', default=None,
                        help='gap pilot declaration for the gap aim (default: the tree\'s configs/obstacles/gap_pilot.json);'
                             ' the file tag gains -gp<stem>')
    parser.add_argument('--near-on-path', action='store_true',
                        help='the gap samples carry the logged near_on_path (tag suffix -nop)')
    parser.add_argument('--descent-view', default=None, metavar='DECLARATION',
                        help='add the view-keeping descent of a frozen descent-view declaration (e.g. '
                             'configs/pilot/descent_view.json) to the variant (version 2 and later with its contact '
                             'support); the file tag gains -dv<version>')
    parser.add_argument('--throttle-column', default='thr', choices=['thr', 'command_thr'],
                        help='the logged column fed to the pilot as its issued throttle (default thr: the earlier '
                             'replays; command_thr is what the pilot saw in the runner, and differs on fast-PD '
                             'flights); '
                             'command_thr adds -cthr to the file tag')
    parser.add_argument('--wall-pilot', default=None, metavar='DECLARATION',
                        help='wall-pilot declaration of the wall rules (default: the tree\'s '
                             'configs/obstacles/wall_pilot.json); the file tag gains -wp<version>')
    parser.add_argument('--cue-drop', default=None, metavar='JSON',
                        help='a JSON file per flight ({flight} in the path) whose rejected_capture_times list the logged '
                             'captures whose ring cue is replaced by no detection (replay(cue_drop=...)); the file tag '
                             'gains -cd')
    parser.add_argument('--stale-evidence', default=None, metavar='DECLARATION',
                        help='add the stale-evidence rules of a frozen declaration (configs/obstacles/stale_evidence.json) '
                             'to a stack variant; the file tag gains -se<version>')
    parser.add_argument('--motor-assist', default=None, metavar='DECLARATION',
                        help='add the motor assist of a frozen motor-assist declaration (configs/pilot/motor_assist.json) '
                             'for the motor contract of each flight; the file tag gains -ma<version>')
    args = parser.parse_args(argv)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    here = str(Path(__file__).resolve().parent)
    if 'haltere' in sys.modules and __name__ == '__main__':
        raise SystemExit('run this file as a script (python path/to/vertical_replay.py) so --tree selects the code')
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != Path(here)]   # run as a script: no shadowing
    sys.path.insert(0, str(Path(args.tree).resolve()))
    from haltere.liftoff import fast_race_cue as frc
    vertical = resolve_vertical(args.stack, args.vertical, hasattr(frc, 'VerticalGuardConfig'))
    tag = variant_tag(args.stack, args.wall, vertical, bool(args.looming_stream))
    if args.gap_pilot:
        tag += f'-gp{Path(args.gap_pilot).stem}'
    if args.near_on_path:
        tag += '-nop'
    descent_view = contact_support = None
    if args.throttle_column != 'thr':
        tag += '-cthr'
    if args.descent_view:
        # the same frozen-declaration checks as the runner (canonical content hash, the tree's rule version)
        from haltere.liftoff.gap_stack import config_sha256
        declaration = json.loads(Path(args.descent_view).read_text(encoding='utf-8'))
        if declaration.get('frozen') is not True or declaration.get('sha256') != config_sha256(declaration):
            raise SystemExit(f'{args.descent_view} is not a frozen descent-view declaration, or it changed after the '
                             'freeze')
        descent_view = frc.descent_view_config(declaration)
        if hasattr(frc, 'contact_support_config'):
            contact_support = frc.contact_support_config(declaration)
        tag += f'-dv{declaration["version"]}'
    extra = {}
    if contact_support is not None or args.throttle_column != 'thr':
        extra.update(contact_support=contact_support, throttle_column=args.throttle_column)
    if args.wall_pilot:
        from haltere.liftoff.gap_stack import config_sha256
        declaration = json.loads(Path(args.wall_pilot).read_text(encoding='utf-8'))
        if declaration.get('frozen') is not True or declaration.get('sha256') != config_sha256(declaration):
            raise SystemExit(f'{args.wall_pilot} is not a frozen wall-pilot declaration, or it changed after the freeze')
        tag += f'-wp{declaration["version"]}'
        extra['wall_pilot'] = args.wall_pilot
    motor_assist = None
    if args.motor_assist:
        from haltere.liftoff.gap_stack import config_sha256
        motor_assist = json.loads(Path(args.motor_assist).read_text(encoding='utf-8'))
        if motor_assist.get('frozen') is not True or motor_assist.get('sha256') != config_sha256(motor_assist):
            raise SystemExit(f'{args.motor_assist} is not a frozen motor-assist declaration, or it changed after the '
                             'freeze')
        tag += f'-ma{motor_assist["version"]}'
    if args.stale_evidence:
        from haltere.liftoff.gap_stack import config_sha256
        declaration = json.loads(Path(args.stale_evidence).read_text(encoding='utf-8'))
        if declaration.get('frozen') is not True or declaration.get('sha256') != config_sha256(declaration):
            raise SystemExit(f'{args.stale_evidence} is not a frozen stale-evidence declaration, or it changed after the '
                             'freeze')
        tag += f'-se{declaration["version"]}'
        extra['stale_evidence'] = declaration
    if args.cue_drop:
        tag += '-cd'
    results = {}
    for flight in args.flights:
        if args.cue_drop:
            extra['cue_drop'] = json.loads(Path(args.cue_drop.replace('{flight}', flight)).read_text(
                encoding='utf-8'))['rejected_capture_times']
        arrays, pilot, info = replay(flight, args.tree, args.runs, stack=args.stack, wall=args.wall,
                                     vertical=vertical, looming_stream=args.looming_stream,
                                     gap_pilot=args.gap_pilot, near_on_path=args.near_on_path,
                                     descent_view=descent_view, motor_assist=motor_assist, **extra)
        np.savez_compressed(f'{args.out}_{tag}_{flight}.npz', **arrays)
        results[flight] = summary(arrays, info)
        meta = pilot.metadata()
        results[flight]['vertical_guard_metadata'] = meta.get('vertical_guard')
        results[flight]['wall_pilot_metadata'] = meta.get('wall_pilot')
        results[flight]['gap_aim_metadata'] = meta.get('gap_aim')
        results[flight]['descent_view_metadata'] = meta.get('descent_view')
        if contact_support is not None:
            results[flight]['contact_support_metadata'] = meta.get('contact_support')
        results[flight]['motor_assist_metadata'] = meta.get('motor_assist')
        if args.stale_evidence:
            results[flight]['stale_evidence_metadata'] = meta.get('stale_evidence')
        print(flight, json.dumps({k: v for k, v in results[flight].items() if not k.endswith('_metadata')},
                                 default=str), flush=True)
    Path(f'{args.out}_{tag}_summary.json').write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')


def _load(prefix, tag, flight):
    return dict(np.load(f'{prefix}_{tag}_{flight}.npz', allow_pickle=False))


def score_all(out, baseline, gates_path=GATES_PATH, runs=RUNS):
    """Score the frozen gates from the replay files: `out` is this tree's prefix, `baseline` the baseline tree's.
    Expected files: {prefix}_{variant_tag}_{flight}.npz as written by main()."""
    gates, digest = load_gates(gates_path)
    if gates['version'] >= 4:
        return score_all_v4(out, baseline, gates, digest, runs)
    if gates['version'] >= 3:
        return score_all_v3(out, baseline, gates, digest, runs)
    g = gates['gates']
    info = {f: json.loads((Path(runs)/f'{f}.json').read_text(encoding='utf-8'))
            for f in g['Identity']['flights']}
    result = dict(gates_sha256=digest, vertical_guard=gates['vertical_guard'], baseline_tree=gates['baseline_tree'])
    # Identity
    straw = set(g['V_Straw']['flights'])
    pairs = []
    for flight in g['Identity']['flights']:
        suffixes = [True, False] if flight in straw else [False]
        for stream in suffixes:
            pairs.append((flight, variant_tag('none', 'off', 'off', stream), variant_tag('none', 'off', 'off', stream),
                          'gate'))
            if stream or flight not in straw:
                pairs.append((flight, variant_tag('shadow', 'off', 'shadow', stream),
                              variant_tag('shadow', 'off', 'off', stream), 'gate'))
            if flight not in straw:
                for mine, base in ((variant_tag('flown', 'on', 'off', stream),)*2,
                                   (variant_tag('flown', 'shadow', 'off', stream),)*2,
                                   (variant_tag('on', 'off', 'off', stream),)*2):
                    pairs.append((flight, mine, base, 'report'))
    identity = []
    for flight, mine, base, kind in pairs:
        try:
            same, keys = identical(_load(out, mine, flight), _load(baseline, base, flight))
        except FileNotFoundError as exc:
            same, keys = None, str(exc)
        identity.append(dict(flight=flight, variant=mine, baseline_variant=base, kind=kind, identical=same,
                             keys=keys))
    gate_rows = [r for r in identity if r['kind'] == 'gate']
    result['Identity'] = dict(pairs=identity, passed=bool(gate_rows and all(r['identical'] for r in gate_rows)))
    # V-Straw (pooled over the laps)
    vs = g['V_Straw']
    laps = {}
    for flight in vs['flights']:
        guard = _load(out, variant_tag('on', 'off', 'on', True), flight)
        control = _load(out, variant_tag('on', 'off', 'off', True), flight)
        flown = _load(out, variant_tag('none', 'off', 'off', False), flight)
        laps[flight] = score_straw(guard, control, flown, vs)
    episodes = [e for lap in laps.values() for e in lap['episodes']]
    fraction = sum(e['limited_before_contact'] for e in episodes)/len(episodes) if episodes else None
    result['V_Straw'] = dict(
        laps=laps, n_episodes=len(episodes), limited_fraction=None if fraction is None else round(fraction, 3),
        passed=bool(episodes and fraction >= vs['limited_fraction']
                    and all(lap['ticks_raised_above_pilot'] == 0 and lap['ticks_horizontal_reduced'] == 0
                            and lap['whole_lap']['ticks_guard_climb_above_limit'] == 0 for lap in laps.values())))
    # V-Minus (and the matched control, report only)
    vm = g['V_Minus']
    minus, control = {}, {}
    for flight, spec in vm['flights'].items():
        side = json.loads((Path(runs)/f'{flight}.json').read_text(encoding='utf-8'))
        minus[flight] = score_minus(_load(out, variant_tag('on', 'off', 'on', False), flight), side, vm,
                                    spec['floor_sink'])
        control[flight] = score_minus(_load(out, variant_tag('on', 'off', 'off', False), flight), side, vm,
                                      spec['floor_sink'])
    result['V_Minus'] = dict(flights=minus, control_report_only=control,
                             passed=all(r.get('passed') for r in minus.values()))
    # V-Pine
    vp = g['V_Pine']
    side = json.loads((Path(runs)/f"{vp['flight']}.json").read_text(encoding='utf-8'))
    pine = score_pine(_load(out, variant_tag('on', 'off', 'on', False), vp['flight']), side, vp)
    pine_control = score_pine(_load(out, variant_tag('on', 'off', 'off', False), vp['flight']), side, vp)
    result['V_Pine'] = dict(pine, control_report_only=pine_control)
    result['passed'] = {k: result[k]['passed'] for k in ('Identity', 'V_Straw', 'V_Minus', 'V_Pine')}
    del info
    return result


def score_all_v3(out, baseline, gates, digest, runs=RUNS):
    """Score gates version 3 (see configs/obstacles/vertical_guard_gates.json): the baseline is the previous guard's
    tree (m2-vertical); stream flights are replayed with the offline looming stream. The v2 gate definitions are also
    scored on the same replays (report: old_definitions)."""
    g = gates['gates']
    stream_flights = set(gates['inputs']['stream_flights'])
    result = dict(gates_sha256=digest, gates_version=gates['version'], vertical_guard=gates['vertical_guard'],
                  baseline_tree=gates['baseline_tree'])

    def load(prefix, stack, wall, vertical, flight, stream=None):
        stream = flight in stream_flights if stream is None else stream
        return _load(prefix, variant_tag(stack, wall, vertical, stream), flight)

    def side(flight):
        return json.loads((Path(runs)/f'{flight}.json').read_text(encoding='utf-8'))
    # Identity: stack off and shadow bit-identical to the baseline tree
    identity = []
    for flight in g['Identity']['flights']:
        streamed = flight in stream_flights
        pairs = [(('none', 'off', 'off', False), 'gate'), (('shadow', 'off', 'shadow', streamed), 'gate'),
                 (('on', 'off', 'off', streamed), 'report')]
        if streamed:
            pairs.append((('none', 'off', 'off', True), 'gate'))
        else:
            pairs += [(('flown', 'on', 'off', False), 'report'), (('flown', 'shadow', 'off', False), 'report')]
        for (stack, wall, vertical, stream), kind in pairs:
            tag = variant_tag(stack, wall, vertical, stream)
            try:
                same, keys = identical(_load(out, tag, flight), _load(baseline, tag, flight))
            except FileNotFoundError as exc:
                same, keys = None, str(exc)
            identity.append(dict(flight=flight, variant=tag, kind=kind, identical=same, keys=keys))
    gate_rows = [r for r in identity if r['kind'] == 'gate']
    result['Identity'] = dict(pairs=identity, gate_pairs=len(gate_rows),
                              identical_gate_pairs=sum(bool(r['identical']) for r in gate_rows),
                              report_pairs=len(identity)-len(gate_rows),
                              identical_report_pairs=sum(bool(r['identical']) for r in identity if r['kind'] != 'gate'),
                              passed=bool(gate_rows and all(r['identical'] for r in gate_rows)))
    # V-Straw downhill: v2's V-Straw definitions without its whole-lap climb criterion
    vd = g['V_Straw_downhill']
    laps = {}
    for flight in vd['flights']:
        laps[flight] = score_straw(load(out, 'on', 'off', 'on', flight, True), load(out, 'on', 'off', 'off', flight, True),
                                   load(out, 'none', 'off', 'off', flight, False), vd)
    episodes = [e for lap in laps.values() for e in lap['episodes']]
    fraction = sum(e['limited_before_contact'] for e in episodes)/len(episodes) if episodes else None
    result['V_Straw_downhill'] = dict(
        laps=laps, n_episodes=len(episodes), limited_fraction=None if fraction is None else round(fraction, 3),
        limited_passed=bool(episodes and fraction >= vd['limited_fraction']),
        not_raised_passed=all(lap['ticks_raised_above_pilot'] == 0 for lap in laps.values()),
        horizontal_passed=all(lap['ticks_horizontal_reduced'] == 0 for lap in laps.values()))
    result['V_Straw_downhill']['passed'] = all(result['V_Straw_downhill'][k] for k in
                                               ('limited_passed', 'not_raised_passed', 'horizontal_passed'))
    # V-Straw uphill: escalated climbs pooled over every Straw Bale lap
    vu = g['V_Straw_uphill']
    result['V_Straw_uphill'] = score_straw_uphill({f: load(out, 'on', 'off', 'on', f, True) for f in vu['flights']}, vu)
    result['V_Straw_uphill']['control_report_only'] = {
        f: dict(climb_s=round(float(durations(c['t'])[np.nan_to_num(c['climb']) > 0].sum()), 2),
                max_climb=round(float(np.nan_to_num(c['climb']).max()), 3))
        for f, c in ((f, load(out, 'on', 'off', 'off', f, True)) for f in vu['flights'])}
    # V-Minus: v2's windows and floor sink, and no escalation on any Minus Two flight
    vm = g['V_Minus']
    minus, control = {}, {}
    for flight, spec in vm['flights'].items():
        minus[flight] = score_minus(load(out, 'on', 'off', 'on', flight), side(flight), vm, spec['floor_sink'])
        control[flight] = score_minus(load(out, 'on', 'off', 'off', flight), side(flight), vm, spec['floor_sink'])
    escalation = {f: score_minus_escalation(load(out, 'on', 'off', 'on', f), vm['no_escalation'])
                  for f in vm['no_escalation']['flights']}
    result['V_Minus'] = dict(flights=minus, control_report_only=control, no_escalation=escalation,
                             windows_passed=all(r.get('passed') for r in minus.values()),
                             no_escalation_passed=all(r['passed'] for r in escalation.values()))
    result['V_Minus']['passed'] = result['V_Minus']['windows_passed'] and result['V_Minus']['no_escalation_passed']
    # V-Pine
    vp = g['V_Pine']
    pine = score_pine_v3(load(out, 'on', 'off', 'on', vp['flight']), side(vp['flight']), vp)
    pine['control_report_only'] = score_pine_v3(load(out, 'on', 'off', 'off', vp['flight']), side(vp['flight']), vp)
    pine['other_flights_report_only'] = {}
    for flight in vp.get('report_flights', []):
        guard = load(out, 'on', 'off', 'on', flight)
        t = np.asarray(guard['t'], float)
        pine['other_flights_report_only'][flight] = dict(
            max_guard_climb=round(float(np.nan_to_num(guard['vertical_climb']).max()), 3),
            escalated_s=round(float(durations(t)[np.nan_to_num(guard['vertical_stage']) == 2].sum()), 2),
            guard_climb_s=round(float(durations(t)[np.nan_to_num(guard['vertical_climb']) > 0].sum()), 2))
    result['V_Pine'] = pine
    # the version-2 gate definitions on the same replays (report)
    v2 = gates['old_definitions']
    straw_v2 = {f: score_straw(load(out, 'on', 'off', 'on', f, True), load(out, 'on', 'off', 'off', f, True),
                               load(out, 'none', 'off', 'off', f, False), v2['V_Straw']) for f in v2['V_Straw']['flights']}
    episodes = [e for lap in straw_v2.values() for e in lap['episodes']]
    fraction = sum(e['limited_before_contact'] for e in episodes)/len(episodes) if episodes else None
    pine_v2 = score_pine(load(out, 'on', 'off', 'on', v2['V_Pine']['flight']), side(v2['V_Pine']['flight']),
                         v2['V_Pine'])
    result['old_definitions'] = dict(
        V_Straw=dict(limited_fraction=None if fraction is None else round(fraction, 3),
                     whole_lap_ticks_above=sum(lap['whole_lap']['ticks_guard_climb_above_limit']
                                               for lap in straw_v2.values()),
                     escalated_s={f: lap['whole_lap']['escalated_s'] for f, lap in straw_v2.items()},
                     passed=bool(episodes and fraction >= v2['V_Straw']['limited_fraction']
                                 and all(lap['ticks_raised_above_pilot'] == 0 and lap['ticks_horizontal_reduced'] == 0
                                         and lap['whole_lap']['ticks_guard_climb_above_limit'] == 0
                                         for lap in straw_v2.values()))),
        V_Pine=dict(answered_fraction=pine_v2['answered_fraction'],
                    late_min_requested_vz=pine_v2['late_min_requested_vz'], passed=pine_v2['passed']))
    result['passed'] = {k: result[k]['passed'] for k in ('Identity', 'V_Straw_downhill', 'V_Straw_uphill', 'V_Minus',
                                                         'V_Pine')}
    return result


def score_r4(guard, gate):
    """V_R4 (gates v4): the guard's terrain climb request never exceeds max_climb and never escalates (vertical_stage 2)
    over the flight, and the issued vertical request stays at most max_climb in the window of the live false
    escalation."""
    t = np.asarray(guard['t'], float)
    climb = np.nan_to_num(np.asarray(guard['vertical_climb'], float))
    stage = np.nan_to_num(np.asarray(guard['vertical_stage'], float))
    window = (t >= gate['window'][0]) & (t <= gate['window'][1])
    escalated = stage == 2
    result = dict(max_guard_climb=round(float(climb.max()), 3), escalated_ticks=int(escalated.sum()),
                  escalated_s=round(float(durations(t)[escalated].sum()), 3),
                  first_escalated_t=round(float(t[escalated][0]), 3) if escalated.any() else None,
                  window_max_requested_vz=round(float(np.max(guard['cvz'][window])), 3),
                  window_logged_max_requested_vz=round(float(np.nanmax(guard['log_cvz'][window])), 3))
    result['passed'] = bool(result['max_guard_climb'] <= gate['max_climb']+EPS and not escalated.any()
                            and result['window_max_requested_vz'] <= gate['max_climb']+EPS)
    return result


def mound_escalation(guard, gate):
    """V_Pine mound_escalation (gates v4): the first tick with vertical_stage 2 inside the first logged climb episode
    (clearance_climb > 0), which must come at the latest at mound_escalated_by_s."""
    t = np.asarray(guard['t'], float)
    climbing = np.nan_to_num(np.asarray(guard['log_climb'], float)) > 0
    starts = onsets(climbing)
    if not len(starts):
        return dict(first_escalated_t=None, passed=False)
    i = j = starts[0]
    while j+1 < len(t) and climbing[j+1]:
        j += 1
    stage = np.nan_to_num(np.asarray(guard['vertical_stage'], float))[i:j+1]
    first = float(t[i:j+1][stage == 2][0]) if (stage == 2).any() else None
    return dict(episode=[round(float(t[i]), 2), round(float(t[j]), 2)],
                first_escalated_t=None if first is None else round(first, 3),
                passed=bool(first is not None and first <= gate['mound_escalated_by_s']+EPS))


def guard_report(guard, baseline=None):
    """V_Keep (gates v4, report): guard climb and escalated seconds, escalations and the maximum guard climb of a
    replay, with the same numbers for the baseline tree's replay when given."""
    def numbers(a):
        t = np.asarray(a['t'], float)
        dt = durations(t)
        climb = np.nan_to_num(np.asarray(a['vertical_climb'], float))
        stage = np.nan_to_num(np.asarray(a['vertical_stage'], float))
        return dict(climb_s=round(float(dt[climb > 0].sum()), 2), escalated_s=round(float(dt[stage == 2].sum()), 2),
                    escalations=int(len(onsets(stage == 2))), max_guard_climb=round(float(climb.max()), 3))
    out = numbers(guard)
    if baseline is not None:
        out['baseline'] = numbers(baseline)
    return out


def score_all_v4(out, baseline, gates, digest, runs=RUNS):
    """Score gates version 4 (see configs/obstacles/vertical_guard_gates.json): the baseline is the previous guard's
    tree (m4, guard v3); stream flights are replayed with the offline looming stream and the round-4 live flights as
    flown (their stack variants carry the inputs.as_flown suffix). The v3 definitions changed in v4 are also scored on
    the same replays (old_definitions), and the baseline's own guard replays are reported beside v4's."""
    g = gates['gates']
    stream_flights = set(gates['inputs']['stream_flights'])
    as_flown = gates['inputs'].get('as_flown', {})
    result = dict(gates_sha256=digest, gates_version=gates['version'], vertical_guard=gates['vertical_guard'],
                  baseline_tree=gates['baseline_tree'])

    def tag(stack, wall, vertical, flight, stream=None, suffix=None):
        stream = flight in stream_flights if stream is None else stream
        base = variant_tag(stack, wall, vertical, stream)
        if suffix is None:
            suffix = as_flown.get(flight, '') if stack in ('on', 'shadow') else ''
        return base+suffix

    def load(prefix, stack, wall, vertical, flight, stream=None, suffix=None):
        return _load(prefix, tag(stack, wall, vertical, flight, stream, suffix), flight)

    def side(flight):
        return json.loads((Path(runs)/f'{flight}.json').read_text(encoding='utf-8'))
    # Identity: stack off and shadow bit-identical to the baseline tree
    identity = []
    for flight in g['Identity']['flights']:
        streamed = flight in stream_flights
        pairs = [(('none', 'off', 'off', False, ''), 'gate'), (('shadow', 'off', 'shadow', streamed, None), 'gate'),
                 (('on', 'off', 'off', streamed, None), 'report')]
        if streamed:
            pairs.append((('none', 'off', 'off', True, ''), 'gate'))
        elif flight in as_flown:
            pairs.append((('none', 'off', 'off', False, '-dv1'), 'gate'))
        else:
            pairs += [(('flown', 'on', 'off', False, ''), 'report'), (('flown', 'shadow', 'off', False, ''), 'report')]
        for (stack, wall, vertical, stream, suffix), kind in pairs:
            name = tag(stack, wall, vertical, flight, stream, suffix)
            try:
                same, keys = identical(_load(out, name, flight), _load(baseline, name, flight))
            except FileNotFoundError as exc:
                same, keys = None, str(exc)
            identity.append(dict(flight=flight, variant=name, kind=kind, identical=same, keys=keys))
    gate_rows = [r for r in identity if r['kind'] == 'gate']
    result['Identity'] = dict(pairs=identity, gate_pairs=len(gate_rows),
                              identical_gate_pairs=sum(bool(r['identical']) for r in gate_rows),
                              report_pairs=len(identity)-len(gate_rows),
                              identical_report_pairs=sum(bool(r['identical']) for r in identity if r['kind'] != 'gate'),
                              passed=bool(gate_rows and all(r['identical'] for r in gate_rows)))
    # V_R4: the live false escalation, as flown
    vr4 = g['V_R4']
    result['V_R4'] = dict(score_r4(load(out, 'on', 'off', 'on', vr4['flight']), vr4),
                          baseline_report_only=score_r4(load(baseline, 'on', 'off', 'on', vr4['flight']), vr4))
    # V_Straw downhill (v3's definitions, unchanged)
    vd = g['V_Straw_downhill']
    laps = {}
    for flight in vd['flights']:
        laps[flight] = score_straw(load(out, 'on', 'off', 'on', flight, True), load(out, 'on', 'off', 'off', flight, True),
                                   load(out, 'none', 'off', 'off', flight, False), vd)
    episodes = [e for lap in laps.values() for e in lap['episodes']]
    fraction = sum(e['limited_before_contact'] for e in episodes)/len(episodes) if episodes else None
    result['V_Straw_downhill'] = dict(
        laps=laps, n_episodes=len(episodes), limited_fraction=None if fraction is None else round(fraction, 3),
        limited_passed=bool(episodes and fraction >= vd['limited_fraction']),
        not_raised_passed=all(lap['ticks_raised_above_pilot'] == 0 for lap in laps.values()),
        horizontal_passed=all(lap['ticks_horizontal_reduced'] == 0 for lap in laps.values()))
    result['V_Straw_downhill']['passed'] = all(result['V_Straw_downhill'][k] for k in
                                               ('limited_passed', 'not_raised_passed', 'horizontal_passed'))
    # V_Straw uphill (tightened to no escalated tick)
    vu = g['V_Straw_uphill']
    result['V_Straw_uphill'] = score_straw_uphill({f: load(out, 'on', 'off', 'on', f, True) for f in vu['flights']}, vu)
    result['V_Straw_uphill']['baseline_report_only'] = score_straw_uphill(
        {f: load(baseline, 'on', 'off', 'on', f, True) for f in vu['flights']}, vu)
    # V_Minus: v3's windows and floor sink, no escalation on every Minus Two flight and on the held-out flights
    vm = g['V_Minus']
    minus, control = {}, {}
    for flight, spec in vm['flights'].items():
        minus[flight] = score_minus(load(out, 'on', 'off', 'on', flight), side(flight), vm, spec['floor_sink'])
        control[flight] = score_minus(load(out, 'on', 'off', 'off', flight), side(flight), vm, spec['floor_sink'])
    no_escalation = vm['no_escalation']
    escalation = {f: score_minus_escalation(load(out, 'on', 'off', 'on', f), no_escalation)
                  for f in no_escalation['flights']+no_escalation.get('held_out_flights', [])}
    result['V_Minus'] = dict(flights=minus, control_report_only=control, no_escalation=escalation,
                             windows_passed=all(r.get('passed') for r in minus.values()),
                             no_escalation_passed=all(r['passed'] for r in escalation.values()),
                             held_out_passed=all(escalation[f]['passed']
                                                 for f in no_escalation.get('held_out_flights', [])))
    result['V_Minus']['passed'] = result['V_Minus']['windows_passed'] and result['V_Minus']['no_escalation_passed']
    # V_Pine: v3's three criteria and the mound escalation
    vp = g['V_Pine']
    pine = score_pine_v3(load(out, 'on', 'off', 'on', vp['flight']), side(vp['flight']), vp)
    pine['mound_escalation'] = mound_escalation(load(out, 'on', 'off', 'on', vp['flight']), vp)
    pine['passed'] = pine['passed'] and pine['mound_escalation']['passed']
    pine['baseline_report_only'] = score_pine_v3(load(baseline, 'on', 'off', 'on', vp['flight']), side(vp['flight']),
                                                 vp)
    pine['baseline_report_only']['mound_escalation'] = mound_escalation(
        load(baseline, 'on', 'off', 'on', vp['flight']), vp)
    pine['control_report_only'] = score_pine_v3(load(out, 'on', 'off', 'off', vp['flight']), side(vp['flight']), vp)
    pine['other_flights_report_only'] = {}
    for flight in vp.get('report_flights', []):
        pine['other_flights_report_only'][flight] = guard_report(load(out, 'on', 'off', 'on', flight),
                                                                 load(baseline, 'on', 'off', 'on', flight))
    result['V_Pine'] = pine
    # V_Keep (report): every guard replay against the baseline's
    result['V_Keep'] = {f: guard_report(load(out, 'on', 'off', 'on', f), load(baseline, 'on', 'off', 'on', f))
                        for f in g['Identity']['flights']}
    # the version-3 definitions changed in v4 (report)
    old = gates['old_definitions']
    uphill = score_straw_uphill({f: load(out, 'on', 'off', 'on', f, True) for f in old['V_Straw_uphill']['flights']},
                                old['V_Straw_uphill'])
    pine_v3 = score_pine_v3(load(out, 'on', 'off', 'on', old['V_Pine']['flight']), side(old['V_Pine']['flight']),
                            old['V_Pine'])
    result['old_definitions'] = dict(
        V_Straw_uphill=dict(escalated_per_min=uphill['escalated_per_min'], passed=uphill['passed']),
        V_Pine=dict(answered_fraction=pine_v3['answered_fraction'], mound_height_fraction=pine_v3['mound_height_fraction'],
                    passed=pine_v3['passed']),
        V_Minus=dict(passed=result['V_Minus']['windows_passed'] and all(
            escalation[f]['passed'] for f in old['V_Minus']['no_escalation']['flights'])))
    result['passed'] = {k: result[k]['passed'] for k in ('Identity', 'V_R4', 'V_Straw_downhill', 'V_Straw_uphill',
                                                         'V_Minus', 'V_Pine')}
    return result


def score_main(argv=None):
    parser = argparse.ArgumentParser(description='Score the frozen vertical-guard gates from replay files')
    parser.add_argument('--out', required=True, help="this tree's replay prefix")
    parser.add_argument('--baseline', required=True, help="the baseline tree's replay prefix")
    parser.add_argument('--gates', default=str(GATES_PATH))
    parser.add_argument('--runs', default=str(RUNS))
    parser.add_argument('--json', required=True, help='where to write the scores')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    result = score_all(args.out, args.baseline, args.gates, args.runs)
    Path(args.json).write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    print(json.dumps(result['passed']))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'score':
        score_main(sys.argv[2:])
    else:
        main()
