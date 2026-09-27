"""FastRaceCue motor assist (`MotorAssistConfig`), declared per motor contract in configs/pilot/motor_assist.json (off by
default; the fast PD contract has no entry).

These tests pin the rule on synthetic states: cap tracking of the binding caps (the pilot's own request, the looming
governor's cap, turn-first's wall ray and creep bound, the stopping model), its bounded reversal and rate limits, the
sag compensation and its exclusions, that the pilot keeps its own request as its state while the support rule reads the
request the motor received, the frozen declaration and runner wiring, and that the pilot with the rule off (default,
and the full round-4 stack) is bit-identical to m4 (3decaac). The surrogate scenarios are in
haltere/liftoff/motor_assist_eval.py. None of this is flight evidence.
"""
import hashlib
import json

import numpy as np
import pytest
import torch

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import FastRaceCue
from haltere.vision.camera import Camera, quat_wxyz_to_mat
from tests.test_fast_race_cue import cue_toward, single_thread  # noqa: F401 (fixture)
from tests.test_visual_assistance import SENSOR, senses

CAMERA = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
CAL = dict(hover_processed=0.13566076755523682, throttle_scale=0.8, hover_stick_sim=-0.43019758448357537)
AHEAD = cue_toward([20., 0., 0.])


def assist(**kw):
    from haltere.liftoff.fast_race_cue import MotorAssistConfig
    return MotorAssistConfig(**kw)


def run(*, seconds=1., velocity=(5., 0., 0.), cue=AHEAD, plant=None, pilot=None, **kw):
    """The pilot at a level attitude and a fixed measured velocity (or a plant: velocity <- f(velocity, command)), a cue
    every tick; rows (t, assisted request, the pilot's own request, state)."""
    history = CameraPoseHistory()
    pilot = pilot or FastRaceCue(SENSOR, history, 6., reference_speed=6., **kw)
    measured = np.array(velocity, float)
    rows = []
    for k in range(int(round(seconds/.01))):
        t = 10.+k*.01
        s = senses(position=(0., 0., 5.), velocity=tuple(measured.tolist()), yaw=0.)
        pilot.pose_history.append(t, [0., 0., 5.], s['quat'][0].numpy())
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(cue)) if cue else None, t-.05, t)
        own = pilot.pilot_command if getattr(pilot, 'pilot_command', None) is not None else pilot.velocity_command
        rows.append((t, pilot.velocity_command.copy(), np.array(own, float), pilot.state))
        if plant is not None:
            measured = plant(measured, pilot.velocity_command)
    return pilot, rows


def bare(config):
    """A pilot past its launch, for direct calls of the rule."""
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., motor_assist=config)
    pilot.launching = False
    return pilot


# ---------------------------------------------------------------------------------------------
# Declaration and configuration
# ---------------------------------------------------------------------------------------------
def test_config_validation():
    from haltere.liftoff.fast_race_cue import MOTOR_ASSIST_SOURCES
    assert set(MOTOR_ASSIST_SOURCES) == {'request', 'governor', 'turn_first', 'stopping'}
    assert assist().cap_sources == MOTOR_ASSIST_SOURCES
    for bad in (dict(cap_gain=0.), dict(cap_deadband=-.1), dict(sag_max=float('nan')), dict(cap_sources=('wall',)),
                dict(cap_sources=('request', 'request')), dict(stop_confirm=1.5), dict(stop_deceleration=0.),
                dict(sag_lead_gain=-.1), dict(request_states=(1,))):
        with pytest.raises(ValueError):
            assist(**bad)
    for removed in ('governor_horizontal', 'turn_first_max_speed', 'climb_gain', 'sag_tilt_gain'):
        with pytest.raises(TypeError):
            assist(**{removed: 0})


def test_contract_entries_and_version():
    from haltere.liftoff.fast_race_cue import MOTOR_ASSIST_VERSION, motor_assist_for_contract
    declaration = dict(version=MOTOR_ASSIST_VERSION, contracts=dict(
        fast_velocity_brain_v1=dict(cap_sources=['request', 'stopping'], request_states=['cue'], cap_gain=2.),
        fast_velocity_pd_v1=None))
    config = motor_assist_for_contract(declaration, 'fast_velocity_brain_v1')
    assert config.cap_sources == ('request', 'stopping') and config.request_states == ('cue',) and config.cap_gain == 2.
    assert motor_assist_for_contract(declaration, 'fast_velocity_pd_v1') is None
    assert motor_assist_for_contract(declaration, 'motor_tracking_teacher_v1') is None
    with pytest.raises(ValueError, match='version'):
        motor_assist_for_contract(dict(declaration, version=MOTOR_ASSIST_VERSION+1), 'fast_velocity_brain_v1')
    with pytest.raises(ValueError, match='contracts'):
        motor_assist_for_contract(dict(version=MOTOR_ASSIST_VERSION), 'fast_velocity_brain_v1')


# ---------------------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------------------
def test_off_by_default_logs_nan_and_adds_no_metadata():
    pilot, rows = run()
    assert pilot.motor_assist is None and 'motor_assist' not in pilot.metadata()
    log = pilot.motor_assist_log()
    assert log['assist_source'] == '' and all(np.isnan(v) for k, v in log.items() if k != 'assist_source')
    assert pilot.pilot_command is None


def test_request_tracking_lowers_the_request_while_the_motor_overshoots_and_releases_after():
    a = assist(cap_sources=('request',), cap_gain=1., cap_deadband=.3, cap_max=2., cap_rise=8., cap_fall=8.,
               sag_gain=0., sag_lead_gain=0.)
    # the pilot's speed schedule asks for less toward a ring 20 deg off the flown course than the motor flies
    off = cue_toward([20., 7.3, 0.])
    pilot, rows = run(velocity=(8., 0., 0.), cue=off, motor_assist=a, seconds=1.5)
    _, flown, own, state = rows[-1]
    u = own[:2]/np.linalg.norm(own[:2])
    excess = 8.*u[0]-np.linalg.norm(own[:2])
    assert state == 'cue' and excess > 1.
    x = pilot.assist_extra['request']
    assert x == pytest.approx(min(2., excess-.3), abs=.02)
    assert np.linalg.norm(flown[:2]) == pytest.approx(np.linalg.norm(own[:2])-x, abs=1e-9)
    assert flown[:2] @ u == pytest.approx(np.linalg.norm(flown[:2]), abs=1e-9)        # same direction, less speed
    assert pilot.assist_source == 'request' and pilot.assist_horizontal > 0
    # the extra reduction grows at cap_rise: after 0.05 s at most 0.4 m/s
    _, first = run(velocity=(8., 0., 0.), cue=off, motor_assist=a, seconds=.05)
    assert np.linalg.norm(first[-1][2][:2])-np.linalg.norm(first[-1][1][:2]) <= 8.*.05+1e-9
    # a motor that follows the request gets no reduction
    _, follow = run(velocity=(0., 0., 0.), cue=off, motor_assist=a, seconds=1.,
                    plant=lambda v, c: np.array(c, float))
    assert all(np.allclose(r[1], r[2]) for r in follow[-50:])


def test_the_request_never_reverses_along_its_own_direction():
    a = assist(cap_sources=('request',), cap_gain=5., cap_max=5., cap_rise=100., sag_gain=0., sag_lead_gain=0.)
    pilot, rows = run(velocity=(9., 0., 0.), cue=cue_toward([6., 4., 0.]), motor_assist=a, seconds=1.)
    for _, flown, own, _ in rows:
        n = np.linalg.norm(own[:2])
        if n > 1e-6:
            assert flown[:2] @ (own[:2]/n) >= -1e-9
    assert min(np.linalg.norm(r[1][:2]) for r in rows[-20:]) == pytest.approx(0., abs=1e-9)    # stopped, not reversed


def test_motor_assist_math_for_wall_sources_and_the_reverse_bound():
    pilot = bare(assist(cap_gain=2., cap_max=3., cap_reverse=1., cap_rise=1000., sag_gain=0.))
    wall = np.array([1., 0.])
    command = np.array([2., 1., 0.])
    # turn-first: no speed toward the wall (c = 0); the motor still closes at 4 m/s -> reverse, bounded at 1 m/s
    out = pilot._motor_assist(command, np.array([4., 0., 0.]), .01, 'side', [('turn_first', wall, 0.)])
    assert out[:2] @ wall == pytest.approx(-1.) and out[1] == pytest.approx(1.) and out[2] == 0.
    # the governor's cap 3 along the wall: measured 4 -> bound 3 - 2*(1 - 0.3) = 1.6
    pilot.assist_extra = dict.fromkeys(pilot.assist_extra, 0.)
    out = pilot._motor_assist(np.array([3., 0., 0.]), np.array([4., 0., 0.]), .01, 'cue', [('governor', wall, 3.)])
    assert out[0] == pytest.approx(1.6) and pilot.assist_source == 'governor'
    # the stopping model's bound applies as a cap even before the motor exceeds it
    pilot.assist_extra = dict.fromkeys(pilot.assist_extra, 0.)
    out = pilot._motor_assist(np.array([5., 0., 0.]), np.array([2., 0., 0.]), .01, 'cue', [('stopping', wall, 2.5)])
    assert out[0] == pytest.approx(2.5) and pilot.assist_source == 'stopping'
    # nothing binds: the request passes unchanged and no source is logged
    out = pilot._motor_assist(np.array([5., 0., 0.]), np.array([5., 0., 0.]), .01, 'cue', [])
    assert np.array_equal(out, [5., 0., 0.]) and pilot.assist_source == '' and pilot.assist_horizontal == 0.


def test_stopping_bound_is_the_declared_stopping_model():
    from haltere.liftoff.fast_race_cue import stopping_speed
    a = assist()
    for distance in (1., 3., 6.):
        v = stopping_speed(distance, a.stop_deceleration, a.stop_latency_s, a.stop_margin_m)
        assert v*a.stop_latency_s+v*v/(2*a.stop_deceleration)+a.stop_margin_m == pytest.approx(distance)


def wall_run(assist_config, seconds=1.5, speed=5., wall_x=12., clear_after=None):
    """Straight at a wall at wall_x with perfect looming samples every 0.055 s (received 0.085 s later) and a motor
    that keeps its speed (a lagging brain): the requests along x. From `clear_after` (x) the samples see no wall (a
    long TTC: a gate arch flown through)."""
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., motor_assist=assist_config)
    x, rows, pending, next_sample = 0., [], [], 0.
    for k in range(int(round(seconds/.01))):
        t = 10.+k*.01
        s = senses(position=(x, 0., 5.), velocity=(speed, 0., 0.), yaw=0.)
        history.append(t, [x, 0., 5.], s['quat'][0].numpy())
        if k*.01 >= next_sample:
            ttc = (wall_x-x)/speed if clear_after is None or x < clear_after else 5.
            pending.append((t+.085, dict(time=t, ttc=ttc, distance=ttc*speed, below_fraction=.5, ttc_lower=ttc)))
            next_sample += .055
        sample = None
        while pending and pending[0][0] <= t:
            sample = pending.pop(0)[1]
        extra = dict(clearance=sample) if sample is not None else {}
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(cue_toward([wall_x+10.-x, 0., 0.]))), t-.05, t, **extra)
        rows.append((t, x, pilot.velocity_command[0], pilot.assist_source if pilot.motor_assist else ''))
        x += speed*.01
    return pilot, rows


def test_stopping_source_starts_braking_before_the_governor():
    base, rows_off = wall_run(None)
    on, rows_on = wall_run(assist(cap_sources=('governor', 'stopping'), sag_gain=0., sag_lead_gain=0.))
    first = lambda rows: next((x for _, x, c, _ in rows if c < 5.), None)
    # without the stopping source the governor's first cut comes later (closer to the wall) than the stopping model's
    assert first(rows_on) is not None and (first(rows_off) is None or first(rows_on) < first(rows_off))
    assert any(src == 'stopping' for *_, src in rows_on)
    assert on.motor_assist_summary()['seconds']['stopping'] > 0


def test_a_clear_sample_ends_the_stopping_bound():
    """A wall sample that stops arriving (a gate arch flown through) does not keep bounding the speed: the stopping
    source needs the newest looming sample to be a confirming wall sample."""
    a = assist(cap_sources=('stopping',), sag_gain=0., sag_lead_gain=0.)
    pilot, rows = wall_run(a, seconds=1.4, wall_x=9., clear_after=5.)
    stopped = [x for _, x, _, src in rows if src == 'stopping']
    assert stopped and min(stopped) < 5.
    # the first clear sample is received by x = 5 + 5 m/s x (0.055 + 0.085) s: no stopping bound after that
    assert max(stopped) <= 5.+5.*(.055+.085)+1e-9
    assert all(src == '' for _, x, _, src in rows if x > 6.)


def test_sag_compensation_adds_a_bounded_climb_and_respects_its_exclusions():
    a = assist(cap_sources=(), sag_gain=1., sag_deadband=.3, sag_max=1., sag_rise=5., sag_fall=2.)
    pilot, rows = run(velocity=(3., 0., -1.5), motor_assist=a, seconds=1.)
    _, flown, own, _ = rows[-1]
    assert flown[2]-own[2] == pytest.approx(1.)                           # (own - (-1.5) - 0.3) clipped to sag_max
    _, early = run(velocity=(3., 0., -1.5), motor_assist=a, seconds=.1)
    assert early[-1][1][2]-early[-1][2][2] <= 5.*.1+1e-9                  # rise rate
    # a motor that tracks its vertical request gets nothing
    _, level = run(velocity=(3., 0., 0.), motor_assist=a, seconds=.5)
    assert all(r[1][2] == r[2][2] for r in level)
    # not while launching
    history = CameraPoseHistory()
    launch = FastRaceCue(SENSOR, history, 6., reference_speed=6., motor_assist=a)
    s = senses(position=(0., 0., .2), velocity=(0., 0., -1.), yaw=0.)
    for k in range(50):
        history.append(10.+k*.01, [0., 0., .2], s['quat'][0].numpy())
        launch.update(s, [0., 0., 0.], dict(race_cue=dict(AHEAD)), 10.+k*.01-.05, 10.+k*.01)
    assert launch.state == 'launch' and launch.assist_vertical == 0.
    # the ceiling guard's overhead bound caps the added climb
    pilot2 = bare(a)
    out = pilot2._motor_assist(np.array([0., 0., -.2]), np.array([0., 0., -2.]), 1., 'cue', [], vertical_cap=0.)
    assert out[2] == pytest.approx(0.)
    # and nothing in a support climb
    pilot3 = bare(a)
    out = pilot3._motor_assist(np.array([0., 0., 1.]), np.array([0., 0., -2.]), 1., 'support_climb', [])
    assert out[2] == 1. and pilot3.assist_vertical == 0.


def test_lead_bias_acts_at_low_speed_before_the_sink():
    a = assist(cap_sources=(), sag_gain=0., sag_lead_gain=.3, sag_lead_deadband=.5, sag_lead_max_speed=2.5,
               sag_rise=1000.)
    pilot = bare(a)
    # at 0.5 m/s asked for 4 m/s: a bias of 0.3 x (3.5 - 0.5) before any sink
    out = pilot._motor_assist(np.array([4., 0., 0.]), np.array([.5, 0., 0.]), .01, 'cue', [])
    assert out[2] == pytest.approx(.9) and pilot.assist_vertical == pytest.approx(.9)
    # not above sag_lead_max_speed, and not while the drone climbs faster than asked
    pilot.assist_sag = 0.
    assert pilot._motor_assist(np.array([6., 0., 0.]), np.array([3., 0., 0.]), .01, 'cue', [])[2] == 0.
    pilot.assist_sag = 0.
    assert pilot._motor_assist(np.array([4., 0., 0.]), np.array([.5, 0., .5]), .01, 'cue', [])[2] == 0.
    # sag_lead_max_speed 0: at any speed
    anywhere = bare(assist(cap_sources=(), sag_gain=0., sag_lead_gain=.3, sag_lead_deadband=.5, sag_lead_max_speed=0.,
                           sag_rise=1000.))
    assert anywhere._motor_assist(np.array([6., 0., 0.]), np.array([3., 0., 0.]), .01, 'cue', [])[2] == pytest.approx(.75)


def test_the_pilot_keeps_its_own_request_as_its_state():
    """Fed the same recorded states, the pilot's own request with the assist equals the request without it (no
    descent is requested here: only the support rule and the descent-path shortfall read the issued request)."""
    off = cue_toward([10., 12., 0.])
    _, plain = run(velocity=(6., 0., -1.), cue=off, seconds=1.5)
    _, assisted = run(velocity=(6., 0., -1.), cue=off, seconds=1.5, motor_assist=assist())
    assert all(np.array_equal(p[1], a[2]) for p, a in zip(plain, assisted))
    assert any(not np.array_equal(a[1], a[2]) for a in assisted)


def test_the_support_rule_reads_the_vertical_request_the_motor_received():
    """A descent the assist withheld (it raised the motor's vertical request) is no evidence of ground contact."""
    def descend(config):
        pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., reference_speed=6., motor_assist=config)
        s = senses(position=(0., 0., 5.), velocity=(3., 0., -.2), yaw=0.)
        pilot.pose_history.append(10., [0., 0., 5.], s['quat'][0].numpy())
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(AHEAD)), 9.95, 10.)       # launch done (z above 0.6 m)
        own = np.array([3., 0., -1.])
        if config is None:
            pilot.velocity_command = own
        else:
            pilot.pilot_command, pilot.velocity_command = own, np.array([3., 0., -.4])
        pilot.pose_history.append(10.01, [0., 0., 5.], s['quat'][0].numpy())
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(AHEAD)), 9.96, 10.01)
        return pilot
    # the pilot asked for 1 m/s of sink and the drone sinks at 0.2 m/s: without the assist that is support evidence
    assert descend(None).support_since is not None
    # the motor was asked for 0.4 m/s (0.6 m/s withheld): no support evidence
    assert descend(assist()).support_since is None


def test_metadata_records_the_rule_and_its_activity():
    pilot, _ = run(motor_assist=assist(), velocity=(6., 0., -1.), cue=cue_toward([10., 12., 0.]), seconds=.5)
    meta = pilot.metadata()['motor_assist']
    assert meta['version'] == 1 and meta['parameters']['cap_sources'] == ['request', 'governor', 'turn_first', 'stopping']
    assert set(meta['seconds']) == {'request', 'governor', 'turn_first', 'stopping', 'sag'}
    assert 'no course' in meta['input']


# ---------------------------------------------------------------------------------------------
# Identity with the rule off: bit-identical to m4 (3decaac)
# ---------------------------------------------------------------------------------------------
def stack_scenario(**extra):
    """A scripted flight through the full round-4 stack as the runner builds it for the fast brain contract (lag turn
    v2, gap aim v5 without gap samples, wall pilot v4, vertical guard v3, descent view v1) with synthetic looming
    samples of a wall and of the ground and a lagging plant. Uses only names that m4 has (the m4 digest below was
    computed with this function on `git archive 3decaac`)."""
    from haltere.liftoff import fast_race_cue as frc
    from haltere.liftoff import visual_brain as vb
    from haltere.liftoff.gap_aim import GapAimConfig
    from haltere.liftoff.gap_stack import load_gap_pilot
    lag, _ = vb.load_lag_turn_declaration(vb.LAG_TURN_DECLARATION)
    wall, _ = vb.load_wall_pilot()
    vertical, _ = vb.load_vertical_guard()
    gap, _ = load_gap_pilot()
    view, _ = vb.load_descent_view()
    contract = 'fast_velocity_brain_v1'
    kw = dict(lag_turn=frc.lag_turn_for_contract(lag, contract), gap_aim=GapAimConfig.from_dict(gap['pilot']),
              vertical_guard=frc.vertical_guard_config(vertical), descent_view=frc.descent_view_config(view),
              **frc.wall_pilot_configs(wall, contract))
    kw.update(extra)
    history = CameraPoseHistory()
    pilot = frc.FastRaceCue(dict(focal_320=100., tilt_deg=30.), history, 6., reference_speed=6., calibration=CAL, **kw)
    position, velocity = np.array([0., 0., 1.2]), np.zeros(3)
    rings = [np.array([20., 0., 1.]), np.array([33., .5, .9]), np.array([29., 9., 1.3]), np.array([5., 16., 1.])]
    wall_x, target, out, pending = 36., 0, [], []
    for k in range(2200):
        now = k*.01
        pitch = np.radians(np.clip(3*(velocity[0]-3.), -25, 25))*.3
        yaw = float(np.arctan2(velocity[1], velocity[0])) if np.hypot(velocity[0], velocity[1]) > .5 else 0.
        q = np.array([np.cos(yaw/2)*np.cos(pitch/2), -np.sin(yaw/2)*np.sin(pitch/2), np.cos(yaw/2)*np.sin(pitch/2),
                      np.sin(yaw/2)*np.cos(pitch/2)])
        history.append(now, position.copy(), q)
        if np.linalg.norm(rings[target]-position) < 2.5 and target < len(rings)-1:
            target += 1
        body = (rings[target]-position) @ quat_wxyz_to_mat(q)
        px, ok = CAMERA.project_body(body[None])
        u, v = px[0, 0]/320, px[0, 1]/180
        edge = not (ok[0] and 0 <= u <= 1 and 0 <= v <= 1)
        cue = dict(u=float(np.clip(u, 0, 1)) if ok[0] else .02, v=float(np.clip(v, 0, 1)) if ok[0] else .5, edge=edge)
        if k % 5 == 0:
            ttc_wall = (wall_x-position[0])/velocity[0] if velocity[0] > .5 else None
            ground = k % 15 == 0 and position[2] < 1.
            sample = dict(time=now, ttc=(ttc_wall if not ground else .9), distance=None,
                          below_fraction=.5 if not ground else .9,
                          ttc_lower=ttc_wall if not ground else .7)
            if sample['ttc'] is not None:
                sample['distance'] = sample['ttc']*max(np.linalg.norm(velocity), .1)
            pending.append((now+.085, sample))
        sample = None
        while pending and pending[0][0] <= now:
            sample = pending.pop(0)[1]
        t = lambda x: torch.tensor([x], dtype=torch.float32)
        senses_ = dict(pos=t(position.tolist()), quat=t(q.tolist()), vel_world=t(velocity.tolist()),
                       vel_body=t((quat_wxyz_to_mat(q).T @ velocity).tolist()), gyro=t([0., 0., 0.]))
        extra_kw = dict(clearance=sample) if sample is not None else {}
        pilot.update(senses_, [0., 0., 0.], dict(race_cue=cue) if k % 9 else None, now-.05, now, **extra_kw)
        pilot.command(np.array([-.45+.1*np.sin(k*.05), 0., 0., 0.]))
        state = ['launch', 'wait', 'cue', 'below', 'below_weak', 'above', 'side', 'coast', 'search',
                 'support_climb'].index(pilot.state)
        out.append(np.r_[pilot.velocity_command, pilot.feedforward, pilot.pilot.sight_yaw, pilot.descent_scale, state])
        velocity = velocity+.05*(pilot.velocity_command-velocity)
        position = position+velocity*.01
        position[2] = max(position[2], 0.)
    return np.asarray(out), pilot


M4_STACK_DIGEST = '34cb4a631c3814539d948f9712be231558a008862429cb7cda4d11834374f043'


def test_full_stack_with_the_rule_off_is_bit_identical_to_m4():
    trace, pilot = stack_scenario()
    digest = hashlib.sha256(np.ascontiguousarray(trace.astype(np.float64)).tobytes()).hexdigest()
    # the scenario exercises the stack: wall brakes, guard climbs, lag turns and the view bound
    assert pilot.clearance.counts['brake_engagements'] > 0 and pilot.clearance.vertical_counts['gentle_climbs'] > 0
    assert pilot.lag_turn_triggers > 0 and pilot.descent_view_summary()['seconds']['limiting'] > 0
    assert digest == M4_STACK_DIGEST
    assisted, _ = stack_scenario(motor_assist=assist())
    assert not np.array_equal(assisted, trace)


# ---------------------------------------------------------------------------------------------
# Runner and replay harness
# ---------------------------------------------------------------------------------------------
def test_runner_flag_columns_and_refusals(tmp_path):
    from types import SimpleNamespace
    from haltere.liftoff.visual_brain import (MOTOR_ASSIST_COLUMNS, MOTOR_ASSIST_DECLARATION, VisualController,
                                              load_motor_assist, motor_assist_row, resolve_motor_assist)
    assert resolve_motor_assist(SimpleNamespace()) is None
    assert resolve_motor_assist(SimpleNamespace(motor_assist='off')) is None
    assert resolve_motor_assist(SimpleNamespace(motor_assist='on')) == str(MOTOR_ASSIST_DECLARATION)
    assert resolve_motor_assist(SimpleNamespace(motor_assist='x.json')) == 'x.json'
    row = motor_assist_row(None)
    assert len(row) == len(MOTOR_ASSIST_COLUMNS) == 6 and all(np.isnan(row[:5])) and row[5] == ''
    pilot, _ = run(motor_assist=assist(), velocity=(6., 0., -1.), cue=cue_toward([10., 12., 0.]), seconds=.5)
    values = motor_assist_row(pilot)
    assert all(np.isfinite(values[:5])) and isinstance(values[5], str)
    with pytest.raises(ValueError, match='fast pilot'):
        VisualController('missing.pt', 'missing.json', 'cpu', pilot_assistance='race-cue', motor_assist='x.json')
    unfrozen = tmp_path/'assist.json'
    unfrozen.write_text(json.dumps(dict(version=1, contracts={})), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen'):
        load_motor_assist(unfrozen)


def test_runner_csv_tail_puts_the_assist_columns_after_the_view_columns():
    import inspect
    import re
    from haltere.liftoff import visual_brain
    source = re.sub(r'\s+', '', inspect.getsource(visual_brain.run))
    assert '*VERTICAL_COLUMNS,*COMMIT_COLUMNS,*view_columns,*assist_columns])' in source
    assert ('*(descent_view_row(controller.assistance)ifview_columnselse()),'
            '*(motor_assist_row(controller.assistance)ifassist_columnselse())])') in source


def test_scenario_wall_geometry():
    from haltere.liftoff.motor_assist_eval import _wall_gap, _wall_hit, hairpin_scenario
    sc = hairpin_scenario(turn_deg=90., arch_m=7., wall_m=2.)
    (a, b), = sc['walls']
    assert np.allclose(a[0], b[0]) and a[0] == pytest.approx(9.)          # a wall across the eastward R2 leg
    hit = _wall_hit(np.array([5., 20., .8]), np.array([4., 0., 0.]), np.array([1., 0.]), sc['walls'])
    assert hit == (pytest.approx(4.), pytest.approx(4.))
    assert _wall_hit(np.array([5., 20., .8]), np.array([-4., 0., 0.]), np.array([-1., 0.]), sc['walls']) is None
    assert _wall_hit(np.array([5., 20., .8]), np.array([4., 0., 0.]), np.array([0., 1.]), sc['walls']) is None
    assert _wall_gap(np.array([8.5, 20., .8]), sc['walls']) == pytest.approx(.5)


# ---------------------------------------------------------------------------------------------
# The frozen declaration and gates
# ---------------------------------------------------------------------------------------------
DECLARATION_SHA256 = 'eefb4a42613cabe5bd9c120825c42274ab72f8f7b970750aadb26334d3781d02'


def test_frozen_declaration_assigns_the_rule_to_the_brain_contract_only(tmp_path):
    from haltere.liftoff.fast_race_cue import MotorAssistConfig, motor_assist_for_contract
    from haltere.liftoff.visual_brain import MOTOR_ASSIST_DECLARATION, load_motor_assist
    declaration, digest = load_motor_assist()
    assert digest == DECLARATION_SHA256 and declaration['version'] == 1
    # the declared brain entry is the code's default rule; the fast PD (and any other contract) has none
    assert motor_assist_for_contract(declaration, 'fast_velocity_brain_v1') == MotorAssistConfig()
    assert motor_assist_for_contract(declaration, 'fast_velocity_pd_v1') is None
    assert 'fast_velocity_pd_v1' in declaration['contracts']
    # an edited copy or another version is refused
    edited = json.loads(MOTOR_ASSIST_DECLARATION.read_text(encoding='utf-8'))
    edited['contracts']['fast_velocity_brain_v1']['cap_gain'] = 3.
    path = tmp_path/'edited.json'
    path.write_text(json.dumps(edited), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen'):
        load_motor_assist(path)
    from haltere.train.brake_gates import gates_sha256
    other = dict(declaration, version=2)
    other['sha256'] = gates_sha256(other)
    path.write_text(json.dumps(other), encoding='utf-8')
    with pytest.raises(ValueError, match='version'):
        load_motor_assist(path)


def test_frozen_gates_name_the_frozen_declaration():
    from haltere.liftoff.motor_assist_gates import load_gates
    gates, digest, declaration = load_gates()
    assert gates['motor_assist']['sha256'] == DECLARATION_SHA256 == declaration['sha256']
    # the held-out scenario sets and seeds differ from the development sets the rule was designed on
    assert gates['hairpin']['sim_seed'] != 17 and gates['accelerate']['sim_seed'] != 17
    assert not set(gates['hairpin']['set']['turn_deg']) & {20., 60., 90.}
    assert not set(gates['accelerate']['set']['bearing_deg']) & {0., 90., 150.}
    assert set(gates['controllers']) == {'brain08', 'brain09b', 'brain10b'}


def test_replay_harness_adds_nothing_for_the_fast_pd():
    """vertical_replay.build: with a motor-assist declaration the fast PD's pilot is built without an assist (the
    replay identity gate then checks the arrays bit for bit), a brain's with it."""
    from pathlib import Path
    from haltere.liftoff.visual_brain import load_motor_assist
    from haltere.obstacles.vertical_replay import build
    declaration, _ = load_motor_assist()
    tree = Path(__file__).resolve().parents[1]
    side = dict(motor_controller=dict(contract='fast_velocity_pd_v1'), pilot_assistance=dict(nominal_speed_mps=6.),
                gate_sensor=dict(focal_320=100., tilt_deg=30.))
    pilot, info = build(side, tree, stack='on', motor_assist=declaration)
    assert pilot.motor_assist is None and info['motor_assist'] is False
    side['motor_controller']['contract'] = 'fast_velocity_brain_v1'
    pilot, info = build(side, tree, stack='on', motor_assist=declaration)
    assert pilot.motor_assist is not None and info['motor_assist'] is True
