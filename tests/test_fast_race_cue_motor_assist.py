"""FastRaceCue motor assist (`MotorAssistConfig`), declared per motor contract in configs/pilot/motor_assist.json (off by
default; the fast PD contract has no entry).

These tests pin the rule on synthetic states: cap tracking of the binding caps (the pilot's own request, the looming
governor's cap, turn-first's wall ray and creep bound, the stopping model), its bounded reversal and rate limits, the
sag compensation and its exclusions, that the pilot keeps its own request as its state while the support rule reads the
request the motor received, the frozen declaration and runner wiring, and that the pilot with the rule off (default,
and the full round-4 stack) is bit-identical to m4 (3decaac). Versions 2 and 3 (version 3 is the declaration the
runner flies: version 2's rule without the approach's climb exclusion): the slew
bound on the assist's change, the wall-ahead gate of the stopping source with its memory, the floored approach source
and its climb exclusion, no governor tracking during a stand-off, and version 1 rebuilt bit-identically from its kept
declaration (the digest computed on `git archive m4b`). The surrogate scenarios are in
haltere/liftoff/motor_assist_eval.py. None of this is flight evidence.
"""
import hashlib
import json
from pathlib import Path

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
    """A version-3 rule (the code's defaults are the declared brain entry; version 2 differs only in
    approach_climb_max)."""
    from haltere.liftoff.fast_race_cue import MotorAssistConfig
    return MotorAssistConfig(**kw)


def assist_v1(**kw):
    """A version-1 rule as version 1 flew it (MOTOR_ASSIST_V1_FIELDS; replays only)."""
    from haltere.liftoff.fast_race_cue import MOTOR_ASSIST_SOURCES, MOTOR_ASSIST_V1_FIELDS, MotorAssistConfig
    return MotorAssistConfig(**{'cap_sources': MOTOR_ASSIST_SOURCES[:4], **MOTOR_ASSIST_V1_FIELDS, 'version': 1, **kw})


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
    assert set(MOTOR_ASSIST_SOURCES) == {'request', 'governor', 'turn_first', 'stopping', 'approach'}
    assert assist().cap_sources == MOTOR_ASSIST_SOURCES and assist().version == 3 and assist().approach_climb_max == 3.5
    assert assist_v1().cap_sources == MOTOR_ASSIST_SOURCES[:4] and assist_v1().slew == 0.
    for bad in (dict(cap_gain=0.), dict(cap_deadband=-.1), dict(sag_max=float('nan')), dict(cap_sources=('wall',)),
                dict(cap_sources=('request', 'request')), dict(stop_confirm=1.5), dict(stop_deceleration=0.),
                dict(sag_lead_gain=-.1), dict(request_states=(1,)), dict(stop_gate='never'), dict(version=4),
                dict(wall_ahead_deg=0.), dict(slew=-1.), dict(approach_climb_max=float('nan')), dict(version=1)):
        with pytest.raises(ValueError):
            assist(**bad)
    with pytest.raises(ValueError):
        assist_v1(cap_sources=('request', 'approach'))
    for removed in ('governor_horizontal', 'turn_first_max_speed', 'climb_gain', 'sag_tilt_gain'):
        with pytest.raises(TypeError):
            assist(**{removed: 0})


def test_contract_entries_and_version():
    from haltere.liftoff.fast_race_cue import (MOTOR_ASSIST_V1_FIELDS, MOTOR_ASSIST_VERSION, MOTOR_ASSIST_VERSIONS,
                                               motor_assist_for_contract)
    assert MOTOR_ASSIST_VERSION == 4 and MOTOR_ASSIST_VERSIONS == (1, 2, 3, 4)
    declaration = dict(version=3, contracts=dict(
        fast_velocity_brain_v1=dict(cap_sources=['request', 'stopping'], request_states=['cue'], cap_gain=2.),
        fast_velocity_pd_v1=None))
    config = motor_assist_for_contract(declaration, 'fast_velocity_brain_v1')
    assert config.cap_sources == ('request', 'stopping') and config.request_states == ('cue',) and config.cap_gain == 2.
    assert config.version == 3 and config.slew == 15.
    # a version-1 entry is rebuilt as version 1 flew it; it carries no version-2 field
    old = dict(version=1, contracts=dict(fast_velocity_brain_v1=dict(cap_gain=2.)))
    rebuilt = motor_assist_for_contract(old, 'fast_velocity_brain_v1')
    assert rebuilt.version == 1 and all(getattr(rebuilt, k) == v for k, v in MOTOR_ASSIST_V1_FIELDS.items())
    assert 'approach' not in rebuilt.cap_sources
    with pytest.raises(ValueError, match='version-2'):
        motor_assist_for_contract(dict(version=1, contracts=dict(fast_velocity_brain_v1=dict(slew=15.))),
                                  'fast_velocity_brain_v1')
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
    a = assist_v1(cap_sources=('request',), cap_gain=1., cap_deadband=.3, cap_max=2., cap_rise=8., cap_fall=8.,
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
    a = assist_v1(cap_sources=('request',), cap_gain=5., cap_max=5., cap_rise=100., sag_gain=0., sag_lead_gain=0.)
    pilot, rows = run(velocity=(9., 0., 0.), cue=cue_toward([6., 4., 0.]), motor_assist=a, seconds=1.)
    for _, flown, own, _ in rows:
        n = np.linalg.norm(own[:2])
        if n > 1e-6:
            assert flown[:2] @ (own[:2]/n) >= -1e-9
    assert min(np.linalg.norm(r[1][:2]) for r in rows[-20:]) == pytest.approx(0., abs=1e-9)    # stopped, not reversed


def test_motor_assist_math_for_wall_sources_and_the_reverse_bound():
    pilot = bare(assist_v1(cap_gain=2., cap_max=3., cap_reverse=1., cap_rise=1000., sag_gain=0.))
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
    on, rows_on = wall_run(assist_v1(cap_sources=('governor', 'stopping'), sag_gain=0., sag_lead_gain=0.))
    first = lambda rows: next((x for _, x, c, _ in rows if c < 5.), None)
    # without the stopping source the governor's first cut comes later (closer to the wall) than the stopping model's
    assert first(rows_on) is not None and (first(rows_off) is None or first(rows_on) < first(rows_off))
    assert any(src == 'stopping' for *_, src in rows_on)
    assert on.motor_assist_summary()['seconds']['stopping'] > 0


def test_a_clear_sample_ends_the_stopping_bound():
    """A wall sample that stops arriving (a gate arch flown through) does not keep bounding the speed: the stopping
    source needs the newest looming sample to be a confirming wall sample."""
    a = assist_v1(cap_sources=('stopping',), sag_gain=0., sag_lead_gain=0.)
    pilot, rows = wall_run(a, seconds=1.4, wall_x=9., clear_after=5.)
    stopped = [x for _, x, _, src in rows if src == 'stopping']
    assert stopped and min(stopped) < 5.
    # the first clear sample is received by x = 5 + 5 m/s x (0.055 + 0.085) s: no stopping bound after that
    assert max(stopped) <= 5.+5.*(.055+.085)+1e-9
    assert all(src == '' for _, x, _, src in rows if x > 6.)


def test_sag_compensation_adds_a_bounded_climb_and_respects_its_exclusions():
    a = assist_v1(cap_sources=(), sag_gain=1., sag_deadband=.3, sag_max=1., sag_rise=5., sag_fall=2.)
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
    a = assist_v1(cap_sources=(), sag_gain=0., sag_lead_gain=.3, sag_lead_deadband=.5, sag_lead_max_speed=2.5,
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
    anywhere = bare(assist_v1(cap_sources=(), sag_gain=0., sag_lead_gain=.3, sag_lead_deadband=.5, sag_lead_max_speed=0.,
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
    assert meta['version'] == 3 and meta['parameters']['cap_sources'] == ['request', 'governor', 'turn_first', 'stopping',
                                                                          'approach']
    assert set(meta['seconds']) == {'request', 'governor', 'turn_first', 'stopping', 'approach', 'sag'}
    assert set(meta['v2_seconds']) == {'wall_ahead', 'slew_limited', 'approach', 'stopping'}
    assert 'no course' in meta['input'] and 'version 2' in meta['rule']
    pilot, _ = run(motor_assist=assist_v1(), velocity=(6., 0., -1.), cue=cue_toward([10., 12., 0.]), seconds=.5)
    meta = pilot.metadata()['motor_assist']
    assert meta['version'] == 1 and 'v2_seconds' not in meta and 'version 2' not in meta['rule']


# ---------------------------------------------------------------------------------------------
# Version 2: slew, wall ahead, approach, stand-off
# ---------------------------------------------------------------------------------------------
SIDE = dict(u=.98, v=.5, edge=True)                 # the marker clamped at the right edge: pilot state 'side'


def wall_run_v2(config, *, seconds=1.8, speed=5., wall_x=12., cue=None, samples_until=None, clear_from=None):
    """wall_run with full rows: a motor at a constant speed straight at a wall at wall_x (perfect looming every 0.055 s,
    received 0.085 s later); `cue(x)` gives the marker (default: a ring straight ahead beyond the wall); no samples
    from `samples_until` (x), long-TTC samples (a clear view) from `clear_from` (x). Rows: dict(t, x, own, flown, source,
    plan, wall_ahead, sources)."""
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., motor_assist=config)
    x, rows, pending, next_sample = 0., [], [], 0.
    for k in range(int(round(seconds/.01))):
        t = 10.+k*.01
        s = senses(position=(x, 0., 5.), velocity=(speed, 0., 0.), yaw=0.)
        history.append(t, [x, 0., 5.], s['quat'][0].numpy())
        if k*.01 >= next_sample:
            if samples_until is None or x < samples_until:
                ttc = (wall_x-x)/speed if clear_from is None or x < clear_from else 5.
                pending.append((t+.085, dict(time=t, ttc=ttc, distance=ttc*speed, below_fraction=.5, ttc_lower=ttc)))
            next_sample += .055
        sample = None
        while pending and pending[0][0] <= t:
            sample = pending.pop(0)[1]
        extra = dict(clearance=sample) if sample is not None else {}
        marker = cue(x) if cue is not None else cue_toward([wall_x+10.-x, 0., 0.])
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(marker)), t-.05, t, **extra)
        rows.append(dict(t=t, x=x, own=np.array(pilot.pilot_command, float), flown=pilot.velocity_command.copy(),
                         source=pilot.assist_source, plan=pilot.assist_plan, wall_ahead=pilot.assist_wall_ahead,
                         sources=[name for name, _, _ in pilot.assist_sources]))
        x += speed*.01
    return pilot, rows


def test_v2_slew_bounds_the_assists_change_per_tick_on_onsets_and_releases():
    """Version 1 stepped the request by metres per second in one tick at a stopping onset; version 2 moves the assist's
    change (flown - own) by at most slew x dt, horizontally and vertically, also when a clear sample releases it."""
    _, rows_v1 = wall_run_v2(assist_v1(), wall_x=9., clear_from=5.5, seconds=1.2)
    steps_v1 = [np.linalg.norm((b['flown']-b['own'])[:2]-(a['flown']-a['own'])[:2]) for a, b in zip(rows_v1, rows_v1[1:])]
    assert max(steps_v1) > .5                                          # > 3 x the slew room
    config = assist()
    for kw in (dict(wall_x=9., clear_from=5.5, seconds=1.2), dict(cue=lambda x: SIDE)):
        pilot, rows = wall_run_v2(config, **kw)
        delta = np.array([r['flown']-r['own'] for r in rows])
        assert np.abs(delta).max() > .5                                    # the assist acted
        room = config.slew*.01+1e-9
        assert np.linalg.norm(np.diff(delta[:, :2], axis=0), axis=1).max() <= room
        assert np.abs(np.diff(delta[:, 2])).max() <= room
    # the release: after the clear sample the change returns to exactly zero, at the slew
    _, rows = wall_run_v2(config, wall_x=9., clear_from=5.5, seconds=1.6)
    assert np.array_equal(rows[-1]['flown'], rows[-1]['own'])


def test_v2_with_the_ring_ahead_the_stopping_model_only_approaches_never_below_the_floor():
    """The ring straight ahead through the looming surface (a gate arch flown through): no wall-ahead condition, the
    stopping model plans the approach and its bound never falls below floor_speed, even at the surface."""
    config = assist()
    pilot, rows = wall_run_v2(config, wall_x=9., seconds=1.8)
    plans = [r['plan'] for r in rows if np.isfinite(r['plan'])]
    assert plans and min(plans) == pytest.approx(config.floor_speed)
    assert not any(r['wall_ahead'] for r in rows) and all('stopping' not in r['sources'] for r in rows)
    assert any(r['source'] == 'approach' for r in rows)
    assert pilot.motor_assist_summary()['v2_seconds']['approach'] > 0
    # the approach bound is the declared stopping model, floored
    from haltere.liftoff.fast_race_cue import stopping_speed
    far = [r for r in rows if 'approach' in r['sources'] and r['plan'] > config.floor_speed+.1]
    assert far and all(r['plan'] >= config.floor_speed for r in far)
    v = stopping_speed(3., config.stop_deceleration, config.stop_latency_s, config.stop_margin_m)
    assert v == pytest.approx(max(config.floor_speed, v)) or v < config.floor_speed


def test_v2_wall_ahead_stops_and_keeps_the_wall_until_a_clear_view():
    """The marker clamped at the side (the next ring does not lie through the surface): the stopping source plans a stop
    at the wall; once the samples stop (no evidence at arm's length) it keeps the latest wall for stop_memory_s after
    the last confirmation, and a clear sample ends it at once."""
    config = assist()
    _, rows = wall_run_v2(config, cue=lambda x: SIDE, wall_x=9., seconds=1.75)
    stops = [r for r in rows if 'stopping' in r['sources']]
    assert stops and all(r['wall_ahead'] for r in stops) and min(r['plan'] for r in stops) == pytest.approx(0.)
    # no samples from x = 3.5 m (TTC 1.17 s there): the stop continues on memory, then ends stop_memory_s after the
    # last confirmation
    short = assist(stop_memory_s=.5)
    _, rows = wall_run_v2(short, cue=lambda x: SIDE, speed=3., wall_x=7., samples_until=3.5, seconds=2.2)
    last_sample = max(r['t'] for r in rows if r['x'] < 3.5)+.085
    kept = [r['t'] for r in rows if 'stopping' in r['sources'] and r['t'] > last_sample+.2]
    # (the last confirmation is up to stop_window_s after the last sample's receipt)
    end = last_sample+short.stop_window_s+short.stop_memory_s+.02
    assert kept and end-.1 <= max(kept) <= end
    assert not any('stopping' in r['sources'] for r in rows if r['t'] > end)
    # a clear sample ends it
    _, rows = wall_run_v2(config, cue=lambda x: SIDE, wall_x=12., clear_from=4., seconds=1.5)
    assert not any('stopping' in r['sources'] for r in rows if r['x'] > 4.+5.*(.055+.085)+1e-9)
    # version 1 has no memory: the samples stop, the bound ends
    _, rows = wall_run_v2(assist_v1(), cue=lambda x: SIDE, speed=3., wall_x=7., samples_until=3.5, seconds=2.2)
    assert not any(r['source'] == 'stopping' for r in rows if r['t'] > last_sample+.3)


def test_v2_wall_ahead_conditions():
    from haltere.liftoff.fast_race_cue import TtcClearanceGovernor
    pilot = bare(assist())
    for state in ('side', 'coast', 'search'):
        assert pilot._assist_wall_ahead(state, 0., 10.)
    assert not pilot._assist_wall_ahead('cue', 0., 10.)                   # no marker bearing yet: nothing off heading
    pilot.direction = np.array([np.cos(np.radians(60.)), np.sin(np.radians(60.)), 0.])
    assert pilot._assist_wall_ahead('cue', 0., 10.) and pilot._assist_wall_ahead('below', 0., 10.)
    assert not pilot._assist_wall_ahead('cue', np.radians(40.), 10.)     # 20 deg off
    pilot.turn_first_active = True
    assert pilot._assist_wall_ahead('cue', np.radians(40.), 10.)
    pilot.turn_first_active = False
    pilot.clearance = TtcClearanceGovernor()
    pilot.clearance.cap, pilot.clearance.standoff_until = 1., 11.
    assert not pilot._assist_wall_ahead('cue', np.radians(40.), 10.)     # the declared rule leaves the stand-off out
    assert bare(assist(wall_ahead_standoff=True))._assist_wall_ahead.__self__ is not None
    other = bare(assist(wall_ahead_standoff=True))
    other.clearance, other.direction = pilot.clearance, pilot.direction
    assert other._assist_wall_ahead('cue', np.radians(40.), 10.)
    launching = bare(assist())
    launching.launching = True
    assert not launching._assist_wall_ahead('side', 0., 10.)


def test_v2_no_approach_while_the_pilots_own_path_climbs():
    """Version 2: a surface looming while the pilot follows its checkpoint up gets no approach bound (the pilot's
    vertical request before the governor and the guard above approach_climb_max 0.3 m/s). Version 3 declares the
    exclusion out (3.5 m/s, the pilot's vertical_up): it also switched the approach off while the pilot climbed back to
    the ring height after a turn."""
    up = lambda x: cue_toward([22.-x, 0., 4.])                           # the ring well above: the pilot climbs
    _, rows = wall_run_v2(assist(approach_climb_max=.3, version=2), cue=up, wall_x=9., seconds=1.6)
    assert not any('approach' in r['sources'] for r in rows)
    _, rows = wall_run_v2(assist(), cue=up, wall_x=9., seconds=1.6)
    assert any('approach' in r['sources'] for r in rows)


def test_v2_no_governor_tracking_during_its_standoff():
    from haltere.liftoff.fast_race_cue import TtcClearanceGovernor
    for tracking, expected in ((False, False), (True, True)):
        pilot = bare(assist(standoff_tracking=tracking))
        pilot.clearance = TtcClearanceGovernor()
        pilot.clearance.standoff_until = 11.
        pilot.pilot_command = np.array([1.5, 0., 0.])
        pilot.clearance_braking = True
        sources = pilot._assist_sources('cue', .8, np.array([1., 0., 0.]), None, np.inf, np.zeros(3), 10.)
        assert any(name == 'governor' for name, _, _ in sources) is expected
        sources = pilot._assist_sources('cue', .8, np.array([1., 0., 0.]), None, np.inf, np.zeros(3), 12.)
        assert any(name == 'governor' for name, _, _ in sources)


def test_v2_approach_and_stopping_share_one_extra_reduction():
    """A wall-ahead condition that starts mid-approach (the arch passed, the next marker to the side) carries the cap
    tracking's extra reduction over instead of restarting it from zero."""
    pilot = bare(assist(slew=0., sag_gain=0., sag_lead_gain=0.))
    h = np.array([1., 0.])
    for _ in range(20):
        pilot._motor_assist(np.array([5., 0., 0.]), np.array([5., 0., 0.]), .01, 'cue', [('approach', h, 3.)])
    carried = pilot.assist_extra['stopping']
    assert carried > .5 and pilot.assist_extra['approach'] == 0.
    out = pilot._motor_assist(np.array([5., 0., 0.]), np.array([5., 0., 0.]), .01, 'side', [('stopping', h, 3.)])
    assert pilot.assist_extra['stopping'] >= carried and out[0] == pytest.approx(3.-pilot.assist_extra['stopping'])
    assert pilot.assist_plan == 3.


def test_v2_passthrough_scenario_geometry():
    from haltere.liftoff.motor_assist_eval import _wall_hit, passthrough_scenario
    sc = passthrough_scenario(turn_deg=90., surface_m=.5, see_through_m=1.5)
    (surface,) = sc['loom_walls']
    assert sc['walls'] == [] and sc['kind'] == 'passthrough' and sc['arch_ring'] == 0
    assert np.allclose(surface[0][1], 24.5) and np.allclose(surface[1][1], 24.5)       # 0.5 m beyond R1 (y 24)
    hit = _wall_hit(np.array([0., 20., .8]), np.array([0., 5., 0.]), np.array([0., 1.]), sc['loom_walls'])
    assert hit == (pytest.approx(4.5), pytest.approx(5.))
    assert np.allclose(sc['rings'][1], sc['rings'][0]+np.array([15., 0., 0.]))          # R2 after a right turn


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


# the full stack with motor assist version 1 (its kept declaration), computed with this function on `git archive m4b`
M4B_V1_ASSIST_DIGEST = 'e3dd0b2490eb9f9565f9f0b6d0017427ae267918c582a8c9844ab1b080372a43'


def test_version_1_rebuilt_from_its_kept_declaration_is_bit_identical_to_m4b():
    from haltere.liftoff.fast_race_cue import motor_assist_for_contract
    from haltere.liftoff.visual_brain import MOTOR_ASSIST_DECLARATION
    kept = json.loads(MOTOR_ASSIST_DECLARATION.with_name('motor_assist_v1.json').read_text(encoding='utf-8'))
    trace, pilot = stack_scenario(motor_assist=motor_assist_for_contract(kept, 'fast_velocity_brain_v1'))
    digest = hashlib.sha256(np.ascontiguousarray(trace.astype(np.float64)).tobytes()).hexdigest()
    assert pilot.motor_assist_summary()['seconds']['stopping'] > 0
    assert digest == M4B_V1_ASSIST_DIGEST


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
    assert len(row) == len(MOTOR_ASSIST_COLUMNS) == 8 and all(np.isnan(row[:5])) and row[5] == ''
    assert MOTOR_ASSIST_COLUMNS[6:] == ('assist_plan', 'assist_wall_ahead') and all(np.isnan(row[6:]))
    pilot, _ = run(motor_assist=assist(), velocity=(6., 0., -1.), cue=cue_toward([10., 12., 0.]), seconds=.5)
    values = motor_assist_row(pilot)
    assert all(np.isfinite(values[:5])) and isinstance(values[5], str) and values[7] in (0., 1.)
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
    # round 5 (arches) appends the stale-evidence columns after them; round 6 appends the marker-jump columns (gates),
    # the early-brake column (brake) and the sighted-descent columns (ground, tests/test_sighted_descent.py), each only
    # when its rule is declared
    assert ('*VERTICAL_COLUMNS,*COMMIT_COLUMNS,*view_columns,*assist_columns,*stale_columns,*marker_columns,'
            '*early_columns,*sighted_columns])') in source
    assert ('*(descent_view_row(controller.assistance)ifview_columnselse()),'
            '*(motor_assist_row(controller.assistance)ifassist_columnselse()),'
            '*(stale_row(controller.assistance)ifstale_columnselse()),'
            '*(marker_jump_row(controller.assistance)ifmarker_columnselse()),'
            '*(early_row(controller.assistance)ifearly_columnselse()),'
            '*(sighted_row(controller.assistance)ifsighted_columnselse())])') in source


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
V3_DECLARATION_SHA256 = '7c3b49e7bcc70ece18b00e68285c427ea430016fc52392122663205fd6be8772'
DECLARATION_SHA256 = '8954a798e127a5594fa4f761f57a238ab0456dd607263b107e17a871185fd5a6'           # version 4 (round 6)
V2_DECLARATION_SHA256 = 'f3f3502247d766b8b59529350746ca53e6c24a66bc818dc1f4494e65e2e36901'
V1_DECLARATION_SHA256 = 'eefb4a42613cabe5bd9c120825c42274ab72f8f7b970750aadb26334d3781d02'
GATES_SHA256 = 'f75a45e4e4f5ccdfda2dcf62cef81ea4ef07e086a79b38aa12e1d959527da5fc'
V2_GATES_SHA256 = '8d7ce785478697b490b370ab09b2cc359be90231a7a43dde249f306ecac47264'
V1_GATES_SHA256 = '9abfb80d9ff95a65d0b6400bef0133966aa0de83644c4f55980d1ef34cb7e3c5'


def test_frozen_declaration_assigns_the_rule_to_the_brain_contract_only(tmp_path):
    from haltere.liftoff.fast_race_cue import MotorAssistConfig, motor_assist_for_contract
    from haltere.liftoff.visual_brain import MOTOR_ASSIST_DECLARATION, load_motor_assist
    declaration, digest = load_motor_assist()
    assert digest == DECLARATION_SHA256 and declaration['version'] == 4
    # the declared brain entry is version 3's rule without the approach, with the tracking floor and the ceiling share;
    # the fast PD (and any other contract) has none
    assert motor_assist_for_contract(declaration, 'fast_velocity_brain_v1') == MotorAssistConfig(
        cap_sources=('request', 'governor', 'turn_first', 'stopping'), track_floor=2.5, ceiling_share=True, version=4)
    assert motor_assist_for_contract(declaration, 'fast_velocity_pd_v1') is None
    assert 'fast_velocity_pd_v1' in declaration['contracts']
    assert [v['sha256'] for v in declaration['previous_versions']] == [V1_DECLARATION_SHA256, V2_DECLARATION_SHA256,
                                                                      V3_DECLARATION_SHA256]
    # an edited copy or another version is refused; the runner refuses the kept versions 1, 2 and 3
    edited = json.loads(MOTOR_ASSIST_DECLARATION.read_text(encoding='utf-8'))
    edited['contracts']['fast_velocity_brain_v1']['cap_gain'] = 3.
    path = tmp_path/'edited.json'
    path.write_text(json.dumps(edited), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen'):
        load_motor_assist(path)
    from haltere.train.brake_gates import gates_sha256
    other = dict(declaration, version=5)
    other['sha256'] = gates_sha256(other)
    path.write_text(json.dumps(other), encoding='utf-8')
    with pytest.raises(ValueError, match='version'):
        load_motor_assist(path)
    for version, digest in ((1, V1_DECLARATION_SHA256), (2, V2_DECLARATION_SHA256), (3, V3_DECLARATION_SHA256)):
        kept = MOTOR_ASSIST_DECLARATION.with_name(f'motor_assist_v{version}.json')
        with pytest.raises(ValueError, match=f'version {version}'):
            load_motor_assist(kept)
        old = json.loads(kept.read_text(encoding='utf-8'))
        assert old['version'] == version and old['sha256'] == digest == gates_sha256(old)
        assert motor_assist_for_contract(old, 'fast_velocity_brain_v1').version == version
    v2 = motor_assist_for_contract(json.loads(MOTOR_ASSIST_DECLARATION.with_name('motor_assist_v2.json').read_text(
        encoding='utf-8')), 'fast_velocity_brain_v1')
    assert v2 == MotorAssistConfig(approach_climb_max=.3, version=2)
    v3 = motor_assist_for_contract(json.loads(MOTOR_ASSIST_DECLARATION.with_name('motor_assist_v3.json').read_text(
        encoding='utf-8')), 'fast_velocity_brain_v1')
    assert v3 == MotorAssistConfig()                  # the dataclass defaults are version 3's


def test_frozen_gates_name_the_frozen_declaration():
    from haltere.liftoff.motor_assist_gates import load_gates
    gates, digest, declaration = load_gates()
    assert digest == GATES_SHA256 and gates['version'] == 3
    # (version 3's gates score the kept version-3 declaration)
    assert gates['motor_assist']['sha256'] == V3_DECLARATION_SHA256 == declaration['sha256']
    assert set(gates['controllers']) == {'brain11cw13', 'brain09b', 'brain10b'}
    # the held-out hairpin set, seeds and hill courses differ from every set either earlier version was designed or
    # scored on
    h = gates['hairpin']
    assert h['sim_seed'] not in (17, 23, 31) and gates['passthrough']['sim_seed'] not in (17, 23, 31)
    assert not set(h['set']['turn_deg']) & {20., 60., 90., 45., 75., 105., 35., 85., 115.}
    assert not set(h['set']['arch_m']) & {7., 11., 8., 13., 9., 12.}
    assert not set(h['set']['wall_m']) & {2.1, 2.6, 2.3, 3., 2.4, 2.8}
    assert gates['course_sets']['hill'] == ['hill:6400-6411'] and gates['gates']['warning']['mode'] == 'final'
    r = gates['replays']
    assert r['minus_heldout'] == [] and r['quiet_heldout'] == [] and len(r['minus_development']) == 16
    # the kept gates still load, each with the declaration it scored
    for path, digest, declared in ((REPO_GATES_V1, V1_GATES_SHA256, V1_DECLARATION_SHA256),
                                   (REPO_GATES_V2, V2_GATES_SHA256, V2_DECLARATION_SHA256)):
        old, old_digest, old_declaration = load_gates(path)
        assert old_digest == digest and old_declaration['sha256'] == declared
    assert [v['sha256'] for v in gates['previous_versions']] == [V1_GATES_SHA256, V2_GATES_SHA256]


def test_gate_replay_metrics_on_a_synthetic_log():
    """motor_assist_gates.replay_metrics / warning_s: planned-crawl episodes before x, the per-tick slew rates and the
    source activity, on a hand-made replay."""
    from haltere.liftoff.motor_assist_gates import replay_metrics, warning_s
    n = 300
    t = np.arange(n)*.01
    a = dict(t=t, now=t+100., x=np.linspace(0., 90., n), assist_pilot_vx=np.full(n, 5.), assist_pilot_vy=np.zeros(n),
             assist_pilot_vz=np.zeros(n), cvx=np.full(n, 5.), cvy=np.zeros(n), cvz=np.zeros(n),
             assist_plan=np.full(n, np.nan), assist_source=np.array(['']*n, dtype='<U12'), assist_wall_ahead=np.zeros(n))
    a['assist_plan'][50:70] = 1.5                        # a planned crawl of 0.19 s at x 15-21 m
    a['assist_plan'][260:280] = 0.                       # a stop beyond x = 73 m (the hairpin): not counted
    a['cvx'][100:120] = 5.-np.minimum(np.arange(20)*.15, 2.)
    a['assist_source'][100:120] = 'approach'
    a['cvx'][280:] = 3.5
    m = replay_metrics(a)
    assert len(m['planned_crawl_episodes']) == 1 and m['planned_crawl_episodes'][0]['min_plan'] == 1.5
    assert m['rate_h'] == pytest.approx(200.) and m['rate_z'] == 0. and m['zero_dt_change'] == 0.   # the 2 m/s release
    assert m['active_s_per_min']['approach'] == pytest.approx(.2/(t[-1]/60), abs=1e-3)
    assert m['stop_model_removed_share'] > 0 and m['removed_share'] >= m['stop_model_removed_share']
    assert warning_s(a) == pytest.approx(t[-1]-t[107]) and warning_s(a, window_s=1.) == pytest.approx(t[-1]-t[280])
    # gates v3: the final continuous cut that reaches the end (the ramp at 100-120 is a separate, earlier cut)
    assert warning_s(a, mode='final') == pytest.approx(t[-1]-t[280])
    a['cvx'][:] = 5.
    assert warning_s(a, mode='final') is None


REPO_GATES_V1 = Path(__file__).resolve().parents[1]/'configs'/'pilot'/'motor_assist_gates_v1.json'
REPO_GATES_V2 = Path(__file__).resolve().parents[1]/'configs'/'pilot'/'motor_assist_gates_v2.json'


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
