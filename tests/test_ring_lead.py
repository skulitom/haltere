"""FastRaceCue near-ring lead (`RingLeadConfig`, runner flag --ring-lead; configs/pilot/ring_lead.json): off by default.

Round 7 (in-gate turns): in both development crashes (minus-brain11cw13-r6-01 at the low floor arch,
straw-brain11cw13-r5-noassist-01 at FAT SHARK) the ring marker never switched before the impact; the brain approached the
gate off its centre and the pursuit of the swinging ring bearing fell behind. With the rule, while the in-view ring-centre
line-of-sight rate is at least rate_deg_s, the course falls further behind the ring bearing and no gap shift is applied,
the lag-aware turn leads the ring bearing as in its switch window. These tests pin: the default pilot is unchanged and
shadow changes nothing, the rate measurement and its restart, the condition (fresh, rate, the pursuit falling behind,
yield to the gap aim), the lead it applies, the declared file and gates, the runner and harness wiring, and the gate
scorer's hindsight triangulation. None of it is flight evidence.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (RING_LEAD_MODES, RING_LEAD_VERSION, FastRaceCue, LagTurnConfig,
                                           RingLeadConfig, ring_lead_for_contract)
from tests.test_fast_race_cue import cue_toward, drive
from tests.test_visual_assistance import SENSOR

RL = RingLeadConfig()
LT = LagTurnConfig()
TREE = Path(__file__).resolve().parents[1]


def azimuth_cue(degrees, distance=20.):
    """In-view cue for a level ring at a world azimuth (drive() keeps yaw 0; positive = left)."""
    a = np.radians(degrees)
    return cue_toward([distance*np.cos(a), distance*np.sin(a), 0.])


def swing(rate_deg_s, start=0., begin=.3):
    """A ring bearing that is still for `begin` s and then turns at rate_deg_s (drive()'s clock starts at 10 s)."""
    def cue(now):
        t = now-10.
        return azimuth_cue(start+rate_deg_s*max(0., t-begin))
    return cue


def run(ring_lead=None, apply=True, cue=None, steps=120, lag_turn=LT, plant='static'):
    """The pilot flown past a ring; plant 'static' keeps the flown course at 0 deg (a motor that has not yet followed:
    the pursuit falls behind a swinging bearing), 'perfect' reports each request as the measured motion."""
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., reference_speed=6., lag_turn=lag_turn,
                        **({} if ring_lead is None else dict(ring_lead=ring_lead, ring_lead_apply=apply)))
    rows = drive(pilot, pilot.pose_history, cue, steps, velocity=(6., 0., 0.), plant=plant)
    return pilot, rows


def test_default_and_shadow_are_unchanged_and_on_leads():
    cue = swing(20.)
    _, plain = run(cue=cue)
    pilot, shadow = run(ring_lead=RL, apply=False, cue=cue)
    assert [r[1] for r in plain] == [r[1] for r in shadow]
    assert all(np.array_equal(x[2], y[2]) for x, y in zip(plain, shadow))
    # shadow measured the swing and logs where it would act
    assert pilot.ring_lead_counts['episodes'] >= 1 and pilot.ring_lead_time > .3
    assert pilot.metadata()['ring_lead']['applied'] is False
    assert 'ring_lead' not in FastRaceCue(SENSOR, CameraPoseHistory(), 6., lag_turn=LT).metadata()
    on, rows = run(ring_lead=RL, cue=cue)
    # applied: the request leads the swinging bearing (more left than without the rule)
    heading = lambda r: float(np.degrees(np.arctan2(r[2][1], r[2][0])))
    later = [(heading(a), heading(b)) for a, b in zip(plain, rows) if a[0] >= 10.8]
    assert all(b >= a-1e-9 for a, b in later) and max(b-a for a, b in later) > 1.
    assert on.metadata()['ring_lead']['applied'] is True and on.metadata()['ring_lead']['seconds'] > .3


def test_a_still_bearing_never_acts():
    pilot, rows = run(ring_lead=RL, cue=azimuth_cue(5.))
    _, plain = run(cue=azimuth_cue(5.))
    assert pilot.ring_lead_counts['episodes'] == 0 and pilot.ring_lead_time == 0.
    assert pilot.ring_los_rate < 1.
    assert all(np.array_equal(x[2], y[2]) for x, y in zip(plain, rows))


def test_the_rate_is_measured_over_the_span_and_needs_three_readings():
    pilot, _ = run(ring_lead=RL, apply=False, cue=swing(12.), steps=100)
    assert abs(pilot.ring_los_rate-12.) < 1.
    assert pilot.ring_lead_log()['ring_lead'] == 1. and abs(pilot.ring_lead_log()['ring_los_rate']-12.) < 1.
    slow, _ = run(ring_lead=RL, apply=False, cue=swing(6.), steps=100)
    assert abs(slow.ring_los_rate-6.) < 1. and slow.ring_lead_counts['episodes'] == 0
    # two readings are no measurement
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., lag_turn=LT, ring_lead=RL)
    centre = np.array([1., 0., 0.])
    pilot._ring_lead_reading(centre, False, 1.)
    pilot._ring_lead_reading(np.array([np.cos(.1), np.sin(.1), 0.]), False, 1.1)
    assert pilot.ring_los_at is None and np.isnan(pilot.ring_los_rate)


def test_a_jump_restarts_and_edge_clamps_are_not_readings():
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., lag_turn=LT, ring_lead=RL)
    ray = lambda deg: np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.])
    for k, deg in enumerate((0., 1., 2.)):
        pilot._ring_lead_reading(ray(deg), False, 1.+.05*k)
    assert abs(pilot.ring_los_rate-20.) < 1e-6
    pilot._ring_lead_reading(ray(40.), False, 1.15)           # a checkpoint switch
    assert pilot.ring_lead_counts['restarts'] == 1 and len(pilot.ring_lead_readings) == 1
    pilot._ring_lead_reading(ray(60.), True, 1.2)             # clamped: ignored
    assert len(pilot.ring_lead_readings) == 1 and pilot.ring_lead_counts['readings'] == 4


def condition_pilot(config=RL):
    """A pilot whose ring lead measured 9 deg/s at 10 s: bearing 10 deg left of a course of 0 deg at the first reading,
    the latest reading's bearing 12.7 deg."""
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., lag_turn=LT, ring_lead=config)
    pilot.launching = False
    pilot.ring_lead_readings = [[9.7, np.radians(10.), 0.], [10., np.radians(12.7), 0.]]
    pilot.ring_los_rate, pilot.ring_los_signed, pilot.ring_los_at = 9., 9., 10.
    pilot.ring_los_az = np.radians(12.7)
    return pilot


def test_the_condition_is_fresh_fast_behind_and_yields_to_the_gap_aim():
    east, v = np.array([6., 0., 0.]), lambda deg: 6.*np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.])
    pilot = condition_pilot()
    assert pilot._ring_lead_condition(10.2, east)
    assert not pilot._ring_lead_condition(10.3, east)                  # older than fresh_s
    assert not pilot._ring_lead_condition(10.1, v(4.))                 # the course catches up: 8.7 < 10 deg behind
    assert pilot._ring_lead_condition(10.1, v(2.))                     # 10.7 >= 10 deg: still falling behind
    assert not pilot._ring_lead_condition(10.1, v(20.))                # the course passed the bearing (other side)
    assert not pilot._ring_lead_condition(10.1, np.array([.5, 0., 0.]))    # too slow for a course
    pilot.ring_lead_readings[0][2] = float('nan')                      # no course at the first reading
    assert not pilot._ring_lead_condition(10.1, east)
    pilot = condition_pilot()
    pilot.ring_los_rate = 7.
    assert not pilot._ring_lead_condition(10.1, east)
    pilot.ring_los_rate, pilot.gap_offset_deg = 9., 3.
    assert not pilot._ring_lead_condition(10.1, east)
    no_yield = condition_pilot(RingLeadConfig(yield_to_gap=False))
    no_yield.gap_offset_deg = 3.
    assert no_yield._ring_lead_condition(10.1, east)
    pilot.gap_offset_deg, pilot.launching = 0., True
    assert not pilot._ring_lead_condition(10.1, east)


def test_a_pursuit_that_keeps_up_is_led_far_less():
    # a motor that tracks its request (the course follows the pursuit): the course falls behind only while the swing
    # starts, so the rule acts about half as long and changes the request about a third as much as for a motor that has
    # not followed (the course held at 0 deg)
    tracking, rows = run(ring_lead=RL, cue=swing(12.), plant='perfect', steps=100)
    _, plain = run(cue=swing(12.), plant='perfect', steps=100)
    static, srows = run(ring_lead=RL, cue=swing(12.), plant='static', steps=100)
    _, splain = run(cue=swing(12.), plant='static', steps=100)
    assert abs(tracking.ring_los_rate-12.) < 1.5
    assert tracking.ring_lead_time < .6*static.ring_lead_time
    change = lambda a, b: max(float(np.hypot(*(x[2][:2]-y[2][:2]))) for x, y in zip(a, b))
    assert change(plain, rows) < .4*change(splain, srows)


@pytest.mark.parametrize('kw', [dict(rate_deg_s=0.), dict(span_s=0.), dict(span_s=3.), dict(min_span_s=.5),
                                dict(min_readings=1), dict(min_readings=2.5), dict(min_readings=True),
                                dict(jump_deg=90.), dict(fresh_s=2.), dict(yield_to_gap=1), dict(rate_deg_s=float('nan'))])
def test_invalid_parameters_are_refused(kw):
    with pytest.raises(ValueError):
        RingLeadConfig(**kw)


def test_the_pilot_needs_the_lag_turn_and_the_config_type():
    with pytest.raises(ValueError, match='lag-aware turn'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., ring_lead=RL)
    with pytest.raises(ValueError, match='RingLeadConfig'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., lag_turn=LT, ring_lead=dict(rate_deg_s=8.))


def test_the_declaration_is_frozen_per_contract(tmp_path):
    from haltere.liftoff import visual_brain as vb
    declaration, digest = vb.load_ring_lead()
    assert declaration['version'] == RING_LEAD_VERSION == 1 and digest == declaration['sha256']
    assert asdict(ring_lead_for_contract(declaration, 'fast_velocity_brain_v1')) == asdict(RL)
    assert ring_lead_for_contract(declaration, 'fast_velocity_pd_v1') is None
    edited = dict(declaration, contracts=dict(declaration['contracts'], fast_velocity_pd_v1=asdict(RL)))
    path = tmp_path/'edited.json'
    path.write_text(json.dumps(edited), encoding='utf-8')
    with pytest.raises(ValueError, match='changed after the freeze'):
        vb.load_ring_lead(path)
    other = dict(declaration, version=2)
    other['sha256'] = vb.lag_turn_declaration_sha256(other)
    path.write_text(json.dumps(other), encoding='utf-8')
    with pytest.raises(ValueError, match='version 2'):
        vb.load_ring_lead(path)
    with pytest.raises(ValueError, match='version'):
        ring_lead_for_contract(dict(declaration, version=2), 'fast_velocity_brain_v1')


def test_the_gates_are_frozen_and_name_the_declaration():
    from haltere.obstacles.ring_lead_gates import load_gates
    gates, digest = load_gates()
    assert gates['declarations']['ring_lead']['version'] == 1 and gates['sha256'] == digest
    assert set(gates['development_logs']) <= set(gates['flights'])


def test_runner_flag_and_columns():
    from haltere.liftoff import visual_brain as vb
    stack = dict(obstacle_stack='on')
    assert vb.resolve_ring_lead(argparse.Namespace(**stack)) == (None, None)
    assert vb.resolve_ring_lead(argparse.Namespace(ring_lead='off', **stack)) == (None, None)
    assert vb.resolve_ring_lead(argparse.Namespace(ring_lead='on', **stack)) == (str(vb.RING_LEAD_DECLARATION), 'on')
    assert vb.resolve_ring_lead(argparse.Namespace(ring_lead='shadow', **stack))[1] == 'shadow'
    with pytest.raises(ValueError, match='obstacle-stack'):
        vb.resolve_ring_lead(argparse.Namespace(ring_lead='on', obstacle_stack=None))
    with pytest.raises(ValueError, match='obstacle-stack'):
        vb.resolve_ring_lead(argparse.Namespace(ring_lead='on', obstacle_stack='on', lag_turn='off'))
    assert RING_LEAD_MODES == ('on', 'shadow')
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., lag_turn=LT)
    assert all(np.isnan(v) for v in vb.ring_lead_row(pilot))
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., lag_turn=LT, ring_lead=RL)
    row = vb.ring_lead_row(pilot)
    assert row[0] == 0. and np.isnan(row[1])
    assert vb.RING_LEAD_COLUMNS == tuple(pilot.ring_lead_log())
    # the runner appends the columns last, after the sighted-descent columns
    source = (TREE/'haltere'/'liftoff'/'visual_brain.py').read_text(encoding='utf-8')
    assert '*sighted_columns,\n                                 *ring_lead_columns])' in source
    parser_help = source[source.index("p.add_argument('--ring-lead'"):]
    assert "choices=['on','off','shadow']" in parser_help[:200]


def test_deployed_pilot_adds_the_rule_only_when_asked():
    from haltere.train.deployed_pilot import deployed_pilot_kwargs
    kw, record = deployed_pilot_kwargs('fast_velocity_brain_v1')
    assert 'ring_lead' not in kw and 'ring_lead' not in record
    kw, record = deployed_pilot_kwargs('fast_velocity_brain_v1', ring_lead='on')
    assert kw['ring_lead'] == RL and kw['ring_lead_apply'] is True and record['ring_lead']['mode'] == 'on'
    kw, record = deployed_pilot_kwargs('fast_velocity_pd_v1', ring_lead='on')
    assert 'ring_lead' not in kw and record['ring_lead']['version'] == 1
    kw, _ = deployed_pilot_kwargs('fast_velocity_brain_v1', ring_lead='shadow')
    assert kw['ring_lead_apply'] is False
    with pytest.raises(ValueError, match='ring-lead mode'):
        deployed_pilot_kwargs('fast_velocity_brain_v1', ring_lead='maybe')
    with pytest.raises(ValueError, match='ring-lead mode'):
        deployed_pilot_kwargs('fast_velocity_brain_v1', stack=False, ring_lead='on')


def test_replay_harness_adds_it_to_stack_variants_only():
    from haltere.obstacles.vertical_replay import build
    side = dict(motor_controller=dict(contract='fast_velocity_brain_v1'), obstacle_stack=dict(mode='on'),
                pilot_assistance=dict(nominal_speed_mps=6.), gate_sensor=SENSOR)
    declaration = json.loads((TREE/'configs'/'pilot'/'ring_lead.json').read_text(encoding='utf-8'))
    pilot, info = build(side, str(TREE), 'on', ring_lead=declaration)
    assert pilot.ring_lead == RL and pilot.ring_lead_apply and info['ring_lead'] == 'applied'
    pilot, info = build(side, str(TREE), 'on', ring_lead=declaration, ring_lead_apply=False)
    assert not pilot.ring_lead_apply and info['ring_lead'] == 'shadow'
    pilot, info = build(side, str(TREE), 'shadow', ring_lead=declaration)
    assert not pilot.ring_lead_apply and info['ring_lead'] == 'shadow'
    pilot, info = build(side, str(TREE), 'none', ring_lead=declaration)
    assert pilot.ring_lead is None and info['ring_lead'] == 'none'
    pd = dict(side, motor_controller=dict(contract='fast_velocity_pd_v1'))
    pilot, info = build(pd, str(TREE), 'on', ring_lead=declaration)
    assert pilot.ring_lead is None and info['ring_lead'] == 'none'


def test_harness_oblique_gate():
    from haltere.liftoff import motor_assist_eval as mae
    plain = mae.gate_scenario(turn_deg=90., half_w=1.5)
    assert 'oblique_deg' not in plain['params']
    np.testing.assert_allclose(plain['posts'][2], [15., 24.-1.5], atol=1e-9)
    oblique = mae.gate_scenario(turn_deg=90., half_w=1.5, oblique_deg=30.)
    assert oblique['params']['oblique_deg'] == 30.
    np.testing.assert_allclose(oblique['posts'][:2], plain['posts'][:2])
    r2 = np.array([15., 24.])
    a, b = oblique['posts'][2]-r2, oblique['posts'][3]-r2
    np.testing.assert_allclose(np.hypot(*a), 1.5)
    np.testing.assert_allclose(a, -b, atol=1e-9)
    # the leg axis is rotated 30 deg (clockwise from above) from across the R1-R2 leg, which runs east: from south
    # toward west
    np.testing.assert_allclose(a/1.5, [-np.sin(np.radians(30.)), -np.cos(np.radians(30.))], atol=1e-9)


def test_gate_scorer_triangulation_and_segments():
    from haltere.obstacles.ring_lead_gates import ring_fit, ring_segments, triangulate
    ring = np.array([10., 2., 1.])
    t = np.arange(20)*.05
    p = np.stack([t*5., np.zeros_like(t), np.ones_like(t)], 1)          # flying along x past a ring 2 m to the left
    d = (ring-p)/np.linalg.norm(ring-p, axis=1, keepdims=True)
    x, eig, rms = triangulate(p, d)
    np.testing.assert_allclose(x, ring, atol=1e-6)
    assert rms < 1e-9 and eig > .01
    # a second ring 40 deg away starts a new segment
    far = np.array([30., -30., 1.])
    p2 = np.stack([5.+t*5., np.zeros_like(t), np.ones_like(t)], 1)
    d2 = (far-p2)/np.linalg.norm(far-p2, axis=1, keepdims=True)
    segs = ring_segments(np.r_[t, 1.+t], np.r_[d, d2], dict(jump_deg=15., gap_s=.6))
    assert segs == [(0, 20), (20, 40)]
    spec = dict(min_rays=5, min_eig=.001, max_rms_m=.2)
    row = ring_fit(t, p, d, np.arange(20), spec)
    assert row['valid'] and np.allclose(row['centre'], ring, atol=1e-3)
    mixed = ring_fit(np.r_[t, 1.+t], np.r_[p, p2], np.r_[d, d2], np.arange(40), spec)
    assert not mixed['valid']
    assert not ring_fit(t, p, d, np.arange(3), spec)['valid']
