"""Offline open-loop replay harness (haltere.obstacles.pilot_replay): variants, planner streams and the V gates.

Pins that the variants build the declared stack (planner shadow/on leave the gap aim unapplied), that planner samples
are published at their pub_time once, that non-perception streams are labelled plumbing, that the identity check is
bitwise, and the V1-V3 scoring rules on small synthetic arrays. The replays of the logged flights live outside the
repository (their results are in docs/free_space_pilot.md); nothing here is flight evidence.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from haltere.liftoff.camera_process import PLAN_FIELDS, PLAN_KINDS, plan_values
from haltere.obstacles import pilot_replay as pr
from tests.test_corridor_aim import left, plan
from tests.test_visual_assistance import SENSOR

TREE = Path(__file__).resolve().parents[1]


def sidecar(contract='fast_velocity_pd_v1', stack=None):
    return dict(motor_controller=dict(contract=contract), obstacle_stack=None if stack is None else dict(mode=stack),
                pilot_assistance=dict(nominal_speed_mps=6., trained_motor_reference_mps=6.), gate_sensor=SENSOR)


def test_variants_build_the_declared_components():
    pilot, info = pr.build(sidecar(), TREE, 'none')
    assert pilot.gap_aim is None and pilot.lag_turn is None and pilot.corridor is None and pilot.turn_first is None
    pilot, info = pr.build(sidecar(stack='on'), TREE, 'flown')
    assert pilot.gap_aim is not None and pilot.gap_apply and pilot.lag_turn is not None and pilot.turn_first is None
    pilot, _ = pr.build(sidecar(), TREE, 'flown')                        # flown without the stack: no components
    assert pilot.gap_aim is None
    pilot, info = pr.build(sidecar(), TREE, 'on', planner='on')
    assert pilot.gap_aim is not None and not pilot.gap_apply and pilot.turn_first is not None and pilot.wall_apply
    assert pilot.corridor is not None and pilot.planner_apply and info['planner'] == 'on'
    pilot, _ = pr.build(sidecar(), TREE, 'shadow', planner='on')         # the stack in shadow applies nothing
    assert not (pilot.gap_apply or pilot.lag_turn_apply or pilot.wall_apply or pilot.planner_apply)
    pilot, _ = pr.build(sidecar(), TREE, 'on', gap_apply='off')
    assert pilot.corridor is None and not pilot.gap_apply
    with pytest.raises(ValueError, match='obstacle stack'):
        pr.build(sidecar(), TREE, 'none', planner='shadow')


def write_stream(path, samples, source='synthetic-test'):
    arrays = {k: np.array([v for v in values], float) for k, values in
              zip(PLAN_FIELDS, zip(*[plan_values(s) for s in samples]))}
    arrays['pub_time'] = arrays['time']+pr.PLAN_OFFLINE_AGE
    arrays['source'] = np.array(source)
    np.savez(path, **arrays)


def test_plan_stream_publishes_each_sample_at_its_pub_time(tmp_path):
    path = tmp_path/'s.npz'
    write_stream(path, [dict(left(1.), seq=1.), dict(plan(1.06, 'no_scale', valid=False), seq=2.),
                        dict(left(1.12, az=6.), seq=3.)])
    stream = pr.PlanStream(path)
    assert stream.source == 'synthetic-test'
    assert stream.at(1.05) is None
    first = stream.at(1.10)
    assert first['seq'] == 1. and first['kind'] == 'shift' and first['valid'] and first['l_az'] == 8.
    assert stream.at(1.15)['seq'] == 1.
    second = stream.at(1.161)
    assert second['kind'] == 'no_scale' and not second['valid'] and second['l_az'] is None
    assert stream.at(5.)['l_az'] == 6.
    assert set(PLAN_FIELDS) <= set(first) and PLAN_KINDS.index('shift') == 8


def test_identity_is_bitwise_with_nan_equal():
    a = dict(cvx=np.array([1., np.nan]), cvy=np.zeros(2), cvz=np.zeros(2), yaw_cmd=np.zeros(2),
             state=np.array(['cue', 'cue']), cap=np.array([np.nan, 2.]), climb=np.zeros(2))
    b = {k: v.copy() for k, v in a.items()}
    assert pr.identical(a, b)[0]
    b['cvx'] = np.array([1.+1e-15, np.nan])
    ok, detail = pr.identical(a, b)
    assert not ok and detail['cvx'] is False and detail['cap'] is True


def gates():
    obj = json.loads((TREE/'configs'/'obstacles'/'free_space_gates.json').read_text(encoding='utf-8'))
    return obj['gates']


def test_the_gates_declaration_is_frozen_and_loadable():
    obj, digest = pr.load_gates()
    assert obj['version'] == 1 and digest == obj['sha256']
    g = obj['gates']
    assert set(g) == {'S1', 'S2', 'S3', 'A1', 'B1', 'T1', 'Q1', 'L1', 'V1', 'V2', 'V3', 'R1', 'I1'}
    assert g['V1']['max_vz'] == 1. and g['V1']['max_vz_low_ceiling'] == .3 and g['V2']['displaced_fraction'] == .9
    assert g['Q1']['max_episodes_per_min'] == 6. and g['T1']['min_lead_s'] == .95


def arrays(n=400, dt=.01, **kw):
    t = np.arange(n)*dt
    base = dict(t=t, now=t+100., x=np.zeros(n), y=np.zeros(n), z=np.full(n, 1.), vx=np.full(n, 6.), vy=np.zeros(n),
                vz=np.zeros(n), cvx=np.full(n, 6.), cvy=np.zeros(n), cvz=np.zeros(n), yaw_cmd=np.zeros(n),
                state=np.array(['cue']*n), log_climb=np.zeros(n), plan_time=t+99.9, plan_h_ceil=np.full(n, 1.2),
                plan_h_floor=np.full(n, 1.), plan_d_h=np.full(n, 9.), plan_cls_confirmed=np.zeros(n),
                plan_intended_el=np.zeros(n), plan_applied_az=np.zeros(n), plan_rise_confirmed=np.zeros(n),
                plan_vertical_applied=np.ones(n))
    base.update(kw)
    return base


def test_v1_scores_the_window_from_the_climb_onset_to_the_impact():
    g = gates()['V1']
    info = dict(stop_reason='Impact detected from flight motion')
    a = arrays(log_climb=np.r_[np.zeros(200), np.full(200, 3.)])
    result = pr.score_v1(a, info, g)
    assert result['scored'] and result['window'][0] == pytest.approx(1.5) and result['passed']
    a['cvz'][350] = 1.2                                                   # a hard climb under the ceiling
    assert not pr.score_v1(a, info, g)['passed']
    a = arrays(log_climb=np.r_[np.zeros(200), np.full(200, 3.)], plan_h_ceil=np.full(400, .7))
    a['cvz'][300] = .5
    result = pr.score_v1(a, info, g)
    assert result['low_ceiling_ticks'] > 0 and not result['passed']
    a = arrays(log_climb=np.r_[np.zeros(200), np.full(200, 3.)], plan_cls_confirmed=np.full(400, 2.),
               plan_intended_el=np.full(400, 9.))
    assert pr.score_v1(a, info, g)['v_class_into_ceiling_ticks'] > 0
    assert not pr.score_v1(arrays(), dict(stop_reason='duration'), g)['scored']


def test_v2_and_v3_rules():
    g = gates()
    a = arrays(z=np.linspace(.7, .3, 400), vz=np.full(400, -.4), log_climb=np.r_[np.zeros(300), np.full(100, 3.5)])
    result = pr.score_v2(a, {}, g['V2'], 'last_3s')
    assert result['passed'] and result['displaced_fraction'] == 1.
    a['cvz'][150] = .8                                                    # a climb from a flat floor above 0.5 m
    assert not pr.score_v2(a, {}, g['V2'], 'last_3s')['passed']
    a = arrays(z=np.linspace(.7, .3, 400), vz=np.full(400, -.4))
    a['cvz'][-1] = -.2                                                    # still descending below 0.5 m
    assert not pr.score_v2(a, {}, g['V2'], [0., 4.])['passed']
    v3 = g['V3']
    a = arrays(log_climb=np.r_[np.zeros(100), np.full(300, 2.)], cvz=np.r_[np.zeros(100), np.full(300, 1.5)])
    result = pr.score_v3(a, dict(stop_reason='Impact detected from flight motion'), v3)
    assert result['passed'] and result['answered_fraction'] == pytest.approx(1., abs=.01)
    a['cvz'][-50:] = -.1
    assert not pr.score_v3(a, dict(stop_reason='Impact detected from flight motion'), v3)['passed']
