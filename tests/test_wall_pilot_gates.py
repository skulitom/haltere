"""Turn-first version-4 gate scoring (haltere/obstacles/wall_pilot_gates.py) on synthetic replay arrays, the frozen
gates declaration, and the replay harness's per-contract wall-pilot build. The replays of the live logs are run
outside the test suite; none of this is flight evidence."""
import json
from pathlib import Path

import numpy as np
import pytest

from haltere.obstacles import vertical_replay as vr
from haltere.obstacles import wall_pilot_gates as wg
from tests.test_visual_assistance import SENSOR

TREE = Path(__file__).resolve().parents[1]
WALL = dict(normal=[1., 0.], plane_offset_m=10.)


def arrays(n=300, dt=.01, **columns):
    t = np.arange(n)*dt
    base = dict(t=t, cvx=np.zeros(n), cvy=np.zeros(n), vx=np.zeros(n), vy=np.zeros(n), x=np.zeros(n),
                y=np.zeros(n), turn_first=np.zeros(n), side_guard=np.zeros(n), log_cvx=np.zeros(n),
                log_cvy=np.zeros(n))
    base.update(columns)
    return base


def test_gates_declaration_is_frozen_and_names_this_trees_wall_pilot(tmp_path):
    gates, digest = wg.load_gates()
    assert gates['version'] == 1 and gates['frozen'] is True and digest == gates['sha256']
    from haltere.liftoff.visual_brain import WALL_PILOT_DECLARATION, load_wall_pilot
    declaration, wall_digest = load_wall_pilot(WALL_PILOT_DECLARATION)
    assert gates['wall_pilot'] == dict(file='configs/obstacles/wall_pilot.json', version=4, sha256=wall_digest)
    g = gates['gates']
    assert g['W_B09']['lead_s'] == .5 and g['W_B09']['max_toward_mps'] == .1 and g['W_B09']['window_s'] == 1.5
    assert g['W_B08']['lead_s'] == .5 and g['W_B08']['max_toward_mps'] is None
    assert g['W_PD']['max_delay_s'] == 1. and g['W_PD']['max_engaged_s'] == 1.
    assert g['W_PD']['max_extra_toward_mps'] == .05 and g['W_V3case']['max_toward_mps'] == .1
    assert set(g['W_Quiet']['straw']) == {'straw-brain08-04', 'straw-brain08-06'}
    assert gates['baseline_tree']['commit'].startswith('935cfdb4b6d7')
    edited = dict(gates, gates=dict(g, W_B09=dict(g['W_B09'], lead_s=.3)))
    path = tmp_path/'gates.json'
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match='frozen'):
        wg.load_gates(path)
    other = dict(declaration, sha256='0'*64)
    wall_path = tmp_path/'wall.json'
    wall_path.write_text(json.dumps(other))
    with pytest.raises(ValueError, match='not the declaration'):
        wg.load_gates(wg.GATES_PATH, wall_path)


def test_lead_gate_needs_an_early_engagement_and_no_request_toward_the_wall():
    gate = dict(window_s=1.5, lead_s=.5, settle_s=.25, max_toward_mps=.1)
    engaged = np.zeros(300)
    engaged[200:] = 1.                                          # engaged at 2.00 s, the log ends at 2.99 s
    toward = np.full(300, 2.)
    toward[220:] = 0.                                           # the request toward the wall gone 0.2 s later
    result = wg.score_lead(arrays(turn_first=engaged, cvx=toward), gate, WALL)
    assert result['passed'] and result['first_engagement_t'] == 2. and result['lead_s'] == pytest.approx(.99)
    toward[260] = .2
    assert not wg.score_lead(arrays(turn_first=engaged, cvx=toward), gate, WALL)['passed']
    late = np.zeros(300)
    late[260:] = 1.
    result = wg.score_lead(arrays(turn_first=late), gate, WALL)
    assert not result['passed'] and result['lead_s'] == pytest.approx(.39)
    assert not wg.score_lead(arrays(), gate, WALL)['passed']    # never engaged
    early = np.zeros(300)
    early[:100] = 1.                                           # an episode before the window does not count
    assert wg.score_lead(arrays(turn_first=early), dict(gate, max_toward_mps=None), WALL)['first_engagement_t'] is None


def test_pd_gate_counts_only_requests_toward_the_wall_and_estimates_the_delay():
    gate = dict(window=[0., 2.99], exit_window=[2., 2.99], max_extra_toward_mps=.05, max_engaged_s=1.,
                max_delay_s=1.)
    base = arrays(cvx=np.full(300, -.9), cvy=np.full(300, 3.))
    mine = arrays(cvx=np.full(300, -.5), cvy=np.full(300, 3.))  # less negative, still away from the wall
    speed = np.full(300, 4.)
    result = wg.score_pd(mine, base, speed, gate, WALL)
    assert result['passed'] and result['max_extra_toward_wall'] == 0.
    mine['cvx'] = np.full(300, .2)
    assert not wg.score_pd(mine, base, speed, gate, WALL)['passed']
    slow = arrays(cvx=np.full(300, -.9), cvy=np.r_[np.full(100, 1.), np.full(200, 3.)])
    slow['turn_first'][:100] = 1.
    result = wg.score_pd(slow, base, speed, gate, WALL)
    deficit = np.hypot(.9, 3.)-np.hypot(.9, 1.)                 # 1 s of the slower request
    assert result['path_deficit_m'] == pytest.approx(deficit, abs=.02)
    assert result['delay_estimate_s'] == pytest.approx(deficit/4., abs=.01)
    assert result['engaged_s'] == pytest.approx(1., abs=.01) and result['passed']


def test_quiet_gate_counts_turn_first_and_side_guard_ticks():
    assert wg.score_quiet(arrays())['passed']
    guard = np.zeros(300)
    guard[10:20] = 1.
    result = wg.score_quiet(arrays(side_guard=guard))
    assert not result['passed'] and result['side_guard_s'] == pytest.approx(.1)
    engaged = np.zeros(300)
    engaged[50:60] = engaged[100:110] = 1.
    result = wg.score_quiet(arrays(turn_first=engaged))
    assert not result['passed'] and result['turn_first_episodes'] == 2
    assert wg.episodes(arrays(turn_first=engaged)) == [(.5, .59, .1), (1., 1.09, .1)]


def test_motor_rollout_follows_the_delayed_request_within_its_acceleration_bound():
    t = np.arange(0., 3., .01)
    request = np.zeros((len(t), 2))
    request[t >= 1.] = [4., 0.]
    rt, rv, rp = wg.motor_rollout([0., 0.], t, request, 0., 3., delay=.3, tau=.1, accel=4.)
    assert np.all(rv[rt <= 1.3, 0] == 0.) and rv[rt > 1.35, 0].min() > 0.
    assert np.max(np.diff(rv[:, 0]))/.01 <= 4.+1e-9 and rv[-1, 0] == pytest.approx(4., abs=.01)
    assert rp[-1, 0] == pytest.approx(np.sum(rv[:, 0])*.01)


def test_the_replay_harness_builds_the_contracts_stopping_model():
    side = dict(motor_controller=dict(contract='fast_velocity_pd_v1'), obstacle_stack=None,
                pilot_assistance=dict(nominal_speed_mps=6.), gate_sensor=SENSOR)
    pilot, info = vr.build(side, str(TREE), 'on', 'off', 'off')
    declaration = json.loads((TREE/'configs'/'obstacles'/'wall_pilot.json').read_text(encoding='utf-8'))
    pd = declaration['turn_first_stopping']['fast_velocity_pd_v1']
    assert info['wall'] == 'on' and pilot.wall_apply
    assert pilot.turn_first.stop_deceleration == pd['stop_deceleration']
    assert pilot.turn_first.stop_latency_s == pd['stop_latency_s']
    side['motor_controller']['contract'] = 'fast_velocity_brain_v1'
    brain, _ = vr.build(side, str(TREE), 'shadow', 'off', 'off')
    assert brain.turn_first.stop_deceleration == declaration['turn_first_stopping']['default']['stop_deceleration']
    assert not brain.wall_apply
