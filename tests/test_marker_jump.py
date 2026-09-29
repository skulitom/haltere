"""FastRaceCue marker-jump confirmation (`MarkerJumpConfig`, runner flag --marker-jump; configs/pilot/marker_jump.json):
off by default.

The ring reader sometimes reads something else as the checkpoint marker for a frame or two where the marker is
unread (straw-brain11cw13-r4b-noassist-02: a banner logo 16.8 deg right of the ring after 0.55 s without a reading, two
captures; the pilot turned toward it and the drone clipped the next arch's leg). With the rule, an in-view marker that
jumps at least jump_deg from the last accepted in-view ring bearing after a gap in the readings is held (no reading)
until its confirm-th reading within agree_deg and window_s. These tests pin: the default pilot is unchanged and shadow
changes nothing, the hold and its release, confirmation, direct switches and edge clamps taken at once, the declared
file and the runner wiring, and the harness pieces the gates use. None of it is flight evidence.
"""
import argparse
import json
from dataclasses import asdict

import numpy as np
import pytest

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (MARKER_JUMP_MODES, MARKER_JUMP_VERSION, FastRaceCue, MarkerJumpConfig,
                                           marker_jump_config)
from tests.test_fast_race_cue import cue_toward, drive
from tests.test_visual_assistance import SENSOR

MJ = MarkerJumpConfig()


def azimuth_cue(degrees, distance=20.):
    """In-view cue for a level ring at a world azimuth (drive() keeps yaw 0; positive = left)."""
    a = np.radians(degrees)
    return cue_toward([distance*np.cos(a), distance*np.sin(a), 0.])


def schedule(*spans):
    """A cue callable of time from (start, end, cue or None) spans (times relative to drive()'s start, 10 s)."""
    def cue(now):
        t = now-10.
        for start, end, c in spans:
            if start <= t < end:
                return c
        return None
    return cue


def run(marker_jump=None, apply=True, cue=None, steps=150):
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., reference_speed=6.,
                        **({} if marker_jump is None else dict(marker_jump=marker_jump, marker_jump_apply=apply)))
    rows = drive(pilot, pilot.pose_history, cue, steps, velocity=(6., 0., 0.))
    return pilot, rows


FALSE = schedule((0., .5, azimuth_cue(0.)), (.8, .82, azimuth_cue(-17.)), (1.2, 1.5, azimuth_cue(1.)))


def direction_deg(pilot):
    return float(np.degrees(np.arctan2(pilot.direction[1], pilot.direction[0])))


def test_default_and_explicit_none_and_shadow_are_unchanged():
    _, plain = run(cue=FALSE)
    _, none = run(marker_jump=None, cue=FALSE)
    pilot, shadow = run(marker_jump=MJ, apply=False, cue=FALSE)
    for a, b in ((plain, none), (plain, shadow)):
        assert [r[1] for r in a] == [r[1] for r in b]
        assert all(np.array_equal(x[2], y[2]) for x, y in zip(a, b))
    # shadow logs what it would hold and holds nothing
    assert pilot.marker_counts['held'] == 2 and pilot.marker_counts['rejected'] == 1
    assert 'marker_jump' not in FastRaceCue(SENSOR, CameraPoseHistory(), 6.).metadata()
    assert pilot.metadata()['marker_jump']['applied'] is False


def test_a_jump_after_a_gap_is_held_and_rejected():
    pilot, rows = run(marker_jump=MJ, cue=FALSE, steps=150)
    # two readings of the false marker (80-81 ms after 0.3 s unread) were held: the bearing stayed at the ring
    during = [r for r in rows if 10.8 <= r[0] < 10.83]
    assert all(abs(float(np.degrees(np.arctan2(r[2][1], r[2][0])))) < 2. for r in during)
    assert abs(direction_deg(pilot)) < 2.
    assert pilot.marker_counts == dict(held=2, candidates=1, confirmed=0, rejected=1)
    # without the rule the pilot turns toward it
    plain, _ = run(cue=FALSE, steps=85)
    assert direction_deg(plain) < -10.


def test_a_jump_confirmed_by_three_readings_is_taken():
    cue = schedule((0., .5, azimuth_cue(0.)), (.8, 1.2, azimuth_cue(-17.)))
    pilot, rows = run(marker_jump=MJ, cue=cue, steps=100)
    assert pilot.marker_counts['confirmed'] == 1 and pilot.marker_counts['held'] == MJ.confirm-1
    assert direction_deg(pilot) < -15.


def test_a_direct_jump_without_a_gap_is_a_switch():
    cue = schedule((0., .5, azimuth_cue(0.)), (.5, 1., azimuth_cue(35.)))
    pilot, _ = run(marker_jump=MJ, cue=cue, steps=60)
    assert pilot.marker_counts['held'] == 0 and direction_deg(pilot) > 30.


def test_a_small_jump_after_a_gap_is_the_same_ring():
    cue = schedule((0., .5, azimuth_cue(0.)), (.9, 1.2, azimuth_cue(6.)))
    pilot, _ = run(marker_jump=MJ, cue=cue, steps=100)
    assert pilot.marker_counts['held'] == 0


def test_edge_clamps_are_readings_and_end_a_candidate():
    side = dict(u=.0, v=.6, edge=True)
    cue = schedule((0., .5, azimuth_cue(0.)), (.8, .81, azimuth_cue(-17.)), (.81, .9, side), (.9, 1., azimuth_cue(40.)))
    pilot, _ = run(marker_jump=MJ, cue=cue, steps=100)
    # the edge clamp ended the candidate and counted as a reading, so the in-view marker after it is taken at once
    assert pilot.marker_counts['rejected'] == 1 and pilot.marker_counts['held'] == 1
    assert direction_deg(pilot) > 30.


def test_readings_beyond_the_window_or_disagreeing_start_a_new_candidate():
    apart = schedule((0., .5, azimuth_cue(0.)), (.8, .81, azimuth_cue(-17.)), (1.3, 1.31, azimuth_cue(-17.)))
    pilot, _ = run(marker_jump=MJ, cue=apart, steps=140)
    assert pilot.marker_counts['candidates'] == 2 and pilot.marker_counts['rejected'] == 1
    other = schedule((0., .5, azimuth_cue(0.)), (.8, .81, azimuth_cue(-17.)), (.81, .82, azimuth_cue(-30.)))
    pilot, _ = run(marker_jump=MJ, cue=other, steps=90)
    assert pilot.marker_counts['candidates'] == 2 and pilot.marker_counts['held'] == 2


@pytest.mark.parametrize('kw', [dict(jump_deg=0.), dict(jump_deg=95.), dict(gap_s=0.), dict(gap_s=2.),
                                dict(confirm=1), dict(confirm=2.5), dict(confirm=True), dict(agree_deg=float('nan')),
                                dict(window_s=0.), dict(window_s=3.)])
def test_invalid_parameters_are_refused(kw):
    with pytest.raises(ValueError):
        MarkerJumpConfig(**kw)


def test_the_pilot_refuses_another_config_type():
    with pytest.raises(ValueError, match='MarkerJumpConfig'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., marker_jump=dict(jump_deg=10.))


def test_the_declaration_is_frozen_and_declares_the_code_defaults(tmp_path):
    from haltere.liftoff import visual_brain as vb
    declaration, digest = vb.load_marker_jump()
    assert declaration['version'] == MARKER_JUMP_VERSION == 1 and digest == declaration['sha256']
    assert asdict(marker_jump_config(declaration)) == asdict(MJ)
    edited = dict(declaration, marker_jump=dict(declaration['marker_jump'], confirm=2))
    path = tmp_path/'edited.json'
    path.write_text(json.dumps(edited), encoding='utf-8')
    with pytest.raises(ValueError, match='changed after the freeze'):
        vb.load_marker_jump(path)
    other = dict(declaration, version=2)
    other['sha256'] = vb.lag_turn_declaration_sha256(other)
    path.write_text(json.dumps(other), encoding='utf-8')
    with pytest.raises(ValueError, match='version 2'):
        vb.load_marker_jump(path)
    with pytest.raises(ValueError, match='version'):
        marker_jump_config(dict(declaration, version=2))


def test_the_gates_are_frozen_and_name_the_declaration():
    from haltere.obstacles.marker_jump_gates import load_gates
    gates, digest = load_gates()
    assert gates['declarations']['marker_jump']['version'] == 1 and gates['sha256'] == digest


def test_runner_flag_and_columns():
    from haltere.liftoff import visual_brain as vb
    fast = dict(pilot_assistance='race-cue', pilot_profile='fast')
    assert vb.resolve_marker_jump(argparse.Namespace(**fast)) == (None, None)
    assert vb.resolve_marker_jump(argparse.Namespace(marker_jump='off', **fast)) == (None, None)
    assert vb.resolve_marker_jump(argparse.Namespace(marker_jump='on', **fast)) == (str(vb.MARKER_JUMP_DECLARATION), 'on')
    assert vb.resolve_marker_jump(argparse.Namespace(marker_jump='shadow', **fast))[1] == 'shadow'
    with pytest.raises(ValueError, match='fast race-cue pilot'):
        vb.resolve_marker_jump(argparse.Namespace(marker_jump='on', pilot_assistance='race-cue',
                                                  pilot_profile='standard'))
    assert MARKER_JUMP_MODES == ('on', 'shadow')
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6.)
    assert all(np.isnan(v) for v in vb.marker_jump_row(pilot))
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., marker_jump=MJ)
    assert vb.marker_jump_row(pilot) == (0., 0.)
    assert vb.MARKER_JUMP_COLUMNS == tuple(pilot.marker_jump_log())


def test_deployed_pilot_adds_the_rule_only_when_asked():
    from haltere.train.deployed_pilot import deployed_pilot_kwargs
    kw, record = deployed_pilot_kwargs('fast_velocity_brain_v1')
    assert 'marker_jump' not in kw and 'marker_jump' not in record
    kw, record = deployed_pilot_kwargs('fast_velocity_brain_v1', marker_jump='shadow')
    assert kw['marker_jump'] == MJ and kw['marker_jump_apply'] is False and record['marker_jump']['version'] == 1
    with pytest.raises(ValueError, match='marker-jump mode'):
        deployed_pilot_kwargs('fast_velocity_brain_v1', marker_jump='maybe')


def test_harness_gate_scenario_and_false_marker():
    from haltere.liftoff import motor_assist_eval as mae
    from haltere.vision.camera import Camera
    sc = mae.gate_scenario(turn_deg=90., half_w=1.5, false_deg=16.6)
    assert len(sc['posts']) == 4 and sc['false_marker']['captures'] == 2
    np.testing.assert_allclose(sc['posts'][0], [1.5, 24.])
    np.testing.assert_allclose(sc['posts'][2], [15., 24.-1.5], atol=1e-9)
    assert mae.gate_scenario()['false_marker'] is None
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    q = np.array([np.cos(np.pi/4), 0., 0., np.sin(np.pi/4)])      # facing north
    true = mae.hud_marker(camera, np.array([0., 20., 0.]), np.zeros(3), q)
    false = mae._false_cue(camera, np.array([0., 20., 0.]), np.zeros(3), q, 16.6)
    assert abs(true['u']-.5) < 1e-6 and false['u'] > .55 and not false['edge']


def test_harness_rows_of_other_scenarios_gain_no_keys():
    from haltere.liftoff import motor_assist_eval as mae
    assert 'posts' not in mae.passthrough_scenario() and 'false_marker' not in mae.hairpin_scenario()


def test_replay_harness_component_switches(tmp_path):
    from haltere.obstacles.vertical_replay import build
    side = dict(motor_controller=dict(contract='fast_velocity_brain_v1'), obstacle_stack=dict(mode='on'),
                pilot_assistance=dict(nominal_speed_mps=6.), gate_sensor=SENSOR)
    tree = str(__import__('pathlib').Path(__file__).resolve().parents[1])
    pilot, info = build(side, tree, 'on')
    assert pilot.lag_turn is not None and pilot.gap_aim is not None and 'lag_turn' not in info
    pilot, info = build(side, tree, 'on', lag_turn=False, gap_aim=False)
    assert pilot.lag_turn is None and pilot.gap_aim is None and info['lag_turn'] is False
    declaration = json.loads((__import__('pathlib').Path(tree)/'configs'/'pilot'/'marker_jump.json')
                             .read_text(encoding='utf-8'))
    pilot, info = build(side, tree, 'none', marker_jump=declaration)
    assert pilot.marker_jump == MJ and pilot.marker_jump_apply and info['marker_jump'] == 'applied'
    pilot, info = build(side, tree, 'shadow', marker_jump=declaration)
    assert not pilot.marker_jump_apply and info['marker_jump'] == 'shadow'
