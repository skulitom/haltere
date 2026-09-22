import cv2
import numpy as np
import pytest

from haltere.vision.race_cues import checkpoint_ring, flag_clearance
from haltere.liftoff.race_cue_assistance import RaceCueAssistance
from haltere.liftoff.camera_pose import CameraPoseHistory
from tests.test_visual_assistance import SENSOR, senses, checkpoint


def picture(*centres):
    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    for centre in centres:
        cv2.circle(rgb, centre, 7, (255, 255, 255), -1)
        cv2.circle(rgb, centre, 4, (20, 40, 25), -1)
    return rgb


def test_ring_survives_touching_arch_and_does_not_confuse_reticle_or_sticks():
    rgb = picture((640, 634), (586, 666), (693, 666))
    cv2.circle(rgb, (640, 360), 8, (255, 255, 255), 1)
    cv2.line(rgb, (633, 628), (610, 605), (255, 255, 255), 4)
    found = checkpoint_ring(rgb)
    assert found is not None and not found['edge']
    assert found['u'] == pytest.approx(.5, abs=.002)
    assert found['v'] == pytest.approx(634/720, abs=.002)


def test_missing_ambiguous_and_edge_cues_are_explicit():
    assert checkpoint_ring(picture()) is None
    assert checkpoint_ring(picture((400, 400), (900, 450))) is None
    assert checkpoint_ring(picture((19, 200)))['edge']
    assert checkpoint_ring(picture((640, 40))) is None  # timer


@pytest.mark.parametrize('side', [-1, 1])
def test_flag_clearance_uses_visible_arrow_side_and_preserves_ring(side):
    rgb = picture((640, 540))
    cv2.rectangle(rgb, (613, 485), (628, 505), (30, 25, 85), -1)
    x = 500 if side < 0 else 760
    cv2.rectangle(rgb, (x, 530), (x+40, 550), (40, 240, 170), -1)
    cue = checkpoint_ring(rgb)
    assert cue is not None and cue['u'] == .5
    assert side*(cue['aim_u']-cue['u']) > .035
    assert flag_clearance(rgb, dict(u=.5, v=.75, edge=True)) == .5
    assert flag_clearance(picture(), dict(u=.5, v=.75, edge=False)) == .5


def test_flag_clearance_follows_mirrored_cloth_not_a_fixed_passing_side():
    rgb = picture((650, 530))
    cv2.fillConvexPoly(rgb, np.array([[615,440],[635,440],[662,550],[640,550]]), (170,175,165))
    cv2.rectangle(rgb, (605, 420), (624, 443), (30,25,85), -1)
    cv2.rectangle(rgb, (750,530), (800,550), (40,240,170), -1)
    cv2.circle(rgb, (650,530), 7, (255,255,255), -1)
    cv2.circle(rgb, (650,530), 4, (20,40,25), -1)
    a = checkpoint_ring(rgb)
    b = checkpoint_ring(rgb[:,::-1].copy())
    assert a['aim_u'] > a['u'] and b['aim_u'] < b['u']
    assert a['aim_u']+b['aim_u'] == pytest.approx(1279/1280, abs=.015)


def test_cloud_holes_and_leaderboard_icons_are_not_race_targets():
    rgb = picture((400, 500), (1253, 410))
    cv2.rectangle(rgb, (850, 250), (1000, 400), (255, 255, 255), -1)
    cv2.circle(rgb, (920, 320), 4, (20, 40, 25), -1)
    found = checkpoint_ring(rgb)
    assert found is not None
    assert found['u'] == pytest.approx(400/1280, abs=.002)


def helper():
    history = CameraPoseHistory()
    history.append(10., [0., 0., 1.5], [1., 0., 0., 0.])
    assist = RaceCueAssistance(SENSOR, history)
    # Use an actual projected horizontal bearing, not an assumed image centre.
    pixels, _ = assist.camera.project_body(np.array([[10., 0., 0.]]))
    cue = dict(u=pixels[0, 0]/320, v=pixels[0, 1]/180, edge=False)
    return assist, cue


def test_cue_is_causal_and_controls_only_goal_senses_and_yaw():
    assist, cue = helper()
    s = senses(velocity=(1., 0., 0.))
    original = {k:v.clone() for k,v in s.items()}
    relative, modified = assist.update(s, [0.,0.,0.], dict(race_cue=cue), 10., 10.08)
    np.testing.assert_allclose(relative, [3.,0.,0.], atol=1e-5)
    assert assist.pilot.mode == 2
    assert all((s[k] == original[k]).all() for k in s)
    assert modified['pos'] is s['pos']
    action = np.array([.1,.2,.3,.4])
    np.testing.assert_array_equal(assist.command(action)[:3], action[:3])
    np.testing.assert_array_equal(action, [.1,.2,.3,.4])
    assist.update(s, [0.,0.,0.], dict(race_cue=cue), 10., 10.09)
    assert assist.frames == 1
    relative, _ = assist.update(s, [0.,0.,0.], dict(race_cue=cue), 10.4, 10.3)
    assert assist.frames == 1  # future image is not evidence
    assert assist.pilot.mode == 4 and relative[0] < 0  # brake, don't circle forward
    assert assist.metadata()['visible_race_cues']
    assert not assist.metadata()['runtime_route_oracle']
    assert assist.metadata()['estimated_passages'] is None


def test_edge_cue_brakes_instead_of_making_a_false_3d_target():
    assist, cue = helper()
    cue.update(u=.01, v=.7, edge=True)
    relative, _ = assist.update(senses(velocity=(2.,0.,0.)), [0.,0.,0.],
                                dict(race_cue=cue), 10., 10.08)
    assert assist.pilot.mode == 4 and relative[0] < 0
    assert assist.pilot.sight_yaw != 0.


def test_bottom_edge_cue_descends_without_a_forced_search_turn():
    assist, cue = helper()
    cue.update(v=.975, edge=True)
    relative, _ = assist.update(senses(velocity=(0.,0.,0.)), [0.,0.,0.],
                                dict(race_cue=cue), 10., 10.08)
    assert assist.pilot.mode == 2
    assert relative[2] == pytest.approx(-1.2)
    assert np.linalg.norm(relative[:2]) == pytest.approx(1.)
    assert assist.pilot.sight_yaw == pytest.approx(0.)


def test_takeoff_clearance_does_not_block_flight_below_start_elevation():
    assist, cue = helper()
    cue.update(v=.975, edge=True)
    s = senses(velocity=(0.,0.,0.))
    s['pos'][0,2] = 0.
    relative, _ = assist.update(s, [0.,0.,0.], dict(race_cue=cue), 10., 10.01)
    assert relative[2] > 0.  # clear the launch surface first
    s['pos'][0,2] = .7
    assist.update(s, [0.,0.,0.], dict(race_cue=cue), 10., 10.02)
    for altitude in (.4, -10.):
        s['pos'][0,2] = altitude
        relative, _ = assist.update(s, [0.,0.,0.], dict(race_cue=cue), 10., 10.03)
        assert relative[2] == pytest.approx(-1.2)
        assert not assist.launching


def test_capture_outage_brakes_without_blind_yaw_or_stale_visual_updates():
    from haltere.liftoff.visual_brain import camera_measurement_fresh
    assist, cue = helper()
    s = senses(velocity=(2., 0., 0.))
    assist.update(s, [0.,0.,0.], dict(race_cue=cue), 10., 10.01)
    relative, _ = assist.update(s, [0.,0.,0.], None, 10., 10.3)
    assert assist.pilot.mode == 4 and relative[0] < 0
    assert assist.pilot.sight_yaw == pytest.approx(0.)
    assert assist.frames == 1
    assert not camera_measurement_fresh(.3, True, allow_braking=True)
    assert not camera_measurement_fresh(.5, True, allow_braking=True)
    for age, memory, braking in [(.501,True,True),(.251,True,False),(.121,False,True)]:
        with pytest.raises(RuntimeError, match='Fresh camera'):
            camera_measurement_fresh(age, memory, allow_braking=braking)
    with pytest.raises(RuntimeError, match='hidden'):
        camera_measurement_fresh(.3, True, foreground=False, allow_braking=True)


def test_runner_keeps_learned_retina_and_neural_motor_outputs_in_cue_mode(checkpoint):
    import torch
    from haltere.liftoff.visual_brain import VisualController
    from haltere.liftoff.telemetry import TelemetryFrame
    controller = VisualController(*checkpoint, 'cpu', pilot_assistance='race-cue')
    retina = torch.ones(1, 720)*.3
    state = {k:v.clone() for k,v in controller.state.items()}
    frame = TelemetryFrame(timestamp=1., motor_rpm=np.ones(4)*1000.)
    action, _, _ = controller.step(frame, retina,
        dict(p=.1, point=np.array([10.,0.,0.]), width=30.,
             race_cue=dict(u=.4,v=.8,edge=False)), 10., 10., 10.)
    with torch.no_grad():
        expected, _, _ = controller.brain(controller.last_observation, state, controller.W)
    np.testing.assert_array_equal(action, expected[0].numpy())
    np.testing.assert_array_equal(controller.last_command[:3], action[:3])
    assert torch.equal(controller.last_observation['retina'], retina)
