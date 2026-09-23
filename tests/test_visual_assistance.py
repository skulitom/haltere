import copy
import json

import numpy as np
import pytest
import torch

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.visual_assistance import VisualPilotAssistance
from haltere.liftoff.visual_brain import VisualController, camera_measurement_fresh
from haltere.vision.camera import quat_wxyz_to_mat


SENSOR = dict(focal_320=100., tilt_deg=30., centre_offset_m=0.,
              missing_gate='zero_goal_neural_search', height_invariant=True,
              gravity_aligned_height=True, raw_retina_active=True, retina_mode='scene_v2')


def senses(position=(0., 0., 1.5), velocity=(1., 0., 0.), yaw=0.):
    q = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
    R = quat_wxyz_to_mat(q)
    t = lambda x: torch.tensor([x], dtype=torch.float32)
    return dict(pos=t(position), quat=t(q.tolist()), vel_world=t(velocity),
                vel_body=t((R.T@velocity).tolist()), gyro=t([0., 0., 0.]),
                gravity_body=t([0., 0., -1.]), altitude=t([position[2]]),
                yaw=t([yaw]), up=torch.ones(1))


def test_assistance_tracks_calibrated_opening_once_per_original_frame():
    history = CameraPoseHistory()
    helper = VisualPilotAssistance(SENSOR, history)
    s = senses()
    world = np.array([9., 0., 1.8])
    # Camera is 80 ms behind the controller. All measurements describe one
    # stationary opening despite the intervening drone translation.
    for i in range(60):
        stamp = 10.+i*.02
        capture_pos = np.array([i*.01, 0., 1.5])
        history.append(stamp, capture_pos, [1., 0., 0., 0.])
        s = senses(position=(i*.01+.04, 0., 1.5))
        relative, modified = helper.update(s, np.zeros(3),
            dict(p=.99, point=world-capture_pos, width=25.), stamp, stamp+.08)
        assert np.isfinite(relative).all()
        assert modified['pos'] is s['pos']
    assert len(helper.pilot.tracks) == 1
    np.testing.assert_allclose(helper.pilot.tracks[0].m, world, atol=1e-5)
    assert helper.pilot.tracks[0].confirmed
    assert helper.pilot.params.centre_offset_m == 0.
    assert helper.pilot.params.range_corr is None
    hits = helper.pilot.tracks[0].hits
    helper.update(s, np.zeros(3), dict(p=.99, point=np.array([9., 0., 3.]), width=25.), stamp, stamp+.09)
    assert helper.pilot.tracks[0].hits == hits
    frames = helper.frames
    helper.update(s, np.zeros(3), None, stamp+.01, stamp+.2)  # over-age pixels discarded
    assert helper.frames == frames
    helper.update(s, np.zeros(3), None, stamp+.3, stamp+.2)  # future sample rejected
    assert helper.frames == frames


def test_speed_scheduling_preserves_actual_motion_and_brain_controls():
    history = CameraPoseHistory()
    s = senses(velocity=(4., 2., 1.), yaw=.7)
    history.append(10., [0., 0., 1.5], s['quat'][0].numpy())
    helper = VisualPilotAssistance(SENSOR, history, speed=4.)
    helper.pilot.flow_f = .7
    original = copy.deepcopy(s)
    _, modified = helper.update(s, np.zeros(3), None, 10., 10.)
    assert all(torch.equal(s[k], original[k]) for k in s)
    np.testing.assert_allclose(modified['vel_world'], [[2.8, 1.4, 1.]], atol=1e-6)
    R = quat_wxyz_to_mat(s['quat'][0].numpy())
    np.testing.assert_allclose(modified['vel_body'], modified['vel_world'].numpy()@R, atol=1e-6)
    helper.pilot.sight_yaw = -.2
    brain = np.array([.1, .2, .3, .4])
    np.testing.assert_array_equal(helper.command(brain), [.1, .2, .3, -.2])
    np.testing.assert_array_equal(brain, [.1, .2, .3, .4])
    assert helper.metadata()['yaw_assistance']
    with pytest.raises(RuntimeError, match='Fresh camera'):
        camera_measurement_fresh(.251, allow_memory=True)


@pytest.mark.parametrize('speed', [0., -1., 5.01, float('nan')])
def test_invalid_assistance_speed_is_rejected(speed):
    with pytest.raises(ValueError, match='speed'):
        VisualPilotAssistance(SENSOR, CameraPoseHistory(), speed)


def test_raw_retina_checkpoint_cannot_silently_become_gate_guided():
    with pytest.raises(ValueError, match='gate sensory contract'):
        VisualPilotAssistance(None, CameraPoseHistory())


@pytest.fixture
def checkpoint(tmp_path):
    from tests.test_human_brain import small_brain
    from haltere.train.human_brain import export
    from haltere.vision.datasets import sha256
    brain, cfg, graph = small_brain()
    cfg.train.graph = str(tmp_path/'flight')
    graph.save(tmp_path/'flight')
    calibration = dict(hover_processed=0., hover_stick_sim=0., throttle_scale=1.,
                       max_rpm=30000., stick_sign=[1., 1., 1.])
    meta = dict(runtime_requires_teacher=False, calibration=calibration,
                graph_sha256=sha256(tmp_path/'flight.npz'), gate_sensor=SENSOR,
                scene_training=dict(iteration=1, iterations=1))
    path = tmp_path/'brain.pt'
    export(path, brain, cfg, meta, 1)
    mapping = tmp_path/'mapping.yaml'
    mapping.write_text(json.dumps(dict(mapping=dict(hover_processed=0., stick_sign=[1., 1., 1.]))))
    return path, mapping


def test_runner_keeps_exact_brain_action_and_labels_assisted_command(checkpoint):
    from haltere.liftoff.telemetry import TelemetryFrame
    controller = VisualController(*checkpoint, 'cpu', pilot_assistance='rabbit')
    retina = torch.ones(1, 720)*.3
    state = {k: v.clone() for k, v in controller.state.items()}
    frame = TelemetryFrame(timestamp=1., motor_rpm=np.ones(4)*1000.)
    action, processed, raw = controller.step(frame, retina,
        dict(p=.99, point=np.array([8., 2., 1.5]), width=30.), 10., 10., 10.)
    with torch.no_grad():
        expected, _, _ = controller.brain(controller.last_observation, state, controller.W)
    np.testing.assert_array_equal(action, expected[0].numpy())
    np.testing.assert_array_equal(controller.last_command[:3], action[:3])
    assert controller.last_command[3] == controller.assistance.pilot.sight_yaw
    assert controller.last_command[3] != action[3]
    np.testing.assert_allclose(processed, controller.last_command)
    assert np.isfinite(raw).all()
    assert torch.equal(controller.last_observation['retina'], retina)
    assert not controller.searching


def test_default_runner_remains_unassisted(checkpoint):
    from haltere.liftoff.telemetry import TelemetryFrame
    controller = VisualController(*checkpoint, 'cpu')
    action, processed, raw = controller.step(
        TelemetryFrame(timestamp=1., motor_rpm=np.ones(4)*1000.), torch.zeros(1,720),
        dict(p=.99, point=np.array([8., 2., 1.5]), width=30.), 10., 10., 10.)
    assert controller.assistance is None
    np.testing.assert_array_equal(controller.last_command, action)
    np.testing.assert_allclose(processed, action)
    assert np.isfinite(raw).all()


@pytest.mark.parametrize('speed', [2., 2.5, 3., 4.])
def test_pd_comparison_uses_same_guidance_and_preserves_shadow_brain(checkpoint, speed):
    from dataclasses import replace
    from haltere.brain.motor_baseline import MotorPD, MotorPDConfig
    from haltere.liftoff.telemetry import TelemetryFrame
    path, mapping = checkpoint
    ck = torch.load(path, weights_only=True)
    ck['visual_brain']['gate_training'] = dict(dynamics=dict(profile=dict(
        vertical_calibration=dict(mean=dict(hover_processed=0., slope_mps2_per_processed=17.)))))
    ck['visual_brain']['motor_tracking'] = dict(nominal_speed_mps=3.)
    torch.save(ck, path)
    brain = VisualController(path, mapping, 'cpu', pilot_assistance='race-cue', assist_speed=speed)
    pd = VisualController(path, mapping, 'cpu', pilot_assistance='race-cue', assist_speed=speed, motor_controller='pd')
    frame = TelemetryFrame(timestamp=1., velocity=np.array([.4, .1, 1.3]), motor_rpm=np.ones(4)*1000.)
    detection = dict(p=.99, point=np.array([8., 0., 1.5]), width=30.,
                     race_cue=dict(u=.55, v=.5, edge=False))
    retina = torch.ones(1, 720)*.3
    brain_action, _, _ = brain.step(frame, retina, detection, 10., 10., 10.)
    pd_shadow, _, _ = pd.step(frame, retina, detection, 10., 10., 10.)
    for key in brain.last_observation:
        torch.testing.assert_close(brain.last_observation[key], pd.last_observation[key])
    np.testing.assert_array_equal(brain_action, pd_shadow)
    assert pd.last_command[3] == brain.last_command[3]
    assert not np.allclose(pd.last_command[:3], pd_shadow[:3])
    assert pd.motor_metadata['brain_controls_motors'] is False
    effective_speed = min(speed, 3.)
    teacher = MotorPD(replace(pd.cfg.quad, **pd.motor_metadata['effective_thrust_curve']),
                      pd.cfg.rates, pd.cfg.ctl.idle,
                      MotorPDConfig(position_gain=max(.8, effective_speed/3.)))
    expected = teacher.command(pd.senses, torch.tensor(pd.relative_gate, dtype=torch.float32)[None],
                               speed=effective_speed)[0].numpy()
    np.testing.assert_allclose(pd.last_command[:3], expected[:3])
    assert pd.motor_metadata['speed_mps'] == effective_speed
    assert pd.motor_baseline.config == teacher.config


@pytest.mark.parametrize('motor', ['pd', 'brain'])
def test_geometry_brake_changes_goal_but_preserves_unmodified_task_and_motor_authority(checkpoint, motor):
    import time
    from types import SimpleNamespace
    from haltere.liftoff.geometry_control import GeometryControlGate
    from haltere.liftoff.telemetry import TelemetryFrame
    path,mapping=checkpoint
    saved=torch.load(path,weights_only=True)
    saved['visual_brain']['gate_training']=dict(dynamics=dict(profile=dict(
        vertical_calibration=dict(mean=dict(hover_processed=0.,slope_mps2_per_processed=17.)))))
    saved['visual_brain']['motor_tracking']=dict(nominal_speed_mps=3.)
    torch.save(saved,path)
    pd=VisualController(path,mapping,'cpu',pilot_assistance='race-cue',assist_speed=2.5,motor_controller=motor)
    frame=TelemetryFrame(timestamp=1.,motor_rpm=np.ones(4)*1000.)
    detection=dict(p=.99,point=np.array([8.,0.,1.5]),width=30.,race_cue=dict(u=.5,v=.5,edge=False))
    stamp=time.monotonic()
    pd.step(frame,torch.zeros(1,720),detection,stamp,stamp,stamp)
    original=pd.relative_gate.copy()
    desired=pd.guidance_velocity.world_velocity(original, np.eye(3))
    stamp=time.monotonic()
    pd.geometry_gate=GeometryControlGate()
    pd.geometry_provider=SimpleNamespace(latest=lambda:dict(available_at=stamp,capture_time=stamp,
        requested_goal_time=stamp,status='observed_obstacle_brake',position=[0,0,0],
        requested_velocity=desired,proposal_velocity=[0,0,0],changed=True))
    action, _, _ = pd.step(frame,torch.zeros(1,720),detection,stamp,stamp,stamp)
    np.testing.assert_allclose(pd.nominal_relative_gate,original)
    np.testing.assert_array_equal(pd.relative_gate,[0,0,0])
    expected=(pd.motor_baseline.command(pd.senses,torch.zeros(1,3),speed=pd.motor_speed)[0].numpy()
              if motor == 'pd' else action)
    np.testing.assert_allclose(pd.last_command[:3],expected[:3])
    assert pd.last_command[3] == pd.assistance.pilot.sight_yaw
    assert pd.motor_metadata['brain_controls_motors'] == (motor == 'brain')
    assert pd.geometry_gate.status=='obstacle_brake'
