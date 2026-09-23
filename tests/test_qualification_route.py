import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haltere.liftoff.challenge_courses import generate_sections
from haltere.liftoff.collection_route import CollectionRoute
from haltere.liftoff.oracle_assistance import OracleCollectionAssistance
from haltere.liftoff.qualification_route import model_clearance, plan_segment, prepare
from haltere.liftoff.telemetry import TelemetryFrame
from haltere.liftoff.visual_brain import VisualController
from tests.test_visual_assistance import checkpoint


def test_oracle_route_goes_around_a_wall_and_preserves_clearance():
    geometry=dict(ground_plane_y=0.,primitives=[dict(center=[0,2.5,4],size=[3,5,1],yaw_deg=0)])
    points=plan_segment([0,2,0],[0,2,8],geometry)
    assert len(points)>2
    samples=np.concatenate([np.linspace(a,b,200) for a,b in zip(points,points[1:])])
    assert model_clearance(samples,geometry).min()>=.9-1e-8


def test_generated_qualification_route_preserves_checkpoint_order_and_oracle_label(tmp_path):
    bundle=tmp_path/'bundle';out=tmp_path/'route.json'
    generate_sections(bundle,123,box_calibration=True)
    result=prepare(bundle,out)
    geometry=json.loads((bundle/'offline-geometry.json').read_text())
    path=np.array(result['waypoints_unity'])
    checkpoint_indices=[np.flatnonzero(np.linalg.norm(path-c['center'],axis=1)<1e-8)[0]
                        for c in geometry['checkpoints']]
    assert np.all(np.diff(checkpoint_indices)>0)
    assert result['minimum_model_clearance_m']>=.9
    assert 'PRIVILEGED' in result['purpose']
    CollectionRoute(out)
    with pytest.raises(FileExistsError):prepare(bundle,out)


def test_oracle_mode_requires_pd_and_cannot_masquerade_as_visual_pilot():
    for motor,pilot in [('brain','none'),('pd','race-cue')]:
        with pytest.raises(ValueError,match='Oracle collection'):
            VisualController('unused','unused',motor_controller=motor,pilot_assistance=pilot,collection_route='unused')


@pytest.mark.parametrize('options',[{},dict(collection_route='unused',motor_controller='pd'),
                                    dict(collection_route='unused',pilot_assistance='race-cue')])
def test_brain_oracle_diagnostic_requires_explicit_brain_route_without_visual_pilot(options):
    with pytest.raises(ValueError,match='Oracle motor diagnostic'):
        VisualController('unused','unused',oracle_motor_diagnostic=True,**options)


def test_collection_teacher_validates_start_and_reports_privilege(tmp_path):
    route=dict(schema='haltere.collection_route.v1',frame='unity_world_xyz_m',loop=False,
               waypoints_unity=[[0,2,0],[0,2,8]],expected_start_unity=[0,0,0],
               speed_mps=2.,lookahead_m=1.5,max_start_error_m=2.,max_start_speed_mps=.5)
    path=tmp_path/'route.json';path.write_text(json.dumps(route))
    assistance=OracleCollectionAssistance(path,reference_speed=2.5)
    with pytest.raises(ValueError,match='Route start'):
        assistance.bind(TelemetryFrame(position=np.array([20,0,0.])))
    assistance.bind(TelemetryFrame())
    senses=dict(pos=torch.zeros(1,3),quat=torch.tensor([[1.,0,0,0]]),vel_world=torch.zeros(1,3))
    goal,_=assistance.update(senses,np.zeros(3),None,None,1.)
    np.testing.assert_allclose(goal,[0,0,1.2])
    meta=assistance.metadata()
    assert meta['runtime_route_oracle'] and not meta['autonomous_evaluation_eligible']
    assert meta['mode']=='oracle-route'


def test_oracle_runner_uses_route_goal_and_preserves_the_shadow_brain(checkpoint,tmp_path):
    checkpoint_path,mapping=checkpoint
    ck=torch.load(checkpoint_path,weights_only=True)
    ck['visual_brain']['gate_training']=dict(dynamics=dict(profile=dict(
        vertical_calibration=dict(mean=dict(hover_processed=0.,slope_mps2_per_processed=17.)))))
    torch.save(ck,checkpoint_path)
    path=tmp_path/'route.json'
    path.write_text(json.dumps(dict(schema='haltere.collection_route.v1',frame='unity_world_xyz_m',loop=False,
        waypoints_unity=[[0,2,0],[0,2,8]],expected_start_unity=[0,0,0],speed_mps=2.,
        lookahead_m=1.5,max_start_error_m=2.,max_start_speed_mps=.5)))
    controller=VisualController(checkpoint_path,mapping,'cpu',motor_controller='pd',collection_route=path)
    original={k:v.clone() for k,v in controller.state.items()}
    action,processed,raw=controller.step(TelemetryFrame(timestamp=1.,motor_rpm=np.ones(4)*1000.),
        torch.zeros(1,720),dict(p=.99,point=np.array([80,20,-5.]),width=3.),1.,1.,1.)
    np.testing.assert_allclose(controller.relative_gate,[0,0,1.2])
    with torch.no_grad():expected,_,_=controller.brain(controller.last_observation,original,controller.W)
    np.testing.assert_array_equal(action,expected[0].numpy())
    assert np.isfinite(np.r_[action,processed,raw]).all()
    assert controller.assistance_mode=='oracle-route'
    assert not controller.motor_metadata['brain_controls_motors']


def test_explicit_oracle_brain_diagnostic_commands_brain_and_keeps_privileged_label(checkpoint,tmp_path):
    checkpoint_path,mapping=checkpoint
    path=tmp_path/'route.json'
    path.write_text(json.dumps(dict(schema='haltere.collection_route.v1',frame='unity_world_xyz_m',loop=False,
        waypoints_unity=[[0,2,0],[0,2,8]],expected_start_unity=[0,0,0],speed_mps=1.,
        lookahead_m=1.5,max_start_error_m=2.,max_start_speed_mps=.5)))
    controller=VisualController(checkpoint_path,mapping,'cpu',motor_controller='brain',
                                collection_route=path,oracle_motor_diagnostic=True)
    action,_,_=controller.step(TelemetryFrame(timestamp=1.,motor_rpm=np.ones(4)*1000.),torch.zeros(1,720))
    np.testing.assert_array_equal(controller.last_command[:3],action[:3])
    assert controller.motor_metadata['brain_controls_motors'] and controller.motor_baseline is None
    meta=controller.assistance.metadata()
    assert meta['runtime_route_oracle'] and not meta['autonomous_evaluation_eligible']
    assert 'Brain throttle' in meta['motor_control']
    senses=dict(pos=torch.tensor([[0.,0.,2.]]),quat=torch.tensor([[1.,0,0,0]]),
                vel_world=torch.tensor([[.2,.3,.4]]),vel_body=torch.tensor([[.2,.3,.4]]))
    _,modified=controller.assistance.update(senses,np.zeros(3),None,None,2.)
    torch.testing.assert_close(modified['vel_world'],torch.tensor([[.4,.6,.4]]))
    torch.testing.assert_close(modified['vel_body'],modified['vel_world'])
    torch.testing.assert_close(senses['vel_world'],torch.tensor([[.2,.3,.4]]))


def test_declared_route_settling_requires_continuous_low_speed(tmp_path):
    path=tmp_path/'route.json'
    path.write_text(json.dumps(dict(schema='haltere.collection_route.v1',frame='unity_world_xyz_m',loop=False,
        waypoints_unity=[[0,2,0],[0,2,2]],expected_start_unity=[0,0,0],speed_mps=2.,
        lookahead_m=1.5,max_start_error_m=2.,max_start_speed_mps=.5,finish_speed_mps=.4,finish_hold_s=2.)))
    pilot=OracleCollectionAssistance(path,reference_speed=2.)
    pilot.bind(TelemetryFrame())
    senses=dict(pos=torch.tensor([[0.,0.,2.]]),quat=torch.tensor([[1.,0,0,0]]),vel_world=torch.zeros(1,3))
    pilot.update(senses,np.zeros(3),None,None,-1.)
    pilot.update(senses,np.zeros(3),None,None,0.)
    senses['pos'][:,0]=2.
    pilot.update(senses,np.zeros(3),None,None,1.)
    assert not pilot.complete
    senses['vel_world'][:,0]=1.
    pilot.update(senses,np.zeros(3),None,None,3.)
    assert not pilot.complete and pilot.stable_since is None
    senses['vel_world'].zero_()
    pilot.update(senses,np.zeros(3),None,None,4.)
    pilot.update(senses,np.zeros(3),None,None,5.9)
    assert not pilot.complete
    pilot.update(senses,np.zeros(3),None,None,6.)
    assert pilot.complete


def test_route_launch_holds_heading_and_accepts_a_measured_hover_volume(tmp_path):
    path=tmp_path/'route.json'
    path.write_text(json.dumps(dict(schema='haltere.collection_route.v1',frame='unity_world_xyz_m',loop=False,
        waypoints_unity=[[0,6,0],[0,6,8]],expected_start_unity=[0,0,0],speed_mps=2.,
        lookahead_m=3.,max_start_error_m=2.,max_start_speed_mps=.5)))
    pilot=OracleCollectionAssistance(path,reference_speed=2.)
    pilot.bind(TelemetryFrame())
    senses=dict(pos=torch.tensor([[.3,0.,5.75]]),quat=torch.tensor([[1.,0,0,0]]),vel_world=torch.tensor([[.2,0,0]]))
    goal,_=pilot.update(senses,np.zeros(3),None,None,0.)
    # Returning toward launch must not point the camera backwards and spin.
    assert abs(pilot.pilot.sight_yaw)<1e-8
    np.testing.assert_allclose(goal,[-.54,0,.25],atol=1e-7)
    pilot.update(senses,np.zeros(3),None,None,.9)
    assert pilot.launching
    pilot.update(senses,np.zeros(3),None,None,1.)
    assert not pilot.launching
    assert not pilot.metadata()['autonomous_evaluation_eligible']


@pytest.mark.parametrize('fps',[0,11,61,float('nan')])
def test_invalid_camera_rate_fails_before_model_loading(fps):
    from haltere.liftoff.visual_brain import run
    with pytest.raises(ValueError,match='Camera rate'):
        run(SimpleNamespace(seconds=1.,max_height=8.,max_speed=5.,max_distance=20.,camera_fps=fps))
