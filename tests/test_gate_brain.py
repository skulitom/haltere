import torch
import pytest

from haltere.brain.gate_senses import aperture_crossing, gate_observation
from haltere.brain.retina import RETINA_DIM
from haltere.sim.tasks import HoverTaskConfig


def test_evaluation_uses_checkpoint_sensory_contract_without_changing_legacy_defaults():
    from haltere.train.evaluate_gate_brain import sensory_contract
    meta=dict(runtime_requires_teacher=False,gate_sensor=dict(focal_320=100.,tilt_deg=30.,
                                                             missing_gate='zero_goal_neural_search'))
    assert sensory_contract(meta)==dict(height_invariant=False,centre_offset=1.5,gravity_aligned_height=False,search_height_anchor=False)
    meta['gate_sensor'].update(height_invariant=True,centre_offset_m=0.,gravity_aligned_height=True,search_height_anchor=True)
    assert sensory_contract(meta)==dict(height_invariant=True,centre_offset=0.,gravity_aligned_height=True,search_height_anchor=True)
    meta['runtime_requires_teacher']=True
    with pytest.raises(ValueError,match='teacher-free'):
        sensory_contract(meta)
    meta['runtime_requires_teacher']=False
    meta['gate_sensor']['tilt_deg']=20.
    with pytest.raises(ValueError,match='calibration'):
        sensory_contract(meta)


def test_crossing_requires_forward_plane_intersection_inside_aperture():
    before = torch.tensor([[0.,0.,1.5],[0.,3.,1.5],[2.,0.,1.5],[0.,0.,4.],[0.,0.,1.5]])
    after = torch.tensor([[2.,0.,1.5],[2.,3.,1.5],[0.,0.,1.5],[2.,0.,4.],[.9,0.,1.5]])
    centre = torch.tensor([1.,0.,1.5]).expand_as(before)
    passed,point = aperture_crossing(before,after,centre)
    assert passed.tolist() == [True,False,False,False,False]
    assert torch.allclose(point[0],centre[0])


def test_gate_senses_keep_vertical_error_and_ignore_world_coordinates():
    s=dict(gyro=torch.zeros(1,3),gravity_body=torch.tensor([[0.,0.,-1.]]),vel_body=torch.zeros(1,3),
           vel_world=torch.zeros(1,3),pos=torch.tensor([[100.,999.,1.]]),quat=torch.tensor([[1.,0.,0.,0.]]),
           up=torch.ones(1),altitude=torch.ones(1,1),yaw=torch.ones(1,1))
    task=HoverTaskConfig()
    obs=gate_observation(s,torch.zeros(1,1),task,torch.zeros(1,RETINA_DIM),torch.tensor([[30.,0.,2.]]))
    assert torch.allclose(obs['goal'][0,:3],torch.tanh(torch.tensor([3.,0.,2.])/task.goal_scale))
    assert torch.count_nonzero(obs['compass'])==0
    other=gate_observation({**s,'pos':torch.zeros(1,3)},torch.zeros(1,1),task,torch.zeros(1,RETINA_DIM),torch.tensor([[30.,0.,2.]]))
    assert torch.equal(obs['goal'],other['goal'])
    pixels=gate_observation(s,torch.zeros(1,1),task,torch.ones(1,RETINA_DIM),torch.tensor([[30.,0.,2.]]))
    assert torch.count_nonzero(pixels['retina'])==0


def test_height_invariant_contract_removes_launch_elevation_from_every_channel():
    s=dict(gyro=torch.ones(1,3),gravity_body=torch.tensor([[0.,0.,-1.]]),vel_body=torch.ones(1,3),
           vel_world=torch.ones(1,3),pos=torch.tensor([[0.,0.,1.]]),quat=torch.tensor([[1.,0.,0.,0.]]),
           up=torch.ones(1),altitude=torch.ones(1,1),yaw=torch.ones(1,1))
    higher={**s,'pos':s['pos']+torch.tensor([0.,0.,20.]),'altitude':s['altitude']+20.}
    args=(torch.zeros(1,1),HoverTaskConfig(),torch.zeros(1,RETINA_DIM),torch.tensor([[20.,0.,4.]]))
    low=gate_observation(s,*args,height_invariant=True)
    high=gate_observation(higher,*args,height_invariant=True)
    assert all(torch.equal(low[k],high[k]) for k in low)
    # The legacy sensory contract is retained for existing checkpoints.
    legacy=gate_observation(s,*args)
    legacy_high=gate_observation(higher,*args)
    assert not torch.equal(legacy['lptc'],legacy_high['lptc'])


def test_visibility_uses_the_configured_opening_offset():
    from haltere.train.gate_brain import gate_in_view
    from haltere.vision.camera import body_to_cam_matrix
    camera=torch.tensor(body_to_cam_matrix(30.),dtype=torch.float32)
    relative=torch.tensor([[4.,0.,-1.]])
    rotation=torch.eye(3)[None]
    assert gate_in_view(relative,rotation,camera,1.5).item()
    assert not gate_in_view(relative,rotation,camera,0.).item()


def test_search_height_reference_is_latched_and_restarts_after_reacquisition():
    from tests.test_human_brain import small_brain
    from haltere.train.gate_brain import GateRollout
    brain,cfg,_=small_brain()
    r=GateRollout(brain,cfg,batch=3,turns=True,search=True,height_invariant=True,search_height_anchor=True)
    r.searching.zero_();r.measurement_age.fill_(float('inf'));r.vs.quad.pos[:,2]=5.
    hidden=torch.tensor([[-20.,0.,0.]]).repeat(3,1);rotation=torch.eye(3)[None].repeat(3,1,1)
    r.camera_measurement(hidden,rotation)
    assert torch.equal(r.search_height,torch.full((3,),5.))
    r.vs.quad.pos[:,2]=6.;r.camera_measurement(hidden,rotation)
    assert torch.equal(r.search_height_cue(),torch.ones(3,1))
    r.camera_measurement(-hidden,rotation)
    assert torch.count_nonzero(r.search_height_cue())==0
    r.vs.quad.pos[:,2]=7.;r.measurement_age.fill_(float('inf'));r.camera_measurement(hidden,rotation)
    assert torch.equal(r.search_height,torch.full((3,),7.))
    assert torch.count_nonzero(r.search_height_cue())==0


def test_gravity_aligned_gate_scaling_does_not_invent_height_when_pitching():
    from haltere.sim.quad import quat_from_euler,quat_to_mat
    angle=torch.tensor([.25]);q=quat_from_euler(angle*.3,angle,angle*2)
    R=quat_to_mat(q)
    up=R[:,2,:]
    s=dict(gyro=torch.zeros(1,3),gravity_body=-up,vel_body=torch.zeros(1,3),vel_world=torch.zeros(1,3),
           pos=torch.tensor([[0.,0.,10.]]),quat=q,up=R[:,2,2],altitude=torch.full((1,1),10.),yaw=(angle*2)[:,None])
    task=HoverTaskConfig()
    for distance in (5.,20.,40.):
        for height in (0.,2.):
            world=torch.tensor([[distance,0.,height]])
            body=(R.transpose(-1,-2)@world[...,None]).squeeze(-1)
            obs=gate_observation(s,torch.zeros(1,1),task,torch.zeros(1,RETINA_DIM),body,
                                  height_invariant=True,gravity_aligned_height=True)
            encoded=torch.atanh(obs['goal'][:,:3])*task.goal_scale
            actual=(R@encoded[...,None]).squeeze(-1)
            torch.testing.assert_close(actual,torch.tensor([[3.,0.,height]]),atol=1e-5,rtol=1e-5)


def test_elevation_training_includes_climbs_and_descents_and_updates_brain():
    from tests.test_human_brain import small_brain
    from haltere.train.gate_brain import GateRollout
    brain,cfg,_=small_brain()
    rollout=GateRollout(brain,cfg,batch=12,evaluation=True,teacher=brain,turns=True,search=True,
                        elevation=True,height_invariant=True,centre_offset=0.,gravity_aligned_height=True,
                        throttle_anchor=4.,height_gain=1.,search_height_anchor=True)
    delta=rollout.gate[:,2]-rollout.vs.quad.pos[:,2]
    assert (delta>4.).any() and (delta< -4.).any()
    assert rollout.vs.quad.pos[:,2].max()>=28.
    loss=rollout.window(12)
    loss.backward()
    assert torch.isfinite(loss)
    assert brain.log_edge_gain.grad.abs().sum()>0
    assert brain.readout.weight.grad.abs().sum()>0


def test_vertical_recovery_exposes_airborne_training_states_without_changing_evaluation():
    from tests.test_human_brain import small_brain
    from haltere.train.gate_brain import GateRollout
    brain,cfg,_=small_brain()
    args=dict(batch=12,turns=True,search=True,elevation=True,vertical_recovery_speed=1.2)
    r=GateRollout(brain,cfg,**args)
    assert torch.count_nonzero(r.vs.quad.vel[:4,2])==0
    assert torch.count_nonzero(r.vs.quad.vel[4:,2])==8
    assert r.vs.quad.vel[:,2].abs().max()<=1.2
    evaluated=GateRollout(brain,cfg,evaluation=True,**args)
    assert torch.count_nonzero(evaluated.vs.quad.vel[:,2])==0


def test_hover_teacher_balances_simulated_gravity_with_randomized_thrust_and_tilt():
    from haltere.train.gate_brain import teacher_hover_command
    from haltere.sim.quad import QuadSim,QuadParams,QuadState,quat_from_euler
    sim=QuadSim(QuadParams(),torch.device('cpu'),substeps=1)
    sim.randomize(3,.1)
    sim.thrust_exp=torch.tensor([[.768],[1.5],[2.]])
    sim.twr=torch.tensor([2.69,3.21,6.])
    pitch=torch.tensor([0.,.2,-.4])
    st=QuadState.hover(3,'cpu',3.)
    st.quat=quat_from_euler(pitch*0,pitch,pitch*0)
    command=teacher_hover_command(sim,pitch.cos())
    st.motor=command[:,None].expand(3,4).clone()
    # Check the resulting simulator acceleration, not the inverse formula.
    after=sim.step(st,st.motor)
    torch.testing.assert_close(after.vel[:,2],torch.zeros(3),atol=2e-6,rtol=0)
    assert torch.count_nonzero(after.vel[1:,:2])>0


def test_nearby_gate_can_be_remembered_briefly_but_never_indefinitely():
    import numpy as np
    from haltere.liftoff.visual_brain import gate_memory_valid
    assert gate_memory_valid(1.5,np.array([3.,.2,.1]))
    assert not gate_memory_valid(2.01,np.array([3.,.2,.1]))
    assert not gate_memory_valid(.6,np.array([20.,.2,.1]))
    assert not gate_memory_valid(.6,np.array([-2.,.2,.1]))
    assert not gate_memory_valid(-.1,np.array([3.,.2,.1]))


def test_expired_camera_memory_is_zeroed_only_for_search_trained_brains():
    import numpy as np
    import pytest
    from haltere.liftoff.visual_brain import gate_measurement_or_search
    old=np.array([20.,5.,1.])
    with pytest.raises(RuntimeError,match='unavailable'):
        gate_measurement_or_search(1.,old)
    measurement,searching=gate_measurement_or_search(1.,old,allow_search=True)
    assert searching and np.count_nonzero(measurement)==0
    measurement,searching=gate_measurement_or_search(.1,old,allow_search=True)
    assert not searching and np.array_equal(measurement,old)


def test_close_gate_outlier_does_not_refresh_or_move_sensory_memory():
    import numpy as np
    from haltere.liftoff.visual_brain import fuse_gate_detection
    previous=np.array([3.,.1,1.2]); jumped=np.array([8.,4.,.5]); position=np.array([1.,0.,1.])
    point,stamp=fuse_gate_detection(previous,10.,jumped,10.1,position,np.eye(3))
    np.testing.assert_array_equal(point,previous)
    assert stamp==10.
    # A bad detector cannot preserve an old target forever.
    point,stamp=fuse_gate_detection(previous,10.,jumped,12.1,position,np.eye(3))
    np.testing.assert_array_equal(point,jumped)
    assert stamp==12.1
    # After passage, do not average two distinct gates into a phantom target.
    point,stamp=fuse_gate_detection(previous,10.,jumped,10.1,np.array([4.,0.,1.]),np.eye(3))
    np.testing.assert_array_equal(point,jumped)
    assert stamp==10.1


def test_close_gate_association_is_invariant_to_global_coordinates():
    import numpy as np
    from haltere.liftoff.visual_brain import fuse_gate_detection
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    shift=np.array([999.,100.,2.])
    old=np.array([3.,0.,1.]); measured=np.array([3.2,.1,1.2]); position=np.array([1.,0.,1.])
    point,stamp=fuse_gate_detection(old,1.,measured,1.1,position,np.eye(3))
    transformed,t=fuse_gate_detection(R@old+shift,1.,R@measured+shift,1.1,R@position+shift,R)
    np.testing.assert_allclose(transformed,R@point+shift)
    assert t==stamp


def test_short_camera_gap_discards_image_and_long_gap_stops():
    import pytest
    from haltere.liftoff.visual_brain import camera_measurement_fresh
    assert camera_measurement_fresh(.1,True)
    assert not camera_measurement_fresh(.14,True)
    assert not camera_measurement_fresh(.25,True)
    for age,memory in [(.251,True),(.121,False),(-.1,True),(float('nan'),True)]:
        with pytest.raises(RuntimeError,match='unavailable'):
            camera_measurement_fresh(age,memory)
    with pytest.raises(RuntimeError,match='hidden'):
        camera_measurement_fresh(.01,True,foreground=False)


def test_search_training_hides_unseen_gate_and_updates_recurrent_weights():
    from tests.test_human_brain import small_brain
    from haltere.train.gate_brain import GateRollout
    brain,cfg,_=small_brain()
    rollout=GateRollout(brain,cfg,batch=3,turns=True,search=True)
    rollout.measurement_age[:]=float('inf')
    relative=torch.tensor([[-10.,0.,0.],[10.,0.,0.],[0.,10.,0.]])
    measured=rollout.camera_measurement(relative,torch.eye(3)[None].expand(3,-1,-1))
    assert torch.count_nonzero(measured[[0,2]])==0
    assert torch.equal(measured[1],relative[1])
    loss=rollout.window(12)
    loss.backward()
    assert torch.isfinite(loss)
    assert brain.log_edge_gain.grad.abs().sum()>0
    assert brain.readout.weight.grad.abs().sum()>0


def test_frozen_cpu_inference_preserves_neural_dynamics_and_cannot_train():
    import pytest
    from tests.test_human_brain import small_brain
    brain,_,_=small_brain()
    csr=brain.inference_matrix()
    assert csr.layout==torch.sparse_csr
    a,b=brain.init_state(2),brain.init_state(2)
    values=brain.weight_matrix().detach()
    with torch.no_grad():
        for _ in range(40):
            obs={k:torch.randn(2,d) for k,d in brain.channel_dims.items()}
            expected,a,_=brain(obs,a,values)
            actual,b,_=brain(obs,b,csr)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(b['v'],a['v'],rtol=1e-5,atol=1e-6)
    with pytest.raises(ValueError,match='cannot train'):
        brain(obs,b,csr)


def test_turn_cost_and_aperture_are_invariant_to_world_heading():
    from haltere.train.gate_brain import turn_objective
    from haltere.sim.quad import quat_from_euler, quat_to_mat
    pos=torch.tensor([[0.,1.,1.5]])
    vel=torch.tensor([[1.,.4,.2]])
    gate=torch.tensor([[5.,4.,1.4]])
    normal=torch.tensor([[.8,.6,0.]])
    R=torch.eye(3)[None]
    yaw=torch.tensor([1.7])
    Q=quat_to_mat(quat_from_euler(yaw*0,yaw*0,yaw))
    rotate=lambda v:(Q@v[...,None]).squeeze(-1)
    crossed=torch.tensor([False])
    cost=turn_objective(pos,vel,R,gate,normal,crossed)
    other=turn_objective(rotate(pos),rotate(vel),Q@R,rotate(gate),rotate(normal),crossed)
    assert torch.allclose(cost,other,atol=1e-5)
    before,after=gate-normal,gate+normal
    passed,point=aperture_crossing(before,after,gate,normal=normal)
    turned,turned_point=aperture_crossing(rotate(before),rotate(after),rotate(gate),normal=rotate(normal))
    assert passed.item() and turned.item()
    assert torch.allclose(rotate(point),turned_point,atol=1e-5)


def test_turn_training_updates_the_recurrent_brain():
    from tests.test_human_brain import small_brain
    from haltere.train.gate_brain import GateRollout
    brain,cfg,_=small_brain()
    rollout=GateRollout(brain,cfg,batch=3,turns=True)
    loss=rollout.window(12)
    loss.backward()
    assert torch.isfinite(loss)
    assert brain.log_edge_gain.grad.abs().sum()>0
    assert brain.readout.weight.grad.abs().sum()>0


def test_offline_track_evaluation_interpolates_tilted_planes(tmp_path):
    import numpy as np
    from haltere.liftoff.gate_evaluation import track_planes, plane_intersections
    xml=tmp_path/'track.xml'
    xml.write_text('<Track><blueprints><TrackBlueprint><itemID>AirgateExample</itemID>'
                   '<instanceID>7</instanceID><position><x>2</x><y>3</y><z>4</z></position>'
                   '<rotation><x>15</x><y>270</y><z>5</z></rotation>'
                   '</TrackBlueprint></blueprints></Track>')
    gate=track_planes(xml,np.zeros(3))[0]
    np.testing.assert_allclose(gate['base'],[4,-2,3])
    point=gate['base']+.4*gate['side']+1.2*gate['up']
    p=np.stack((point-2*gate['normal'],point+gate['normal']))
    result=plane_intersections(p,[10.,13.],[gate])
    assert len(result)==1
    assert abs(result[0]['seconds']-12.)<1e-6
    assert abs(result[0]['lateral_m']-.4)<1e-6
    assert abs(result[0]['height_above_base_m']-1.2)<1e-6
    assert plane_intersections(p[::-1],[10.,13.],[gate])==[]
    reverse=plane_intersections(p[::-1],[10.,13.],[gate],both_directions=True)
    assert len(reverse)==1 and reverse[0]['normal_direction']==-1
    assert abs(reverse[0]['seconds']-11.)<1e-6
    assert abs(reverse[0]['height_above_base_m']-1.2)<1e-6


def test_camera_pose_uses_capture_time_and_handles_quaternion_sign_flip():
    import numpy as np
    from haltere.liftoff.camera_pose import CameraPoseHistory
    history=CameraPoseHistory()
    history.append(10.,[0.,0.,0.],[1.,0.,0.,0.])
    history.append(11.,[2.,0.,0.],[-2**-.5,0.,0.,-2**-.5])
    pos,q=history.at(10.5)
    np.testing.assert_allclose(pos,[1.,0.,0.])
    np.testing.assert_allclose(q,[np.cos(np.pi/8),0.,0.,np.sin(np.pi/8)],atol=1e-7)
    np.testing.assert_allclose(history.at(9.)[0],[0.,0.,0.])
    np.testing.assert_allclose(history.at(12.)[0],[2.,0.,0.])


def test_dynamics_calibration_preserves_brain_and_task_and_rejects_other_overrides():
    import pytest
    from haltere.train.bptt import ExperimentConfig
    from haltere.train.gate_brain import calibrated_dynamics
    cfg=ExperimentConfig.from_dict({})
    new=calibrated_dynamics(cfg,{'overrides':{'rates':{'expo':[.3,.3,.3]},'quad':{'drag_lin':[.01,.01,.1]}}})
    assert new.rates.expo==(.3,.3,.3)
    assert cfg.rates.expo==(.0,.0,.0)
    assert new.brain==cfg.brain and new.task==cfg.task
    with pytest.raises(ValueError):
        calibrated_dynamics(cfg,{'overrides':{'brain':{'dt':.02}}})
    with pytest.raises(ValueError):
        calibrated_dynamics(cfg,{'overrides':{'quad':{'misspelled_drag':.01}}})
