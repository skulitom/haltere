import torch

from haltere.brain.gate_senses import aperture_crossing, gate_observation
from haltere.brain.retina import RETINA_DIM
from haltere.sim.tasks import HoverTaskConfig


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


def test_nearby_gate_can_be_remembered_briefly_but_never_indefinitely():
    import numpy as np
    from haltere.liftoff.visual_brain import gate_memory_valid
    assert gate_memory_valid(1.5,np.array([3.,.2,.1]))
    assert not gate_memory_valid(2.01,np.array([3.,.2,.1]))
    assert not gate_memory_valid(.6,np.array([20.,.2,.1]))
    assert not gate_memory_valid(.6,np.array([-2.,.2,.1]))
    assert not gate_memory_valid(-.1,np.array([3.,.2,.1]))


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
