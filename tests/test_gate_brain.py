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
