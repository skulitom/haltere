import copy
import pytest
import torch
from haltere.vision.model import GateNet
from haltere.vision.scene_features import scene_map,fit_projection,project_scene


def test_observed_sticks_never_enter_scene_convolutions():
    torch.manual_seed(83);net=GateNet(4).eval()
    pixels=torch.rand(1,3,180,320);other=pixels.clone()
    other[...,round(.84*180):,round(.40*320):round(.60*320)]=1.
    with torch.no_grad():
        actual=scene_map(net,pixels);masked=scene_map(net,other)
    assert torch.equal(actual,masked)
    obstacle=pixels.clone();obstacle[...,150:175,210:220]=0.
    with torch.no_grad():changed=scene_map(net,obstacle)
    assert not torch.equal(actual,changed)


def test_frozen_projection_preserves_spatial_information_and_rejects_wrong_contract():
    torch.manual_seed(18);training=torch.randn(12,16,6,10)
    projection=fit_projection(training);before=copy.deepcopy(projection)
    heldout=torch.randn(3,16,6,10)
    output=project_scene(heldout,projection)
    assert output.shape==(3,720) and output.abs().max()<=1
    assert projection==before
    shifted=project_scene(heldout.roll(1,-1),projection).reshape(3,12,6,10)
    assert torch.equal(shifted,output.reshape(3,12,6,10).roll(1,-1))
    with pytest.raises(ValueError,match='contract'):
        project_scene(heldout,{**projection,'grid':[12,20]})
