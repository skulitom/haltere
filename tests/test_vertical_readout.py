import copy

import pytest
import torch

from haltere.train.vertical_readout import fit_readout,replace_throttle_readout


def test_static_fit_recovers_vertical_response_on_unseen_sensory_states():
    g=torch.Generator().manual_seed(518)
    X=torch.cat((torch.randn(80,5,generator=g),torch.ones(80,1)),1)
    V=torch.cat((torch.randn(30,5,generator=g),torch.ones(30,1)),1)
    target=torch.tensor([.1,-.3,.05,.15,-.08,-.4])
    prior=torch.tensor([.1,-.6,.05,.15,-.08,-.4])
    weights,report=fit_readout(X,torch.tanh(X@target),V,torch.tanh(V@target),prior)
    assert report['validation_rmse']<report['baseline_validation_rmse']/10
    assert torch.allclose(torch.tanh(V.double()@weights),torch.tanh(V@target).double(),atol=.01)


def test_export_keeps_graph_and_every_other_motor_row_exact():
    checkpoint=dict(model={'readout.weight':torch.randn(4,7),'readout.bias':torch.randn(4),
                           'edge_index':torch.tensor([[0,1],[1,2]]),'sign_fixed':torch.tensor([1.,-1.]),
                           'log_edge_gain':torch.randn(2)},visual_brain={'qualified':False})
    original=copy.deepcopy(checkpoint)
    weights=torch.randn(8,dtype=torch.float64)
    exported=replace_throttle_readout(checkpoint,weights)
    for key,value in checkpoint['model'].items():
        assert torch.equal(value,original['model'][key])
        if key.startswith('readout.'):
            assert torch.equal(value[1:],exported['model'][key][1:])
        else:
            assert torch.equal(value,exported['model'][key])
    assert torch.equal(exported['model']['readout.weight'][0],weights[:-1].float())
    with pytest.raises(ValueError,match='Invalid'):
        replace_throttle_readout(checkpoint,torch.full((8,),float('nan')))
