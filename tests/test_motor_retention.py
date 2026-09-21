import torch
from haltere.train.motor_retention import outside_basis,constrain_readout


def test_projected_update_preserves_measured_features_and_can_change_new_ones():
    basis=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    delta=torch.tensor([[3.,-4.,2.]])
    projected=outside_basis(delta,basis)
    assert torch.equal(projected@basis.T,torch.zeros(1,2))
    assert projected[0,2]==2


def test_readout_norm_bound_does_not_break_prefix_constraint():
    basis=torch.tensor([[1.,0.,0.]])
    parent=torch.tensor([[1.,2.,3.],[-2.,1.,1.]])
    weight=parent+torch.tensor([[2.,4.,6.],[-1.,5.,4.]])
    constrain_readout(weight,parent,basis,.2)
    delta=weight-parent
    torch.testing.assert_close(delta@basis.T,torch.zeros(2,1))
    assert torch.all(delta.norm(dim=1)<=.200001*parent.norm(dim=1))
