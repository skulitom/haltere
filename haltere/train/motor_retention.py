"""Constrain a motor-readout refit on measured successful parent activity."""
import torch


@torch.no_grad()
def prefix_motor_basis(brain,replay,relative_threshold=1e-5):
    features=[]
    for k,(entry,data) in enumerate(replay.takes):
        if entry['correction_ticks']!=0:continue
        lookup,states,_=replay.states[k]
        rows=[row for start,row in lookup.items() if start<len(data['action'])]
        v=states[rows].to(brain.device)[:,brain.motor_idx]
        m=brain.cfg.rate_max*torch.sigmoid(v)
        if brain.motor_norm is None:raise ValueError('Expected fixed motor normalization')
        bn=brain.motor_norm
        features.append((m-bn.running_mean)/torch.sqrt(bn.running_var+bn.eps))
    if not features:raise ValueError('No measured successful-prefix states')
    x=torch.cat(features)
    _,singular,basis=torch.linalg.svd(x,full_matrices=False)
    basis=basis[singular>singular[0]*relative_threshold]
    return basis,singular


def outside_basis(value,basis):
    """Remove components that change the sampled successful motor activity."""
    return value-(value@basis.T)@basis


@torch.no_grad()
def constrain_readout(weight,parent,basis,max_relative_norm=.5):
    delta=outside_basis(weight-parent,basis)
    limit=parent.norm(dim=1,keepdim=True)*max_relative_norm
    delta*=torch.clamp(limit/delta.norm(dim=1,keepdim=True).clamp_min(1e-12),max=1.)
    weight.copy_(parent+delta)

