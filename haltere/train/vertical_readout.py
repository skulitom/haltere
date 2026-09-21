"""Calibrate the neural throttle readout using training-only vertical targets.

The already-trained recurrent connectome supplies every readout feature. Only
the throttle row changes in this stage. Static fitting is not flight evidence;
the exported candidate still needs teacher-free closed-loop and live checks.
"""
import argparse,copy,json
from pathlib import Path
import torch
from haltere.train.bptt import load_checkpoint
from haltere.train.evaluate_gate_brain import sensory_contract
from haltere.train.gate_brain import teacher_hover_command
from haltere.brain.gate_senses import gate_observation
from haltere.brain.retina import RETINA_DIM
from haltere.sim.quad import QuadSim,QuadState,quat_from_euler,quat_to_mat
from haltere.vision.datasets import sha256


def fit_readout(X,y,V,vy,prior,tolerance=.2):
    """Regularized residual fit; preserve the prior where data adds no evidence."""
    if not 0<=tolerance<=.5:
        raise ValueError('Validation tolerance must be in [0, .5]')
    X,y,V,vy,prior=(v.detach().cpu().double() for v in (X,y,V,vy,prior))
    if (X.ndim!=2 or V.ndim!=2 or X.shape[1]!=V.shape[1] or prior.shape!=(X.shape[1],)
            or y.shape!=(len(X),) or vy.shape!=(len(V),) or min(len(X),len(V))<2):
        raise ValueError('Incompatible training, validation and prior shapes')
    if not all(torch.isfinite(v).all() for v in (X,y,V,vy,prior)) or y.abs().max()>=1 or vy.abs().max()>=1:
        raise ValueError('Expected finite features and targets strictly inside (-1, 1)')
    residual=torch.atanh(y)-X@prior
    baseline=float((torch.tanh(V@prior)-vy).square().mean().sqrt())
    gram=X@X.T;rows=[];weights=[]
    for ridge in (.01,.1,1.,10.,100.,1000.):
        delta=X.T@torch.linalg.solve(gram+ridge*torch.eye(len(X)),residual)
        w=prior+delta;weights.append(w)
        error=float((torch.tanh(V@w)-vy).square().mean().sqrt())
        rows.append(dict(ridge=ridge,validation_rmse=error,weight_change=float(delta.abs().max())))
    limit=min(r['validation_rmse'] for r in rows)*(1+tolerance)
    index=max(i for i,r in enumerate(rows) if r['validation_rmse']<=limit)
    return weights[index],dict(baseline_validation_rmse=baseline,**rows[index],candidates=rows,
                               selection='strongest ridge within relative tolerance of lowest validation RMSE',
                               validation_tolerance=tolerance)


def replace_throttle_readout(checkpoint,weights):
    """Leave every recurrent/sensory weight, graph buffer and other motor row exact."""
    changed=copy.deepcopy(checkpoint)
    model=changed['model'];expected=model['readout.weight'].shape[1]+1
    if weights.shape!=(expected,) or not torch.isfinite(weights).all():
        raise ValueError('Invalid throttle readout weights')
    model['readout.weight'][0]=weights[:-1].to(model['readout.weight'])
    model['readout.bias'][0]=weights[-1].to(model['readout.bias'])
    return changed


@torch.no_grad()
def features(brain,cfg,contract,count,seed,velocity_gain):
    torch.manual_seed(seed)
    sim=QuadSim(cfg.quad,brain.device);W=brain.weight_matrix();captured={}
    hook=brain.readout.register_forward_pre_hook(lambda m,args:captured.update(x=args[0].detach()))
    xx,yy=[],[]
    for start in range(0,count,64):
        B=min(64,count-start);rand=lambda *shape:torch.rand(*shape,device=brain.device)
        q=QuadState.hover(B,brain.device,15.)
        roll,pitch=(rand(B)*2-1)*.15,(rand(B)*2-1)*.15
        yaw=(rand(B)*2-1)*torch.pi
        q.quat=quat_from_euler(roll,pitch,yaw);R=quat_to_mat(q.quat)
        vb=torch.stack((rand(B)*3,(rand(B)*2-1)*.6,(rand(B)*2-1)*1.5),-1)
        q.vel=(R@vb[...,None]).squeeze(-1)
        q.omega=(rand(B,3)*2-1)*.12
        height=(rand(B)*2-1)*4.;angle=(rand(B)*2-1)*1.2+yaw
        distance=5.+rand(B)*25.
        world=torch.stack((distance*angle.cos(),distance*angle.sin(),height),-1)
        relative=(R.transpose(-1,-2)@world[...,None]).squeeze(-1)
        missing=rand(B)<.2;relative[missing]=0.
        error=(rand(B)*2-1)*1.5
        cue=torch.where(missing,error,0.)[:,None] if contract['search_height_anchor'] else None
        obs=gate_observation(sim.sensors(q,noise=False),q.motor.mean(-1,keepdim=True),cfg.task,
                             torch.zeros(B,RETINA_DIM,device=brain.device),relative,
                             contract['height_invariant'],contract['gravity_aligned_height'],cue)
        state=brain.init_state(B)
        for _ in range(160):_,state,_=brain(obs,state,W)
        desired=height.clamp(-1.2,1.2)
        desired=torch.where(missing,(-.6*error).clamp(-1.2,1.2),desired)
        target=(2*teacher_hover_command(sim,R[:,2,2])-1+velocity_gain*(desired-q.vel[:,2])).clamp(-.8,.2)
        xx.append(torch.cat((captured['x'],torch.ones(B,1,device=brain.device)),-1).cpu())
        yy.append(target.cpu())
        print('features',seed,start+B,flush=True)
    hook.remove()
    return torch.cat(xx),torch.cat(yy)

def run(checkpoint,out,velocity_gain=.3):
    if not 0<velocity_gain<=.5:
        raise ValueError('Use 0 < vertical velocity gain <= .5')
    torch.set_num_threads(2);out=Path(out)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True);meta=ck['visual_brain']
    contract=sensory_contract(meta)
    if not all(contract[k] for k in ('height_invariant','gravity_aligned_height','search_height_anchor')):
        raise ValueError('Requires the trained height-invariant, gravity-aligned, search-height contract')
    if ck['iter']<meta['gate_training']['iterations']:
        raise ValueError('Parent training is incomplete')
    brain,cfg,_=load_checkpoint(checkpoint,'cuda');brain.requires_grad_(False)
    if not cfg.brain.mask_motor_feedback:
        raise ValueError('Motor feedback must be masked in the exported brain')
    if sha256(Path(cfg.train.graph).with_suffix('.npz'))!=meta['graph_sha256']:
        raise ValueError('Parent connectome fingerprint differs from loaded graph')
    out.mkdir(parents=True,exist_ok=False)
    X,y=features(brain,cfg,contract,1024,11917,velocity_gain);V,vy=features(brain,cfg,contract,256,32941,velocity_gain)
    torch.save(dict(X=X,y=y,V=V,vy=vy),out/'features.pt')
    prior=torch.cat((brain.readout.weight[0].detach().cpu(),brain.readout.bias[:1].detach().cpu())).double()
    w,fit=fit_readout(X,y,V,vy,prior)
    changed=replace_throttle_readout(ck,w)
    for k,v in ck['model'].items():
        if k not in ('readout.weight','readout.bias'):assert torch.equal(v,changed['model'][k])
    assert torch.equal(ck['model']['readout.weight'][1:],changed['model']['readout.weight'][1:])
    assert torch.equal(ck['model']['readout.bias'][1:],changed['model']['readout.bias'][1:])
    provenance=dict(parent_sha256=sha256(checkpoint),method='ridge residual on normalized motor-neuron activity',
                    training_samples=1024,validation_samples=256,training_seed=11917,validation_seed=32941,
                    teacher='calibrated nominal thrust law and measured relative height/vertical speed; training only',
                    changed_weights=['readout.weight[0]','readout.bias[0]'],recurrent_weights_unchanged=True,
                    graph_unchanged=True,velocity_gain=velocity_gain,
                    training_code_sha256=sha256(__file__),features_sha256=sha256(out/'features.pt'),
                    synthetic_static_senses=True,release_qualified=False,**fit)
    previous=changed['visual_brain'].pop('perception_rebind',None)
    if previous:
        provenance['parent_perception_rebind']=previous
    changed['visual_brain']['vertical_readout_training']=provenance
    changed['visual_brain']['qualified']=False
    torch.save(changed,out/'candidate.pt');(out/'evaluation.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('--out',required=True)
    p.add_argument('--velocity-gain',type=float,default=.3)
    a=p.parse_args();run(a.checkpoint,a.out,a.velocity_gain)
