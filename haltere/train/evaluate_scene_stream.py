"""Continuous, teacher-absent student replay plus offline diagnostic targets.

This checks accumulated motor drift that short windows can hide. The recorded
trajectory is fixed: even a passing replay result does not certify live flight.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .bptt import load_checkpoint
from .onpolicy_scene import continuous_groups
from ..vision.datasets import sha256


@torch.no_grad()
def evaluate(checkpoint,parent,prepared,out):
    torch.set_num_threads(2)
    checkpoint,parent,prepared,out=map(Path,(checkpoint,parent,prepared,out))
    if out.exists():raise FileExistsError(out)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    original=torch.load(parent,map_location='cpu',weights_only=True)
    scene=ck['visual_brain']['scene_training']
    if scene['iteration']<scene['iterations']:raise ValueError('Training is incomplete')
    m=json.loads((prepared/'manifest.json').read_text())
    if scene['parent_sha256']!=sha256(parent) or scene['prepared_sha256']!=sha256(prepared/'manifest.json'):
        raise ValueError('Training provenance differs')
    if scene.get('input_contract')=='retinal_navigation_currents_v1':
        from .navigation_scene import validate_navigation_retention
        validate_navigation_retention(ck,original)
    else:
        changed={k for k,v in ck['model'].items() if k not in original['model'] or not torch.equal(v,original['model'][k])}
        allowed={'encoders.retina__lptc.U','encoders.retina__lptc.log_gain'}
        if scene.get('motor_refit'):allowed.add('readout.weight')
        if changed-allowed:
            raise ValueError('This ablation requires exact nonvisual parent retention')
    brain,_,_=load_checkpoint(checkpoint,'cuda');brain.eval().requires_grad_(False)
    parent_weight=original['model']['readout.weight'].to('cuda')
    parent_bias=original['model']['readout.bias'].to('cuda')
    def parent_readout(module,inputs,output):
        # The recurrent graph is unchanged. Blank parent/target branches use
        # the original motor readout, including their own low-pass action state.
        output=output.clone()
        output[[0,2]]=torch.nn.functional.linear(inputs[0][[0,2]],parent_weight,parent_bias)
        return output
    hook=brain.readout.register_forward_hook(parent_readout)
    W=brain.inference_matrix();results=[]
    for entry in m['takes']:
        if entry['split']!='validation':continue
        if sha256(prepared/entry['arrays'])!=entry['sha256']:raise ValueError('Prepared targets changed')
        with np.load(prepared/entry['arrays']) as z:labels={k:z[k] for k in z.files}
        direct=None
        if 'continuous' in entry:
            path=prepared/entry['continuous']
            if sha256(path)!=entry['continuous_sha256']:raise ValueError('Continuous demonstration changed')
            with np.load(path) as z:raw={k:z[k] for k in z.files}
            n=len(raw['action']);correct=raw['correct'];goals=raw['goal']
            good=raw['fresh'].astype(bool);direct=raw['teacher_action']
            elapsed=raw['clock']-raw['clock'][0]
        else:
            path=Path(entry['source_metadata'])
            if sha256(path)!=entry['source_metadata_sha256']:raise ValueError('Flight metadata changed')
            flight=json.loads(path.read_text());record=flight['neural_replay']
            if sha256(record['path'])!=entry['raw_replay_sha256']:raise ValueError('Flight replay changed')
            with np.load(record['path']) as z:raw={k:z[k] for k in z.files}
            good=(raw['fresh'][:,0]>0)&(raw['clock'][:,3]>=0)&(raw['clock'][:,3]<=.12)
            good&=np.r_[True,(np.diff(raw['clock'][:,1])>0)&(np.diff(raw['clock'][:,1])<.03)]
            ids,_=continuous_groups(good)
            if 'teacher_action' in labels:
                # Successful-prefix retention is explicitly shorter than the
                # original failed flight. Exclude its unlabelled tail from scores.
                ids=ids[:len(labels['action'])]
                good[:]=False;good[ids]=True
            if not np.array_equal(raw['goal'][ids],labels['goal']):raise ValueError('Target alignment differs')
            n=len(raw['action']);correct=np.zeros((n,1),np.float32);correct[ids]=labels['correct']
            goals=raw['goal'].copy()
            if 'teacher_action' in labels:
                direct=raw['action'].copy();direct[ids]=labels['teacher_action']
            else:
                goals[ids]=labels['teacher_goal']*labels['correct']+labels['goal']*(1-labels['correct'])
            elapsed=raw['clock'][:,1]-raw['clock'][0,1]
        # Parent, student, offline target, shuffled images, blank student.
        inputs={k:torch.tensor(raw[k],device='cuda') for k in brain.channel_dims}
        teacher_goal=torch.tensor(goals,device='cuda')
        order=np.arange(n);rng=np.random.default_rng(292)
        for mask in (correct[:,0]<.5,correct[:,0]>=.5):
            group=np.flatnonzero(mask);order[group]=rng.permutation(group)
        outputs=[];state=brain.init_state(5)
        for i in range(n):
            obs={k:v[i:i+1].repeat(5,1) for k,v in inputs.items()}
            obs['retina'][[0,2,4]]=0.;obs['retina'][3]=inputs['retina'][order[i]]
            obs['goal'][2]=teacher_goal[i]
            action,state,_=brain(obs,state,W);outputs.append(action.cpu().numpy())
            if i%5000==0:print('Continuous replay',entry['id'],i,n,flush=True)
        a=np.array(outputs);difference=a[:,0]-raw['action']
        retention=float(abs(difference).max())
        if retention>1e-3:raise RuntimeError('Blank parent does not reproduce live flight')
        if direct is not None:
            a[:,2]=a[:,0]*(1-correct)+direct*correct
        item=dict(id=entry['id'],parent_reference_max_delta=retention,ticks=n,phases={})
        for name,mask in dict(arming=elapsed<5.,prefix=(correct[:,0]<.01)&(elapsed>=5.),
                              correction=correct[:,0]>.99).items():
            mask&=good
            if not mask.any():continue
            delta=a[mask,1]-a[mask,0]
            errors={key:np.mean((a[mask,j]-a[mask,2])**2,axis=0).tolist()
                    for key,j in [('parent',0),('student',1),('shuffled',3),('blank_student',4)]}
            item['phases'][name]=dict(samples=int(mask.sum()),mse_axes=errors,
                motor_delta_rmse=np.sqrt(np.mean(delta**2,axis=0)).tolist(),
                motor_delta_max=np.max(abs(delta),axis=0).tolist())
        out.parent.mkdir(parents=True,exist_ok=True)
        arrays=out.with_name(out.stem+'-'+entry['id']+'.npz')
        with arrays.open('xb') as f:np.savez_compressed(f,actions=a,correct=correct,elapsed=elapsed)
        item['diagnostic_arrays']=str(arrays);results.append(item)
    hook.remove()
    result=dict(checkpoint_sha256=sha256(checkpoint),prepared_sha256=sha256(prepared/'manifest.json'),
                student_requires_teacher=False,offline_target_branch_only=True,closed_loop=False,
                release_qualified=False,state_reset='only once at the original start of each complete flight',
                takes=results)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','parent','prepared','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();evaluate(a.checkpoint,a.parent,a.prepared,a.out)
