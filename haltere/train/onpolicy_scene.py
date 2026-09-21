"""Correct a visual brain on its own causal flight states, offline only.

The training route supplies a missing-gate label after a reviewed successful
prefix. Neither route, gate identity, position nor progress enters the student.
All four targets come from the frozen parent connectome given that offline label.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .bptt import load_checkpoint
from .human_brain import Replay,rollout
from .scene_brain import visual_parameters_only
from .thermal import wait_if_hot
from ..brain.gate_senses import gate_observation
from ..liftoff.gate_evaluation import track_planes,ordered_race_planes
from ..liftoff.frames import unity_vec_to_sim
from ..sim.quad import quat_to_mat
from ..vision.datasets import sha256


def next_gate_labels(positions,planes,chain):
    """Conservative training-only order tracking; not a certified race scorer."""
    by_id={g['id']:g for g in planes}
    index=0;progress=np.zeros(len(positions),np.int64)
    for i in range(1,len(positions)):
        expected=chain[index % len(chain)];g=by_id[expected['gate_id']]
        normal=g['normal']*expected['normal_direction']
        before=float((positions[i-1]-g['base'])@normal)
        after=float((positions[i]-g['base'])@normal)
        if before<0<=after:
            p=positions[i-1]+(positions[i]-positions[i-1])*(-before/(after-before))
            offset=p-g['base']
            if abs(float(offset@g['side']))<1.3 and .2<float(offset@g['up'])<2.2:
                index+=1
        progress[i]=index
    centres=np.array([by_id[chain[int(p)%len(chain)]['gate_id']]['base']+
                      1.2*by_id[chain[int(p)%len(chain)]['gate_id']]['up'] for p in progress])
    return centres,progress


def continuous_groups(good):
    starts=np.flatnonzero(np.diff(np.r_[False,good,False].astype(int))==1)
    ends=np.flatnonzero(np.diff(np.r_[False,good,False].astype(int))==-1)
    ids=[];groups=[]
    for start,end in zip(starts,ends):
        if end-start>=64:
            ids.extend(range(start,end));groups.extend([len(starts)+start]*(end-start))
    return np.array(ids,dtype=np.int64),np.array(groups,dtype=np.int64)


def bounded_route_lookahead(positions,progress,points,distance,passage_distances,lookahead=4.):
    """Offline route labels retain intervening terrain/obstacle clearance.

    Restrict the nearest-point search to the current ordered route segment so
    a nearby earlier/later leg cannot become a corrective target by accident.
    """
    if lookahead<=0 or len(passage_distances)<=int(np.max(progress)):
        raise ValueError('Route must extend through every corrective segment')
    result=np.empty_like(positions,dtype=float)
    for k in np.unique(progress):
        lo=max(0.,float(passage_distances[k-1])-2.) if k else 0.
        hi=min(float(distance[-1]),float(passage_distances[k])+2.)
        s=np.arange(lo,hi,.25)
        candidates=np.stack([np.interp(s,distance,points[:,j]) for j in range(3)],-1)
        for i in np.flatnonzero(progress==k):
            nearest=int(np.argmin(np.linalg.norm(candidates-positions[i],axis=1)))
            target=min(hi,float(s[nearest])+lookahead)
            result[i]=[np.interp(target,distance,points[:,j]) for j in range(3)]
    return result


@torch.no_grad()
def reconstruct_states(brain,raw,ids,groups):
    """Replay the entire causal history once, saving PRE-step window states.

    These are the deployed parent's states, never states driven by route labels.
    Exact flight actions independently check that this reconstruction is valid.
    """
    starts=[]
    for group in np.unique(groups):
        rows=np.flatnonzero(groups==group)
        starts.extend(range(int(rows[0]),int(rows[-1])-64+2,32))
    needed={int(ids[s]):s for s in starts}
    state=brain.init_state(1);W=brain.inference_matrix();vs=[];acts=[];indices=[]
    inputs={k:torch.tensor(raw[k],device=brain.device) for k in brain.channel_dims}
    errors=[]
    for i in range(len(raw['action'])):
        if i in needed:
            indices.append(needed[i]);vs.append(state['v'][:,0].cpu().numpy().copy())
            acts.append(state['act'][0].cpu().numpy().copy())
        obs={k:v[i:i+1] for k,v in inputs.items()};obs['retina']=torch.zeros_like(obs['retina'])
        action,state,_=brain(obs,state,W)
        errors.append(action[0].cpu().numpy()-raw['action'][i])
        if i%5000==0:print('Reconstructing actual neural history',i,len(raw['action']),flush=True)
    error=np.asarray(errors)
    check=dict(max_absolute_error=float(abs(error).max()),rmse_axes=np.sqrt((error**2).mean(0)).tolist())
    if check['max_absolute_error']>1e-3:
        raise RuntimeError(f'Full-history parent replay differs from live controls: {check}')
    return dict(start=np.array(indices),v=np.array(vs),act=np.array(acts)),check


class WarmReplay(Replay):
    def __init__(self,path,split,length=128):
        super().__init__(path,split,length=length,stride=32)
        self.states=[]
        for entry,_ in self.takes:
            p=Path(path)/entry['states']
            if sha256(p)!=entry['states_sha256']:raise ValueError('Reconstructed states changed')
            with np.load(p) as z:
                self.states.append((dict(zip(z['start'].tolist(),range(len(z['start'])))),
                                    torch.tensor(z['v']),torch.tensor(z['act'])))

    def batch(self,ids,device):
        batch=super().batch(ids,device);v=[];act=[]
        for i in ids:
            take,start=self.windows[i];lookup,states,actions=self.states[take]
            v.append(states[lookup[start]]);act.append(actions[lookup[start]])
        batch.update(initial_v=torch.stack(v).to(device),initial_act=torch.stack(act).to(device))
        return batch


@torch.no_grad()
def prepare(parent,flights,track,race,route,out,after_passages=11):
    parent,out=Path(parent),Path(out)
    torch.set_num_threads(2)
    brain,cfg,_=load_checkpoint(parent,'cuda');brain.eval().requires_grad_(False)
    checkpoint=torch.load(parent,map_location='cpu',weights_only=True);meta=checkpoint['visual_brain']
    if meta['gate_sensor'].get('raw_retina_active'):
        raise ValueError('This correction stage requires a gate-only parent')
    if sha256(Path(cfg.train.graph).with_suffix('.npz'))!=meta['graph_sha256']:
        raise ValueError('Connectome changed')
    if sha256(meta['gate_sensor']['checkpoint'])!=meta['gate_sensor']['sha256']:
        raise ValueError('Detector changed')
    if after_passages<1:
        raise ValueError('A reviewed prefix must be explicitly retained')
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema=1,parent_sha256=sha256(parent),teacher_sha256=sha256(parent),
                  teacher_training_only=True,gate_sensor=meta['gate_sensor'],
                  retina_mode='gatenet_scene_v1',scene_projection=meta['gate_sensor']['scene_projection'],
                  preparation_code_sha256=sha256(__file__),closed_loop=False,
                  path_supervision=dict(source='offline recorded route after reviewed prefix',
                      track_sha256=sha256(track),race_sha256=sha256(race),after_passages=after_passages,
                      route_sha256=sha256(route),lookahead_m=4.,
                      teacher='frozen parent connectome; all four controls, including descent',
                      prefix='Parent action retention before the correction region',
                      inputs='Current image features and recorded body/gate senses only'),takes=[])
    for number,(path,split) in enumerate(flights):
        path=Path(path);m=json.loads(path.read_text());record=m['neural_replay']
        if m['checkpoint_sha256']!=sha256(parent) or record['sha256']!=sha256(record['path']):
            raise ValueError('Changed flight checkpoint or raw replay')
        with np.load(record['path']) as z:a={k:z[k] for k in z.files}
        planes=track_planes(track,m['origin_sim']);order=ordered_race_planes(race,planes)
        # The spawn lead-in first crosses the finish arch, then the race Start.
        order=[order[-1],*order[:-1]]
        centres,progress=next_gate_labels(a['position'],planes,order)
        route_data=json.loads(Path(route).read_text())
        if route_data.get('frame')!='unity_world_xyz_m' or route_data.get('loop') is not False:
            raise ValueError('Expected an explicitly prepared offline collection route')
        points=unity_vec_to_sim(np.array(route_data['waypoints_unity']))-m['origin_sim']
        distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        _,route_progress=next_gate_labels(points,planes,order)
        passages=[distance[np.flatnonzero(route_progress>=k)[0]] for k in range(1,int(route_progress.max())+1)]
        centres=bounded_route_lookahead(a['position'],progress,points,distance,passages)
        R=quat_to_mat(torch.tensor(a['quaternion']))
        relative=(R.transpose(-1,-2)@torch.tensor(centres-a['position'],dtype=torch.float32)[...,None]).squeeze(-1)
        n=len(relative);zero=torch.zeros(n,3)
        senses=dict(gyro=zero,gravity_body=-R[:,2,:],vel_body=zero,vel_world=zero,
                    pos=zero,up=R[:,2,2],altitude=torch.zeros(n,1),yaw=torch.zeros(n,1))
        goal=gate_observation(senses,torch.zeros(n,1),cfg.task,torch.zeros(n,720),relative,True,True)['goal'].numpy()
        correct=(progress>=after_passages).astype(np.float32)[:,None]
        # Do not abruptly ask for a turn at the instant of plane intersection.
        first=np.flatnonzero(correct[:,0])
        if len(first):
            age=a['clock'][:,1]-a['clock'][first[0],1]
            correct*=np.clip(age[:,None],0,1)
        good=(a['fresh'][:,0]>0)&(a['clock'][:,3]>=0)&(a['clock'][:,3]<=.12)
        good&=np.r_[True,(np.diff(a['clock'][:,1])>0)&(np.diff(a['clock'][:,1])<.03)]
        ids,groups=continuous_groups(good)
        arrays={k:a[k][ids] for k in brain.channel_dims}
        arrays.update(action=a['action'][ids],teacher_goal=goal[ids],correct=correct[ids],run_id=groups)
        states,retention=reconstruct_states(brain,a,ids,groups)
        state_name=f'flight_{number:02d}_states.npz';np.savez_compressed(out/state_name,**states)
        name=f'flight_{number:02d}.npz';np.savez_compressed(out/name,**arrays)
        entry=dict(id=path.parent.name,split=split,arrays=name,sha256=sha256(out/name),ticks=len(ids),
                   correction_ticks=int((correct[ids]>.5).sum()),source_metadata=str(path),
                   source_metadata_sha256=sha256(path),raw_replay_sha256=record['sha256'],
                   progression_max=int(progress.max()),states=state_name,states_sha256=sha256(out/state_name),
                   neural_state_source='Full causal parent flight replay, before each window; no route labels',
                   parent_replay_check=retention)
        manifest['takes'].append(entry);print(json.dumps(entry),flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


@torch.no_grad()
def targets(parent,batch):
    base,_,_=rollout(parent,batch,blank=True)
    if 'teacher_action' in batch:
        return base,base*(1-batch['correct'])+batch['teacher_action']*batch['correct']
    goal=batch['goal']*(1-batch['correct'])+batch['teacher_goal']*batch['correct']
    target,_,_=rollout(parent,{**batch,'goal':goal},blank=True)
    return base,target


@torch.no_grad()
def assess(brain,parent,replay,ids,require_blank_parent=True):
    sums={k:np.zeros(4) for k in ('prefix','correction','parent_correction','shuffled_correction','blank_correction')}
    counts=dict(prefix=0,correction=0);blank_max=0.
    for start in range(0,len(ids),8):
        b=replay.batch(ids[start:start+8],brain.device);base,target=targets(parent,b)
        actual,_,_=rollout(brain,b);blank,_,_=rollout(brain,b,blank=True)
        other={**b,'retina':b['retina'].flip(1).roll(1,0)}
        shuffled,_,_=rollout(brain,other)
        blank_max=max(blank_max,float((blank-base).abs().max()))
        for name,mask in [('prefix',b['correct'][:,20:,0]<.01),('correction',b['correct'][:,20:,0]>.99)]:
            counts[name]+=int(mask.sum())
            sums[name]+=((actual[:,20:]-target[:,20:])[mask]**2).sum(0).cpu().numpy()
            if name=='correction':
                sums['parent_correction']+=((base[:,20:]-target[:,20:])[mask]**2).sum(0).cpu().numpy()
                sums['shuffled_correction']+=((shuffled[:,20:]-target[:,20:])[mask]**2).sum(0).cpu().numpy()
                sums['blank_correction']+=((blank[:,20:]-target[:,20:])[mask]**2).sum(0).cpu().numpy()
    if require_blank_parent and blank_max>1e-5:raise RuntimeError('Blank-retina retention failed')
    result={k:(v/max(1,counts['prefix' if k=='prefix' else 'correction'])).tolist() for k,v in sums.items()}
    return dict(mse_axes=result,sample_counts=counts,blank_parent_max_delta=blank_max)


def train(parent,prepared,out,iterations=400,motor_refit=False,prefix_lock=False,navigation_input=False):
    if iterations<=0:raise ValueError('Positive iteration count required')
    if prefix_lock and not motor_refit:raise ValueError('Prefix lock requires motor refitting')
    if navigation_input and motor_refit:raise ValueError('Navigation input keeps the original readout frozen')
    torch.set_num_threads(2);torch.manual_seed(1839);rng=np.random.default_rng(1839)
    parent,prepared,out=map(Path,(parent,prepared,out));m=json.loads((prepared/'manifest.json').read_text())
    if sha256(parent)!=m['parent_sha256']:raise ValueError('Parent changed')
    brain,cfg,graph=load_checkpoint(parent,'cuda');frozen,_,_=load_checkpoint(parent,'cuda')
    frozen.eval().requires_grad_(False)
    if navigation_input:
        from .navigation_scene import add_navigation_input,validate_navigation_retention,ADDED_PARAMETERS
        brain,cfg,parameters=add_navigation_input(frozen,cfg,graph)
    else:parameters=visual_parameters_only(brain)
    initial_gain=.1 if navigation_input else (.01 if motor_refit else .003)
    max_gain=2. if navigation_input else .15
    with torch.no_grad():parameters[1].fill_(math.log(initial_gain))
    groups=[dict(params=[parameters[0]],lr=.003),dict(params=[parameters[1]],lr=.008)]
    if motor_refit:
        brain.readout.weight.requires_grad_(True)
        parameters.append(brain.readout.weight)
        groups.append(dict(params=[brain.readout.weight],lr=.0003 if prefix_lock else .00001))
    opt=torch.optim.Adam(groups)
    original=torch.load(parent,map_location='cpu',weights_only=True)
    tr,val=WarmReplay(prepared,'train'),WarmReplay(prepared,'validation')
    basis=singular=None
    if prefix_lock:
        from .motor_retention import prefix_motor_basis,constrain_readout,outside_basis
        basis,singular=prefix_motor_basis(frozen,tr)
    def partitions(replay):
        prefix=[];correction=[]
        for i,(k,s) in enumerate(replay.windows):
            mask=replay.takes[k][1]['correct'][s:s+replay.length]
            if bool((mask<.01).all()):prefix.append(i)
            elif bool((mask>.99).all()):correction.append(i)
        if not prefix or not correction:raise ValueError('Both retention and correction windows required')
        return prefix,correction
    prefix,correction=partitions(tr);vp,vc=partitions(val)
    ids=[*rng.choice(vp,min(16,len(vp)),replace=False),*rng.choice(vc,min(24,len(vc)),replace=False)]
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    lock=None
    if basis is not None:
        np.savez_compressed(out/'prefix-basis.npz',basis=basis.cpu().numpy(),singular_values=singular.cpu().numpy())
        lock=dict(rank=len(basis),sha256=sha256(out/'prefix-basis.npz'),source='training successful-prefix parent neural states',
                  relative_singular_threshold=1e-5,max_relative_readout_norm=.5)
    (out/'config.json').write_text(json.dumps(dict(parent=str(parent),prepared=str(prepared),iterations=iterations,
        seed=1839,batch_size=8,window_ticks=128,initial_gain=initial_gain,max_gain=max_gain,
        navigation_input=navigation_input,
        motor_refit=motor_refit,motor_readout_learning_rate=(.0003 if prefix_lock else .00001) if motor_refit else None,
        motor_prefix_lock=lock,
        training_code_sha256=sha256(__file__)),indent=2))
    for it in range(1,iterations+1):
        if it%10==1:wait_if_hot(70.)
        chosen=[*rng.choice(prefix,4),*rng.choice(correction,4)]
        b=tr.batch(chosen,'cuda');base,target=targets(frozen,b)
        action,_,_=rollout(brain,b,detach_every=8)
        scale=torch.where(b['correct']>.5,action.new_tensor([.015,.04,.04,.04]),
                          action.new_tensor([.004,.012,.012,.012]))
        loss=(((action-target)[:,20:]/scale[:,20:])**2).mean()
        if motor_refit:
            # Retain body/gate control even if the camera scene is blank. This
            # is measured retention, not the exact invariance of encoder-only fits.
            blank,_,_=rollout(brain,b,blank=True,detach_every=8)
            prefix_mask=(b['correct'][:,20:,0]<.01)
            loss+=(((blank-base)[:,20:]/blank.new_tensor([.004,.012,.012,.012]))[prefix_mask]**2).mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        if basis is not None:
            brain.readout.weight.grad.copy_(outside_basis(brain.readout.weight.grad,basis))
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError('Nonfinite update')
        opt.step()
        if basis is not None:constrain_readout(brain.readout.weight,frozen.readout.weight,basis)
        with torch.no_grad():parameters[1].clamp_(max=math.log(max_gain))
        row=dict(iteration=it,loss=float(loss.detach()),elapsed_s=time.monotonic()-started)
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if it==1 or it%10==0:print(json.dumps(row),flush=True)
        if it%50==0 or it==iterations:
            result=assess(brain,frozen,val,ids,not motor_refit);print('validation',it,json.dumps(result),flush=True)
            candidate=copy.deepcopy(original);state=brain.state_dict()
            keys=set(original['model'])|(ADDED_PARAMETERS if navigation_input else set())
            candidate['model']={k:state[k].detach().cpu() for k in sorted(keys)}
            if navigation_input:
                from ..config import dataclass_to_dict
                candidate['config']['brain']=dataclass_to_dict(cfg.brain)
                validate_navigation_retention(candidate,original)
            changed=[k for k,v in candidate['model'].items() if k not in original['model'] or not torch.equal(v,original['model'][k])]
            allowed={'encoders.retina__lptc.U','encoders.retina__lptc.log_gain'}|({'readout.weight'} if motor_refit else set())
            if navigation_input:allowed=ADDED_PARAMETERS
            if set(changed)-allowed:
                raise RuntimeError('Unexpected weight changes')
            meta=candidate['visual_brain'];meta['qualified']=False
            meta['scene_training']=dict(parent_sha256=sha256(parent),prepared_sha256=sha256(prepared/'manifest.json'),
                teacher_training_only=True,iterations=iterations,iteration=it,training_code_sha256=sha256(__file__),
                path_supervision=m['path_supervision'],changed_weights=changed,closed_loop=False,
                motor_refit=motor_refit,recurrent_weights_unchanged=True,
                input_contract='retinal_navigation_currents_v1' if navigation_input else 'retinal_motion_currents_v1',
                original_weights_unchanged=navigation_input,
                trained_weights=[name for name,value in brain.named_parameters() if value.requires_grad],
                motor_prefix_lock=lock,
                blank_retina_parent_tolerance=None if motor_refit else 1e-5,validation=result,
                purpose='on-policy visual correction with successful-prefix retention')
            meta['gate_sensor']['raw_retina_active']=True
            torch.save(candidate,out/'last.pt')
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=it,**result))+'\n')
        time.sleep(.1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--parent',required=True);p.add_argument('--out',required=True)
    p.add_argument('--prepared');p.add_argument('--train-flight',action='append',default=[])
    p.add_argument('--validation-flight',action='append',default=[])
    p.add_argument('--track');p.add_argument('--race');p.add_argument('--route');p.add_argument('--after-passages',type=int,default=11)
    p.add_argument('--motor-refit',action='store_true',help='Also fit motor readout weights with successful-prefix retention')
    p.add_argument('--prefix-lock',action='store_true',help='Project motor updates outside measured successful-prefix activity')
    p.add_argument('--navigation-input',action='store_true',help='Learn only added zero-centred image currents into existing goal neurons')
    p.add_argument('--iterations',type=int,default=400);a=p.parse_args()
    if a.command=='prepare':
        prepare(a.parent,[(f,'train') for f in a.train_flight]+[(f,'validation') for f in a.validation_flight],
                a.track,a.race,a.route,a.out,a.after_passages)
    else:train(a.parent,a.prepared,a.out,a.iterations,a.motor_refit,a.prefix_lock,a.navigation_input)
