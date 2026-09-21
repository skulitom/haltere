"""Experimental scene-to-connectome distillation; navigation targets are training-only.

Only visual population tuning and gain change. With a blank retinal channel the
parent's complete recurrent dynamics and motor behaviour remain exactly intact.
Recorded replay is not a closed-loop obstacle-avoidance qualification.
"""
import argparse,copy,json,math,time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

from .bptt import load_checkpoint
from .human_brain import Replay,rollout
from .evaluate_gate_brain import sensory_contract
from .thermal import wait_if_hot
from ..brain.gate_senses import gate_observation
from ..brain.retina import retina_input,RETINA_DIM
from ..liftoff.frames import unity_quat_to_sim,unity_vec_to_sim
from ..liftoff.visual_brain import fuse_gate_detection,gate_measurement_or_search
from ..sim.quad import quat_to_mat,yaw_of
from ..vision.camera import Camera,quat_wxyz_to_mat
from ..vision.runtime import detection_geometry
from ..vision.train import load_gatenet
from ..vision.model import decode
from ..vision.datasets import sha256


def visual_parameters_only(brain):
    """Zero retinal input stays identical because encoder thresholds stay frozen."""
    brain.eval().requires_grad_(False)
    enc=brain.encoders['retina__lptc']
    enc.U.requires_grad_(True);enc.log_gain.requires_grad_(True)
    return [enc.U,enc.log_gain]


@torch.no_grad()
def prepare(parent,prepared,dataset,out):
    torch.set_num_threads(2)
    parent,prepared,dataset,out=map(Path,(parent,prepared,dataset,out))
    manifest=json.loads((prepared/'manifest.json').read_text())
    if sha256(dataset/'manifest.json')!=manifest['dataset_sha256']:
        raise ValueError('Prepared replay and immutable source manifest differ')
    brain,cfg,_=load_checkpoint(parent,'cpu')
    meta=torch.load(parent,map_location='cpu',weights_only=True)['visual_brain'];contract=sensory_contract(meta)
    if (meta['graph_sha256']!=manifest['graph_sha256']
            or sha256(Path(cfg.train.graph).with_suffix('.npz'))!=meta['graph_sha256']):
        raise ValueError('Replay and parent connectomes differ')
    if manifest['calibration']!=meta['calibration'] or not cfg.brain.mask_motor_feedback:
        raise ValueError('Parent sensory/motor calibration differs from source replay')
    if not all(contract[k] for k in ('height_invariant','gravity_aligned_height','search_height_anchor')):
        raise ValueError('Scene replay requires the current gate sensory contract')
    if sha256(meta['gate_sensor']['checkpoint'])!=meta['gate_sensor']['sha256']:
        raise ValueError('Detector changed')
    detector=load_gatenet(meta['gate_sensor']['checkpoint'],'cuda')
    camera=Camera(320,180,100.,30.)
    sources={t['id']:t for t in json.loads((dataset/'manifest.json').read_text())['takes']}
    result=dict(schema=1,parent_sha256=sha256(parent),prepared_sha256=sha256(prepared/'manifest.json'),
                dataset_sha256=sha256(dataset/'manifest.json'),teacher_sha256=manifest['teacher_sha256'],
                teacher_training_only=True,retina_mode='scene_v2',gate_sensor=meta['gate_sensor'],
                closed_loop=False,source_replay_limitations=manifest['limitation'],takes=[],
                preparation_code_sha256=sha256(__file__),
                validation_scope='Adapter development validation; inspect upstream detector/brain lineage separately')
    out.mkdir(parents=True,exist_ok=False)
    for entry in manifest['takes']:
        source=sources[entry['id']];root=(dataset/source['source']).resolve()
        for name,digest in source['source_hashes'].items():
            if sha256(root/name)!=digest:raise ValueError('Raw recording changed')
        if sha256(prepared/entry['arrays'])!=entry['sha256']:raise ValueError('Prepared replay changed')
        with np.load(prepared/entry['arrays']) as f:a={k:f[k] for k in f.files}
        index=pd.read_csv(root/'index.csv');raw=pd.read_csv(root/'telemetry.csv')
        image_ids=np.unique(a['image_row']);retinas=np.zeros((len(index),RETINA_DIM),np.float32)
        predictions=np.zeros((len(index),4),np.float32)
        for start in range(0,len(image_ids),32):
            ids=image_ids[start:start+32];pixels=[];scene=[]
            for i in ids:
                rgb=cv2.cvtColor(cv2.imread(str(root/'frames'/index.iloc[i]['file'])),cv2.COLOR_BGR2RGB)
                pixels.append(cv2.resize(rgb,(320,180),interpolation=cv2.INTER_AREA))
                scene.append(cv2.resize(rgb,(160,90),interpolation=cv2.INTER_AREA))
            x=torch.tensor(np.stack(pixels).transpose(0,3,1,2),device='cuda',dtype=torch.float32)/255
            predictions[ids]=decode(detector(x)).cpu().numpy()
            x=torch.tensor(np.stack(scene).transpose(0,3,1,2),dtype=torch.float32)/255
            retinas[ids]=retina_input(x,mode='scene_v2').numpy()
        rows=raw.iloc[a['raw_row']];N=len(rows)
        q=torch.tensor(np.array([unity_quat_to_sim(v) for v in rows[['qx','qy','qz','qw']].values]),dtype=torch.float32)
        R=quat_to_mat(q)
        origin=np.array(json.loads((root/'capture.json').read_text())['origin_sim'])
        pos=unity_vec_to_sim(rows[['px','py','pz']].values)-origin
        vel=torch.tensor(unity_vec_to_sim(rows[['vx','vy','vz']].values),dtype=torch.float32)
        sensor=dict(gyro=torch.zeros(N,3),gravity_body=-R[:,2,:],vel_body=(R.transpose(-1,-2)@vel[...,None]).squeeze(-1),
                    vel_world=vel,pos=torch.tensor(pos,dtype=torch.float32),quat=q,up=R[:,2,2],
                    altitude=torch.tensor(pos[:,2:3],dtype=torch.float32),yaw=yaw_of(q)[:,None])
        relative=np.zeros((N,3),np.float32);teacher_relative=relative.copy();height_error=np.zeros((N,1),np.float32)
        point=stamp=last_image=search_height=None;last_group=None
        for j,(image_id,group) in enumerate(zip(a['image_row'],a['run_id'])):
            if group!=last_group:point=stamp=last_image=search_height=None;last_group=group
            frame=index.iloc[image_id];rotation=R[j].numpy();now=float(rows.iloc[j].recv_time)
            capture_R=quat_wxyz_to_mat(frame[['qw','qx','qy','qz']].to_numpy(dtype=float))
            capture_pos=frame[['px','py','pz']].to_numpy(dtype=float)
            if image_id!=last_image:
                p,u,v,width=predictions[image_id]
                if p>.8:
                    direction,distance=detection_geometry(camera,(u+1)*160,(v+1)*90,width)
                    measured=capture_pos+capture_R@(direction*distance)
                    measured[2]-=meta['gate_sensor']['centre_offset_m']
                    point,stamp=fuse_gate_detection(point,stamp,measured,float(frame.capture_end),pos[j],rotation)
                last_image=image_id
            rel=rotation.T@(point-pos[j]) if point is not None else np.zeros(3)
            relative[j],searching=gate_measurement_or_search(now-stamp if stamp is not None else float('inf'),rel,True)
            if searching:
                if search_height is None:search_height=pos[j,2]
                height_error[j,0]=pos[j,2]-search_height;point=stamp=None
            else:search_height=None
            teacher_relative[j]=rotation.T@(capture_pos+capture_R@a['teacher'][j,-1]-pos[j])
        obs=gate_observation(sensor,torch.zeros(N,1),cfg.task,torch.zeros(N,RETINA_DIM),torch.tensor(relative),
                             True,True,torch.tensor(height_error))
        # Retain the original causal gyro filtering, while fixing its velocity
        # normalization to the deployed height-invariant contract.
        obs['haltere']=torch.tensor(a['haltere']);obs['lptc'][:,:3]=obs['haltere']
        obs['retina']=torch.tensor(retinas[a['image_row']])
        teacher_goal=gate_observation(sensor,torch.zeros(N,1),cfg.task,obs['retina'],torch.tensor(teacher_relative),True,True)['goal']
        arrays={k:v.numpy() for k,v in obs.items()}
        arrays.update(teacher_goal=teacher_goal.numpy(),action=a['action'],run_id=a['run_id'],raw_row=a['raw_row'],image_row=a['image_row'])
        np.savez_compressed(out/entry['arrays'],**arrays)
        result['takes'].append(dict(id=entry['id'],split=entry['split'],arrays=entry['arrays'],sha256=sha256(out/entry['arrays']),ticks=N))
        print('Prepared scene replay',entry['id'],N,flush=True)
    (out/'manifest.json').write_text(json.dumps(result,indent=2))


@torch.no_grad()
def targets(parent,batch):
    base,_,_=rollout(parent,batch,blank=True)
    teacher_batch={**batch,'goal':batch['teacher_goal']}
    target,_,_=rollout(parent,teacher_batch,blank=True)
    target=target.clone();target[:,:,0]=base[:,:,0]  # retain the calibrated climb
    return base,target


@torch.no_grad()
def assess(brain,parent,replay,ids,device):
    totals=np.zeros(3);count=0;maximum=0.
    for start in range(0,len(ids),8):
        b=replay.batch(ids[start:start+8],device);base,target=targets(parent,b)
        action,_,_=rollout(brain,b);blank,_,_=rollout(brain,b,blank=True)
        totals+=np.array([float((a[:,20:]-target[:,20:]).square().mean()) for a in (base,action,blank)])*len(b['action'])
        count+=len(b['action'])
        maximum=max(maximum,float((blank-base).abs().max()))
        # CUDA sparse accumulation varies by about 1e-6 even when replaying the
        # same frozen parent twice. CPU retention is additionally tested exactly.
        if not torch.allclose(blank,base,atol=1e-5,rtol=0):raise RuntimeError('Blank-retina parent retention was lost')
    return dict(zip(('parent_mse','scene_mse','blank_mse'),(totals/count).tolist()),blank_parent_max_delta=maximum)


def train(parent,prepared,out,iterations=200):
    torch.set_num_threads(2);torch.manual_seed(3821);rng=np.random.default_rng(3821)
    parent,prepared,out=map(Path,(parent,prepared,out));manifest=json.loads((prepared/'manifest.json').read_text())
    if sha256(parent)!=manifest['parent_sha256']:raise ValueError('Replay parent differs')
    brain,cfg,_=load_checkpoint(parent,'cuda');frozen,_,_=load_checkpoint(parent,'cuda');frozen.eval().requires_grad_(False)
    parameters=visual_parameters_only(brain)
    with torch.no_grad():parameters[1].clamp_(max=math.log(.03))
    opt=torch.optim.Adam([{'params':[parameters[0]],'lr':.003},{'params':[parameters[1]],'lr':.01}])
    original=torch.load(parent,map_location='cpu',weights_only=True)
    if sha256(Path(cfg.train.graph).with_suffix('.npz'))!=original['visual_brain']['graph_sha256']:
        raise ValueError('Parent connectome changed')
    tr,val=Replay(prepared,'train'),Replay(prepared,'validation',stride=64)
    ids=rng.choice(len(val.windows),min(24,len(val.windows)),replace=False).tolist()
    by_take=[[i for i,(k,_) in enumerate(tr.windows) if k==n] for n in range(len(tr.takes))]
    out.mkdir(parents=True,exist_ok=False)
    code_hash=sha256(__file__)
    (out/'config.json').write_text(json.dumps(dict(parent=str(parent),prepared=str(prepared),iterations=iterations,
        seed=3821,batch_size=8,tuning_lr=.003,gain_lr=.01,training_code_sha256=code_hash),indent=2))
    baseline=assess(brain,frozen,val,ids,'cuda');print('baseline',json.dumps(baseline),flush=True)
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2));best=float('inf');started=time.monotonic()
    for it in range(1,iterations+1):
        if it%10==1:wait_if_hot(70.)
        chosen=[int(rng.choice(by_take[int(rng.integers(len(by_take)))])) for _ in range(8)]
        b=tr.batch(chosen,'cuda');base,target=targets(frozen,b)
        action,_,_=rollout(brain,b,detach_every=8)
        scale=action.new_tensor([.04,.04,.04,.04])
        loss=(((action-target)[:,20:]/scale)**2).mean()+.1*(((action-base)[:,20:]/scale)**2).mean()
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError('Nonfinite update')
        opt.step();time.sleep(.1)
        with (out/'training.jsonl').open('a') as f:
            f.write(json.dumps(dict(iteration=it,loss=float(loss.detach()),elapsed_s=time.monotonic()-started))+'\n')
        if it==1 or it%10==0:print(json.dumps(dict(iteration=it,loss=float(loss.detach()),elapsed_s=time.monotonic()-started)),flush=True)
        if it%50==0 or it==iterations:
            result=assess(brain,frozen,val,ids,'cuda');print('validation',it,json.dumps(result),flush=True)
            candidate=copy.deepcopy(original);state=brain.state_dict()
            candidate['model']={k:state[k].detach().cpu() for k in original['model']}
            changed=[]
            for k,v in candidate['model'].items():
                if not torch.equal(v,original['model'][k]):changed.append(k)
            if set(changed)-{'encoders.retina__lptc.U','encoders.retina__lptc.log_gain'}:raise RuntimeError('Unexpected weight changes')
            meta=candidate['visual_brain'];meta['qualified']=False
            previous=meta.pop('perception_rebind',None)
            meta['scene_training']=dict(parent_sha256=sha256(parent),prepared_sha256=sha256(prepared/'manifest.json'),
                navigation_teacher_sha256=manifest['teacher_sha256'],teacher_training_only=True,iterations=iterations,iteration=it,
                training_code_sha256=code_hash,
                changed_weights=changed,blank_retina_parent_tolerance=1e-5,closed_loop=False,validation=result,
                purpose='experimental visual correction of gate approaches from recorded training-only paths')
            if previous:meta['scene_training']['parent_perception_rebind']=previous
            meta['gate_sensor'].update(raw_retina_active=True,retina_mode=manifest['retina_mode'])
            if manifest.get('scene_projection'):
                meta['gate_sensor']['scene_projection']=manifest['scene_projection']
            torch.save(candidate,out/'last.pt')
            if result['scene_mse']<best:best=result['scene_mse'];torch.save(candidate,out/'best.pt')
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=it,**result))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--parent',required=True);p.add_argument('--prepared',required=True);p.add_argument('--dataset')
    p.add_argument('--out',required=True);p.add_argument('--iterations',type=int,default=200);a=p.parse_args()
    if a.command=='prepare':prepare(a.parent,a.prepared,a.dataset,a.out)
    else:train(a.parent,a.prepared,a.out,a.iterations)
