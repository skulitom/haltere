"""Offline observed-control supervision for the visual connectome.

Images and telemetry are held causally on the deployed 100 Hz clock. The route
teacher's recorded controls are labels only. Full parent neural history is
replayed before extracting training windows, including missing-gate intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from .onpolicy_scene import continuous_groups,reconstruct_states,next_gate_labels
from ..liftoff.visual_brain import VisualController
from ..liftoff.telemetry import TelemetryFrame
from ..liftoff.gate_evaluation import track_planes,ordered_race_planes
from ..vision.camera import Camera
from ..vision.datasets import sha256
from ..vision.model import decode
from ..vision.runtime import detection_geometry
from ..vision.scene_features import scene_map,project_scene
from ..vision.train import load_gatenet


def causal_indices(available,ticks):
    available=np.asarray(available)
    if np.any(np.diff(available)<0):raise ValueError('Availability clock moved backwards')
    return np.searchsorted(available,ticks,side='right')-1


def measured_brain_actions(processed,calibration):
    """Invert deployed calibration; input here is [throttle,roll,pitch,yaw]."""
    a=np.array(processed,dtype=np.float32,copy=True)
    a[:,0]=(a[:,0]-calibration['hover_processed'])/calibration['throttle_scale']+calibration['hover_stick_sim']
    a[:,1:]/=np.asarray(calibration['stick_sign'])
    return a


@torch.no_grad()
def image_features(source,index,sensor,out):
    if sha256(sensor['checkpoint'])!=sensor['sha256'] or sensor['scene_projection']['detector_sha256']!=sensor['sha256']:
        raise ValueError('Frozen detector or scene projection differs')
    net=load_gatenet(sensor['checkpoint'],'cuda');camera=Camera(320,180,sensor['focal_320'],sensor['tilt_deg'])
    features=[];detections=[];image_hash=hashlib.sha256()
    for start in range(0,len(index),32):
        images=[]
        for name in index.file.iloc[start:start+32]:
            encoded=(source/'frames'/name).read_bytes()
            image_hash.update(name.encode()+b'\0'+encoded)
            bgr=cv2.imdecode(np.frombuffer(encoded,dtype=np.uint8),cv2.IMREAD_COLOR)
            if bgr is None:raise ValueError(f'Missing image {name}')
            images.append(cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(320,180),interpolation=cv2.INTER_AREA))
        pixels=torch.tensor(np.stack(images).transpose(0,3,1,2),device='cuda',dtype=torch.float32)/255
        pred=decode(net(pixels)).cpu().numpy()
        features.append(project_scene(scene_map(net,pixels),sensor['scene_projection']).cpu().numpy())
        for p,u,v,width in pred:
            direction,distance=detection_geometry(camera,(u+1)*160,(v+1)*90,width)
            detections.append([p,* (direction*distance),width])
        if start%512==0:print('Image features',start,len(index),flush=True)
    features=np.concatenate(features);detections=np.array(detections,dtype=np.float32)
    np.savez_compressed(out,retina=features,detection=detections)
    return features,detections,image_hash.hexdigest()


@torch.no_grad()
def prepare(parent,mapping,source,retention,out,split,track,race):
    torch.set_num_threads(2)
    code_sha=sha256(__file__)
    parent,source,retention,out=map(Path,(parent,source,retention,out))
    meta=json.loads((source/'capture.json').read_text())
    if meta['stop_reason']=='recording' or meta['source']!='route_teacher_live':
        raise ValueError('A completed, explicitly teacher-guided recording is required')
    if sha256(meta['teacher_route']['path'])!=meta['teacher_route']['sha256']:
        raise ValueError('Collection route changed')
    control=VisualController(parent,mapping,'cuda',stop_on_search_timeout=False)
    control.W=control.brain.inference_matrix()
    if control.meta['gate_sensor'].get('raw_retina_active'):raise ValueError('Expected a gate-only parent')
    camera=yaml.safe_load((source/'camera.yaml').read_text());sensor=control.meta['gate_sensor']
    if (sha256(source/'camera.yaml')!=meta['camera_sha256'] or camera['width']!=640 or camera['height']!=360
            or not np.isclose(camera['f']/2,sensor['focal_320']) or not np.isclose(camera['tilt_deg'],sensor['tilt_deg'])):
        raise ValueError('Recording camera differs from deployed calibration')
    out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['parent_sha256']!=sha256(parent):raise ValueError('Mixed parent brains')
        if any(t['split']==split for t in manifest['takes']):raise ValueError('Split already prepared')
    else:
        manifest=dict(schema=1,parent_sha256=sha256(parent),teacher_sha256=sha256(parent),
            teacher_training_only=True,closed_loop=False,retina_mode='gatenet_scene_v1',
            gate_sensor=control.meta['gate_sensor'],scene_projection=control.meta['gate_sensor']['scene_projection'],
            preparation_code_sha256=sha256(__file__),takes=[],
            path_supervision=dict(source='measured controls from separate slow route-teacher flights',
                teacher='observed processed controls, inverted with deployed brain calibration',
                prefix='retain separately recorded successful autonomous opening',
                inputs='latest available images and telemetry only; no route, position, progress or teacher controls'))
    index=pd.read_csv(source/'index.csv');raw=pd.read_csv(source/'telemetry.csv')
    if len(index)!=meta['frames']:raise ValueError('Capture index incomplete')
    if np.any(np.diff(raw.timestamp)<-.1):raise ValueError('Game reset inside recording')
    feat,det,image_sha=image_features(source,index,control.meta['gate_sensor'],out/f'{split}_features.npz')
    dt=control.cfg.brain.dt
    ticks=np.arange(float(raw.recv_time.iloc[0]),float(raw.recv_time.iloc[-1]),dt)
    rows=causal_indices(raw.recv_time.values,ticks);images=causal_indices(index.capture_end.values,ticks)
    age=np.where(images>=0,ticks-index.capture_end.values[images.clip(0)],np.inf)
    age_telemetry=ticks-raw.recv_time.values[rows]
    data=raw.to_numpy();cols={k:i for i,k in enumerate(raw.columns)}
    def values(row,names):return np.array([row[cols[k]] for k in names])
    replay={k:[] for k in control.brain.channel_dims};actions=[];positions=[]
    for i,(tick,r,j) in enumerate(zip(ticks,rows,images)):
        row=data[r]
        fr=TelemetryFrame(timestamp=row[cols['timestamp']],recv_time=row[cols['recv_time']],
            position=values(row,['px','py','pz']),attitude=values(row,['qx','qy','qz','qw']),
            velocity=values(row,['vx','vy','vz']),gyro=values(row,['gyro_pitch','gyro_roll','gyro_yaw']),
            input=values(row,['in_throttle','in_yaw','in_pitch','in_roll']),
            motor_rpm=values(row,['rpm_lf','rpm_rf','rpm_lb','rpm_rb']))
        fresh=j>=0 and age[i]<=.12
        retina=torch.tensor(feat[j:j+1] if fresh else np.zeros((1,720)),device='cuda',dtype=torch.float32)
        measurement=dict(p=float(det[j,0]),point=det[j,1:4],width=float(det[j,4])) if fresh else None
        # Capture END determines availability, START approximates pose time.
        stamp=float(index.wall_time.iloc[j]) if fresh else tick
        action,_,_=control.step(fr,retina,measurement,stamp,fr.recv_time,tick)
        for k in replay:
            replay[k].append((retina if k=='retina' else control.last_observation[k])[0].cpu().numpy().copy())
        actions.append(action);positions.append(control.senses['pos'][0].cpu().numpy().copy())
        if i%5000==0:print('Causal teacher observation replay',i,len(ticks),flush=True)
    a={k:np.asarray(v,dtype=np.float32) for k,v in replay.items()}
    a['action']=np.asarray(actions,dtype=np.float32)
    planes=track_planes(track,meta['origin_sim']);chain=ordered_race_planes(race,planes);chain=[chain[-1],*chain[:-1]]
    _,progress=next_gate_labels(np.asarray(positions),planes,chain)
    if progress.max()<18:raise ValueError('Demonstration did not traverse the full ordered course')
    correction=(progress>=11).astype(np.float32)[:,None]
    first=np.flatnonzero(correction[:,0])[0]
    correction*=np.clip((ticks-ticks[first])[:,None]/.5,0,1)
    good=(images>=0)&(age<=.12)&(age_telemetry<=.05)&(ticks-ticks[0]>6.)
    # Teacher trajectory supplies correction supervision only. Preserve the
    # opening on actual autonomous states, not this counterfactual trajectory.
    good&=correction[:,0]>.99
    ids,groups=continuous_groups(good)
    if len(ids)<1000:raise ValueError('Insufficient fresh correction data')
    controls=raw[['in_throttle','in_roll','in_pitch','in_yaw']].values[rows]
    labels=measured_brain_actions(controls,control.calibration)
    arrays={k:a[k][ids] for k in control.brain.channel_dims}
    arrays.update(action=a['action'][ids],teacher_action=labels[ids],correct=correction[ids],run_id=groups)
    states,check=reconstruct_states(control.brain,a,ids,groups)
    name=f'{split}_teacher.npz';state_name=f'{split}_teacher_states.npz'
    np.savez_compressed(out/name,**arrays);np.savez_compressed(out/state_name,**states)
    np.savez_compressed(out/f'{split}_continuous.npz',**a,teacher_action=labels,correct=correction,
                        clock=ticks,position=np.array(positions),fresh=good)
    manifest['takes'].append(dict(id=source.name,split=split,arrays=name,sha256=sha256(out/name),
        states=state_name,states_sha256=sha256(out/state_name),ticks=len(ids),correction_ticks=len(ids),
        progression_max=int(progress.max()),parent_replay_check=check,source=str(source),
        source_metadata_sha256=sha256(source/'capture.json'),source_telemetry_sha256=sha256(source/'telemetry.csv'),
        source_index_sha256=sha256(source/'index.csv'),feature_sha256=sha256(out/f'{split}_features.npz'),
        source_images_sha256=image_sha,preparation_code_sha256=code_sha,
        continuous=f'{split}_continuous.npz',continuous_sha256=sha256(out/f'{split}_continuous.npz'),
        source_route_sha256=meta['teacher_route']['sha256'],timing=meta['timing']))
    original=json.loads((retention/'manifest.json').read_text())
    if original['parent_sha256']!=sha256(parent):raise ValueError('Retention parent differs')
    for entry in original['takes']:
        if entry['split']!=split:continue
        if sha256(retention/entry['arrays'])!=entry['sha256']:raise ValueError('Retention changed')
        with np.load(retention/entry['arrays']) as z:b={k:z[k] for k in z.files}
        end=int(np.flatnonzero(b['correct'][:,0]>.01)[0])
        b={k:v[:end] for k,v in b.items() if k!='teacher_goal'}
        b['correct'][:]=0;b['teacher_action']=b['action'].copy()
        name=f'{split}_retention.npz';np.savez_compressed(out/name,**b)
        if sha256(retention/entry['states'])!=entry['states_sha256']:raise ValueError('Retention states changed')
        state_name=f'{split}_retention_states.npz';(out/state_name).write_bytes((retention/entry['states']).read_bytes())
        manifest['takes'].append(dict(**{k:v for k,v in entry.items() if k not in ('arrays','sha256','states','states_sha256','ticks','correction_ticks')},
            arrays=name,sha256=sha256(out/name),states=state_name,states_sha256=sha256(out/state_name),ticks=end,correction_ticks=0))
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print('Prepared',split,'correction ticks',len(ids),'parent reconstruction',check,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','mapping','source','retention','out','track','race'):p.add_argument('--'+name,required=True)
    p.add_argument('--split',choices=['train','validation'],required=True)
    prepare(**vars(p.parse_args()))
