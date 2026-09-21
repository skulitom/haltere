"""Add measured human future paths as labels, never as visual/student inputs."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .bptt import load_checkpoint
from ..brain.gate_senses import gate_observation
from ..liftoff.frames import unity_quat_to_sim,unity_vec_to_sim
from ..sim.quad import quat_to_mat,yaw_of
from ..vision.datasets import sha256


@torch.no_grad()
def prepare(parent,prepared,dataset,out):
    torch.set_num_threads(2);parent,prepared,dataset,out=map(Path,(parent,prepared,dataset,out))
    m=json.loads((prepared/'manifest.json').read_text())
    if m['parent_sha256']!=sha256(parent) or m['dataset_sha256']!=sha256(dataset/'manifest.json'):
        raise ValueError('Parent or dataset changed')
    _,cfg,_=load_checkpoint(parent,'cpu')
    sources={t['id']:t for t in json.loads((dataset/'manifest.json').read_text())['takes']}
    out.mkdir(parents=True,exist_ok=False);result=copy.deepcopy(m)
    result.update(parent_replay_sha256=sha256(prepared/'manifest.json'),target_preparation_code_sha256=sha256(__file__),
                  path_supervision=dict(measured_human_future_weight=.8,navigation_predictor_weight=.2,
                                        inputs='Current image and measured body senses only; both future paths are loss labels'),takes=[])
    for e in m['takes']:
        s=sources[e['id']];root=(dataset/s['source']).resolve()
        if sha256(prepared/e['arrays'])!=e['sha256'] or sha256(dataset/s['arrays'])!=s['arrays_sha256']:
            raise ValueError('Prepared arrays changed')
        for name,digest in s['source_hashes'].items():
            if sha256(root/name)!=digest:raise ValueError('Raw recording changed')
        with np.load(prepared/e['arrays']) as z:a={k:z[k] for k in z.files}
        with np.load(dataset/s['arrays']) as z:labels={k:z[k] for k in z.files}
        lookup={int(v):i for i,v in enumerate(labels['source_row'])}
        ids=np.array([lookup[int(i)] for i in a['image_row']]);h=int(np.flatnonzero(np.isclose(labels['horizons_s'],1.))[0])
        index=pd.read_csv(root/'index.csv').iloc[a['image_row']]
        raw=pd.read_csv(root/'telemetry.csv').iloc[a['raw_row']]
        origin=np.array(json.loads((root/'capture.json').read_text())['origin_sim'])
        pos=torch.tensor(unity_vec_to_sim(raw[['px','py','pz']].values)-origin,dtype=torch.float32)
        q=torch.tensor(np.array([unity_quat_to_sim(v) for v in raw[['qx','qy','qz','qw']].values]),dtype=torch.float32)
        R=quat_to_mat(q);capture_R=quat_to_mat(torch.tensor(index[['qw','qx','qy','qz']].values,dtype=torch.float32))
        future=torch.tensor(index[['px','py','pz']].values,dtype=torch.float32)+(capture_R@torch.tensor(labels['future_body'][ids,h])[...,None]).squeeze(-1)
        relative=(R.transpose(-1,-2)@(future-pos)[...,None]).squeeze(-1);N=len(pos)
        # Only relative geometry is needed for this loss label. The zero motion
        # fields below do not replace any of the student's recorded sensors.
        sensor=dict(gyro=torch.zeros(N,3),gravity_body=-R[:,2,:],vel_body=torch.zeros(N,3),vel_world=torch.zeros(N,3),
                    pos=pos,quat=q,up=R[:,2,2],altitude=pos[:,2:3],yaw=yaw_of(q)[:,None])
        a['demonstration_goal']=gate_observation(sensor,torch.zeros(N,1),cfg.task,torch.zeros(N,720),relative,True,True)['goal'].numpy()
        np.savez_compressed(out/e['arrays'],**a)
        result['takes'].append({**e,'sha256':sha256(out/e['arrays'])})
        print('Added measured path labels',e['id'],N,flush=True)
    (out/'manifest.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','prepared','dataset','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a.parent,a.prepared,a.dataset,a.out)
