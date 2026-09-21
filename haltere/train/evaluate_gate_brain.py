"""Teacher-free gate checks using the sensory contract of the exact checkpoint.

These are synthetic-perception checks, not live-flight or release qualification.
No fallback checkpoint is selected and no training teacher is loaded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .bptt import load_checkpoint
from .gate_brain import evaluate
from ..vision.datasets import sha256


def sensory_contract(meta):
    sensor=meta.get('gate_sensor')
    if not sensor or meta.get('runtime_requires_teacher',True):
        raise ValueError('Expected a teacher-free camera-gate brain')
    if sensor.get('raw_retina_active',False):
        raise ValueError('This synthetic gate check does not evaluate raw retina')
    if sensor.get('missing_gate')!='zero_goal_neural_search':
        raise ValueError('Search checks require the deployed zero-goal search contract')
    if sensor.get('focal_320')!=100. or sensor.get('tilt_deg')!=30.:
        raise ValueError('Synthetic camera does not match the checkpoint calibration')
    offset=float(sensor.get('centre_offset_m',1.5))
    if not math.isfinite(offset) or not 0<=offset<=3:
        raise ValueError('Invalid checkpoint gate centre offset')
    return dict(height_invariant=bool(sensor.get('height_invariant',False)),centre_offset=offset,
                gravity_aligned_height=bool(sensor.get('gravity_aligned_height',False)),
                search_height_anchor=bool(sensor.get('search_height_anchor',False)))


def run(checkpoint,out,device='cuda',seed=813):
    path,out=Path(checkpoint),Path(out)
    if out.exists():
        raise FileExistsError(out)
    fingerprint=sha256(path)
    ck=torch.load(path,map_location='cpu',weights_only=True)
    meta=ck['visual_brain']; contract=sensory_contract(meta)
    if ck['iter']<meta.get('gate_training',{}).get('iterations',0):
        raise ValueError('Training has not reached its requested final iteration')
    torch.set_num_threads(2)
    brain,cfg,_=load_checkpoint(path,device)
    graph_hash=sha256(Path(cfg.train.graph).with_suffix('.npz'))
    if graph_hash!=meta['graph_sha256']:
        raise ValueError('Checkpoint connectome fingerprint differs from the loaded graph')
    result=dict(checkpoint=str(path),checkpoint_sha256=fingerprint,graph_sha256=graph_hash,
                iteration=ck['iter'],teacher_absent=True,synthetic_perception=True,
                release_qualified=False,complete=False,sensory_contract=contract,evaluations={})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:
        json.dump(result,f,indent=2)
    cases=[('level_retention',dict(seconds=25,search=False,elevation=False)),
           ('search',dict(seconds=45,search=True,elevation=False)),
           ('elevation',dict(seconds=45,search=True,elevation=True))]
    for name,case in cases:
        item=evaluate(brain,cfg,seed=seed,turns=True,**case,**contract)
        result['evaluations'][name]=item
        out.write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(case=name,**{k:item[k] for k in
                         ('episodes','crossings','camera_viable_crossings','crashed','search_timeouts')})),flush=True)
    if sha256(path)!=fingerprint:
        raise RuntimeError('Checkpoint changed during evaluation; results are not complete')
    result['complete']=True
    out.write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=813)
    a=p.parse_args();run(a.checkpoint,a.out,a.device,a.seed)


if __name__=='__main__':
    main()
