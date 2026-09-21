"""Fit horizontal drag per mass from recorded motion, without motor/thrust labels.

The two body axes perpendicular to rotor thrust contain gravity-corrected drag.
Use free-flight windows only: collisions and ground support violate the model.
Weak lateral excitation cannot identify separate lateral drag coefficients.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..vision.datasets import sha256


def fit_drag(frame, start_s=5., end_s=None, reference=None):
    d=frame.drop_duplicates('ts').copy()
    ts=d.ts.values-d.ts.iloc[0]
    if end_s is not None:
        keep=ts<end_s
        d,ts=d.loc[keep],ts[keep]
    t=np.arange(ts[0],ts[-1],.01)
    get=lambda columns:np.column_stack([np.interp(t,ts,d[k]) for k in columns])
    v=get(['vx','vy','vz'])
    q=get(['qx','qy','qz','qw'])
    q/=np.linalg.norm(q,axis=1,keepdims=True)
    R=Rotation.from_quat(q).as_matrix().transpose(0,2,1)
    acceleration=gaussian_filter1d(v,4,axis=0,order=1)/.01
    specific=np.einsum('bij,bj->bi',R,acceleration+np.array([0,0,9.80665]))
    velocity_body=np.einsum('bij,bj->bi',R,v)
    good=(get(['z'])[:,0]>.5)&(np.linalg.norm(acceleration,axis=1)<5)&(t>start_s)
    result={}
    for axis,name in enumerate(('forward','lateral')):
        u=velocity_body[good,axis]
        y=-specific[good,axis]
        keep=np.abs(u)>.15
        u,y=u[keep],y[keep]
        if len(y)<100:
            raise ValueError(f'Insufficient free-flight excitation for {name}')
        X=np.stack((u,np.abs(u)*u),axis=1)
        solution=least_squares(lambda x:X@x-y,[.01,.004],bounds=(0,2),loss='soft_l1',f_scale=.1)
        result[name]=dict(linear_per_s=float(solution.x[0]),quadratic_per_m=float(solution.x[1]),
                          samples=len(y),speed_range_mps=[float(u.min()),float(u.max())],
                          acceleration_rms=float(np.sqrt(np.mean((X@solution.x-y)**2))))
        if reference is not None:
            result[name]['reference_acceleration_rms']=float(np.sqrt(np.mean((X@np.asarray(reference)-y)**2)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log'); p.add_argument('--out',required=True)
    p.add_argument('--start',type=float,default=5.); p.add_argument('--end',type=float,required=True)
    p.add_argument('--reference',nargs=2,type=float,help='Evaluate fixed linear/quadratic coefficients per mass without refitting them')
    args=p.parse_args()
    out=Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    result=dict(source=args.log,source_sha256=sha256(args.log),start_s=args.start,end_s=args.end,
                fit=fit_drag(pd.read_csv(args.log),args.start,args.end,args.reference),reference=args.reference,
                limits='Partial horizontal fit; verify on another flight. No vertical/rotational calibration.')
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
