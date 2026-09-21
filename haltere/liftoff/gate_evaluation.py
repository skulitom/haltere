"""Offline track-plane checks for camera-brain telemetry, never pilot inputs.

Asset dimensions are not inferred here. The result reports plane intersections
and offsets; video or verified opening dimensions must establish a gate pass.
It does not turn an unscored free-flight session into a completed race.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def track_planes(path, origin):
    # Unity world (right, up, forward) -> simulator (forward, left, up).
    C = np.array([[0.,0.,1.],[-1.,0.,0.],[0.,1.,0.]])
    planes = []
    for b in ET.parse(path).getroot().findall('./blueprints/TrackBlueprint'):
        item = b.findtext('itemID','')
        if not item.startswith('Airgate'):
            continue
        position = np.array([float(b.findtext('position/'+k,'0')) for k in 'xyz'])
        angles = [float(b.findtext('rotation/'+k,'0')) for k in 'zxy']
        R = Rotation.from_euler('zxy',angles,degrees=True).as_matrix()
        planes.append(dict(id=int(b.findtext('instanceID')),item=item,
                           base=C@position-np.asarray(origin),normal=C@R[:,2],
                           side=-C@R[:,0],up=C@R[:,1]))
    return planes


def plane_intersections(positions,times,planes,both_directions=False):
    results=[]
    p,t=np.asarray(positions),np.asarray(times)
    for gate in planes:
        distance=(p-gate['base'])@gate['normal']
        crossings=(distance[:-1]<0)&(distance[1:]>=0)
        if both_directions:
            crossings|=(distance[:-1]>0)&(distance[1:]<=0)
        for i in np.flatnonzero(crossings):
            fraction=-distance[i]/(distance[i+1]-distance[i])
            point=p[i]+fraction*(p[i+1]-p[i])
            offset=point-gate['base']
            lateral=float(offset@gate['side'])
            height=float(offset@gate['up'])
            if abs(lateral)>10 or abs(height)>10:
                continue
            results.append(dict(gate_id=gate['id'],item=gate['item'],
                                normal_direction=1 if distance[i+1]>distance[i] else -1,
                                seconds=float(t[i]+fraction*(t[i+1]-t[i])),
                                position=point.tolist(),lateral_m=lateral,height_above_base_m=height))
    return sorted(results,key=lambda r:r['seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log'); parser.add_argument('--track',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--both-directions',action='store_true',
                        help='Include both mesh-normal directions; mesh facing is not race direction')
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    d=pd.read_csv(args.log)
    meta=json.loads(Path(args.log).with_suffix('.json').read_text())
    intersections=plane_intersections(d[['x','y','z']].values,d.wall.values-d.wall.iloc[0],
                                    track_planes(args.track,meta['origin_sim']),args.both_directions)
    result=dict(checkpoint_sha256=meta['checkpoint_sha256'],stop_reason=meta['stop_reason'],
                runtime_requires_teacher=meta['runtime_requires_teacher'],
                runtime_route_oracle=meta['runtime_route_oracle'],
                aperture_verified=False,race_completion_verified=False,
                plane_crossing_directions='both' if args.both_directions else 'positive normal',
                intersections=intersections)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
