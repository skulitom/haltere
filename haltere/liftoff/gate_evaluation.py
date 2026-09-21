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


def ordered_race_planes(race_path, planes):
    """Read a single race chain for OFFLINE diagnostics only.

    Directionality strings describe prefab trigger axes, which do not always
    match a mesh's root normal (the finish and last Straw Bale arch differ).
    Infer approach direction from the preceding checkpoint instead. This is
    explicitly a geometric diagnostic, never the game's race score.
    """
    passages = ET.parse(race_path).getroot().findall('./checkPointPassages/RaceCheckpointPassage')
    by_id = {p.findtext('uniqueId'): p for p in passages}
    starts = [p for p in passages if p.findtext('passageType') == 'Start']
    if len(starts) != 1 or len(by_id) != len(passages):
        raise ValueError('Expected one start and unique race passages')
    chain, seen, current = [], set(), starts[0]
    while True:
        uid = current.findtext('uniqueId')
        if uid in seen:
            raise ValueError('Cyclic passage graph is unsupported')
        seen.add(uid)
        chain.append(int(current.findtext('checkPointID')))
        following = current.findall('./nextPassageIDs/string')
        if current.findtext('passageType') == 'Finish':
            if following:
                raise ValueError('Finish has unexpected successors')
            break
        if len(following) != 1 or following[0].text not in by_id:
            raise ValueError('Expected a single connected race chain')
        current = by_id[following[0].text]
    if len(seen) != len(passages):
        raise ValueError('Disconnected race passages')
    gates = {g['id']: g for g in planes}
    result = []
    for i, gate_id in enumerate(chain):
        if gate_id not in gates or chain[i-1] not in gates:
            raise ValueError('Race includes checkpoints without evaluated planes')
        gate, previous = gates[gate_id], gates[chain[i-1]]
        approach = float((gate['base'] - previous['base']) @ gate['normal'])
        if abs(approach) < .1:
            raise ValueError('Cannot infer approach direction for coplanar checkpoints')
        result.append(dict(gate_id=gate_id, normal_direction=1 if approach > 0 else -1))
    return result


def ordered_plane_progress(intersections, chain):
    """Reject wrong-side and out-of-order crossings; aperture remains unverified."""
    if not chain:
        raise ValueError('Empty race chain')
    index, cycles, accepted, rejected = 0, 0, [], []
    for event in sorted(intersections, key=lambda r:r['seconds']):
        expected = chain[index]
        if event['gate_id'] != expected['gate_id']:
            rejected.append(dict(**event, reason='not_expected_checkpoint'))
            continue
        if event['normal_direction'] != expected['normal_direction']:
            rejected.append(dict(**event, reason='wrong_approach_side'))
            continue
        accepted.append(event)
        index += 1
        if index == len(chain):
            index = 0
            cycles += 1
    return dict(completed_plane_cycles=cycles, current_cycle_plane_count=index,
                next_expected=chain[index], accepted=accepted, rejected=rejected,
                approach_direction_source='inferred from previous checkpoint centre and mesh normal',
                aperture_verified=False, race_completion_verified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log'); parser.add_argument('--track',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--race',help='Optional race XML for order/approach-side diagnostics; never a race score')
    parser.add_argument('--both-directions',action='store_true',
                        help='Include both mesh-normal directions; mesh facing is not race direction')
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    d=pd.read_csv(args.log)
    meta=json.loads(Path(args.log).with_suffix('.json').read_text())
    planes=track_planes(args.track,meta['origin_sim'])
    intersections=plane_intersections(d[['x','y','z']].values,d.wall.values-d.wall.iloc[0],
                                    planes,args.both_directions or bool(args.race))
    result=dict(checkpoint_sha256=meta['checkpoint_sha256'],stop_reason=meta['stop_reason'],
                runtime_requires_teacher=meta['runtime_requires_teacher'],
                runtime_route_oracle=meta['runtime_route_oracle'],
                aperture_verified=False,race_completion_verified=False,
                plane_crossing_directions='both' if args.both_directions or args.race else 'positive normal',
                intersections=intersections)
    if args.race:
        result['ordered_plane_progress']=ordered_plane_progress(intersections,ordered_race_planes(args.race,planes))
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
