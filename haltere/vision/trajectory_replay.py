"""Audit local proposals on a recorded trajectory; never report counterfactual finishes.

The geometry cache contains causal image/motion predictions, not collider labels.
Task goals and velocities use only controller rows already recorded by each image
receipt. Proposals do not change the replayed vehicle or subsequent observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .camera import Camera, quat_wxyz_to_mat
from .local_trajectory import LocalTrajectoryPlanner
from .surface_memory import SurfaceMemory


def replay(dataset, geometry_replay, log, out):
    dataset, geometry_replay, log, out = map(Path, (dataset, geometry_replay, log, out))
    out.mkdir(parents=True, exist_ok=False)
    geometry_meta = json.loads((geometry_replay/'results.json').read_text())
    # Refuse a geometry cache paired with a different take.
    for name in ('index.csv', 'capture.json', 'camera.yaml'):
        expected = [value for key, value in geometry_meta['source_hashes'].items()
                    if Path(key).name == name]
        actual = hashlib.sha256((dataset/name).read_bytes()).hexdigest()
        if expected != [actual]:
            raise ValueError(f'Geometry cache does not match dataset {name}')
    meta = json.loads(log.with_suffix('.json').read_text())
    motor = meta['motor_controller']
    if motor['kind'] != 'pd':
        raise ValueError('This audit reconstructs the logged PD desired-velocity contract only')
    capture = json.loads((dataset/'capture.json').read_text())
    import yaml
    camera_config = yaml.safe_load((dataset/'camera.yaml').read_text())
    camera = Camera(**{k:camera_config[k] for k in ('width','height','f','tilt_deg')})
    origin = np.asarray(capture['origin_sim'])
    with (dataset/'index.csv').open() as file:
        frames = list(csv.DictReader(file))
    with log.open() as file:
        controls = list(csv.DictReader(file))
    # wall is a receipt/control wall-clock time; unlike game time it also
    # prevents a delayed control row becoming available prematurely in replay.
    control_times = np.array([float(row['wall']) for row in controls])
    if np.any(np.diff(control_times) < 0):
        raise ValueError('Controller wall clock moved backwards; split this recording')
    cloud = np.load(geometry_replay/'cloud.npz')['points']
    memory, planner = SurfaceMemory(), LocalTrajectoryPlanner()
    results, timings, paths = [], [], []
    for i, frame in enumerate(frames[:geometry_meta['frames']]):
        timestamp = float(frame['capture_end'])
        j = int(np.searchsorted(control_times, timestamp, side='right')-1)
        if j < 0 or timestamp-control_times[j] > .2:
            continue
        control = controls[j]
        position = origin+np.array([float(frame[k]) for k in ('px','py','pz')])
        velocity = np.array([float(control[k]) for k in ('vx','vy','vz')])
        rotation = quat_wxyz_to_mat([float(control[k]) for k in ('qw','qx','qy','qz')])
        relative = rotation @ np.array([float(control[k]) for k in ('gate_bx','gate_by','gate_bz')])
        params = motor['parameters']
        requested = relative*np.array([params['position_gain'], params['position_gain'],
                                        params['vertical_position_gain']])
        requested[:2] *= min(1., motor['speed_mps']/max(1e-9, np.linalg.norm(requested[:2])))
        requested[2] = np.clip(requested[2], -params['max_vertical_speed'], params['max_vertical_speed'])
        observed = cloud[cloud[:,0] == i]
        surfaces = memory.update(position, timestamp, observed[:,3:6], observed[:,-1])
        start = time.perf_counter()
        quaternion = np.array([float(frame[k]) for k in ('qw','qx','qy','qz')])
        proposal = planner.propose(position, velocity, requested, surfaces, timestamp,
                                   view=(camera, position, quaternion))
        timings.append(1000*(time.perf_counter()-start))
        results.append(dict(frame=i, ts=float(frame['ts']), status=proposal['status'],
                            control_row=j, control_age_s=timestamp-control_times[j],
                            requested_velocity=requested.tolist(), velocity=proposal['velocity'].tolist(),
                            changed=proposal['changed'], candidates_checked=proposal['candidates_checked'],
                            nominal_margin_m=proposal['nominal_margin_m'] if np.isfinite(proposal['nominal_margin_m']) else None,
                            selected_margin_m=proposal['selected_margin_m'] if np.isfinite(proposal['selected_margin_m']) else None,
                            coverage_certified=False))
        if proposal['changed']:
            path = proposal['path']
            paths.extend([i,t,*p] for t,p in zip(path['times'], path['positions']))
    files = [Path(__file__), Path(__file__).with_name('local_trajectory.py'),
             Path(__file__).with_name('surface_memory.py'), geometry_replay/'cloud.npz',
             geometry_replay/'results.json', log, log.with_suffix('.json')]
    report = dict(schema=1, live_authority=False, runtime_course_geometry=False,
                  source_flight_runtime_route_oracle=meta.get('runtime_route_oracle'),
                  contract='Causal cached image surfaces and past logged PD goals; point-mass proposals only',
                  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  config=asdict(planner.config), frames=len(results),
                  status_counts=dict(Counter(row['status'] for row in results)),
                  timing_ms=dict(p50=float(np.median(timings)),p95=float(np.percentile(timings,95)),
                                 maximum=float(np.max(timings))) if timings else {},
                  limits='Recorded states are unchanged by proposals. No counterfactual completion or clearance claim; motor response, sparse coverage, display latency and live timing remain unvalidated.',
                  rows=results)
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(out/'proposals.npz', points=np.asarray(paths).reshape(-1,5),
                        columns=['frame','future_s','x','y','z'])
    return {k:v for k,v in report.items() if k != 'rows'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','geometry-replay','log','out'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    print(json.dumps(replay(args.dataset,args.geometry_replay,args.log,args.out),indent=2))


if __name__=='__main__':
    main()
