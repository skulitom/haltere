"""Replay causal image geometry without opening a controller or reading a route.

Optional --offline-labels reads generated collider geometry only AFTER prediction
for scoring. It does not change feature tracking, surfaces or warning decisions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .camera import Camera, quat_wxyz_to_mat
from .geometry_mask import liftoff_geometry_mask
from .surface_memory import SurfaceMemory, observed_path_margin
from .temporal_depth import TemporalDepth


def replay(dataset, out, *, offline_labels=None, stop_timestamp=None):
    import cv2
    import yaml

    cv2.setNumThreads(2)
    dataset, out = Path(dataset), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((dataset/'capture.json').read_text())
    config = yaml.safe_load((dataset/'camera.yaml').read_text())
    camera = Camera(**{k: config[k] for k in ['width', 'height', 'f', 'tilt_deg']}).scaled(640, 360)
    origin = np.asarray(meta['origin_sim'], float)
    rows = list(csv.DictReader((dataset/'index.csv').open()))
    tracker, memory = TemporalDepth(camera), SurfaceMemory()
    previous = None
    results, clouds, labels, timing = [], [], [], []
    geometry = json.loads(Path(offline_labels).read_text()) if offline_labels else None
    for i, row in enumerate(rows):
        ts, stamp = float(row['ts']), float(row['capture_end'])
        if stop_timestamp is not None and ts >= stop_timestamp:
            break
        rgb = cv2.cvtColor(cv2.imread(str(dataset/'frames'/row['file'])), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        position = origin+np.array([float(row[k]) for k in ['px', 'py', 'pz']])
        quaternion = np.array([float(row[k]) for k in ['qw', 'qx', 'qy', 'qz']])
        begin = time.perf_counter()
        result = tracker.update(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), position,
                                quaternion, stamp, liftoff_geometry_mask(rgb))
        points, sigma, pixels, ranges = np.empty((0, 3)), np.empty(0), np.empty((0, 2)), np.empty(0)
        if result is not None:
            good = result['valid'] & (result['range_sigma_m'] < .25*result['range_m'])
            points, sigma = result['position_world'][good], result['range_sigma_m'][good]
            pixels, ranges = result['pixels'][good], result['range_m'][good]
        inferred = time.perf_counter()
        surfaces = memory.update(position, stamp, points, sigma)
        velocity = np.zeros(3) if previous is None else (position-previous[0])/max(.01, ts-previous[1])
        previous = position, ts
        # Diagnostic extrapolation only. This does not simulate a feasible flight.
        path = position+np.linspace(0, 1.2, 31)[:, None]*velocity
        warning = observed_path_margin(path, surfaces)
        finished = time.perf_counter()
        timing.append([1000*(inferred-begin), 1000*(finished-inferred)])
        results.append(dict(frame=i, ts=ts, capture_time=stamp, valid_points=len(points),
                            memory_points=len(surfaces['points']), triangles=len(surfaces['triangles']),
                            margin_m=warning['margin_m'] if np.isfinite(warning['margin_m']) else None,
                            observed_collision=warning['observed_collision'], coverage_certified=False))
        clouds.extend([i, stamp, ts, *p, float(s)] for p, s in zip(points, sigma))
        if geometry is not None and len(points):
            from ..liftoff.frames import sim_vec_to_unity
            from ..liftoff.section_geometry import collision_depth
            rays = sim_vec_to_unity(camera.unproject_body(pixels)@quat_wxyz_to_mat(quaternion).T)
            truth = collision_depth(geometry, sim_vec_to_unity(position), rays, max_range=30.)['range_m']
            known = np.isfinite(truth) & (truth > 1.)
            labels.extend(dict(frame=i, predicted=float(p), collider=float(t), ratio=float(p/t))
                          for p, t in zip(ranges[known], truth[known]))
    ratios = np.array([row['ratio'] for row in labels])
    durations = np.array(timing).reshape(-1, 2)
    files = [Path(__file__), Path(__file__).with_name('temporal_depth.py'),
             Path(__file__).with_name('surface_memory.py'), Path(__file__).with_name('geometry_mask.py'),
             dataset/'index.csv', dataset/'capture.json', dataset/'camera.yaml']
    if offline_labels:
        files.append(Path(offline_labels))
    report = dict(schema=1, dataset=str(dataset), source_controller=meta['course'],
                  live_authority=False, runtime_course_geometry=False,
                  contract='Causal image/motion surfaces; 1.2 s constant-velocity diagnostic; no free-space certification',
                  source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  stop_timestamp=stop_timestamp, frames=len(results),
                  frames_with_points=sum(r['valid_points'] > 0 for r in results),
                  warning_frames=sum(r['observed_collision'] for r in results),
                  timings_ms={name: dict(p50=float(np.median(durations[:, k])),
                                         p95=float(np.percentile(durations[:, k], 95)))
                              for k, name in enumerate(['tracking_and_mask', 'surfaces_and_query'])} if len(durations) else {},
                  offline_depth=dict(points=len(ratios),
                                     median_ratio=float(np.median(ratios)) if len(ratios) else None,
                                     mean_absolute_relative_error=float(np.mean(abs(ratios-1))) if len(ratios) else None,
                                     overestimate_25pct_fraction=float(np.mean(ratios > 1.25)) if len(ratios) else None),
                  limits='Correlated development observations; primitive collider labels; display latency unmeasured; sparse holes and interpolated patches remain uncertain.',
                  rows=results, labels=labels)
    np.savez_compressed(out/'cloud.npz', points=np.asarray(clouds).reshape(-1, 7),
                        columns=['frame', 'wall', 'ts', 'x', 'y', 'z', 'sigma'])
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return {k: v for k, v in report.items() if k not in ('rows', 'labels')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--offline-labels')
    p.add_argument('--stop-timestamp', type=float)
    args = p.parse_args()
    print(json.dumps(replay(args.dataset, args.out, offline_labels=args.offline_labels,
                            stop_timestamp=args.stop_timestamp), indent=2))


if __name__ == '__main__':
    main()
