"""Offline generated-course route for labelled oracle collection, never evaluation.

Known collider geometry is privileged information. This tool qualifies course
playability and collects images; its output must not enter autonomous racing.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def model_clearance(points, geometry):
    """Signed point distance to the offline box model and ground (Unity axes)."""
    points = np.asarray(points, float).reshape(-1, 3)
    distance = points[:, 1]-geometry['ground_plane_y']
    for box in geometry['primitives']:
        yaw = np.deg2rad(box['yaw_deg'])
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        delta = abs((points-np.array(box['center']))@rotation)-np.array(box['size'])/2
        signed = np.linalg.norm(np.maximum(delta, 0), axis=1)+np.minimum(delta.max(axis=1), 0)
        distance = np.minimum(distance, signed)
    return distance


def segment_clear(a, b, geometry, clearance):
    count = max(2, int(np.ceil(np.linalg.norm(b-a)/.2))+1)
    return bool((model_clearance(np.linspace(a, b, count), geometry) >= clearance).all())


def plan_segment(start, goal, geometry, *, clearance=.9, spacing=.5):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if segment_clear(start, goal, geometry, clearance):
        return [start, goal]
    lower = np.floor((np.minimum(start, goal)-[8., 6., 4.])/spacing)*spacing
    lower[1] = max(spacing, lower[1])
    upper = np.ceil((np.maximum(start, goal)+[8., 8., 4.])/spacing)*spacing
    shape = np.rint((upper-lower)/spacing).astype(int)+1
    nodes = np.indices(shape).reshape(3, -1).T
    world = lower+nodes*spacing
    # Half a cell diagonal covers straight motion between occupied-grid samples.
    free = (model_clearance(world, geometry) >= clearance+.5*spacing*np.sqrt(3)).reshape(shape)
    source = tuple(np.rint((start-lower)/spacing).astype(int))
    target = tuple(np.rint((goal-lower)/spacing).astype(int))
    if not free[source] or not free[target]:
        raise ValueError('Grid endpoints lack the requested clearance')
    moves = [(np.array(d), float(np.linalg.norm(d))*spacing)
             for d in itertools.product([-1, 0, 1], repeat=3) if any(d)]
    queue, cost, parent = [(0., source)], {source: 0.}, {}
    expanded = set()
    while queue:
        _, node = heapq.heappop(queue)
        if node in expanded:
            continue
        if node == target:
            chain = [node]
            while chain[-1] != source:
                chain.append(parent[chain[-1]])
            path = [start, *[lower+np.array(k)*spacing for k in reversed(chain)], goal]
            # Greedy visibility simplification is still checked against colliders.
            simplified, i = [path[0]], 0
            while i < len(path)-1:
                j = len(path)-1
                while j > i+1 and not segment_clear(path[i], path[j], geometry, clearance):
                    j -= 1
                if not segment_clear(path[i], path[j], geometry, clearance):
                    raise ValueError('Grid path has an unsafe edge')
                if np.linalg.norm(path[j]-simplified[-1]) > 1e-8:
                    simplified.append(path[j])
                i = j
            return simplified
        expanded.add(node)
        for delta, length in moves:
            neighbor = np.array(node)+delta
            if (neighbor < 0).any() or (neighbor >= shape).any():
                continue
            neighbor = tuple(neighbor)
            if not free[neighbor]:
                continue
            distance = cost[node]+length
            if distance < cost.get(neighbor, np.inf):
                cost[neighbor], parent[neighbor] = distance, node
                estimate = np.linalg.norm(np.array(neighbor)-target)*spacing
                heapq.heappush(queue, (distance+estimate, neighbor))
    raise ValueError('No route with the requested clearance in the bounded search region')


def prepare(bundle, out, *, speed=2., clearance=.9):
    bundle, out = Path(bundle), Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((bundle/'manifest.json').read_text())
    geometry_path = bundle/manifest['offline_geometry']['path']
    geometry_bytes = geometry_path.read_bytes()
    if hashlib.sha256(geometry_bytes).hexdigest() != manifest['offline_geometry']['sha256']:
        raise ValueError('Course geometry hash changed')
    geometry = json.loads(geometry_bytes)
    if geometry.get('runtime_geometry_allowed') is not False or geometry.get('unknown_geometry'):
        raise ValueError('Qualification requires a complete explicitly offline collider model')
    if not np.isfinite([speed, clearance]).all() or not 0 < speed <= 3 or clearance < .6:
        raise ValueError('Use bounded collection speed and at least 0.6 m model clearance')
    track = ET.parse(next(bundle.rglob('*.track'))).getroot()
    spawn = next(b for b in track.findall('./blueprints/TrackBlueprint') if b.findtext('itemID') == 'SpawnPointSingle02')
    position = np.array([float(spawn.findtext('position/'+axis)) for axis in 'xyz'])
    # The known spawn asset sits above its support. Start the airborne route at
    # that declared point; the collection controller separately validates reset.
    path = [position]
    for checkpoint in geometry['checkpoints']:
        center, normal = np.array(checkpoint['center']), np.array(checkpoint['normal'])
        for target in [center-2*normal, center, center+2*normal]:
            path.extend(plan_segment(path[-1], target, geometry, clearance=clearance)[1:])
    samples = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b-a)/.1)+1))
                              for a, b in zip(path, path[1:])])
    minimum = float(model_clearance(samples, geometry).min())
    if minimum < clearance-1e-8:
        raise ValueError('Final route violates requested collider clearance')
    path = np.asarray(path)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    route = dict(schema='haltere.collection_route.v1', frame='unity_world_xyz_m', loop=False,
                 purpose='PRIVILEGED oracle course qualification and image collection; not autonomous evaluation',
                 source=str(geometry_path), source_sha256=hashlib.sha256(geometry_bytes).hexdigest(),
                 source_kind='generated collider model', environment='TheDrawingBoard',
                 race_id=manifest['race_id'], track_id=manifest['track_id'],
                 expected_start_unity=position.tolist(), max_start_error_m=2., max_start_speed_mps=.5,
                 speed_mps=speed, lookahead_m=1.5, length_m=length,
                 minimum_model_clearance_m=minimum, required_model_clearance_m=clearance,
                 live_alignment_verified=False, waypoints_unity=path.tolist())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as file:
        json.dump(route, file, indent=2)
    return route


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--speed', type=float, default=2.)
    args = p.parse_args()
    result = prepare(args.bundle, args.out, speed=args.speed)
    print(json.dumps({k: v for k, v in result.items() if k != 'waypoints_unity'}, indent=2))


if __name__ == '__main__':
    main()
