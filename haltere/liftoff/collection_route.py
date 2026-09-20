"""Prepare a bounded bot-route collection experiment in original Unity coordinates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .frames import unity_vec_to_sim, quat_xyzw_to_mat
from .replay import read_replay


def prepare_route(recording, out, *, length_m=80., speed_mps=2.):
    """Preserve the recorded polyline, clipping at a distance or the first lap end.

    No smoothing, altitude clamping, inferred actions or automatic loop closure.
    """
    if not np.isfinite([length_m, speed_mps]).all() or length_m <= 0 or not 0 < speed_mps <= 3:
        raise ValueError('Use a positive route length and a collection speed in (0, 3] m/s')
    recording = Path(recording)
    raw = recording.read_bytes()
    states, meta = read_replay(raw)
    starts = meta['lap_start_source_indices']
    if meta['gamemode'] != 'Race' or len(starts) < 3 or starts[0] != 0:
        raise ValueError('Expected a race recording with lead-in and a complete first lap')
    # In the inspected race format, segment 0 is the lead-in, then lap 1.
    points = states[:starts[2] + 1, :3]
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if len(points) < 2 or distance[-1] < 1:
        raise ValueError('Source route is too short')
    length = min(float(length_m), float(distance[-1]))
    endpoint = [float(np.interp(length, distance, points[:, k])) for k in range(3)]
    prefix = np.vstack([points[distance < length], endpoint])
    route = {
        'schema': 'haltere.collection_route.v1', 'frame': 'unity_world_xyz_m',
        'source': str(recording.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'environment': meta['environment'], 'race_id': meta['race_id'],
        'track_id': meta['track_id'], 'source_game_version': meta['game_version'],
        'expected_start_unity': states[0, :3].tolist(),
        'source_start_attitude_xyzw': states[0, 3:7].tolist(),
        'max_start_error_m': 2., 'max_start_speed_mps': .5,
        'speed_mps': float(speed_mps), 'lookahead_m': 1.5,
        'length_m': length, 'nominal_moving_seconds': length / speed_mps,
        'first_lap_with_lead_in_m': float(distance[-1]),
        'loop': False, 'purpose': 'oracle route teacher for development-data collection',
        'live_alignment_verified': False,
        'waypoints_unity': prefix.tolist(),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        json.dump(route, f, indent=2)
    return route


class CollectionRoute:
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding='utf-8'))
        d = self.data
        if d.get('schema') != 'haltere.collection_route.v1' or d.get('frame') != 'unity_world_xyz_m' or d.get('loop') is not False:
            raise ValueError('Expected a prepared, non-looping collection route')
        self.points = np.asarray(d['waypoints_unity'], dtype=float)
        self.origin = np.asarray(d['expected_start_unity'], dtype=float)
        values = [d['speed_mps'], d['lookahead_m'], d['max_start_error_m'], d['max_start_speed_mps']]
        self.cross_track_gain = float(d.get('cross_track_gain', 0.))
        if not np.isfinite(self.cross_track_gain) or not 0 <= self.cross_track_gain <= 4:
            raise ValueError('Cross-track gain must be finite and between 0 and 4')
        self.cross_track_damping = float(d.get('cross_track_damping', 0.))
        if not np.isfinite(self.cross_track_damping) or not 0 <= self.cross_track_damping <= 3:
            raise ValueError('Cross-track damping must be finite and between 0 and 3')
        self.cross_track_integral = float(d.get('cross_track_integral', 0.))
        if not np.isfinite(self.cross_track_integral) or not 0 <= self.cross_track_integral <= 2:
            raise ValueError('Cross-track integral must be finite and between 0 and 2')
        if (self.points.ndim != 2 or self.points.shape[1] != 3 or len(self.points) < 2
                or self.origin.shape != (3,) or not np.isfinite(self.points).all()
                or not np.isfinite(self.origin).all() or not np.isfinite(values).all()
                or not 0 < values[0] <= 3 or any(v <= 0 for v in values[1:])
                or np.any(np.linalg.norm(np.diff(self.points, axis=0), axis=1) <= 1e-8)):
            raise ValueError('Invalid collection route geometry or limits')

    def relative_waypoints(self, frame):
        """Bind absolute geometry to this live reset without rotating world axes."""
        if not np.isfinite([*frame.position, *frame.velocity, *frame.attitude]).all():
            raise ValueError('Non-finite initial telemetry')
        if abs(np.linalg.norm(frame.attitude) - 1) > .01:
            raise ValueError('Invalid initial attitude')
        error = float(np.linalg.norm(frame.position - self.origin))
        if error > self.data['max_start_error_m']:
            raise ValueError(f'Route start is {error:.2f} m from this reset; select the correct course and spawn')
        if np.linalg.norm(frame.velocity) > self.data['max_start_speed_mps']:
            raise ValueError('Route collection must start with the drone stationary after reset')
        if quat_xyzw_to_mat(frame.attitude)[1, 1] < .9:
            raise ValueError('Route collection must start upright after reset')
        return unity_vec_to_sim(self.points - frame.position)

    def bind(self, pilot, frame):
        points = self.relative_waypoints(frame)
        pilot.loop = False
        pilot.set_waypoints(points)
        pilot.reset(frame)
        pilot.path_speed = self.data['speed_mps']
        pilot.path_lookahead = self.data['lookahead_m']
        pilot.path_cross_track_gain = self.cross_track_gain
        pilot.path_cross_track_damping = self.cross_track_damping
        pilot.path_cross_track_integral = self.cross_track_integral
        return float(np.linalg.norm(frame.position - self.origin))
