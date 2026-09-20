"""Offline pose import for modern Liftoff LocalGhostStatesRecording files.

Replay inputs remain unnamed channels: their mapping to our actuator contract has
not been verified. These files are trajectories, not telemetry or labelled images.
"""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .frames import unity_quat_to_sim, unity_vec_to_sim


def read_replay(data: bytes) -> tuple[np.ndarray, dict]:
    """Return validated source rows and metadata, retaining duplicate boundaries.

    Supported rows: Unity position xyz, attitude xyzw, optional input xyzw, time.
    Only the modern POSITIONAL and POSITION_STICKS layouts are established here.
    """
    if data.startswith(b'\x1f\x8b'):
        data = gzip.decompress(data)
    root = ET.fromstring(data)
    if root.tag != 'LocalGhostStatesRecording':
        raise ValueError('Expected LocalGhostStatesRecording XML')
    version = root.findtext('gameVersion', '')
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:\.\d+)?', version):
        raise ValueError('Missing or unsupported gameVersion')
    if tuple(map(int, version.split('.'))) < (0, 14, 0):
        raise ValueError('Legacy replay field order is not supported')
    layout = root.findtext('stateLayout', '')
    widths = {'POSITIONAL': 8, 'POSITION_STICKS': 12}
    if layout not in widths:
        raise ValueError(f'Unsupported replay stateLayout: {layout!r}')
    payload = base64.b64decode(''.join(root.findtext('statesByte', '').split()), validate=True)
    width = widths[layout]
    if len(payload) < 2 * width * 4 or len(payload) % (width * 4):
        raise ValueError('Replay payload has incomplete rows or fewer than two states')
    states = np.frombuffer(payload, dtype='<f4').reshape(-1, width).astype(np.float64)
    if not np.isfinite(states).all():
        raise ValueError('Non-finite replay state')
    norms = np.linalg.norm(states[:, 3:7], axis=1)
    if np.max(np.abs(norms - 1)) > .01:
        raise ValueError('Replay contains invalid attitude quaternions')
    dt = np.diff(states[:, -1])
    if np.any(dt < 0) or not np.any(dt > 0):
        raise ValueError('Replay timestamps reverse or never advance')
    duplicates = np.flatnonzero(dt == 0)
    if any(not np.array_equal(states[i], states[i + 1]) for i in duplicates):
        raise ValueError('Different states share a timestamp')
    starts = [int(x.text) for x in root.findall('lapStartIndices/int')]
    if (any(i < 0 or i >= len(states) for i in starts)
            or any(a >= b for a, b in zip(starts, starts[1:]))):
        raise ValueError('Invalid lap start indices')
    lap_times = [float(x.text) for x in root.findall('lapTimes/float')]
    total = float(root.findtext('totalTime', '0'))
    if not np.isfinite([total, *lap_times]).all() or min([total, *lap_times]) < 0:
        raise ValueError('Invalid recorded race/lap times')
    metadata = {
        'game_version': version, 'name': root.findtext('name'),
        'environment': root.findtext('environment'), 'gamemode': root.findtext('gamemode'),
        'track_id': root.findtext('trackID/str'), 'race_id': root.findtext('raceID/str'),
        'track_version': root.findtext('trackID/version'),
        'race_version': root.findtext('raceID/version'),
        'is_crashed': root.findtext('isCrashed'), 'state_layout': layout,
        'reported_total_time_s': total, 'reported_lap_times_s': lap_times,
        'lap_start_source_indices': starts,
        'source_rows': len(states), 'duplicate_boundary_rows': len(duplicates),
    }
    return states, metadata


def import_replay(source, out, *, provenance=None):
    """Export one immutable source recording, CSV trajectory and an audit report."""
    source = Path(source)
    return export_replay(source.read_bytes(), out, source=str(source.resolve()), provenance=provenance)


def export_replay(raw, out, *, source, provenance=None):
    out = Path(out)
    states, meta = read_replay(raw)
    # Keep the last of identical boundary rows, which belongs to the new segment.
    keep = np.r_[np.diff(states[:, -1]) > 0, True]
    indices, clean = np.flatnonzero(keep), states[keep]
    origin = states[0, :3]
    pos = unity_vec_to_sim(clean[:, :3] - origin)
    quat = np.array([unity_quat_to_sim(q / np.linalg.norm(q)) for q in clean[:, 3:7]])
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] *= -1
    dt = np.diff(clean[:, -1])
    distance = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed = distance / dt
    sticks = clean[:, 7:11] if clean.shape[1] == 12 else None
    meta.update({
        'schema': 1, 'source': source,
        'source_sha256': hashlib.sha256(raw).hexdigest(), 'provenance': provenance or {},
        'rows': len(clean), 'duration_s': float(clean[-1, -1] - clean[0, -1]),
        'path_length_m': float(distance.sum()), 'median_dt_s': float(np.median(dt)),
        'max_dt_s': float(dt.max()), 'median_speed_mps': float(np.median(speed)),
        'p95_speed_mps': float(np.quantile(speed, .95)), 'max_speed_mps': float(speed.max()),
        'origin_unity_xyz': origin.tolist(),
        'position_frame': 'sim x forward, y left, z up; relative to first source pose, NOT current reset',
        'attitude_frame': 'sim body-to-world wxyz, world axes retained',
        'input_channels_present': sticks is not None,
        'input_channels_vary': bool(sticks is not None and np.any(np.ptp(sticks, axis=0) > 1e-6)),
        'input_min': sticks.min(axis=0).tolist() if sticks is not None else None,
        'input_max': sticks.max(axis=0).tolist() if sticks is not None else None,
        'action_labels_verified': False, 'camera_alignment_verified': False,
        'current_course_completion_verified': False,
        'warnings': [
            'Recorded stick channels have unverified order, scale and vehicle dynamics.',
            'Playback speed can differ from source timestamps; do not pair with live player telemetry.',
            'Source lap indices/times are retained as metadata, not proof of current HUD completion.',
            'Align the source world origin and verify the current course before flying this route.',
        ],
    })
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source.recording').write_bytes(raw)
    with (out / 'trajectory.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        columns = ['source_index', 'segment', 't', 'source_timestamp', 'px', 'py', 'pz',
                   'qw', 'qx', 'qy', 'qz', 'unity_x', 'unity_y', 'unity_z']
        if sticks is not None:
            columns += ['recorded_input_0', 'recorded_input_1', 'recorded_input_2', 'recorded_input_3']
        writer.writerow(columns)
        for i, source_index in enumerate(indices):
            segment = int(np.searchsorted(meta['lap_start_source_indices'], source_index, side='right') - 1)
            row = [source_index, segment, clean[i, -1] - states[0, -1], clean[i, -1],
                   *pos[i], *quat[i], *clean[i, :3]]
            if sticks is not None:
                row.extend(sticks[i])
            writer.writerow(row)
    (out / 'report.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta
