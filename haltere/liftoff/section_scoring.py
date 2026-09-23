"""Offline section scoring from logged pose and generated checkpoint planes.

Geometric crossings are estimates until checked against game checkpoint progress.
They never establish a game finish. This scorer uses geometry only after a run;
the flight sidecar separately declares any privileged runtime route use.
Report course outcomes as well as correlated sections.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .course_pool import digest
from .frames import sim_vec_to_unity


def crossing(a, b, checkpoint):
    center, normal = np.asarray(checkpoint['center']), np.asarray(checkpoint['normal'])
    if not np.allclose(np.linalg.norm(normal), 1.) or abs(normal[1]) > 1e-6:
        raise ValueError('Checkpoint must have a unit horizontal normal')
    d0, d1 = (a-center)@normal, (b-center)@normal
    if not d0 < 0 <= d1:
        return None
    fraction = -d0/(d1-d0)
    point = a+fraction*(b-a)-center
    tangent = np.cross([0., 1., 0.], normal)
    if (abs(point@tangent) <= checkpoint['half_width'] and
            abs(point[1]) <= checkpoint['half_height']):
        return float(fraction)
    return None


def score_sections(geometry, times, positions, *, impact_time=None, runtime_stop=False, max_gap=.2,
                   runtime_geometry_used=None):
    times, positions = np.asarray(times, dtype=float), np.asarray(positions, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or positions.shape != (len(times), 3)
            or not np.isfinite(times).all() or not np.isfinite(positions).all()
            or (np.diff(times) < 0).any()):
        raise ValueError('Need finite chronological timestamps and world positions; resets must split attempts')
    if impact_time is not None and not np.isfinite(impact_time):
        raise ValueError('Impact time must be finite')
    gates = geometry['checkpoints']
    events, index, gap_time = {}, 0, None
    for i in range(1, len(times)):
        if times[i]-times[i-1] > max_gap:
            gap_time = float(times[i-1])
            break  # do not invent unseen crossings across a capture/export gap
        if impact_time is not None and times[i-1] >= impact_time:
            break
        if index == len(gates):
            break
        fraction = crossing(positions[i-1], positions[i], gates[index])
        if fraction is not None:
            stamp = float(times[i-1]+fraction*(times[i]-times[i-1]))
            if impact_time is not None and stamp >= impact_time:
                break  # touching while crossing is not a clean passage
            events[gates[index]['instance_id']] = stamp
            index += 1
    outcomes = []
    for section in geometry['sections']:
        entry, exit_time = events.get(section['entry_checkpoint']), events.get(section['exit_checkpoint'])
        status = 'unattempted'
        if entry is not None:
            if exit_time is not None:
                status = 'geometric_success'
            elif gap_time is not None or runtime_stop and impact_time is None:
                status = 'censored_runtime'
            elif impact_time is not None:
                status = 'impact_failure'
            else:
                status = 'incomplete'
        outcomes.append(dict(id=section['id'], kind=section['kind'], status=status,
                             entry_time=entry, exit_time=exit_time,
                             traversal_s=exit_time-entry if exit_time is not None else None))
    groups = {}
    for row in outcomes:
        counts = groups.setdefault(row['kind'], {})
        counts[row['status']] = counts.get(row['status'], 0)+1
    return dict(schema=1, method='offline directed checkpoint-plane crossings',
                game_finish_confirmed=False, game_checkpoint_progress_confirmed=False,
                runtime_geometry_used=runtime_geometry_used, telemetry_gap_at=gap_time, impact_time=impact_time,
                runtime_stop=runtime_stop, sections=outcomes, per_obstacle_type=groups,
                per_course=dict(geometric_sections_completed=sum(r['status']=='geometric_success' for r in outcomes),
                                total_sections=len(outcomes), impact=impact_time is not None),
                limitation='Sections in a course are correlated. Geometry does not replace HUD/finish evidence.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--log', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    bundle, log = Path(a.bundle), Path(a.log)
    manifest = json.loads((bundle/'manifest.json').read_text())
    geometry_path = (bundle/manifest['offline_geometry']['path']).resolve()
    if not geometry_path.is_relative_to(bundle.resolve()) or digest(geometry_path) != manifest['offline_geometry']['sha256']:
        raise ValueError('Offline geometry does not match the frozen course manifest')
    meta = json.loads(log.with_suffix('.json').read_text())
    rows = np.genfromtxt(log, delimiter=',', names=True)
    positions = np.column_stack([rows[k] for k in ['x', 'y', 'z']])+np.asarray(meta['origin_sim'])
    reason = meta['stop_reason']
    runtime_stop = any(word in reason.lower() for word in ['telemetry', 'camera', 'geometry', 'deadline', 'recorder', 'hidden'])
    image_control = bool((meta.get('geometry_control') or {}).get('live_authority'))
    report = score_sections(json.loads(geometry_path.read_text()), rows['ts'], sim_vec_to_unity(positions),
                            impact_time=(meta.get('impact') or {}).get('timestamp'), runtime_stop=runtime_stop,
                            runtime_geometry_used=True if image_control else meta.get('runtime_route_oracle'))
    report['runtime_course_geometry_used'] = meta.get('runtime_route_oracle')
    report['runtime_image_geometry_control'] = image_control
    report['runtime_route_oracle'] = meta.get('runtime_route_oracle')
    report['autonomous_evaluation_eligible'] = (False if meta.get('runtime_route_oracle') is True
                                               else meta.get('autonomous_evaluation_eligible'))
    report['sources'] = {str(f): digest(f) for f in [log, log.with_suffix('.json'), bundle/'manifest.json', geometry_path]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
