"""Where are the gates? Gate positions recovered from lap flights, and gate labels projected into frames.

Liftoff does not expose its track layout, so the gates are found from the flight itself: when the
drone flies through a truss gate the dark truss members fill the border of the FPV image for a
few tenths of a second while the drone's position (from telemetry) is known. Those passages give the
gate positions along the taught path; the path direction there gives each gate's facing. Once the
gates are known, every frame of every dataset can be labelled by projecting the next gate into the
camera, which is what the detector network is trained on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .camera import Camera, world_to_body

GATE_WIDTH_M = 3.0      # nominal inner width of the truss gates (used to turn apparent size into distance)


def border_darkness(gray: np.ndarray, band: float = 0.08) -> float:
    """Fraction of dark pixels in the outer band of the image (the truss fills it while passing through)."""
    h, w = gray.shape
    bh, bw = max(1, int(h * band)), max(1, int(w * band))
    border = np.concatenate([gray[:bh].ravel(), gray[-bh:].ravel(), gray[:, :bw].ravel(), gray[:, -bw:].ravel()])
    return float((border < 70).mean())


def passages_from_dataset(dataset: str | Path, thresh: float = 0.35, min_gap_m: float = 8.0, verbose: bool = True) -> list[dict]:
    """Detect gate passages: peaks of border darkness along a flight, at least min_gap_m apart along the path.
    Returns [{pos, heading, frame, darkness}] in the dataset's (sim) world frame."""
    import cv2
    from .calibrate import load_index
    rows = load_index(dataset)
    frames_dir = Path(dataset) / 'frames'
    dark = np.zeros(len(rows))
    for i, r in enumerate(rows):
        g = cv2.imread(str(frames_dir / r['file']), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            dark[i] = border_darkness(g)
    pos = np.array([[r['px'], r['py'], r['pz']] for r in rows])
    # heading of travel: smoothed position differences
    k = 5
    vel = np.zeros_like(pos)
    vel[k:-k] = pos[2 * k:] - pos[:-2 * k]
    passages = []
    order = np.argsort(-dark)
    for i in order:
        if dark[i] < thresh:
            break
        if any(np.linalg.norm(pos[i] - p['pos']) < min_gap_m for p in passages):
            continue
        v = vel[i, :2]
        heading = float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 1e-3 else 0.0
        passages.append({'pos': pos[i].tolist(), 'heading': heading, 'frame': rows[i]['file'], 'darkness': float(dark[i])})
    passages.sort(key=lambda p: p['frame'])
    if verbose:
        print(f'{len(rows)} frames, border darkness median {np.median(dark):.2f}, max {dark.max():.2f}; '
              f'{len(passages)} passages above {thresh}')
        for p in passages:
            print(f"  {p['frame']}: pos {np.round(p['pos'], 1)} heading {np.degrees(p['heading']):.0f} deg darkness {p['darkness']:.2f}")
    return passages


def gates_from_passage_frames(dataset: str | Path, frames: list[int], track_yaml: str | Path | None = None,
                              verbose: bool = True, min_gap_m: float = 6.0) -> list[dict]:
    """Gate positions from the frames in which the drone passes a gate (read off thumbnail sheets by eye).
    The gate is placed at the taught path's point nearest the drone at that moment (the human flew through
    the gate centres, so the path is more accurate than the brain's own line); its facing is the path
    direction there. Without a track file the drone's own position and travel direction are used."""
    from .calibrate import load_index
    rows = load_index(dataset)
    by_name = {r['file']: r for r in rows}
    pos_all = np.array([[r['px'], r['py'], r['pz']] for r in rows])
    path = None
    if track_yaml:
        import yaml
        wps = np.array(yaml.safe_load(Path(track_yaml).read_text(encoding='utf-8'))['waypoints'], dtype=np.float64)
        # densify the polyline to 0.25 m so the nearest point is precise
        pts = [wps[0]]
        for a, b in zip(wps[:-1], wps[1:]):
            n = max(1, int(np.linalg.norm(b - a) / 0.25))
            pts.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
        path = np.array(pts)
    gates = []
    for gi, fr in enumerate(frames):
        name = f'{fr:06d}.jpg'
        if name not in by_name:
            if verbose:
                print(f'frame {name} not in the dataset, skipped')
            continue
        i = rows.index(by_name[name])
        pos = pos_all[i]
        k0, k1 = max(0, i - 8), min(len(rows) - 1, i + 8)
        travel = pos_all[k1] - pos_all[k0]
        if path is not None:
            j = int(np.argmin(np.linalg.norm(path - pos, axis=1)))
            gpos = path[j]
            j0, j1 = max(0, j - 8), min(len(path) - 1, j + 8)
            travel = path[j1] - path[j0]
        else:
            gpos = pos
        heading = float(np.arctan2(travel[1], travel[0])) if np.linalg.norm(travel[:2]) > 1e-3 else 0.0
        if any(np.linalg.norm(np.asarray(g['pos']) - gpos) < min_gap_m for g in gates):
            if verbose:
                print(f'frame {name}: within {min_gap_m} m of a gate already listed, skipped')
            continue
        gates.append({'pos': gpos.tolist(), 'heading': heading, 'frame': name, 'gate': len(gates),
                      'drone_offset_m': float(np.linalg.norm(gpos - pos))})
        if verbose:
            print(f'gate {gates[-1]["gate"]} at frame {name}: pos {np.round(gpos, 1)} heading {np.degrees(heading):.0f} deg '
                  f'(drone was {gates[-1]["drone_offset_m"]:.1f} m from the path point)')
    return gates


def save_gates(passages: list[dict], out: str | Path) -> None:
    Path(out).write_text(json.dumps({'gate_width_m': GATE_WIDTH_M, 'gates': passages}, indent=1), encoding='utf-8')


def load_gates(path: str | Path) -> list[dict]:
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    return d['gates']


def next_gate_index(pos: np.ndarray, gates: list[dict], passed_margin: float = 1.0) -> int | None:
    """The next gate along the course for a drone at pos: the first gate (in course order, cyclic) that the
    drone has not passed yet, judged by the signed distance along the gate's facing direction."""
    if not gates:
        return None
    best = None
    for i, g in enumerate(gates):
        d = np.asarray(g['pos']) - pos
        h = g['heading']
        along = d[0] * np.cos(h) + d[1] * np.sin(h)        # > 0: the gate is still ahead of the drone
        if along > -passed_margin:
            dist = float(np.linalg.norm(d))
            if best is None or dist < best[1]:
                best = (i, dist)
    return None if best is None else best[0]


def gate_label(pos: np.ndarray, quat_wxyz: np.ndarray, gates: list[dict], cam: Camera, next_only: bool = True) -> dict:
    """Label for one frame: the next gate's pixel centre, apparent width and distance (if it is in the image).
    Returns {'visible': 0/1, 'u', 'v' (pixels), 'width_px', 'dist_m', 'gate': index}."""
    i = next_gate_index(pos, gates)
    if i is None:
        return {'visible': 0}
    g = gates[i]
    centre = np.asarray(g['pos'], dtype=np.float64)
    # gate corners: a square of GATE_WIDTH_M facing the heading, centred at the passage point
    h = g['heading']
    side = np.array([-np.sin(h), np.cos(h), 0.0]) * GATE_WIDTH_M / 2
    up = np.array([0.0, 0.0, GATE_WIDTH_M / 2])
    corners = np.stack([centre + side + up, centre - side + up, centre - side - up, centre + side - up])
    pts_b = world_to_body(np.vstack([centre[None], corners]), pos, quat_wxyz)
    px, ok = cam.project_body(pts_b)
    dist = float(np.linalg.norm(pts_b[0]))
    if not ok.all() or dist < 0.8:
        return {'visible': 0, 'gate': i}
    u, v = px[0]
    inside = -0.1 * cam.width <= u <= 1.1 * cam.width and -0.1 * cam.height <= v <= 1.1 * cam.height
    width_px = float(np.linalg.norm(px[1] - px[2]))
    return {'visible': int(inside), 'u': float(u), 'v': float(v), 'width_px': width_px, 'dist_m': dist, 'gate': i}
