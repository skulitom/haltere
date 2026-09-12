"""Gate positions from pixel observations in posed frames (triangulation), and contact sheets to read them off.

A gate observed at pixel (u, v) in a frame whose drone pose is known defines a ray in the world; the
same gate seen from several frames gives a set of rays whose least-squares intersection is the gate's
position. Observations come from a JSON list of {"file", "u", "v", "gate"} entries (pixel coordinates
in the dataset's frame size); ``sheet`` renders frames with a labelled grid so those can be read off.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibrate import load_index
from .camera import Camera, quat_wxyz_to_mat


def ray_world(row: dict, u: float, v: float, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Origin and unit direction (world/sim frame) of the ray through pixel (u, v) of a dataset frame."""
    pos = np.array([row['px'], row['py'], row['pz']])
    R = quat_wxyz_to_mat(np.array([row['qw'], row['qx'], row['qy'], row['qz']]))
    d_b = cam.unproject_body(np.array([[u, v]]))[0]
    return pos, R @ d_b


def intersect_rays(origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares point closest to all rays; returns (point, rms distance to the rays)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    p = np.linalg.lstsq(A, b, rcond=None)[0]
    res = [np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - o)) for o, d in zip(origins, dirs)]
    return p, float(np.sqrt(np.mean(np.square(res))))


def gates_from_observations(dataset: str | Path, observations: list[dict], cam: Camera, verbose: bool = True) -> list[dict]:
    """observations: [{'file', 'u', 'v', 'gate'}] -> [{'pos', 'heading', 'rms', 'n', 'gate'}] sorted by gate id.
    The heading of each gate is the direction of travel of the drone when it was closest to the gate."""
    rows = {r['file']: r for r in load_index(dataset)}
    order = list(rows)
    by_gate: dict = {}
    for o in observations:
        by_gate.setdefault(o['gate'], []).append(o)
    gates = []
    for gid in sorted(by_gate):
        obs = by_gate[gid]
        origins, dirs = [], []
        for o in obs:
            if o['file'] not in rows:
                continue
            org, d = ray_world(rows[o['file']], float(o['u']), float(o['v']), cam)
            origins.append(org)
            dirs.append(d)
        if len(origins) < 2:
            if verbose:
                print(f'gate {gid}: fewer than two usable observations, skipped')
            continue
        p, rms = intersect_rays(np.array(origins), np.array(dirs))
        # heading: travel direction of the drone at its closest approach to the gate
        pos_all = np.array([[rows[f]['px'], rows[f]['py'], rows[f]['pz']] for f in order])
        k = int(np.argmin(np.linalg.norm(pos_all - p, axis=1)))
        k0, k1 = max(0, k - 5), min(len(order) - 1, k + 5)
        vel = pos_all[k1] - pos_all[k0]
        heading = float(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 1e-3 else 0.0
        gates.append({'pos': p.tolist(), 'heading': heading, 'rms': rms, 'n': len(origins), 'gate': gid,
                      'closest_m': float(np.linalg.norm(pos_all[k] - p))})
        if verbose:
            print(f'gate {gid}: {len(origins)} rays -> {np.round(p, 2)} (rms {rms:.2f} m, heading {np.degrees(heading):.0f} deg, '
                  f'closest approach {gates[-1]["closest_m"]:.1f} m)')
    return gates


def sheet(dataset: str | Path, files: list[str], out: str | Path, cols: int = 3, grid: int = 40) -> Path:
    """Contact sheet of dataset frames with a labelled pixel grid (for reading gate positions off by eye)."""
    from PIL import Image, ImageDraw
    frames_dir = Path(dataset) / 'frames'
    ims = []
    for f in files:
        im = Image.open(frames_dir / f).convert('RGB')
        d = ImageDraw.Draw(im)
        w, h = im.size
        for x in range(0, w, grid):
            d.line([(x, 0), (x, h)], fill=(255, 255, 0) if x % (grid * 2) == 0 else (120, 120, 0), width=1)
            if x % (grid * 2) == 0:
                d.text((x + 2, 2), str(x), fill=(255, 255, 0))
        for y in range(0, h, grid):
            d.line([(0, y), (w, y)], fill=(255, 255, 0) if y % (grid * 2) == 0 else (120, 120, 0), width=1)
            if y % (grid * 2) == 0:
                d.text((2, y + 2), str(y), fill=(255, 255, 0))
        d.rectangle([(0, h - 14), (150, h)], fill=(0, 0, 0))
        d.text((3, h - 13), f, fill=(255, 80, 80))
        ims.append(im)
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * w, rows * h), (20, 20, 20))
    for i, im in enumerate(ims):
        canvas.paste(im, ((i % cols) * w, (i // cols) * h))
    canvas.save(out)
    return Path(out)


def save_observations(obs: list[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(obs, indent=1), encoding='utf-8')


def load_observations(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding='utf-8'))
