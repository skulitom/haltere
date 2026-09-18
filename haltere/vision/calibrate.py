"""Calibrate the FPV camera's focal length (and check its tilt) from a flight dataset.

Consecutive dataset frames come with the drone's attitude. Distant features (sky, far scenery) move in
the image almost purely because of the rotation between the two frames, so the focal length that
makes the rotated first-frame rays land on the matched second-frame features is the camera's.
Sparse ORB features are matched between frame pairs; for each candidate (f, tilt) the median angular
error of the prediction is scored and the best pair is reported.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .camera import Camera, quat_wxyz_to_mat


def load_index(dataset: str | Path) -> list[dict]:
    rows = []
    with open(Path(dataset) / 'index.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (v if k == 'file' else float(v)) for k, v in r.items()})
    return rows


def _matches(img_a, img_b, orb, matcher, top_frac: float = 0.65):
    """ORB matches restricted to the upper part of the image (far scenery), as (N, 2), (N, 2) pixel arrays."""
    import cv2
    h = img_a.shape[0]
    mask = np.zeros(img_a.shape[:2], dtype=np.uint8)
    mask[: int(h * top_frac)] = 255
    ka, da = orb.detectAndCompute(img_a, mask)
    kb, db = orb.detectAndCompute(img_b, mask)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None
    m = matcher.knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 8:
        return None
    pa = np.array([ka[g.queryIdx].pt for g in good])
    pb = np.array([kb[g.trainIdx].pt for g in good])
    return pa, pb


def angular_errors(pa, pb, R_rel_body, cam: Camera) -> np.ndarray:
    """Angle between the rotated first-frame rays and the second-frame rays (radians, per match).
    R_rel_body takes body-frame vectors of frame a to body-frame vectors of frame b."""
    ra = cam.unproject_body(pa)                 # body frame of frame a
    rb_pred = ra @ R_rel_body.T                 # same rays expressed in frame b's body
    rb = cam.unproject_body(pb)
    cosang = np.clip((rb_pred * rb).sum(1), -1.0, 1.0)
    return np.arccos(cosang)


def calibrate(dataset: str | Path, step: int = 2, max_pairs: int = 150, min_rot_deg: float = 1.5,
              f_grid=None, tilt_grid=None, verbose: bool = True) -> dict:
    import cv2
    rows = load_index(dataset)
    frames_dir = Path(dataset) / 'frames'
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=12)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = []
    for i in range(0, len(rows) - step, step):
        a, b = rows[i], rows[i + step]
        qa = np.array([a['qw'], a['qx'], a['qy'], a['qz']]); qb = np.array([b['qw'], b['qx'], b['qy'], b['qz']])
        Ra, Rb = quat_wxyz_to_mat(qa), quat_wxyz_to_mat(qb)      # world-from-body
        R_rel = Rb.T @ Ra                                         # body_a -> body_b
        ang = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
        if ang < min_rot_deg or ang > 25.0:
            continue
        ia = cv2.imread(str(frames_dir / a['file']), cv2.IMREAD_GRAYSCALE)
        ib = cv2.imread(str(frames_dir / b['file']), cv2.IMREAD_GRAYSCALE)
        if ia is None or ib is None:
            continue
        m = _matches(ia, ib, orb, matcher)
        if m is None:
            continue
        pairs.append((m[0], m[1], R_rel, ang))
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        raise RuntimeError('no usable frame pairs (need rotation between consecutive frames and matchable features)')
    h, w = ia.shape
    f_grid = np.arange(150, 900, 5.0) if f_grid is None else np.asarray(f_grid, dtype=float)
    tilt_grid = [30.0] if tilt_grid is None else list(tilt_grid)
    best = None
    table = {}
    for tilt in tilt_grid:
        for f in f_grid:
            cam = Camera(w, h, float(f), float(tilt))
            errs = [np.median(angular_errors(pa, pb, R, cam)) for pa, pb, R, _ in pairs]
            score = float(np.median(errs))
            table[(tilt, float(f))] = score
            if best is None or score < best[0]:
                best = (score, float(f), float(tilt))
    score, f, tilt = best
    cam = Camera(w, h, f, tilt)
    if verbose:
        print(f'{len(pairs)} frame pairs, rotation {np.mean([p[3] for p in pairs]):.1f} deg on average')
        print(f'best focal length {f:.0f} px at {w}x{h} (horizontal FOV {cam.hfov_deg:.1f} deg), tilt {tilt:.0f} deg, '
              f'median angular error {np.degrees(score):.2f} deg')
    return {'f': f, 'tilt_deg': tilt, 'width': w, 'height': h, 'hfov_deg': cam.hfov_deg, 'error_deg': float(np.degrees(score)),
            'pairs': len(pairs), 'table': table}
