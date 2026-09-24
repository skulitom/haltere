"""L2: hindsight triangulation fused into per-flight voxel maps. OFFLINE ONLY (uses future frames).

Pipeline per flight (store frames of one run, time order; tracks break at frame gaps > MAX_GAP_S
and at position jumps > MAX_JUMP_M, i.e. resets):

1. Tracks. Chained pyramidal Lucas-Kanade tracks over consecutive store frames with the LK
   parameters of haltere.vision.temporal_depth.TemporalDepth (21 x 21, 3 levels, forward-backward
   error < 1 px), seeded with the rotation-only prediction from the logged attitudes, on a
   CLAHE-equalised grey image (dark scenes); a step is kept only within EPIPOLAR_PX of the
   epipolar line given by the logged poses (static-scene check that also drops ghost racers and
   other moving objects the overlay mask misses). Up to MAX_CORNERS corners are kept alive; new ones are
   detected per image tile (goodFeaturesToTrack, quality QUALITY relative to the tile, min distance
   5 px, block 5) so that dark or low-contrast regions also get features. Features are detected and
   kept only on scene pixels: ``feature_mask`` removes HUD glyphs, the checkpoint ring and cyan
   volumes, ghost trails (haltere.obstacles.overlays) and the propeller zone, dilated by
   MASK_DILATE_PX; a point that lands on a masked pixel ends its track. Tracks that stay put in the image
   while the camera turns (``camera_fixed``: HUD glyph halos the mask misses, lens flare, vignetting) are
   dropped with all their observations.
2. Multi-baseline triangulation, forward AND backward. For every observation of a track in
   frame a and every keyframe offset in KEYFRAME_OFFSETS_S (-2, -1, -0.5, +0.5, +1, +2 s),
   the track's observation in the frame b nearest t_a + offset is intersected with the one
   in a by haltere.vision.temporal_depth.triangulate_motion (reused unchanged: telemetry
   poses, parallax >= 0.5 deg, ray gap <= 0.1 m, range 0.25-30 m, first-order image-noise
   sigma with 1 px at 448 px) and must reproject within THIRD_VIEW_PX into the frame midway
   between a and b (the third-view check of TemporalDepth). The lowest-sigma estimate with
   sigma < MAX_SIGMA_FRACTION * range (the precise-keyframe rule of MultiBaselineDepth) is
   kept per (track, frame). The neighbouring store frames (ADJACENT_STEPS) are partners too:
   at the store's ~6 Hz and 10-20 m/s, LK tracks rarely survive the three frames to the
   +-0.5 s keyframe (mean track length 1.1 frames on fast Pine/Minus captures), while one step
   already gives metres of baseline. An adjacent-pair estimate must pass the third-view check on
   the other neighbour when the track reaches it; otherwise it is kept as a two-view estimate.
3. Fusion into VOXEL_M voxels, one frame (launch-relative FLU) per flight. Occupied: voxels
   holding estimates of >= MIN_OCC_TRACKS tracks or from >= MIN_OCC_FRAMES reference frames.
   Visibility carving: every kept estimate marks the voxels along its line of sight free up to
   the point minus max(2 sigma, 2 voxels). A voxel carved more than FREE_OVER_OCC times its
   occupied count is not occupied (moving/ghost objects and outliers are carved away).
4. The flown path is free: occupied voxels within PATH_CLEAR_M of it are removed and voxels within
   PATH_FREE_M are marked free (except near contacts), which also removes ghost-trail points that
   lie on the racing line (``clear_flown_path``).
5. Projection into every frame of the flight (``hindsight_labels``), per sub-ray of
   corridors.SUB_SHAPE (3 x 3 per grid cell):

   - e = distance to which the sub-ray stays in carved-free voxels from the camera, marched in
     MARCH_STEP_M steps up to FREE_MAX_M (the first VOXEL_M around the camera counts as free);
   - hits: every occupied voxel mean point within MARCH_MAX_M is splatted onto the sub-rays within
     SPLAT_RADIUS_M of it (at least the nearest one); the nearest one per sub-ray is the hit s;
   - hit and e >= s - max(0.1 s, 2 voxels) -> EXACT s; own-frame track points (the feature was seen
     in this frame, so its line of sight is free) are EXACT on their nearest sub-ray unless a map
     point is clearly nearer; other hits UPPER s; no hit and e >= MIN_LOWER_M -> LOWER e; else UNKNOWN.

   Cells take labels.min_over of their sub-rays (strict: a cell with some UNKNOWN sub-rays gets at
   most UPPER); a cell whose minimum is only bracketed (a hit on some sub-rays, a shorter carved extent
   on others) keeps UPPER hit within WIDE_UPPER_MAX_M (8 m) and LOWER extent beyond
   (labels.resolve_interval). Fan corridors: occupied voxel means (and own-frame points) within
   FAN_CORRIDOR_RADIUS_M of the axis give the hit distance; the corridor is observed free to e when,
   at every MARCH_STEP_M slice, the axis sample and at least FAN_FREE_FRACTION of the cross-section
   samples (axis, 8 at 0.25 m, 8 at 0.5 m) are carved free; then corridors.fan_constraints.

The FAN_FREE_FRACTION rule is an approximation (sparse lines of sight cannot certify a whole
0.5 m cylinder); it is recorded in the label manifest. Unreliable alignments get no L2.
Chunked per flight; ChunkGuard.before_chunk() before each flight and every CHUNK_FRAMES frames.

K0b checks (labels/build.py 'quality'): L2 vs colliders median |err|/r <= 10 % at 2-10 m;
L2 vs L3 within 15 % on >= 80 % of frames 2.0-0.5 s before impact where both exist; >= 30 % of
near-travel fan cells (<= 8 m, +-20 deg) uncensored in the Minus Two and Pine Valley test sets.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import contract
from ..contract import FAN_CORRIDOR_RADIUS_M, FAN_MAX_M, FAN_SHAPE, IMAGE_H, IMAGE_W
from . import EXACT_TOL, WIDE_UPPER_MAX_M, LabelKind, clip_fan
from .corridors import (SUB, SUB_SHAPE, fan_constraints, grid_from_subrays, subray_dirs_world, unknown_fan,
                        unknown_grid, unknown_subrays)

VOXEL_M = 0.2
KEYFRAME_OFFSETS_S = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
ADJACENT_STEPS = (-1, 1)      # also the neighbouring store frames (short baselines; see module doc)
MAX_SIGMA_FRACTION = 0.1
PARTNER_TOL = 0.35            # partner frame within 0.35 |offset| of t_a + offset
MAX_GAP_S = 0.45              # tracks break at larger frame gaps
MAX_JUMP_M = 3.0              # ... and at position jumps (resets)
MAX_CORNERS = 800
MIN_DISTANCE_PX = 5
QUALITY = 0.005               # goodFeaturesToTrack quality, relative to the strongest corner of each tile
DETECT_TILES = (3, 6)         # detection tiles (rows, cols): corners spread over the image
CLAHE_CLIP = 2.0              # contrast-limited equalisation of the grey image before tracking (dark scenes)
MASK_DILATE_PX = 2            # features keep this distance from masked overlay pixels
FB_MAX_PX = 1.0
SEED_DEPTHS_M = (np.inf,)     # LK seeds: static point at these depths (inf = rotation only); finite-depth seeds
                              # were tried (24, 10, 5, 2.5 m) and added no tracks at the store frame rate
EPIPOLAR_PX = 6.0             # a tracked point must stay this close to its epipolar line (logged poses)
CAM_FIXED_PX = 1.0            # a track that stays this close to its start ...
CAM_FIXED_MIN_ROT_PX = 3.0    # ... while the camera turned enough to move a point at infinity this far is camera-fixed
                              # (a static pixel under a 3 px / 1.2 deg turn triangulates to baseline / 0.021 m: 4 m
                              # at the box course's 0.1-0.2 m frame baselines)
THIRD_VIEW_PX = 2.0          # TemporalDepth's third-view limit (2 px); here at 448 px
PIXEL_SIGMA = 1.0
MIN_PARALLAX_DEG = 0.5
MAX_RANGE_M = 30.0
MIN_OCC_TRACKS = 2            # occupied voxel: estimates of >= 2 distinct tracks ...
MIN_OCC_FRAMES = 2            # ... or in >= 2 reference frames (3 frames: 40-60 % fewer occupied voxels on
                              # the box course and Straw for UPPER error 5.7 % instead of 6.6 %; not worth it)
CARVE_SUPPORTED_ONLY = False  # carve along lines of sight to every estimate (supported-only halves free space
                              # on the box course for no measurable gain: LOWER violations 0.0000 vs 0.0002)
FREE_OVER_OCC = 2.0
MARCH_STEP_M = 0.2           # free-space march step (= voxel size)
FREE_MAX_M = 20.0             # free extents are marched to this distance (LOWER bounds beyond are not claimed)
MARCH_MAX_M = 30.0            # occupied points are projected to this range
SPLAT_RADIUS_M = 0.14         # an occupied voxel covers the sub-rays within this radius of its mean point
SPLAT_MAX_SUBRAYS = 4
MIN_LOWER_M = 1.0
FAN_FREE_FRACTION = 0.6
PATH_CLEAR_M = 0.25           # occupied voxels this close to the flown path are removed (the drone flew there)
PATH_FREE_M = 0.15            # voxel centres this close to the flown path are free (inside the 0.35 m envelope)
PATH_STOP_MARGIN_S = 0.5      # ... except within this time of a contact / impact / reset
CHUNK_FRAMES = 400
BLOCK = 8                     # voxels per block side of the sparse map (1.6 m blocks)
LK = dict(winSize=(21, 21), maxLevel=3)
_KEY_OFF = 1 << 20

RULES = dict(
    tracks=('chained pyramidal LK (TemporalDepth parameters), rotation-seeded, fwd-bwd < 1 px, '
            f'epipolar distance <= {EPIPOLAR_PX} px (logged poses), overlay-masked; camera-fixed tracks (within '
            f'{CAM_FIXED_PX} px of their start after a turn that moves infinity by > {CAM_FIXED_MIN_ROT_PX} px) dropped'),
    triangulation=('triangulate_motion (unchanged) against partner frames at t +- 0.5/1/2 s and the neighbouring '
                   f'store frames, third-view reprojection <= {THIRD_VIEW_PX} px (adjacent pairs: when the track has a '
                   f'third view), lowest sigma per (track, frame), sigma < {MAX_SIGMA_FRACTION} r'),
    fusion=(f'{VOXEL_M} m voxels; occupied with estimates of >= {MIN_OCC_TRACKS} tracks or from >= {MIN_OCC_FRAMES} '
            f'reference frames and carved <= {FREE_OVER_OCC} x occupied count; lines of sight of '
            f'{"supported" if CARVE_SUPPORTED_ONLY else "all"} estimates carved free up to point - max(2 sigma, 2 voxels)'),
    grid=('sub-ray march; first occupied voxel or own-frame point = hit; EXACT if carved free to within '
          'max(10 %, 2 voxels) of the hit or an own-frame observation, else UPPER; no hit: LOWER carved extent '
          f'if >= {MIN_LOWER_M} m; cells = strict min over 3 x 3 sub-rays, a bracketed cell minimum keeps UPPER '
          f'hit when <= {WIDE_UPPER_MAX_M} m, else LOWER extent'),
    fan=(f'occupied points within {FAN_CORRIDOR_RADIUS_M} m of the axis = hit; observed free while the axis and '
         f'>= {FAN_FREE_FRACTION:.0%} of 17 cross-section samples are carved free (approximation); free extents '
         f'< {MIN_LOWER_M} m dropped'),
)


# ----------------------------------------------------------------------------- masks

def _box_mask_448() -> np.ndarray:
    """Conservative fallback: survey fixed-HUD boxes + propeller boxes (1280 x 720 units) at 448 x 252."""
    boxes = [(24, 36, 112, 72), (520, 20, 780, 84), (500, 104, 780, 150), (1130, 36, 1280, 150),
             (1040, 250, 1280, 480), (424, 170, 446, 560), (834, 170, 856, 560), (500, 300, 780, 400),
             (616, 400, 664, 720), (530, 610, 750, 720), (0, 680, 90, 720), (1180, 650, 1280, 720),
             (0, 420, 400, 700), (880, 420, 1280, 700)]
    m = np.zeros((IMAGE_H, IMAGE_W), bool)
    sx, sy = IMAGE_W / 1280.0, IMAGE_H / 720.0
    for x0, y0, x1, y1 in boxes:
        m[int(np.floor(y0 * sy)):int(np.ceil(y1 * sy)), int(np.floor(x0 * sx)):int(np.ceil(x1 * sx))] = True
    return m


_FALLBACK = None
_PROVENANCE = None
_OVERLAYS_FILE = None


def use_overlays_file(path) -> str:
    """Offline builds only: take the overlay masks from ``path`` (an overlays.py whose assets sit in
    ``<path>/../../../configs/obstacles``, e.g. a pinned snapshot of the overlays branch) instead of the
    package module, so the label feature mask does not change while that module is still being developed.
    Returns the new mask provenance (module + asset hashes; it enters the hindsight stage input hash)."""
    import importlib.util
    import sys
    global _PROVENANCE, _OVERLAYS_FILE
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location('haltere.obstacles.overlays', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['haltere.obstacles.overlays'] = mod
    spec.loader.exec_module(mod)
    from ... import obstacles as _pkg
    _pkg.overlays = mod
    _PROVENANCE, _OVERLAYS_FILE = None, str(path).replace('\\', '/')
    return mask_provenance()


def mask_provenance() -> str:
    """Which feature mask the labels use: 'overlays sha256=<module + asset files>' or the fallback rule."""
    global _PROVENANCE
    if _PROVENANCE is None:
        import hashlib
        try:
            from .. import overlays
            overlays.overlay_masks(np.zeros((IMAGE_H, IMAGE_W, 3), np.uint8))
            h = hashlib.sha256(Path(overlays.__file__).read_bytes())
            adir = getattr(overlays, 'ASSET_DIR', None)
            for name in (getattr(overlays, 'STATIC_HUD_ASSET', None), getattr(overlays, 'PROPELLER_ASSET', None)):
                if adir and name and (Path(adir) / name).exists():
                    h.update((Path(adir) / name).read_bytes())
            _PROVENANCE = f'overlays sha256={h.hexdigest()[:16]}'
        except (NotImplementedError, FileNotFoundError, OSError, ImportError, AttributeError):
            _PROVENANCE = 'fallback: survey HUD + propeller boxes and ring colour (overlays unavailable)'
    return _PROVENANCE


def feature_mask(rgb: np.ndarray) -> tuple[np.ndarray, str]:
    """(252, 448) bool: True where features must NOT be detected/kept, and the mask provenance.

    Uses haltere.obstacles.overlays (HUD glyphs, ring, ghost trails, propeller zone). When that
    module or its assets are unavailable, falls back to the survey boxes plus the ring colour rule.
    """
    global _FALLBACK
    try:
        from .. import overlays
        m = overlays.overlay_masks(rgb)
        return np.asarray(m.masked | m.propeller, bool), 'overlays'
    except (NotImplementedError, FileNotFoundError, OSError, ImportError, AttributeError):
        pass
    import cv2
    if _FALLBACK is None:
        _FALLBACK = _box_mask_448()
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    ring = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 100) & (hsv[..., 1] >= 75) & (hsv[..., 2] >= 75)
    ring = cv2.dilate(ring.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return _FALLBACK | ring, 'fallback boxes + ring colour'


# ----------------------------------------------------------------------------- tracking

def _unproject_c(uv):
    uv = np.asarray(uv, np.float64)
    d = np.stack([(uv[:, 0] - IMAGE_W / 2) / contract.FOCAL_PX, (uv[:, 1] - IMAGE_H / 2) / contract.FOCAL_PX,
                  np.ones(len(uv))], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def segment_breaks(t, pos) -> np.ndarray:
    """(n,) bool: frame k starts a new tracking segment (first frame, time gap or position jump)."""
    t = np.asarray(t, np.float64)
    pos = np.asarray(pos, np.float64)
    b = np.ones(len(t), bool)
    if len(t) > 1:
        dt = np.diff(t)
        jump = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        b[1:] = (dt > MAX_GAP_S) | (dt <= 0) | (jump > MAX_JUMP_M) | ~np.isfinite(jump)
    return b


def _predict(pts, p0, p1, R0, R1, depth: float):
    """Pixel in the next frame of a static point seen at ``pts`` in the previous one at ``depth`` m (inf = rotation only)."""
    d_w = _unproject_c(pts) @ R0.T
    d_c = d_w @ R1 if np.isinf(depth) else (p0[None] + d_w * depth - p1[None]) @ R1
    return contract.project_camera(d_c)


def epipolar_distance(pts0, pts1, p0, p1, R0, R1) -> np.ndarray:
    """Pixel distance of ``pts1`` (next frame) from the epipolar lines of ``pts0`` (previous frame), known poses.

    The line joins the epipole (previous camera centre seen from the next one) and the vanishing point of the
    previous ray, both as homogeneous image points (valid for either sign of their depth). A baseline below
    1 cm leaves only the vanishing point: the distance to it is returned.
    """
    K = np.array([[contract.FOCAL_PX, 0.0, IMAGE_W / 2.0], [0.0, contract.FOCAL_PX, IMAGE_H / 2.0], [0.0, 0.0, 1.0]])
    v = (_unproject_c(pts0) @ R0.T @ R1) @ K.T                    # vanishing points (homogeneous)
    x1 = np.c_[np.asarray(pts1, np.float64), np.ones(len(pts1))]
    base = np.asarray(p0, np.float64) - np.asarray(p1, np.float64)
    if np.linalg.norm(base) < 0.01:
        vz = np.where(np.abs(v[:, 2:3]) > 1e-9, v[:, 2:3], 1e-9)
        return np.linalg.norm(v[:, :2] / vz - x1[:, :2], axis=1)
    e = K @ (base @ R1)                                         # epipole (homogeneous)
    line = np.cross(np.broadcast_to(e, v.shape), v)
    return np.abs((line * x1).sum(1)) / np.maximum(np.hypot(line[:, 0], line[:, 1]), 1e-12)


def _track_step(cv2, prev_grey, grey, pts, p0, p1, R0, R1, bad):
    """One LK step with pose-seeded hypotheses: (new positions (n, 2), accepted (n,) bool).

    Seeds: the point's pixel in the new frame if it were static at each SEED_DEPTHS_M depth (inf first:
    rotation only). A point keeps the first seed whose LK result passes status, forward-backward < FB_MAX_PX,
    stays on an unmasked pixel and lies within EPIPOLAR_PX of its epipolar line (static-scene check with the
    logged poses). Fast flight at the store's ~6 Hz moves near features by tens of pixels beyond the
    rotation-only prediction; finite-depth seeds recover them.
    """
    n = len(pts)
    out = pts.astype(np.float64).copy()
    good = np.zeros(n, bool)
    todo = np.arange(n)
    for depth in SEED_DEPTHS_M:
        if len(todo) == 0:
            break
        pred, okp = _predict(pts[todo], p0, p1, R0, R1, depth)
        okp &= contract.in_image(pred, margin_px=-LK['winSize'][0])
        sel = todo[okp]
        if len(sel) == 0:
            continue
        seed = pred[okp].astype(np.float32).reshape(-1, 1, 2)
        q1, st, _ = cv2.calcOpticalFlowPyrLK(prev_grey, grey, pts[sel].reshape(-1, 1, 2), seed,
                                             flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01), **LK)
        q0b, st2, _ = cv2.calcOpticalFlowPyrLK(grey, prev_grey, q1, pts[sel].reshape(-1, 1, 2).copy(),
                                               flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **LK)
        q1, q0b = q1.reshape(-1, 2), q0b.reshape(-1, 2)
        fin = np.isfinite(q1).all(1) & np.isfinite(q0b).all(1)
        xy = np.rint(np.nan_to_num(q1, nan=-1)).astype(int)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < IMAGE_W) & (xy[:, 1] >= 0) & (xy[:, 1] < IMAGE_H)
        allowed = np.zeros(len(sel), bool)
        allowed[inside] = ~bad[xy[inside, 1], xy[inside, 0]]
        ok = (fin & inside & allowed & (st.ravel() > 0) & (st2.ravel() > 0)
              & (np.linalg.norm(q0b - pts[sel], axis=1) < FB_MAX_PX))
        if ok.any() and EPIPOLAR_PX is not None:
            j = np.flatnonzero(ok)
            ok[j] = epipolar_distance(pts[sel[j]], q1[j], p0, p1, R0, R1) <= EPIPOLAR_PX
        out[sel[ok]] = q1[ok]
        good[sel[ok]] = True
        todo = np.setdiff1d(todo, sel[ok], assume_unique=True)
    return out, good


@dataclass
class Tracks:
    """Observations of chained tracks: one row per (track, frame)."""
    track: np.ndarray     # (m,) int64
    frame: np.ndarray     # (m,) int32 (position in the run's time-ordered frame list)
    uv: np.ndarray        # (m, 2) float32
    breaks: np.ndarray    # (n,) bool
    mask_source: str = ''

    def by_frame(self):
        order = np.lexsort((self.track, self.frame))
        f = self.frame[order]
        starts = np.searchsorted(f, np.arange(len(self.breaks) + 1))
        return order, starts


def camera_fixed(uv0, R0, uv, R) -> np.ndarray:
    """(n,) bool: tracks that stayed within CAM_FIXED_PX of where they started although the camera turned enough to
    move a world point at infinity by more than CAM_FIXED_MIN_ROT_PX: features fixed to the camera (HUD glyph halos
    the overlay mask misses, lens flare, vignetting), not to the world. Such tracks triangulate to spurious surfaces a
    few metres from the camera (box course: collider-free sky cells at 3-5 m next to the HUD)."""
    uv0 = np.asarray(uv0, np.float64)
    if len(uv0) == 0:
        return np.zeros(0, bool)
    d_w = np.einsum('nij,nj->ni', R0, _unproject_c(uv0))        # camera -> world ray at the track's first frame
    pred, ok = contract.project_camera(d_w @ R)                 # the same (infinitely far) ray seen now
    turned = ~ok | (np.linalg.norm(pred - uv0, axis=1) > CAM_FIXED_MIN_ROT_PX)
    return turned & (np.linalg.norm(np.asarray(uv, np.float64) - uv0, axis=1) < CAM_FIXED_PX)


def track_run(frames, t, pos, quat, *, mask_fn=None, guard=None, log=None) -> Tracks:
    """Chained LK tracks over a time-ordered frame sequence (n, 252, 448, 3) RGB with poses. Tracks found fixed to
    the camera (``camera_fixed``) are dropped with all their observations."""
    import cv2
    n = len(frames)
    breaks = segment_breaks(t, pos)
    R_wc = contract.camera_to_world(np.asarray(quat, np.float64))
    tr_ids, tr_frames, tr_uv = [], [], []
    prev_grey, pts, ids = None, np.zeros((0, 2), np.float32), np.zeros(0, np.int64)
    first_uv, first_k = np.zeros((0, 2), np.float32), np.zeros(0, np.int64)     # where/when each live track began
    fixed_ids = []
    next_id = 0
    src = ''
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8)) if CLAHE_CLIP else None
    for k in range(n):
        if guard is not None and k and k % CHUNK_FRAMES == 0:
            guard.before_chunk()
        rgb = np.asarray(frames[k])
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if clahe is not None:
            grey = clahe.apply(grey)
        bad, src = (mask_fn or feature_mask)(rgb)
        if MASK_DILATE_PX:
            bad = cv2.dilate(bad.astype(np.uint8), np.ones((2 * MASK_DILATE_PX + 1,) * 2, np.uint8)) > 0
        if breaks[k] or prev_grey is None or len(pts) == 0:
            pts, ids = np.zeros((0, 2), np.float32), np.zeros(0, np.int64)
            first_uv, first_k = np.zeros((0, 2), np.float32), np.zeros(0, np.int64)
        else:
            p1, good = _track_step(cv2, prev_grey, grey, pts, pos[k - 1], pos[k], R_wc[k - 1], R_wc[k], bad)
            pts, ids = p1[good].astype(np.float32), ids[good]
            first_uv, first_k = first_uv[good], first_k[good]
            fixed = camera_fixed(first_uv, R_wc[first_k], pts, R_wc[k])
            if fixed.any():
                fixed_ids.append(ids[fixed])
                keep = ~fixed
                pts, ids, first_uv, first_k = pts[keep], ids[keep], first_uv[keep], first_k[keep]
        # replenish features on scene pixels away from live tracks, tile by tile (uniform coverage)
        room = MAX_CORNERS - len(pts)
        if room > 20:
            allow = (~bad).astype(np.uint8) * 255
            if len(pts):
                q = np.rint(pts).astype(int)
                for (x, y) in q:
                    cv2.circle(allow, (int(x), int(y)), MIN_DISTANCE_PX, 0, -1)
            found = []
            th, tw = IMAGE_H // DETECT_TILES[0], IMAGE_W // DETECT_TILES[1]
            per = max(2, int(room // (DETECT_TILES[0] * DETECT_TILES[1])))
            for ty in range(DETECT_TILES[0]):
                for tx in range(DETECT_TILES[1]):
                    sl = (slice(ty * th, (ty + 1) * th), slice(tx * tw, (tx + 1) * tw))
                    if not allow[sl].any():
                        continue
                    new = cv2.goodFeaturesToTrack(np.ascontiguousarray(grey[sl]), maxCorners=per, qualityLevel=QUALITY,
                                                  minDistance=MIN_DISTANCE_PX, mask=np.ascontiguousarray(allow[sl]),
                                                  blockSize=5)
                    if new is not None and len(new):
                        found.append(new.reshape(-1, 2) + np.array([tx * tw, ty * th], np.float32))
            if found:
                new = np.concatenate(found).astype(np.float32)
                pts = np.vstack([pts, new])
                ids = np.r_[ids, np.arange(next_id, next_id + len(new))]
                first_uv = np.vstack([first_uv, new])
                first_k = np.r_[first_k, np.full(len(new), k, np.int64)]
                next_id += len(new)
        tr_ids.append(ids.copy())
        tr_frames.append(np.full(len(ids), k, np.int32))
        tr_uv.append(pts.copy())
        prev_grey = grey
    if not tr_ids:
        return Tracks(np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros((0, 2), np.float32), breaks, src)
    ids, frames_, uv = np.concatenate(tr_ids), np.concatenate(tr_frames), np.concatenate(tr_uv)
    if fixed_ids:
        keep = ~np.isin(ids, np.concatenate(fixed_ids))
        ids, frames_, uv = ids[keep], frames_[keep], uv[keep]
    return Tracks(ids, frames_, uv, breaks, src)


# ----------------------------------------------------------------------------- triangulation

@dataclass
class Estimates:
    """Accepted hindsight depth estimates, one per (track, reference frame)."""
    frame: np.ndarray      # (k,) int32 reference frame
    track: np.ndarray      # (k,) int64
    uv: np.ndarray         # (k, 2) float32 pixel in the reference frame
    point: np.ndarray      # (k, 3) float64 world point (launch-relative FLU)
    range_m: np.ndarray    # (k,) float64 range from the reference camera
    sigma_m: np.ndarray    # (k,) float64
    offset_s: np.ndarray   # (k,) float32 keyframe offset that gave the estimate

    @staticmethod
    def empty():
        return Estimates(np.zeros(0, np.int32), np.zeros(0, np.int64), np.zeros((0, 2), np.float32),
                         np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0, np.float32))


def _segment_id(breaks):
    return np.cumsum(breaks) - 1


def triangulate(tracks: Tracks, t, pos, quat, *, offsets=KEYFRAME_OFFSETS_S) -> Estimates:
    """Forward and backward multi-baseline triangulation of every track observation (see module doc)."""
    from ...vision.temporal_depth import triangulate_motion
    cam = contract.store_camera()
    t = np.asarray(t, np.float64)
    pos = np.asarray(pos, np.float64)
    q = np.asarray(quat, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    R_wc = contract.camera_to_world(q)
    n = len(t)
    seg = _segment_id(tracks.breaks)
    order, starts = tracks.by_frame()
    ids_f = [tracks.track[order[starts[k]:starts[k + 1]]] for k in range(n)]
    uv_f = [tracks.uv[order[starts[k]:starts[k + 1]]] for k in range(n)]
    out_rows = []
    for a in range(n):
        if len(ids_f[a]) == 0:
            continue
        cand = []
        partners = []
        for off in offsets:
            target = t[a] + off
            b = int(np.argmin(np.abs(t - target)))
            if b == a or abs(t[b] - target) > PARTNER_TOL * abs(off) or seg[b] != seg[a]:
                continue
            partners.append((off, b))
        for step in ADJACENT_STEPS:
            b = a + step
            if 0 <= b < n and seg[b] == seg[a] and all(b != p[1] for p in partners):
                partners.append((float(t[b] - t[a]), b))
        for off, b in partners:
            _, ia, ib = np.intersect1d(ids_f[a], ids_f[b], assume_unique=True, return_indices=True)
            if len(ia) < 1:
                continue
            res = triangulate_motion(cam, uv_f[b][ib], uv_f[a][ia], pos[b], q[b], pos[a], q[a],
                                     min_parallax_deg=MIN_PARALLAX_DEG, max_range=MAX_RANGE_M,
                                     pixel_sigma=PIXEL_SIGMA)
            ok = res['valid'] & (res['range_sigma_m'] < MAX_SIGMA_FRACTION * res['range_m'])
            if not ok.any():
                continue
            # third view: the frame midway between a and b (by index) must see the point where it was tracked;
            # for an adjacent partner, the neighbour of a on the other side, when the track reaches it
            adjacent = abs(b - a) == 1
            c = (a + b) // 2 if not adjacent else a - (b - a)
            if c in (a, b) or c < 0 or c >= n or seg[c] != seg[a]:
                c = None
            third = np.zeros(len(ia), bool)
            seen = np.zeros(len(ia), bool)
            if c is not None:
                _, ic_a, ic = np.intersect1d(ids_f[a][ia], ids_f[c], assume_unique=True, return_indices=True)
                if len(ic_a):
                    P = res['position_world'][ic_a]
                    d_c = (P - pos[c]) @ R_wc[c]
                    px, front = contract.project_camera(d_c)
                    resid = np.linalg.norm(px - uv_f[c][ic], axis=1)
                    third[ic_a] = front & (resid <= THIRD_VIEW_PX)
                    seen[ic_a] = True
            # adjacent pairs: a two-view estimate is kept when the track has no third observation
            ok &= third | (adjacent & ~seen)
            if ok.any():
                cand.append((off, ia[ok], res['position_world'][ok], res['range_m'][ok], res['range_sigma_m'][ok]))
        if not cand:
            continue
        # lowest sigma per track across offsets
        off_a = np.concatenate([np.full(len(c[1]), c[0], np.float32) for c in cand])
        ia_a = np.concatenate([c[1] for c in cand])
        P_a = np.concatenate([c[2] for c in cand])
        r_a = np.concatenate([c[3] for c in cand])
        s_a = np.concatenate([c[4] for c in cand])
        o = np.lexsort((s_a, ia_a))
        first = np.r_[True, ia_a[o][1:] != ia_a[o][:-1]]
        sel = o[first]
        out_rows.append((np.full(len(sel), a, np.int32), ids_f[a][ia_a[sel]], uv_f[a][ia_a[sel]], P_a[sel], r_a[sel],
                         s_a[sel], off_a[sel]))
    if not out_rows:
        return Estimates.empty()
    cols = list(zip(*out_rows))
    return Estimates(*(np.concatenate(c) for c in cols))


# ----------------------------------------------------------------------------- voxel map

def voxel_index(p) -> np.ndarray:
    return np.floor(np.asarray(p, np.float64) / VOXEL_M).astype(np.int64)


def voxel_key(ijk) -> np.ndarray:
    ijk = np.asarray(ijk, np.int64) + _KEY_OFF
    return (ijk[..., 0] << 42) | (ijk[..., 1] << 21) | ijk[..., 2]


@dataclass
class VoxelMap:
    """Per-flight fused hindsight map (launch-relative FLU)."""
    occ_key: np.ndarray        # (m,) sorted int64 voxel keys of occupied voxels
    occ_pos: np.ndarray        # (m, 3) mean point
    occ_n: np.ndarray          # (m,) estimates
    occ_frames: np.ndarray     # (m,) distinct reference frames
    occ_sigma: np.ndarray      # (m,) min sigma
    free_key: np.ndarray       # (f,) sorted int64 keys of carved-free voxels (not occupied)
    meta: dict = field(default_factory=dict)
    _blocks: tuple | None = None

    def save(self, path) -> None:
        """Atomic write (a killed job never leaves a half-written map behind)."""
        import os
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, occ_key=self.occ_key, occ_pos=self.occ_pos.astype(np.float32), occ_n=self.occ_n,
                            occ_frames=self.occ_frames, occ_sigma=self.occ_sigma.astype(np.float32),
                            free_key=self.free_key, meta=np.array(repr(self.meta)))
        os.replace(tmp, path)

    @staticmethod
    def load(path) -> 'VoxelMap':
        d = np.load(path, allow_pickle=False)
        import ast
        return VoxelMap(d['occ_key'], d['occ_pos'].astype(np.float64), d['occ_n'], d['occ_frames'],
                        d['occ_sigma'].astype(np.float64), d['free_key'], ast.literal_eval(str(d['meta'])))

    # Sparse dense-block layout: a coarse grid of block ids over the map's bounding box and
    # BLOCK^3 uint8 states per touched block (0 unknown, 1 free, 2 occupied; last block = empty).
    def blocks(self):
        if self._blocks is None:
            occ_ijk, free_ijk = key_to_ijk(self.occ_key), key_to_ijk(self.free_key)
            all_b = np.floor_divide(np.r_[occ_ijk, free_ijk], BLOCK)
            if len(all_b) == 0:
                self._blocks = (np.zeros(3, np.int64), np.full((1, 1, 1), 0, np.int32),
                                np.zeros((1, BLOCK, BLOCK, BLOCK), np.uint8))
                return self._blocks
            bmin = all_b.min(axis=0)
            dims = all_b.max(axis=0) - bmin + 1
            ub, binv = np.unique(all_b - bmin, axis=0, return_inverse=True)
            coarse = np.full(tuple(dims), len(ub), np.int32)
            coarse[ub[:, 0], ub[:, 1], ub[:, 2]] = np.arange(len(ub), dtype=np.int32)
            state = np.zeros((len(ub) + 1, BLOCK, BLOCK, BLOCK), np.uint8)
            binv = binv.reshape(-1)
            nocc = len(occ_ijk)
            for sl, val in ((slice(nocc, None), 1), (slice(0, nocc), 2)):
                ijk = np.r_[occ_ijk, free_ijk][sl]
                if len(ijk):
                    loc = np.mod(ijk, BLOCK)
                    state[binv[sl], loc[:, 0], loc[:, 1], loc[:, 2]] = val
            self._blocks = (bmin, coarse, state)
        return self._blocks

    def lookup(self, points) -> np.ndarray:
        """State of the voxels holding ``points`` (..., 3): 0 unknown, 1 free, 2 occupied."""
        bmin, coarse, state = self.blocks()
        ijk = np.floor(np.asarray(points) / np.float32(VOXEL_M)).astype(np.int64)
        b = (ijk >> 3) - bmin if BLOCK == 8 else np.floor_divide(ijk, BLOCK) - bmin
        dims = coarse.shape
        inside = ((b[..., 0] >= 0) & (b[..., 0] < dims[0]) & (b[..., 1] >= 0) & (b[..., 1] < dims[1])
                  & (b[..., 2] >= 0) & (b[..., 2] < dims[2]))
        flat_b = np.where(inside, (b[..., 0] * dims[1] + b[..., 1]) * dims[2] + b[..., 2], 0)
        bid = np.where(inside, coarse.reshape(-1)[flat_b], len(state) - 1)
        loc = ijk & (BLOCK - 1) if BLOCK == 8 else np.mod(ijk, BLOCK)
        cell = (loc[..., 0] * BLOCK + loc[..., 1]) * BLOCK + loc[..., 2]
        return state.reshape(-1)[bid * BLOCK ** 3 + cell]

    def occupied_slot(self, points) -> np.ndarray:
        """Index into occ_* of the occupied voxel holding each point (-1 if none)."""
        key = voxel_key(voxel_index(points))
        j = np.searchsorted(self.occ_key, key)
        j = np.minimum(j, max(len(self.occ_key) - 1, 0))
        ok = (len(self.occ_key) > 0) & (self.occ_key[j] == key) if len(self.occ_key) else np.zeros(key.shape, bool)
        return np.where(ok, j, -1)


def key_to_ijk(keys) -> np.ndarray:
    keys = np.asarray(keys, np.int64)
    m = (1 << 21) - 1
    return np.stack([(keys >> 42) & m, (keys >> 21) & m, keys & m], axis=-1) - _KEY_OFF


def fuse(est: Estimates, cam_pos, *, carve_stride: int = 1) -> VoxelMap:
    """Fuse estimates (reference-frame camera positions ``cam_pos[est.frame]``) into a VoxelMap."""
    cam_pos = np.asarray(cam_pos, np.float64)
    if len(est.frame) == 0:
        return VoxelMap(np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0, np.int32), np.zeros(0, np.int32),
                        np.zeros(0), np.zeros(0, np.int64), dict(estimates=0))
    key = voxel_key(voxel_index(est.point))
    uk, inv, n = np.unique(key, return_inverse=True, return_counts=True)
    w = 1.0 / np.maximum(est.sigma_m, 1e-3) ** 2
    pos = np.stack([np.bincount(inv, weights=w * est.point[:, i], minlength=len(uk)) for i in range(3)], 1)
    pos /= np.bincount(inv, weights=w, minlength=len(uk))[:, None]
    fr = np.unique(inv.astype(np.int64) * (1 << 31) + est.frame.astype(np.int64))
    n_frames = np.bincount((fr >> 31).astype(np.int64), minlength=len(uk))
    tr = np.unique(inv.astype(np.int64) * (1 << 40) + est.track.astype(np.int64))
    n_tracks = np.bincount((tr >> 40).astype(np.int64), minlength=len(uk))
    # support: estimates of two tracks, or from two reference frames (note: one two-view pair puts the same
    # point into both of its frames; MIN_OCC_FRAMES = 3 would demand a third view)
    support = (n_tracks >= MIN_OCC_TRACKS) | (n_frames >= MIN_OCC_FRAMES)
    smin = np.full(len(uk), np.inf)
    np.minimum.at(smin, inv, est.sigma_m)
    # carving: lines of sight up to point - max(2 sigma, 2 voxels), from supported estimates only (an
    # unsupported outlier with a too-long range would carve through real surfaces)
    sel = np.arange(0, len(est.frame), max(1, carve_stride))
    if CARVE_SUPPORTED_ONLY:
        sel = sel[support[inv[sel]]]
    c = cam_pos[est.frame[sel]]
    d = est.point[sel] - c
    r = np.linalg.norm(d, axis=1)
    d /= np.maximum(r, 1e-9)[:, None]
    s_end = r - np.maximum(2 * est.sigma_m[sel], 2 * VOXEL_M)
    steps = np.maximum(np.floor(s_end / (VOXEL_M / 2)).astype(np.int64), 0)
    total = int(steps.sum())
    free_keys = np.zeros(0, np.int64)
    if total:
        rep = np.repeat(np.arange(len(sel)), steps)
        k_in = np.arange(total) - np.repeat(np.cumsum(steps) - steps, steps)
        s = (k_in + 0.5) * (VOXEL_M / 2)
        pts = c[rep] + d[rep] * s[:, None]
        free_keys = voxel_key(voxel_index(pts))
    fk, fn = np.unique(free_keys, return_counts=True)
    free_at_occ = np.zeros(len(uk), np.int64)
    j = np.searchsorted(fk, uk)
    j = np.minimum(j, max(len(fk) - 1, 0))
    if len(fk):
        m = fk[j] == uk
        free_at_occ[m] = fn[j[m]]
    occupied = support & (free_at_occ <= FREE_OVER_OCC * n)
    free_only = ~np.isin(fk, uk[occupied], assume_unique=True)
    meta = dict(estimates=int(len(est.frame)), voxels_with_estimates=int(len(uk)), occupied=int(occupied.sum()),
                rejected_unsupported=int((~support).sum()),
                rejected_carved=int((support & (free_at_occ > FREE_OVER_OCC * n)).sum()),
                carving_estimates=int(len(sel)),
                free_voxels=int(free_only.sum()), carve_samples=total)
    return VoxelMap(uk[occupied], pos[occupied], n[occupied].astype(np.int32), n_frames[occupied].astype(np.int32),
                    smin[occupied], fk[free_only], meta)


# ----------------------------------------------------------------------------- projection

_S = ((np.arange(int(round(FREE_MAX_M / MARCH_STEP_M))) + 0.5) * MARCH_STEP_M).astype(np.float32)
_SUB_STEP = contract.PATCH_PX / SUB


def free_extent(vmap: VoxelMap, origin, dirs, s_max: float = FREE_MAX_M, chunk: int = 10) -> np.ndarray:
    """Distance to which each ray from ``origin`` stays in carved-free voxels (the drone's own voxel counts free)."""
    S = _S[_S <= s_max]
    dirs = np.asarray(dirs, np.float32)
    o = np.asarray(origin, np.float32)
    e = np.full(len(dirs), float(s_max))
    alive = np.arange(len(dirs))
    for a in range(0, len(S), chunk):
        Sc = S[a:a + chunk]
        st = vmap.lookup(o[None, None, :] + dirs[alive, None, :] * Sc[None, :, None])
        near = Sc <= VOXEL_M
        if near.any():
            st[:, near] = np.where(st[:, near] == 2, 2, 1)
        nf = st != 1
        stop = nf.any(1)
        if stop.any():
            e[alive[stop]] = Sc[nf[stop].argmax(1)] - MARCH_STEP_M / 2
        alive = alive[~stop]
        if len(alive) == 0:
            break
    return e


def splat_hits(ranges, uv, radius_m: float = SPLAT_RADIUS_M, max_px: int = SPLAT_MAX_SUBRAYS):
    """Points projected at ``uv`` (n, 2) with ``ranges`` (n,) -> per-sub-ray minimum range (54 * 96,), inf if none.

    Each point covers the sub-rays within its angular footprint (radius_m at its range, at least the
    nearest sub-ray; at most max_px sub-ray spacings).
    """
    out = np.full(SUB_SHAPE[0] * SUB_SHAPE[1], np.inf)
    if len(ranges) == 0:
        return out
    rad = np.minimum(contract.FOCAL_PX * radius_m / np.maximum(ranges, 1e-3), max_px * _SUB_STEP)
    col = uv[:, 0] / _SUB_STEP - 0.5
    row = uv[:, 1] / _SUB_STEP - 0.5
    c0, r0 = np.rint(col).astype(int), np.rint(row).astype(int)
    K = int(np.ceil(rad.max() / _SUB_STEP)) if len(rad) else 0
    for dy in range(-K, K + 1):
        for dx in range(-K, K + 1):
            c, r = c0 + dx, r0 + dy
            dist = np.hypot((c - col) * _SUB_STEP, (r - row) * _SUB_STEP)
            m = ((dist <= rad) | ((dx == 0) & (dy == 0))) & (c >= 0) & (c < SUB_SHAPE[1]) & (r >= 0) & (r < SUB_SHAPE[0])
            if m.any():
                np.minimum.at(out, r[m] * SUB_SHAPE[1] + c[m], ranges[m])
    return out


def _visible_points(points, origin, R_wc, max_range: float = MARCH_MAX_M):
    rel = np.asarray(points, np.float64) - origin[None]
    rng = np.linalg.norm(rel, axis=1)
    uv, front = contract.project_camera(rel @ R_wc)
    ok = front & contract.in_image(uv) & (rng <= max_range) & (rng > 0.05)
    return uv[ok], rng[ok], ok


def subray_constraints(vmap: VoxelMap, pos, quat_wb, *, own_uv=None, own_range=None):
    """(54, 96) sub-ray (value, kind) for one frame (see module doc)."""
    origin = np.asarray(pos, np.float64)
    q = np.asarray(quat_wb, np.float64)
    dirs = subray_dirs_world(q).reshape(-1, 3)
    R_wc = contract.camera_to_world(q)
    e = free_extent(vmap, origin, dirs)
    uv, rng, _ = _visible_points(vmap.occ_pos, origin, R_wc)
    s_hit = splat_hits(rng, uv)
    val = np.full(len(dirs), np.nan)
    kind = np.zeros(len(dirs), np.uint8)
    hit = np.isfinite(s_hit)
    with np.errstate(invalid="ignore"):
        exact = hit & (e >= s_hit - np.maximum(EXACT_TOL * s_hit, 2 * VOXEL_M))
    val[hit] = s_hit[hit]
    kind[hit] = LabelKind.UPPER
    kind[exact] = LabelKind.EXACT
    if own_uv is not None and len(own_uv):
        own_uv = np.asarray(own_uv, np.float64)
        own_range = np.asarray(own_range, np.float64)
        s_own = splat_hits(own_range, own_uv, radius_m=0.0)       # nearest sub-ray only
        j = np.flatnonzero(np.isfinite(s_own))
        r = s_own[j]
        # an own-frame observation is visible: EXACT, unless an occupied voxel is clearly nearer
        keep = ~(hit[j] & (s_hit[j] < r / (1 + EXACT_TOL)))
        val[j[keep]] = r[keep]
        kind[j[keep]] = LabelKind.EXACT
    free = ~hit & (e >= MIN_LOWER_M) & (kind == LabelKind.UNKNOWN)
    val[free] = e[free]
    kind[free] = LabelKind.LOWER
    return val.reshape(SUB_SHAPE), kind.reshape(SUB_SHAPE)


def _corridor_offsets(d):
    a = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    e1 = np.cross(d, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)
    offs = [np.zeros_like(d)]
    for r in (0.25, FAN_CORRIDOR_RADIUS_M):
        for k in range(8):
            ang = 2 * np.pi * (k + 0.5 * (r < FAN_CORRIDOR_RADIUS_M)) / 8
            offs.append(r * (np.cos(ang) * e1 + np.sin(ang) * e2))
    return np.stack(offs, axis=1)          # (m, 17, 3)


def fan_free_extent(vmap: VoxelMap, origin, dirs, s_max: float = FAN_MAX_M, chunk: int = 10):
    """Along-axis distance to which each corridor is observed free (FAN_FREE_FRACTION rule)."""
    origin = np.asarray(origin, np.float64)
    dirs = np.asarray(dirs, np.float64).reshape(-1, 3)
    base = (origin[None, None] + _corridor_offsets(dirs)).astype(np.float32)        # (m, 17, 3)
    d32 = dirs.astype(np.float32)
    S = _S[_S <= s_max]
    e = np.full(len(dirs), float(s_max))
    alive = np.arange(len(dirs))
    own = VOXEL_M + FAN_CORRIDOR_RADIUS_M
    for a in range(0, len(S), chunk):
        Sc = S[a:a + chunk]
        st = vmap.lookup(base[alive, :, None, :] + d32[alive, None, None, :] * Sc[None, None, :, None])  # (m, 17, k)
        near = Sc <= own
        if near.any():
            st[:, :, near] = np.where(st[:, :, near] == 2, 2, 1)
        free = st == 1
        ok = free[:, 0] & (free.mean(axis=1) >= FAN_FREE_FRACTION) & ~(st == 2).any(axis=1)
        stop = ~ok.all(1)
        if stop.any():
            e[alive[stop]] = Sc[(~ok[stop]).argmax(1)] - MARCH_STEP_M / 2
        alive = alive[~stop]
        if len(alive) == 0:
            break
    return e


def fan_labels(vmap: VoxelMap, pos, quat_wb, *, own_points=None):
    """(4, 9) fan (value, kind) from the fused map (plus own-frame points), clipped."""
    from .corridors import corridor_hits
    q = np.asarray(quat_wb, np.float64)
    dirs = contract.fan_directions_world(q).reshape(-1, 3)
    if not np.isfinite(dirs).all():
        return unknown_fan()
    origin = np.asarray(pos, np.float64)
    near = np.linalg.norm(vmap.occ_pos - origin[None], axis=1) <= FAN_MAX_M + FAN_CORRIDOR_RADIUS_M
    pts = vmap.occ_pos[near]
    if own_points is not None and len(own_points):
        pts = np.vstack([pts, np.asarray(own_points, np.float64).reshape(-1, 3)])
    hit = corridor_hits(pts, origin, dirs, FAN_CORRIDOR_RADIUS_M, FAN_MAX_M)
    free = fan_free_extent(vmap, origin, dirs, FAN_MAX_M)
    v, k = fan_constraints(hit, free, s_max=FAN_MAX_M)
    # free extents shorter than MIN_LOWER_M are the forced drone-own region of the march, not observations (and
    # wrong when the drone flies within 0.5 m of the ground or a wall): dropped, as for the grid and the tube
    short = (k == LabelKind.LOWER) & (v < MIN_LOWER_M)
    v, k = np.where(short, np.nan, v), np.where(short, LabelKind.UNKNOWN, k).astype(np.uint8)
    return clip_fan(v.reshape(FAN_SHAPE), k.reshape(FAN_SHAPE))


def hindsight_labels(voxel_map: VoxelMap, pos, quat_wb, *, own_uv=None, own_range=None, own_points=None,
                     return_subrays: bool = False):
    """-> (grid_value (18, 32), grid_kind, fan_value (4, 9), fan_kind) for one frame (LabelSource.HINDSIGHT)."""
    q = np.asarray(quat_wb, np.float64)
    if not (np.isfinite(pos).all() and np.isfinite(q).all()):
        gv, gk = unknown_grid()
        fv, fk = unknown_fan()
        sv, sk = unknown_subrays()
        return (gv, gk, fv, fk, sv, sk) if return_subrays else (gv, gk, fv, fk)
    sv, sk = subray_constraints(voxel_map, pos, q, own_uv=own_uv, own_range=own_range)
    # a cell with a triangulated surface on some sub-rays and a carving that stops short on others keeps the
    # surface (UPPER) within 8 m, the free extent (LOWER) beyond (labels.resolve_interval)
    gv, gk = grid_from_subrays(sv, sk, wide_upper_max_m=WIDE_UPPER_MAX_M)
    fv, fk = fan_labels(voxel_map, pos, q, own_points=own_points)
    return (gv, gk, fv, fk, sv, sk) if return_subrays else (gv, gk, fv, fk)


# ----------------------------------------------------------------------------- per-flight driver

def run_rows(store, run_id: int, rows=None) -> np.ndarray:
    """Time-ordered store rows of one run eligible for L2 (grade <= FAIR)."""
    from ..store import Grade
    ix = store.index
    if rows is None:
        rows = np.flatnonzero(ix['run_id'] == run_id)
    rows = np.asarray(rows, np.int64)
    rows = rows[ix['grade'][rows] <= int(Grade.FAIR)]
    return rows[np.argsort(ix['t_wall'][rows], kind='stable')]


class _LazyFrames:
    """Frames of store rows read one at a time (a long run would otherwise need GBs of RAM)."""

    def __init__(self, store, rows):
        self.store, self.rows = store, np.asarray(rows, np.int64)
        ix = store.index[self.rows]
        self.slots = ix['slot'] if hasattr(store, '_frames') else None
        self.mm = store._frames(int(ix['run_id'][0])) if hasattr(store, '_frames') and len(ix) else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        if self.mm is not None:
            return np.asarray(self.mm[int(self.slots[k])])
        return self.store.frames(self.rows[k:k + 1])[0]


def build_flight_map(store, run_id: int, guard, *, rows=None, mask_fn=None, log=print):
    """Track, triangulate and fuse one flight.

    Returns (rows, VoxelMap, Estimates, Tracks); rows are the time-ordered store rows used.
    """
    t0 = time.monotonic()
    rows = run_rows(store, run_id, rows)
    if guard is not None:
        guard.before_chunk()
    if len(rows) < 3:
        return rows, fuse(Estimates.empty(), np.zeros((0, 3))), Estimates.empty(), None
    ix = store.index[rows]
    frames = _LazyFrames(store, rows)
    t = ix['t_wall'].astype(np.float64)
    pos = ix['pos'].astype(np.float64)
    quat = ix['quat'].astype(np.float64)
    tracks = track_run(frames, t, pos, quat, mask_fn=mask_fn, guard=guard)
    t1 = time.monotonic()
    est = triangulate(tracks, t, pos, quat)
    t2 = time.monotonic()
    vmap = fuse(est, pos)
    vmap.meta.update(run_id=int(run_id), frames=int(len(rows)), observations=int(len(tracks.track)),
                     tracks=int(len(np.unique(tracks.track))), mask=tracks.mask_source,
                     map_seconds=dict(track=round(t1 - t0, 1), triangulate=round(t2 - t1, 1),
                                  fuse=round(time.monotonic() - t2, 1)))
    return rows, vmap, est, tracks


def clear_flown_path(vmap: VoxelMap, path_t, path_pos, stops=(), *, clear_m: float = PATH_CLEAR_M,
                     free_m: float = PATH_FREE_M, stop_margin_s: float = PATH_STOP_MARGIN_S) -> VoxelMap:
    """The flown volume is free: drop occupied voxels whose mean point lies within ``clear_m`` of the flown path
    and mark voxels whose centre lies within ``free_m`` of it free (path samples within ``stop_margin_s`` of a
    contact/impact/reset are skipped). Removes ghost-trail and outlier points on the racing line."""
    t = np.asarray(path_t, np.float64)
    p = np.asarray(path_pos, np.float64)
    if len(t) < 2:
        return vmap
    keep = np.ones(len(t), bool)
    for s in stops:
        keep &= np.abs(t - s) > stop_margin_s
    t, p = t[keep], p[keep]
    if len(p) < 2:
        return vmap
    # resample the path at <= VOXEL_M / 2 spacing (breaks at gaps: jumps > 1 m are not interpolated)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    pts = [p[:1]]
    for a, b, L in zip(p[:-1], p[1:], seg):
        if L > 1.0 or L == 0:
            pts.append(b[None])
            continue
        n = int(np.ceil(L / (VOXEL_M / 2)))
        pts.append(a[None] + (b - a)[None] * (np.arange(1, n + 1) / n)[:, None])
    P = np.concatenate(pts)
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    occ_keep = np.ones(len(vmap.occ_key), bool)
    if len(vmap.occ_key):
        d, _ = tree.query(vmap.occ_pos, k=1)
        occ_keep = d > clear_m
    # free voxels around the path: voxel centres within free_m of a path sample
    ijk = voxel_index(P)
    r = int(np.ceil(free_m / VOXEL_M)) + 1
    off = np.stack(np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing='ij'), -1).reshape(-1, 3)
    cand = np.unique((ijk[:, None, :] + off[None]).reshape(-1, 3), axis=0)
    centres = (cand + 0.5) * VOXEL_M
    d, _ = tree.query(centres, k=1)
    new_free = voxel_key(cand[d <= free_m])
    occ_key = vmap.occ_key[occ_keep]
    free_key = np.union1d(vmap.free_key, new_free)
    free_key = free_key[~np.isin(free_key, occ_key, assume_unique=True)]
    meta = dict(vmap.meta, path_cleared_occupied=int((~occ_keep).sum()), path_free_voxels=int(len(new_free)))
    return VoxelMap(occ_key, vmap.occ_pos[occ_keep], vmap.occ_n[occ_keep], vmap.occ_frames[occ_keep],
                    vmap.occ_sigma[occ_keep], free_key, meta)


def own_frame_points(est: Estimates, vmap: VoxelMap, k: int):
    """Own-frame observations of frame k whose voxel survived fusion: (uv, range, points)."""
    m = est.frame == k
    if not m.any():
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 3))
    keep = vmap.lookup(est.point[m]) == 2
    return est.uv[m][keep].astype(np.float64), est.range_m[m][keep], est.point[m][keep]
