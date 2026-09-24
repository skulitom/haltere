"""Per-video clock refinement and the K0a timing check (offline only).

``python -m haltere.obstacles.timing refine --store runs/obstacle-store-v1 --search-ms 60 --step-ms 5
--flight-lock PATH [--controls]``

A run video's frame times are ``t = csv.wall[0] + t_video + offset`` (store_build). The survey fitted
``offset`` per video with a yaw-flow correlation and an epipolar refinement at 160 x 90. This module
checks and refines it against the telemetry poses at 640 x 360, the resolution of the K0a criterion.

Residual (per run and candidate delta):

1. Windows: up to ``MAX_WINDOWS`` runs of ``WINDOW`` consecutive valid new frames (store_build drop
   reason '' in parts/rNNNNN.frames.npz), no frame gap above ``MAX_GAP_S``, mean speed >= ``MIN_SPEED``,
   spread evenly over the flight. The video is decoded once at 640 x 360 (ffmpeg area scaling of the
   1280 x 720 gameplay crop).
2. Bidirectional tracks: Shi-Tomasi corners in the first frame inside the Liftoff geometry mask
   (haltere.vision.geometry_mask: HUD, propellers, coloured task cues and white overlay-like pixels),
   pyramidal Lucas-Kanade forward through the window and back again; a track is kept when the
   backward track returns within ``FB_MAX_PX`` of its start and every point stays in the mask of its frame.
3. For each delta in [-search, +search] (step) the telemetry pose (100 Hz CSV on its row clock) is
   interpolated at ``t + delta``; each track is triangulated (least-squares ray midpoint) with the
   store camera model scaled to 640 px (f = 2 * focal_320 = 200 px, 30 deg uptilt, pixel centres at
   integer + 0.5) and reprojected into every frame of its window. Track residual = RMS reprojection
   error (px at 640 x 360). Run residual = weighted median over tracks; windows whose peak |omega_z|
   exceeds 2 rad/s (Flag.HIGH_YAW_RATE) weigh ``HIGH_YAW_WEIGHT``.
4. delta = argmin. Accepted when the minimum is inside the search range, at least ``MIN_TRACKS``
   tracks exist, the residual improves on delta = 0 and a window bootstrap puts the minimum within
   ``MAX_BOOT_STD_MS`` (std). ``store repose`` applies accepted deltas.

K0a: after refinement the run residual is <= 1.0 px at 640 px on >= 70 % of good/fair runs. If it
fails, geometry labels are built from the exact PNG and capture sources only and the report says so.

``--controls`` also scores sources whose image time is exact (geometry PNG, worker pose) or logged
(DatasetWriter capture sets, grab-start time + raw UDP pose): the same residual on correctly timed
frames is the method's floor. Older recorder capture sets are scored with the pose logged next to
each image, which checks the camera model (``--focal-scan``) rather than the clock. Controls are
reported under ``controls`` and never re-posed.

Output ``<store>/timing/refined.json``::

    {"schema": "haltere.obstacles.timing.v1", "search_ms": 60, "step_ms": 5, "store_index_sha256": str,
     "runs": {"<run_id>": {"source_id": str, "delta_s": float, "residual_px_before": float,
                            "residual_px_after": float, "n_tracks": int, "accepted": bool, ...}},
     "controls": {...}, "pose_lag": {...}, "k0a": {...}}

``pose_lag`` holds the older recorder capture sets (pose logged with the image, SOURCE_RAW): the same scan
over [-search, +300] ms estimates how far the logged pose trails the image; ``store repose`` replaces
their poses by the logged pose series interpolated at t_wall + delta (SOURCE_COMPENSATED) when accepted.

Per-run results are cached in ``timing/runs/rNNNNN.json`` so an interrupted job resumes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

TIMING_SCHEMA = 'haltere.obstacles.timing.v1'
SEARCH_MS = 60
STEP_MS = 5
WIDTH, HEIGHT = 640, 360
WINDOW = 5                    # consecutive new frames per track window
MAX_WINDOWS = 48
MAX_GAP_S = 0.15              # no missing new frame inside a window (13-18 Hz sources)
MIN_SPEED = 1.0               # m/s: translation for triangulation
MAX_CORNERS = 300
FB_MAX_PX = 0.5
MIN_TRACKS = 200
HIGH_YAW_WEIGHT = 0.25
HIGH_YAW = 2.0                # rad/s
BOOTSTRAP = 64
MAX_BOOT_STD_MS = 10.0
K0A_PX = 1.0
K0A_FRACTION = 0.70
FAR_M = 200.0
LAG_SEARCH_MAX_MS = 300       # older recorder capture sets: the logged pose may trail the image by > 60 ms
CONTROL_MIN_SPEED = 0.3       # the box-course flights are slow (median ~0.6 m/s)


# ----------------------------------------------------------------------------- geometry

def camera_640(focal_320: float | None = None):
    from . import contract
    from ..vision.camera import Camera
    f = 2.0 * (contract.FOCAL_320 if focal_320 is None else focal_320)
    return Camera(WIDTH, HEIGHT, f, contract.TILT_DEG)


def rays_world(uv_cv, R_wc, cam):
    """Unit world rays of OpenCV pixel coordinates (..., 2) (pixel centres at integers) for camera->world R."""
    u = uv_cv[..., 0] + 0.5
    v = uv_cv[..., 1] + 0.5
    d = np.stack([(u - cam.cx) / cam.f, (v - cam.cy) / cam.f, np.ones_like(u)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return np.einsum('...ij,...j->...i', R_wc, d)


def triangulate(C, D):
    """Least-squares midpoint of rays C + s D. C, D: (n, W, 3) -> X (n, 3), ok (n,)."""
    I = np.eye(3)
    P = I - D[..., :, None] * D[..., None, :]              # (n, W, 3, 3) projectors onto the ray normal plane
    A = P.sum(1) + 1e-9 * I
    b = np.einsum('nwij,nwj->ni', P, C)
    X = np.linalg.solve(A, b[..., None])[..., 0]
    s = np.einsum('nwi,nwi->nw', X[:, None, :] - C, D)      # along-ray distances
    ok = (s > 0.3).all(1) & (s < FAR_M).all(1)
    # (near-)parallel rays: a point far along the mean ray tests the rotations alone
    far = C[:, 0] + FAR_M * _unit(D.mean(1))
    X = np.where(ok[:, None], X, far)
    return X, ok


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def reprojection_rms(tracks, C, R_wc, cam):
    """Per-track RMS reprojection error (px) of tracks (n, W, 2) for poses C (n, W, 3), R_wc (n, W, 3, 3)."""
    D = rays_world(tracks, R_wc, cam)
    X, _ = triangulate(C, D)
    Xc = np.einsum('nwji,nwj->nwi', R_wc, X[:, None, :] - C)   # camera coordinates (R_wc^T (X - C))
    z = np.maximum(Xc[..., 2], 1e-6)
    u = cam.cx + cam.f * Xc[..., 0] / z - 0.5
    v = cam.cy + cam.f * Xc[..., 1] / z - 0.5
    e2 = (u - tracks[..., 0]) ** 2 + (v - tracks[..., 1]) ** 2
    e2 = np.where(Xc[..., 2] > 1e-3, e2, 1e6)
    return np.sqrt(e2.mean(1))


def weighted_median(x, w):
    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    if not len(x):
        return float('nan')
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


# ----------------------------------------------------------------------------- tracks

def usable_mask(rgb):
    from ..vision.geometry_mask import liftoff_geometry_mask
    return liftoff_geometry_mask(rgb) > 0


def track_window(greys, masks):
    """Bidirectional KLT tracks through one window -> (n, W, 2) OpenCV pixel coordinates."""
    import cv2
    W = len(greys)
    p0 = cv2.goodFeaturesToTrack(greys[0], MAX_CORNERS, 0.01, 7, mask=masks[0].astype(np.uint8) * 255, blockSize=7)
    if p0 is None or len(p0) < 8:
        return np.zeros((0, W, 2), np.float32)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pts = [p0.reshape(-1, 2).astype(np.float32)]
    ok = np.ones(len(p0), bool)
    for k in range(1, W):
        p, st, _ = cv2.calcOpticalFlowPyrLK(greys[k - 1], greys[k], pts[-1].reshape(-1, 1, 2), None, **lk)
        pts.append(p.reshape(-1, 2))
        ok &= st.reshape(-1) == 1
    back = pts[-1]
    for k in range(W - 1, 0, -1):
        back, st, _ = cv2.calcOpticalFlowPyrLK(greys[k], greys[k - 1], back.reshape(-1, 1, 2), None, **lk)
        back = back.reshape(-1, 2)
        ok &= st.reshape(-1) == 1
    ok &= np.linalg.norm(back - pts[0], axis=1) < FB_MAX_PX
    T = np.stack(pts, 1)
    for k in range(W):
        x = np.round(T[:, k, 0]).astype(int)
        y = np.round(T[:, k, 1]).astype(int)
        inside = (x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)
        ok &= inside
        ok[inside] &= masks[k][y[inside], x[inside]]
    return T[ok]


# ----------------------------------------------------------------------------- per-run scoring

def choose_windows(t, valid, speed, max_windows=MAX_WINDOWS, window=WINDOW, max_gap=MAX_GAP_S, min_speed=MIN_SPEED):
    """Start indices (into the frame list) of evenly spread windows of consecutive valid moving frames."""
    n = len(t)
    starts = []
    for i in range(n - window + 1):
        seg = slice(i, i + window)
        if not valid[seg].all():
            continue
        if np.max(np.diff(t[seg])) > max_gap:
            continue
        if np.mean(speed[seg]) < min_speed:
            continue
        starts.append(i)
    starts = np.array(starts, int)
    if len(starts) <= max_windows:
        return starts
    # evenly over time, without overlap where possible
    pick = np.unique(np.round(np.linspace(0, len(starts) - 1, max_windows)).astype(int))
    return starts[pick]


def score_curve(tracks, win_of, t_frames, weights, tel, deltas, cam, pose=None):
    """Residual (weighted median of track RMS) per delta. t_frames: (n_windows, W) epoch times."""
    from . import contract
    out = []
    per_track = []
    for d in deltas:
        if pose is not None:
            C, Q = pose(t_frames + d)
        else:
            s = tel.sample((t_frames + d).reshape(-1))
            C = s['pos'].reshape(t_frames.shape + (3,))
            Q = s['quat'].reshape(t_frames.shape + (4,))
        R = contract.camera_to_world(Q)
        r = reprojection_rms(tracks, C[win_of], R[win_of], cam)
        per_track.append(r)
        out.append(weighted_median(r, weights))
    return np.array(out), np.stack(per_track) if per_track else np.zeros((0, 0))


def bootstrap_delta(per_track, win_of, weights, deltas, n_windows, rng, n=BOOTSTRAP):
    """Std (ms) of the argmin delta over window bootstrap resamples."""
    best = []
    for _ in range(n):
        pick = rng.integers(0, n_windows, n_windows)
        cnt = np.bincount(pick, minlength=n_windows)
        w = weights * cnt[win_of]
        if w.sum() <= 0:
            continue
        curve = [weighted_median(per_track[k], w) for k in range(len(deltas))]
        best.append(deltas[int(np.argmin(curve))])
    return float(np.std(best) * 1000.0) if best else float('nan')


def decode_frames(path, wanted, guard=None, chunk=500):
    """{decoded index: RGB 640x360} for the wanted decoded frame indices of a run video."""
    from .store_build import GAMEPLAY_H, VideoReader, video_props
    fps, W, H = video_props(path)
    if H != GAMEPLAY_H:
        raise RuntimeError(f'{path}: unexpected height {H}')
    wanted = sorted(set(int(i) for i in wanted))
    out = {}
    rd = VideoReader(path, W, H, scale=(WIDTH, HEIGHT))
    try:
        k, j, since = 0, 0, 0
        while j < len(wanted):
            fr = rd.read()
            if fr is None:
                raise RuntimeError(f'{path}: video ended at frame {k} before frame {wanted[j]}')
            if k == wanted[j]:
                out[k] = fr.copy()
                j += 1
            k += 1
            since += 1
            if guard is not None and since >= chunk:
                guard.before_chunk()
                since = 0
    finally:
        rd.close()
    return out


def _tracks_for_windows(images_of, starts, window):
    """Tracks of every window; images_of(i) -> RGB 640x360 of new frame i."""
    import cv2
    tracks, win_of = [], []
    for w, i0 in enumerate(starts):
        rgbs = [images_of(i) for i in range(i0, i0 + window)]
        greys = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in rgbs]
        masks = [usable_mask(x) for x in rgbs]
        T = track_window(greys, masks)
        tracks.append(T)
        win_of.append(np.full(len(T), w, int))
    if not tracks:
        return np.zeros((0, window, 2)), np.zeros(0, int)
    return np.concatenate(tracks).astype(np.float64), np.concatenate(win_of)


def refine_video_run(store_root, rec, tel, search_ms, step_ms, guard=None, cam=None, focal_scan=None):
    """Timing refinement of one run video (see the module docstring)."""
    from .store_build import data_path, run_stem, store_data_root
    manifest = json.loads((Path(store_root) / 'manifest.json').read_text(encoding='utf-8'))
    root = store_data_root(manifest)
    cam = cam or camera_640()
    fr = np.load(Path(store_root) / 'parts' / f'{run_stem(rec["run_id"])}.frames.npz')
    idx, t_wall, valid = fr['frame_idx'], fr['t_wall'], fr['why'] == 0
    s = tel.sample(t_wall)
    speed = np.linalg.norm(s['vel'], axis=1)
    starts = choose_windows(t_wall, valid & s['valid'], speed)
    res = dict(source_id=rec['source_id'], grade=rec['alignment']['grade'], offset_s=rec['alignment']['offset_s'],
               n_windows=int(len(starts)))
    if len(starts) < 4:
        return dict(res, delta_s=0.0, accepted=False, reason='fewer than 4 usable windows', n_tracks=0,
                    residual_px_before=None, residual_px_after=None)
    wanted = np.unique(np.concatenate([idx[i:i + WINDOW] for i in starts]))
    imgs = decode_frames(data_path(root, rec['path']), wanted, guard)
    tracks, win_of = _tracks_for_windows(lambda i: imgs[int(idx[i])], starts, WINDOW)
    t_frames = np.stack([t_wall[i:i + WINDOW] for i in starts])
    omega_z = np.abs(tel.sample(t_frames.reshape(-1))['omega'][:, 2]).reshape(t_frames.shape).max(1)
    wwin = np.where(omega_z > HIGH_YAW, HIGH_YAW_WEIGHT, 1.0)
    weights = wwin[win_of]
    return dict(res, **_scan(tracks, win_of, t_frames, weights, tel, search_ms, step_ms, cam, len(starts),
                             focal_scan=focal_scan),
                high_yaw_windows=int((omega_z > HIGH_YAW).sum()))


def _scan(tracks, win_of, t_frames, weights, tel, search_ms, step_ms, cam, n_windows, pose=None, focal_scan=None):
    lo, hi = (-search_ms, search_ms) if np.isscalar(search_ms) else search_ms
    deltas = np.arange(lo, hi + 1e-9, step_ms) / 1000.0
    n_tracks = int(len(tracks))
    if n_tracks < 8:
        return dict(delta_s=0.0, accepted=False, reason='too few tracks', n_tracks=n_tracks,
                    residual_px_before=None, residual_px_after=None)
    curve, per_track = score_curve(tracks, win_of, t_frames, weights, tel, deltas, cam, pose)
    k0 = int(np.argmin(np.abs(deltas)))
    kb = int(np.argmin(curve))
    rng = np.random.default_rng(0)
    boot = bootstrap_delta(per_track, win_of, weights, deltas, n_windows, rng)
    interior = 0 < kb < len(deltas) - 1
    improves = curve[kb] < curve[k0]
    accepted = bool(interior and n_tracks >= MIN_TRACKS and improves and boot <= MAX_BOOT_STD_MS)
    reason = None if accepted else ('minimum at the search edge' if not interior else
                                    f'fewer than {MIN_TRACKS} tracks' if n_tracks < MIN_TRACKS else
                                    'no improvement over delta = 0' if not improves else
                                    f'bootstrap std {boot:.1f} ms > {MAX_BOOT_STD_MS:.0f} ms')
    out = dict(delta_s=round(float(deltas[kb]), 4) if accepted else 0.0, best_delta_s=round(float(deltas[kb]), 4),
               accepted=accepted, reason=reason, n_tracks=n_tracks,
               residual_px_before=round(float(curve[k0]), 4),
               residual_px_after=round(float(curve[kb] if accepted else curve[k0]), 4),
               residual_px_at_best=round(float(curve[kb]), 4), bootstrap_std_ms=round(boot, 2),
               track_rms_p90_before=round(float(np.percentile(per_track[k0], 90)), 3),
               curve_ms=[int(round(d * 1000)) for d in deltas], curve_px=[round(float(c), 4) for c in curve])
    if focal_scan:
        from ..vision.camera import Camera
        k = kb if accepted else k0
        fs = []
        for f in focal_scan:
            c = Camera(WIDTH, HEIGHT, float(f), cam.tilt_deg)
            r, _ = score_curve(tracks, win_of, t_frames, weights, tel, deltas[k:k + 1], c, pose)
            fs.append(round(float(r[0]), 4))
        out['focal_scan'] = dict(f_640=[float(f) for f in focal_scan], residual_px=fs,
                                 best_f_640=float(focal_scan[int(np.argmin(fs))]))
    return out


# ----------------------------------------------------------------------------- controls (exact image times)

def control_max_gap(t):
    """Largest frame gap allowed inside a control window: 1.6 x the source's median frame interval."""
    dt = np.diff(np.asarray(t, np.float64))
    dt = dt[dt > 1e-4]
    return max(MAX_GAP_S, 1.6 * float(np.median(dt))) if len(dt) else MAX_GAP_S


def refine_geometry_control(store_root, rec, search_ms, step_ms, guard=None, cam=None, focal_scan=None):
    """The same residual on a geometry-PNG run: worker poses at the exact grab time (expected delta ~ 0)."""
    import cv2
    from .store_build import data_path, load_run_csv, read_rgb, store_data_root
    manifest = json.loads((Path(store_root) / 'manifest.json').read_text(encoding='utf-8'))
    root = store_data_root(manifest)
    cam = cam or camera_640()
    tel = load_run_csv(data_path(root, rec['telemetry_csv']), clock='receipt')
    recs = [json.loads(x) for x in open(data_path(root, rec['archive']), encoding='utf-8') if x.strip()]
    png = data_path(root, rec['path'])
    items = [r for r in recs if r.get('image_file') and r.get('camera_position') is not None
             and (png / r['image_file']).exists()]
    t = np.array([r['capture_time'] for r in items]) + tel.epoch_offset
    o = np.argsort(t)
    items, t = [items[i] for i in o], t[o]
    s = tel.sample(t)
    speed = np.linalg.norm(s['vel'], axis=1)
    starts = choose_windows(t, s['valid'], speed, max_gap=control_max_gap(t), min_speed=CONTROL_MIN_SPEED)
    res = dict(source_id=rec['source_id'], kind='geometry_png', n_windows=int(len(starts)))
    if len(starts) < 4:
        return dict(res, reason='fewer than 4 usable windows')
    cache = {}

    def image(i):
        if i not in cache:
            cache[i] = cv2.resize(read_rgb(png / items[i]['image_file']), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        return cache[i]
    tracks, win_of = _tracks_for_windows(image, starts, WINDOW)
    t_frames = np.stack([t[i:i + WINDOW] for i in starts])
    omega_z = np.abs(tel.sample(t_frames.reshape(-1))['omega'][:, 2]).reshape(t_frames.shape).max(1)
    weights = np.where(omega_z > HIGH_YAW, HIGH_YAW_WEIGHT, 1.0)[win_of]
    return dict(res, **_scan(tracks, win_of, t_frames, weights, tel, search_ms, step_ms, cam, len(starts),
                             focal_scan=focal_scan))


def refine_capture_control(store_root, rec, search_ms, step_ms, guard=None, cam=None, focal_scan=None):
    """The same residual on a capture set: DatasetWriter sets use the grab-start time and the raw UDP pose;
    older recorder sets (no raw log) use the pose logged with each image (SOURCE_RAW, lag unknown), which
    still tests the camera model (``--focal-scan``) but not the clock."""
    from .store_build import capture_index_telemetry, data_path, load_capture_telemetry, read_rgb, store_data_root
    manifest = json.loads((Path(store_root) / 'manifest.json').read_text(encoding='utf-8'))
    root = store_data_root(manifest)
    cam = cam or camera_640()
    ds = data_path(root, rec['path'])
    tel = load_capture_telemetry(ds, rec.get('origin_sim'))
    idx_tel, idx = capture_index_telemetry(ds)
    t = idx['wall_time'].to_numpy(np.float64)
    pose = 'raw UDP log at the grab start'
    if tel is None:
        tel = idx_tel
        pose = 'pose logged with the image (SOURCE_RAW)'
        search_ms = (-search_ms, LAG_SEARCH_MAX_MS)
    s = tel.sample(t)
    speed = np.linalg.norm(np.nan_to_num(s['vel']), axis=1)
    starts = choose_windows(t, s['valid'], speed, max_gap=control_max_gap(t), min_speed=CONTROL_MIN_SPEED)
    res = dict(source_id=rec['source_id'], kind='capture_dataset', pose=pose, n_windows=int(len(starts)))
    if len(starts) < 4:
        return dict(res, reason='fewer than 4 usable windows')
    cache = {}

    def image(i):
        if i not in cache:
            cache[i] = read_rgb(ds / 'frames' / idx['file'][i])
        return cache[i]
    tracks, win_of = _tracks_for_windows(image, starts, WINDOW)
    t_frames = np.stack([t[i:i + WINDOW] for i in starts])
    omega_z = np.abs(tel.sample(t_frames.reshape(-1))['omega'][:, 2]).reshape(t_frames.shape).max(1)
    weights = np.where(omega_z > HIGH_YAW, HIGH_YAW_WEIGHT, 1.0)[win_of]
    return dict(res, **_scan(tracks, win_of, t_frames, weights, tel, search_ms, step_ms, cam, len(starts),
                             focal_scan=focal_scan))


# ----------------------------------------------------------------------------- CLI

def _summary(res: dict) -> dict:
    def med(key):
        v = [r[key] for r in res.values() if r.get(key) is not None]
        return round(float(np.median(v)), 4) if v else None
    best = med('best_delta_s')
    return dict(n=len(res), accepted=sum(bool(r.get('accepted')) for r in res.values()),
                residual_px_median_at_0=med('residual_px_before'), residual_px_median_at_best=med('residual_px_at_best'),
                best_delta_ms_median=None if best is None else round(1000 * best, 1))


def k0a_summary(runs: dict) -> dict:
    graded = [r for r in runs.values() if r.get('grade') in ('good', 'fair')]
    scored = [r for r in graded if r.get('residual_px_after') is not None]
    ok = [r for r in scored if r['residual_px_after'] <= K0A_PX]
    frac = len(ok) / len(graded) if graded else 0.0
    return dict(criterion=f'median reprojection residual <= {K0A_PX} px at 640 px on >= {K0A_FRACTION:.0%} of '
                          'good/fair runs', n_runs=len(graded), n_scored=len(scored), n_within=len(ok),
                fraction=round(frac, 4), passed=bool(frac >= K0A_FRACTION),
                residual_after_median=round(float(np.median([r['residual_px_after'] for r in scored])), 4)
                if scored else None,
                residual_before_median=round(float(np.median([r['residual_px_before'] for r in scored])), 4)
                if scored else None,
                accepted=sum(bool(r.get('accepted')) for r in graded),
                on_fail='geometry labels from exact PNG and capture sources only (~105k targets); video frames keep '
                        'image/pose data but get no pose-derived geometry labels')


def refine(args):
    from . import thermal
    from .store import FrameStore, index_sha256
    from .store import _json_dump
    from .store_build import data_path, load_run_csv, run_stem, store_data_root
    lock = thermal.require_flight_lock_path(args.flight_lock)
    thermal.limit_threads(2, cv2=True)
    guard = thermal.ChunkGuard(lock)
    store = FrameStore(args.store)
    root = Path(args.store)
    data_root = store_data_root(store.manifest)
    out_dir = root / 'timing' / 'runs'
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = camera_640(getattr(args, 'focal_320', None))
    focal_scan = [float(x) for x in args.focal_scan.split(',')] if getattr(args, 'focal_scan', None) else None
    index = np.asarray(store.index)
    sha = index_sha256(index)

    def run_sha(rid):             # per-run cache key: a rebuild of another run does not invalidate this one
        return index_sha256(index[index['run_id'] == rid])
    only = set(int(x) for x in args.runs) if getattr(args, 'runs', None) else None
    targets = [r for r in store.runs if r.get('n_frames') and r['source'] == 'run_video'
               and r['alignment']['grade'] in ('good', 'fair')]
    controls = []
    if getattr(args, 'controls', False):
        controls = [r for r in store.runs if r.get('n_frames') and r['source'] in ('geometry_png', 'capture_dataset')]
    results, ctrl = {}, {}
    tag = f'{args.search_ms}_{args.step_ms}_{cam.f:g}' + ('_fs' if focal_scan else '')
    for r in targets + controls:
        rid = r['run_id']
        if only is not None and rid not in only:
            continue
        cache = out_dir / f'{run_stem(rid)}.json'
        if cache.exists():
            got = json.loads(cache.read_text(encoding='utf-8'))
            if got.get('tag') == tag and got.get('run_index_sha256') == run_sha(rid):
                (results if r['source'] == 'run_video' else ctrl)[str(rid)] = got
                continue
        guard.before_chunk()
        t0 = time.monotonic()
        try:
            if r['source'] == 'run_video':
                tel = load_run_csv(data_path(data_root, r['telemetry_csv']), clock='wall')
                res = refine_video_run(root, r, tel, args.search_ms, args.step_ms, guard, cam, focal_scan)
            elif r['source'] == 'geometry_png':
                res = refine_geometry_control(root, r, args.search_ms, args.step_ms, guard, cam, focal_scan)
            else:
                res = refine_capture_control(root, r, args.search_ms, args.step_ms, guard, cam, focal_scan)
        except Exception as ex:   # noqa: BLE001 - reported per run
            res = dict(source_id=r['source_id'], accepted=False, reason=f'error: {ex!r}'[:300], delta_s=0.0,
                       residual_px_before=None, residual_px_after=None, n_tracks=0)
        res.update(tag=tag, run_index_sha256=run_sha(rid), env=r['env'], seconds=round(time.monotonic() - t0, 1))
        _json_dump(cache, res)
        (results if r['source'] == 'run_video' else ctrl)[str(rid)] = res
        print(f'timing run {rid:4d} {r["source_id"][:55]:55s} before {res.get("residual_px_before")} '
              f'after {res.get("residual_px_after")} delta {res.get("best_delta_s")} '
              f'{"ACCEPTED" if res.get("accepted") else res.get("reason")} tracks {res.get("n_tracks")} '
              f'{res["seconds"]:.0f}s', flush=True)
    k0a = k0a_summary(results)
    pose_lag = {k: v for k, v in ctrl.items() if str(v.get('pose', '')).startswith('pose logged')}
    exact = {k: v for k, v in ctrl.items() if k not in pose_lag}
    out = dict(schema=TIMING_SCHEMA, search_ms=args.search_ms, step_ms=args.step_ms, store_index_sha256=sha,
               camera=dict(width=WIDTH, height=HEIGHT, f=cam.f, tilt_deg=cam.tilt_deg),
               method=dict(window=WINDOW, max_windows=MAX_WINDOWS, min_speed_mps=MIN_SPEED, fb_max_px=FB_MAX_PX,
                           min_tracks=MIN_TRACKS, high_yaw_rad_s=HIGH_YAW, high_yaw_weight=HIGH_YAW_WEIGHT,
                           bootstrap=BOOTSTRAP, max_bootstrap_std_ms=MAX_BOOT_STD_MS,
                           mask='haltere.vision.geometry_mask.liftoff_geometry_mask',
                           residual='weighted median over tracks of per-track RMS reprojection error, px at 640x360'),
               runs=results, controls=exact, pose_lag=pose_lag, k0a=k0a,
               controls_summary=_summary(exact), pose_lag_summary=_summary(pose_lag),
               guard=guard.summary())
    if only is None:
        _json_dump(root / 'timing' / 'refined.json', out)
    print(json.dumps(dict(k0a=k0a, controls_summary=out['controls_summary'], pose_lag_summary=out['pose_lag_summary']),
                     indent=1), flush=True)
    return out


def main(argv=None):
    import argparse
    from .store import DEFAULT_STORE
    p = argparse.ArgumentParser(prog='python -m haltere.obstacles.timing')
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('refine')
    r.add_argument('--store', type=Path, default=DEFAULT_STORE)
    r.add_argument('--search-ms', type=int, default=SEARCH_MS)
    r.add_argument('--step-ms', type=int, default=STEP_MS)
    r.add_argument('--flight-lock', default=None)
    r.add_argument('--controls', action='store_true', help='also score exact-time geometry PNG and capture sets')
    r.add_argument('--focal-320', type=float, default=None, help='camera focal override (diagnostic only)')
    r.add_argument('--focal-scan', default=None, help='comma list of f at 640 px: residual per focal (diagnostic)')
    r.add_argument('--runs', nargs='*', default=None, help='restrict to these run ids (no refined.json written)')
    args = p.parse_args(argv)
    return refine(args)


if __name__ == '__main__':
    main()
