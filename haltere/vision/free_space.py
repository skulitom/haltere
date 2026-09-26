"""FreeSpace Corridor Planner v1, perception half: metric free corridors toward the checkpoint ring (runtime-safe).

OFF BY DEFAULT (``enabled: false`` in configs/obstacles/free_space.json). The flight stack runs it only inside the
obstacle stack's depth process (haltere.liftoff.gap_stack) with ``--obstacle-planner shadow|on``. Imports numpy
and the standard library at module level; OpenCV only when tracks are computed. No course geometry, routes,
per-course parameters, offline labels or bench code (tests/test_obstacle_label_isolation.py).

Inputs per processed frame, all causal: the 448 x 252 RGB frame, its DA-V2-Small block disparity (36 x 64, the
depth process's existing path), per-block validity of the overlay masks (hud, ring, ghost, propeller), the
world-from-body attitude and world velocity at capture (``no_pose`` without them), the checkpoint-ring cue of the
same frame (``stale`` when it did not arrive in time, ``no_ring`` when it is not in view or clamped at an edge)
and the declared motor (0 fast_pd, 1 brain08: the response model of configs/obstacles/response_models.json).

Steps (docs/free_space_planner.md, spec sections 4-5):

P1  Valid blocks: no pixel covered by a mask layer. Unknown blocks are never free space.
P2  Tracks (`ScaleTracker`): Shi-Tomasi corners on the usable pixels of the previous processed frame (valid blocks
    and `looming2.hud_mask`), pyramidal LK into the current frame, forward-backward error <= 0.5 px. The pair's
    attitude delta is removed with the homography K M R_cur^T R_prev M^T K^-1. The translation t = M R_cur^T v
    (v: the mean world velocity of the pair) gives the focus of expansion K t / t_z and each track's inverse depth
    invZ = (dq . r) / (t_z dt |q - FOE|), where q is the derotated previous position (the geometrically exact
    choice for the current frame's depth), dq = q_cur - q and r the unit radial direction from the FOE.
P3  Scale: invZ = a d + b by weighted Huber IRLS (weights (t_z dt |q - FOE|)^2, normalised per frame as in the
    design study), normal equations pooled over frames with forgetting; valid when a > 0, the pooled anchor count
    >= 60 and a (d_p90 - d_p10) >= 0.05 1/m over the frame's valid blocks; else the last valid fit is held 0.3 s.
P4  Cloud: every valid block at Z = 1 / max(a d + b, 1/30) in the path frame (x = horizontal course, y left, z
    up, origin at the camera); Z >= 30 m, x < 0.3 m and x > D_h + 1.5 m dropped; D_h = max(2 m, |v| T_h),
    T_h = delay + sqrt(2 * 1.5 / a_lat) + 0.3 s.
P5  Candidates: ring-relative offsets (daz -30..30 step 2) x (del 0..12 step 3), no descending candidates;
    a static (straight ray) and a lag-aware path (straight along the 3-D velocity for |v| delay, then turning
    toward the candidate at a_lat / |v|^2 and 5 / |v|^2 per metre) each; blocked by >= 2 cloud points within
    |dy| < 0.75 m and |dz| < 0.35 m of the path (0.3 m < x < D_h); eligibility: the aim ray inside the image
    with a 2 deg margin and >= 60 % valid blocks in the corridor footprint at min(D_h, 4 m) (ineligible counts
    as blocked; so an up-escape needs observed free space). Aperture exception for the ring corridor.
P6  Decision: clear / aperture (ring lag-aware path free), else per class (L, R, V) the cheapest static-free
    eligible candidate pushed outward until its lag-aware path is free (ok) or saturated at the bound (urgent);
    the cheapest class is proposed (shift), none: blocked. v_cap from the stopping distance. Eligibility applies
    to the escape candidates and the blocked v_cap; the ring corridor's own clear test is the corridor test alone
    (the ring overlay masks the ring's own footprint, so its footprint is often mostly unknown by construction).
P7  Vertical profiles in the |y| < 0.75 m strip: floor height, ceiling height, rising ground (rise, rise_x) and
    the share of the ring corridor's blocking points below the path.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .camera import body_to_cam_matrix, quat_wxyz_to_mat

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'obstacles' / 'free_space.json'
CONFIG_META_KEYS = ('frozen', 'frozen_at', 'sha256')
SCHEMA = 'haltere.obstacles.free_space.v1'
VERSION = 1       # the declaration version this code implements; the runtime refuses others

PLAN_KINDS = ('off', 'no_ring', 'no_pose', 'slow', 'no_scale', 'stale', 'clear', 'aperture', 'shift', 'blocked')
VALID_KINDS = ('clear', 'aperture', 'shift', 'blocked')
PLAN_FIELDS = ('time', 'seq', 'valid', 'kind', 'motor', 'speed', 'course_az', 'ring_az', 'ring_el',
               'cls', 'az', 'el', 'feasible',
               'l_az', 'l_el', 'l_ok', 'r_az', 'r_el', 'r_ok', 'v_el', 'v_ok',
               'd_free_ring', 'd_h', 'v_cap', 'n_block_ring', 'below_frac',
               'h_floor', 'h_ceil', 'rise', 'rise_x',
               'scale_a', 'scale_b', 'scale_n', 'scale_spread', 'scale_held', 'n_tracks',
               'age', 'frame_ms', 'overlay_ms', 'depth_ms', 'lk_ms', 'plan_ms')
MOTORS = ('fast_pd', 'brain08')          # the ``motor`` field: index into this tuple
CLS_LEFT, CLS_RIGHT, CLS_VERTICAL, CLS_NONE = 1, -1, 2, 0
MIN_FRAME_ANCHORS = 8                    # a frame's own IRLS fit needs this many anchors (design study lkscale.py)
PSI_CLIP = 1.4                           # rad: path headings are clipped to +-80 deg (x-parameterised paths)


def kind_index(kind: str) -> float:
    return float(PLAN_KINDS.index(kind))


def kind_name(index) -> str:
    if index is None or not np.isfinite(index) or not 0 <= int(index) < len(PLAN_KINDS):
        return ''
    return PLAN_KINDS[int(index)]


# ----------------------------------------------------------------------------- configuration

def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')


def config_sha256(obj: dict) -> str:
    """sha256 of a config object without its meta keys (as gap_stack.config_sha256)."""
    body = {k: v for k, v in obj.items() if k not in CONFIG_META_KEYS}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def _section(cls, d: dict, name: str):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    missing = names - set(d)
    if unknown or missing:
        raise ValueError(f'free_space {name}: unknown {sorted(unknown)}, missing {sorted(missing)}')
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


@dataclass(frozen=True)
class CameraSection:
    width: int
    height: int
    focal_px: float
    cx: float
    cy: float
    uptilt_deg: float
    blocks: tuple
    block_px: int


@dataclass(frozen=True)
class TracksSection:
    max_corners: int
    quality: float
    min_distance_px: float
    lk_window_px: int
    lk_levels: int
    fb_max_px: float
    perp_max_px: float
    min_foe_px: float
    min_tz_mps: float
    inv_z_range: tuple
    pair_dt_s: tuple


@dataclass(frozen=True)
class ScaleSection:
    huber_k: float
    irls_iters: int
    sigma_floor: float
    forget: float
    reset_gap_s: float
    reset_speed_mps: float
    min_anchors: float
    min_spread: float
    spread_pct: tuple
    hold_s: float
    max_range_m: float


@dataclass(frozen=True)
class PlanSection:
    min_speed_mps: float
    az_offsets_deg: tuple
    el_offsets_deg: tuple
    cost_el_weight: float
    max_az_deg: float
    max_el_deg: float
    path_step_m: float
    horizon_lateral_m: float
    horizon_extra_s: float
    min_horizon_m: float
    a_vert_mps2: float
    r_lat_m: float
    r_vert_m: float
    x_min_m: float
    min_points: int
    footprint_x_m: float
    footprint_valid: float
    image_margin_deg: float
    aperture_ratio: float
    aperture_gap_m: float
    aperture_edge_m: float
    brake_mps2: float
    stop_margin_m: float
    cloud_beyond_horizon_m: float


@dataclass(frozen=True)
class ProfileSection:
    bin_m: float
    near_x_m: tuple
    lateral_m: float
    min_bins: int
    ceil_min_m: float
    rise_level_m: float
    rise_bins: int
    rise_max_step_m: float


@dataclass(frozen=True)
class ResponseModel:
    name: str
    delay_s: float
    a_lat_mps2: float


@dataclass(frozen=True)
class FreeSpaceConfig:
    """The perception sections of configs/obstacles/free_space.json (the pilot, vertical and runtime sections are
    kept as plain dicts for the pilot side and the depth process)."""
    enabled: bool
    camera: CameraSection
    tracks: TracksSection
    scale: ScaleSection
    plan: PlanSection
    profile: ProfileSection
    masks: dict
    depth: dict
    pilot: dict
    vertical: dict
    runtime: dict
    response_models: str
    response_model_for_contract: dict
    version: int = VERSION

    @classmethod
    def from_dict(cls, obj: dict) -> 'FreeSpaceConfig':
        if obj.get('schema') != SCHEMA:
            raise ValueError(f'free_space schema {obj.get("schema")!r}, expected {SCHEMA!r}')
        return cls(enabled=bool(obj['enabled']), camera=_section(CameraSection, obj['camera'], 'camera'),
                   tracks=_section(TracksSection, obj['tracks'], 'tracks'),
                   scale=_section(ScaleSection, obj['scale'], 'scale'),
                   plan=_section(PlanSection, obj['plan'], 'plan'),
                   profile=_section(ProfileSection, obj['profile'], 'profile'),
                   masks=dict(obj['masks']), depth=dict(obj['depth']), pilot=dict(obj['pilot']),
                   vertical=dict(obj['vertical']), runtime=dict(obj['runtime']),
                   response_models=str(obj['response_models']),
                   response_model_for_contract=dict(obj['response_model_for_contract']),
                   version=int(obj['version']))

    @property
    def mask_layers(self) -> tuple:
        return tuple(self.masks['layers'])


def load_config(path: str | Path = CONFIG_PATH, *, require_frozen: bool = False,
                version: int | None = VERSION) -> tuple[FreeSpaceConfig, dict, str]:
    """(config, raw object, content sha256). Refuses a frozen file that changed, a draft when ``require_frozen``
    and a declaration version other than ``version`` (this code's VERSION)."""
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    sha = config_sha256(obj)
    if obj.get('frozen') and obj.get('sha256') != sha:
        raise ValueError(f'{path} changed after the freeze (sha256 mismatch)')
    if require_frozen and obj.get('frozen') is not True:
        raise ValueError(f'{path} is not frozen')
    if version is not None and obj.get('version') != version:
        raise ValueError(f'{path} declares free_space version {obj.get("version")}; this code implements {version}')
    return FreeSpaceConfig.from_dict(obj), obj, sha


def load_response_models(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    obj = json.loads(path.read_text(encoding='utf-8'))
    return {name: ResponseModel(name, float(m['delay_s']), float(m['a_lat_mps2'])) for name, m in obj['models'].items()}


def motor_name(motor) -> str:
    if isinstance(motor, str):
        if motor not in MOTORS:
            raise ValueError(f'unknown motor {motor!r}; expected one of {MOTORS}')
        return motor
    return MOTORS[int(motor)]


def horizon_s(model: ResponseModel, plan: PlanSection) -> float:
    """T_h = delay + sqrt(2 * horizon_lateral / a_lat) + horizon_extra (brain-08 1.53 s, fast PD 1.21 s)."""
    return model.delay_s + math.sqrt(2 * plan.horizon_lateral_m / model.a_lat_mps2) + plan.horizon_extra_s


def stop_speed(distance: float, delay: float, brake: float = 8.0, margin: float = 0.5) -> float:
    """v with v delay + v^2 / (2 brake) + margin = distance: 8 (sqrt(delay^2 + (D - 0.5) / 4) - delay) for the
    frozen brake 8 m/s^2; 0 when D <= margin."""
    room = float(distance) - margin
    if not np.isfinite(room) or room <= 0:
        return 0.0
    return float(brake * (math.sqrt(delay ** 2 + 2 * room / brake) - delay))


# ----------------------------------------------------------------------------- camera geometry

class Geometry:
    """Pinhole camera of the 448 x 252 model frame with the FPV uptilt, and its block grid."""

    def __init__(self, cam: CameraSection):
        self.W, self.H = int(cam.width), int(cam.height)
        self.f, self.cx, self.cy = float(cam.focal_px), float(cam.cx), float(cam.cy)
        self.grid = (int(cam.blocks[0]), int(cam.blocks[1]))
        self.bpx = int(cam.block_px)
        if self.grid[0] * self.bpx != self.H or self.grid[1] * self.bpx != self.W:
            raise ValueError('the block grid must tile the frame')
        self.K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.]])
        self.Kinv = np.linalg.inv(self.K)
        self.M = body_to_cam_matrix(float(cam.uptilt_deg))           # body (FLU) -> camera (right, down, fwd)
        rr, cc = np.mgrid[0:self.grid[0], 0:self.grid[1]]
        px = (cc.ravel() + .5) * self.bpx
        py = (rr.ravel() + .5) * self.bpx
        self.rays_cam = np.stack([(px - self.cx) / self.f, (py - self.cy) / self.f, np.ones(len(px))], 1)  # z = 1
        self.rays_body = self.rays_cam @ self.M                        # (n, 3) body, optical z = 1 scaling
        self.half_h = (math.atan(self.cx / self.f), math.atan((self.W - self.cx) / self.f))
        self.half_v = (math.atan(self.cy / self.f), math.atan((self.H - self.cy) / self.f))

    def pixel_ray_world(self, u_px: float, v_px: float, R: np.ndarray) -> np.ndarray:
        c = np.array([(u_px - self.cx) / self.f, (v_px - self.cy) / self.f, 1.])
        w = R @ (self.M.T @ c)
        return w / np.linalg.norm(w)

    def in_image(self, dirs_world: np.ndarray, R: np.ndarray, margin_deg: float) -> np.ndarray:
        """(k, 3) world directions -> bool: inside the image with an angular margin on both axes."""
        c = dirs_world @ R @ self.M.T
        z = c[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            ah = np.arctan2(c[:, 0], z)
            av = np.arctan2(c[:, 1], z)
        m = math.radians(margin_deg)
        return ((z > 1e-6) & (ah > -self.half_h[0] + m) & (ah < self.half_h[1] - m)
                & (av > -self.half_v[0] + m) & (av < self.half_v[1] - m))


def bilinear(grid: np.ndarray, xy: np.ndarray, bpx: int = 7) -> np.ndarray:
    """Block values at block centres sampled bilinearly at pixel coordinates (the design study's lkscale.bilinear)."""
    h, w = grid.shape
    gx = xy[:, 0] / bpx - .5
    gy = xy[:, 1] / bpx - .5
    x0 = np.clip(np.floor(gx).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(gy).astype(int), 0, h - 2)
    fx = np.clip(gx - x0, 0, 1)
    fy = np.clip(gy - y0, 0, 1)
    g = grid.astype(np.float64)
    return ((1 - fx) * (1 - fy) * g[y0, x0] + fx * (1 - fy) * g[y0, x0 + 1] + (1 - fx) * fy * g[y0 + 1, x0]
            + fx * fy * g[y0 + 1, x0 + 1])


def hud_usable(rgb: np.ndarray) -> np.ndarray:
    """`looming2.hud_mask` (usable pixels: no white/grey HUD graphics, no fixed HUD panels), computed with OpenCV
    channel min/max on uint8 (the same result, about 5x faster)."""
    import cv2
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    c0, c1, c2 = cv2.split(rgb)
    lo = cv2.min(cv2.min(c0, c1), c2)
    hi = cv2.max(cv2.max(c0, c1), c2)
    white = ((lo > 185) & (cv2.subtract(hi, lo) < 60)).astype(np.uint8)
    usable = ~cv2.dilate(white, np.ones((3, 3), np.uint8)).astype(bool)
    usable[:int(.2 * h)] = False
    usable[int(.3 * h):int(.8 * h), int(.82 * w):] = False
    usable[int(.85 * h):, int(.41 * w):int(.59 * w)] = False
    usable[int(.87 * h):, :int(.1 * w)] = False
    return usable


def usable_pixels(rgb: np.ndarray, valid: np.ndarray, bpx: int = 7) -> np.ndarray:
    """Pixels of valid blocks outside the fixed HUD panels and white HUD graphics (`hud_usable`)."""
    u = hud_usable(rgb)
    v = np.asarray(valid, bool)
    h, w = v.shape
    return u & np.broadcast_to(v[:, None, :, None], (h, bpx, w, bpx)).reshape(h * bpx, w * bpx)


def derotation_homography(geo: Geometry, R_prev: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Pixel map of a pure attitude change from the previous to the current frame: K M R_cur^T R_prev M^T K^-1."""
    return geo.K @ geo.M @ R_cur.T @ R_prev @ geo.M.T @ geo.Kinv


def parallax_anchors(P: np.ndarray, Q: np.ndarray, R_prev: np.ndarray, R_cur: np.ndarray, v_world: np.ndarray,
                     dt: float, geo: Geometry, tr: TracksSection) -> dict | None:
    """Tracks (previous pixel P -> current pixel Q) -> derotated translational parallax anchors (P2)."""
    t = geo.M @ (R_cur.T @ np.asarray(v_world, np.float64))
    tz = float(t[2])
    if tz < tr.min_tz_mps or len(P) == 0:
        return None
    H = derotation_homography(geo, R_prev, R_cur)
    ph = np.c_[P, np.ones(len(P))] @ H.T
    ph = ph[:, :2] / ph[:, 2:]
    foe = np.array([geo.cx + geo.f * t[0] / tz, geo.cy + geo.f * t[1] / tz])
    q = ph - foe
    qn = np.linalg.norm(q, axis=1)
    rhat = q / np.maximum(qn, 1e-9)[:, None]
    dq = Q - ph
    rad = (dq * rhat).sum(1)
    perp = dq[:, 0] * rhat[:, 1] - dq[:, 1] * rhat[:, 0]
    base = tz * dt * np.maximum(qn, 1e-9)
    inv_z = rad / base
    lo, hi = tr.inv_z_range
    keep = (qn >= tr.min_foe_px) & (np.abs(perp) <= tr.perp_max_px) & (inv_z >= lo) & (inv_z <= hi)
    return dict(Q=Q[keep], inv_z=inv_z[keep], w=base[keep] ** 2, foe=foe, tz=tz, n_raw=int(len(P)),
                perp_abs_p50=float(np.median(np.abs(perp))) if len(perp) else float('nan'))


def huber_fit(x: np.ndarray, y: np.ndarray, w0: np.ndarray, iters: int, k: float, sigma_floor: float):
    """y = a x + b by weighted Huber IRLS (the design study's lkscale.huber_line with a sigma floor): returns
    ((a, b), normal matrix, rhs) with the last iteration's weights, or None."""
    A = np.stack([x, np.ones(len(x))], 1)
    ww = w0.copy()
    for _ in range(int(iters)):
        Aw = A * ww[:, None]
        N = Aw.T @ A
        try:
            sol = np.linalg.solve(N, Aw.T @ y)
        except np.linalg.LinAlgError:
            return None
        r = y - A @ sol
        s = max(1.4826 * float(np.median(np.abs(r))), sigma_floor)
        ww = w0 * np.where(np.abs(r) <= k * s, 1., k * s / np.maximum(np.abs(r), 1e-12))
    Aw = A * ww[:, None]
    N = Aw.T @ A
    rhs = Aw.T @ y
    try:
        sol = np.linalg.solve(N, rhs)
    except np.linalg.LinAlgError:
        return None
    return (float(sol[0]), float(sol[1])), N, rhs


# ----------------------------------------------------------------------------- P2-P3 scale

class ScaleTracker:
    """Sparse LK parallax anchors and the pooled metric scale of the relative disparity (P2, P3)."""

    def __init__(self, config: FreeSpaceConfig, geometry: Geometry | None = None):
        self.cfg = config
        self.geo = geometry or Geometry(config.camera)
        self.reset()

    def reset(self):
        self.prev = None
        self.reset_pool()
        self.last_valid = None           # (capture time, a, b)

    def reset_pool(self):
        self.N = np.zeros((2, 2))
        self.rhs = np.zeros(2)
        self.n_eff = 0.0

    def _tracks(self, prev, gray, usable):
        import cv2
        tr = self.cfg.tracks
        pts = cv2.goodFeaturesToTrack(prev['gray'], int(tr.max_corners), float(tr.quality), float(tr.min_distance_px),
                                      mask=prev['usable'].astype(np.uint8))
        if pts is None or len(pts) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        lk = dict(winSize=(int(tr.lk_window_px), int(tr.lk_window_px)), maxLevel=int(tr.lk_levels),
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
        nb, st, _ = cv2.calcOpticalFlowPyrLK(prev['gray'], gray, pts, None, **lk)
        ok = st.ravel() == 1
        P, Q = pts.reshape(-1, 2)[ok], nb.reshape(-1, 2)[ok]
        if not len(P):
            return P, Q
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev['gray'], Q.reshape(-1, 1, 2).astype(np.float32), None, **lk)
        fb = np.linalg.norm(pb.reshape(-1, 2) - P, axis=1)
        good = (st2.ravel() == 1) & (fb <= tr.fb_max_px)
        qi = np.rint(Q).astype(int)
        inside = (qi[:, 0] >= 0) & (qi[:, 0] < self.geo.W) & (qi[:, 1] >= 0) & (qi[:, 1] < self.geo.H)
        good &= inside
        good[good] &= usable[qi[good, 1], qi[good, 0]]
        return P[good].astype(np.float64), Q[good].astype(np.float64)

    def update(self, t: float, rgb: np.ndarray, valid: np.ndarray, R: np.ndarray, v: np.ndarray,
               disparity: np.ndarray, *, usable: np.ndarray | None = None) -> dict:
        """One processed frame with a pose; returns the scale state for this frame (see ``state``)."""
        import cv2
        sc, tr = self.cfg.scale, self.cfg.tracks
        t0 = time.perf_counter()
        v = np.asarray(v, np.float64)
        speed = float(np.linalg.norm(v))
        out = dict(n_tracks=0, anchors=None)
        if self.prev is not None and t - self.prev['t'] > sc.reset_gap_s:
            self.reset_pool()
            self.prev = None
        if speed < sc.reset_speed_mps:
            self.reset_pool()
        self.N *= sc.forget
        self.rhs *= sc.forget
        self.n_eff *= sc.forget
        gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        if usable is None:
            usable = usable_pixels(rgb, valid, self.geo.bpx)
        prev = self.prev
        if prev is not None and speed >= tr.min_tz_mps:
            dt = float(t - prev['t'])
            if tr.pair_dt_s[0] <= dt <= tr.pair_dt_s[1]:
                P, Q = self._tracks(prev, gray, usable)
                anchors = parallax_anchors(P, Q, prev['R'], R, .5 * (prev['v'] + v), dt, self.geo, tr)
                if anchors is not None:
                    out['n_tracks'] = int(len(anchors['inv_z']))
                    out['anchors'] = anchors
                    if len(anchors['inv_z']) >= MIN_FRAME_ANCHORS:
                        d = bilinear(np.asarray(disparity, np.float64), anchors['Q'], self.geo.bpx)
                        w = anchors['w'] / max(float(anchors['w'].mean()), 1e-12)
                        fit = huber_fit(d, anchors['inv_z'], w, sc.irls_iters, sc.huber_k, sc.sigma_floor)
                        anchors['d'] = d
                        if fit is not None:
                            _, Nf, rf = fit
                            self.N += Nf
                            self.rhs += rf
                            self.n_eff += len(d)
                            out['frame_fit'] = fit[0]
        self.prev = dict(t=float(t), gray=gray, usable=usable, R=np.asarray(R, np.float64), v=v)
        out.update(self.state(t, disparity, valid))
        out['lk_ms'] = 1000 * (time.perf_counter() - t0)
        return out

    def pooled(self):
        if self.n_eff <= 0 or abs(np.linalg.det(self.N)) < 1e-12:
            return None
        try:
            a, b = np.linalg.solve(self.N, self.rhs)
        except np.linalg.LinAlgError:
            return None
        return float(a), float(b)

    def state(self, t, disparity, valid) -> dict:
        sc = self.cfg.scale
        fit = self.pooled()
        spread = float('nan')
        ok = False
        a = b = float('nan')
        if fit is not None:
            a, b = fit
            dv = np.asarray(disparity, np.float64)[np.asarray(valid, bool)]
            dv = dv[np.isfinite(dv)]
            if len(dv):
                lo, hi = np.percentile(dv, sc.spread_pct)
                spread = float(a * (hi - lo))
            ok = a > 0 and self.n_eff >= sc.min_anchors and np.isfinite(spread) and spread >= sc.min_spread
        if ok:
            self.last_valid = (float(t), a, b)
            return dict(scale_ok=True, a=a, b=b, held=False, n_eff=float(self.n_eff), spread=spread)
        if self.last_valid is not None and 0 <= t - self.last_valid[0] <= sc.hold_s:
            return dict(scale_ok=True, a=self.last_valid[1], b=self.last_valid[2], held=True, n_eff=float(self.n_eff),
                        spread=spread, fit_a=a, fit_b=b)
        return dict(scale_ok=False, a=a, b=b, held=False, n_eff=float(self.n_eff), spread=spread)


# ----------------------------------------------------------------------------- P4-P7 planner

def _wrap_rad(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


def _grid(spec) -> np.ndarray:
    lo, hi, step = (float(v) for v in spec)
    return np.round(np.arange(lo, hi + step / 2, step), 6)


class FreeSpacePlanner:
    """Per-frame free-space corridor planner (P1-P8). ``process`` returns the PLAN_FIELDS values as a dict; the
    last frame's intermediate arrays are kept in ``self.last`` (diagnostics and offline evaluation)."""

    def __init__(self, config: FreeSpaceConfig, motor, *, response_models: dict | None = None,
                 enabled: bool | None = None):
        self.cfg = config
        self.enabled = config.enabled if enabled is None else bool(enabled)
        self.motor = motor_name(motor)
        self.motor_index = MOTORS.index(self.motor)
        models = response_models if response_models is not None else load_response_models(config.response_models)
        self.model = models[self.motor]
        self.geo = Geometry(config.camera)
        self.tracker = ScaleTracker(config, self.geo)
        p = config.plan
        self.az_grid = _grid(p.az_offsets_deg)
        self.el_grid = _grid(p.el_offsets_deg)
        if (self.el_grid < 0).any():
            raise ValueError('v1 has no descending candidates')
        A, E = np.meshgrid(self.az_grid, self.el_grid, indexing='ij')
        self.cand_az, self.cand_el = A.ravel(), E.ravel()          # (155,), index = ia * n_el + ie
        self.cost = self.cand_az ** 2 + p.cost_el_weight * self.cand_el ** 2
        self.t_h = horizon_s(self.model, p)
        self.last = {}
        self.counts = {k: 0 for k in PLAN_KINDS}
        self.counts['held_scale'] = 0

    def reset(self):
        self.tracker.reset()
        self.last = {}

    def snapshot(self):
        """A copy of the time state (tracks, scale pool) for leak tests: `restore` it to re-run a frame."""
        return copy.deepcopy((self.tracker.prev, self.tracker.N, self.tracker.rhs, self.tracker.n_eff,
                              self.tracker.last_valid))

    def restore(self, snap):
        (self.tracker.prev, self.tracker.N, self.tracker.rhs, self.tracker.n_eff,
         self.tracker.last_valid) = copy.deepcopy(snap)

    # ------------------------------------------------------------------ one frame
    def process(self, rgb, disparity, valid, quat_wxyz, velocity, cue, t_capture: float, *, stale: bool = False,
                seq: float = 0., timings: dict | None = None, scale_override: tuple | None = None) -> dict:
        """One processed frame. ``cue``: None, (u, v) normalised, or the race-cue dict (u, v, edge); ``stale``:
        the cue did not arrive in time. ``quat_wxyz``/``velocity`` None: no pose. ``scale_override`` (a, b)
        replaces the tracked scale (offline HUD-only leak test)."""
        nan = float('nan')
        vals = {k: nan for k in PLAN_FIELDS}
        vals.update(time=float(t_capture), seq=float(seq), valid=0., motor=float(self.motor_index))
        for k in ('frame_ms', 'overlay_ms', 'depth_ms'):
            if timings and k in timings:
                vals[k] = float(timings[k])
        self.last = {}
        if not self.enabled:
            return self._finish(vals, 'off')
        if quat_wxyz is None or velocity is None or not np.isfinite(np.asarray(quat_wxyz, float)).all() \
                or not np.isfinite(np.asarray(velocity, float)).all():
            self.tracker.reset()
            return self._finish(vals, 'no_pose')
        q = np.asarray(quat_wxyz, np.float64)
        R = quat_wxyz_to_mat(q / np.linalg.norm(q))
        v = np.asarray(velocity, np.float64)
        valid = np.asarray(valid, bool)
        disparity = np.asarray(disparity, np.float64)
        sc = self.tracker.update(float(t_capture), rgb, valid, R, v, disparity)
        vals['lk_ms'] = sc['lk_ms']
        vals['n_tracks'] = float(sc['n_tracks'])
        vals['scale_n'] = sc['n_eff']
        vals['scale_spread'] = sc['spread']
        if scale_override is not None:
            sc = dict(sc, scale_ok=True, a=float(scale_override[0]), b=float(scale_override[1]), held=False)
        vals = self._decide(vals, disparity, valid, R, v, cue, sc, stale)
        self.last['anchors'] = sc.get('anchors')
        return vals

    def decide_with_scale(self, disparity, valid, quat_wxyz, velocity, cue, t_capture: float, a: float, b: float,
                          *, stale: bool = False) -> dict:
        """P4-P7 for one frame with a given scale (a, b), without touching the tracks or the scale pool (offline
        HUD-only leak test and parity checks)."""
        nan = float('nan')
        vals = {k: nan for k in PLAN_FIELDS}
        vals.update(time=float(t_capture), seq=0., valid=0., motor=float(self.motor_index))
        self.last = {}
        if quat_wxyz is None or velocity is None:
            return self._finish(vals, 'no_pose')
        q = np.asarray(quat_wxyz, np.float64)
        R = quat_wxyz_to_mat(q / np.linalg.norm(q))
        sc = dict(scale_ok=True, a=float(a), b=float(b), held=False, n_eff=nan, spread=nan)
        return self._decide(vals, np.asarray(disparity, np.float64), np.asarray(valid, bool), R,
                            np.asarray(velocity, np.float64), cue, sc, stale)

    def _decide(self, vals, disparity, valid, R, v, cue, sc, stale):
        if sc['scale_ok']:
            vals.update(scale_a=sc['a'], scale_b=sc['b'], scale_held=float(sc['held']))
        self.last['scale'] = sc
        t0 = time.perf_counter()
        speed = float(np.linalg.norm(v))
        vh = math.hypot(v[0], v[1])
        vals['speed'] = speed
        if vh > .5:
            xh = np.array([v[0] / vh, v[1] / vh, 0.])
        else:                                   # nearly vertical flight: the body's horizontal heading
            bx = R[:, 0]
            n = math.hypot(bx[0], bx[1])
            xh = np.array([bx[0] / n, bx[1] / n, 0.]) if n > 1e-6 else np.array([1., 0., 0.])
        yh = np.array([-xh[1], xh[0], 0.])
        course_az = math.atan2(xh[1], xh[0])
        vals['course_az'] = math.degrees(course_az)
        gamma = math.atan2(v[2], max(vh, 1e-6))
        ring = self._ring(cue, R)
        if ring is not None:
            vals['ring_az'], vals['ring_el'] = math.degrees(ring[0]), math.degrees(ring[1])
        p = self.cfg.plan
        D_h = max(p.min_horizon_m, speed * self.t_h)
        vals['d_h'] = D_h
        cloud = None
        if sc['scale_ok'] and speed >= p.min_speed_mps:
            cloud = self._cloud(disparity, valid, R, xh, yh, sc['a'], sc['b'], D_h)
            self._profiles(vals, cloud, gamma)
        if stale:
            kind = 'stale'
        elif ring is None:
            kind = 'no_ring'
        elif speed < p.min_speed_mps:
            kind = 'slow'
        elif not sc['scale_ok']:
            kind = 'no_scale'
        else:
            kind = self._plan(vals, cloud, valid, R, xh, yh, course_az, gamma, ring, speed, D_h)
        vals['plan_ms'] = 1000 * (time.perf_counter() - t0)
        if sc['scale_ok'] and sc['held']:
            self.counts['held_scale'] += 1
        return self._finish(vals, kind)

    def _finish(self, vals, kind):
        vals['kind'] = kind_index(kind)
        vals['valid'] = float(kind in VALID_KINDS)
        self.counts[kind] += 1
        self.last['kind'] = kind
        return vals

    def _ring(self, cue, R):
        """(world azimuth, gravity-levelled elevation) in rad of the ring cue ray, or None (not in view / edge)."""
        if cue is None:
            return None
        if isinstance(cue, dict):
            if cue.get('edge'):
                return None
            uv = (cue.get('u'), cue.get('v'))
        else:
            uv = tuple(np.asarray(cue, np.float64).ravel()[:2])
        if len(uv) != 2 or uv[0] is None or uv[1] is None or not np.isfinite(uv).all():
            return None
        w = self.geo.pixel_ray_world(float(uv[0]) * self.geo.W, float(uv[1]) * self.geo.H, R)
        if not self.geo.in_image(w[None], R, self.cfg.plan.image_margin_deg)[0]:
            return None
        if math.hypot(w[0], w[1]) < 1e-3:
            return None
        return math.atan2(w[1], w[0]), math.asin(float(np.clip(w[2], -1, 1)))

    def _cloud(self, disparity, valid, R, xh, yh, a, b, D_h):
        sc, p = self.cfg.scale, self.cfg.plan
        inv = a * disparity.ravel() + b
        Z = 1.0 / np.maximum(inv, 1.0 / sc.max_range_m)
        keep = valid.ravel() & np.isfinite(Z) & (Z < sc.max_range_m)
        P_w = (self.geo.rays_body[keep] * Z[keep, None]) @ R.T          # world vectors from the camera
        P = np.stack([P_w @ xh, P_w @ yh, P_w[:, 2]], 1)
        m = (P[:, 0] >= p.x_min_m) & (P[:, 0] <= D_h + p.cloud_beyond_horizon_m)
        idx = np.flatnonzero(keep)[m]
        self.last['cloud'] = P[m]
        self.last['cloud_block'] = idx
        return P[m]

    # ------------------------------------------------------------------ paths
    def _paths(self, phi_t, el_t, speed, gamma0, D_h):
        """Static and lag-aware path profiles Y(x), Z(x) (k, n) on x = 0, dx, ... for candidate azimuths relative
        to the course ``phi_t`` and elevations ``el_t`` (rad)."""
        p = self.cfg.plan
        dx = p.path_step_m
        n = int(math.ceil(D_h / dx)) + 2
        xs = np.arange(n) * dx
        phi_c = np.clip(phi_t, -PSI_CLIP, PSI_CLIP)
        el_c = np.clip(el_t, -PSI_CLIP, PSI_CLIP)
        Ys = xs[None, :] * np.tan(phi_c)[:, None]
        Zs = xs[None, :] * (np.tan(el_c) / np.cos(phi_c))[:, None]
        k = len(phi_c)
        v = max(speed, 1e-3)
        k_lat = self.model.a_lat_mps2 / v ** 2
        k_v = p.a_vert_mps2 / v ** 2
        s_straight = speed * self.model.delay_s
        psi = np.zeros(k)
        gam = np.full(k, float(np.clip(gamma0, -PSI_CLIP, PSI_CLIP)))
        s = np.zeros(k)
        Y = np.zeros((k, n))
        Z = np.zeros((k, n))
        for i in range(1, n):
            ds_h = dx / np.cos(psi)
            ds = ds_h / np.cos(gam)
            turn = s >= s_straight
            psi = np.where(turn, psi + np.clip(phi_c - psi, -k_lat * ds_h, k_lat * ds_h), psi)
            gam = np.where(turn, gam + np.clip(el_c - gam, -k_v * ds, k_v * ds), gam)
            Y[:, i] = Y[:, i - 1] + dx * np.tan(psi)
            Z[:, i] = Z[:, i - 1] + dx * np.tan(gam) / np.cos(psi)
            s += ds
        return xs, Ys, Zs, Y, Z

    def _corridor(self, Q, xs, Yp, Zp, D_h, az_rows=None):
        """Blocking counts, free distances and blocking masks of path profiles (k, n) against points Q.
        ``az_rows``: (unique lateral profiles (u, n), row index of each path) when paths share lateral profiles
        (the lateral test then runs once per unique profile)."""
        p = self.cfg.plan
        k = Yp.shape[0]
        m = (Q[:, 0] > p.x_min_m) & (Q[:, 0] < D_h)
        dx = xs[1] - xs[0]
        if m.any():
            # the corridors' envelope at each point's x (a point outside it blocks no path)
            xi = Q[:, 0] / dx
            i0 = np.clip(np.floor(xi).astype(int), 0, len(xs) - 2)
            f = xi - i0
            env = [Yp.min(0) - p.r_lat_m, Yp.max(0) + p.r_lat_m, Zp.min(0) - p.r_vert_m, Zp.max(0) + p.r_vert_m]
            lo_y, hi_y, lo_z, hi_z = ((np.minimum(e[i0], e[i0 + 1]) if j % 2 == 0 else np.maximum(e[i0], e[i0 + 1]))
                                      for j, e in enumerate(env))
            m &= (Q[:, 1] > lo_y) & (Q[:, 1] < hi_y) & (Q[:, 2] > lo_z) & (Q[:, 2] < hi_z)
        idx = np.flatnonzero(m)
        if not len(idx):
            return np.zeros(k, int), np.full(k, D_h), idx, np.zeros((k, 0), bool)
        Qm = Q[idx]
        xi = Qm[:, 0] / dx
        i0 = np.clip(np.floor(xi).astype(int), 0, len(xs) - 2)
        f = (xi - i0).astype(np.float32)
        g = 1 - f
        qy, qz = Qm[:, 1].astype(np.float32), Qm[:, 2].astype(np.float32)
        if az_rows is None:
            Yu, rows = Yp.astype(np.float32), np.arange(k)
        else:
            Yu, rows = az_rows[0].astype(np.float32), az_rows[1]
        lat = np.abs(qy[None, :] - (Yu[:, i0] * g + Yu[:, i0 + 1] * f)) < p.r_lat_m
        Zf = Zp.astype(np.float32)
        blk = lat[rows] & (np.abs(qz[None, :] - (Zf[:, i0] * g + Zf[:, i0 + 1] * f)) < p.r_vert_m)
        cnt = blk.sum(1)
        free = np.full(k, D_h)
        mp = int(p.min_points)
        hit = cnt >= mp
        if hit.any():
            xb = np.where(blk[hit], Qm[None, :, 0], np.inf)
            free[hit] = np.partition(xb, mp - 1, axis=1)[:, mp - 1]
        return cnt, free, idx, blk

    def _footprint_ok(self, R, xh, yh, valid, xs, Yp, Zp, D_h):
        """>= footprint_valid of the blocks whose rays cross the corridor cross-section at min(D_h, 4 m) are valid."""
        p = self.cfg.plan
        rw = self.geo.rays_body @ R.T
        rx, ry, rz = rw @ xh, rw @ yh, rw[:, 2]
        fwd = rx > 1e-3
        x_f = min(D_h, p.footprint_x_m)
        yb = np.where(fwd, x_f * ry / np.where(fwd, rx, 1.), np.inf)
        zb = np.where(fwd, x_f * rz / np.where(fwd, rx, 1.), np.inf)
        dx = xs[1] - xs[0]
        i0 = min(int(x_f // dx), len(xs) - 2)
        fr = x_f / dx - i0
        yc = Yp[:, i0] * (1 - fr) + Yp[:, i0 + 1] * fr
        zc = Zp[:, i0] * (1 - fr) + Zp[:, i0 + 1] * fr
        sel = fwd & (yb > yc.min() - p.r_lat_m) & (yb < yc.max() + p.r_lat_m) & \
            (zb > zc.min() - p.r_vert_m) & (zb < zc.max() + p.r_vert_m)
        vb = valid.ravel()[sel]
        inside = (np.abs(yb[sel][None, :] - yc[:, None]) < p.r_lat_m) & (np.abs(zb[sel][None, :] - zc[:, None]) < p.r_vert_m)
        n_in = inside.sum(1)
        n_valid = (inside & vb[None, :]).sum(1)
        frac = np.where(n_in > 0, n_valid / np.maximum(n_in, 1), 0.)
        return (n_in > 0) & (frac >= p.footprint_valid), frac, n_in

    def _aperture(self, Qb, xs, Y, Z):
        """Ring corridor aperture: blocking points split left / right of the path at similar nearest range with a
        wide enough gap that the path passes inside both edges."""
        p = self.cfg.plan
        if len(Qb) < 2:
            return False
        dy = Qb[:, 1] - np.interp(Qb[:, 0], xs, Y)
        L, Rr = dy > 0, dy < 0
        if not L.any() or not Rr.any():
            return False
        aL, aR = Qb[L, 0].min(), Qb[Rr, 0].min()
        if max(aL, aR) / max(min(aL, aR), 1e-6) > p.aperture_ratio:
            return False
        inner_l, inner_r = dy[L].min(), dy[Rr].max()
        return bool(inner_l - inner_r >= p.aperture_gap_m and inner_l >= p.aperture_edge_m
                    and -inner_r >= p.aperture_edge_m)

    def _plan(self, vals, Q, valid, R, xh, yh, course_az, gamma, ring, speed, D_h):
        p = self.cfg.plan
        ring_az, ring_el = ring
        cand_az_w = ring_az + np.radians(self.cand_az)                    # world azimuths
        cand_el = ring_el + np.radians(self.cand_el)
        phi = _wrap_rad(cand_az_w - course_az)
        dirs = np.stack([np.cos(cand_el) * np.cos(cand_az_w), np.cos(cand_el) * np.sin(cand_az_w), np.sin(cand_el)], 1)
        in_img = self.geo.in_image(dirs, R, p.image_margin_deg)
        xs, Ys, Zs, Yl, Zl = self._paths(phi, cand_el, speed, gamma, D_h)
        k = len(phi)
        Yall, Zall = np.vstack([Ys, Yl]), np.vstack([Zs, Zl])
        # lateral profiles depend on the candidate azimuth only (both path types): test them once per azimuth
        n_el = len(self.el_grid)
        ia = np.arange(k) // n_el
        az_rows = (np.vstack([Ys[::n_el], Yl[::n_el]]), np.concatenate([ia, ia + len(self.az_grid)]))
        cnt, free, idx, blk = self._corridor(Q, xs, Yall, Zall, D_h, az_rows)
        fp_ok, fp_frac, fp_n = self._footprint_ok(R, xh, yh, valid, xs, Yall, Zall, D_h)
        mp = int(p.min_points)
        static_free, lag_free = cnt[:k] < mp, cnt[k:] < mp
        static_elig, lag_elig = in_img & fp_ok[:k], in_img & fp_ok[k:]
        d_lag = free[k:]
        i0 = int(np.flatnonzero((self.cand_az == 0) & (self.cand_el == 0))[0])
        ring_blk = blk[k + i0] if blk.shape[1] else np.zeros(0, bool)
        Qb = Q[idx[ring_blk]] if len(idx) else np.zeros((0, 3))
        vals['d_free_ring'] = float(d_lag[i0])
        vals['n_block_ring'] = float(cnt[k + i0])
        if len(Qb):
            vals['below_frac'] = float((Qb[:, 2] < np.interp(Qb[:, 0], xs, Zl[i0])).mean())
        options = self._options(static_free, static_elig, lag_free, lag_elig, d_lag)
        for key, cls in (('l', CLS_LEFT), ('r', CLS_RIGHT)):
            o = options.get(cls)
            if o is not None:
                vals[f'{key}_az'], vals[f'{key}_el'], vals[f'{key}_ok'] = o['az'], o['el'], float(o['ok'])
        o = options.get(CLS_VERTICAL)
        if o is not None:
            vals['v_el'], vals['v_ok'] = o['el'], float(o['ok'])
        aperture = False
        if not lag_free[i0]:
            aperture = self._aperture(Qb, xs, Yl[i0], Zl[i0])
        self.last.update(xs=xs, Ys=Ys, Zs=Zs, Yl=Yl, Zl=Zl, cnt=cnt, free=free, static_free=static_free,
                         lag_free=lag_free, static_elig=static_elig, lag_elig=lag_elig, in_image=in_img,
                         footprint_frac=fp_frac, footprint_n=fp_n, options=options, ring_points=Qb, i0=i0,
                         ring=(ring_az, ring_el), course_az=course_az, gamma=gamma)
        if lag_free[i0] or aperture:
            vals.update(cls=float(CLS_NONE), az=0., el=0.)
            return 'aperture' if (aperture and not lag_free[i0]) else 'clear'
        if not options:
            vals.update(cls=float(CLS_NONE), az=0., el=0.)
            elig = lag_elig
            D = float(d_lag[elig].max()) if elig.any() else float(d_lag[i0])
            vals['v_cap'] = stop_speed(D, self.model.delay_s, p.brake_mps2, p.stop_margin_m)
            return 'blocked'
        order = {CLS_LEFT: 0, CLS_RIGHT: 1, CLS_VERTICAL: 2}
        best = min(options.items(), key=lambda kv: (kv[1]['cost'], -kv[1]['ok'], -kv[1]['d_free'], order[kv[0]]))
        cls, o = best
        vals.update(cls=float(cls), az=float(o['az']), el=float(o['el']), feasible=float(o['ok']))
        if not o['ok']:
            vals['v_cap'] = stop_speed(o['d_free'], self.model.delay_s, p.brake_mps2, p.stop_margin_m)
        return 'shift'

    def _options(self, static_free, static_elig, lag_free, lag_elig, d_lag) -> dict:
        """Per class: c* (cheapest static-free eligible candidate) pushed outward until its lag-aware path is free
        and eligible (ok), else saturated at the bound (urgent). No option for a class without c*."""
        p = self.cfg.plan
        n_el = len(self.el_grid)
        az_g, el_g = self.az_grid, self.el_grid
        ok_static = static_free & static_elig
        classes = {CLS_LEFT: (self.cand_az > 0) | ((self.cand_az == 0) & (self.cand_el == 0)),
                   CLS_RIGHT: (self.cand_az < 0) | ((self.cand_az == 0) & (self.cand_el == 0)),
                   CLS_VERTICAL: (self.cand_az == 0) & (self.cand_el > 0)}
        out = {}
        for cls, member in classes.items():
            cands = np.flatnonzero(member & ok_static)
            if not len(cands):
                continue
            # min cost; ties: smaller |daz|, then smaller del
            j = min(cands, key=lambda c: (self.cost[c], abs(self.cand_az[c]), self.cand_el[c]))
            ia, ie = divmod(int(j), n_el)
            steps = []
            if cls == CLS_VERTICAL:
                ie_max = int(np.flatnonzero(el_g <= p.max_el_deg + 1e-9)[-1])
                steps = [ia * n_el + e for e in range(ie, ie_max + 1)]
                sat = ia * n_el + ie_max
            else:
                sgn = 1 if cls == CLS_LEFT else -1
                bound = sgn * p.max_az_deg
                ib = int(np.argmin(np.abs(az_g - bound)))
                a = ia
                while 0 <= a < len(az_g) and sgn * az_g[a] <= p.max_az_deg + 1e-9:
                    steps.append(a * n_el + ie)
                    a += sgn
                sat = ib * n_el + ie
            chosen, ok = sat, False
            for c in steps:
                if lag_free[c] and lag_elig[c]:
                    chosen, ok = c, True
                    break
            out[cls] = dict(az=float(self.cand_az[chosen]), el=float(self.cand_el[chosen]), ok=bool(ok),
                            d_free=float(d_lag[chosen]), cost=float(self.cost[j]), c_star=int(j), index=int(chosen))
        return out

    # ------------------------------------------------------------------ P7
    def _profiles(self, vals, Q, gamma):
        pr = self.cfg.profile
        strip = Q[np.abs(Q[:, 1]) < pr.lateral_m]
        if len(strip) < 2:
            return
        b = np.floor(strip[:, 0] / pr.bin_m).astype(int)
        order = np.argsort(b, kind='stable')
        b, strip = b[order], strip[order]
        ub, start, cnt = np.unique(b, return_index=True, return_counts=True)
        g_up = math.tan(max(gamma, 0.))
        lo_x, hi_x = pr.near_x_m
        g_near, c_near, bins = [], [], []
        for bb, s0, n in zip(ub, start, cnt):
            if n < 2:
                continue
            z = strip[s0:s0 + n, 2]
            x = strip[s0:s0 + n, 0]
            zs = np.sort(z)
            g, c = zs[1], zs[-2]
            x_lo = bb * pr.bin_m
            if lo_x <= x_lo < hi_x:
                if g < 0:
                    g_near.append(g)
                if c > pr.ceil_min_m:
                    c_near.append(c)
            zr = np.sort(z - x * g_up)
            bins.append((x_lo + pr.bin_m / 2, zr[1]))
        if len(g_near) >= pr.min_bins:
            vals['h_floor'] = float(-np.median(g_near))
        if len(c_near) >= pr.min_bins:
            vals['h_ceil'] = float(np.median(c_near))
        # Rising ground: rise_bins consecutive data bins with g' > -rise_level, reached from the nearest data bin
        # without a step > rise_max_step between consecutive data bins (a wall with a masked base jumps; a pillar
        # is one bin long).
        run, run_start, found = 0, None, None
        for i, (x, g) in enumerate(bins):
            if i > 0 and abs(g - bins[i - 1][1]) > pr.rise_max_step_m:
                break                           # later bins are not reached
            if g > -pr.rise_level_m:
                if run == 0:
                    run_start = i
                run += 1
                if run >= pr.rise_bins:
                    found = run_start
                    break
            else:
                run = 0
        if found is not None:
            # the reached chain from the first rising bin until a step > rise_max_step (or the last bin)
            chain = [bins[found]]
            for i in range(found + 1, len(bins)):
                if abs(bins[i][1] - bins[i - 1][1]) > pr.rise_max_step_m:
                    break
                chain.append(bins[i])
            vals['rise_x'] = float(bins[found][0])
            vals['rise'] = float(max(g for _, g in chain) + pr.rise_level_m)

    def status(self) -> dict:
        return dict(motor=self.motor, horizon_s=round(self.t_h, 4), counts=dict(self.counts))


def plan_values(sample: dict) -> list:
    """PLAN_FIELDS values (list of floats) of a process() result."""
    return [float(sample[k]) for k in PLAN_FIELDS]


def from_config(path: str | Path = CONFIG_PATH, *, motor='brain08', enabled: bool | None = None,
                require_frozen: bool = False) -> FreeSpacePlanner:
    config, _, _ = load_config(path, require_frozen=require_frozen)
    return FreeSpacePlanner(config, motor, enabled=enabled)
