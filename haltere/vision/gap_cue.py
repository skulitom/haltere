"""Gap cue: a gravity-levelled free-interval aim shift from one frame's relative disparity (runtime-safe).

OFF BY DEFAULT (``enabled: false`` in configs/obstacles/gap_cue.json); nothing in the flight stack calls it yet.
Imports numpy and the standard library only.

Inputs, all causal and from one frame: a relative-disparity map of the current frame (any (h, w) block grid
that tiles the camera image, e.g. the (36, 64) 7 x 7 px block means of haltere.vision.relative_depth),
optional per-block validity (``block_validity`` of the frame's overlay masks: HUD glyphs, ring marker and
checkpoint volumes, ghost trails, propeller zone; haltere.obstacles.overlays), the camera intrinsics and
uptilt, the capture attitude (world-from-body quaternion, wxyz), the ring cue (normalised image position of
the next-checkpoint marker detected in the same frame) and the declared world velocity. No course geometry,
routes, per-course parameters or offline labels.

Rule (ported from the M2 design study ``m2design/hybrid/gapcue.py``; parameters in GapCueParams):

1. Band: block-centre rays rotated to the world by the capture attitude; blocks whose gravity-levelled
   elevation lies in ``band_deg`` = (-4, +6) deg and that are valid.
2. Profile: azimuth relative to the ring bearing (positive = LEFT, the FLU yaw sense) in 1-degree bins over
   +-60 deg; a bin holds the maximum disparity of the band blocks within +-1.5 deg of it.
3. Background b = median of the profile within +-35 deg of the ring; ratio r = profile / b. A column is
   NEAR where r > kappa (1.8). Columns without valid band blocks are UNKNOWN: neither near nor observed free.
4. Free interval: the run of not-near columns containing the ring (kind ``gap``, or ``clear`` when no column
   within +-25 deg is near). When the ring column is near, the run containing the nearest observed-free
   column within +-25 deg (kind ``occluded``); none: kind ``blocked``, shift 0. An interval edge bounded by
   near columns keeps a margin mu0 * clip(r_edge / kappa, 1, 2.5), mu0 = 5 deg, r_edge = the peak ratio of
   the adjacent near run; unknown columns and the profile ends do not bound. Target = the point of
   [lo + margin_lo, hi - margin_hi] nearest the ring (the interval midpoint when the margins overlap);
   shift = target clipped to +-12 deg (positive = aim LEFT of the ring).
5. Diagnostics: r_ring (max r within +-2 deg of the ring), r_peak (max r within +-8 deg), lr = ln(median
   band disparity 8-35 deg left / median 8-35 deg right), the terrain side statistic (> 0: left side nearer).
6. near_on_path: a near column lies within +-path_halfwidth_deg of the predicted flown path over
   path_horizon_s: straight along the declared horizontal velocity for the motor's response delay, then
   turning toward the aim (ring + shift) at the motor's lateral acceleration (declared per motor in
   configs/obstacles/response_models.json). It flags obstacles on the path the vehicle is committed to
   (e.g. a course that still lags a new ring) even when the ring line itself is free; it does not change
   the shift.

``GapCue`` adds the time logic: a shift is CONFIRMED after ``confirm_frames`` consecutive fresh frames with
|shift| >= active_deg on the same side and released after ``release_frames`` fresh frames without (the last
confirmed shift is held meanwhile, and follows the latest active frame); an invalid frame (no ring cue,
too few band blocks) or a side change releases at once; a frame older than ``max_age_s`` (capture to now)
gives shift 0 at once (kind ``stale``); a gap between fresh frames > ``max_gap_s`` restarts confirmation.
The output shift is 0 unless confirmed.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from .camera import Camera, quat_wxyz_to_mat

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'obstacles' / 'gap_cue.json'
RESPONSE_MODELS_PATH = REPO_ROOT / 'configs' / 'obstacles' / 'response_models.json'
CONFIG_META_KEYS = ('frozen', 'frozen_at', 'sha256')

# The store / relative_depth image: 448 x 252, f = 100 px at 320 wide = 140 px, 30 deg uptilt
# (haltere.obstacles.contract; not imported here so the vision package stays free of obstacle code).
DEFAULT_CAMERA = Camera(448, 252, 140.0, 30.0)

KINDS = ('off', 'no_ring', 'stale', 'invalid', 'clear', 'gap', 'occluded', 'blocked')


@dataclass(frozen=True)
class GapCueParams:
    enabled: bool = False
    band_deg: tuple = (-4.0, 6.0)             # gravity-levelled elevation band (lo, hi), deg
    az_span_deg: float = 60.0                 # profile covers +-span around the ring bearing
    az_step_deg: float = 1.0
    bin_halfwidth_deg: float = 1.5            # a bin takes the max over blocks within this of its centre
    background_halfwidth_deg: float = 35.0    # background = median profile within this of the ring
    min_background_bins: int = 10
    kappa: float = 1.8                        # near: disparity ratio to the background above this
    mu0_deg: float = 5.0                      # margin at a near edge: mu0 * clip(r / kappa, 1, cap)
    margin_ratio_cap: float = 2.5
    search_halfwidth_deg: float = 25.0        # occluded ring: nearest observed-free column within this
    shift_clip_deg: float = 12.0
    ring_halfwidth_deg: float = 2.0           # r_ring window
    peak_halfwidth_deg: float = 8.0           # r_peak window
    lr_inner_deg: float = 8.0                 # terrain side statistic: band samples 8-35 deg each side
    lr_outer_deg: float = 35.0
    lr_min_blocks: int = 5
    min_band_blocks: int = 20
    cue_edge_margin: float = 0.03             # the ring cue must lie inside [m, 1 - m] (not edge-clamped)
    block_mask_max_fraction: float = 0.0      # block_validity: a block is invalid above this masked fraction
    active_deg: float = 2.0                   # |shift| counted towards confirmation
    confirm_frames: int = 2
    release_frames: int = 2
    max_age_s: float = 0.25                   # capture-to-now age above which the output is 0 (stale)
    max_gap_s: float = 0.3                    # fresh-frame gap that restarts confirmation
    path_halfwidth_deg: float = 4.0           # near_on_path corridor half-width around the predicted path
    path_horizon_s: float = 1.0
    path_min_speed_mps: float = 1.0
    motor: str = 'brain08'                    # response model used for near_on_path

    @classmethod
    def from_dict(cls, d: dict) -> 'GapCueParams':
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f'unknown gap cue parameters: {sorted(unknown)}')
        d = dict(d)
        if 'band_deg' in d:
            d['band_deg'] = tuple(float(x) for x in d['band_deg'])
        return cls(**d)

    def as_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out['band_deg'] = list(out['band_deg'])
        return out


@dataclass(frozen=True)
class ResponseModel:
    """Declared vehicle response to an aim change: straight for ``delay_s``, then lateral acceleration."""
    name: str
    delay_s: float
    a_lat_mps2: float


# ----------------------------------------------------------------------------- configuration

def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')


def config_sha256(obj: dict) -> str:
    """sha256 of a config object without its meta keys (frozen, frozen_at, sha256)."""
    body = {k: v for k, v in obj.items() if k not in CONFIG_META_KEYS}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def load_config(path: str | Path = CONFIG_PATH, *, require_frozen: bool = False) -> tuple[GapCueParams, dict, str]:
    """(params, raw object, content sha256). ``require_frozen`` refuses a draft or a changed frozen file."""
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    sha = config_sha256(obj)
    if obj.get('frozen') and obj.get('sha256') != sha:
        raise RuntimeError(f'{path} changed after freezing (sha256 mismatch)')
    if require_frozen and not obj.get('frozen'):
        raise RuntimeError(f'{path} is not frozen')
    return GapCueParams.from_dict(obj.get('params', {})), obj, sha


def load_response_models(path: str | Path = RESPONSE_MODELS_PATH) -> dict:
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    return {name: ResponseModel(name, float(m['delay_s']), float(m['a_lat_mps2']))
            for name, m in obj['models'].items()}


# ----------------------------------------------------------------------------- geometry

def wrap_deg(a):
    return (np.asarray(a, np.float64) + 180.0) % 360.0 - 180.0


@lru_cache(maxsize=16)
def _block_rays(h: int, w: int, width: int, height: int, f: float, tilt_deg: float) -> np.ndarray:
    cam = Camera(width, height, f, tilt_deg)
    bw, bh = width / w, height / h
    rr, cc = np.mgrid[0:h, 0:w]
    px = np.stack([(cc.ravel() + 0.5) * bw, (rr.ravel() + 0.5) * bh], 1)
    rays = cam.unproject_body(px)
    rays.setflags(write=False)
    return rays


def block_rays_body(shape, camera: Camera = DEFAULT_CAMERA) -> np.ndarray:
    """Unit body-frame (FLU) rays through the centres of an (h, w) block grid tiling the camera image."""
    h, w = int(shape[0]), int(shape[1])
    return _block_rays(h, w, int(camera.width), int(camera.height), float(camera.f), float(camera.tilt_deg))


def block_validity(mask, shape, max_fraction: float = GapCueParams.block_mask_max_fraction) -> np.ndarray:
    """(H, W) pixel mask (True = overlay, not scene) -> (h, w) bool, True where at most ``max_fraction`` of
    the block's pixels are masked. The block grid must tile the pixel grid."""
    m = np.asarray(mask, bool)
    H, W = m.shape
    h, w = int(shape[0]), int(shape[1])
    if H % h or W % w:
        raise ValueError(f'block grid {h}x{w} does not tile a {H}x{W} mask')
    frac = m.reshape(h, H // h, w, W // w).mean(axis=(1, 3))
    return frac <= max_fraction


def ring_bearing_deg(cue_uv, quat_wxyz, camera: Camera = DEFAULT_CAMERA, edge_margin: float = 0.03):
    """World azimuth (deg, FLU: 0 = +x, positive towards +y) of the ring cue ray, or None."""
    if cue_uv is None:
        return None
    cue = np.asarray(cue_uv, np.float64).ravel()
    if cue.size != 2 or not np.isfinite(cue).all():
        return None
    if not (edge_margin < cue[0] < 1 - edge_margin and edge_margin < cue[1] < 1 - edge_margin):
        return None
    R = quat_wxyz_to_mat(np.asarray(quat_wxyz, np.float64) / np.linalg.norm(quat_wxyz))
    ray = R @ camera.unproject_body(np.array([[cue[0] * camera.width, cue[1] * camera.height]]))[0]
    if math.hypot(ray[0], ray[1]) < 1e-3:
        return None
    return float(np.degrees(np.arctan2(ray[1], ray[0])))


def predicted_path(course_rel_deg: float, speed_mps: float, aim_rel_deg: float, model: ResponseModel,
                   horizon_s: float, dt: float = 0.02) -> tuple[float, float]:
    """Bearings (deg, relative to the ring) swept by the predicted flown path over ``horizon_s``: straight
    along the current course for the model's delay, then turning towards the aim at a_lat / speed."""
    h = math.radians(course_rel_deg)
    aim = math.radians(aim_rel_deg)
    omega = model.a_lat_mps2 / max(speed_mps, 0.1)
    x = y = 0.0
    lo = hi = course_rel_deg
    n = max(1, int(round(horizon_s / dt)))
    for k in range(1, n + 1):
        t = k * dt
        if t > model.delay_s:
            err = (aim - h + math.pi) % (2 * math.pi) - math.pi
            h += max(-omega * dt, min(omega * dt, err))
        x += speed_mps * dt * math.cos(h)
        y += speed_mps * dt * math.sin(h)
        b = math.degrees(math.atan2(y, x))
        lo, hi = min(lo, b), max(hi, b)
    return lo, hi


# ----------------------------------------------------------------------------- one frame

@dataclass
class GapDecision:
    valid: bool
    kind: str
    shift_deg: float = 0.0                 # clipped target relative to the ring, positive = LEFT
    target_deg: float = 0.0                # unclipped target
    ring_bearing_deg: float | None = None  # world azimuth of the ring
    r_ring: float = float('nan')
    r_peak: float = float('nan')
    lr: float = float('nan')
    background: float = float('nan')
    n_band: int = 0
    interval_deg: tuple | None = None      # (lo, hi) free interval used, relative to the ring
    margins_deg: tuple = (0.0, 0.0)        # (at lo, at hi)
    near_on_path: bool | None = None
    path_deg: tuple | None = None          # (lo, hi) predicted path bearings relative to the ring
    course_rel_deg: float | None = None
    az_deg: np.ndarray | None = field(default=None, repr=False)     # profile azimuths (relative)
    ratio: np.ndarray | None = field(default=None, repr=False)      # profile / background (NaN unknown)

    @property
    def blocked(self) -> bool:
        return self.kind == 'blocked'

    @property
    def near(self) -> np.ndarray | None:
        if self.ratio is None:
            return None
        return np.isfinite(self.ratio) & (self.ratio > self._kappa)

    _kappa: float = field(default=GapCueParams.kappa, repr=False)


def _profile(disp, rays_b, quat_wxyz, ring_deg, valid, p: GapCueParams):
    R = quat_wxyz_to_mat(np.asarray(quat_wxyz, np.float64) / np.linalg.norm(quat_wxyz))
    w = rays_b @ R.T
    el = np.degrees(np.arcsin(np.clip(w[:, 2], -1.0, 1.0)))
    az = np.degrees(np.arctan2(w[:, 1], w[:, 0]))
    d = np.asarray(disp, np.float64).ravel()
    m = (el > p.band_deg[0]) & (el < p.band_deg[1]) & np.isfinite(d)
    if valid is not None:
        m &= np.asarray(valid, bool).ravel()
    rel = wrap_deg(az[m] - ring_deg)
    dm = d[m]
    AZ = np.arange(-p.az_span_deg, p.az_span_deg + p.az_step_deg / 2, p.az_step_deg)
    prof = np.full(AZ.shape, -np.inf)
    # bins k with |rel - AZ[k]| < halfwidth: a small set of neighbours of the nearest bin
    k_near = np.rint((rel + p.az_span_deg) / p.az_step_deg).astype(np.int64)
    reach = int(math.ceil(p.bin_halfwidth_deg / p.az_step_deg))
    for off in range(-reach, reach + 1):
        k = k_near + off
        ok = (k >= 0) & (k < len(AZ))
        ok[ok] &= np.abs(rel[ok] - AZ[k[ok]]) < p.bin_halfwidth_deg
        if ok.any():
            np.maximum.at(prof, k[ok], dm[ok])
    prof[~np.isfinite(prof)] = np.nan
    return AZ, prof, rel, dm


def _run_peak(ratio, near, k, step):
    """Peak ratio of the contiguous near run starting at column k and extending in direction ``step``."""
    best = 0.0
    n = len(ratio)
    while 0 <= k < n and near[k]:
        best = max(best, float(ratio[k]))
        k += step
    return best


def decide(disp, quat_wxyz, cue_uv, *, velocity=None, valid=None, params: GapCueParams | None = None,
           camera: Camera = DEFAULT_CAMERA, response: ResponseModel | None = None,
           keep_profile: bool = False) -> GapDecision:
    """Stateless gap-cue decision for one frame (see the module docstring). ``valid``: (h, w) bool blocks."""
    p = params or GapCueParams()
    rb = ring_bearing_deg(cue_uv, quat_wxyz, camera, p.cue_edge_margin)
    if rb is None:
        return GapDecision(False, 'no_ring', _kappa=p.kappa)
    disp = np.asarray(disp)
    rays = block_rays_body(disp.shape, camera)
    AZ, prof, rel, dm = _profile(disp, rays, quat_wxyz, rb, valid, p)
    n_band = int(len(dm))
    out = GapDecision(False, 'invalid', ring_bearing_deg=rb, n_band=n_band, _kappa=p.kappa)
    if n_band < p.min_band_blocks:
        return out
    inw = np.abs(AZ) <= p.background_halfwidth_deg
    if np.isfinite(prof[inw]).sum() < p.min_background_bins:
        return out
    b = float(np.nanmedian(prof[inw]))
    if not np.isfinite(b) or b <= 0:
        return out
    r = prof / b
    known = np.isfinite(r)
    near = known & (r > p.kappa)
    c = int(np.argmin(np.abs(AZ)))
    with np.errstate(all='ignore'):
        rr = r[np.abs(AZ) <= p.ring_halfwidth_deg]
        rp = r[np.abs(AZ) <= p.peak_halfwidth_deg]
        out.r_ring = float(np.nanmax(rr)) if np.isfinite(rr).any() else float('nan')
        out.r_peak = float(np.nanmax(rp)) if np.isfinite(rp).any() else float('nan')
        L = dm[(rel > p.lr_inner_deg) & (rel < p.lr_outer_deg)]
        Rt = dm[(rel < -p.lr_inner_deg) & (rel > -p.lr_outer_deg)]
        if len(L) >= p.lr_min_blocks and len(Rt) >= p.lr_min_blocks:
            out.lr = float(np.log(max(np.median(L), 1e-6) / max(np.median(Rt), 1e-6)))
    out.background = b
    out.valid = True
    if keep_profile:
        out.az_deg, out.ratio = AZ, r
    win = np.abs(AZ) <= p.search_halfwidth_deg
    if not near[c]:
        k0 = c
        out.kind = 'gap' if near[win].any() else 'clear'
    else:
        cand = np.flatnonzero(known & ~near & win)
        if len(cand) == 0:
            out.kind = 'blocked'
            _path(out, near, AZ, 0.0, velocity, rb, p, response)
            return out
        k0 = int(cand[np.argmin(np.abs(AZ[cand]))])
        out.kind = 'occluded'
    free = ~near
    lo = hi = k0
    while lo - 1 >= 0 and free[lo - 1]:
        lo -= 1
    while hi + 1 < len(AZ) and free[hi + 1]:
        hi += 1
    m_lo = m_hi = 0.0
    if lo - 1 >= 0 and near[lo - 1]:
        m_lo = p.mu0_deg * float(np.clip(_run_peak(r, near, lo - 1, -1) / p.kappa, 1.0, p.margin_ratio_cap))
    if hi + 1 < len(AZ) and near[hi + 1]:
        m_hi = p.mu0_deg * float(np.clip(_run_peak(r, near, hi + 1, +1) / p.kappa, 1.0, p.margin_ratio_cap))
    lo_b, hi_b = AZ[lo] + m_lo, AZ[hi] - m_hi
    target = float(np.clip(0.0, lo_b, hi_b)) if lo_b <= hi_b else float((AZ[lo] + AZ[hi]) / 2)
    out.target_deg = target
    out.shift_deg = float(np.clip(target, -p.shift_clip_deg, p.shift_clip_deg))
    out.interval_deg = (float(AZ[lo]), float(AZ[hi]))
    out.margins_deg = (m_lo, m_hi)
    _path(out, near, AZ, out.shift_deg, velocity, rb, p, response)
    return out


def _path(out: GapDecision, near, AZ, aim_rel, velocity, ring_deg, p: GapCueParams, response):
    if velocity is None or response is None:
        return
    v = np.asarray(velocity, np.float64).ravel()
    if v.size < 2 or not np.isfinite(v[:2]).all():
        return
    speed = float(math.hypot(v[0], v[1]))
    if speed < p.path_min_speed_mps:
        out.near_on_path = False
        return
    course_rel = float(wrap_deg(math.degrees(math.atan2(v[1], v[0])) - ring_deg))
    lo, hi = predicted_path(course_rel, speed, aim_rel, response, p.path_horizon_s)
    out.course_rel_deg = course_rel
    out.path_deg = (lo, hi)
    on = (AZ >= lo - p.path_halfwidth_deg) & (AZ <= hi + p.path_halfwidth_deg)
    out.near_on_path = bool((near & on).any())


# ----------------------------------------------------------------------------- time logic

@dataclass
class GapCueOutput:
    shift_deg: float                 # aim shift relative to the ring (positive = LEFT); 0 unless confirmed
    confirmed: bool
    kind: str
    age_s: float
    episode: int                     # confirmed episodes so far
    decision: GapDecision | None


class GapCue:
    """Stateful gap cue: stateless ``decide`` per fresh frame plus confirmation, release and staleness."""

    def __init__(self, params: GapCueParams | None = None, camera: Camera = DEFAULT_CAMERA,
                 response: ResponseModel | None = None, keep_profile: bool = False):
        self.params = params or GapCueParams()
        self.camera = camera
        self.response = response
        self.keep_profile = keep_profile          # keep each decision's profile (diagnostics / evaluation)
        self.reset()

    def reset(self):
        self._sign = 0
        self._count = 0
        self._miss = 0
        self._confirmed = False
        self._conf_sign = 0
        self._shift = 0.0
        self._last_t = None
        self._last_decision = None
        self.episodes = 0

    def _release(self):
        self._confirmed = False
        self._conf_sign = 0
        self._shift = 0.0
        self._miss = 0

    def _out(self, kind, age, decision):
        return GapCueOutput(self._shift if self._confirmed else 0.0, self._confirmed, kind, float(age),
                            self.episodes, decision)

    def update(self, disparity, quat_wxyz, cue_uv, t_capture: float, now: float | None = None, *,
               velocity=None, valid=None) -> GapCueOutput:
        p = self.params
        now = t_capture if now is None else now
        age = float(now - t_capture)
        if not p.enabled:
            return GapCueOutput(0.0, False, 'off', age, self.episodes, None)
        if age > p.max_age_s:
            self._release()
            self._count = 0
            self._sign = 0
            return self._out('stale', age, None)
        if self._last_t is not None and t_capture <= self._last_t:
            # the same frame again (or an older one): no new evidence
            d = self._last_decision
            return self._out(d.kind if d is not None else 'invalid', age, d)
        if self._last_t is not None and t_capture - self._last_t > p.max_gap_s:
            self._release()
            self._count = 0
            self._sign = 0
        self._last_t = t_capture
        d = decide(disparity, quat_wxyz, cue_uv, velocity=velocity, valid=valid, params=p, camera=self.camera,
                   response=self.response, keep_profile=self.keep_profile)
        self._last_decision = d
        active = d.valid and abs(d.shift_deg) >= p.active_deg
        if active:
            s = 1 if d.shift_deg > 0 else -1
            self._count = self._count + 1 if s == self._sign else 1
            self._sign = s
            self._miss = 0
            if self._confirmed and s != self._conf_sign:
                self._release()
            if not self._confirmed and self._count >= p.confirm_frames:
                self._confirmed = True
                self._conf_sign = s
                self.episodes += 1
            if self._confirmed:
                self._shift = d.shift_deg
        else:
            self._count = 0
            self._sign = 0
            if self._confirmed:
                self._miss += 1
                if not d.valid or self._miss >= p.release_frames:
                    self._release()
        return self._out(d.kind, age, d)


def from_config(path: str | Path = CONFIG_PATH, *, enabled: bool | None = None, motor: str | None = None,
                camera: Camera = DEFAULT_CAMERA) -> GapCue:
    """A GapCue from configs/obstacles/gap_cue.json (``enabled`` / ``motor`` override the file)."""
    params, _, _ = load_config(path)
    if enabled is not None:
        params = replace(params, enabled=enabled)
    if motor is not None:
        params = replace(params, motor=motor)
    models = load_response_models()
    return GapCue(params, camera, models.get(params.motor))
