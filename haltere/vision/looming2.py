"""Time-to-contact along the flight path from image expansion: a fly-style looming cue (v2).

Changes from haltere/vision/looming.py (v1):
  * rotation: exact attitude-delta homography from the two frames' telemetry quaternions (not omega*dt), then a
    small visual correction fitted from the flow component perpendicular to the focus-of-expansion (FOE) ray.
    Translational flow is radial from the FOE whatever the depth, so that component is rotation only; the
    correction absorbs camera/telemetry timing error.
  * model: instead of an affine divergence (TTC = 2/div, fronto-parallel only), the translational flow with known
    direction is fitted with a planar inverse-depth field,  flow = (u0,v0) + rho(q)*Tz*q,
    rho(q) = c0 + c1*qx/f + c2*qy/f,  q = pixel - FOE.  rho at the FOE (c0, 1/m) is where the fitted surface
    crosses the flight path, so TTC_path = 1/(c0 * forward speed). A ground plane parallel to the path gives
    c0 = 0 (no alarm); rising terrain or a wall gives c0 > 0. Fits are robust (Huber IRLS).
  * windows (default = the low-false-alarm variant selected offline): planar fits in a wide (+/-0.22 W) and a
    central (+/-0.11 W) window, most urgent of the two, AND a fronto-parallel fit in an inner window (+/-0.08 W)
    at the FOE: both must alarm. The inner window suppresses race-gate openings and passing structure whose frame
    looms around, but not on, the flight path (gate proximity was ~2/3 of v1's clean-flight alarms).
    Windows may be offset from the FOE (dx, dy) for corridor layouts.
  * contrast: CLAHE before flow (dark, low-contrast walls: evidence fraction near the Minus Two wall 0.25 -> 0.45).
  * evidence and time: a window needs >= 35 % usable textured pixels and the alarm uses rho - 1 standard error;
    when a frame has no evidence the last TTC keeps counting down for `hold` s; output is the median of the last
    3 frames (a gap >= 0.3 s restarts the median).
Offline evaluation (recorded videos, 12 impacts / 10 clean segments) is in the same folder: final_table.md.
  * where along the vertical (report only; the alarm above is unchanged): two planar windows 0.12 W above ('up')
    and below ('lo') the FOE (half-size 0.14 W x 0.08 W) with the same model. Each window's c0 is where ITS fitted
    surface crosses the flight path: ttc_upper / ttc_lower = 1/((c0 - se) * forward), urgency U = 1/ttc (0 when the
    window has evidence but no crossing). below_fraction = U_lo / (U_lo + U_up) when both windows have evidence and
    one is urgent (TTC < max_ttc), else None; below_fraction_1side also reads 1.0 (0.0) when only the lower (upper)
    window has evidence and it is urgent. ~0.5 for a wall facing the drone, -> 1 for terrain rising into the path.
    The bottom-centre HUD panel covers ~63 % of the lower window when the FOE is low; `vertical_frac_of_usable`
    counts the evidence fraction over the pixels it leaves visible (more terrain evidence, but more walls and clean
    flight read as terrain). Offline (17 impacts, last 2 s, TTC < 1.5 s frames), default / usable-only:
    >= 0.7 on 67 / 92 % of terrain frames (2 Pine mounds), 1 / 7-8 % of wall frames; +0.7 ms/frame. The default
    was selected by a closed-loop replay of the fast pilot's TTC policy (fewer false climbs).
It reads only causal frames and telemetry (attitude, velocity), never a course, route or map. No evidence is not
free space.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class Looming2Config:
    width: int = 160
    height: int = 90
    focal_320: float = 100.         # focal length in pixels at 320 px width
    tilt_deg: float = 30.           # camera uptilt
    contrast: str = 'clahe'         # 'none' | 'clahe'
    # (name, half-width, half-height as fractions of image width, model 'planar' | 'fronto'[, dx, dy offset from FOE])
    windows: tuple = (('big', .22, .176, 'planar'), ('c', .11, .088, 'planar'), ('i8', .08, .064, 'fronto'))
    # output TTC = max over groups of (min over the group's windows): every group must alarm
    groups: tuple = (('big', 'c'), ('i8',))
    # report-only windows (not in any group): where along the vertical the expansion lies; () disables them
    vertical_windows: tuple = (('up', .14, .08, 'planar', 0., -.12), ('lo', .14, .08, 'planar', 0., .12))
    vertical_frac_of_usable: bool = False   # True: evidence fraction over the pixels the HUD mask leaves visible
    vertical_min_usable: float = .2         # ... of which at least this share of the window must be visible
    vertical_min_count: int = 100           # textured pixels a vertical window needs
    min_texture: float = 4.         # Sobel magnitude/8 (grey levels per pixel) on the image used for evidence
    min_frac: float = .35           # fraction of a window's pixels that must be usable and textured
    min_forward: float = 1.         # m/s along the optical axis
    sig_k: float = 1.               # alarm uses rho - sig_k * se(rho): one standard error of evidence
    vision_derotation: bool = True
    hold: float = .5                # s to keep counting down the last TTC without evidence
    median: int = 3
    max_ttc: float = 10.

    def __post_init__(self):
        if self.width < 32 or self.height < 18 or self.focal_320 <= 0 or self.min_frac <= 0 or self.max_ttc <= 0:
            raise ValueError('Use positive looming parameters')
        if self.contrast not in ('none', 'clahe'):
            raise ValueError('contrast must be none or clahe')


def body_to_cam(tilt_deg):
    t = np.deg2rad(tilt_deg)
    fwd = np.array([np.cos(t), 0., np.sin(t)])
    right = np.array([0., -1., 0.])
    return np.stack([right, np.cross(fwd, right), fwd])


def quat_wxyz_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def hud_mask(rgb_small):
    """True where the pixel is usable: excludes white/grey HUD graphics and fixed HUD panels (same as v1)."""
    rgb = np.asarray(rgb_small, dtype=np.int16)
    h, w = rgb.shape[:2]
    lo, hi = rgb.min(-1), rgb.max(-1)
    white = (lo > 185) & (hi - lo < 60)
    white = cv2.dilate(white.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    usable = ~white
    usable[:int(.2 * h)] = False
    usable[int(.3 * h):int(.8 * h), int(.82 * w):] = False
    usable[int(.85 * h):, int(.41 * w):int(.59 * w)] = False
    usable[int(.87 * h):, :int(.1 * w)] = False
    return usable


def _wls(A, b, w):
    Aw = A * w[:, None]
    N = Aw.T @ A
    try:
        x = np.linalg.solve(N, Aw.T @ b)
        cov = np.linalg.inv(N)
    except np.linalg.LinAlgError:
        return None, None, None
    r = b - A @ x
    s2 = float((w * r * r).sum() / max(w.sum() - A.shape[1], 1.))
    return x, r, cov * s2


def _irls(A, b, iters=2):
    x, r, cov = _wls(A, b, np.ones(len(b)))
    for _ in range(iters):
        if x is None:
            break
        k = 1.345 * (1.4826 * np.median(np.abs(r)) + 1e-6)
        a = np.abs(r)
        x, r, cov = _wls(A, b, np.where(a <= k, 1., k / a))
    return x, r, cov


class LoomingEstimator2:
    """Stateful: feed consecutive frames with capture time, world-from-body attitude and world velocity."""

    def __init__(self, config: Looming2Config | None = None):
        self.config = c = config or Looming2Config()
        self.f = c.focal_320 * c.width / 320.
        self.cx, self.cy = c.width / 2., c.height / 2.
        ys, xs = np.mgrid[0:c.height, 0:c.width].astype(np.float64)
        self.X, self.Y = xs, ys
        self.xn, self.yn = (xs - self.cx) / self.f, (ys - self.cy) / self.f
        self.K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.]])
        self.Kinv = np.linalg.inv(self.K)
        self.M = body_to_cam(c.tilt_deg)
        self.clahe = cv2.createCLAHE(3., (max(2, round(c.width / 40)), max(2, round(c.width / 40 * 9 / 16)))) \
            if c.contrast == 'clahe' else None
        self.previous = None
        self.history = []            # (time, raw ttc) of recent frames for the median
        self.last_evidence = None    # (time, ttc)
        self.last = None

    def reset(self):
        self.previous, self.history, self.last_evidence, self.last = None, [], None, None

    def _prepare(self, image, usable):
        c = self.config
        img = np.asarray(image)
        if img.shape[:2] != (c.height, c.width):
            img = cv2.resize(img, (c.width, c.height), interpolation=cv2.INTER_AREA)
        if img.ndim == 3:
            if usable is None:
                usable = hud_mask(img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            if usable is None:
                usable = np.ones(gray.shape, bool)
        flow_img = self.clahe.apply(gray) if self.clahe is not None else gray
        return gray, flow_img, usable

    def update(self, image, capture_time, quat_wxyz, velocity_world, usable=None):
        """image: RGB (any size; resized) or grey at config size. Returns a result dict or None for the first frame."""
        c = self.config
        gray, flow_img, usable = self._prepare(image, usable)
        R = quat_wxyz_to_mat(quat_wxyz)
        v = np.asarray(velocity_world, float)
        prev, self.previous = self.previous, (flow_img, usable, float(capture_time), R, v)
        if prev is None:
            return None
        dt = float(capture_time) - prev[2]
        if not .005 < dt < .25:
            self.history = []
            return None
        raw, info = self._measure(prev, flow_img, usable, dt, R, v)
        t = float(capture_time)
        evidence = raw is not None
        if evidence:
            self.last_evidence = (t, raw)
        elif self.last_evidence is not None and t - self.last_evidence[0] <= c.hold and np.isfinite(self.last_evidence[1]):
            raw = max(.05, self.last_evidence[1] - (t - self.last_evidence[0]))
        else:
            raw = np.inf
        if self.history and t - self.history[-1][0] >= .3:
            self.history = []
        self.history = self.history[-(c.median - 1):] + [(t, raw)] if c.median > 1 else [(t, raw)]
        vals = [x for _, x in self.history] + [np.inf] * (c.median - len(self.history))
        ttc = float(np.median(vals))
        speed = float(np.linalg.norm(v))
        result = dict(time=t, dt=dt, ttc=ttc if np.isfinite(ttc) else None, ttc_raw=raw if np.isfinite(raw) else None,
                      evidence=evidence, distance=ttc * speed if np.isfinite(ttc) else None, **info)
        self.last = result
        return result

    def _measure(self, prev, flow_img, usable, dt, R, v):
        c = self.config
        Rp, vp = prev[3], prev[4]
        vw = .5 * (vp + v)
        Vc = self.M @ (Rp.T @ vw)                  # velocity in the previous camera frame
        fwd = float(Vc[2])
        info = dict(forward=fwd, reason=None)
        if fwd < c.min_forward:
            info['reason'] = 'slow'
            return None, info
        fx, fy = self.cx + self.f * Vc[0] / fwd, self.cy + self.f * Vc[1] / fwd
        info['foe'] = (float(fx), float(fy))
        flow = cv2.calcOpticalFlowFarneback(prev[0], flow_img, None, .5, 2 if c.width <= 160 else 3,
                                            9 if c.width <= 160 else 13, 3, 5, 1.1, 0)
        # rotation from the attitude change (exact homography, points at infinity)
        H = self.K @ self.M @ R.T @ Rp @ self.M.T @ self.Kinv
        X, Y = self.X, self.Y
        d = H[2, 0] * X + H[2, 1] * Y + H[2, 2]
        u = flow[..., 0] - ((H[0, 0] * X + H[0, 1] * Y + H[0, 2]) / d - X)
        vv = flow[..., 1] - ((H[1, 0] * X + H[1, 1] * Y + H[1, 2]) / d - Y)
        # evidence: texture of the previous frame's flow image (CLAHE output when contrast='clahe')
        gx = cv2.Sobel(prev[0], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(prev[0], cv2.CV_32F, 0, 1)
        base = prev[1] & usable & (np.hypot(gx, gy) / 8. > c.min_texture)
        qx, qy = X - fx, Y - fy
        if c.vision_derotation:
            w = self._rotation_correction(u, vv, base, qx, qy)
            u = u - self.f * (self.xn * self.yn * w[0] - (1 + self.xn ** 2) * w[1] + self.yn * w[2])
            vv = vv - self.f * ((1 + self.yn ** 2) * w[0] - self.xn * self.yn * w[1] - self.xn * w[2])
            info['rotation_correction_deg'] = float(np.degrees(np.linalg.norm(w)))
        Tz = fwd * dt
        windows = {}
        for spec in c.windows:
            windows[spec[0]] = self._fit_window(spec, base, u, vv, qx, qy, Tz, fwd, fx, fy)
        valid = prev[1] & usable if c.vertical_frac_of_usable else None
        for spec in c.vertical_windows:
            windows[spec[0]] = self._fit_window(spec, base, u, vv, qx, qy, Tz, fwd, fx, fy, valid)
        if c.vertical_windows:
            info.update(self._vertical(windows))
        info['windows'] = windows
        best = -np.inf
        for group in c.groups:
            vals = [np.inf if windows[w]['ttc'] is None else windows[w]['ttc'] for w in group if 'rho_path' in windows.get(w, {})]
            if not vals:                            # a group without evidence: no measurement this frame
                info['reason'] = 'low_texture'
                return None, info
            best = max(best, min(vals))
        return best, info

    def _fit_window(self, spec, base, u, vv, qx, qy, Tz, fwd, fx, fy, valid=None):
        c = self.config
        name, hx, hy, model = spec[:4]
        dx, dy = (spec[4], spec[5]) if len(spec) > 5 else (0., 0.)   # optional offset from the FOE (fraction of width)
        wx, wy = fx + dx * c.width, fy + dy * c.width
        x0, x1 = int(max(0, round(wx - hx * c.width))), int(min(c.width, round(wx + hx * c.width)))
        y0, y1 = int(max(0, round(wy - hy * c.width))), int(min(c.height, round(wy + hy * c.width)))
        full = 4 * hx * hy * c.width ** 2
        area = max(0, x1 - x0) * max(0, y1 - y0)
        if area < .15 * full or x1 - x0 < 6 or y1 - y0 < 4:
            return dict(reason='outside')
        m = base[y0:y1, x0:x1]
        frac = float(m.sum()) / area
        if valid is not None:                   # vertical windows: fraction of the pixels the HUD leaves visible
            visible = float(valid[y0:y1, x0:x1].sum())
            if visible < c.vertical_min_usable * area:
                return dict(reason='hud_masked', frac=frac)
            frac = float(m.sum()) / visible
            if frac < c.min_frac or m.sum() < c.vertical_min_count:
                return dict(reason='low_texture', frac=frac)
        elif frac < c.min_frac or m.sum() < 25:
            return dict(reason='low_texture', frac=frac)
        a, b = qx[y0:y1, x0:x1][m], qy[y0:y1, x0:x1][m]
        n = len(a)
        one, zero = np.ones(n), np.zeros(n)
        if model == 'planar':
            A = np.r_[np.stack((one, zero, Tz * a, Tz * a * a / self.f, Tz * a * b / self.f), 1),
                      np.stack((zero, one, Tz * b, Tz * b * a / self.f, Tz * b * b / self.f), 1)]
        else:                                   # fronto-parallel: constant inverse depth in the window
            A = np.r_[np.stack((one, zero, Tz * a), 1), np.stack((zero, one, Tz * b), 1)]
        x, _, cov = _irls(A, np.r_[u[y0:y1, x0:x1][m], vv[y0:y1, x0:x1][m]])
        if x is None:
            return dict(reason='singular', frac=frac)
        c0, se = float(x[2]), float(np.sqrt(max(cov[2, 2], 0.)))
        cc = c0 - c.sig_k * se
        ttc = min(c.max_ttc, 1. / (cc * fwd)) if cc > 1. / (c.max_ttc * fwd) else np.inf
        return dict(frac=frac, rho_path=c0, rho_se=se, ttc=ttc if np.isfinite(ttc) else None, urgency=max(0., cc) * fwd)

    def _vertical(self, windows):
        """Where the expansion lies: TTC of the surfaces fitted above/below the path and their urgency share."""
        up, lo = windows.get('up', {}), windows.get('lo', {})
        eu, el = 'rho_path' in up, 'rho_path' in lo
        urgent = 1. / self.config.max_ttc
        uu, ul = (up['urgency'] if eu else 0.), (lo['urgency'] if el else 0.)
        strict = one_side = None
        if eu and el and max(uu, ul) > urgent:
            strict = one_side = ul / (ul + uu)
        elif el and not eu and ul > urgent:
            one_side = 1.
        elif eu and not el and uu > urgent:
            one_side = 0.
        return dict(ttc_upper=up.get('ttc') if eu else None, ttc_lower=lo.get('ttc') if el else None,
                    evidence_upper=eu, evidence_lower=el, urgency_upper=uu if eu else None,
                    urgency_lower=ul if el else None, below_fraction=strict, below_fraction_1side=one_side)

    def _rotation_correction(self, u, v, m, qx, qy):
        r = np.hypot(qx, qy)
        k = m & (r > .02 * self.config.width)
        if k.sum() < 100:
            return np.zeros(3)
        x, y = self.xn[k], self.yn[k]
        nx, ny = qy[k] / r[k], -qx[k] / r[k]
        A = self.f * np.stack((x * y * nx + (1 + y * y) * ny, -(1 + x * x) * nx - x * y * ny, y * nx - x * ny), 1)
        w, _, _ = _irls(A, u[k] * nx + v[k] * ny)
        return np.zeros(3) if w is None else w

    def metadata(self):
        return dict(cue='time-to-contact along the velocity ray from a planar inverse-depth fit of de-rotated flow '
                        'around the focus of expansion (looming v2)',
                    parameters=asdict(self.config), establishes_free_space=False,
                    below_fraction='U_lo/(U_lo+U_up), U = max(0, c0 - se) * forward from planar windows 0.12 W '
                                   'above/below the FOE; None unless both have evidence and one is urgent '
                                   '(report only: the alarm TTC does not use it)',
                    limitations='Needs texture; no evidence is not free space; planar fit per window; attitude and '
                                'velocity from telemetry; HUD mask tuned to the Liftoff HUD')
