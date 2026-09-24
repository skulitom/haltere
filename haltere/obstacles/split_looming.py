"""Baseline B1: split-field looming side cue (lateral study, candidate A), vendored for the obstacle bench.

OFFLINE BASELINE ONLY (haltere.obstacles.evaluate B1); not part of the flight stack. This is a verbatim copy
of the lateral study's ``split_looming.py`` (sha256 31e988eaa1ae6da5..., scratchpad
lateral/cand-split-looming, 2026-09-24) without the unused ``corridor_threat`` experiment, so the bench
does not depend on a session scratchpad. The study's selected parameter set (``config_selected.json``,
sha256 62b111f9770428f6...) is ``SELECTED_CONFIG`` below and is recorded in
configs/obstacles/thresholds.json (baselines.B1). Study result with it: correct free side held >= 1 s on
1 of 11 lateral impacts, no wrong side, 0.31 clean episodes/min.

Original docstring:

Split-field looming: which side of the flight path is closing in (a causal lateral obstacle cue).

Candidate A of the lateral-obstacle study. Built from copies of haltere/vision/looming2.py internals (the repo is not
edited): CLAHE -> Farneback flow between consecutive frames -> exact attitude-delta de-rotation from the telemetry
quaternions + a small visual rotation correction fitted on the flow component perpendicular to the FOE ray -> the
focus of expansion (FOE) from the telemetry velocity.

New here: the de-rotated flow in a window around the FOE is fitted with ONE shared offset (u0, v0) and a SEPARATE
inverse-depth field on each side of the flight path,
    flow(q) = (u0, v0) + Tz * rho_side(q) * q,      rho_side(q) = c0 + c1 * s/f + c2 * t/f  ('planar')
                                                    rho_side(q) = c0                        ('fronto')
where q = pixel - FOE, and (s, t) are q expressed on roll-compensated axes: s along the image direction of the
world-horizontal 'left of the velocity', t along the image direction of 'up, perpendicular to the velocity'. The left
half-window is s > 0, the right half-window s < 0. c0 is where that side's fitted surface crosses the flight path
(1/m); a ground plane or a side wall parallel to the path gives c0 ~ 0, a pillar/boulder/trunk face at the path edge
gives c0 = 1/distance. Per side, urgency U = max(0, c0 - k*se) * forward speed (1/s) = inverse time to contact.

Decision (per frame, then temporal filtering): when the more urgent side exceeds 1/ttc_on and dominates the other side
(U_min <= ratio * U_max), indicate the OTHER side (steer away from the closing side). Both sides urgent and similar ->
'centre' (no side; brake territory, left to the path TTC governor). Hysteresis: once active, stay on the same side
while its trigger side stays above 1/ttc_off; a side flip needs `switch_frames` consecutive opposite decisions; short
evidence gaps hold the last decision for `hold` s.

Fly analogy: the fly's escape direction is set by where on the retina a looming stimulus expands (LPLC2/giant-fibre
pathways compare expansion across the visual field); here the comparison is between the two halves of the frontal
field around the FOE, after removing self-rotation (haltere/ocellar-style rotation compensation).

Causal: reads only the current and previous frame, their capture times, the attitude quaternions and the telemetry
velocity. No course, route or map. No evidence is not free space: a side without texture gives no side decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ----------------------------------------------------------------------------- copied from haltere/vision/looming2.py
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
    """True where the pixel is usable: excludes white/grey HUD graphics and fixed HUD panels (same as looming2)."""
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
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitLoomingConfig:
    width: int = 240
    height: int = 135
    focal_320: float = 100.
    tilt_deg: float = 30.
    contrast: str = 'clahe'
    min_texture: float = 4.          # Sobel magnitude/8 on the flow image (looming2 default)
    vision_derotation: bool = True
    min_forward: float = 1.          # m/s along the optical axis
    baseline: int = 1                # flow from the frame `baseline` captures back (longer displacement, same cost)
    # geometry of the split window (fractions of the image width), roll-compensated axes around the FOE
    model: str = 'planar'            # 'planar' | 'fronto'
    half_width: float = .16          # lateral extent of each half-window from the FOE
    up: float = .12                  # extent above the FOE
    down: float = .06                # extent below the FOE
    gap: float = 0.                  # dead zone |s| < gap around the split line
    min_frac: float = .25            # textured usable fraction of a half-window's in-image area
    min_inside: float = .3           # in-image share of a half-window's nominal area
    min_count: int = 60              # textured pixels per half-window
    sig_k: float = 1.                # urgency uses c0 - sig_k * se
    exact_scale: bool = True         # rho = k / (1 + k Tz): large-displacement correction
    # decision
    ttc_on: float = 1.5              # s: the closing side's TTC must be below this to act
    ttc_off: float = 2.0             # s: release when the trigger side's (filtered) TTC rises above this
    ratio: float = .5                # dominance: U_other <= ratio * U_trigger
    median: int = 3                  # median over the last N frames of each side's urgency (gap >= .3 s restarts)
    switch_frames: int = 2           # consecutive opposite decisions needed to flip the active side
    hold: float = .3                 # s to keep the last decision through frames without a decision (evidence gaps)
    both_evidence: bool = True       # a side decision needs evidence on both sides
    tau: float = .15                 # s: leaky integration of each side's urgency after the median (0 = off)
    onset_frames: int = 2            # consecutive raw side decisions needed to activate
    u_clip: float = 8.               # 1/s: clip single-frame urgencies before filtering

    def __post_init__(self):
        if self.model not in ('planar', 'fronto'):
            raise ValueError('model must be planar or fronto')
        if self.contrast not in ('none', 'clahe'):
            raise ValueError('contrast must be none or clahe')
        if not 0 < self.ttc_on <= self.ttc_off:
            raise ValueError('need 0 < ttc_on <= ttc_off')


class FlowFront:
    """Shared front end: CLAHE, Farneback flow, de-rotation, FOE and roll-compensated lateral axes (looming2 copy)."""

    def __init__(self, width=240, height=135, focal_320=100., tilt_deg=30., contrast='clahe', min_texture=4.,
                 vision_derotation=True, min_forward=1., baseline=1, max_span=.4):
        self.width, self.height = width, height
        self.baseline, self.max_span = int(baseline), float(max_span)
        self.min_texture, self.vision_derotation, self.min_forward = min_texture, vision_derotation, min_forward
        self.f = focal_320 * width / 320.
        self.cx, self.cy = width / 2., height / 2.
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
        self.X, self.Y = xs, ys
        self.xn, self.yn = (xs - self.cx) / self.f, (ys - self.cy) / self.f
        self.K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.]])
        self.Kinv = np.linalg.inv(self.K)
        self.M = body_to_cam(tilt_deg)
        self.clahe = cv2.createCLAHE(3., (max(2, round(width / 40)), max(2, round(width / 40 * 9 / 16)))) \
            if contrast == 'clahe' else None
        self.buffer = []            # last `baseline` frames: (flow_img, usable, time, R, v)

    def reset(self):
        self.buffer = []

    def _prepare(self, image, usable):
        img = np.asarray(image)
        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if img.ndim == 3:
            if usable is None:
                usable = hud_mask(img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            if usable is None:
                usable = np.ones(gray.shape, bool)
        flow_img = self.clahe.apply(gray) if self.clahe is not None else gray
        return flow_img, usable

    def step(self, image, capture_time, quat_wxyz, velocity_world, usable=None):
        """Returns None (first frame / bad dt) or a dict with the de-rotated flow of the previous frame's pixels."""
        flow_img, usable = self._prepare(image, usable)
        R = quat_wxyz_to_mat(quat_wxyz)
        v = np.asarray(velocity_world, float)
        cur = (flow_img, usable, float(capture_time), R, v)
        buf = self.buffer
        if buf and not .005 < float(capture_time) - buf[-1][2] < .25:
            self.buffer = [cur]                       # gap or clock jump: restart
            return dict(time=float(capture_time), dt=float(capture_time) - buf[-1][2], reason='dt')
        # reference frame: `baseline` frames back (longer displacement), else the oldest usable one
        prev = None
        for k in range(min(self.baseline, len(buf)), 0, -1):
            if float(capture_time) - buf[-k][2] <= self.max_span:
                prev = buf[-k]
                vs = [b[4] for b in buf[-k:]] + [v]
                break
        self.buffer = (buf + [cur])[-self.baseline:]
        if prev is None:
            return None
        dt = float(capture_time) - prev[2]
        Rp = prev[3]
        vw = np.mean(vs, axis=0)
        Vc = self.M @ (Rp.T @ vw)
        fwd = float(Vc[2])
        out = dict(time=float(capture_time), dt=dt, forward=fwd, speed=float(np.linalg.norm(vw)), reason=None)
        if fwd < self.min_forward:
            out['reason'] = 'slow'
            return out
        fx, fy = self.cx + self.f * Vc[0] / fwd, self.cy + self.f * Vc[1] / fwd
        # roll-compensated axes at the FOE: image directions of world-left and world-up perpendicular to the velocity
        vhat = vw / max(np.linalg.norm(vw), 1e-6)
        h = np.cross([0., 0., 1.], vhat)
        if np.linalg.norm(h) < .2:                  # near-vertical flight: fall back to the body left axis
            h = Rp[:, 1] - (Rp[:, 1] @ vhat) * vhat
        h /= np.linalg.norm(h)
        upv = np.cross(vhat, h)
        Hc, Uc = self.M @ (Rp.T @ h), self.M @ (Rp.T @ upv)
        eL = Hc[:2] * fwd - Vc[:2] * Hc[2]
        eL /= max(np.linalg.norm(eL), 1e-9)
        eU = Uc[:2] * fwd - Vc[:2] * Uc[2]
        eU = eU - (eU @ eL) * eL                    # orthogonalise (image axes stay orthonormal)
        eU /= max(np.linalg.norm(eU), 1e-9)
        flow = cv2.calcOpticalFlowFarneback(prev[0], flow_img, None, .5, 2 if self.width <= 160 else 3,
                                            9 if self.width <= 160 else 13, 3, 5, 1.1, 0)
        H = self.K @ self.M @ R.T @ Rp @ self.M.T @ self.Kinv
        X, Y = self.X, self.Y
        d = H[2, 0] * X + H[2, 1] * Y + H[2, 2]
        u = flow[..., 0] - ((H[0, 0] * X + H[0, 1] * Y + H[0, 2]) / d - X)
        vv = flow[..., 1] - ((H[1, 0] * X + H[1, 1] * Y + H[1, 2]) / d - Y)
        gx = cv2.Sobel(prev[0], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(prev[0], cv2.CV_32F, 0, 1)
        base = prev[1] & usable & (np.hypot(gx, gy) / 8. > self.min_texture)
        qx, qy = X - fx, Y - fy
        if self.vision_derotation:
            w = self._rotation_correction(u, vv, base, qx, qy)
            u = u - self.f * (self.xn * self.yn * w[0] - (1 + self.xn ** 2) * w[1] + self.yn * w[2])
            vv = vv - self.f * ((1 + self.yn ** 2) * w[0] - self.xn * self.yn * w[1] - self.xn * w[2])
            out['rotation_correction_deg'] = float(np.degrees(np.linalg.norm(w)))
        out.update(foe=(float(fx), float(fy)), eL=eL, eU=eU, Tz=fwd * dt, u=u, v=vv, base=base, qx=qx, qy=qy,
                   visible=prev[1] & usable,
                   roll_img_deg=float(np.degrees(np.arctan2(-eL[1], -eL[0]))))
        return out

    def _rotation_correction(self, u, v, m, qx, qy):
        r = np.hypot(qx, qy)
        k = m & (r > .02 * self.width)
        if k.sum() < 100:
            return np.zeros(3)
        x, y = self.xn[k], self.yn[k]
        nx, ny = qy[k] / r[k], -qx[k] / r[k]
        A = self.f * np.stack((x * y * nx + (1 + y * y) * ny, -(1 + x * x) * nx - x * y * ny, y * nx - x * ny), 1)
        w, _, _ = _irls(A, u[k] * nx + v[k] * ny)
        return np.zeros(3) if w is None else w


def fit_split(fr, f, width, height, model='planar', half_width=.16, up=.12, down=.06, gap=0., min_frac=.25,
              min_inside=.3, min_count=60, sig_k=1., exact_scale=True, quantiles=False, qmin=5.):
    """Joint robust fit: shared offset + a separate inverse-depth field per side. Returns dict(L=..., R=..., ...)."""
    W = width
    fx, fy = fr['foe']
    eL, eU = fr['eL'], fr['eU']
    A_, U_, D_, G_ = half_width * W, up * W, down * W, gap * W
    r = np.hypot(A_, max(U_, D_)) + 1
    x0, x1 = int(max(0, np.floor(fx - r))), int(min(width, np.ceil(fx + r) + 1))
    y0, y1 = int(max(0, np.floor(fy - r))), int(min(height, np.ceil(fy + r) + 1))
    nominal = A_ * (U_ + D_)
    res = dict(L=dict(reason='outside'), R=dict(reason='outside'), foe=(fx, fy))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return res
    qx, qy = fr['qx'][y0:y1, x0:x1], fr['qy'][y0:y1, x0:x1]
    s = qx * eL[0] + qy * eL[1]
    t = qx * eU[0] + qy * eU[1]
    inwin = (np.abs(s) <= A_) & (t <= U_) & (t >= -D_) & (np.abs(s) >= G_)
    base = fr['base'][y0:y1, x0:x1] & inwin
    sides = {}
    for name, sel in (('L', s > 0), ('R', s < 0)):
        inside = float((inwin & sel).sum())
        m = base & sel
        n = int(m.sum())
        frac = n / inside if inside > 0 else 0.
        if inside < min_inside * nominal:
            sides[name] = dict(reason='outside', inside=inside / nominal, frac=frac, n=n)
        elif frac < min_frac or n < min_count:
            sides[name] = dict(reason='low_texture', inside=inside / nominal, frac=frac, n=n)
        else:
            sides[name] = dict(reason=None, inside=inside / nominal, frac=frac, n=n, mask=m)
    ok = [k for k in ('L', 'R') if sides[k]['reason'] is None]
    for k in ('L', 'R'):
        sides[k].pop('mask', None) if k not in ok else None
    if not ok:
        res.update(L=sides['L'], R=sides['R'])
        return res
    Tz = fr['Tz']
    cols_u, cols_v, bu, bv = [], [], [], []
    rows = []
    for k in ok:
        m = sides[k]['mask']
        rows.append((k, m))
    npx = sum(int(m.sum()) for _, m in rows)
    nper = 3 if model == 'planar' else 1
    ncol = 2 + nper * len(rows)
    A = np.zeros((2 * npx, ncol))
    b = np.zeros(2 * npx)
    i = 0
    for j, (k, m) in enumerate(rows):
        a, c = qx[m], qy[m]
        ss, tt = s[m], t[m]
        n = len(a)
        A[i:i + n, 0] = 1.
        A[npx + i:npx + i + n, 1] = 1.
        col = 2 + nper * j
        A[i:i + n, col] = Tz * a
        A[npx + i:npx + i + n, col] = Tz * c
        if model == 'planar':
            A[i:i + n, col + 1] = Tz * a * ss / f
            A[npx + i:npx + i + n, col + 1] = Tz * c * ss / f
            A[i:i + n, col + 2] = Tz * a * tt / f
            A[npx + i:npx + i + n, col + 2] = Tz * c * tt / f
        b[i:i + n] = fr['u'][y0:y1, x0:x1][m]
        b[npx + i:npx + i + n] = fr['v'][y0:y1, x0:x1][m]
        i += n
    x, _, cov = _irls(A, b)
    fwd = fr['forward']
    for j, (k, m) in enumerate(rows):
        d = sides[k]
        d.pop('mask', None)
        if x is None:
            d['reason'] = 'singular'
            continue
        col = 2 + nper * j
        c0, se = float(x[col]), float(np.sqrt(max(cov[col, col], 0.)))
        if exact_scale:                               # flow = q * rho Tz / (1 - rho Tz)  ->  rho = k / (1 + k Tz)
            den = 1. + max(c0, -.5 / Tz) * Tz
            c0, se = c0 / den, se / den ** 2
        d.update(c0=c0, se=se, U=max(0., c0 - sig_k * se) * fwd)
        if model == 'planar':
            d.update(c1=float(x[col + 1]), c2=float(x[col + 2]))
    if quantiles and x is not None:                 # per-pixel inverse depth (radial flow) after removing the offset
        for j, (k, m) in enumerate(rows):
            a, c = qx[m], qy[m]
            r2 = a * a + c * c
            keep = r2 >= qmin * qmin
            if keep.sum() < 10:
                continue
            uu = fr['u'][y0:y1, x0:x1][m][keep] - x[0]
            vv = fr['v'][y0:y1, x0:x1][m][keep] - x[1]
            kk = (uu * a[keep] + vv * c[keep]) / r2[keep] / Tz
            rho = kk / (1. + np.maximum(kk, -.5 / Tz) * Tz) if exact_scale else kk
            q = np.percentile(rho, (50, 75, 90))
            sides[k].update(q50=float(q[0]), q75=float(q[1]), q90=float(q[2]))
    res.update(L=sides['L'], R=sides['R'], offset=None if x is None else (float(x[0]), float(x[1])))
    return res


class SideDecision:
    """Temporal filter + decision on per-side urgencies. update(time, UL, UR) with None for a side without evidence.

    Filter per side: median of the last `median` frames (spike rejection), then a leaky integrator with time constant
    `tau` s (evidence accumulation; 0 = off). A gap >= .3 s between frames restarts both. Activation needs
    `onset_frames` consecutive frames with the same raw side decision."""

    def __init__(self, ttc_on=1.5, ttc_off=2.0, ratio=.5, median=3, switch_frames=2, hold=.3, both_evidence=True,
                 tau=0., onset_frames=1, u_clip=8.):
        self.ttc_on, self.ttc_off, self.ratio = ttc_on, ttc_off, ratio
        self.median, self.switch_frames, self.hold, self.both_evidence = median, switch_frames, hold, both_evidence
        self.tau, self.onset_frames, self.u_clip = tau, onset_frames, u_clip
        self.reset()

    def reset(self):
        self.hist = []              # (time, UL, UR)
        self.ema = [None, None]
        self.last_t = None
        self.side = None            # 'left' | 'right' | None: direction to steer (the free side)
        self.trigger = None         # 'L' | 'R': the closing side
        self.last_decision_time = -np.inf
        self.pending, self.pending_n = None, 0
        self.onset, self.onset_n = None, 0

    def _filtered(self, t, UL, UR):
        if self.last_t is not None and t - self.last_t >= .3:
            self.hist, self.ema = [], [None, None]
        dt = 0. if self.last_t is None else max(0., t - self.last_t)
        self.last_t = t
        clip = lambda x: None if x is None else min(float(x), self.u_clip)
        self.hist = (self.hist + [(t, clip(UL), clip(UR))])[-self.median:]
        out = []
        for i in (1, 2):
            vals = [h[i] for h in self.hist if h[i] is not None]
            med = float(np.median(vals + [0.] * (self.median - len(self.hist)))) if vals else None
            if self.tau > 0:
                if med is None:
                    pass                              # no evidence this frame: keep the integrator as it is
                elif self.ema[i - 1] is None:
                    self.ema[i - 1] = med
                else:
                    a = 1. - np.exp(-dt / self.tau)
                    self.ema[i - 1] += a * (med - self.ema[i - 1])
                out.append(self.ema[i - 1] if med is not None else None)
            else:
                out.append(med)
        return out

    def update(self, t, UL, UR):
        fUL, fUR = self._filtered(t, UL, UR)
        on, off = 1. / self.ttc_on, 1. / self.ttc_off
        raw, trig = None, None
        have = (fUL is not None and fUR is not None) if self.both_evidence else (fUL is not None or fUR is not None)
        if have:
            a, b = (fUL or 0.), (fUR or 0.)
            hi, lo = max(a, b), min(a, b)
            if hi >= on and lo <= self.ratio * hi:
                trig = 'L' if a > b else 'R'
                raw = 'right' if trig == 'L' else 'left'
            elif hi >= on:
                raw = 'centre'
        if raw in ('left', 'right'):
            self.onset_n = self.onset_n + 1 if self.onset == raw else 1
            self.onset = raw
        else:
            self.onset, self.onset_n = None, 0
        # hysteresis on the active side
        if self.side is not None:
            tv = fUL if self.trigger == 'L' else fUR
            still = tv is not None and tv >= off
            if raw in ('left', 'right') and raw != self.side:
                self.pending_n = self.pending_n + 1 if self.pending == raw else 1
                self.pending = raw
                if self.pending_n >= self.switch_frames:
                    self.side, self.trigger = raw, trig
                    self.pending, self.pending_n = None, 0
                self.last_decision_time = t
            elif raw == self.side:
                self.pending, self.pending_n = None, 0
                self.last_decision_time = t
            elif still:
                self.last_decision_time = t
            elif tv is None and t - self.last_decision_time <= self.hold:
                pass                                   # evidence gap: hold
            else:
                self.side, self.trigger = None, None
        elif raw in ('left', 'right') and self.onset_n >= self.onset_frames:
            self.side, self.trigger = raw, trig
            self.last_decision_time = t
            self.pending, self.pending_n = None, 0
        state = self.side if self.side is not None else ('centre' if raw == 'centre' else None)
        urg = max(fUL or 0., fUR or 0.)
        return dict(side=self.side, state=state, raw=raw, UL=fUL, UR=fUR, urgency=urg,
                    ttc_min=1. / urg if urg > 0 else None,
                    lateral_fraction=(fUL / (fUL + fUR)) if (fUL is not None and fUR is not None and fUL + fUR > 0)
                    else None)


class SplitLoomingEstimator:
    """Streaming cue: update(frame, time, quat_wxyz, velocity_world[, usable]) -> dict or None (first frame).

    Output: side ('left'|'right'|None: the side to steer toward, i.e. away from the closing side), state (side,
    'centre' when both sides close in alike, or None), urgency (1/s, the larger side's inverse TTC, filtered), UL/UR,
    ttc_min, lateral_fraction (UL/(UL+UR)), evidence per side and the FOE."""

    def __init__(self, config: SplitLoomingConfig | None = None):
        self.config = c = config or SplitLoomingConfig()
        self.front = FlowFront(c.width, c.height, c.focal_320, c.tilt_deg, c.contrast, c.min_texture,
                               c.vision_derotation, c.min_forward, c.baseline)
        self.decision = SideDecision(c.ttc_on, c.ttc_off, c.ratio, c.median, c.switch_frames, c.hold, c.both_evidence,
                                     c.tau, c.onset_frames, c.u_clip)
        self.last = None

    def reset(self):
        self.front.reset()
        self.decision.reset()
        self.last = None

    def update(self, image, capture_time, quat_wxyz, velocity_world, usable=None):
        c = self.config
        fr = self.front.step(image, capture_time, quat_wxyz, velocity_world, usable)
        if fr is None:
            return None
        UL = UR = None
        ev = dict(L=None, R=None)
        if fr.get('reason') is None:
            fit = fit_split(fr, self.front.f, c.width, c.height, c.model, c.half_width, c.up, c.down, c.gap,
                            c.min_frac, c.min_inside, c.min_count, c.sig_k, c.exact_scale)
            UL, UR = fit['L'].get('U'), fit['R'].get('U')
            ev = dict(L=fit['L'], R=fit['R'])
        if fr.get('reason') == 'dt':
            self.decision.hist, self.decision.ema = [], [None, None]
        out = self.decision.update(fr['time'], UL, UR)
        out.update(time=fr['time'], foe=fr.get('foe'), forward=fr.get('forward'), reason=fr.get('reason'),
                   raw_UL=UL, raw_UR=UR, evidence_left=ev['L'], evidence_right=ev['R'])
        self.last = out
        return out

    def metadata(self):
        return dict(cue='split-field looming: per-side inverse TTC from a joint planar inverse-depth fit of de-rotated '
                        'flow left/right of the FOE (roll-compensated), steer away from the dominant closing side',
                    parameters=asdict(self.config), establishes_free_space=False,
                    limitations='needs texture on both sides; no evidence is not free space; symmetric threats give '
                                "'centre' (no side); attitude and velocity from telemetry")


SOURCE_SHA256 = '31e988eaa1ae6da5b65a784b90f23eb01770622dee7ca480a78e0cedd527f661'
SELECTED_CONFIG_SHA256 = '62b111f9770428f6225b293df281492e285a419c143f10a163d35a102a134ced'
# config_selected.json of the lateral study
SELECTED_CONFIG = dict(width=240, height=135, focal_320=100.0, tilt_deg=30.0, contrast='clahe', min_texture=4.0,
                       vision_derotation=True, min_forward=1.0, baseline=3, model='planar', half_width=0.16, up=0.12,
                       down=0.06, gap=0.0, min_frac=0.25, min_inside=0.3, min_count=60, sig_k=1.0, exact_scale=True,
                       ttc_on=1.0, ttc_off=1.3333, ratio=0.33, median=3, switch_frames=2, hold=0.3,
                       both_evidence=True, tau=0.3, onset_frames=3, u_clip=8.0)
SIDE_CODE = {None: 0, 'left': 1, 'right': 2, 'centre': 3}
