"""The by-sight pilot "Rabbit+": a virtual lead vehicle flies a smooth course through the gates the detector sees.

``TelemetryPilot.sight == 'rabbit'`` hands ``vision_goal`` to ``SightPilot.goal`` and the yaw stick to
``SightPilot.sight_yaw``; ``'legacy'`` keeps the older remembered-gate pilot.

Perception. Every detector frame is used once, placed with the pose at its screen grab (``TelemetryPilot.pose_at``).
A sighting is a world ray (the bearing is the accurate part) plus a range from the apparent width (the rough part,
bias-corrected by ``range_corr``); it updates one of several per-arch Kalman filters with a covariance that is tight
across the ray and loose along it (twice as loose when the arch is cropped by the image edge). Arches are associated
in metres along and across the ray and in log-range, so two arches lined up on one bearing stay two tracks. A track
confirms after three sightings over 0.2 s at a plausible height (hay bales and shadows on the ground do not); a
confirmed estimate that the camera should see but does not for 2 s is a ghost. Passed arches absorb their own
sightings from behind.

Guidance. The target is the nearest confirmed arch ahead. Its approach axis runs along the course (from the last gate,
turned toward the bisector with the next gate when that one is known) and pivots onto the exact bearing close up.
A rabbit, a point with bounded speed, acceleration, curvature and curvature rate, steers onto that axis with
line-of-sight guidance and flies through the gate, on for a few metres and, without a gate, around a search circle;
it never stops, and it waits for the drone (it keeps about 3 m ahead along its own trail). The brain's goal is the
rabbit (its horizontal part clipped to 5 m, its height to +-1.2 m), pulled onto the drone-to-gate bearing on the last
metres. The nose follows the rabbit's heading. Nothing that perception decides can make the goal jump: a new target
or a changed estimate only changes the rabbit's desired curvature and speed.

Speed. The brain holds a sensed cruise speed (about ``flow_ref`` m/s in the game), so the optic-flow/airflow gain is
scheduled as flow_ref / v_nom with a slow integral trim on the measured speed, between ``flow_min`` and ``flow_max``
(the fly command's --flow-gain).

Everything is in the pilot's world frame (x forward at the reset, y left, z up, metres from the reset point);
angles in the log are degrees. ``step`` never raises: an internal error is reported once and the goal holds the last
carrot while the yaw stick returns to centre.
"""
from __future__ import annotations

import math
import sys
import traceback
from bisect import bisect_left
from dataclasses import dataclass, fields
from operator import itemgetter

import numpy as np

CENTRE_UP_M = 1.5                  # the arch's visual centre above the passage point (gates.CENTRE_UP_M)
DEG = math.pi / 180.0
TWO_PI = 2.0 * math.pi
# true range / range from the apparent width, by the latter: measured on 2036 GateNet (gatenet8) detections of runs 17-21
# matched to the arch within 6 deg of their ray (median per bin; below 4.5 m extrapolated, beyond 30 m assumed)
RANGE_CORR = ((3.5, 0.75), (6.3, 0.81), (9.2, 0.88), (12.7, 0.99), (18.2, 1.08), (23.4, 1.08), (30.0, 1.10), (45.0, 1.0))
# the prototype's table (fitted to the rehearsal's synthetic detector; 5-10 % shorter than GateNet's at 6-18 m)
RANGE_CORR_SPEC = ((3.5, 0.58), (5.6, 0.72), (8.9, 0.84), (12.8, 0.90), (18.5, 1.0), (25.0, 1.15), (35.0, 1.0), (45.0, 0.90))

# numeric log columns (fly --log and the rehearsal), in this order; angles in degrees
LOG_COLUMNS = ['det_u', 'det_v', 'det_t',
               'rb_x', 'rb_y', 'rb_z', 'rb_psi', 'rb_kappa', 'rb_v', 'rb_vnom',
               'look', 'yaw_ref', 'sight_yaw',
               'tgt_id', 'tgt_x', 'tgt_y', 'tgt_z', 'tgt_sdlat', 'tgt_hits', 'tgt_age', 'axis_deg', 'next_id',
               'n_conf', 'n_tent', 'mode', 'n_passes', 'pass_kind',
               'flow_gain', 'rej_elev', 'rej_stale', 'absorbed', 'low', 'reseeds',
               'ghosts', 'unpasses', 'behind', 'goal_clips', 'sight_errors']
PASS_KIND = {'cross': 1, 'travel': 2, 'beside': 3, 'ghost': 4, 'unpass': 5}
MODE_NAMES = {0: 'ground', 1: 'cruise', 2: 'target', 3: 'search'}


def wrap(a: float) -> float:
    return (a + math.pi) % TWO_PI - math.pi


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class SightParams:
    """Every tunable of the rabbit pilot (stage A defaults of the merged spec). Speeds m/s, angles degrees."""
    # --- perception
    p_min: float = 0.5
    stale_s: float = 0.35                 # older detections (grab time) are not used
    vision_lag: float = 0.0               # s subtracted from det.t before pairing with a pose
    elev_deg: tuple = (-40.0, 35.0)       # world elevation window of a sighting (attitude at the grab)
    range_corr: tuple | None = RANGE_CORR  # range from width -> multiplied by interp(range, table); None = off
    sig_along: tuple = (0.22, 0.3)        # along-ray std = a * range + b
    sig_cross: tuple = (0.05, 0.2)        # across-ray std = a * range + b
    crop_px: float = 3.0                  # the arch's box within this of the image edge = cropped
    crop_mult: float = 2.0                # along-ray std multiplier when cropped
    q: float = 0.02                       # m^2/s process noise per track
    perp_gate: tuple = (1.5, 0.12)        # association: ray passes within max(a, b * along) of the estimate
    log_gate: float = 0.45                # |ln(range / along)| for confirmed tracks
    log_gate_tent: float = 0.60           # ... for tentative tracks
    confirm_hits: int = 3
    confirm_span: float = 0.2
    z_plaus_sigma: float = 0.7            # height plausibility applies once sigma_z < this ...
    z_plaus_above: float = 0.5            # ... centre >= last passage height + this
    low_sigma: float = 0.5                # confirmed with sigma_z < this and centre < passage + low_above: deleted
    low_above: float = 0.3
    no_update_d: float = 2.5              # the target is not updated within this horizontal distance ...
    no_update_crop_rng: float = 4.0       # ... nor when cropped and closer than this
    tent_life: float = 2.0
    conf_life: float = 30.0
    passed_life: float = 90.0
    ghost_s: float = 2.0
    ghost_range: tuple = (6.0, 30.0)
    merge_d: float = 1.5
    merge_lat: tuple = (1.5, 0.06)
    merge_log: float = 0.7
    gate_track_sigma: bool = True         # widen the association gate by the track's own across-ray uncertainty
    stale_penalty: float = 2.0            # selection score per second a track has gone unseen beyond stale_grace (cap 8)
    stale_grace: float = 1.0
    # --- target and approach axis
    eligible_d: float = 1.0
    eligible_bearing: float = 110.0
    switch_s: float = 0.3
    lock_s: tuple = (0.3, 0.6)            # on the latest ray up to a, blended to the filter until b
    bisector_cap: float = 45.0
    turn_rot_min: float = 12.0            # no next gate: rotate the axis by min(0.5 |alpha|, turn_rot_max) beyond this
    turn_rot_max: float = 20.0
    turn_gate_alpha: float = 25.0
    turn_gate_bisector: float = 15.0
    pivot_max: float = 60.0               # axis may pivot this far from the ray at pivot_d[1], none at pivot_d[0]
    pivot_d: tuple = (3.0, 12.0)
    pivot_rate: float = 40.0              # deg/s
    pivot_tau: float = 0.4
    pivot_freeze_d: float = 4.0
    turn_hints: tuple | None = None       # optional: heading change (deg, + left) at each gate in course order
    # --- pass
    airborne_guard: float = 2.0
    fresh_hits: int = 5
    fresh_age: float = 2.5
    fresh_rng: float = 10.0
    cross_before: float = -2.0
    cross_after: float = 0.3
    cross_lat: float = 4.0
    travel_age: float = 0.4
    travel_rng: float = 7.0
    travel_frac: float = 0.8
    beside_a: float = 3.0
    beside_d: float = 8.0
    beside_bearing: float = 100.0
    seen_ahead_age: float = 0.25
    seen_ahead_along: float = 3.0
    unpass_s: float = 1.5
    behind_s: float = 20.0
    pass_absorb_d: float = 4.0            # unpassed tracks this close to a gate being passed are its fragments
    frag_ahead: float = 10.0              # a pass is a short fragment when a fresh confirmed track lies this far ahead ...
    frag_lat: float = 3.0                 # ... within this of the approach line ...
    frag_fresh_s: float = 0.5             # ... seen this recently (gates on a course are further apart)
    # --- rabbit
    v_launch: float = 2.5
    v_cruise: float = 2.5
    v_exit: float | None = None           # None = v_cruise
    v_gate: float = 3.0
    v_gate_turn: float = 2.5
    v_unsure: float = 2.5
    v_blind: float = 2.0
    v_search: float = 2.0
    a_lat: float = 1.2
    a_acc: float = 0.8
    a_brk: float = 1.2
    t_lag: float = 1.2
    kappa_max: float = 0.25
    sharp: float = 0.06                   # 1/m^2
    k_head: float = 1.5                   # 1/s
    lead: float = 3.0                     # m at the gate; + lead_open in the open
    lead_open: float = 0.5
    lead_band: float = 1.5
    lead_fmax: float = 1.3
    rabbit_stop: float = 4.5              # the rabbit waits when this far from the drone
    reseed_d: float = 6.0
    reseed_s: float = 1.0
    bump_tau: float = 0.4
    bump_vmax: float = 2.5                # m/s: the reseed offset fades no faster than this
    d_on: float = 4.0                     # m flown straight on after a pass
    launch_t: float = 6.0
    search_radius: float = 8.0
    search_radius_wide: float = 12.0      # after a full circle
    search_leash: float = 20.0
    search_side: float = 1.0              # +1 left, -1 right (default side before the course has turned)
    tent_side_hits: int = 2               # a tentative track steers the search side after this many sightings
    snap_start: float = 9.0
    snap_max: float = 1.5
    snap_rate: float = 0.8
    goal_max: float = 5.0
    goal_z: float = 1.2
    # --- altitude
    z_start: float = 1.5
    z_pass0: float = 1.2
    z_aim: float = 0.3
    z_min: float = 1.2
    up_bias: float = 0.5                  # + min(up_bias, up_bias * sigma_z)
    climb_front: float = 0.5
    vz_frac: float = 0.35
    vz_min: float = 0.4
    vz_max: float = 1.0
    az_max: float = 0.8
    grade_max: float = 0.35
    grade_len: float = 15.0
    z_window: tuple = (3.0, 12.0)         # target height within [z_aim_last - a, z_aim_last + b]
    # --- yaw
    yaw_rate: float = 2.3                 # rad/s per unit yaw stick (Liftoff 2.3; the simulator's rates 3.8)
    yaw_gain: float = 1.5                 # 1/s (2.5 limit-cycled against the simulator's lagging yaw-rate response)
    yaw_lead: float = 0.15                # s: phase lead from the measured heading rate, e + lead * (r_ref - psi_dot)
    yaw_rate_tau: float = 0.05            # s low-pass on the measured heading rate
    yaw_db: float = 2.0
    yaw_rate_cap: float = 1.0             # rad/s
    yaw_max: float = 0.35
    yaw_slew: float = 3.0                 # stick per s
    look_free: float = 35.0
    look_max: float = 25.0
    look_tau: float = 0.3
    look_kappa: float = 0.04
    sweep: float = 20.0
    sweep_period: float = 5.0
    sweep_ramp: float = 1.5
    # --- speed sense
    flow_ref: float = 2.45                # m/s the brain holds as sensed cruise
    flow_min: float | None = None         # None: min(0.7, 0.9 flow_ref / max(v_cruise, v_gate))
    flow_max: float = 1.0                 # the fly command's --flow-gain
    flow_ki: float = 0.15
    flow_trim_range: tuple = (0.5, 1.3)
    flow_tau: float = 1.0
    flow_mode: str = 'global'             # 'along': scale only the speed along the rabbit's heading
    flow_alt: str = 'start'               # 'ground': altitude sense above the ground under the rabbit's reference

    def flow_bounds(self) -> tuple[float, float]:
        top = max(self.v_cruise, self.v_gate, self.flow_ref)
        lo = min(0.7, 0.9 * self.flow_ref / top) if self.flow_min is None else self.flow_min
        return min(lo, self.flow_max), self.flow_max

    def describe(self) -> str:
        lo, hi = self.flow_bounds()
        return (f'cruise {self.v_cruise} m/s, gate {self.v_gate} (turn {self.v_gate_turn}), lead {self.lead} m, '
                f'a_lat {self.a_lat}, a_brk {self.a_brk}, yaw {self.yaw_rate} rad/s per stick gain {self.yaw_gain} '
                f'lead {self.yaw_lead} s max {self.yaw_max}, flow {lo:.2f}-{hi:.2f} (ref {self.flow_ref}, {self.flow_mode}, altitude '
                f'{self.flow_alt}), elevation {self.elev_deg}, vision lag {self.vision_lag} s, z aim {self.z_aim}, '
                f'snap from {self.snap_start} m, search {"left" if self.search_side > 0 else "right"} '
                f'r {self.search_radius} m' + (f', turn hints {self.turn_hints}' if self.turn_hints else ''))


class Track:
    """One arch: a world Kalman estimate of its visual centre and what the pass logic needs."""
    __slots__ = ('id', 'm', 'P', 'r0', 'dir_first', 'hits', 't_first', 't_last', 'p_last', 'r_last', 'rng_last_h',
                 'last_along', 'confirmed', 'passed', 't_passed', 'n_pass', 'min_a', 'unseen_in_view')

    def __init__(self, tid: int, now: float, z: np.ndarray, Rm: np.ndarray, r: np.ndarray, pg: np.ndarray, rho: float):
        self.id = tid
        self.m = z.copy()
        self.P = Rm.copy()
        self.r0 = r.copy()
        dx, dy = float(z[0] - pg[0]), float(z[1] - pg[1])
        n = math.hypot(dx, dy)
        self.dir_first = (dx / n, dy / n) if n > 1e-6 else (1.0, 0.0)
        self.hits = 1
        self.t_first = self.t_last = now
        self.p_last = pg.copy()
        self.r_last = r.copy()
        self.rng_last_h = n
        self.last_along = rho
        self.confirmed = False
        self.passed = False
        self.t_passed = -math.inf
        self.n_pass = (1.0, 0.0)
        self.min_a = math.inf
        self.unseen_in_view = 0.0

    def dist_h(self, p) -> float:
        return math.hypot(float(self.m[0] - p[0]), float(self.m[1] - p[1]))


class SightPilot:
    """State and per-tick logic of the rabbit pilot. ``host`` is the TelemetryPilot (clock, last_R, last_vel, vision,
    pose_at, flow_gain; vision_gate_w / _passed / vision_passed_t / vision_status are kept up to date for logs)."""

    def __init__(self, host, params: SightParams | None = None):
        self.host = host
        self.params = params or SightParams()
        self.errors = 0
        self._err_logged = False
        self._consec_err = 0
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        P = self.params
        self._init = False
        self.t_prev = None
        self.t_air = None
        self.grounded_for = 0.0
        self.det_seen = -1
        self.sight_yaw = 0.0
        self.flow_f = P.flow_max
        self.flow_trim = 1.0
        for k in ('rej_elev', 'rej_stale', 'absorbed', 'behind', 'ghosts', 'low', 'unpasses', 'reseeds', 'goal_clips'):
            setattr(self, k, 0)
        self._next_id = 0
        self.pass_kind = 0
        self.carrot = None
        self._last_rel = None
        self._full_init(np.zeros(3), 0.0, 0.0)
        self._init = False

    def _full_init(self, p: np.ndarray, psi_n: float, now: float) -> None:
        P = self.params
        self._init = True
        self.tracks: list[Track] = []
        self.target: Track | None = None
        self.pending: Track | None = None
        self.pending_since = 0.0
        self.next: Track | None = None
        self.n_ang: float | None = None
        self.turn_gate = False
        self.sd_lat = 0.0
        self.g_s = None
        self.d_gate = math.inf
        self.psi_launch = psi_n
        self.launch_p = (float(p[0]), float(p[1]))
        self.c = [float(p[0]) + 2.5 * math.cos(psi_n), float(p[1]) + 2.5 * math.sin(psi_n)]
        self.psi = psi_n
        self.kappa = 0.0
        self.v = 0.0
        self.v_nom = 0.0
        self.v_des = 0.0
        self.s = 0.0
        self.z_c = P.z_start
        self.vz_c = 0.0
        self.trail = [(0.0, self.c[0], self.c[1])]
        self.s_d_prev = -math.inf
        self.far_since = None
        self.o_snap = 0.0
        self.o_bump = np.zeros(3)
        self.last_pass = None                 # dict(t, m, n, s, kind)
        self.pass_backup = None
        self.n_passes = 0
        self.z_pass_last = P.z_pass0
        self.z_aim_last = P.z_start
        self.grade_last = 0.0
        self.side = 1.0 if P.search_side >= 0 else -1.0
        self.mode = 0
        self.mode_since = now
        self.search_turned = 0.0
        self.last_tent_b = 0.0
        self.last_tent_t = -math.inf
        self.look = 0.0
        self.look_prev = 0.0
        self.psi_dot = 0.0
        self.yaw_ref = psi_n
        self.lead_now = 0.0

    @property
    def ready(self) -> bool:
        """Initialised: the rabbit's heading and reference height mean something (for the optional flow modes)."""
        return self._init

    # ------------------------------------------------------------------ public tick
    def goal(self, pos_w) -> np.ndarray:
        """Body-frame goal vector for the brain. Never raises."""
        try:
            rel_b = self._goal(np.asarray(pos_w, dtype=np.float64))
            if not np.all(np.isfinite(rel_b)):
                raise FloatingPointError(f'non-finite goal {rel_b}')
            self._consec_err = 0
            return rel_b
        except Exception:                                    # noqa: BLE001 - the 100 Hz loop must go on
            self.errors += 1
            self._consec_err += 1
            if not self._err_logged:
                self._err_logged = True
                print('SIGHT PILOT ERROR (reported once; holding the last carrot):\n' + traceback.format_exc(),
                      file=sys.stderr, flush=True)
            if self._consec_err >= 20:
                self._init = False                            # start over on the next tick
            return self._safe_goal(np.asarray(pos_w, dtype=np.float64))

    def _safe_goal(self, p: np.ndarray) -> np.ndarray:
        try:
            R = self.host.last_R
            P = self.params
            if self.carrot is not None and np.all(np.isfinite(self.carrot)):
                rel = np.asarray(self.carrot, dtype=np.float64) - p
            else:
                rel = np.array([0.0, 0.0, P.z_start - float(p[2])])
            h = math.hypot(rel[0], rel[1])
            if h > P.goal_max:
                rel[:2] *= P.goal_max / h
            rel[2] = clip(float(rel[2]), -P.goal_z, P.goal_z)
            self.sight_yaw = clip(0.0, self.sight_yaw - P.yaw_slew * 0.01, self.sight_yaw + P.yaw_slew * 0.01)
            out = R.T @ rel
            return out if np.all(np.isfinite(out)) else np.zeros(3)
        except Exception:                                    # noqa: BLE001
            self.sight_yaw = 0.0
            return np.zeros(3)

    # ------------------------------------------------------------------ the tick
    def _goal(self, p: np.ndarray) -> np.ndarray:
        P = self.params
        h = self.host
        now = float(h.clock())
        self._p = p
        R = h.last_R
        vel = np.asarray(h.last_vel, dtype=np.float64)
        psi_n = math.atan2(R[1, 0], R[0, 0])
        if self.t_prev is None:
            self.t_prev = now
        dt = clip(now - self.t_prev, 0.0, 0.05)
        self.t_prev = now
        self.pass_kind = 0
        on_ground = p[2] < 0.3 and float(vel @ vel) < 0.25
        self.grounded_for = self.grounded_for + dt if on_ground else 0.0
        if not self._init or (on_ground and (self.t_air is None or self.grounded_for > 1.5)):
            self._full_init(p, psi_n, now)
            self.t_air = None
        elif self.t_air is None and p[2] > 0.8:
            self.t_air = now

        # 1 intake: each detector frame once
        det = h.vision.get()
        if det.frames != self.det_seen:
            self.det_seen = det.frames
            if det.frames > 0 and det.p_visible >= P.p_min and det.dist_m > 0.5:
                tg = float(det.t) - P.vision_lag
                pose = h.pose_at(tg) if now - tg <= P.stale_s else None
                if pose is None:
                    self.rej_stale += 1
                else:
                    self._intake(now, det, pose, p)

        # 2 maintenance
        self._maintain(now, dt, p, R)

        # 3 target
        self._select(now, p)
        T = self.target
        self.next = None
        self.g_s = None
        self.d_gate = math.inf
        n_vec = None
        if T is not None:
            # 4 locked point, 5 approach axis
            age = now - T.t_last
            g_s = self._locked(T, age)
            self.g_s = g_s
            n_vec = self._axis(T, g_s, p, now, dt)
            # 6 pass
            T = self._pass_check(T, n_vec, p, now, age)
            if T is None:
                n_vec = None
                self.g_s = None
                self.d_gate = math.inf
        h.vision_gate_w = None if self.target is None else self.target.m.copy()

        # guidance (mode and desired speed / curvature)
        self._guidance(now, dt, p, vel)
        # rabbit integration, reseed
        self._integrate(now, dt, p, vel, R)
        # altitude reference
        self._altitude(dt)
        # goal vector
        rel = self._goal_vector(dt, p, on_ground)
        # yaw and speed sense
        self._yaw(now, dt, psi_n, on_ground)
        self._flow(now, dt, vel)
        self._status(now)
        self._last_rel = rel
        return R.T @ rel

    # ------------------------------------------------------------------ 1 intake
    def _range(self, dist: float) -> float:
        P = self.params
        rho = dist
        if P.range_corr:
            d, k = zip(*P.range_corr)
            rho = dist * float(np.interp(dist, d, k))
        return clip(rho, 1.0, 45.0)

    def _intake(self, now: float, det, pose, p: np.ndarray) -> None:
        P = self.params
        pg = np.asarray(pose[0], dtype=np.float64)
        Rg = pose[1]
        r = Rg @ np.asarray(det.direction_body, dtype=np.float64)
        r = r / max(float(np.linalg.norm(r)), 1e-9)
        el = math.asin(clip(float(r[2]), -1.0, 1.0))
        if not (P.elev_deg[0] * DEG < el < P.elev_deg[1] * DEG):
            self.rej_elev += 1
            return
        rho = self._range(float(det.dist_m))
        cam = getattr(self.host.vision, 'cam', None)
        W, H = (float(cam.width), float(cam.height)) if cam is not None else (320.0, 180.0)
        hw = 0.5 * float(det.width_px)
        crop = (det.u - hw < P.crop_px or det.u + hw > W - P.crop_px or det.v - hw < P.crop_px
                or det.v + hw > H - P.crop_px)
        sa = (P.sig_along[0] * rho + P.sig_along[1]) * (P.crop_mult if crop else 1.0)
        sc = P.sig_cross[0] * rho + P.sig_cross[1]
        rr = np.outer(r, r)
        Rm = sa * sa * rr + sc * sc * (np.eye(3) - rr)
        z = pg + rho * r
        T = self._update(now, z, Rm, r, rho, pg, crop)
        # the side to search: a tentative arch seen twice (a single sighting is as often a phantom or a flip)
        if T is not None and not T.confirmed and T.hits >= P.tent_side_hits:
            self.last_tent_b = wrap(math.atan2(float(T.m[1] - p[1]), float(T.m[0] - p[0])) - self.psi)
            self.last_tent_t = now

    def _fit(self, T: Track, pg: np.ndarray, r: np.ndarray, rho: float, g: float) -> float | None:
        rel = T.m - pg
        al = float(rel @ r)
        if al <= 0.5:
            return None
        perp = float(np.linalg.norm(rel - al * r))
        P = self.params
        lim = max(P.perp_gate[0], P.perp_gate[1] * al)
        if P.gate_track_sigma:
            # a young estimate is itself uncertain across the ray: widen the gate by the total/measurement std ratio
            var_t = max(0.5 * (float(T.P[0, 0] + T.P[1, 1] + T.P[2, 2]) - float(r @ T.P @ r)), 0.0)
            sc = P.sig_cross[0] * al + P.sig_cross[1]
            lim *= math.sqrt(1.0 + var_t / (sc * sc))
        lr = abs(math.log(rho / al))
        if perp > lim or lr > g:
            return None
        return perp / lim + lr / g

    def _update(self, now, z, Rm, r, rho, pg, crop) -> Track | None:
        P = self.params
        for T in self.tracks:
            if not T.passed:
                continue
            nx, ny = T.n_pass
            ahead = float((z[0] - pg[0]) * nx + (z[1] - pg[1]) * ny) > 1.0
            # a passed arch absorbs its sightings from behind; after the un-pass window also from its entry side (the
            # drone came round in a search: the course does not go through the same arch again)
            if ((not ahead or now - T.t_passed > P.unpass_s)
                    and (math.hypot(float(z[0] - T.m[0]), float(z[1] - T.m[1])) < 4.0
                         or self._fit(T, pg, r, rho, 0.35) is not None)):
                self.absorbed += 1
                return None
            if (ahead and now - T.t_passed < P.unpass_s and self._fit(T, pg, r, rho, 0.45) is not None
                    and float((T.m[0] - pg[0]) * nx + (T.m[1] - pg[1]) * ny) > 2.0):
                self._unpass(T, now)
        lp = self.last_pass
        if lp is not None and now - lp['t'] < P.behind_s:
            if float((z[0] - lp['m'][0]) * lp['n'][0] + (z[1] - lp['m'][1]) * lp['n'][1]) < -1.0:
                self.behind += 1
                return None
        best, cost = None, math.inf
        for T in self.tracks:
            if T.passed:
                continue
            c = self._fit(T, pg, r, rho, P.log_gate if T.confirmed else P.log_gate_tent)
            if c is not None and c < cost:
                best, cost = T, c
        if best is None:
            self._next_id += 1
            T = Track(self._next_id, now, z, Rm, r, pg, rho)
            self.tracks.append(T)
            return T
        T = best
        T.hits += 1
        T.t_last = now
        T.p_last = pg.copy()
        T.r_last = r.copy()
        T.unseen_in_view = 0.0
        T.last_along = min(rho, float((T.m - pg) @ r))
        if not (T is self.target and (T.dist_h(pg) < P.no_update_d or (crop and rho < P.no_update_crop_rng))):
            S = T.P + Rm
            K = T.P @ np.linalg.inv(S)
            T.m = T.m + K @ (z - T.m)
            T.P = (np.eye(3) - K) @ T.P
            T.P = 0.5 * (T.P + T.P.T)
            # successive width ranges share their error: the along-ray variance may only shrink with parallax
            par = math.degrees(math.acos(clip(float(r @ T.r0), -1.0, 1.0)))
            sf = 0.08 * rho * (1.0 - min(par / 8.0, 1.0)) + 0.15
            vr = float(r @ T.P @ r)
            if vr < sf * sf:
                T.P = T.P + (sf * sf - vr) * np.outer(r, r)
        T.rng_last_h = T.dist_h(pg)
        if (not T.confirmed and T.hits >= P.confirm_hits and now - T.t_first >= P.confirm_span
                and (math.sqrt(max(T.P[2, 2], 0.0)) >= P.z_plaus_sigma or T.m[2] >= self.z_pass_last + P.z_plaus_above)):
            T.confirmed = True
        return T

    # ------------------------------------------------------------------ 2 maintenance
    def _in_view(self, T: Track, p: np.ndarray, R: np.ndarray, cam) -> bool:
        rel = T.m - p
        rng = float(np.linalg.norm(rel))
        lo, hi = self.params.ghost_range
        if cam is None or not (lo <= rng <= hi):
            return False
        px, ok = cam.project_body((R.T @ rel)[None])
        u, v = px[0]
        return bool(ok[0]) and 0.05 * cam.width < u < 0.95 * cam.width and 0.05 * cam.height < v < 0.95 * cam.height

    def _maintain(self, now: float, dt: float, p: np.ndarray, R: np.ndarray) -> None:
        P = self.params
        cam = getattr(self.host.vision, 'cam', None)
        near, near_d = None, math.inf
        for T in self.tracks:
            T.P = T.P + (P.q * dt) * np.eye(3)
            if T.confirmed and not T.passed:
                d = float(np.linalg.norm(T.m - p))
                if d < near_d and self._in_view(T, p, R, cam):
                    near, near_d = T, d
        if near is not None:
            near.unseen_in_view += dt
        keep = []
        for T in self.tracks:
            if T.passed:
                if now - T.t_passed <= P.passed_life:
                    keep.append(T)
                continue
            if not T.confirmed:
                if now - T.t_last <= P.tent_life:
                    keep.append(T)
                continue
            if T is not self.target and now - T.t_last > P.conf_life:
                continue
            if T.unseen_in_view > P.ghost_s:
                self.ghosts += 1
                self.pass_kind = PASS_KIND['ghost']
                if T is self.target:
                    self.target = None
                    self.n_ang = None
                continue
            if math.sqrt(max(T.P[2, 2], 0.0)) < P.low_sigma and T.m[2] < self.z_pass_last + P.low_above:
                self.low += 1
                if T is self.target:
                    self.target = None
                    self.n_ang = None
                continue
            keep.append(T)
        # merge fragments of one arch (confirmed, unpassed), fusing the smaller into the larger
        out: list[Track] = []
        for T in sorted(keep, key=lambda t: -t.hits):
            twin = None
            if T.confirmed and not T.passed:
                for o in out:
                    if o.confirmed and not o.passed and self._same(o, T, p):
                        twin = o
                        break
            if twin is None:
                out.append(T)
                continue
            S = twin.P + T.P
            K = twin.P @ np.linalg.inv(S)
            twin.m = twin.m + K @ (T.m - twin.m)
            twin.P = (np.eye(3) - K) @ twin.P
            twin.P = 0.5 * (twin.P + twin.P.T)
            twin.hits += T.hits
            if T.t_last > twin.t_last:
                twin.t_last, twin.p_last, twin.r_last = T.t_last, T.p_last, T.r_last
            twin.unseen_in_view = min(twin.unseen_in_view, T.unseen_in_view)
            if self.target is T:
                self.target = twin
            if self.pending is T:
                self.pending = twin
        self.tracks = out
        if self.target is not None and self.target not in out:
            self.target = None
            self.n_ang = None

    def _same(self, a: Track, b: Track, p: np.ndarray) -> bool:
        P = self.params
        if math.hypot(float(a.m[0] - b.m[0]), float(a.m[1] - b.m[1])) < P.merge_d:
            return True
        va, vb = a.m - p, b.m - p
        ra, rb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        if min(ra, rb) < 1.0 or float(va @ vb) <= 0.0:
            return False                      # the cross product below cannot tell a track behind from one ahead
        if ra <= rb:
            lat = float(np.linalg.norm(np.cross(va / ra, vb)))
        else:
            lat = float(np.linalg.norm(np.cross(vb / rb, va)))
        return lat < max(P.merge_lat[0], P.merge_lat[1] * min(ra, rb)) and abs(math.log(ra / rb)) < P.merge_log

    # ------------------------------------------------------------------ 3 target
    def _select(self, now: float, p: np.ndarray) -> None:
        P = self.params
        best, best_score = None, math.inf
        for T in self.tracks:
            if not T.confirmed or T.passed:
                continue
            dx, dy = float(T.m[0] - p[0]), float(T.m[1] - p[1])
            d = math.hypot(dx, dy)
            b = wrap(math.atan2(dy, dx) - self.psi)
            is_t = T is self.target
            if not is_t and (d < P.eligible_d or abs(b) > P.eligible_bearing * DEG):
                continue
            score = d + 8.0 * (1.0 - math.cos(b)) - (4.0 if is_t else 0.0)
            # an estimate not seen for a while loses its standing against arches seen now
            score += min(P.stale_penalty * max(now - T.t_last - P.stale_grace, 0.0), 8.0)
            if score < best_score:
                best, best_score = T, score
        if self.target is None:
            if best is not None:
                self.target = best
                self.n_ang = None
            self.pending = None
        elif best is not self.target and best is not None:
            if self.pending is not best:
                self.pending, self.pending_since = best, now
            elif now - self.pending_since >= P.switch_s:
                self.target = best
                self.n_ang = None
                self.pending = None
        else:
            self.pending = None

    # ------------------------------------------------------------------ 4-5 locked point and approach axis
    def _locked(self, T: Track, age: float) -> np.ndarray:
        a0, a1 = self.params.lock_s
        if age > a1:
            return T.m
        lock = T.p_last + T.r_last * max(float((T.m - T.p_last) @ T.r_last), 0.5)
        if age <= a0:
            return lock
        f = (age - a0) / max(a1 - a0, 1e-6)
        return lock + f * (T.m - lock)

    def _axis(self, T: Track, g_s: np.ndarray, p: np.ndarray, now: float, dt: float) -> tuple[float, float]:
        P = self.params
        gx, gy = float(g_s[0]), float(g_s[1])
        rx, ry = gx - float(p[0]), gy - float(p[1])
        d = math.hypot(rx, ry)
        self.d_gate = d
        a_ray = math.atan2(ry, rx)
        lp = self.last_pass
        if lp is not None and now - lp['t'] < 60.0 and math.hypot(gx - lp['m'][0], gy - lp['m'][1]) > 4.0:
            a_cin = math.atan2(gy - lp['m'][1], gx - lp['m'][0])
        else:
            a_cin = math.atan2(T.dir_first[1], T.dir_first[0])
        cx, cy = math.cos(a_cin), math.sin(a_cin)
        nxt, nxt_d = None, math.inf
        for o in self.tracks:
            if o is T or o.passed or not o.confirmed:
                continue
            vx, vy = float(o.m[0]) - gx, float(o.m[1]) - gy
            dv = math.hypot(vx, vy)
            if 3.0 < dv < 45.0 and vx * cx + vy * cy > 3.0 and dv < nxt_d:
                nxt, nxt_d = o, dv
        self.next = nxt
        hints = P.turn_hints
        if nxt is not None:
            ux, uy = (float(nxt.m[0]) - gx) / nxt_d, (float(nxt.m[1]) - gy) / nxt_d
            a_bis = math.atan2(cy + uy, cx + ux)
            rot = wrap(a_bis - a_cin)
            a_nom = a_cin + clip(rot, -P.bisector_cap * DEG, P.bisector_cap * DEG)
            self.turn_gate = abs(rot) > P.turn_gate_bisector * DEG
        elif hints and self.n_passes < len(hints):
            hint = float(hints[self.n_passes]) * DEG
            a_nom = a_cin + 0.5 * hint
            self.turn_gate = abs(hint) >= P.turn_gate_alpha * DEG
        elif lp is not None:
            alpha = wrap(a_cin - math.atan2(lp['n'][1], lp['n'][0]))
            if abs(alpha) >= P.turn_rot_min * DEG:
                a_nom = a_cin + math.copysign(min(0.5 * abs(alpha), P.turn_rot_max * DEG), alpha)
            else:
                a_nom = a_cin
            self.turn_gate = abs(alpha) >= P.turn_gate_alpha * DEG
        else:
            a_nom = a_cin
            self.turn_gate = False
        beta = wrap(a_nom - a_ray)
        lim = P.pivot_max * DEG * clip((d - P.pivot_d[0]) / max(P.pivot_d[1] - P.pivot_d[0], 1e-6), 0.0, 1.0)
        a_app = a_ray + clip(beta, -lim, lim)
        if self.n_ang is None:
            self.n_ang = wrap(a_app)
        elif d >= P.pivot_freeze_d:
            step = wrap(a_app - self.n_ang) * min(1.0, dt / P.pivot_tau)
            self.n_ang = wrap(self.n_ang + clip(step, -P.pivot_rate * DEG * dt, P.pivot_rate * DEG * dt))
        nx, ny = math.cos(self.n_ang), math.sin(self.n_ang)
        lx, ly = -ny, nx
        Pm = T.P
        self.sd_lat = math.sqrt(max(lx * lx * Pm[0, 0] + 2 * lx * ly * Pm[0, 1] + ly * ly * Pm[1, 1], 0.0))
        return nx, ny

    # ------------------------------------------------------------------ 6 pass
    def _pass_check(self, T: Track, n_vec, p: np.ndarray, now: float, age: float) -> Track | None:
        P = self.params
        nx, ny = n_vec
        dx, dy = float(p[0] - T.m[0]), float(p[1] - T.m[1])
        a_d = dx * nx + dy * ny
        lat_d = -dx * ny + dy * nx
        T.min_a = min(T.min_a, a_d)
        crossed = T.min_a <= P.cross_before and a_d >= P.cross_after and abs(lat_d) < P.cross_lat
        travelled = (age > P.travel_age and T.rng_last_h < P.travel_rng
                     and float((p[0] - T.p_last[0]) * nx + (p[1] - T.p_last[1]) * ny) >= P.travel_frac * T.rng_last_h)
        bear = abs(wrap(math.atan2(-dy, -dx) - self.psi))
        beside = a_d > P.beside_a or (self.d_gate < P.beside_d and bear > P.beside_bearing * DEG)
        seen_ahead = age < P.seen_ahead_age and T.last_along > P.seen_ahead_along
        self.dbg = (a_d, lat_d, T.min_a, crossed, travelled, beside, seen_ahead, self.d_gate, math.degrees(bear))
        if not (self.t_air is not None and now - self.t_air > P.airborne_guard and (crossed or travelled or beside)
                and not seen_ahead):
            return T
        frag = self._fragment_ahead(T, n_vec, now)
        if frag is not None:
            # T is a short estimate of an arch seen just ahead on the same line: not a gate of its own
            if T in self.tracks:
                self.tracks.remove(T)
            self.absorbed += 1
            frag.hits += T.hits
            self.target = frag
            self.pending = None
            return None
        if T.hits >= P.fresh_hits and (age < P.fresh_age or T.rng_last_h < P.fresh_rng):
            self._pass(T, now, 'cross' if crossed else 'travel' if travelled else 'beside', n_vec)
        else:
            if T in self.tracks:
                self.tracks.remove(T)
            self.ghosts += 1
            self.pass_kind = PASS_KIND['ghost']
            self.target = None
            self.n_ang = None
        return None

    def _fragment_ahead(self, T: Track, n_vec, now: float) -> Track | None:
        P = self.params
        if P.frag_ahead <= 0:
            return None
        nx, ny = n_vec
        best, best_a = None, math.inf
        for o in self.tracks:
            if o is T or o.passed or not o.confirmed or now - o.t_last > P.frag_fresh_s:
                continue
            dx, dy = float(o.m[0] - T.m[0]), float(o.m[1] - T.m[1])
            a = dx * nx + dy * ny
            if 0.5 < a <= P.frag_ahead and abs(-dx * ny + dy * nx) < P.frag_lat and a < best_a:
                best, best_a = o, a
        return best

    def _pass(self, T: Track, now: float, kind: str, n_vec) -> None:
        P = self.params
        lp = self.last_pass
        self.pass_backup = (self.z_pass_last, self.z_aim_last, self.grade_last, lp, self.side, self.n_passes)
        T.passed = True
        T.t_passed = now
        T.n_pass = (float(n_vec[0]), float(n_vec[1]))
        zp = float(T.m[2]) - CENTRE_UP_M
        if lp is not None:
            dist = math.hypot(float(T.m[0] - lp['m'][0]), float(T.m[1] - lp['m'][1]))
            self.grade_last = clip((zp - self.z_pass_last) / max(dist, 5.0), 0.0, P.grade_max)
            turn = wrap(self.n_ang - math.atan2(lp['n'][1], lp['n'][0])) if self.n_ang is not None else 0.0
            if abs(turn) > 10.0 * DEG:
                self.side = math.copysign(1.0, turn)
        else:
            self.grade_last = 0.0
        self.z_pass_last = zp
        self.z_aim_last = clip(zp + P.z_aim, P.z_min, 30.0)
        self.last_pass = {'t': now, 'm': (float(T.m[0]), float(T.m[1]), float(T.m[2])), 'n': T.n_pass, 's': self.s,
                          'kind': kind, 'track': T}
        self.n_passes += 1
        self.target = None
        self.n_ang = None
        self.pending = None
        self.pass_kind = PASS_KIND[kind]
        # fragments of the same arch (made while the target was frozen close up) would be passed again a moment later
        for o in [o for o in self.tracks if o is not T and not o.passed]:
            if math.hypot(float(o.m[0] - T.m[0]), float(o.m[1] - T.m[1])) < P.pass_absorb_d:
                self.tracks.remove(o)
                self.absorbed += 1
        h = self.host
        if hasattr(h, '_passed'):
            h._passed.append((now, T.m.copy()))
        h.vision_passed_t = now

    def _unpass(self, T: Track, now: float) -> None:
        T.passed = False
        T.min_a = math.inf
        if self.pass_backup is not None:
            (self.z_pass_last, self.z_aim_last, self.grade_last, self.last_pass, self.side,
             self.n_passes) = self.pass_backup
            self.pass_backup = None
        self.target = T
        self.n_ang = None
        self.pending = None
        self.unpasses += 1
        self.pass_kind = PASS_KIND['unpass']
        h = self.host
        if getattr(h, '_passed', None):
            h._passed.pop()

    # ------------------------------------------------------------------ guidance
    def _guidance(self, now: float, dt: float, p: np.ndarray, vel: np.ndarray) -> None:
        P = self.params
        T = self.target
        mode_prev = self.mode
        lp = self.last_pass
        if self.t_air is None:
            self.mode = 0
            v_des, k_des = 0.0, 0.0
        elif T is not None and self.g_s is not None:
            self.mode = 2
            nx, ny = math.cos(self.n_ang), math.sin(self.n_ang)
            lx, ly = -ny, nx
            relx, rely = self.c[0] - float(self.g_s[0]), self.c[1] - float(self.g_s[1])
            a_r = relx * nx + rely * ny
            e = relx * lx + rely * ly
            if a_r < 0.0:
                psi_des = self.n_ang + math.atan2(-e, clip(0.6 * -a_r, 3.0, 8.0))
            else:
                psi_des = self.n_ang
            err = wrap(psi_des - self.psi)
            k_des = P.k_head * err / max(self.v_nom, 1.0)
            D = max(-a_r - 1.0, 1.0)
            k_req = min(max(abs(wrap(self.n_ang - self.psi)) / D, 4.0 * abs(e) / (D * D), abs(err) / 8.0), P.kappa_max)
            v_gate = (P.v_gate_turn if self.turn_gate else P.v_gate) * clip(1.15 - 0.5 * self.sd_lat, 0.75, 1.0)
            v_brake = math.sqrt(v_gate * v_gate + 2.0 * P.a_brk * max(0.0, -a_r - self.v_nom * P.t_lag))
            v_des = min(P.v_cruise, math.sqrt(P.a_lat / max(k_req, 1e-3)), v_brake)
            if self.d_gate < 10.0 and self.sd_lat > 0.7:
                v_des = min(v_des, P.v_unsure)
            if now - T.t_last > 2.5:
                v_des = min(v_des, P.v_blind)
        elif (lp is not None and self.s - lp['s'] < P.d_on) or (lp is None and now - self.t_air < P.launch_t):
            self.mode = 1
            h_ref = math.atan2(lp['n'][1], lp['n'][0]) if lp is not None else self.psi_launch
            hints = P.turn_hints
            if lp is not None and hints and 0 < self.n_passes <= len(hints):
                h_ref += 0.5 * float(hints[self.n_passes - 1]) * DEG
            k_des = P.k_head * wrap(h_ref - self.psi) / max(self.v_nom, 1.0)
            v_des = (P.v_exit if P.v_exit is not None else P.v_cruise) if lp is not None else P.v_launch
        else:
            self.mode = 3
            if mode_prev != 3:
                self.mode_since = now
                self.search_turned = 0.0
            s_side = None
            hints = P.turn_hints
            if hints and 0 < self.n_passes <= len(hints) and hints[self.n_passes - 1]:
                s_side = math.copysign(1.0, float(hints[self.n_passes - 1]))
            if s_side is None and now - self.last_tent_t < 3.0 and 10.0 * DEG < abs(self.last_tent_b) < 120.0 * DEG:
                s_side = math.copysign(1.0, self.last_tent_b)
            if s_side is None:
                others = [o for o in self.tracks if o.confirmed and not o.passed]
                if others:
                    o = min(others, key=lambda o: o.dist_h(p))
                    b = wrap(math.atan2(float(o.m[1] - p[1]), float(o.m[0] - p[0])) - self.psi)
                    s_side = math.copysign(1.0, b)
            if s_side is None:
                s_side = self.side
            if lp is not None:
                ax, ay = lp['m'][0] + 6.0 * lp['n'][0], lp['m'][1] + 6.0 * lp['n'][1]
            else:
                ax = self.launch_p[0] + 10.0 * math.cos(self.psi_launch)
                ay = self.launch_p[1] + 10.0 * math.sin(self.psi_launch)
            if math.hypot(float(p[0]) - ax, float(p[1]) - ay) > P.search_leash:
                k_des = P.k_head * wrap(math.atan2(ay - self.c[1], ax - self.c[0]) - self.psi) / max(self.v_nom, 1.0)
            else:
                radius = P.search_radius if self.search_turned < TWO_PI else P.search_radius_wide
                k_des = s_side / radius
            v_des = P.v_search
            self.search_turned += abs(self.kappa * self.v * dt)
        self.v_des = v_des
        self.k_des = k_des

    # ------------------------------------------------------------------ rabbit integration
    def _drone_s(self, p: np.ndarray) -> float:
        tr = self.trail
        if len(tr) < 2:
            return self.s - math.hypot(self.c[0] - float(p[0]), self.c[1] - float(p[1]))
        lo = max(self.s_d_prev - 1.0, self.s - 15.0)
        i0 = max(bisect_left(tr, lo, key=itemgetter(0)) - 1, 0)
        seg = tr[i0:]
        if len(seg) < 2:
            seg = tr[-2:]
        A = np.array(seg + [(self.s, self.c[0], self.c[1])])
        S, X, Y = A[:, 0], A[:, 1], A[:, 2]
        ax, ay = X[:-1], Y[:-1]
        bx, by = X[1:] - ax, Y[1:] - ay
        L2 = np.maximum(bx * bx + by * by, 1e-9)
        f = np.clip(((float(p[0]) - ax) * bx + (float(p[1]) - ay) * by) / L2, 0.0, 1.0)
        qx, qy = ax + f * bx - float(p[0]), ay + f * by - float(p[1])
        i = int(np.argmin(qx * qx + qy * qy))
        self.s_d_prev = float(S[i] + f[i] * (S[i + 1] - S[i]))
        return self.s_d_prev

    def _carrot_world(self) -> np.ndarray:
        lx, ly = -math.sin(self.psi), math.cos(self.psi)
        return np.array([self.c[0] + self.o_snap * lx, self.c[1] + self.o_snap * ly, self.z_c]) + self.o_bump

    def _integrate(self, now: float, dt: float, p: np.ndarray, vel: np.ndarray, R: np.ndarray) -> None:
        P = self.params
        self.v_nom += clip(self.v_des - self.v_nom, -P.a_brk * dt, P.a_acc * dt)
        s_d = self._drone_s(p)
        lead0 = P.lead + P.lead_open * clip((self.d_gate - 4.0) / 6.0, 0.0, 1.0)
        self.lead_now = self.s - s_d
        f = clip(1.0 - (self.lead_now - lead0) / P.lead_band, 0.0, P.lead_fmax)
        dist_c = math.hypot(self.c[0] - float(p[0]), self.c[1] - float(p[1]))
        if dist_c > P.rabbit_stop:
            f = 0.0
        self.v += clip(self.v_nom * f - self.v, -2.0 * dt, 2.0 * dt)
        kmax = min(P.kappa_max, P.a_lat / max(self.v_nom, 0.5) ** 2)
        dk = P.sharp * max(self.v, 0.3) * dt
        self.kappa += clip(clip(self.k_des, -kmax, kmax) - self.kappa, -dk, dk)
        ds = self.v * dt
        self.psi = wrap(self.psi + self.kappa * ds)
        self.c[0] += ds * math.cos(self.psi)
        self.c[1] += ds * math.sin(self.psi)
        self.s += ds
        if self.s - self.trail[-1][0] >= 0.2:
            self.trail.append((self.s, self.c[0], self.c[1]))
            if len(self.trail) > 400:
                del self.trail[:100]
        # reseed when the drone has been far from the rabbit for a while (a bounce, a recovery)
        if dist_c > P.reseed_d:
            if self.far_since is None:
                self.far_since = now
            elif now - self.far_since > P.reseed_s:
                old = self._carrot_world()
                vh = math.hypot(float(vel[0]), float(vel[1]))
                if vh > 1.0:
                    ux, uy = float(vel[0]) / vh, float(vel[1]) / vh
                else:
                    a = math.atan2(R[1, 0], R[0, 0])
                    ux, uy = math.cos(a), math.sin(a)
                self.c = [float(p[0]) + 2.5 * ux, float(p[1]) + 2.5 * uy]
                self.psi = math.atan2(uy, ux)
                self.kappa = 0.0
                self.trail = [(self.s, self.c[0], self.c[1])]
                self.s_d_prev = -math.inf
                self.o_bump = np.zeros(3)
                self.o_bump = old - self._carrot_world()
                self.far_since = None
                self.reseeds += 1
        else:
            self.far_since = None
        nb = float(np.linalg.norm(self.o_bump))
        if nb > 0.0:
            shrink = min(nb * (1.0 - math.exp(-dt / P.bump_tau)), P.bump_vmax * dt)
            self.o_bump *= max(nb - shrink, 0.0) / nb

    # ------------------------------------------------------------------ altitude
    def _altitude(self, dt: float) -> None:
        P = self.params
        T = self.target
        lp = self.last_pass
        if T is not None and self.g_s is not None:
            sz = math.sqrt(max(T.P[2, 2], 0.0))
            z_tgt = float(T.m[2]) - CENTRE_UP_M + P.z_aim + min(P.up_bias, P.up_bias * sz)
            z_tgt = clip(z_tgt, max(P.z_min, self.z_aim_last - P.z_window[0]), self.z_aim_last + P.z_window[1])
            T_z = max((self.d_gate - 1.0) / max(self.v_nom, 1.0), 1.0)
            if z_tgt > self.z_c:
                T_z = max(P.climb_front * T_z, 1.0)
        else:
            z_tgt = self.z_aim_last
            if lp is not None:
                z_tgt += self.grade_last * min(self.s - lp['s'], P.grade_len)
            T_z = 1.5
        vlim = min(max(P.vz_frac * self.v_nom, P.vz_min), P.vz_max)
        vz_des = clip((z_tgt - self.z_c) / T_z, -vlim, vlim)
        self.vz_c += clip(vz_des - self.vz_c, -P.az_max * dt, P.az_max * dt)
        self.z_c += self.vz_c * dt

    # ------------------------------------------------------------------ goal
    def _goal_vector(self, dt: float, p: np.ndarray, on_ground: bool) -> np.ndarray:
        P = self.params
        snap_des = 0.0
        if self.mode == 2 and self.g_s is not None and self.d_gate < P.snap_start:
            rx, ry = float(self.g_s[0] - p[0]), float(self.g_s[1] - p[1])
            rn = max(math.hypot(rx, ry), 1e-6)
            rx, ry = rx / rn, ry / rn
            qx, qy = self.c[0] - float(p[0]), self.c[1] - float(p[1])
            al = qx * rx + qy * ry
            ox, oy = qx - al * rx, qy - al * ry
            lx, ly = -math.sin(self.psi), math.cos(self.psi)
            w = clip((P.snap_start - self.d_gate) / 4.0, 0.0, 1.0)
            snap_des = clip(-w * (ox * lx + oy * ly), -P.snap_max, P.snap_max)
        self.o_snap += clip(snap_des - self.o_snap, -P.snap_rate * dt, P.snap_rate * dt)
        carrot = self._carrot_world()
        self.carrot = carrot
        rel = carrot - p
        hh = math.hypot(float(rel[0]), float(rel[1]))
        if hh > P.goal_max:
            rel[0] *= P.goal_max / hh
            rel[1] *= P.goal_max / hh
            self.goal_clips += 1
        rel[2] = P.goal_z if on_ground else clip(float(rel[2]), -P.goal_z, P.goal_z)
        return rel

    # ------------------------------------------------------------------ yaw and speed sense
    def _yaw(self, now: float, dt: float, psi_n: float, on_ground: bool) -> None:
        P = self.params
        look_des = 0.0
        if self.mode == 2 and self.g_s is not None:
            p = self.host_pos
            b = wrap(math.atan2(float(self.g_s[1] - p[1]), float(self.g_s[0] - p[0])) - self.psi)
            look_des = (math.copysign(1.0, b) * clip(abs(b) - P.look_free * DEG, 0.0, P.look_max * DEG)
                        * clip((self.d_gate - 4.0) / 4.0, 0.0, 1.0) * clip(1.0 - abs(self.kappa) / P.look_kappa, 0.0, 1.0))
        elif self.mode == 3:
            ts = now - self.mode_since
            look_des = P.sweep * DEG * clip(ts / P.sweep_ramp, 0.0, 1.0) * math.sin(TWO_PI * ts / P.sweep_period)
        self.look += (look_des - self.look) * min(1.0, dt / P.look_tau)
        look_rate = clip((self.look - self.look_prev) / dt, -1.0, 1.0) if dt > 0 else 0.0
        self.look_prev = self.look
        self.yaw_ref = wrap(self.psi + self.look)
        om = getattr(self.host, 'omega', None)
        if om is not None and dt > 0:
            rate = float((self.host.last_R @ np.asarray(om, dtype=np.float64))[2])     # world heading rate
            self.psi_dot += (rate - self.psi_dot) * min(1.0, dt / P.yaw_rate_tau)
        r_ref = self.v * self.kappa + look_rate
        e = wrap(self.yaw_ref - psi_n) + P.yaw_lead * (r_ref - self.psi_dot)
        db = P.yaw_db * DEG
        e_s = e * e * e / (e * e + db * db)
        r_cmd = clip(r_ref + P.yaw_gain * e_s, -P.yaw_rate_cap, P.yaw_rate_cap)
        y_des = clip(-r_cmd / P.yaw_rate, -P.yaw_max, P.yaw_max)
        self.sight_yaw += clip(y_des - self.sight_yaw, -P.yaw_slew * dt, P.yaw_slew * dt)
        if on_ground:
            self.sight_yaw = 0.0

    def _flow(self, now: float, dt: float, vel: np.ndarray) -> None:
        P = self.params
        lo, hi = P.flow_bounds()
        f_ff = P.flow_ref / max(self.v_nom, P.flow_ref)
        if self.t_air is not None and now - self.t_air > 3.0 and self.v_nom > 1.0 and abs(self.v_des - self.v_nom) < 0.2:
            vh = math.hypot(float(vel[0]), float(vel[1]))
            self.flow_trim = clip(self.flow_trim + dt * P.flow_ki * (vh - self.v_nom) / self.v_nom, *P.flow_trim_range)
        self.flow_f += (clip(f_ff * self.flow_trim, lo, hi) - self.flow_f) * min(1.0, dt / P.flow_tau)
        self.host.flow_gain = self.flow_f

    def _status(self, now: float) -> None:
        T = self.target
        if self.mode == 2 and T is not None:
            seen = now - T.t_last < 0.3
            s = (f'gate {"seen" if seen else "remembered"} {self.d_gate:.1f} m; target {T.id} hits {T.hits} '
                 f'sd_lat {self.sd_lat:.2f}; axis {math.degrees(self.n_ang):.0f}{" turn" if self.turn_gate else ""}')
        elif self.mode == 1:
            s = 'flying on'
        elif self.mode == 3:
            s = 'no gate: searching'
        else:
            s = 'on the ground'
        self.host.vision_status = (f'{s}; rabbit v {self.v:.1f}/{self.v_nom:.1f} lead {self.lead_now:.1f}; '
                                   f'passes {self.n_passes}; flow {self.flow_f:.2f}')

    @property
    def host_pos(self) -> np.ndarray:
        return self._p

    # ------------------------------------------------------------------ log
    def log_values(self, det=None, offset=(0.0, 0.0, 0.0)) -> list[float]:
        """Numeric values of LOG_COLUMNS (offset shifts world positions, e.g. to the rehearsal's gate frame)."""
        nan = math.nan
        ox, oy, oz = (float(x) for x in offset)
        du, dv, dtt = ((float(det.u), float(det.v), float(det.t)) if det is not None and det.frames else (nan, nan, nan))
        T = self.target
        now = self.t_prev if self.t_prev is not None else nan
        if T is not None:
            tgt = [T.id, float(T.m[0]) + ox, float(T.m[1]) + oy, float(T.m[2]) + oz, self.sd_lat, T.hits,
                   now - T.t_last, math.degrees(self.n_ang) if self.n_ang is not None else nan,
                   self.next.id if self.next is not None else -1]
        else:
            tgt = [-1, nan, nan, nan, nan, nan, nan, nan, -1]
        n_conf = sum(1 for t in self.tracks if t.confirmed and not t.passed)
        n_tent = sum(1 for t in self.tracks if not t.confirmed)
        return [du, dv, dtt,
                self.c[0] + ox, self.c[1] + oy, self.z_c + oz, math.degrees(self.psi), self.kappa, self.v, self.v_nom,
                math.degrees(self.look), math.degrees(self.yaw_ref), self.sight_yaw,
                *tgt,
                n_conf, n_tent, self.mode, self.n_passes, self.pass_kind,
                self.flow_f, self.rej_elev, self.rej_stale, self.absorbed, self.low, self.reseeds,
                self.ghosts, self.unpasses, self.behind, self.goal_clips, self.errors]

    @staticmethod
    def empty_log_values(det=None) -> list[float]:
        nan = math.nan
        du, dv, dtt = ((float(det.u), float(det.v), float(det.t)) if det is not None and det.frames else (nan, nan, nan))
        return [du, dv, dtt] + [nan] * (len(LOG_COLUMNS) - 3)


# ---------------------------------------------------------------------------------------------------- CLI helpers

def add_cli_args(q, yaw_rate_default: float = 2.3) -> None:
    """The --sight* flags, shared by `liftoff fly` and `vision rehearse`."""
    g = q.add_argument_group('by-sight pilot (--sight rabbit)')
    g.add_argument('--sight', choices=['legacy', 'rabbit'], default='legacy',
                   help='by-sight pilot: legacy (remembered gate + carrot line) or rabbit (tracks + virtual lead vehicle)')
    g.add_argument('--sight-speed', type=float, default=2.5, help='rabbit cruise speed V_CRUISE (m/s); above flow_ref '
                   'the speed sense is scaled by flow_ref/speed (4 -> about 0.6)')
    g.add_argument('--sight-gate-speed', type=float, default=3.0, help='speed through a straight gate (m/s)')
    g.add_argument('--sight-turn-gate-speed', type=float, default=2.5, help='speed through a gate on a turn (m/s)')
    g.add_argument('--sight-lead', type=float, default=3.0, help='rabbit lead ahead of the drone at the gate (m)')
    g.add_argument('--sight-a-lat', type=float, default=1.2, help='rabbit lateral acceleration limit (m/s^2)')
    g.add_argument('--sight-a-brk', type=float, default=1.2, help='rabbit braking limit (m/s^2)')
    g.add_argument('--sight-yaw-rate', type=float, default=yaw_rate_default,
                   help='yaw rate per unit yaw stick (rad/s; Liftoff 2.3, the simulator 3.8)')
    g.add_argument('--sight-yaw-gain', type=float, default=1.5, help='heading error gain (1/s)')
    g.add_argument('--sight-yaw-lead', type=float, default=0.15,
                   help='phase lead on the heading error from the measured heading rate (s)')
    g.add_argument('--sight-yaw-max', type=float, default=0.35, help='yaw stick cap')
    g.add_argument('--sight-flow-ref', type=float, default=2.45, help='sensed cruise speed the brain holds (m/s)')
    g.add_argument('--sight-flow-min', type=float, default=None,
                   help='lowest speed-sense gain (default min(0.7, 0.9 flow_ref / max(speed, gate speed))); '
                        'the highest is --flow-gain')
    g.add_argument('--sight-flow', choices=['global', 'along'], default='global',
                   help='scale the whole horizontal speed sense, or only its part along the rabbit heading')
    g.add_argument('--sight-flow-alt', choices=['start', 'ground'], default='start',
                   help='optic-flow altitude: above the reset point (as trained), or above the ground under the '
                        'altitude reference (hills)')
    g.add_argument('--sight-search-side', choices=['left', 'right'], default='left',
                   help='search turn before the course has turned')
    g.add_argument('--sight-search-radius', type=float, default=8.0, help='search circle radius (m)')
    g.add_argument('--sight-elev', default='-40,35',
                   help='world elevation window of sightings, deg (write --sight-elev=-40,35)')
    g.add_argument('--vision-lag', type=float, default=0.0, help='s subtracted from the detection grab time')
    g.add_argument('--sight-range-corr', default='default',
                   help="range correction table 'd:k,d:k,...', 'default' (measured on GateNet), 'spec' (the prototype's) "
                        "or 'none'")
    g.add_argument('--sight-z-aim', type=float, default=0.3, help='fly this far above the passage point (m)')
    g.add_argument('--sight-climb-front', type=float, default=0.5, help='climbs finish by this fraction of the time to go')
    g.add_argument('--sight-snap-start', type=float, default=9.0,
                   help='pull the goal onto the gate bearing within this distance (m)')
    g.add_argument('--sight-turn-hints', default='', help='heading change at each gate, deg (+ left), e.g. 5,40,90')
    g.add_argument('--sight-set', action='append', default=[], metavar='NAME=VALUE',
                   help='override any SightParams field, e.g. --sight-set v_blind=2.5 --sight-set lead_open=1')


def params_from_args(a, flow_gain: float = 1.0) -> SightParams:
    import yaml
    P = SightParams(v_cruise=a.sight_speed, v_gate=a.sight_gate_speed, v_gate_turn=a.sight_turn_gate_speed,
                    lead=a.sight_lead, a_lat=a.sight_a_lat, a_brk=a.sight_a_brk, yaw_rate=a.sight_yaw_rate,
                    yaw_gain=a.sight_yaw_gain, yaw_lead=a.sight_yaw_lead, yaw_max=a.sight_yaw_max, flow_ref=a.sight_flow_ref,
                    flow_min=a.sight_flow_min, flow_max=flow_gain, flow_mode=a.sight_flow, flow_alt=a.sight_flow_alt,
                    search_side=1.0 if a.sight_search_side == 'left' else -1.0, search_radius=a.sight_search_radius,
                    vision_lag=a.vision_lag, z_aim=a.sight_z_aim, climb_front=a.sight_climb_front,
                    snap_start=a.sight_snap_start)
    P.search_radius_wide = 1.5 * P.search_radius
    lo, hi = (float(x) for x in str(a.sight_elev).split(','))
    P.elev_deg = (lo, hi)
    rc = str(a.sight_range_corr).strip().lower()
    if rc == 'none':
        P.range_corr = None
    elif rc == 'spec':
        P.range_corr = RANGE_CORR_SPEC
    elif rc != 'default':
        P.range_corr = tuple(tuple(float(y) for y in x.split(':')) for x in rc.split(','))
    if a.sight_turn_hints:
        P.turn_hints = tuple(float(x) for x in a.sight_turn_hints.split(','))
    names = {f.name: f for f in fields(SightParams)}
    for kv in a.sight_set:
        k, _, v = kv.partition('=')
        k = k.strip()
        if k not in names:
            raise SystemExit(f'--sight-set: SightParams has no field {k!r}')
        val = yaml.safe_load(v)
        if isinstance(val, list):
            val = tuple(tuple(x) if isinstance(x, list) else x for x in val)
        setattr(P, k, val)
    return P
