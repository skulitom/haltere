"""Run the trained brain on live Liftoff telemetry and produce stick commands."""
from __future__ import annotations

import time
from bisect import bisect_left
from dataclasses import dataclass, field
from operator import itemgetter

import numpy as np

IN_W_PX = 320                              # GateNet input width: detections' pixel units
import torch

from ..sim.tasks import HoverTaskConfig, observe_from_sensors
from .frames import omega_from_quats, quat_wxyz_to_mat, unity_quat_to_sim, unity_vec_to_sim, yaw_of
from .telemetry import TelemetryFrame


@dataclass
class LiftoffMapping:
    """How Liftoff's conventions relate to the simulator's (filled in by ``haltere liftoff fit``)."""
    stick_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)   # multiply brain (roll, pitch, yaw) before sending
    gyro_axis: tuple[int, int, int] = (1, 0, 2)                # telemetry gyro index for body x, y, z (roll, pitch, yaw)
    gyro_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_quat_rates: bool = True                                # derive body rates from attitude instead of Gyro
    max_rpm: float = 30000.0
    throttle_scale: float = 1.0                                # gain around the hover point
    hover_stick_sim: float | None = None                       # brain's throttle stick at hover (from its training physics)
    hover_stick_game: float | None = None                      # Liftoff's raw throttle stick at hover (measured by autotest)
    hover_processed_game: float | None = None                  # Liftoff's processed throttle input at hover
    stick_curves: object = None                                # StickCurves: undo Liftoff's deadband/expo per axis
    stick_model: object = None                                 # RadialSticks: undo Liftoff's per-stick radial deadzone
    notes: dict = field(default_factory=dict)

    def processed_throttle(self, a: float) -> float | None:
        """Brain throttle stick -> the processed throttle input the game should see (None when unknown)."""
        if self.hover_stick_sim is None or self.hover_processed_game is None:
            return None
        return self.hover_processed_game + self.throttle_scale * (a - self.hover_stick_sim)

    def to_raw(self, cmd: np.ndarray) -> np.ndarray:
        """Brain commands [throttle, roll, pitch, yaw] -> raw pad axes [throttle, roll, pitch, yaw].

        With the radial stick model, each physical stick (throttle+yaw, roll+pitch) is inverted as one 2D vector,
        which is how the game applies its deadzone: inverting the axes one at a time over-drives both whenever both
        are active (a 0.05 roll correction during a 0.3 pitch cruise arrived as 0.21)."""
        roll = float(cmd[1]) * self.stick_sign[0]
        pitch = float(cmd[2]) * self.stick_sign[1]
        yaw = float(cmd[3]) * self.stick_sign[2]
        thr_p = self.processed_throttle(float(cmd[0]))
        if self.stick_model is not None and thr_p is not None:
            thr, yaw_r = self.stick_model.raw_for('throttle', 'yaw', thr_p, yaw)
            roll_r, pitch_r = self.stick_model.raw_for('roll', 'pitch', roll, pitch)
            return np.array([thr, roll_r, pitch_r, yaw_r])
        sticks = np.array([self.map_throttle(float(cmd[0])), self.map_axis('roll', roll),
                           self.map_axis('pitch', pitch), self.map_axis('yaw', yaw)])
        # the game normalises each stick to the unit circle: keep (throttle, yaw) and (roll, pitch) inside it
        for i, j in ((0, 3), (1, 2)):
            n = float(np.hypot(sticks[i], sticks[j]))
            if n > 0.97:
                sticks[i] *= 0.97 / n
                sticks[j] *= 0.97 / n
        return sticks

    def map_throttle(self, a: float) -> float:
        """Brain throttle stick -> raw game throttle stick: same hover point, scaled deviation, through the
        game's throttle curve when it is known."""
        if self.hover_stick_sim is None:
            return a * self.throttle_scale
        if self.stick_curves is not None and 'throttle' in self.stick_curves.inv and self.hover_processed_game is not None:
            target = self.hover_processed_game + self.throttle_scale * (a - self.hover_stick_sim)
            return self.stick_curves.raw_for('throttle', target)
        if self.hover_stick_game is None:
            return a * self.throttle_scale
        return self.hover_stick_game + self.throttle_scale * (a - self.hover_stick_sim)

    def map_axis(self, ax: str, value: float) -> float:
        """Brain roll/pitch/yaw stick (the processed input the FC should see) -> raw pad axis."""
        if self.stick_curves is None:
            return value
        return self.stick_curves.raw_for(ax, value)


def pattern_target(name: str, t: float, offset: np.ndarray, radius: float, period: float,
                   amplitude: float, ramp: float = 4.0) -> np.ndarray:
    """Moving targets for freestyle-style flights (sim frame, relative to the reset point).

    orbit: circle of `radius` around the offset point, one lap per `period`.
    climbdive: sweep back and forth over `radius` m while the altitude swings by +-`amplitude`.
    figure8: a figure of eight of `radius`, one figure per `period`.
    The pattern fades in over `ramp` seconds after take-off."""
    g = min(1.0, max(0.0, (t - 3.0) / ramp))       # fade in after take-off
    w = 2 * np.pi * (t / period)
    if name == 'orbit':
        d = np.array([radius * np.cos(w) - radius, radius * np.sin(w), 0.0])
    elif name == 'climbdive':
        d = np.array([radius * np.sin(w), 0.0, amplitude * np.sin(2 * w)])
    elif name == 'figure8':
        d = np.array([radius * np.sin(w), radius * np.sin(2 * w) * 0.5, 0.0])
    else:
        raise ValueError(f'unknown pattern {name!r}')
    return offset + g * d


class TelemetryPilot:
    def __init__(self, brain, task_cfg: HoverTaskConfig, mapping: LiftoffMapping, device,
                 offset=(0.0, 0.0, 2.0), waypoints: list | None = None, dwell: float = 4.0,
                 advance_radius: float = 0.0, loop: bool = True, pattern: str = '', radius: float = 3.0,
                 period: float = 12.0, amplitude: float = 1.5, stick_gain: float = 1.0, stick_lpf: float = 0.0,
                 face_gain: float = 0.0, face_max: float = 0.25):
        self.brain = brain
        self.cfg = task_cfg
        self.map = mapping
        self.device = device
        self.offset = np.asarray(offset, dtype=np.float64)
        self.waypoints = [np.asarray(w, dtype=np.float64) for w in (waypoints or [])]
        self.dwell = dwell
        self.advance_radius = advance_radius   # > 0: advance to the next waypoint when this close (racing)
        self.loop = loop
        self.pattern = pattern
        self.radius, self.period, self.amplitude = radius, period, amplitude
        # scales roll/pitch/yaw commands (smoothness / latency margin): one number or one per axis
        self.stick_gain = np.array(stick_gain if np.ndim(stick_gain) else [stick_gain] * 3, dtype=np.float64)
        self.stick_lpf = stick_lpf             # s; low-pass on the sticks sent to the game (0 = off)
        # > 0: yaw the nose toward the target (stick per radian of heading error, capped at face_max). The brain
        # has no camera and no heading objective, so on its own it flies sideways; this keeps the FPV view
        # looking along the path. Positive yaw stick = nose right (Betaflight convention, verified in the sim).
        self.face_gain, self.face_max = face_gain, face_max
        # wall clock of the by-sight logic (detection ages, fly-on/creep/search timers, the facing wobble); the
        # simulator rehearsal (haltere.vision.rehearse) swaps in its simulated clock
        self.clock = time.time
        self.face_ahead = 0.0                  # m; path mode: face the path this far beyond the carrot (0 = face the carrot)
        self.face_target = None
        self.face_wobble_deg = 0.0             # > 0: sweep the facing heading +-this many degrees (sinusoidal)
        self.face_wobble_period = 10.0         # s; so datasets see the gates all over the image, not only centred
        self._wobble_t0 = None
        # vision: a GateVision object supplies the goal instead of telemetry positions (see haltere.vision.runtime)
        self.vision = None
        self.vision_thresh = 0.5
        self.vision_stale = 0.5                # s; older detections are not trusted
        self.vision_gate_w = None              # remembered position of the gate last seen (world, sim frame)
        self.vision_lookahead = 2.0            # m; the carrot runs this far ahead of the progress along the line
        self.vision_speed = 2.0                # m/s; how fast the carrot advances toward the gate
        self._cand_hist = []                   # (time, gate candidate, drone position) of recent sightings
        self._sightings = []                   # (time, apparent width) of recent plausible detections
        self._last_wide_t = None               # when the arch last filled the view (about to fly through)
        self._passed = []                      # (time, world position) of gates already passed: an arch looks the same from behind
        self._rays = []                        # (time, origin, world direction) of recent sightings, for triangulation
        self._gate_anchor = (np.zeros(3), 3.0) # where the believed gate was first confirmed, and the allowed drift
        self._agree_t = 0.0                    # last time a sighting agreed with the remembered gate
        self._line = None                      # (start, gate) of the straight line the carrot runs along
        self._line_s = 0.0                     # progress along that line (m)
        self._last_goal_t = None
        self._z_ref = None                     # altitude to hold while no gate is in sight
        self._hold_w = None                    # position to hold while no gate is in sight
        self._no_gate_since = None             # when the drone last lost sight of every gate (search yaw after a while)
        self.vision_search_yaw = -0.2          # yaw stick while searching (negative = nose turns left)
        self.vision_fly_on = 4.0               # s to keep flying straight after passing a gate, then look around
        self.vision_creep = 8.0                # s to creep ahead sweeping the view after that, before hovering and turning
        self._yaw_override = None
        self.vision_z_min, self.vision_z_max = 1.4, 9.0   # m above the start: the altitude band flown by sight (gates sit on hills too)
        self.last_vel = np.zeros(3)
        self.vision_passed_t = None            # when the remembered gate was passed (fly on for a moment)
        self.vision_status = 'no vision'
        self.flow_gain = 1.0                   # scale on the horizontal speed the brain senses (< 1: it flies faster)
        self.vision_goal_max = 0.0             # m; > 0: clip the horizontal by-sight goal to this length (legacy pilot)
        self.follower = None                   # PathFollower: projection, speed profile, tangent control frame (lap mode)
        self.follow_frame = True               # give the brain the line's tangent as its control frame
        self.follow_line_alt = True            # optic flow scaled by the height above the taught line, not above the start
        self.follow_info = None
        self.frame_delta = 0.0
        self.prev_step_t = None
        self._pose_hist = []                   # (wall time, position, attitude) of the last second of telemetry
        # which by-sight pilot runs vision_goal: 'legacy' (below) or 'rabbit' (haltere.liftoff.sightpilot: gate tracks
        # and a virtual lead vehicle; it also commands the yaw stick and schedules flow_gain)
        self.sight = 'legacy'
        self.sight_params = None               # SightParams for the rabbit (None: defaults)
        self.sightpilot = None
        self._gate_clearance = None            # m, measured on this course: the first gate's height above the
                                               # terrain it stands on (see sensors(); None until a gate is flown)
        self.path_speed = 0.0                  # > 0: follow the waypoint polyline as a moving target at this speed
        self.path_lookahead = 1.5              # m ahead of the drone's progress along the path
        self.path_z_lead = None                # m; the carrot's height is taken this far ahead (None: at the carrot)
        self.path_cross_track_gain = 0.0       # extra lateral goal correction, bounded to 0.8 m
        self.path_cross_track_damping = 0.0    # damp sideways velocity before crossing the line
        self.path_cross_track_integral = 0.0   # slowly cancel persistent vehicle tracking bias
        self.set_waypoints(self.waypoints)
        self.W = brain.weight_matrix().detach()
        self.reset(None)

    def set_waypoints(self, waypoints) -> None:
        self.waypoints = [np.asarray(w, dtype=np.float64) for w in waypoints]
        if self.waypoints:
            P = np.stack(self.waypoints)
            self.path_pts = np.vstack([P, P[:1]]) if self.loop else P
            self.path_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.path_pts, axis=0), axis=1))]

    def path_point(self, s: float) -> np.ndarray:
        """Point on the waypoint polyline, clamped at the final point for an open route."""
        total = self.path_s[-1]
        if total <= 0:
            return self.path_pts[0].copy()
        s = s % total if self.loop else np.clip(s, 0, total)
        i = int(np.searchsorted(self.path_s, s, side='right') - 1)
        i = min(max(i, 0), len(self.path_pts) - 2)
        f = (s - self.path_s[i]) / max(self.path_s[i + 1] - self.path_s[i], 1e-6)
        return self.path_pts[i] + f * (self.path_pts[i + 1] - self.path_pts[i])

    def reset(self, frame: TelemetryFrame | None) -> None:
        self.state = self.brain.init_state(1)
        self.pos0 = None if frame is None else unity_vec_to_sim(frame.position)
        self.prev_quat = None
        self.prev_t = None
        self.last_timestamp = -1.0
        self.t_start = None if frame is None else frame.timestamp
        self.omega = np.zeros(3)
        self.wp_index = 0
        self.wp_since = None
        self.last_pos = np.zeros(3)
        self.filtered = None
        self.path_progress = 0.0
        self._path_lateral_bias = 0.0
        self._path_airborne = False
        self._last_t = None
        self.vision_gate_w = None
        self.vision_passed_t = None
        self._cand_hist = []
        self._sightings = []
        self._last_wide_t = None
        self._rays = []
        self._passed = []
        self._rays = []
        self._line = None
        self._line_s = 0.0
        self._last_goal_t = None
        self._z_ref = None
        self._hold_w = None
        self._no_gate_since = None
        self.prev_step_t = None
        self._pose_hist = []                   # poses of the previous reset frame would place sightings wrongly
        self._yaw_override = None
        if getattr(self, 'follower', None) is not None:
            self.follower.s_p = None
            self.follower.flow = 1.0
        if self.sightpilot is not None:
            self.sightpilot.reset()

    @property
    def rabbit(self) -> bool:
        """The rabbit by-sight pilot is flying (vision on and sight == 'rabbit')."""
        return self.vision is not None and self.sight == 'rabbit'

    def target_at(self, t: float) -> np.ndarray:
        if self.pattern:
            return pattern_target(self.pattern, t, self.offset, self.radius, self.period, self.amplitude)
        if not self.waypoints:
            return self.offset
        if self.path_speed > 0:
            # Height is relative to launch, not terrain. After takeoff a valley
            # can legitimately put the route below zero without grounding us.
            self._path_airborne = self._path_airborne or self.last_pos[2] >= .3
            # progress along the path only as fast as the drone keeps up: the carrot slows down smoothly as the
            # drone falls behind it (a hard stop/go gate excited a ~0.5 Hz pitch oscillation in Liftoff)
            dt = 0.0 if self._last_t is None else float(np.clip(t - self._last_t, 0.0, 0.05))
            self._last_t = t
            carrot = self.path_point(self.path_progress + self.path_lookahead)
            if self.path_z_lead is not None:
                # the height comes from just ahead: a carrot far up a climbing line pulls the drone above the line, into
                # the top of a round arch sitting on the slope
                carrot[2] = self.path_point(self.path_progress + min(self.path_z_lead, self.path_lookahead))[2]
            gap = float(np.linalg.norm(self.last_pos - carrot)) - self.path_lookahead
            if (self.path_cross_track_gain > 0 or self.path_cross_track_damping > 0
                    or self.path_cross_track_integral > 0):
                # Correct sideways bias without increasing the forward or vertical goal.
                # A moving target alone can settle beside the line, clipping nearby trees.
                centre = self.path_point(self.path_progress)
                tangent = (self.path_point(self.path_progress + .25)
                           - self.path_point(self.path_progress - .25))[:2]
                length = float(np.linalg.norm(tangent))
                if length > 1e-6:
                    normal = np.array([-tangent[1], tangent[0]]) / length
                    error = float(np.dot(self.last_pos[:2] - centre[:2], normal))
                    drift = float(np.dot(self.last_vel[:2], normal))
                    correction = (self.path_cross_track_gain * error
                                  + self.path_cross_track_damping * drift + self._path_lateral_bias)
                    # Freeze integration when saturated in the same direction, or grounded.
                    if self._path_airborne and (abs(correction) < .8 or error * correction < 0):
                        self._path_lateral_bias = float(np.clip(
                            self._path_lateral_bias + self.path_cross_track_integral * error * dt, -.8, .8))
                        correction = (self.path_cross_track_gain * error
                                      + self.path_cross_track_damping * drift + self._path_lateral_bias)
                    carrot[:2] -= normal * np.clip(correction, -.8, .8)
            keep_up = float(np.clip(1.0 - gap / 1.5, 0.0, 1.0))
            if not self._path_airborne:   # hold only until the first takeoff after reset
                keep_up = 0.0
            self.path_progress += self.path_speed * keep_up * dt
            if not self.loop:
                self.path_progress = min(self.path_progress, self.path_s[-1])
            if self.face_ahead > 0:
                self.face_target = self.path_point(self.path_progress + self.path_lookahead + self.face_ahead)
            progress = self.path_progress % self.path_s[-1] if self.loop and self.path_s[-1] > 0 else self.path_progress
            self.wp_index = min(len(self.waypoints) - 1, int(np.searchsorted(self.path_s, progress, side='right') - 1))
            return carrot
        if self.advance_radius > 0:
            if self.wp_index < len(self.waypoints):
                d = np.linalg.norm(self.last_pos - self.waypoints[self.wp_index])
                if d < self.advance_radius:
                    self.wp_index += 1
                    if self.wp_index >= len(self.waypoints):
                        self.wp_index = 0 if self.loop else len(self.waypoints) - 1
            return self.waypoints[self.wp_index]
        i = int(t // self.dwell)
        i = i % len(self.waypoints) if self.loop else min(i, len(self.waypoints) - 1)
        self.wp_index = i
        return self.waypoints[i]

    def vision_goal(self, pos_w: np.ndarray) -> np.ndarray:
        """Body-frame goal vector from the gate detector.

        A sighting is a world ray (from the drone through the detected centre) plus a rough range from the
        apparent width. The remembered gate sits on the LATEST ray, at a range smoothed over the sightings: the
        bearing is the accurate part of a detection and must not lag, the range is the rough part and may.
        A sighting agrees with the memory when its ray passes within 2 m (or 15% of the range) of it; two
        agreeing sightings establish a gate, and sightings that disagree for a while replace it. The goal is a
        carrot along the straight line from where the approach began to the gate. After passing a gate the drone
        flies on along its nose for a moment, then creeps ahead sweeping its view, then hovers and turns.

        With ``sight == 'rabbit'`` the goal comes from ``SightPilot`` instead (haltere.liftoff.sightpilot)."""
        if self.sight == 'rabbit':
            if self.sightpilot is None:
                from .sightpilot import SightPilot
                self.sightpilot = SightPilot(self, self.sight_params)
            return self.sightpilot.goal(pos_w)
        det = self.vision.get()
        now = self.clock()
        dt = 0.0 if self._last_goal_t is None else float(np.clip(now - self._last_goal_t, 0.0, 0.1))
        self._last_goal_t = now
        R = self.last_R
        self._yaw_override = None
        d = det.direction_body
        elevation = float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))
        plausible = (det.p_visible >= self.vision_thresh and now - det.t < self.vision_stale and det.dist_m > 0.5
                     and -35.0 < elevation < 25.0)         # gates are near the ground, never up in the sky
        self._passed = [(t, g) for t, g in self._passed if now - t < 60.0]
        ray = None
        if plausible:
            dw = R @ d
            rng = float(np.clip(det.dist_m, 1.0, 40.0))
            if any(np.linalg.norm(pos_w + dw * rng - g) < 6.0 for _, g in self._passed):
                plausible = False                    # the arch just flown through, seen from behind
            else:
                # two arches often line up (the next gate shows small right behind the first) and the detector
                # flips between them: only trust a sighting that is not much narrower than the widest of the
                # last moment, i.e. follow the nearest arch
                self._sightings = [(t, w) for t, w in self._sightings if now - t < 0.7] + [(now, det.width_px)]
                if det.width_px >= 0.75 * max(w for _, w in self._sightings):
                    if det.width_px >= 0.35 * IN_W_PX:
                        self._last_wide_t = now      # the arch fills the view: we are within a few metres of it
                    # the apparent width misjudges the range of an arch seen obliquely; when the drone's motion
                    # has opened a few degrees of parallax, the sighting rays' intersection is a better range
                    self._rays = [(t, o, r) for t, o, r in self._rays if now - t < 3.0] + [(now, pos_w.copy(), dw)]
                    if len(self._rays) >= 3:
                        origins = np.array([o for _, o, _ in self._rays])
                        dirs = np.array([r for _, _, r in self._rays])
                        spread = float(np.degrees(np.arccos(np.clip((dirs @ dw).min(), -1.0, 1.0))))
                        if spread >= 6.0 and np.linalg.norm(origins - origins[-1], axis=1).max() >= 2.5:
                            from ..vision.triangulate import intersect_rays
                            pt, rms = intersect_rays(origins, dirs)
                            rel = pt - pos_w
                            r_par = float(np.linalg.norm(rel))
                            if rms < 1.0 and 1.0 < r_par < 40.0 and rel @ dw > 0 and 0.5 < r_par / rng < 2.0:
                                rng = 0.5 * rng + 0.5 * r_par
                    ray = (dw, rng)

        def near(point, dw, rng):
            """Does the sighting ray pass close to the point (which must lie ahead along the ray)?"""
            rel = point - pos_w
            along = float(rel @ dw)
            perp = float(np.linalg.norm(rel - along * dw))
            return along > 0.5 and perp < max(2.0, 0.15 * max(along, rng))

        gate = self.vision_gate_w
        if ray is not None:
            dw, rng = ray
            if gate is not None and near(gate, dw, rng):
                r_mem = float(np.linalg.norm(gate - pos_w))
                w = 0.6 if r_mem < 8.0 else 0.35                       # close up the sighting's range is good too
                self.vision_gate_w = pos_w + dw * ((1.0 - w) * r_mem + w * rng)
                self._agree_t = now
                self._cand_hist = []
            else:
                # a sighting that does not fit the memory (or there is none): two of them agreeing with each
                # other a moment apart make a gate; a phantom that rides along with the drone does not stay put
                self._cand_hist = [(t, c) for t, c in self._cand_hist if now - t < 1.5]
                match = next((t for t, c in self._cand_hist if now - t >= 0.3 and near(c, dw, rng)), None)
                self._cand_hist.append((now, pos_w + dw * rng))
                far_from_memory = gate is None or np.linalg.norm(gate - pos_w) > 5.0
                if match is not None and far_from_memory and (gate is None or now - self._agree_t > 0.7):
                    self.vision_gate_w = pos_w + dw * rng
                    self._agree_t = now
                    self._line = (pos_w.copy(), self.vision_gate_w.copy())
                    self._line_s = 0.0
                    self.vision_passed_t = None
                    self._cand_hist = []
        if self.vision_gate_w is not None:
            gate = self.vision_gate_w
            to_gate_b = R.T @ (gate - pos_w)
            dist_gate = float(np.linalg.norm(gate - pos_w))
            flew_through = (ray is None and self._last_wide_t is not None and 0.7 < now - self._last_wide_t < 2.5
                            and now - self._agree_t > 0.7)
            if to_gate_b[0] < -0.5 or flew_through:                            # the gate is behind: passed it
                self._passed.append((now, gate.copy()))
                self._last_wide_t = None
                self.vision_gate_w = None
                self._line = None
                self.vision_passed_t = now
            else:
                entry = None
                if self._passed:
                    course = gate - self._passed[-1][1]
                    course[2] = 0.0
                    if np.linalg.norm(course) > 3.0:
                        course /= np.linalg.norm(course)
                        entry = gate - 5.0 * course              # 5 m before the gate, on the course line through it
                if entry is not None and dist_gate > 8.0 and (pos_w - gate) @ (entry - gate) > 0:
                    # a gate on a turn: fly to the entry point first, then through the gate along the course
                    start = entry
                    line = gate - start
                    L = float(np.linalg.norm(line))
                    u = line / max(L, 1e-6)
                    along_drone = float((pos_w - start) @ u)
                    if along_drone < -1.0 or np.linalg.norm((pos_w - start) - along_drone * u) > 2.5:
                        carrot = entry.copy()                    # not on the line yet: head for its start
                        self._line_s = 0.0
                    else:
                        self._line_s = max(self._line_s, along_drone)
                        carrot = start + u * min(self._line_s + self.vision_lookahead, L + 4.0)
                    self._line = (start, gate.copy())
                    L_run = L + 4.0
                else:
                    if self._line is None:
                        self._line = (pos_w.copy(), gate.copy())
                        self._line_s = 0.0
                    start, _ = self._line
                    if dist_gate > 12.0 and np.linalg.norm(pos_w - start) > 6.0:
                        # far from the gate the line is re-anchored every few metres so it always runs straight from
                        # near the drone to the gate; inside 12 m it stays put so the final approach is straight
                        self._line = (pos_w.copy(), gate.copy())
                        self._line_s = 0.0
                        start = pos_w.copy()
                    self._line = (start, gate.copy())
                    line = gate - start
                    L = float(np.linalg.norm(line))
                    u = line / max(L, 1e-6)
                    L_run = L + 4.0                  # the carrot runs past the estimate; passing is judged by the gate falling behind
                    carrot = start + u * min(self._line_s + self.vision_lookahead, L_run)
                if self._line_s >= L_run - 0.5:                                   # ran out of line without passing it
                    self._passed.append((now, gate.copy()))
                    self.vision_gate_w = None
                    self._line = None
                    self.vision_passed_t = now
                gap = float(np.linalg.norm(pos_w - carrot)) - self.vision_lookahead
                keep_up = float(np.clip(1.0 - gap / 1.5, 0.0, 1.0))
                self._line_s = min(self._line_s + self.vision_speed * keep_up * dt, L_run)
                rel_w = carrot - pos_w
                # altitude: the visual centre sits 1.5 m above the line through the arch; aim 1.2 m below it. The
                # height of a far sighting is rough, so the band sits around the last gate passed (or the start)
                # and climbs above it only on close sightings (the sixth gate of this lap is 5 m up a hill)
                z_ref = self._passed[-1][1][2] - 1.2 if self._passed else 1.4
                z_lo = max(self.vision_z_min, z_ref - 1.0)
                z_hi = min(self.vision_z_max, z_ref + (5.0 if dist_gate < 8.0 else 1.5))
                z_goal = float(np.clip(gate[2] - 1.2, z_lo, z_hi))
                rel_w[2] = float(np.clip(z_goal - pos_w[2], -1.5, 1.0))
                self._z_ref = z_goal
                self._hold_w = None
                self._no_gate_since = None
                speed = float(np.linalg.norm(self.last_vel))
                self.vision_status = (f'gate {"seen" if plausible else "remembered"} {dist_gate:.1f} m, '
                                      f'carrot {self._line_s:.1f}/{L:.1f} m, speed {speed:.1f} m/s'
                                      + (f' p={det.p_visible:.2f}' if plausible else ''))
                return R.T @ rel_w
        if self._z_ref is None:
            self._z_ref = float(np.clip(pos_w[2], self.vision_z_min, self.vision_z_max))
        dz = float(np.clip(self._z_ref - pos_w[2], -2.0, 2.0))
        fwd = R @ np.array([1.0, 0.0, 0.0])
        fwd[2] = 0.0
        fwd /= max(float(np.linalg.norm(fwd)), 1e-6)
        if len(self._passed) >= 2:
            # the course direction at the last gate: the next gate is usually on from there, not along the nose
            course = self._passed[-1][1] - self._passed[-2][1]
            course[2] = 0.0
            if np.linalg.norm(course) > 3.0:
                fwd = course / np.linalg.norm(course)
        seen = f' (unconfirmed sighting p={det.p_visible:.2f})' if plausible else ''
        if self.vision_passed_t is not None and now - self.vision_passed_t < self.vision_fly_on:
            self._hold_w = None
            self.vision_status = 'flying on past the gate' + seen
            return R.T @ (2.0 * fwd + np.array([0.0, 0.0, dz]))     # straight ahead along the nose
        if self.vision_passed_t is not None and now - self.vision_passed_t < self.vision_fly_on + self.vision_creep:
            # creep ahead slowly while sweeping the view left and right: the next gate is usually somewhere ahead
            self._hold_w = None
            phase = now - self.vision_passed_t - self.vision_fly_on
            self._yaw_override = 0.2 * (1.0 if np.sin(2 * np.pi * phase / 6.0) >= 0 else -1.0) * np.sign(self.vision_search_yaw)
            self.vision_status = 'creeping ahead, sweeping' + seen
            return R.T @ (1.0 * fwd + np.array([0.0, 0.0, dz]))
        if self._hold_w is None:                     # hold the spot where the drone lost sight of the course
            self._hold_w = pos_w.copy()
            self._no_gate_since = now
        if now - self._no_gate_since > 2.0:
            self._yaw_override = self.vision_search_yaw
        rel_w = self._hold_w - pos_w
        rel_w[2] = dz
        self.vision_status = 'no gate: holding position' + (', searching' if now - self._no_gate_since > 2.0 else '') + seen
        return R.T @ rel_w

    def pose_at(self, t: float, tol: float = 0.015) -> tuple[np.ndarray, np.ndarray] | None:
        """Drone position and attitude matrix at wall time t from the last second of telemetry: the position
        interpolated between the two neighbouring frames, the attitude of the nearer one. None when t lies outside
        the history (by more than ``tol``, about one telemetry frame)."""
        h = self._pose_hist
        if not h or t < h[0][0] - tol or t > h[-1][0] + tol:
            return None
        i = bisect_left(h, t, key=itemgetter(0))
        if i <= 0:
            return h[0][1].copy(), h[0][2]
        if i >= len(h):
            return h[-1][1].copy(), h[-1][2]
        t0, p0, R0 = h[i - 1]
        t1, p1, R1 = h[i]
        f = (t - t0) / max(t1 - t0, 1e-9)
        return p0 + f * (p1 - p0), (R0 if f < 0.5 else R1)

    def rates(self) -> np.ndarray | None:
        """Current firing rates of all neurons (for the live recorder)."""
        if 'v' not in self.state:
            return None
        return (self.brain.cfg.rate_max * torch.sigmoid(self.state['v'][:, 0])).float().cpu().numpy()

    def gate_clearance(self, sp) -> float:
        """How high this course's gates stand above the ground under them, measured on the course being flown.

        Latched at the first gate actually flown through: the drone starts on the ground, so the reset datum is
        the terrain under the start, and the height at which it crosses the first gate is that gate's clearance
        above it. Until then ``z_pass0`` stands in as a prior. It used to stand in permanently, at Straw Bale's
        1.2 m, which put a measurement of one course inside the speed sense of every other.
        """
        if self._gate_clearance is None and getattr(sp, 'n_passes', 0) >= 1:
            self._gate_clearance = float(sp.z_pass_last)
        return sp.params.z_pass0 if self._gate_clearance is None else self._gate_clearance

    def sensors(self, fr: TelemetryFrame) -> dict[str, torch.Tensor]:
        if self.pos0 is None:
            self.reset(fr)
        q = unity_quat_to_sim(fr.attitude)
        pos = unity_vec_to_sim(fr.position) - self.pos0
        vel = unity_vec_to_sim(fr.velocity)
        R = quat_wxyz_to_mat(q)
        self.last_R = R
        self.last_vel = vel
        # the last second of poses, so a detection can be placed with the pose at its screen grab (vision runs ~15 fps,
        # its frames are 60-100 ms old by the time the pilot sees them)
        now = self.clock()
        self._pose_hist.append((now, pos.copy(), R.copy()))
        while self._pose_hist and now - self._pose_hist[0][0] > 1.0:
            self._pose_hist.pop(0)
        gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
        # the brain senses its horizontal speed through optic flow and airflow; like a fly in a flight arena whose
        # visual feedback gain is turned down, it flies faster when that sense reports less than the truth
        sp = self.sightpilot if self.rabbit and self.sightpilot is not None and self.sightpilot.ready else None
        altitude = pos[2]
        if sp is not None and sp.params.flow_mode == 'along':
            # only the speed along the rabbit's heading is scaled: the sensed drift direction stays true
            u = np.array([np.cos(sp.psi), np.sin(sp.psi), 0.0])
            along = float(vel @ u)
            vel = vel + (self.flow_gain - 1.0) * along * u
        else:
            vel = vel * np.array([self.flow_gain, self.flow_gain, 1.0])
        if sp is not None and sp.params.flow_alt == 'ground':
            # Height above the ground under the altitude reference, not above the reset point (13 m up the hill).
            # The ground is taken to be the gate height less the gates' clearance above the terrain, and that
            # clearance used to be z_pass0 = 1.2 m - a measurement of Straw Bale, baked into the SPEED sense of
            # every course. It biases the optic flow the brain flies on wherever gates sit differently above the
            # ground, which is not a navigation shortcut but a corrupted input. Measure it here instead: the drone
            # starts on the ground, so the reset datum is the terrain under the start, and the height at which it
            # crosses the first gate is that gate's clearance above it. z_pass0 survives only as the prior held
            # until the first gate has actually been flown through.
            clearance = self.gate_clearance(sp)
            altitude = float(np.clip(pos[2] - (sp.z_c - sp.params.z_aim - clearance), 0.3, 6.0))
        vel_body = R.T @ vel
        if self.map.use_quat_rates and self.prev_quat is not None and self.prev_t is not None:
            dt = max(fr.timestamp - self.prev_t, 1e-3)
            self.omega = 0.5 * self.omega + 0.5 * omega_from_quats(self.prev_quat, q, dt)
        elif not self.map.use_quat_rates:
            g = np.asarray(fr.gyro, dtype=np.float64)
            self.omega = np.deg2rad(np.array([g[i] for i in self.map.gyro_axis]) * np.asarray(self.map.gyro_sign))
        self.prev_quat, self.prev_t = q, fr.timestamp
        t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device)[None]  # noqa: E731
        return {
            'gyro': t(self.omega), 'gravity_body': t(gravity_body), 'vel_body': t(vel_body), 'vel_world': t(vel),
            'pos': t(pos), 'quat': t(q), 'up': t([R[2, 2]])[0], 'altitude': t([altitude]), 'yaw': t([yaw_of(q)]),
        }

    @torch.no_grad()
    def step(self, fr: TelemetryFrame) -> np.ndarray:
        """Return sticks [throttle, roll, pitch, yaw] in [-1, 1] for the virtual pad."""
        if fr.timestamp < self.last_timestamp - 0.5:  # the drone was reset in Liftoff
            self.reset(fr)
        self.last_timestamp = fr.timestamp
        s = self.sensors(fr)
        rel_b_t = None
        if self.vision is not None:
            rel_b = self.vision_goal(s['pos'][0].cpu().numpy())
            if self.vision_goal_max > 0 and not self.rabbit:
                # never hand the brain a goal far outside its training (a far entry carrot made it sprint and climb)
                h = float(np.hypot(rel_b[0], rel_b[1]))
                if h > self.vision_goal_max:
                    rel_b = rel_b * np.array([self.vision_goal_max / h, self.vision_goal_max / h, 1.0])
            rel_b_t = torch.as_tensor(rel_b, dtype=torch.float32, device=self.device)[None]
            target = torch.as_tensor(s['pos'][0].cpu().numpy() + self.last_R @ rel_b, dtype=torch.float32,
                                     device=self.device)[None]
        elif self.follower is not None:
            pos = s['pos'][0].cpu().numpy()
            dt = 0.0 if self.prev_step_t is None else float(np.clip(fr.timestamp - self.prev_step_t, 0.0, 0.05))
            info = self.follower.update(pos, self.last_vel, dt)
            self.follow_info = info
            if pos[2] < 0.3:                          # arming on the ground: aim just above the start
                info = dict(info, carrot=np.array([pos[0], pos[1], 1.2]))
            target = torch.as_tensor(info['carrot'], dtype=torch.float32, device=self.device)[None]
            self.flow_gain = info['flow']              # used by sensors() from the next frame on
            if self.face_ahead > 0:
                self.face_target = self.follower.point(info['s'] + info['d_c'] + self.face_ahead)
            if self.follow_line_alt:
                s['altitude'] = torch.full_like(s['altitude'], info['alt'])
            if self.follow_frame and pos[2] > 0.3:
                from .pathfollow import rotate_senses, wrap
                self.frame_delta = wrap(info['psi_c'] - float(s['yaw'].flatten()[0]))
                rel = self.last_R.T @ (info['carrot'] - pos)
                s, rel = rotate_senses(s, rel, self.frame_delta, info['psi_c'], torch)
                rel_b_t = torch.as_tensor(rel, dtype=torch.float32, device=self.device)[None]
            else:
                self.frame_delta = 0.0
        else:
            target = torch.as_tensor(self.target_at(fr.timestamp - (self.t_start or 0.0)), dtype=torch.float32,
                                     device=self.device)[None]
        self.prev_step_t = fr.timestamp
        rpm = np.asarray(fr.motor_rpm, dtype=np.float64)
        motor_mean = float(np.clip(rpm.mean() / self.map.max_rpm, 0, 1)) if rpm.size else 0.5
        obs = observe_from_sensors(s, target, torch.tensor([[motor_mean]], device=self.device), self.cfg, rel_b=rel_b_t)
        act, self.state, _ = self.brain(obs, self.state, self.W)
        a = act[0].cpu().numpy().astype(np.float64)
        self.last_brain = a.copy()
        self.last_rel_b = (rel_b_t[0].cpu().numpy() if rel_b_t is not None
                           else self.last_R.T @ (target[0].cpu().numpy() - s['pos'][0].cpu().numpy()))
        if self.follower is not None and self.vision is None and self.frame_delta:
            from .pathfollow import rotate_commands
            a[1], a[2] = rotate_commands(a[1], a[2], self.frame_delta)
        a[1:] *= self.stick_gain
        rabbit = self.rabbit
        if self.face_gain > 0 and not rabbit:
            if rel_b_t is not None and self.follower is None:
                rel_b = rel_b_t[0].cpu().numpy()
                err = float(np.arctan2(rel_b[1], rel_b[0])) if np.hypot(rel_b[0], rel_b[1]) > 0.8 else 0.0
            else:
                aim = self.face_target if self.face_target is not None else target[0].cpu().numpy()
                rel = aim - s['pos'][0].cpu().numpy()
                err = 0.0
                if np.hypot(rel[0], rel[1]) > 0.8:
                    err = np.angle(np.exp(1j * (np.arctan2(rel[1], rel[0]) - float(s['yaw'].flatten()[0]))))
            if self.face_wobble_deg > 0:
                now = self.clock()
                self._wobble_t0 = now if self._wobble_t0 is None else self._wobble_t0
                err += np.radians(self.face_wobble_deg) * np.sin(2 * np.pi * (now - self._wobble_t0) / self.face_wobble_period)
            a[3] = float(np.clip(-self.face_gain * err, -self.face_max, self.face_max))
        if self.vision is not None and self._yaw_override is not None and not rabbit:
            a[3] = float(self._yaw_override)            # searching: the by-sight logic steers the nose
        if rabbit:
            a[3] = float(self.sightpilot.sight_yaw) if self.sightpilot is not None else 0.0   # the nose follows the rabbit
        if self.stick_lpf > 0 and not rabbit:
            dt = max(fr.timestamp - self.prev_t, 1e-3) if self.prev_t is not None else 0.01
            alpha = min(1.0, dt / self.stick_lpf)
            self.filtered = a.copy() if self.filtered is None else self.filtered + alpha * (a - self.filtered)
            a = self.filtered
        self.last_cmd = a.copy()
        sticks = self.map.to_raw(a)
        self.last_obs = obs
        self.last_quat = s['quat'][0].cpu().numpy()
        self.last_target = target[0].cpu().numpy()
        self.last_pos = s['pos'][0].cpu().numpy()
        return np.clip(sticks, -1.0, 1.0)
