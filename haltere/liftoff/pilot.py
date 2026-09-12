"""Run the trained brain on live Liftoff telemetry and produce stick commands."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..sim.quad import G
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
    notes: dict = field(default_factory=dict)

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
        # vision: a GateVision object supplies the goal instead of telemetry positions (see haltere.vision.runtime)
        self.vision = None
        self.vision_thresh = 0.5
        self.vision_stale = 0.5                # s; older detections are not trusted
        self.vision_gate_w = None              # remembered position of the gate last seen (world, sim frame)
        self.vision_passed_t = None            # when the remembered gate was passed (fly on for a moment)
        self.vision_status = 'no vision'
        self.path_speed = 0.0                  # > 0: follow the waypoint polyline as a moving target at this speed
        self.path_lookahead = 1.5              # m ahead of the drone's progress along the path
        if self.waypoints:
            P = np.stack(self.waypoints)
            seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
            self.path_s = np.r_[0.0, np.cumsum(seg)]          # arc length at each vertex (closed loop)
            self.path_pts = np.vstack([P, P[:1]])
        self.W = brain.weight_matrix().detach()
        self.reset(None)

    def path_point(self, s: float) -> np.ndarray:
        """Point on the closed waypoint polyline at arc length s."""
        total = self.path_s[-1]
        s = s % total if self.loop else min(s, total)
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
        self._last_t = None
        self.vision_gate_w = None
        self.vision_passed_t = None

    def target_at(self, t: float) -> np.ndarray:
        if self.pattern:
            return pattern_target(self.pattern, t, self.offset, self.radius, self.period, self.amplitude)
        if not self.waypoints:
            return self.offset
        if self.path_speed > 0:
            # progress along the path only as fast as the drone keeps up: the carrot slows down smoothly as the
            # drone falls behind it (a hard stop/go gate excited a ~0.5 Hz pitch oscillation in Liftoff)
            dt = 0.0 if self._last_t is None else float(np.clip(t - self._last_t, 0.0, 0.05))
            self._last_t = t
            carrot = self.path_point(self.path_progress + self.path_lookahead)
            gap = float(np.linalg.norm(self.last_pos - carrot)) - self.path_lookahead
            keep_up = float(np.clip(1.0 - gap / 1.5, 0.0, 1.0))
            if self.last_pos[2] < 0.3:      # still on the ground (arming): hold the path
                keep_up = 0.0
            self.path_progress += self.path_speed * keep_up * dt
            self.wp_index = int(np.searchsorted(self.path_s, self.path_progress % self.path_s[-1], side='right') - 1)
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
        """Body-frame goal vector from the gate detector: the gate in view, else the remembered gate, else fly
        on briefly after passing a gate, else hover in place."""
        import time as _time
        det = self.vision.get()
        now = _time.time()
        R = self.last_R
        if det.p_visible >= self.vision_thresh and now - det.t < self.vision_stale and det.dist_m > 0.5:
            rel_b = det.direction_body * det.dist_m
            self.vision_gate_w = pos_w + R @ rel_b
            self.vision_passed_t = None
            self.vision_status = f'gate seen p={det.p_visible:.2f} {det.dist_m:.1f} m'
            return rel_b
        if self.vision_gate_w is not None:
            rel_b = R.T @ (self.vision_gate_w - pos_w)
            if rel_b[0] > -0.5 and np.linalg.norm(rel_b) > 0.5:
                self.vision_status = f'remembered gate {np.linalg.norm(rel_b):.1f} m'
                return rel_b
            self.vision_gate_w = None          # passed it: fly on a little so the next gate comes into view
            self.vision_passed_t = now
        if self.vision_passed_t is not None and now - self.vision_passed_t < 4.0:
            self.vision_status = 'flying on past the gate'
            return np.array([2.5, 0.0, 0.0])
        self.vision_status = 'no gate: hovering'
        return np.zeros(3)

    def rates(self) -> np.ndarray | None:
        """Current firing rates of all neurons (for the live recorder)."""
        if 'v' not in self.state:
            return None
        return (self.brain.cfg.rate_max * torch.sigmoid(self.state['v'][:, 0])).float().cpu().numpy()

    def sensors(self, fr: TelemetryFrame) -> dict[str, torch.Tensor]:
        if self.pos0 is None:
            self.reset(fr)
        q = unity_quat_to_sim(fr.attitude)
        pos = unity_vec_to_sim(fr.position) - self.pos0
        vel = unity_vec_to_sim(fr.velocity)
        R = quat_wxyz_to_mat(q)
        self.last_R = R
        gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
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
            'pos': t(pos), 'quat': t(q), 'up': t([R[2, 2]])[0], 'altitude': t([pos[2]]), 'yaw': t([yaw_of(q)]),
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
            rel_b_t = torch.as_tensor(rel_b, dtype=torch.float32, device=self.device)[None]
            target = torch.as_tensor(s['pos'][0].cpu().numpy() + self.last_R @ rel_b, dtype=torch.float32,
                                     device=self.device)[None]
        else:
            target = torch.as_tensor(self.target_at(fr.timestamp - (self.t_start or 0.0)), dtype=torch.float32,
                                     device=self.device)[None]
        rpm = np.asarray(fr.motor_rpm, dtype=np.float64)
        motor_mean = float(np.clip(rpm.mean() / self.map.max_rpm, 0, 1)) if rpm.size else 0.5
        obs = observe_from_sensors(s, target, torch.tensor([[motor_mean]], device=self.device), self.cfg, rel_b=rel_b_t)
        act, self.state, _ = self.brain(obs, self.state, self.W)
        a = act[0].cpu().numpy().astype(np.float64)
        a[1:] *= self.stick_gain
        if self.face_gain > 0:
            if rel_b_t is not None:
                rel_b = rel_b_t[0].cpu().numpy()
                err = float(np.arctan2(rel_b[1], rel_b[0])) if np.hypot(rel_b[0], rel_b[1]) > 0.8 else 0.0
            else:
                rel = target[0].cpu().numpy() - s['pos'][0].cpu().numpy()
                err = 0.0
                if np.hypot(rel[0], rel[1]) > 0.8:
                    err = np.angle(np.exp(1j * (np.arctan2(rel[1], rel[0]) - float(s['yaw'].flatten()[0]))))
            a[3] = float(np.clip(-self.face_gain * err, -self.face_max, self.face_max))
        if self.stick_lpf > 0:
            dt = max(fr.timestamp - self.prev_t, 1e-3) if self.prev_t is not None else 0.01
            alpha = min(1.0, dt / self.stick_lpf)
            self.filtered = a.copy() if self.filtered is None else self.filtered + alpha * (a - self.filtered)
            a = self.filtered
        sticks = np.array([self.map.map_throttle(float(a[0])),
                           self.map.map_axis('roll', float(a[1]) * self.map.stick_sign[0]),
                           self.map.map_axis('pitch', float(a[2]) * self.map.stick_sign[1]),
                           self.map.map_axis('yaw', float(a[3]) * self.map.stick_sign[2])])
        # the game normalises each stick to the unit circle: keep (throttle, yaw) and (roll, pitch) inside it
        for i, j in ((0, 3), (1, 2)):
            n = float(np.hypot(sticks[i], sticks[j]))
            if n > 0.97:
                sticks[i] *= 0.97 / n
                sticks[j] *= 0.97 / n
        self.last_obs = obs
        self.last_quat = s['quat'][0].cpu().numpy()
        self.last_target = target[0].cpu().numpy()
        self.last_pos = s['pos'][0].cpu().numpy()
        return np.clip(sticks, -1.0, 1.0)
