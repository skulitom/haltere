"""Flight tasks: initial-state distributions, sensory observations for the brain, and costs."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .quad import QuadState, QuadSim, quat_from_euler, rotate_inv


@dataclass
class HoverTaskConfig:
    height: float = 2.0
    target_box: tuple[float, float, float] = (3.0, 3.0, 1.0)  # +- range of hover target from origin (x,y,z)
    init_pos_jitter: float = 1.0        # m
    init_vel_jitter: float = 1.0        # m/s
    init_tilt_deg: float = 25.0
    init_rate_deg_s: float = 90.0
    episode_steps: int = 400            # at the control rate
    min_height: float = 0.3             # soft floor penalty starts here
    difficulty: float = 0.3             # 0..1 scales all jitters
    # cost weights
    w_pos: float = 1.0
    w_vel: float = 0.05
    w_rate: float = 0.02
    w_up: float = 2.0
    w_act: float = 0.02
    w_dact: float = 0.2
    w_floor: float = 5.0
    w_crash: float = 20.0
    pos_huber: float = 0.0              # m; > 0: position cost grows linearly beyond this distance (Huber) so that
                                        # far-behind moving targets do not dominate the smoothness terms
    goal_scale: float = 3.0             # m, saturating scale of goal-vector encoding
    vel_scale: float = 6.0              # m/s
    rate_scale: float = 6.0             # rad/s
    switch_target_every: int = 0        # steps; 0 = fixed target per episode (waypoint mode if > 0)
    # moving target (path following): the target drifts at a random speed along a slowly turning heading
    path_speed: float = 0.0             # m/s, ceiling of the target speed (0 = static targets)
    path_turn_deg_s: float = 30.0       # max heading turn rate of the moving target
    path_climb: float = 0.3             # vertical speed as a fraction of the horizontal speed
    path_z_range: tuple[float, float] = (1.2, 3.5)   # the moving target bounces between these altitudes
    path_static_frac: float = 0.0       # fraction of episodes whose target does not move (keeps hover precise)
    control_dt: float = 0.01            # s per control step (target motion)


def observe_from_sensors(s: dict[str, torch.Tensor], target: torch.Tensor, motor_mean: torch.Tensor,
                         c: HoverTaskConfig) -> dict[str, torch.Tensor]:
    """Turn a sensor dict (see QuadSim.sensors; also produced from Liftoff telemetry) into the brain's
    sensory channels. Shared by the simulator and the Liftoff runtime so both see identical inputs."""
    rel = target - s['pos']
    rel_b = rotate_inv(s['quat'], rel)
    dist = rel.norm(dim=1, keepdim=True)
    goal = torch.cat([torch.tanh(rel_b / c.goal_scale), torch.tanh(dist / c.goal_scale)], dim=1)
    yaw = s['yaw']
    compass = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=1)
    alt = s['altitude'].clamp(min=0.3)
    flow = torch.cat([torch.tanh(s['gyro'] / c.rate_scale), torch.tanh(s['vel_body'] / alt)], dim=1)
    wing = torch.cat([s['up'][:, None], torch.tanh(s['vel_world'][:, 2:3] / c.vel_scale), motor_mean * 2 - 1], dim=1)
    return {
        'haltere': torch.tanh(s['gyro'] / c.rate_scale),
        'ocelli': s['gravity_body'],
        'lptc': flow,
        'jo': torch.tanh(s['vel_body'] / c.vel_scale),
        'wing_cs': wing,
        'compass': compass,
        'goal': goal,
    }


class HoverTask:
    """Fly to and hold a target point. Also serves as a waypoint task when switch_target_every > 0."""

    # name -> dimensionality of each sensory channel handed to the brain
    channels = {
        'haltere': 3,        # angular rates (body)
        'ocelli': 3,         # gravity direction in body frame (attitude)
        'lptc': 6,           # optic-flow proxies: rotation rates + translation/altitude
        'jo': 3,             # airflow: body-frame velocity
        'wing_cs': 3,        # wing-load proxies: up-vector, vertical speed, mean motor command
        'compass': 2,        # heading (cos, sin)
        'goal': 4,           # goal vector in body frame (saturated) + distance
    }

    def __init__(self, cfg: HoverTaskConfig, sim: QuadSim, B: int, device):
        self.cfg = cfg
        self.sim = sim
        self.B = B
        self.device = device
        self.target = torch.zeros(B, 3, device=device)
        self.target_vel = torch.zeros(B, 3, device=device)
        self.turn_rate = torch.zeros(B, device=device)
        self.t = torch.zeros(B, dtype=torch.long, device=device)
        self.prev_action = torch.zeros(B, 4, device=device)

    # ------------------------------------------------------------------ resets
    def _difficulty(self, difficulty: float | None) -> float:
        return self.cfg.difficulty if difficulty is None else difficulty

    def _sample_motion(self, n: int, d: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Random target motion: speed ~ U(0, d * path_speed), random heading, turn rate and climb."""
        c, dev = self.cfg, self.device
        speed = d * c.path_speed * torch.rand(n, device=dev)
        if c.path_static_frac > 0:
            speed = torch.where(torch.rand(n, device=dev) < c.path_static_frac, torch.zeros_like(speed), speed)
        heading = (2 * torch.rand(n, device=dev) - 1) * torch.pi
        climb = c.path_climb * (2 * torch.rand(n, device=dev) - 1)
        vel = torch.stack([speed * torch.cos(heading), speed * torch.sin(heading), speed * climb], dim=1)
        turn = torch.deg2rad(torch.tensor(c.path_turn_deg_s, device=dev)) * (2 * torch.rand(n, device=dev) - 1)
        return vel, turn

    def _sample_targets(self, n: int) -> torch.Tensor:
        box = torch.tensor(self.cfg.target_box, device=self.device)
        target = (2 * torch.rand(n, 3, device=self.device) - 1) * box
        target[:, 2] = self.cfg.height + target[:, 2]
        return target

    def sample_state(self, n: int, difficulty: float | None = None) -> tuple[QuadState, torch.Tensor]:
        c = self.cfg
        d = c.difficulty if difficulty is None else difficulty
        dev = self.device
        target = self._sample_targets(n)
        pos = target + d * c.init_pos_jitter * torch.randn(n, 3, device=dev)
        pos[:, 2] = pos[:, 2].clamp(min=c.min_height + 0.2)
        vel = d * c.init_vel_jitter * torch.randn(n, 3, device=dev)
        tilt = torch.deg2rad(torch.tensor(c.init_tilt_deg, device=dev)) * d
        roll = tilt * torch.randn(n, device=dev)
        pitch = tilt * torch.randn(n, device=dev)
        yaw = (2 * torch.rand(n, device=dev) - 1) * torch.pi
        quat = quat_from_euler(roll, pitch, yaw)
        omega = torch.deg2rad(torch.tensor(c.init_rate_deg_s, device=dev)) * d * torch.randn(n, 3, device=dev)
        motor = torch.full((n, 4), self.sim.hover_command(), device=dev)
        st = QuadState(pos, vel, quat, omega, motor, torch.zeros(n, dtype=torch.bool, device=dev))
        return st, target

    def reset_all(self, difficulty: float | None = None) -> QuadState:
        st, target = self.sample_state(self.B, difficulty)
        self.target = target
        self.target_vel, self.turn_rate = self._sample_motion(self.B, self._difficulty(difficulty))
        self.t = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.prev_action = torch.zeros(self.B, 4, device=self.device)
        return st

    def reset_where(self, st: QuadState, mask: torch.Tensor, difficulty: float | None = None) -> QuadState:
        if not bool(mask.any()):
            return st
        new_st, new_target = self.sample_state(self.B, difficulty)
        self.target = torch.where(mask[:, None], new_target, self.target)
        vel, turn = self._sample_motion(self.B, self._difficulty(difficulty))
        self.target_vel = torch.where(mask[:, None], vel, self.target_vel)
        self.turn_rate = torch.where(mask, turn, self.turn_rate)
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
        self.prev_action = torch.where(mask[:, None], torch.zeros_like(self.prev_action), self.prev_action)
        return st.where(mask, new_st)

    # ------------------------------------------------------------------ observation
    def observe(self, st: QuadState) -> dict[str, torch.Tensor]:
        s = self.sim.sensors(st)
        return observe_from_sensors(s, self.target, st.motor.mean(dim=1, keepdim=True), self.cfg)

    # ------------------------------------------------------------------ cost / termination
    def cost(self, st: QuadState, action: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        s = self.sim.sensors(st, noise=False)
        pos_err = (st.pos - self.target).pow(2).sum(1)
        if c.pos_huber > 0:
            dist = (pos_err + 1e-8).sqrt()
            pos_err = torch.where(dist < c.pos_huber, pos_err, c.pos_huber * (2 * dist - c.pos_huber))
        vel = st.vel.pow(2).sum(1)
        rate = st.omega.pow(2).sum(1)
        tilt = 1.0 - s['up']
        act = action.pow(2).sum(1)
        dact = (action - self.prev_action).pow(2).sum(1)
        floor = (c.min_height - st.pos[:, 2]).clamp(min=0.0).pow(2)
        crash = st.crashed.float()
        self.prev_action = action.detach()   # the smoothness term must not tie BPTT windows together
        return (c.w_pos * pos_err + c.w_vel * vel + c.w_rate * rate + c.w_up * tilt + c.w_act * act
                + c.w_dact * dact + c.w_floor * floor + c.w_crash * crash)

    def advance_target(self) -> None:
        """Move the target one control step along its (slowly turning) heading; bounce off the altitude band."""
        c = self.cfg
        if c.path_speed <= 0:
            return
        a = self.turn_rate * c.control_dt
        ca, sa = torch.cos(a), torch.sin(a)
        vx, vy, vz = self.target_vel[:, 0], self.target_vel[:, 1], self.target_vel[:, 2]
        z = self.target[:, 2]
        bounce = ((z < c.path_z_range[0]) & (vz < 0)) | ((z > c.path_z_range[1]) & (vz > 0))
        vz = torch.where(bounce, -vz, vz)
        self.target_vel = torch.stack([ca * vx - sa * vy, sa * vx + ca * vy, vz], dim=1)
        self.target = self.target + self.target_vel * c.control_dt

    def tick(self, st: QuadState) -> torch.Tensor:
        """Advance episode clocks (and waypoints / moving targets); return the mask of environments to reset."""
        self.t = self.t + 1
        self.advance_target()
        if self.cfg.switch_target_every > 0:
            switch = (self.t % self.cfg.switch_target_every) == 0
            if bool(switch.any()):
                self.target = torch.where(switch[:, None], self._sample_targets(self.B), self.target)
        return st.crashed | (self.t >= self.cfg.episode_steps)

    def metrics(self, st: QuadState) -> dict[str, float]:
        d = (st.pos - self.target).norm(dim=1)
        return {'dist_mean': float(d.mean()), 'within_0.5m': float((d < 0.5).float().mean()),
                'crashed': float(st.crashed.float().mean()), 'speed': float(st.vel.norm(dim=1).mean()),
                'target_speed': float(self.target_vel.norm(dim=1).mean())}
