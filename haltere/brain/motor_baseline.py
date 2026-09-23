"""Conventional motor baseline for matched guidance comparisons.

This is not a fly brain. It consumes the same local target and measured motion;
it has no course, gate-order or obstacle knowledge. Yaw may be supplied by the
same visual assistant used with the connectome.
"""
from dataclasses import asdict, dataclass
import math

import torch

from ..sim.quad import quat_to_mat
from ..sim.rates import betaflight_rate_curve


@dataclass(frozen=True)
class MotorPDConfig:
    position_gain: float = 0.8
    velocity_gain: float = 1.6
    vertical_position_gain: float = 0.8
    vertical_velocity_gain: float = 2.5
    attitude_gain: float = 4.0
    angular_damping: float = 0.15
    max_acceleration: float = 5.0
    max_vertical_speed: float = 1.2
    max_rate: float = 2.0


def inverse_rate(rate, rc_rate, super_rate, expo):
    """Monotone inverse in brain stick coordinates, independent of pad mapping."""
    low, high = torch.full_like(rate, -1.), torch.ones_like(rate)
    for _ in range(14):
        middle = (low + high) * .5
        below = betaflight_rate_curve(middle, rc_rate, super_rate, expo) < rate
        low, high = torch.where(below, middle, low), torch.where(below, high, middle)
    return (low + high) * .5


class MotorPD:
    def __init__(self, quad, rates, idle=.04, config=None):
        self.quad, self.rates, self.idle = quad, rates, idle
        self.config = config or MotorPDConfig()

    def metadata(self):
        return dict(kind='pd', parameters=asdict(self.config),
                    inputs='local body-relative target, attitude, velocity and angular rates',
                    course_geometry=False, brain_controls_motors=False)

    def command(self, sensors, relative_body, speed=2.):
        p = self.config
        rotation = quat_to_mat(sensors['quat'])
        relative = torch.einsum('bij,bj->bi', rotation, relative_body)
        desired_xy = p.position_gain * relative[:, :2]
        desired_xy *= (speed / desired_xy.norm(dim=-1, keepdim=True).clamp_min(speed))
        desired_z = (p.vertical_position_gain * relative[:, 2]).clamp(
            -p.max_vertical_speed, p.max_vertical_speed)
        desired = torch.cat((desired_xy, desired_z[:, None]), -1)
        gain = relative.new_tensor([p.velocity_gain, p.velocity_gain, p.vertical_velocity_gain])
        acceleration = (gain * (desired - sensors['vel_world'])).clamp(
            -p.max_acceleration, p.max_acceleration)
        force = acceleration + relative.new_tensor([0., 0., 9.81])
        direction = force / force.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        body_direction = torch.einsum('bji,bj->bi', rotation, direction)
        tilt_error = torch.stack((-body_direction[:, 1], body_direction[:, 0]), -1)
        rates = (p.attitude_gain * tilt_error - p.angular_damping * sensors['gyro'][:, :2]).clamp(
            -p.max_rate, p.max_rate)
        sticks = [inverse_rate(torch.rad2deg(rates[:, i]), self.rates.rc_rate[i],
                               self.rates.super_rate[i], self.rates.expo[i]) for i in range(2)]
        thrust = force[:, 2] / rotation[:, 2, 2].clamp_min(.4)
        motor = (thrust / (9.81 * self.quad.twr)).clamp_min(0.).pow(1. / self.quad.thrust_exp)
        throttle = 2 * ((motor - self.idle) / (1 - self.idle)) - 1
        return torch.stack((throttle.clamp(-1., 1.), *sticks, torch.zeros_like(throttle)), -1)


@dataclass(frozen=True)
class FastPDConfig:
    """Declared high-envelope profile; the teacher contract above is unchanged."""
    velocity_gain: float = 3.0
    vertical_velocity_gain: float = 3.0
    max_acceleration: float = 15.0
    max_climb_acceleration: float = 8.0
    max_sink_acceleration: float = 6.0
    max_tilt_deg: float = 60.0
    attitude_gain: float = 8.0
    angular_damping: float = 0.15
    max_rate: float = 8.0
    feedforward: float = 1.0
    stick_time_constant: float = 0.02

    def __post_init__(self):
        values = list(asdict(self).values())
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Use finite nonnegative fast PD parameters')
        if not 0 < self.max_tilt_deg < 80 or min(self.velocity_gain, self.max_acceleration, self.max_rate) <= 0:
            raise ValueError('Use a bounded tilt and positive gains/limits')


def measured_inverse_rate(rate_deg, coefficient, super_rate, expo):
    """Invert the measured Liftoff rate curve, with saturation applied after expo."""
    magnitude = rate_deg.abs()
    shaped = (magnitude / (coefficient + super_rate * magnitude)).clamp(max=.999)
    low, high = torch.zeros_like(shaped), torch.ones_like(shaped)
    for _ in range(18):
        middle = (low + high) * .5
        below = middle * (1 - expo + expo * middle.pow(3)) < shaped
        low, high = torch.where(below, middle, low), torch.where(below, high, middle)
    return torch.sign(rate_deg) * (low + high) * .5


class FastMotorPD:
    """Velocity-command motor baseline for the measured original drone.

    Consumes a world velocity request and its feedforward acceleration from a
    causal pilot. It uses the measured full-throttle thrust curve and the
    measured post-expo rate curve instead of the near-hover teacher fit. It has
    no course, gate or obstacle knowledge and is not a fly brain.
    """
    def __init__(self, profile, calibration, config=None):
        thrust = profile['thrust']
        self.twr, self.exponent = float(thrust['full_input_twr']), float(thrust['exponent'])
        axes = profile['axes']
        self.rate_curve = [(float(axes[a]['coefficient_deg_s']), float(axes[a]['super_rate']),
                            float(axes[a]['expo'])) for a in ('roll', 'pitch')]
        if any(not axes[a].get('super_after_expo') for a in ('roll', 'pitch')):
            raise ValueError('Fast PD requires the measured post-expo rate curve')
        if not (self.twr > 1 and self.exponent > 0):
            raise ValueError('Use a measured thrust-to-weight ratio above one')
        self.calibration = {k: calibration[k] for k in ('hover_processed', 'hover_stick_sim', 'throttle_scale')}
        self.config = config or FastPDConfig()
        self.previous = None

    def reset(self):
        self.previous = None

    def metadata(self):
        return dict(kind='fast_pd', parameters=asdict(self.config), thrust_twr=self.twr,
                    thrust_exponent=self.exponent, rate_curve=self.rate_curve,
                    inputs='world velocity request, feedforward acceleration, attitude, velocity and rates',
                    course_geometry=False, brain_controls_motors=False)

    def command(self, sensors, velocity, feedforward=None, dt=.01):
        p = self.config
        rotation = quat_to_mat(sensors['quat'])
        velocity = torch.as_tensor(velocity, dtype=rotation.dtype, device=rotation.device).reshape(-1, 3)
        error = velocity - sensors['vel_world']
        acceleration = torch.cat((p.velocity_gain * error[:, :2],
                                  p.vertical_velocity_gain * error[:, 2:]), -1)
        if feedforward is not None:
            acceleration = acceleration + p.feedforward * torch.as_tensor(
                feedforward, dtype=rotation.dtype, device=rotation.device).reshape(-1, 3)
        horizontal = acceleration[:, :2]
        horizontal = horizontal * (p.max_acceleration / horizontal.norm(dim=-1, keepdim=True).clamp_min(p.max_acceleration))
        vertical = acceleration[:, 2:].clamp(-p.max_sink_acceleration, p.max_climb_acceleration)
        force = torch.cat((horizontal, vertical + 9.81), -1)
        # Keep the requested thrust within the declared tilt cone; vertical
        # support has priority over horizontal acceleration.
        limit = torch.tan(torch.deg2rad(force.new_tensor(p.max_tilt_deg)))
        lateral = force[:, :2].norm(dim=-1, keepdim=True)
        allowed = force[:, 2:].clamp_min(1.) * limit
        force = torch.cat((force[:, :2] * (allowed / lateral.clamp_min(1e-6)).clamp(max=1.), force[:, 2:]), -1)
        direction = force / force.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        body_direction = torch.einsum('bji,bj->bi', rotation, direction)
        tilt_error = torch.stack((-body_direction[:, 1], body_direction[:, 0]), -1)
        rates = (p.attitude_gain * tilt_error - p.angular_damping * sensors['gyro'][:, :2]).clamp(
            -p.max_rate, p.max_rate)
        sticks = torch.stack([measured_inverse_rate(torch.rad2deg(rates[:, i]), *self.rate_curve[i])
                              for i in range(2)], -1)
        thrust = (force * rotation[:, :, 2]).sum(-1).clamp_min(0.)
        drive = (thrust / (9.81 * self.twr)).clamp(0., 1.).pow(1. / self.exponent)
        c = self.calibration
        throttle = c['hover_stick_sim'] + (2 * drive - 1 - c['hover_processed']) / c['throttle_scale']
        action = torch.cat((throttle[:, None].clamp(-1., 1.), sticks.clamp(-1., 1.),
                            torch.zeros_like(throttle)[:, None]), -1)
        if p.stick_time_constant > 0 and self.previous is not None and self.previous.shape == action.shape:
            alpha = 1 - torch.exp(action.new_tensor(-dt / p.stick_time_constant))
            action = torch.cat((action[:, :1], self.previous[:, 1:3] + alpha * (action[:, 1:3] - self.previous[:, 1:3]),
                                action[:, 3:]), -1)
        self.previous = action.detach().clone()
        return action
