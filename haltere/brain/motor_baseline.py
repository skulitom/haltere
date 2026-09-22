"""Conventional motor baseline for matched guidance comparisons.

This is not a fly brain. It consumes the same local target and measured motion;
it has no course, gate-order or obstacle knowledge. Yaw may be supplied by the
same visual assistant used with the connectome.
"""
from dataclasses import asdict, dataclass

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
