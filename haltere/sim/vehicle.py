"""Vehicle = quadrotor physics + Liftoff-style flight controller, driven by stick commands.

Stick vector (as the brain outputs it, and as the virtual gamepad receives it):
``[throttle, roll, pitch, yaw]`` each in [-1, 1]; throttle -1 = motors idle, +1 = full.

Sign conventions inside the simulator (body frame FLU: x forward, y left, z up):
roll stick + -> roll right (+omega_x); pitch stick + (forward) -> nose down (+omega_y);
yaw stick + (right) -> clockwise seen from above (-omega_z). The Liftoff adapter carries its own
per-axis sign vector that system identification fits, so these conventions never leak outside.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .controller import RateController, RateControllerParams, RateControllerState
from .quad import QuadParams, QuadSim, QuadState
from .rates import betaflight_rate_curve, throttle_curve

STICK_TO_RATE_SIGN = (1.0, 1.0, -1.0)   # stick (roll, pitch, yaw) -> rate setpoint sign about body x,y,z
AXIS_TO_MIX_SIGN = (1.0, -1.0, 1.0)     # PID output about body x,y,z -> Liftoff mixer (roll, pitch, yaw) columns


@dataclass
class RatesConfig:
    rc_rate: tuple[float, float, float] = (1.0, 1.0, 1.0)
    super_rate: tuple[float, float, float] = (0.7, 0.7, 0.7)
    expo: tuple[float, float, float] = (0.0, 0.0, 0.0)
    throttle: tuple[float, float, float, float] = (0.0, 0.5, 1.0, 0.0)  # start, mid, end, expo


@dataclass
class VehicleState:
    quad: QuadState
    ctl: RateControllerState

    @property
    def B(self) -> int:
        return self.quad.B

    def detach(self) -> "VehicleState":
        return VehicleState(self.quad.detach(), self.ctl.detach())

    def where(self, mask: torch.Tensor, other: "VehicleState") -> "VehicleState":
        return VehicleState(self.quad.where(mask, other.quad), self.ctl.where(mask, other.ctl))


class Vehicle:
    def __init__(self, quad: QuadParams, ctl: RateControllerParams, rates: RatesConfig, device,
                 dt: float = 0.01, substeps: int = 4):
        self.sim = QuadSim(quad, device, dt=dt, substeps=substeps)
        self.ctl = RateController(ctl, device)
        self.rates = rates
        self.device = device
        self.dt = dt
        self.stick_sign = torch.tensor(STICK_TO_RATE_SIGN, device=device)
        self.mix_sign = torch.tensor(AXIS_TO_MIX_SIGN, device=device)

    def wrap(self, quad: QuadState) -> VehicleState:
        return VehicleState(quad, RateControllerState.zeros(quad.B, self.device))

    def setpoints(self, sticks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """sticks [B,4] -> (throttle [B] in [0,1], rate setpoints [B,3] deg/s about body x,y,z)."""
        thr = throttle_curve((sticks[:, 0] + 1.0) * 0.5, *self.rates.throttle)
        cols = []
        for i in range(3):
            cols.append(betaflight_rate_curve(sticks[:, 1 + i], self.rates.rc_rate[i], self.rates.super_rate[i],
                                              self.rates.expo[i]))
        sp = torch.stack(cols, dim=1) * self.stick_sign
        return thr, sp

    def step(self, st: VehicleState, sticks: torch.Tensor) -> VehicleState:
        thr, sp = self.setpoints(sticks.clamp(-1.0, 1.0))
        quad, ctl = st.quad, st.ctl
        for _ in range(self.sim.substeps):
            gyro_deg = torch.rad2deg(quad.omega)
            axis, ctl = self.ctl.pid(ctl, sp, gyro_deg, self.sim.sub_dt)
            motor = self.ctl.mixer(thr, axis * self.mix_sign)
            quad = self.sim._substep(quad, motor, self.sim.sub_dt)
        return VehicleState(quad, ctl)
