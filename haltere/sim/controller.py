"""Betaflight-like rate PID controller and motor mixer, batched and differentiable (PyTorch).

Gains are expressed in Betaflight units (the numbers you see in Liftoff's drone configuration,
e.g. P=29 I=36 D=22) and converted with Betaflight's internal scales, so the simulator's
behaviour is in the right ballpark before system identification tunes the ``axis_gain`` multipliers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# Betaflight src/main/flight/pid.c
PTERM_SCALE = 0.032029
ITERM_SCALE = 0.244381
DTERM_SCALE = 0.000529
PID_MIXER_SCALING = 1000.0

# Liftoff default motor mixing table: rows = motors (LF, RF, LB, RB), cols = (throttle, roll, pitch, yaw)
LIFTOFF_MIX = [[1.0, 1.0, 1.0, -1.0],
               [1.0, -1.0, 1.0, 1.0],
               [1.0, 1.0, -1.0, 1.0],
               [1.0, -1.0, -1.0, -1.0]]


@dataclass
class RateControllerParams:
    P: tuple[float, float, float] = (29.0, 29.0, 30.0)
    I: tuple[float, float, float] = (34.0, 36.0, 38.0)
    D: tuple[float, float, float] = (22.0, 22.0, 0.0)
    axis_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)  # sysid multipliers on the whole PID sum
    iterm_limit: float = 400.0          # Betaflight itermLimit, in PID units
    dterm_lpf_hz: float = 100.0
    idle: float = 0.04                  # idleMotorThrottlePercentage
    airmode: bool = True
    mix: list[list[float]] = field(default_factory=lambda: [row[:] for row in LIFTOFF_MIX])


@dataclass
class RateControllerState:
    iterm: torch.Tensor      # [B,3]
    prev_gyro: torch.Tensor  # [B,3] deg/s
    dterm: torch.Tensor      # [B,3] filtered derivative (PID units)

    @staticmethod
    def zeros(B: int, device) -> "RateControllerState":
        z = torch.zeros(B, 3, device=device)
        return RateControllerState(z.clone(), z.clone(), z.clone())

    def where(self, mask: torch.Tensor, other: "RateControllerState") -> "RateControllerState":
        m = mask[:, None]
        return RateControllerState(torch.where(m, other.iterm, self.iterm),
                                   torch.where(m, other.prev_gyro, self.prev_gyro),
                                   torch.where(m, other.dterm, self.dterm))

    def detach(self) -> "RateControllerState":
        return RateControllerState(self.iterm.detach(), self.prev_gyro.detach(), self.dterm.detach())


class RateController:
    """Rate-mode (acro) PID + mixer. All tensors are batched [B, ...]."""

    def __init__(self, params: RateControllerParams, device):
        self.p = params
        self.device = device
        self.kp = torch.tensor(params.P, device=device) * PTERM_SCALE
        self.ki = torch.tensor(params.I, device=device) * ITERM_SCALE
        self.kd = torch.tensor(params.D, device=device) * DTERM_SCALE
        self.axis_gain = torch.tensor(params.axis_gain, device=device)
        self.mix = torch.tensor(params.mix, device=device, dtype=torch.float32)  # [4,4]

    def pid(self, st: RateControllerState, setpoint_deg_s: torch.Tensor, gyro_deg_s: torch.Tensor,
            dt: float) -> tuple[torch.Tensor, RateControllerState]:
        err = setpoint_deg_s - gyro_deg_s
        pterm = self.kp * err
        iterm = (st.iterm + self.ki * err * dt).clamp(-self.p.iterm_limit, self.p.iterm_limit)
        dgyro = (gyro_deg_s - st.prev_gyro) / dt
        raw_d = -self.kd * dgyro
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.p.dterm_lpf_hz * dt)
        dterm = st.dterm + alpha * (raw_d - st.dterm)
        axis_cmd = (pterm + iterm + dterm) * self.axis_gain / PID_MIXER_SCALING
        return axis_cmd, RateControllerState(iterm, gyro_deg_s, dterm)

    def mixer(self, throttle: torch.Tensor, axis_cmd: torch.Tensor) -> torch.Tensor:
        """throttle [B] in [0,1], axis_cmd [B,3] -> motor commands [B,4] in [0,1] (after idle)."""
        cmd = torch.cat([throttle[:, None], axis_cmd], dim=1)  # [B,4]
        m = cmd @ self.mix.T                                     # [B,4]
        if self.p.airmode:
            mmax = m.max(dim=1).values
            mmin = m.min(dim=1).values
            span = (mmax - mmin).clamp(min=1e-6)
            scale = torch.where(span > 1.0, 1.0 / span, torch.ones_like(span))
            m = throttle[:, None] + (m - throttle[:, None]) * scale[:, None]
            mmax = m.max(dim=1).values
            mmin = m.min(dim=1).values
            shift = torch.where(mmax > 1.0, 1.0 - mmax, torch.zeros_like(mmax))
            shift = torch.where(mmin < 0.0, -mmin, shift)
            m = m + shift[:, None]
        m = m.clamp(0.0, 1.0)
        return self.p.idle + (1.0 - self.p.idle) * m
