"""Betaflight-style stick -> rate-setpoint curves, as used by Liftoff's Zetaflight/Betaflight controllers.

Liftoff's drone configuration exposes ``Rate`` / ``Expo`` / ``SuperExpo`` per axis in percent
(e.g. Rate=100, Expo=0, SuperExpo=70), which map onto Betaflight's classic "Betaflight rates":
``rcRate = Rate/100``, ``rcExpo = Expo/100``, ``superRate = SuperExpo/100``.
"""
from __future__ import annotations

import torch

RC_RATE_INCREMENTAL = 14.54


def betaflight_rate_curve(stick: torch.Tensor, rc_rate: float = 1.0, super_rate: float = 0.7,
                          expo: float = 0.0) -> torch.Tensor:
    """Map a stick deflection in [-1, 1] to an angular-rate setpoint in deg/s.

    Port of Betaflight's ``applyBetaflightRates`` (src/main/fc/rc.c)."""
    a = stick.abs()
    if expo:
        stick = stick * a.pow(3) * expo + stick * (1.0 - expo)
    rr = rc_rate
    if rr > 2.0:
        rr += RC_RATE_INCREMENTAL * (rr - 2.0)
    rate = 200.0 * rr * stick
    if super_rate:
        rate = rate / (1.0 - a * super_rate).clamp(0.01, 1.0)
    return rate


def max_rate_deg_s(rc_rate: float = 1.0, super_rate: float = 0.7, expo: float = 0.0) -> float:
    return float(betaflight_rate_curve(torch.tensor(1.0), rc_rate, super_rate, expo))


def throttle_curve(stick: torch.Tensor, start: float = 0.0, mid: float = 0.5, end: float = 1.0,
                   expo: float = 0.0) -> torch.Tensor:
    """Liftoff throttle curve (ThrottleCurveStart/Mid/End in [0,1], expo in [0,1]).

    ``stick`` is in [0, 1]. Piecewise-linear through (0,start),(0.5,mid),(1,end) with an optional
    cubic expo blend around the midpoint. Default is the identity."""
    s = stick.clamp(0.0, 1.0)
    lo = start + (mid - start) * (s / 0.5)
    hi = mid + (end - mid) * ((s - 0.5) / 0.5)
    y = torch.where(s < 0.5, lo, hi)
    if expo:
        c = (s - 0.5) * 2.0
        y = y * (1.0 - expo) + (mid + (end - mid) * c.pow(3) * torch.sign(c).abs() * 0.5 + (c.pow(3)) * 0.5 * (mid - start)) * expo
    return y.clamp(0.0, 1.0)
