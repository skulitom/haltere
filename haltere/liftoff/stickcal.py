"""Stick-curve calibration: how Liftoff transforms the virtual pad's raw axis into the processed input
it feeds its flight controller (deadband, expo, range), measured on the ground by ramping each axis
while recording the ``Input`` telemetry field. The inverse curve lets the pilot command the
processed input the simulator's controller expects.
"""
from __future__ import annotations

import csv

import numpy as np

AXES = ['throttle', 'roll', 'pitch', 'yaw']
TELE_COL = {'throttle': 'in_throttle', 'roll': 'in_roll', 'pitch': 'in_pitch', 'yaw': 'in_yaw'}


def load_pad_log(path: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
    return {k: np.array([float(r[k]) for r in rows]) for k in ['wall_time'] + AXES}


def load_telemetry(path: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
    out = {'recv_time': np.array([float(r['recv_time']) for r in rows])}
    for ax, col in TELE_COL.items():
        out[ax] = np.array([float(r[col]) for r in rows])
    return out


def fit_curves(pad_log: str, telemetry_csv: str, n_bins: int = 41, latency: float = 0.03,
               extra_throttle_points: list | None = None) -> dict[str, dict]:
    """Per axis: the processed input as a function of the raw stick, from time-aligned samples."""
    pad = load_pad_log(pad_log)
    tele = load_telemetry(telemetry_csv)
    t_pad = pad['wall_time'] + latency
    # which axis is moving at each pad sample (change over the previous ~0.2 s)
    moving = {}
    for ax in AXES:
        v = pad[ax]
        k = 40
        d = np.abs(v - np.concatenate([np.full(k, v[0]), v[:-k]]))
        moving[ax] = d > 0.01
    curves = {}
    for ax in AXES:
        # the telemetry sample closest after each pad sample
        idx = np.searchsorted(tele['recv_time'], t_pad)
        ok = (idx < len(tele['recv_time'])) & (idx > 0)
        raw = pad[ax][ok]
        proc = tele[ax][idx[ok]]
        # only samples where this axis is the one moving and the others are still
        others = [o for o in AXES if o != ax]
        quiet = moving[ax][ok] & ~np.any([moving[o][ok] for o in others], axis=0)
        raw, proc = raw[quiet], proc[quiet]
        if len(raw) < 50:
            continue
        # the axis may be inverted by the game profile: fit the curve of sign * processed, remember the sign
        sign = 1.0 if np.corrcoef(raw, proc)[0, 1] >= 0 else -1.0
        proc = sign * proc
        lo, hi = float(raw.min()), float(raw.max())
        grid = np.linspace(lo, hi, n_bins)
        half = (hi - lo) / (2 * (n_bins - 1)) + 1e-6
        binned = np.array([proc[np.abs(raw - g) <= half].mean() if np.any(np.abs(raw - g) <= half) else np.nan for g in grid])
        good = ~np.isnan(binned)
        grid, binned = grid[good], binned[good]
        binned = np.maximum.accumulate(binned)   # monotone non-decreasing so it can be inverted
        curves[ax] = {'raw': grid.round(4).tolist(), 'processed': binned.round(4).tolist(), 'sign': sign,
                      'samples': int(len(raw))}
    if 'throttle' in curves and extra_throttle_points:
        # above the deadband the ground ramp cannot go (the drone would lift): add points measured in flight
        c = curves['throttle']
        raw = np.array(c['raw'] + [p[0] for p in extra_throttle_points])
        proc = np.array(c['processed'] + [p[1] for p in extra_throttle_points])
        order = np.argsort(raw)
        raw, proc = raw[order], np.maximum.accumulate(proc[order])
        curves['throttle'] = {'raw': raw.round(4).tolist(), 'processed': proc.round(4).tolist(), 'sign': c['sign'],
                              'samples': c['samples']}
    return curves


class RadialSticks:
    """Liftoff's (Rewired's) processing of a gamepad stick, measured with ``haltere liftoff sticktest``: the deadzone
    is applied to each stick's 2D vector, the magnitude beyond it is rescaled to [0, 1], the result is clamped to the
    unit circle, and the profile may invert an axis. Per stick (a, b) and per-axis sign s:

        processed = s * raw / |raw| * clip((|raw| - dz) / (1 - dz), 0, 1)

    ``raw_for`` inverts it exactly for one stick, so the flight controller sees what the brain commanded."""

    def __init__(self, deadzone: float = 0.25, sign: dict | None = None):
        self.dz = float(deadzone)
        self.sign = {'throttle': 1.0, 'yaw': 1.0, 'roll': 1.0, 'pitch': 1.0, **(sign or {})}

    def processed(self, ax_a: str, ax_b: str, raw_a: float, raw_b: float) -> tuple[float, float]:
        m = float(np.hypot(raw_a, raw_b))
        if m <= self.dz:
            return 0.0, 0.0
        k = min((m - self.dz) / (1.0 - self.dz), 1.0) / m
        return self.sign[ax_a] * raw_a * k, self.sign[ax_b] * raw_b * k

    def raw_for(self, ax_a: str, ax_b: str, p_a: float, p_b: float) -> tuple[float, float]:
        a, b = self.sign[ax_a] * p_a, self.sign[ax_b] * p_b
        m = float(np.hypot(a, b))
        if m < 1e-6:
            return 0.0, 0.0
        if m > 1.0:                        # beyond the unit circle the game would clamp it: keep the direction
            a, b, m = a / m, b / m, 1.0
        k = (self.dz + (1.0 - self.dz) * m) / m
        return a * k, b * k

    def describe(self) -> str:
        return (f'radial deadzone {self.dz:.3f} per stick (throttle+yaw, roll+pitch), unit-circle clamp, signs '
                + ', '.join(f'{k} {v:+.0f}' for k, v in self.sign.items()))


def fit_radial(csv_path: str) -> tuple[RadialSticks, dict]:
    """Fit the radial model to a ``sticktest`` recording (raw pad axes and the telemetry's processed input)."""
    rows = list(csv.DictReader(open(csv_path, newline='', encoding='utf-8')))
    raw = {k: np.array([float(r[k]) for r in rows]) for k in AXES}
    proc = {k: np.array([float(r['p_' + k]) for r in rows]) for k in AXES}
    sign = {}
    for ax in AXES:
        big = np.abs(raw[ax]) > 0.5
        sign[ax] = 1.0 if np.sum(raw[ax][big] * proc[ax][big]) >= 0 else -1.0
    best = None
    for dz in np.arange(0.0, 0.401, 0.005):
        model = RadialSticks(dz, sign)
        err = []
        for a, b in (('throttle', 'yaw'), ('roll', 'pitch')):
            pa, pb = np.array([model.processed(a, b, x, y) for x, y in zip(raw[a], raw[b])]).T
            err += [pa - proc[a], pb - proc[b]]
        rms = float(np.sqrt(np.mean(np.concatenate(err) ** 2)))
        if best is None or rms < best[1]:
            best = (dz, rms)
    return RadialSticks(best[0], sign), {'deadzone': round(float(best[0]), 3), 'rms': best[1], 'samples': len(rows)}


class StickCurves:
    """Inverse curves: processed (what the FC should see, i.e. the simulator's stick) -> raw pad axis.
    The game's own axis inversion (``sign``) is undone here, so callers pass the *intended* processed value."""

    def __init__(self, curves: dict[str, dict]):
        self.inv = {}
        self.sign = {}
        for ax, c in curves.items():
            raw = np.asarray(c['raw'], dtype=float)
            proc = np.asarray(c['processed'], dtype=float)
            keep = np.concatenate([[True], np.diff(proc) > 1e-6])   # strictly increasing for interpolation
            self.inv[ax] = (proc[keep], raw[keep])
            self.sign[ax] = float(c.get('sign', 1.0))

    def raw_for(self, ax: str, processed: float) -> float:
        if ax not in self.inv:
            return float(processed)
        proc, raw = self.inv[ax]
        return float(np.interp(self.sign[ax] * processed, proc, raw, left=raw[0], right=raw[-1]))

    def describe(self) -> str:
        lines = []
        for ax, (proc, raw) in self.inv.items():
            lines.append(f'{ax:8s}: sign {self.sign[ax]:+.0f}; raw {raw[0]:+.2f}..{raw[-1]:+.2f} -> processed '
                         f'{proc[0]:+.2f}..{proc[-1]:+.2f}; raw for processed +0.35: {self.raw_for(ax, 0.35):+.2f}, '
                         f'-0.35: {self.raw_for(ax, -0.35):+.2f}, +0.10: {self.raw_for(ax, 0.10):+.2f}')
        return '\n'.join(lines)
