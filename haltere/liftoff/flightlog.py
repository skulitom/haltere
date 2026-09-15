"""Score a flight from its ``fly --log`` CSV (every telemetry frame, 100 Hz).

Two kinds of numbers. Progress: gates flown through (the gate list's crossing test), time and average speed between
the first and last gate, ground speed, distance to the taught line. Wobble, i.e. what makes the FPV view shake:
the horizon's roll and pitch above 1 Hz (degrees RMS; the drone's slow, deliberate banking is below it), the body
rates above 1 Hz (deg/s RMS), the per-frame change of the processed input the flight controller received, and
vertical speed jitter. A flight that crashes is split at every reset; each attempt is scored on its own.
"""
from __future__ import annotations

import csv
import json

import numpy as np


def load_log(path: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
    out = {}
    for k in rows[0].keys():
        if k == 'status':
            out[k] = np.array([r[k] for r in rows], dtype=object)
            continue
        out[k] = np.array([float(r[k]) if r[k] not in ('', None) else np.nan for r in rows])
    # a log still being written ends in a partial row
    ok = np.all([np.isfinite(out[k]) for k in ('ts', 'px', 'vz', 'qw', 'wz', 'in_roll', 's_yaw', 'phase')], axis=0)
    return {k: v[ok] for k, v in out.items()}


def attempts(log: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Index arrays of the separate attempts: the game clock jumps back at every reset."""
    ts = log['ts']
    cuts = [0] + [i for i in range(1, len(ts)) if ts[i] < ts[i - 1] - 0.5] + [len(ts)]
    return [np.arange(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 200]


def _highpass(x: np.ndarray, dt: float, fc: float) -> np.ndarray:
    """Zero-phase first-order high-pass (forward-backward), enough to split deliberate motion from shake."""
    a = np.exp(-2 * np.pi * fc * dt)

    def one(sig):
        y = np.zeros_like(sig)
        for i in range(1, len(sig)):
            y[i] = a * (y[i - 1] + sig[i] - sig[i - 1])
        return y
    return one(one(x)[::-1])[::-1]


def euler_deg(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = q.T
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    return roll, pitch


def gate_crossings(P: np.ndarray, t: np.ndarray, gates: list[dict], half_width: float = 2.0) -> list[dict]:
    out = []
    for i, g in enumerate(gates):
        gp = np.asarray(g['pos'], dtype=float)
        h = g['heading']
        n = np.array([np.cos(h), np.sin(h), 0.0])
        s = np.array([-np.sin(h), np.cos(h), 0.0])
        along = (P - gp) @ n
        for c in np.where((along[:-1] <= 0) & (along[1:] > 0))[0]:
            lat = float((P[c] - gp) @ s)
            dz = float(P[c, 2] - gp[2])
            if abs(lat) > 12.0:                    # crossing the gate's plane far away is not an attempt at it
                continue
            out.append({'gate': i, 't': float(t[c]), 'lateral_m': lat, 'dz_m': dz,
                        'through': abs(lat) < half_width and -0.5 < dz < 3.0})
    return sorted(out, key=lambda p: p['t'])


def path_error(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    a, b = W[:-1], W[1:]
    ab = b - a
    err = np.full(len(P), np.inf)
    for k in range(len(a)):
        tt = np.clip(((P - a[k]) @ ab[k]) / max(float(ab[k] @ ab[k]), 1e-9), 0, 1)
        err = np.minimum(err, np.linalg.norm(a[k] + tt[:, None] * ab[k] - P, axis=1))
    return err


def score_attempt(log: dict[str, np.ndarray], idx: np.ndarray, gates: list[dict] | None = None,
                  track: np.ndarray | None = None) -> dict:
    ts = log['ts'][idx] - log['ts'][idx][0]
    dt = float(np.median(np.diff(ts)))
    P = np.c_[log['px'][idx], log['py'][idx], log['pz'][idx]]
    V = np.c_[log['vx'][idx], log['vy'][idx], log['vz'][idx]]
    air = (P[:, 2] > 0.5) & (log['phase'][idx] > 3.0)
    if air.sum() < 100:
        return {'airborne_s': float(air.sum() * dt)}
    a0 = int(np.argmax(air))
    sl = slice(a0, len(idx))
    roll, pitch = euler_deg(np.c_[log['qw'][idx], log['qx'][idx], log['qy'][idx], log['qz'][idx]])
    hp = lambda x: _highpass(x[sl], dt, 1.0)  # noqa: E731
    w = np.degrees(np.c_[log['wx'][idx], log['wy'][idx], log['wz'][idx]])
    speed = np.linalg.norm(V[:, :2], axis=1)
    d_in = np.abs(np.diff(np.c_[log['in_thr'][idx], log['in_roll'][idx], log['in_pitch'][idx], log['in_yaw'][idx]][sl],
                          axis=0)).mean(0)
    res = {
        'airborne_s': float((len(idx) - a0) * dt),
        'speed_median': float(np.median(speed[sl])), 'speed_p90': float(np.percentile(speed[sl], 90)),
        'horizon_shake_deg': float(np.sqrt(np.mean(hp(roll) ** 2 + hp(pitch) ** 2))),
        'rate_shake_dps': float(np.sqrt(np.mean(hp(w[:, 0]) ** 2 + hp(w[:, 1]) ** 2))),
        'yaw_shake_dps': float(np.sqrt(np.mean(hp(w[:, 2]) ** 2))),
        'vz_shake': float(np.sqrt(np.mean(hp(V[:, 2]) ** 2))),
        'input_chatter': [round(float(x), 4) for x in d_in],     # thr, roll, pitch, yaw per frame
        'z_range': [float(P[sl, 2].min()), float(P[sl, 2].max())],
    }
    if track is not None:
        e = path_error(P[sl], track)
        res.update({'path_err_mean': float(e.mean()), 'path_err_p90': float(np.percentile(e, 90))})
    if gates is not None:
        cr = gate_crossings(P, ts, gates)
        res['crossings'] = cr
        through = [c for c in cr if c['through']]
        res['gates_through'] = sorted({c['gate'] for c in through})
        if len(through) >= 2:
            t0, t1 = through[0]['t'], through[-1]['t']
            seg = (ts >= t0) & (ts <= t1)
            length = float(np.linalg.norm(np.diff(P[seg], axis=0), axis=1).sum())
            res.update({'first_last_gate': [through[0]['gate'], through[-1]['gate']], 'gate_span_s': t1 - t0,
                        'gate_span_speed': length / max(t1 - t0, 1e-6)})
    return res


def score_log(path: str, gates_json: str | None = 'configs/gates_strawbale.json',
              track_yaml: str | None = None) -> list[dict]:
    log = load_log(path)
    gates = json.load(open(gates_json, encoding='utf-8'))['gates'] if gates_json else None
    track = None
    if track_yaml:
        import yaml
        d = yaml.safe_load(open(track_yaml, encoding='utf-8'))
        track = np.array(d['waypoints'] if isinstance(d, dict) else d, dtype=float)[:, :3]
    return [score_attempt(log, idx, gates, track) for idx in attempts(log)]


def describe(name: str, results: list[dict]) -> str:
    lines = []
    for k, r in enumerate(results):
        if 'speed_median' not in r:
            lines.append(f'{name} attempt {k + 1}: airborne {r["airborne_s"]:.0f} s only')
            continue
        s = (f'{name} attempt {k + 1}: {r["airborne_s"]:.0f} s airborne, speed median {r["speed_median"]:.2f} '
             f'(p90 {r["speed_p90"]:.2f}) m/s | shake: horizon {r["horizon_shake_deg"]:.2f} deg, rates '
             f'{r["rate_shake_dps"]:.1f} deg/s, yaw {r["yaw_shake_dps"]:.1f} deg/s, vz {r["vz_shake"]:.2f} m/s, '
             f'input chatter thr/roll/pitch/yaw {r["input_chatter"]}')
        if 'path_err_mean' in r:
            s += f' | path error {r["path_err_mean"]:.2f} m (p90 {r["path_err_p90"]:.2f})'
        if 'gates_through' in r:
            s += f' | gates through {r["gates_through"]} ({len(r["gates_through"])}/7)'
            if 'gate_span_s' in r:
                g0, g1 = r['first_last_gate']
                s += f', gate {g0} -> {g1} in {r["gate_span_s"]:.0f} s at {r["gate_span_speed"]:.2f} m/s'
            s += '; ' + ', '.join(f'g{c["gate"]}@{c["t"]:.0f}s {c["lateral_m"]:+.1f}m{"" if c["through"] else " miss"}'
                                  for c in r['crossings'])
        lines.append(s)
    return '\n'.join(lines)
