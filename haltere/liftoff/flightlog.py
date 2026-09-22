"""Score a flight from a legacy ``fly`` or visual-brain telemetry CSV.

Two kinds of numbers. Progress: gates flown through (the gate list's crossing test), time and average speed between
the first and last gate, ground speed, distance to the taught line. Wobble, i.e. what makes the FPV view shake:
the horizon's roll and pitch above 1 Hz (degrees RMS; the drone's slow, deliberate banking is below it), the roll
and pitch rates above 1 Hz (deg/s RMS; the yaw rate is reported separately), the per-frame change of the processed input the flight controller received, and
vertical speed jitter. A flight that crashes is split at every reset; each attempt is scored on its own.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_log(path: str) -> dict[str, np.ndarray]:
    with open(path, newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    if not rows:
        return {}
    out = {}
    for k in rows[0].keys():
        if k in ('status', 'motor_controller'):
            out[k] = np.array([r[k] for r in rows], dtype=object)
            continue
        out[k] = np.array([{'True': 1., 'False': 0.}.get(r[k], r[k])
                           if r[k] not in ('', None) else np.nan for r in rows], dtype=float)
    # The visual runner records the same real pose/rates under these names.
    if 'shadow' in out:
        for dst, src in {'px': 'x', 'py': 'y', 'pz': 'z',
                         'wx': 'omega_x', 'wy': 'omega_y', 'wz': 'omega_z'}.items():
            if dst not in out and src in out:
                out[dst] = out[src]
    required = ('ts', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz',
                'wx', 'wy', 'wz', 'in_thr', 'in_roll', 'in_pitch', 'in_yaw')
    missing = set(required) - out.keys()
    if missing:
        raise ValueError(f'Flight log is missing telemetry columns: {", ".join(sorted(missing))}')
    # a log still being written ends in a partial row
    validity = required + tuple(k for k in ('phase', 'shadow', 'pilot_assisted') if k in out)
    ok = np.all([np.isfinite(out[k]) for k in validity], axis=0)
    out = {k: v[ok] for k, v in out.items()}
    if 'phase' not in out:
        if 'shadow' not in out:
            raise ValueError('Legacy flight log is missing phase')
        # Older visual logs omitted elapsed flight time. Their arming hold/ramp
        # finishes at 3 s; infer elapsed time separately for each reset.
        ts = out['ts']
        phase = np.zeros(len(ts))
        start = 0
        for i in range(len(ts)):
            if i and ts[i] < ts[i - 1] - .5:
                start = i
            phase[i] = ts[i] - ts[start]
        out['phase'] = phase
    return out


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


def body_up(q: np.ndarray) -> np.ndarray:
    """The world direction the propellers push in: the drone's own up axis, from its [w, x, y, z] quaternion."""
    w, x, y, z = np.asarray(q, dtype=float).T
    return np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], axis=1)


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
            # dz is measured against the passage point (where the human flew through), not the arch's floor: the
            # opening extends about a metre below it before the ground, and the round top limits the height
            out.append({'gate': i, 't': float(t[c]), 'lateral_m': lat, 'dz_m': dz,
                        'through': abs(lat) < half_width and -1.0 < dz < 2.5})
    return sorted(out, key=lambda p: p['t'])


def collisions(P: np.ndarray, V: np.ndarray, Q: np.ndarray, t: np.ndarray, airborne: np.ndarray,
               threshold: float = 20.0, unexplained: float = 6.0) -> list[dict]:
    """Impacts: the velocity changing faster than ``threshold`` m/s^2 (3-frame mean) while airborne, by a force the
    drone cannot have produced itself.

    Magnitude on its own does not mean contact, because a racing quad makes about 3 g of its own. Both events that
    have cost the odd-course bench a gate the drone actually flew through were the drone's own doing: a hard corner
    on home (1d3d62e), and on clockwise and gate_pair a throttle punch a second after the gate - 30 m/s^2 straight
    up while the drone sat dead centre in the arch, 1.7 m clear of the nearest post.

    What a contact has and a manoeuvre has not is a force from outside. The propellers can only push along the
    drone's own up axis, and they push, never pull, so in free flight the specific force (the acceleration less
    gravity, what an accelerometer reads) lies on that ray. Whatever distance is left to it is the arch, the bale or
    the ground. Ordinary flight leaves under 4 m/s^2 of it - drag, and the rehearsal's own drag model - while the
    contacts in the game logs leave 12 to 1100, so anything past ``unexplained`` is a contact and nothing between
    3 and 8 m/s^2 changes a single count on the 26 recorded flights or on the bench.

    This subsumes the corner test it replaces: a coordinated turn is thrust along a banked axis, and leaves nothing
    over. A tumble after hitting the ground still counts, and so does a clip that barely slows the drone.

    Where it is weakest is a sideways clip taken in a steep bank, because a thrust axis tilted into the blow absorbs
    ``cos(bank)`` of it. The margin is wide rather than absent: the weakest contact in the game logs leaves
    12 m/s^2 and would need 59 degrees of bank to fall under ``unexplained``, and contacts there arrive at a median
    bank of 10 degrees. A push straight up the thrust axis is invisible to this test by construction, which the
    course generator cannot produce - its posts push sideways and its top bars down, and ground contact is below
    the ``airborne`` floor - but a real bale top could.
    """
    dt = np.maximum(np.diff(t), 1e-3)
    A = np.diff(V, axis=0) / dt[:, None]                       # acceleration vector, m/s^2
    k = np.ones(3) / 3
    A = np.stack([np.convolve(A[:, i], k, mode='same') for i in range(3)], axis=1)
    acc = np.linalg.norm(A, axis=1)
    spec = A + np.array([0.0, 0.0, 9.81])                      # the specific force: acceleration less gravity
    up = body_up(Q)[:len(A)]
    thrust = np.maximum(np.einsum('ij,ij->i', spec, up), 0.0)  # the drone's own share of it
    left = np.linalg.norm(spec - thrust[:, None] * up, axis=1)  # and what no propeller of its could have made
    out, last = [], -1e9
    for e in np.where(airborne[1:] & (acc > threshold) & (left > unexplained))[0]:
        if t[e] - last < 1.0:
            if out:
                out[-1]['peak'] = max(out[-1]['peak'], float(acc[e]))
            continue
        last = t[e]
        out.append({'t': float(t[e]), 'pos': [round(float(x), 1) for x in P[e]], 'peak': float(acc[e]),
                    'unexplained': float(left[e]),
                    'speed_before': float(np.linalg.norm(V[max(e - 10, 0)])),
                    'speed_after': float(np.linalg.norm(V[min(e + 10, len(V) - 1)]))})
    return out


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
    attribution = {}
    if 'shadow' in log:
        shadow = log['shadow'][idx].astype(bool)
        attribution['control_mode'] = ('shadow: no control output' if shadow.all() else
                                       'live control' if not shadow.any() else 'mixed shadow/live')
        if 'motor_controller' in log:
            motors = np.unique(log['motor_controller'][idx])
            attribution['motor_controller'] = motors[0] if len(motors) == 1 else 'mixed'
            if not shadow.any() and attribution['motor_controller'] == 'pd':
                attribution['control_mode'] = 'PD motor baseline; brain in shadow'
        if 'pilot_assisted' in log:
            assisted = log['pilot_assisted'][idx].astype(bool)
            attribution['pilot_assistance'] = ('rabbit' if assisted.all() else
                                               'none' if not assisted.any() else 'mixed')
            if 'pilot_kind' in log:
                kinds = np.unique(log['pilot_kind'][idx])
                attribution['pilot_assistance'] = ({0:'none',1:'rabbit',2:'race-cue'}.get(kinds[0], 'unknown')
                                                   if len(kinds)==1 else 'mixed')
    ts = log['ts'][idx] - log['ts'][idx][0]
    dt = float(np.median(np.diff(ts)))
    P = np.c_[log['px'][idx], log['py'][idx], log['pz'][idx]]
    V = np.c_[log['vx'][idx], log['vy'][idx], log['vz'][idx]]
    Q = np.c_[log['qw'][idx], log['qx'][idx], log['qy'][idx], log['qz'][idx]]
    air = (P[:, 2] > 0.5) & (log['phase'][idx] > 3.0)
    if air.sum() < 100:
        return dict(attribution, airborne_s=float((np.diff(ts)*air[:-1]).sum()))
    a0 = int(np.argmax(air))
    sl = slice(a0, len(idx))
    roll, pitch = euler_deg(Q)
    hp = lambda x: _highpass(x[sl], dt, 1.0)  # noqa: E731
    w = np.degrees(np.c_[log['wx'][idx], log['wy'][idx], log['wz'][idx]])
    speed = np.linalg.norm(V[:, :2], axis=1)
    d_in = np.abs(np.diff(np.c_[log['in_thr'][idx], log['in_roll'][idx], log['in_pitch'][idx], log['in_yaw'][idx]][sl],
                          axis=0)).mean(0)
    res = {
        **attribution,
        'airborne_s': float(ts[-1]-ts[a0]),
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
        res['n_gates'] = len(gates)
    hits = collisions(P, V, Q, ts, air)
    res['collisions'] = hits
    if gates is not None:
        cr = gate_crossings(P, ts, gates)
        for c in cr:                                   # a bounce off the arch is not a pass
            g = np.asarray(gates[c['gate']]['pos'], dtype=float)
            c['hit'] = any(abs(h['t'] - c['t']) < 1.0 and np.linalg.norm(np.asarray(h['pos'])[:2] - g[:2]) < 5.0 for h in hits)
            c['through'] = c['through'] and not c['hit']
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


def score_log(path: str, gates_json: str | None = None,
              track_yaml: str | None = None) -> list[dict]:
    log = load_log(path)
    if not log or not len(log['ts']):
        return []
    gates = json.load(open(gates_json, encoding='utf-8'))['gates'] if gates_json else None
    track = None
    if track_yaml:
        import yaml
        d = yaml.safe_load(open(track_yaml, encoding='utf-8'))
        track = np.array(d['waypoints'] if isinstance(d, dict) else d, dtype=float)[:, :3]
    results = [score_attempt(log, idx, gates, track) for idx in attempts(log)]
    sidecar = Path(path).with_suffix('.json')
    has_reset = np.any(np.diff(log['ts']) < -.5)
    if 'shadow' in log and len(results) == 1 and not has_reset and sidecar.is_file():
        try:
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        except json.JSONDecodeError:  # the runner may still be writing it
            metadata = {}
        # A visual runner stops on reset. Do not attach one terminal event to
        # combined attempts or a sidecar from a differently sized recording.
        if isinstance(metadata, dict) and metadata.get('ticks') == len(log['ts']):
            if isinstance(metadata.get('stop_reason'), str):
                results[0]['stop_reason'] = metadata['stop_reason']
            if isinstance(metadata.get('impact'), dict):
                # The guard stops before the impact sample is written to CSV.
                # Preserve its evidence separately from CSV contact estimates;
                # adding the counts would risk counting the same event twice.
                results[0]['terminal_impact'] = metadata['impact']
    return results


def describe(name: str, results: list[dict]) -> str:
    lines = []
    for k, r in enumerate(results):
        mode = f' [{r["control_mode"]}; pilot={r.get("pilot_assistance", "unspecified")}]' if 'control_mode' in r else ''
        label = f'{name} attempt {k + 1}{mode}'
        stop = f' | stop: {r["stop_reason"]}' if 'stop_reason' in r else ''
        if 'speed_median' not in r:
            terminal = ' | terminal impact recorded' if 'terminal_impact' in r else ''
            lines.append(f'{label}: airborne {r["airborne_s"]:.0f} s only{terminal}{stop}')
            continue
        s = (f'{label}: {r["airborne_s"]:.0f} s airborne, speed median {r["speed_median"]:.2f} '
             f'(p90 {r["speed_p90"]:.2f}) m/s | shake: horizon {r["horizon_shake_deg"]:.2f} deg, rates '
             f'{r["rate_shake_dps"]:.1f} deg/s, yaw {r["yaw_shake_dps"]:.1f} deg/s, vz {r["vz_shake"]:.2f} m/s, '
             f'input chatter thr/roll/pitch/yaw {r["input_chatter"]}')
        n_contacts = len(r.get('collisions', []))
        s += (f' | terminal impact recorded; CSV contact estimates {n_contacts}' if 'terminal_impact' in r
              else f' | estimated contacts {n_contacts}') + ''.join(
            f' @{h["t"]:.0f}s({h["pos"][0]:.0f},{h["pos"][1]:.0f},{h["pos"][2]:.0f}) {h["speed_before"]:.1f}->'
            f'{h["speed_after"]:.1f}m/s [{h["unexplained"]:.0f} off-axis]'
            for h in r.get('collisions', [])[:6])
        if 'path_err_mean' in r:
            s += f' | path error {r["path_err_mean"]:.2f} m (p90 {r["path_err_p90"]:.2f})'
        if 'gates_through' in r:
            s += f' | gates through {r["gates_through"]} ({len(r["gates_through"])}/{r.get("n_gates", "?")})'
            if 'gate_span_s' in r:
                g0, g1 = r['first_last_gate']
                s += f', gate {g0} -> {g1} in {r["gate_span_s"]:.0f} s at {r["gate_span_speed"]:.2f} m/s'
            s += '; ' + ', '.join(f'g{c["gate"]}@{c["t"]:.0f}s {c["lateral_m"]:+.1f}m{"" if c["through"] else (" hit" if c.get("hit") else " miss")}'
                                  for c in r['crossings'])
        lines.append(s + stop)
    return '\n'.join(lines)
