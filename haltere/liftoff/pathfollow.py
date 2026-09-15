"""Following a taught line fast: projection, a speed profile, and the brain's control frame.

The path pilot used to run a carrot at a fixed speed and let the nose look ahead. Two things limit that at speed.
The brain reads its goal and its drift in its own body frame, so when the nose looks into a bend the carrot appears
on the outside of the turn and the brain asks for no inward roll: the drone drifts wide and clips arches and
obstacles near the line. And the brain cruises at a speed it senses, not at the carrot's speed.

``PathFollower`` measures progress by projecting the drone onto the line, builds a speed profile from the line's
curvature (lateral acceleration limit, braking and acceleration passes, caps at gates), commands that speed through
the gain on the brain's speed senses, places the carrot so that its lateral offset stays in the linear part of the
brain's goal encoding, and gives the brain a control frame aligned with the line's tangent: its horizontal senses
are rotated into that frame and its roll and pitch commands rotated back, so the camera can look ahead without
changing how the brain steers.
"""
from __future__ import annotations

import numpy as np


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class PathFollower:
    def __init__(self, waypoints: np.ndarray, loop: bool = True, ds: float = 0.5, v_max: float = 5.0,
                 a_lat: float = 1.5, a_brake: float = 1.0, a_acc: float = 0.8, gates: list | None = None,
                 v_gate: float = 0.0, cruise_sensed: float = 2.45, flow_min: float = 0.4, height_above_line: float = 1.2,
                 obstacles: list | None = None, clearance: float = 1.8):
        P = np.asarray(waypoints, dtype=np.float64)[:, :3]
        if loop and len(P) > 10:
            # a lap taught by hand usually runs on past its own start: close it where the tail passes the start instead
            # of jumping back (the Straw Bale lap's last 33 m overlap its first, which made the carrot reverse)
            tail = np.arange(int(0.7 * len(P)), len(P))
            j = int(tail[np.argmin(np.linalg.norm(P[tail] - P[0], axis=1))])
            if np.linalg.norm(P[j] - P[0]) < 8.0:
                P = P[:j + 1] if np.linalg.norm(P[j] - P[0]) > 1.0 else P[:j]
        if loop:
            P = np.vstack([P, P[:1]])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s_v = np.r_[0.0, np.cumsum(seg)]
        self.length = float(s_v[-1])
        self.loop = loop
        self.ds = ds
        self.s = np.arange(0.0, self.length, ds)
        self.xyz = np.stack([np.interp(self.s, s_v, P[:, k]) for k in range(3)], axis=1)
        n = len(self.s)
        self.shift = np.zeros(n)
        if obstacles:
            # bend the line away from known obstacles (arch legs a hand-flown line passes a metre from): keep `clearance`
            # horizontally from each, shifting smoothly over +-10 m of line; an arch is 4 m wide, so clearing a leg by
            # 1.8 m centres the line in it
            disp = np.zeros((n, 2))
            for o in obstacles:
                o = np.asarray(o, dtype=np.float64)
                dxy = self.xyz[:, :2] - o[:2]
                d = np.linalg.norm(dxy, axis=1)
                d = np.where(np.abs(self.xyz[:, 2] - o[2]) < 1.2, d, np.inf)   # above or below the line: a height matter
                j = int(np.argmin(d))
                need = clearance - d[j]
                if not np.isfinite(d[j]) or need <= 0:
                    continue
                u = dxy[j] / max(d[j], 1e-6)
                rel = self.s - self.s[j]
                if loop:
                    rel = (rel + self.length / 2) % self.length - self.length / 2
                w = 0.5 * (1 + np.cos(np.pi * np.clip(rel / 10.0, -1, 1)))
                disp += need * w[:, None] * u
            self.xyz[:, :2] += disp
            self.shift = np.linalg.norm(disp, axis=1)
        # heading from a centred difference over +-3 m, curvature from its derivative, both smoothed
        k = max(1, int(round(3.0 / ds)))
        idx = np.arange(n)
        nxt = (idx + k) % n if loop else np.minimum(idx + k, n - 1)
        prv = (idx - k) % n if loop else np.maximum(idx - k, 0)
        d = self.xyz[nxt, :2] - self.xyz[prv, :2]
        self.psi = np.arctan2(d[:, 1], d[:, 0])
        dpsi = np.array([wrap(a) for a in (self.psi[nxt] - self.psi[prv])])
        span = np.maximum((nxt - prv) % n if loop else (nxt - prv), 1) * ds
        kappa = np.abs(dpsi) / span
        self.kappa = kappa
        # the curvature the drone must respect soon: max over [s - 2, s + 4]
        w0, w1 = int(round(2.0 / ds)), int(round(4.0 / ds))
        self.kappa_eff = np.array([kappa[np.arange(i - w0, i + w1 + 1) % n].max() if loop else
                                   kappa[max(0, i - w0):min(n, i + w1 + 1)].max() for i in range(n)])
        self.slope = np.gradient(self.xyz[:, 2], ds)
        self.v_max, self.a_lat, self.a_brake, self.a_acc = v_max, a_lat, a_brake, a_acc
        self.cruise_sensed, self.flow_min, self.height_above_line = cruise_sensed, flow_min, height_above_line
        self.gate_s = []
        if gates:
            for g in gates:
                self.gate_s.append(float(self.s[int(np.argmin(np.linalg.norm(self.xyz - np.asarray(g), axis=1)))]))
        self.v_gate = v_gate
        self.v_ref = self._profile()
        self.s_p = None
        self.flow = 1.0

    # ------------------------------------------------------------------ geometry
    def _i(self, s: float) -> int:
        if self.loop:
            return int(round((s % self.length) / self.ds)) % len(self.s)
        return int(np.clip(round(s / self.ds), 0, len(self.s) - 1))

    def point(self, s: float) -> np.ndarray:
        return self.xyz[self._i(s)].copy()

    def heading(self, s: float) -> float:
        return float(self.psi[self._i(s)])

    def _profile(self) -> np.ndarray:
        n = len(self.s)
        v = np.minimum(self.v_max, np.sqrt(self.a_lat / np.maximum(self.kappa_eff, 1e-6)))
        if self.v_gate > 0:
            for gs in self.gate_s:
                for i in range(n):
                    rel = (self.s[i] - gs + self.length / 2) % self.length - self.length / 2 if self.loop else self.s[i] - gs
                    if -8.0 <= rel <= 3.0:
                        v[i] = min(v[i], self.v_gate)
        reps = 2 if self.loop else 1
        for _ in range(reps):                       # braking: v[i]^2 <= v[i+1]^2 + 2 a ds, run backwards (twice round a loop)
            for i in range(n - 1, -1, -1):
                j = (i + 1) % n if self.loop else min(i + 1, n - 1)
                v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * self.a_brake * self.ds))
        for _ in range(reps):                       # acceleration forwards
            for i in range(n):
                j = (i - 1) % n if self.loop else max(i - 1, 0)
                v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * self.a_acc * self.ds))
        return v

    def project(self, pos: np.ndarray) -> float:
        """Arc length of the drone's projection on the line, searched forward from the last one."""
        if self.s_p is None:
            i = int(np.argmin(np.linalg.norm(self.xyz - pos, axis=1)))
            self.s_p = float(self.s[i])
            return self.s_p
        lo, hi = int(round(-2.0 / self.ds)), int(round(10.0 / self.ds))
        base = self._i(self.s_p)
        cand = (base + np.arange(lo, hi + 1)) % len(self.s) if self.loop else np.clip(base + np.arange(lo, hi + 1), 0, len(self.s) - 1)
        j = cand[int(np.argmin(np.linalg.norm(self.xyz[cand, :2] - pos[:2], axis=1)))]
        step = (self.s[j] - self.s[base] + self.length / 2) % self.length - self.length / 2 if self.loop else self.s[j] - self.s[base]
        self.s_p = self.s_p + float(step)
        return self.s_p

    # ------------------------------------------------------------------ per tick
    def update(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> dict:
        s_p = self.project(pos)
        v = float(np.linalg.norm(vel[:2]))
        v_ref = float(self.v_ref[self._i(s_p + 0.6 * v)])
        k_eff = float(self.kappa_eff[self._i(s_p + 0.6 * v)])
        d_c = float(np.clip(min(5.5, np.sqrt(2.0 / max(k_eff, 1e-6))) - 2.0 * max(0.0, v - v_ref), 2.5, 5.5))
        carrot = self.point(s_p + d_c)
        carrot[2] = self.point(s_p + 1.0)[2] + float(self.slope[self._i(s_p)]) * 0.3 * v
        flow_des = float(np.clip(self.cruise_sensed / max(v_ref, 1e-3), self.flow_min, 1.0))
        self.flow += (flow_des - self.flow) * min(1.0, dt / 0.5)
        psi_c = self.heading(s_p + 0.3 * v)
        alt_line = float(np.clip(self.height_above_line + pos[2] - self.point(s_p)[2], 0.8, 3.5))
        return {'s': s_p, 'carrot': carrot, 'v_ref': v_ref, 'flow': self.flow, 'psi_c': psi_c, 'd_c': d_c,
                'alt': alt_line, 'kappa': k_eff}


def rotate_senses(s: dict, rel_b: np.ndarray, delta: float, psi_c: float, torch_mod) -> tuple[dict, np.ndarray]:
    """Express the horizontal body-frame senses in a frame yawed by delta from the nose (the path tangent)."""
    c, sn = np.cos(delta), np.sin(delta)

    def rot(t):
        x = t[:, 0].clone()
        y = t[:, 1].clone()
        out = t.clone()
        out[:, 0] = c * x + sn * y
        out[:, 1] = -sn * x + c * y
        return out
    s = dict(s)
    for k in ('gyro', 'gravity_body', 'vel_body'):
        s[k] = rot(s[k])
    s['yaw'] = torch_mod.full_like(s['yaw'], psi_c)
    r = np.array([c * rel_b[0] + sn * rel_b[1], -sn * rel_b[0] + c * rel_b[1], rel_b[2]])
    return s, r


def rotate_commands(roll_v: float, pitch_v: float, delta: float) -> tuple[float, float]:
    """Roll and pitch asked for in the tangent frame -> roll and pitch about the drone's body axes.
    The tilt request (pitch, -roll) is a horizontal vector: body = Rz(delta) * frame."""
    c, sn = np.cos(delta), np.sin(delta)
    pitch = c * pitch_v + sn * roll_v
    roll = -sn * pitch_v + c * roll_v
    return roll, pitch
