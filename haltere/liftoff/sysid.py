"""System identification: fit the simulator to a Liftoff telemetry recording.

Two stages:
1. Sign/axis inference by correlation (which way each stick turns the drone, which telemetry gyro
   component is which body axis).
2. Gradient descent through the differentiable simulator on short windows of the recording,
   fitting thrust, thrust curve, motor lag, drag and the per-axis controller gains.
"""
from __future__ import annotations

import csv
import math
from dataclasses import asdict

import numpy as np
import torch

from ..sim.controller import RateControllerParams
from ..sim.quad import QuadParams, QuadState
from ..sim.vehicle import RatesConfig, Vehicle
from .frames import omega_from_quats, unity_quat_to_sim, unity_vec_to_sim
from .pilot import LiftoffMapping
from .telemetry import TelemetryFrame


def load_recording(path: str) -> dict[str, np.ndarray]:
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append([float(row[c]) for c in TelemetryFrame.columns()])
    a = np.asarray(rows, dtype=np.float64)
    cols = {c: i for i, c in enumerate(TelemetryFrame.columns())}
    T = len(a)
    pos_u = a[:, [cols['px'], cols['py'], cols['pz']]]
    vel_u = a[:, [cols['vx'], cols['vy'], cols['vz']]]
    q_u = a[:, [cols['qx'], cols['qy'], cols['qz'], cols['qw']]]
    q_s = np.stack([unity_quat_to_sim(q) for q in q_u])
    # keep quaternion sign continuous for finite differences
    for i in range(1, T):
        if np.dot(q_s[i], q_s[i - 1]) < 0:
            q_s[i] = -q_s[i]
    t = a[:, cols['timestamp']]
    omega = np.zeros((T, 3))
    for i in range(1, T):
        dt = t[i] - t[i - 1]
        omega[i] = omega_from_quats(q_s[i - 1], q_s[i], dt) if 0 < dt < 0.1 else omega[i - 1]
    return {
        't': t,
        'pos': unity_vec_to_sim(pos_u), 'vel': unity_vec_to_sim(vel_u), 'quat': q_s, 'omega': omega,
        'gyro': a[:, [cols['gyro_pitch'], cols['gyro_roll'], cols['gyro_yaw']]],
        'input': a[:, [cols['in_throttle'], cols['in_yaw'], cols['in_pitch'], cols['in_roll']]],
        'rpm': a[:, [cols['rpm_lf'], cols['rpm_rf'], cols['rpm_lb'], cols['rpm_rb']]],
        'ground_z': unity_vec_to_sim(pos_u)[0, 2],
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def infer_mapping(rec: dict[str, np.ndarray]) -> tuple[LiftoffMapping, dict]:
    """Stick signs and gyro axis mapping from correlations between inputs and measured body rates."""
    om = rec['omega']
    inp = rec['input']
    stick = {'roll': inp[:, 3], 'pitch': inp[:, 2], 'yaw': inp[:, 1]}
    c_r = _corr(stick['roll'], om[:, 0])
    c_p = _corr(stick['pitch'], om[:, 1])
    c_y = _corr(stick['yaw'], om[:, 2])
    sgn = lambda c: 1.0 if c >= 0 else -1.0  # noqa: E731
    # simulator: roll+ -> +omega_x, pitch+ -> +omega_y, yaw+ -> -omega_z
    stick_sign = (sgn(c_r), sgn(c_p), -sgn(c_y))
    gyro_axis, gyro_sign = [], []
    corr_table = np.zeros((3, 3))
    for k in range(3):  # body axis
        for j in range(3):  # telemetry component
            corr_table[k, j] = _corr(om[:, k], np.deg2rad(rec['gyro'][:, j]))
        j = int(np.argmax(np.abs(corr_table[k])))
        gyro_axis.append(j)
        gyro_sign.append(sgn(corr_table[k, j]))
    notes = {'corr_roll': c_r, 'corr_pitch': c_p, 'corr_yaw': c_y, 'gyro_corr': corr_table.round(3).tolist(),
             'throttle_range': [float(inp[:, 0].min()), float(inp[:, 0].max())],
             'max_rpm_seen': float(rec['rpm'].max())}
    m = LiftoffMapping(stick_sign=stick_sign, gyro_axis=tuple(gyro_axis), gyro_sign=tuple(gyro_sign),
                       max_rpm=max(float(rec['rpm'].max()), 1.0), notes=notes)
    return m, notes


def to_sim_sticks(inp: np.ndarray, mapping: LiftoffMapping) -> np.ndarray:
    """Liftoff Input rows (throttle, yaw, pitch, roll) -> simulator sticks (throttle, roll, pitch, yaw)."""
    s = mapping.stick_sign
    return np.stack([inp[:, 0], inp[:, 3] * s[0], inp[:, 2] * s[1], inp[:, 1] * s[2]], axis=1)


def make_windows(rec: dict[str, np.ndarray], window: int = 100, stride: int = 50, min_alt: float = 0.3):
    T = len(rec['t'])
    alt = rec['pos'][:, 2] - rec['ground_z']
    idx = []
    for s in range(0, T - window, stride):
        seg_t = rec['t'][s:s + window]
        if np.any(np.diff(seg_t) <= 0) or np.any(np.diff(seg_t) > 0.05):
            continue  # reset or gap inside the window
        if alt[s:s + window].min() < min_alt:
            continue
        idx.append(s)
    return np.asarray(idx)


def fit_dynamics(rec: dict[str, np.ndarray], mapping: LiftoffMapping, quad: QuadParams, ctl: RateControllerParams,
                 rates: RatesConfig, iters: int = 300, window: int = 100, device='cuda', verbose=True) -> dict:
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    starts = make_windows(rec, window)
    if len(starts) == 0:
        raise RuntimeError('no usable airborne windows in the recording (fly for a while above 0.3 m)')
    dt = float(np.median(np.diff(rec['t'])))
    B = len(starts)
    sticks_all = to_sim_sticks(rec['input'], mapping)
    f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
    ground = np.array([0.0, 0.0, rec['ground_z']])
    pos = f32(np.stack([rec['pos'][s:s + window] - ground for s in starts]))  # [B,W,3], z relative to ground
    vel = f32(np.stack([rec['vel'][s:s + window] for s in starts]))
    om = f32(np.stack([rec['omega'][s:s + window] for s in starts]))
    quat0 = f32(np.stack([rec['quat'][s] for s in starts]))
    sticks = f32(np.stack([sticks_all[s:s + window] for s in starts]))      # [B,W,4]
    rpm0 = f32(np.stack([rec['rpm'][s] for s in starts])) / mapping.max_rpm

    veh = Vehicle(quad, ctl, rates, device, dt=dt, substeps=4)
    sim = veh.sim
    # learnable physical parameters (log-parametrised)
    lp = {
        'twr': torch.tensor(math.log(quad.twr), device=device, requires_grad=True),
        'thrust_exp': torch.tensor(math.log(quad.thrust_exp), device=device, requires_grad=True),
        'motor_tau': torch.tensor(math.log(quad.motor_tau), device=device, requires_grad=True),
        'drag_lin': torch.log(f32(quad.drag_lin)).requires_grad_(True),
        'drag_quad': torch.log(f32(quad.drag_quad)).requires_grad_(True),
        'axis_gain': torch.zeros(3, device=device, requires_grad=True),
        'yaw_coeff': torch.tensor(math.log(quad.yaw_coeff), device=device, requires_grad=True),
    }
    opt = torch.optim.Adam(lp.values(), lr=0.03)

    def apply_params():
        sim.twr = torch.exp(lp['twr'])
        sim.thrust_exp = torch.exp(lp['thrust_exp'])
        sim.motor_tau = torch.exp(lp['motor_tau'])
        sim.drag_lin = torch.exp(lp['drag_lin'])
        sim.drag_quad = torch.exp(lp['drag_quad'])
        sim.yaw_coeff = torch.exp(lp['yaw_coeff'])
        veh.ctl.axis_gain = torch.exp(lp['axis_gain'])

    hist = []
    for it in range(iters):
        apply_params()
        st = QuadState(pos[:, 0], vel[:, 0], quat0, om[:, 0], rpm0.clamp(0.0, 1.0),
                       torch.zeros(B, dtype=torch.bool, device=device))
        vs = veh.wrap(st)
        lp_, lv, lo = 0.0, 0.0, 0.0
        for k in range(1, window):
            vs = veh.step(vs, sticks[:, k - 1])
            lp_ = lp_ + (vs.quad.pos - pos[:, k]).pow(2).sum(1).mean()
            lv = lv + (vs.quad.vel - vel[:, k]).pow(2).sum(1).mean()
            lo = lo + (vs.quad.omega - om[:, k]).pow(2).sum(1).mean()
        loss = (lp_ + 0.2 * lv + 0.02 * lo) / window
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
        if verbose and (it % 20 == 0 or it == iters - 1):
            print(f'fit it {it:4d} loss {float(loss):.4f} twr {float(torch.exp(lp["twr"])):.2f} '
                  f'exp {float(torch.exp(lp["thrust_exp"])):.2f} tau {float(torch.exp(lp["motor_tau"])):.3f} '
                  f'axis_gain {[round(float(x), 2) for x in torch.exp(lp["axis_gain"])]}', flush=True)
    out_quad = asdict(quad)
    out_quad.update({'twr': float(torch.exp(lp['twr'])), 'thrust_exp': float(torch.exp(lp['thrust_exp'])),
                     'motor_tau': float(torch.exp(lp['motor_tau'])),
                     'drag_lin': [float(x) for x in torch.exp(lp['drag_lin'])],
                     'drag_quad': [float(x) for x in torch.exp(lp['drag_quad'])],
                     'yaw_coeff': float(torch.exp(lp['yaw_coeff']))})
    out_ctl = asdict(ctl)
    out_ctl['axis_gain'] = [float(x) for x in torch.exp(lp['axis_gain'])]
    return {'quad': out_quad, 'ctl': out_ctl, 'loss': hist, 'windows': int(B), 'dt': dt}
