"""Surrogate braking gates for fast brain motors (development checks in the measured surrogate; not flight evidence).

Runs the tests declared in a frozen gate file (configs/brain09_gates.json) for one motor controller: a fast-contract
brain checkpoint, or 'pd' (FastMotorPD with the reference brain's calibration), writes the measurements and, for a
frozen file, the verdict of every gate. Tests (all plants: IdentifiedSim with the measured profile, 3-tick command
delay, quadratic drag 0.0075, as haltere.train.fast_motor_tracking.rollout; brain retina blanked per contract):

- hover: scripted straight line from a hover at `height` (no pilot). 'cap': the request slews to the cruise speed
  (command acceleration with the pilot's 0.25 s taper), at cap_at_s a governor-like cap drops it to X at brake_slew
  (no taper) for cap_hold_s, then it is released. 'sustained': the request slews to X and holds it. A height loop
  (vz = vz_gain (height - z), clipped to +-1 m/s) keeps the drone level; feedforward is the pilot's 0.05 s low-pass
  of the request derivative; yaw stick 0.
- live: the brain is warmed for warm_s on the exact recorded inputs of a logged flight up to t0; the plant starts from
  the logged state at t0. For pre_s the logged horizontal request at t0 is held, then it drops to X at brake_slew and
  is held hold_s (height loop to the logged height at t0; yaw 0). One drone per target X.
- swaps: warmed the same way, the logged request of the window is replayed from the logged state for up to
  horizon_s (cut before the logged impact) with the logged yaw stick; modes: this controller, FastMotorPD (its stick
  filter warmed on the logged trajectory) and the logged commands open loop (plant validity).
- rollout_speed: fast_motor_tracking.rollout with pilot speed S on the declared courses; cruise excess over the
  request along the track on aligned, level cruise ticks (rollout brake metrics).
- in_course_caps: rollout on the declared courses with SyntheticCaps on every drone, held-out cap seed.
- evaluation8: the distillation's own 8-course evaluation rollout.
- regression16: the 16-course switch harness (scratch tool, pinned by sha256) run as a subprocess on this code
  tree, plus steep-sink tracking and descending gate passes from its trace.
- audit: the weights changed against the parent are only readout rows 0-2 and their biases.

usage: python -m haltere.train.brake_gates CHECKPOINT|pd --gates configs/brain09_gates.json --out report.json
       [--parts hover,live,swaps,rollout_speed,in_course_caps,evaluation8,regression16,audit] [--run RUN_DIR]
"""
from __future__ import annotations

import argparse
from collections import deque
import datetime as _dt
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .bptt import load_checkpoint
from .fast_motor_tracking import SyntheticCapsConfig, brain_observation, rollout
from ..brain.motor_baseline import FastMotorPD
from ..brain.retina import RETINA_DIM
from ..liftoff.fast_rehearsal import synthetic_course
from ..sim.identified import IdentifiedSim
from ..vision.datasets import sha256

META_KEYS = ('frozen', 'frozen_at', 'sha256')
PARTS = ('hover', 'live', 'swaps', 'rollout_speed', 'in_course_caps', 'evaluation8', 'regression16', 'audit')
DT = .01
CONTRACT_KEYS = ('nominal_speed_mps', 'goal_seconds', 'velocity_scale', 'vertical_goal_seconds')


def gates_sha256(obj):
    body = {k: v for k, v in obj.items() if k not in META_KEYS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def load_gates(path, require_frozen=False):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    sha = gates_sha256(obj)
    if obj.get('frozen') and obj.get('sha256') != sha:
        raise RuntimeError(f'{path} changed after freezing (sha256 mismatch): bump its version and freeze again')
    if require_frozen and not obj.get('frozen'):
        raise RuntimeError(f'{path} is not frozen')
    return obj, sha


def freeze(path):
    """Mark a gate file frozen with its content sha256 (refuses a frozen file that changed)."""
    obj, sha = load_gates(path)
    if not obj.get('frozen'):
        obj.update(frozen=True, frozen_at=_dt.datetime.now().isoformat(timespec='seconds'), sha256=sha)
        Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    return obj


class Controller:
    """A fast-contract brain checkpoint, or FastMotorPD with the reference brain's calibration and contract."""

    def __init__(self, spec, reference):
        self.kind = 'pd' if spec == 'pd' else 'brain'
        self.path = reference if spec == 'pd' else spec
        self.brain, self.cfg, _ = load_checkpoint(self.path, 'cpu')
        self.brain.eval()
        self.meta = torch.load(self.path, map_location='cpu', weights_only=True)['visual_brain']
        self.contract = self.meta['fast_motor_tracking']
        self.calibration = self.meta['calibration']
        self.W = self.brain.inference_matrix()
        self.channels = list(self.brain.channel_dims)
        if self.kind == 'brain' and self.contract.get('recorded_scene_currents') is not False:
            raise ValueError('The braking gates are declared for brains without scene currents (retina blanked)')

    def expand(self, state, batch):
        return {'v': state['v'].repeat(1, batch), 'act': state['act'].repeat(batch, 1)}

    def act(self, senses, motor, request, state):
        obs = brain_observation(self.meta, senses, motor, self.cfg.task, torch.zeros(len(request), RETINA_DIM),
                                request, self.contract)
        action, state, _ = self.brain(obs, state, self.W)
        return action, state


def _feedforward(request, dt=DT):
    """The pilot's 0.05 s low-pass of the request derivative, per tick (T, B, 3)."""
    ff = np.zeros_like(request)
    alpha = 1-np.exp(-dt/.05)
    for k in range(1, len(request)):
        ff[k] = ff[k-1]+alpha*((request[k]-request[k-1])/dt-ff[k-1])
    return ff


def _drop(start, target, rate, n):
    """Speed schedule falling from `start` to `target` at `rate` m/s^2 (no taper), n ticks."""
    return np.maximum(target, start-rate*DT*np.arange(1, n+1)) if start > target else np.full(n, float(target))


def _slew(start, goal, n, acceleration=10., tc=.25):
    """The pilot's straight-line request slew with its 0.25 s taper, n ticks."""
    out, cur = np.zeros(n), float(start)
    for k in range(n):
        chord = goal-cur
        step = np.sign(chord)*min(abs(chord), acceleration*DT, abs(chord)*DT/tc)
        cur += step
        out[k] = cur
    return out


def _fly(ctl, profile, sim_state_fn, horizontal, height_ref, feedforward, *, brain_state=None, queue=None, yaw=None,
         vz_gain=.5, pd_warm=None, controller=None):
    """Closed loop of one batch: horizontal (T, B, 2) world request per tick, vertical from the height loop;
    returns velocity (T, B, 3) and height (T, B)."""
    T, B = horizontal.shape[:2]
    sim = IdentifiedSim(profile, ctl.calibration, 'cpu', DT)
    sim.randomize(B, 0.)
    s = sim_state_fn(sim, B)
    controller = controller or ctl.kind
    pd = FastMotorPD(profile, ctl.calibration)
    pd.reset()
    if pd_warm is not None:
        pd_warm(pd)
    idle = torch.tensor([[-1., 0., 0., 0.]]).repeat(B, 1)
    queue = deque(queue) if queue is not None else deque(idle.clone() for _ in range(3))
    request = np.zeros((T, B, 3))
    request[..., :2] = horizontal
    velocities, heights = np.zeros((T, B, 3)), np.zeros((T, B))
    state = brain_state
    with torch.no_grad():
        for k in range(T):
            senses = sim.sensors(s)
            z = s.quad.pos[:, 2].numpy().astype(float)
            request[k, :, 2] = np.clip(vz_gain*(height_ref-z), -1., 1.)
            r = torch.tensor(request[k], dtype=torch.float32)
            ff = torch.tensor(feedforward[k], dtype=torch.float32)
            if controller == 'pd':
                command = pd.command(senses, r, ff, dt=DT).clone()
            else:
                if state is None:
                    state = ctl.brain.init_state(B)
                    obs = brain_observation(ctl.meta, senses, s.quad.motor.mean(-1, keepdim=True), ctl.cfg.task,
                                            torch.zeros(B, RETINA_DIM), r, ctl.contract)
                    for _ in range(50):
                        _, state, _ = ctl.brain(obs, state, ctl.W)
                command, state = ctl.act(senses, s.quad.motor.mean(-1, keepdim=True), r, state)
                command = command.clone()
            command[:, 3] = 0. if yaw is None else torch.as_tensor(yaw[k], dtype=torch.float32)
            queue.append(command)
            s = sim.step(s, queue.popleft())
            v = s.quad.vel
            s.quad.vel = v-.0075*v.norm(dim=-1, keepdim=True)*v*DT
            s.quad.crashed[:] = False
            velocities[k] = s.quad.vel.numpy()
            heights[k] = s.quad.pos[:, 2].numpy()
    return velocities, heights


def hover_tests(ctl, profile, spec):
    """Scripted cap steps and sustained requests from a hover (see the module docstring)."""
    out = {}
    for kind in ('cap', 'sustained'):
        targets = np.asarray(spec['cap_targets' if kind == 'cap' else 'sustained_targets'], float)
        B = len(targets)
        end = spec['cap_end_s'] if kind == 'cap' else spec['sustained_end_s']
        T = int(round(end/DT))
        speed = np.zeros((T, B))
        k1 = int(round(1./DT))
        for i, x in enumerate(targets):
            if kind == 'cap':
                kc, kr = int(round(spec['cap_at_s']/DT)), int(round((spec['cap_at_s']+spec['cap_hold_s'])/DT))
                speed[k1:kc, i] = _slew(0., spec['cruise'], kc-k1)
                speed[kc:kr, i] = _drop(speed[kc-1, i], x, spec['brake_slew'], kr-kc)
                speed[kr:, i] = _slew(speed[kr-1, i], spec['cruise'], T-kr)
            else:
                speed[k1:, i] = _slew(0., x, T-k1)
        horizontal = np.zeros((T, B, 2))
        horizontal[..., 0] = speed

        def start(sim, batch):
            s = sim.hover(batch, spec['height'])
            s.quad.pos[:, :2] = 0.
            s.quad.vel[:] = 0.
            return s
        request3 = np.concatenate((horizontal, np.zeros((T, B, 1))), -1)
        vel, z = _fly(ctl, profile, start, horizontal, spec['height'], _feedforward(request3), vz_gain=spec['vz_gain'])
        vh = np.linalg.norm(vel[..., :2], axis=-1)
        rows = []
        for i, x in enumerate(targets):
            if kind == 'cap':
                kc = int(round(spec['cap_at_s']/DT))
                s0, s1 = (int(round((spec['cap_at_s']+a)/DT)) for a in spec['cap_settle_s'])
                reach = np.flatnonzero(vel[kc:kc+int(round(spec['cap_hold_s']/DT)), i, 0] <= x+spec['within'])
                settled = float(vh[s0:s1, i].mean())
                rows.append(dict(target=float(x), pre_cap=round(float(vh[kc-1, i]), 3), settled=round(settled, 3),
                                 settled_excess=round(settled-x, 3),
                                 t_within=round(float(reach[0]*DT), 2) if len(reach) else None,
                                 z_range=[round(float(z[:, i].min()), 2), round(float(z[:, i].max()), 2)]))
            else:
                s0, s1 = (int(round(a/DT)) for a in spec['sustained_settle_s'])
                settled = float(vh[s0:s1, i].mean())
                rows.append(dict(target=float(x), settled=round(settled, 3), settled_excess=round(settled-x, 3),
                                 z_range=[round(float(z[:, i].min()), 2), round(float(z[:, i].max()), 2)]))
        out[kind] = rows
    return out


class Flight:
    """A logged flight: CSV rows and the recorded brain inputs (replay npz)."""

    def __init__(self, root, name):
        import pandas as pd
        self.name = name
        self.d = pd.read_csv(Path(root)/f'{name}.csv', low_memory=False)
        self.z = {k: v for k, v in np.load(Path(root)/f'{name}-replay.npz').items()}
        if len(self.d) != len(self.z['action']):
            raise ValueError(f'{name}: CSV and replay lengths differ')
        self.t = self.d.phase.to_numpy(float)
        self.vel = self.d[['vx', 'vy', 'vz']].to_numpy(float)
        self.req = self.d[['cmd_vx', 'cmd_vy', 'cmd_vz']].to_numpy(float)
        self.cmds = self.d[['command_thr', 'command_roll', 'command_pitch', 'command_yaw']].to_numpy(float)
        dv = np.linalg.norm(np.diff(self.vel, axis=0), axis=1)
        self.jumps = np.flatnonzero(dv > .6)
        self.ff = np.zeros_like(self.req)
        for k in range(1, len(self.d)):
            dt = float(np.clip(self.t[k]-self.t[k-1], 0., .1))
            ok = np.isfinite(self.req[k]).all() and np.isfinite(self.req[k-1]).all()
            raw = (self.req[k]-self.req[k-1])/max(dt, 1e-3) if ok else 0.
            self.ff[k] = self.ff[k-1]+(1-np.exp(-dt/.05))*(raw-self.ff[k-1])

    def index(self, t0):
        return int(np.searchsorted(self.t, t0))

    def impact_after(self, k0):
        later = self.jumps[self.jumps >= k0]
        return int(later[0]) if len(later) else len(self.d)-1

    def state_fn(self, k):
        d = self.d

        def start(sim, batch):
            s = sim.hover(batch, 0.)
            s.quad.pos[:] = torch.tensor(d[['x', 'y', 'z']].iloc[k].to_numpy(float), dtype=torch.float32)
            s.quad.vel[:] = torch.tensor(d[['vx', 'vy', 'vz']].iloc[k].to_numpy(float), dtype=torch.float32)
            s.quad.quat[:] = torch.tensor(d[['qw', 'qx', 'qy', 'qz']].iloc[k].to_numpy(float), dtype=torch.float32)
            s.quad.omega[:] = torch.tensor(d[['omega_x', 'omega_y', 'omega_z']].iloc[k].to_numpy(float),
                                           dtype=torch.float32)
            s.sensed_omega[:] = s.quad.omega
            s.drive[:] = float((d.in_thr.iloc[k]+1)/2)
            s.quad.crashed[:] = False
            return s
        return start

    def warm(self, ctl, k0, seconds):
        """The controller's brain state after the recorded inputs of [t0 - seconds, t0) (retina blanked)."""
        kw = self.index(self.t[k0]-seconds)
        state = ctl.brain.init_state(1)
        with torch.no_grad():
            for k in range(kw, k0):
                obs = {c: torch.from_numpy(self.z[c][k]).float()[None] for c in ctl.channels}
                obs['retina'] = torch.zeros_like(obs['retina'])
                _, state, _ = ctl.brain(obs, state, ctl.W)
        return state

    def pd_warm(self, profile, ctl, k0, ticks=30):
        def warm(pd):
            sim = IdentifiedSim(profile, ctl.calibration, 'cpu', DT)
            sim.randomize(1, 0.)
            for k in range(k0-ticks, k0):
                pd.command(sim.sensors(self.state_fn(k)(sim, 1)), torch.tensor(self.req[k:k+1], dtype=torch.float32),
                           torch.tensor(self.ff[k:k+1], dtype=torch.float32), dt=DT)
        return warm


def _check_recorded_contract(ctl, reference_contract):
    if ctl.kind == 'brain':
        mine = {k: ctl.contract[k] for k in CONTRACT_KEYS}
        if any(abs(mine[k]-reference_contract[k]) > 1e-9 for k in CONTRACT_KEYS):
            raise ValueError(f'The recorded goal inputs were encoded with {reference_contract}, the checkpoint uses {mine}')


def live_tests(ctl, profile, spec, reference_contract):
    """Cap steps and sustained requests from logged live states (see the module docstring)."""
    _check_recorded_contract(ctl, reference_contract)
    targets = np.asarray(spec['targets'], float)
    B = len(targets)
    rows, flights = [], {}
    for window in spec['windows']:
        name, t0 = window.rsplit(':', 1)
        flight = flights.setdefault(name, Flight(spec['flights_dir'], name))
        k0 = flight.index(float(t0))
        pre, hold = int(round(spec['pre_s']/DT)), int(round(spec['hold_s']/DT))
        req0 = flight.req[k0, :2]
        cruise = float(np.linalg.norm(req0))
        u = req0/max(cruise, 1e-6)
        speed = np.zeros((pre+hold, B))
        speed[:pre] = cruise
        for i, x in enumerate(targets):
            # a cap below the logged request falls at brake_slew; a request above it rises with the pilot's slew
            speed[pre:, i] = _drop(cruise, x, spec['brake_slew'], hold) if x < cruise else _slew(cruise, x, hold)
        horizontal = speed[..., None]*u[None, None, :]
        request3 = np.concatenate((horizontal, np.zeros((pre+hold, B, 1))), -1)
        z0 = float(flight.d.z.iloc[k0])
        queue = [torch.tensor(flight.cmds[k0-3+i], dtype=torch.float32)[None].repeat(B, 1) for i in range(3)]
        state = ctl.expand(flight.warm(ctl, k0, spec['warm_s']), B) if ctl.kind == 'brain' else None
        vel, z = _fly(ctl, profile, flight.state_fn(k0), horizontal, z0, _feedforward(request3), brain_state=state,
                      queue=queue, vz_gain=spec['vz_gain'],
                      pd_warm=flight.pd_warm(profile, ctl, k0) if ctl.kind == 'pd' else None)
        vh = np.linalg.norm(vel[..., :2], axis=-1)
        along = vel[..., :2] @ u
        for i, x in enumerate(targets):
            reach = np.flatnonzero(along[pre:, i] <= x+spec['within'])
            s0, s1 = (pre+int(round(a/DT)) for a in spec['settle_s'])
            q0, q1 = (pre+int(round(a/DT)) for a in spec['steady_s'])
            rows.append(dict(window=window, target=float(x), live_request=round(cruise, 3),
                             live_speed=round(float(np.linalg.norm(flight.vel[k0, :2])), 3),
                             pre_cap=round(float(vh[pre-1, i]), 3), t_within=round(float(reach[0]*DT), 2) if len(reach) else None,
                             settled=round(float(vh[s0:s1, i].mean()), 3),
                             settled_excess=round(float(vh[s0:s1, i].mean())-x, 3),
                             steady=round(float(vh[q0:q1, i].mean()), 3),
                             steady_excess=round(float(vh[q0:q1, i].mean())-x, 3),
                             z_range=[round(float(z[:, i].min()), 2), round(float(z[:, i].max()), 2)]))
    return rows


def swap_tests(ctl, profile, spec, reference_contract):
    """Logged request replayed from logged states: this controller vs FastMotorPD vs logged commands."""
    _check_recorded_contract(ctl, reference_contract)
    rows, flights = [], {}
    for window in spec['windows']:
        name, t0 = window.rsplit(':', 1)
        flight = flights.setdefault(name, Flight(spec['flights_dir'], name))
        k0 = flight.index(float(t0))
        H = min(int(round(spec['horizon_s']/DT)), flight.impact_after(k0)-k0-1)
        queue = [torch.tensor(flight.cmds[k0-3+i], dtype=torch.float32)[None] for i in range(3)]
        result = dict(window=window, horizon_s=round(H*DT, 2), request_h_1s=None)
        k1 = min(99, H-1)
        live_vh = np.linalg.norm(flight.vel[k0+1:k0+H+1, :2], axis=-1)
        result['live_vh_1s'] = round(float(live_vh[k1]), 3)
        result['request_h_1s'] = round(float(np.linalg.norm(flight.req[k0+k1, :2])), 3)
        modes = (('controller', ctl.kind), ('pd', 'pd'), ('logged', 'logged'))
        for tag, mode in modes:
            if mode == 'logged':
                vel = _open_loop(ctl, profile, flight, k0, H)
            else:
                state = flight.warm(ctl, k0, spec['warm_s']) if mode == 'brain' else None
                vel = _replay(ctl, profile, flight, k0, H, queue, state, mode)
            result[f'{tag}_vh_1s'] = round(float(np.linalg.norm(vel[k1, 0, :2])), 3)
            result[f'{tag}_vh_end'] = round(float(np.linalg.norm(vel[-1, 0, :2])), 3)
        result['controller_minus_pd_1s'] = round(result['controller_vh_1s']-result['pd_vh_1s'], 3)
        result['logged_minus_live_1s'] = round(result['logged_vh_1s']-result['live_vh_1s'], 3)
        rows.append(result)
    return rows


def _replay(ctl, profile, flight, k0, H, queue, state, mode):
    """Closed loop with the logged world request (all three axes) and the logged yaw stick, clamped for throttle
    priority as the runtime does."""
    sim = IdentifiedSim(profile, ctl.calibration, 'cpu', DT)
    sim.randomize(1, 0.)
    s = flight.state_fn(k0)(sim, 1)
    pd = FastMotorPD(profile, ctl.calibration)
    pd.reset()
    if mode == 'pd':
        flight.pd_warm(profile, ctl, k0)(pd)
    q = deque(queue)
    cal = ctl.calibration
    out = np.zeros((H, 1, 3))
    with torch.no_grad():
        for j in range(H):
            k = k0+j
            senses = sim.sensors(s)
            r = torch.tensor(flight.req[k:k+1], dtype=torch.float32)
            if mode == 'pd':
                a = pd.command(senses, r, torch.tensor(flight.ff[k:k+1], dtype=torch.float32), dt=DT)
            else:
                a, state = ctl.act(senses, s.quad.motor.mean(-1, keepdim=True), r, state)
            command = a.clone()
            thr = cal['hover_processed']+cal['throttle_scale']*(float(command[0, 0])-cal['hover_stick_sim'])
            room = float(np.sqrt(max(.97**2-min(thr*thr, .97**2), 0.)))
            command[0, 3] = float(np.clip(flight.cmds[k, 3], -room, room))
            q.append(command)
            s = sim.step(s, q.popleft())
            v = s.quad.vel
            s.quad.vel = v-.0075*v.norm(dim=-1, keepdim=True)*v*DT
            s.quad.crashed[:] = False
            out[j] = s.quad.vel.numpy()
    return out


def _open_loop(ctl, profile, flight, k0, H):
    sim = IdentifiedSim(profile, ctl.calibration, 'cpu', DT)
    sim.randomize(1, 0.)
    s = flight.state_fn(k0)(sim, 1)
    q = deque(torch.tensor(flight.cmds[k0-3+i], dtype=torch.float32)[None] for i in range(3))
    out = np.zeros((H, 1, 3))
    with torch.no_grad():
        for j in range(H):
            q.append(torch.tensor(flight.cmds[k0+j], dtype=torch.float32)[None])
            s = sim.step(s, q.popleft())
            v = s.quad.vel
            s.quad.vel = v-.0075*v.norm(dim=-1, keepdim=True)*v*DT
            s.quad.crashed[:] = False
            out[j] = s.quad.vel.numpy()
    return out


def _courses(spec):
    return [synthetic_course(s, steep=steep) for _, steep in spec['groups'] for s in spec['seeds']]


def rollout_tests(ctl, profile, spec):
    rows = []
    courses = _courses(spec)
    for S in spec['speeds']:
        row, _ = rollout(ctl.brain, ctl.cfg, ctl.meta, profile, ctl.contract, courses, controller=ctl.kind,
                         speed=float(S), seconds=spec['seconds'], seed=spec['sim_seed'], record_brake=True)
        rows.append(dict(S=float(S), finished=row['finished'], crashed=row['crashed'], mean_speed=row['mean_speed'],
                         stick_chatter=row['stick_chatter'], **row['brake']))
        time.sleep(spec.get('rest_s', 0.))
    return rows


def in_course_caps(ctl, profile, spec):
    caps = SyntheticCapsConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec['caps'].items()})
    row, _ = rollout(ctl.brain, ctl.cfg, ctl.meta, profile, ctl.contract, _courses(spec), controller=ctl.kind,
                     seconds=spec['seconds'], seed=spec['sim_seed'], caps=caps, cap_fraction=spec['cap_fraction'],
                     cap_seed=spec['cap_seed'], record_brake=True)
    return row


def evaluation8(ctl, profile, spec):
    row, _ = rollout(ctl.brain, ctl.cfg, ctl.meta, profile, ctl.contract, _courses(spec), controller=ctl.kind,
                     seconds=spec['seconds'], seed=spec['sim_seed'], record_brake=True)
    return row


def _run_lengths(mask):
    out, c = np.zeros(len(mask), int), 0
    for i, m in enumerate(mask):
        c = c+1 if m else 0
        out[i] = c
    return out


def trace_metrics(trace_path, controller, settle_s=1., steep_vz=-1.5, drop_m=3., high_m=1.5):
    """Steep-request sink tracking (requests <= steep_vz held >= settle_s, t > 3 s, not in the last 1 s of a drone's
    flight) and descending gate passes (legs dropping > drop_m): height above the gate at its vertical plane."""
    z = np.load(trace_path, allow_pickle=True)
    ids, courses = list(z['ids']), z['courses']
    pos, vel, req, tgt, act = (z[f'{controller}_{k}'] for k in ('pos', 'vel', 'req', 'target', 'active'))
    shortfall, high, desc = [], 0, 0
    for i in range(len(ids)):
        a = act[:, i]
        if not a.any():
            continue
        stop = int(np.flatnonzero(a)[-1])+1
        r = np.nan_to_num(req[:stop, i].astype(float)); v = vel[:stop, i].astype(float)
        t = np.arange(stop)*DT
        ok = ~np.isnan(req[:stop, i]).any(1) & (t > 3) & (t < t[-1]-1.)
        steep = ok & (r[:, 2] <= steep_vz)
        settled = steep & (_run_lengths(steep)*DT >= settle_s)
        shortfall.append(v[settled, 2]-r[settled, 2])
        course = np.asarray(courses[i], float)
        target = tgt[:, i].astype(int)
        stop2 = min(len(a), stop+1)
        P = pos[:stop2, i].astype(float)
        for k in np.flatnonzero(np.diff(target[:stop2]) > 0)+1:
            g = int(target[k])-1
            gate, prev = course[g], (course[g-1] if g else np.zeros(3))
            if prev[2]-gate[2] <= drop_m:
                continue
            n = gate[:2]-prev[:2]; n = n/max(np.linalg.norm(n), 1e-9)
            lo, hi = max(1, k-300), min(stop2-1, k+300)
            s = (P[lo-1:hi, :2]-gate[:2]) @ n
            cross = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
            if len(cross):
                j = lo-1+int(cross[0]); f = -s[cross[0]]/(s[cross[0]+1]-s[cross[0]])
                desc += 1
                high += int(P[j, 2]+f*(P[j+1, 2]-P[j, 2])-gate[2] > high_m)
    values = np.concatenate(shortfall) if shortfall else np.zeros(0)
    return dict(steep_settled_s=round(float(len(values))*DT, 2),
                steep_sink_shortfall_mps=round(float(values.mean()), 3) if len(values) else None,
                descending_passes=desc, descending_passes_high=high)


def regression16(ctl, spec, out_dir, tree):
    harness = Path(spec['harness'])
    if sha256(harness) != spec['harness_sha256']:
        raise RuntimeError(f'{harness} is not the pinned harness')
    out_dir.mkdir(parents=True, exist_ok=True)
    result, trace = out_dir/'regression16.json', out_dir/'regression16_trace.npz'
    env = dict(os.environ, HALTERE_TREE=str(tree), OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    cmd = [sys.executable, str(harness), '--out', str(result), '--controllers', ctl.kind, '--checkpoint', ctl.path,
           '--trace', str(trace), '--rest', '0', '--label', spec.get('label', 'brake-gates')]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summary = json.loads(result.read_text())['controllers'][ctl.kind]
    bins = summary['switch_bins']
    lateral = summary.get('lateral_switch_bins', {})
    keep = ('n', 'speed_min', 'regain_s', 'stick_after', 'pitch_swing')
    return dict(finished=summary['finished'], crashed=summary['crashed'], mean_speed=summary['mean_speed'],
                stick_chatter=summary['stick_chatter'], mean_finish_s_own=summary['mean_finish_s_own'],
                switch_bins={b: {k: bins[b].get(k) for k in keep} for b in bins},
                lateral_1s={b: (lateral[b].get('lat_1.0') or {}).get('mean') for b in lateral},
                **trace_metrics(trace, ctl.kind), result=str(result))


def audit(ctl, parent):
    if ctl.kind != 'brain':
        return None
    p = torch.load(parent, map_location='cpu', weights_only=True)['model']
    c = torch.load(ctl.path, map_location='cpu', weights_only=True)['model']
    changed = {k: float((v-p[k]).abs().max()) for k, v in c.items() if not torch.equal(v, p[k])} if p.keys() == c.keys() else None
    clean = (changed is not None and set(changed) <= {'readout.weight', 'readout.bias'}
             and all(torch.equal(c[k][3:], p[k][3:]) for k in ('readout.weight', 'readout.bias')))
    return dict(parent_sha256=sha256(parent), candidate_sha256=sha256(ctl.path), changed_tensors=changed,
                only_readout_rows_0_2=bool(clean))


# ----------------------------------------------------------------------------- verdict

def _get(report, path):
    node = report
    for key in path.split('.'):
        if node is None:
            return None
        node = node[int(key)] if isinstance(node, list) else node.get(key)
    return node


def derived(report):
    """Values that combine parts: history_gap = the live-state steady speed minus the hover-start sustained speed for
    the same request (the brain's speed equilibrium should not depend on where it came from)."""
    out = {}
    hover, live = report.get('hover'), report.get('live')
    if hover and live:
        sustained = {r['target']: r['settled'] for r in hover['sustained']}
        out['history_gap'] = [dict(window=r['window'], target=r['target'], gap=round(r['steady']-sustained[r['target']], 3))
                              for r in live if r['target'] in sustained]
    return out


def verdict(report, gates):
    """Apply every declared gate to a report. A check reads one value (`path`) or the `field` of every row of a list
    (`rows`), optionally only rows whose keys take listed values (`select`) and rows meeting a numeric condition
    (`where`: field, op, value); it passes when every value satisfies `op` `value` (and at least one value exists)."""
    out = {}
    for name, gate in gates['gates'].items():
        checks = []
        for check in gate['checks']:
            if 'rows' in check:
                rows = _get(report, check['rows']) or []
                selected = [r for r in rows if all(r.get(k) in v for k, v in check.get('select', {}).items())]
                where = check.get('where')
                kept = [r for r in selected if where is None or _compare(r.get(where['field']), where['op'], where['value'])]
                values = [r.get(check['field']) for r in kept]
                excluded = len(selected)-len(kept)
            else:
                values, excluded = [_get(report, check['path'])], 0
            ok = bool(values) and all(_compare(v, check['op'], check['value']) for v in values)
            entry = dict(check=check.get('name', check.get('path', check.get('field'))), ok=ok,
                         values=values if len(values) > 1 else values[0] if values else None)
            if excluded:
                entry['excluded_rows'] = excluded
            checks.append(entry)
        out[name] = dict(passed=all(c['ok'] for c in checks), primary=gate.get('primary', True), checks=checks)
    primary = [g for g in out.values() if g['primary']]
    return dict(gates=out, passed_primary=sum(g['passed'] for g in primary), primary_total=len(primary),
                all_primary_passed=all(g['passed'] for g in primary))


def _compare(value, op, limit):
    if value is None:
        return op == 'is_none'
    if op == 'le':
        return value <= limit
    if op == 'ge':
        return value >= limit
    if op == 'abs_le':
        return abs(value) <= limit
    if op == 'between':
        return limit[0] <= value <= limit[1]
    if op == 'eq':
        return value == limit
    if op == 'not_none':
        return True
    raise ValueError(f'Unknown comparison {op}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('controller', help="a fast-contract brain checkpoint, or 'pd'")
    parser.add_argument('--gates', default='configs/brain09_gates.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--parts', default=','.join(PARTS))
    parser.add_argument('--label', default='')
    parser.add_argument('--freeze', action='store_true', help='freeze the gate file (content sha256) and exit')
    args = parser.parse_args()
    if args.freeze:
        obj = freeze(args.gates)
        print(f'{args.gates}: frozen {obj["frozen_at"]} sha256 {obj["sha256"]}')
        return
    torch.set_num_threads(2)
    gates, sha = load_gates(args.gates)
    tests = gates['tests']
    ctl = Controller(args.controller, gates['reference']['brain08'])
    profile = json.loads(Path(gates['reference']['profile']).read_text())
    reference_contract = torch.load(gates['reference']['brain08'], map_location='cpu',
                                    weights_only=True)['visual_brain']['fast_motor_tracking']
    out = Path(args.out)
    tree = Path(__file__).resolve().parents[2]
    report = dict(label=args.label, controller=args.controller, kind=ctl.kind,
                  checkpoint_sha256=sha256(ctl.path) if ctl.kind == 'brain' else None,
                  gates_file=str(args.gates), gates_version=gates.get('version'), gates_sha256=sha,
                  tests_sha256=gates_sha256(dict(tests=tests, reference=gates['reference'])),
                  gates_frozen=bool(gates.get('frozen')), tree=str(tree),
                  source_sha256=sha256(__file__), created=time.strftime('%Y-%m-%d %H:%M:%S'),
                  scope='measured surrogate (IdentifiedSim) development checks; not flight evidence', timing={})
    if out.exists():
        previous = json.loads(out.read_text())
        if (previous.get('tests_sha256') == report['tests_sha256'] and previous.get('source_sha256') == report['source_sha256']
                and previous.get('checkpoint_sha256') == report['checkpoint_sha256']):
            report.update({k: v for k, v in previous.items() if k in PARTS})
    parts = [p for p in args.parts.split(',') if p]
    for part in parts:
        if part not in PARTS:
            raise ValueError(f'Unknown part {part}')
        begin = time.time()
        if part == 'hover':
            report['hover'] = hover_tests(ctl, profile, tests['hover'])
        elif part == 'live':
            report['live'] = live_tests(ctl, profile, tests['live'], reference_contract)
        elif part == 'swaps':
            report['swaps'] = swap_tests(ctl, profile, tests['swaps'], reference_contract)
        elif part == 'rollout_speed':
            report['rollout_speed'] = rollout_tests(ctl, profile, tests['rollout_speed'])
        elif part == 'in_course_caps':
            report['in_course_caps'] = in_course_caps(ctl, profile, tests['in_course_caps'])
        elif part == 'evaluation8':
            report['evaluation8'] = evaluation8(ctl, profile, tests['evaluation8'])
        elif part == 'regression16':
            report['regression16'] = regression16(ctl, tests['regression16'], out.parent/(out.stem+'_harness'), tree)
        elif part == 'audit':
            report['audit'] = audit(ctl, gates['reference']['parent'])
        report['timing'][part] = round(time.time()-begin, 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, default=float))
        print(f'[{part}] {report["timing"][part]} s', flush=True)
    report['derived'] = derived(report)
    if 'gates' in gates:
        report['verdict'] = verdict(report, gates)
        if not gates.get('frozen'):
            report['verdict']['note'] = 'draft gate file: not a frozen verdict'
    out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report.get('verdict', {}), indent=1))


if __name__ == '__main__':
    main()
