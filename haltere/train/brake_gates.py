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

brain-11 parts (configs/brain11_gates.json; a gate file without their tests does not run them):
- r4_windows: logged round-4 live windows replayed from their logged states with the logged request (all three axes;
  past the log's end the last request is held): this controller, FastMotorPD and the logged commands open loop against
  the live flight (speed excess over the request, height loss).
- swaps_delayed: the swaps windows with a FastMotorPD reference given extra command delay (the brain's output lag).
- capped_turns: governor caps during turns and climbs, from a hover and from logged cruise states.
- accelerate: height kept while accelerating hard from low speed (creep, or a stop) toward several bearings.
- in_course_caps2: in_course_caps with another declared cap configuration (e.g. wider rays and climbing caps).
- full_pilot: the deployed pilot (--obstacle-stack on --descent-view on) in the descent surrogate.

brain-12 parts (configs/brain12_gates.json; run only when declared, so earlier gate files run as before):
- capped_turns_right / accelerate_right: capped_turns / accelerate with other declared cases (the mirror images of the
  brain-11 cases: turns and bearings to the right).
- r4b_windows: round-4b live windows replayed from their logged states: a hairpin approach (speed over the capped
  request and at the logged impact, with and without the motor assist computed from the surrogate's state) and a
  downhill (height kept at the live contact along the live path, horizontal speed kept on the descent).
- hairpins: the motor-assist harness's hairpin scenarios (haltere.liftoff.motor_assist_eval) under the deployed pilot,
  with and without the motor assist.
- full_pilot with `motor_assist: true` flies the deployed pilot with the motor assist (--motor-assist on); regression16
  with declared `seeds`, `groups` and `sim_seed` runs the pinned harness on those courses.

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
# brain-11 parts: run by default only when the gate file declares their tests (a brain-09/10 file runs PARTS only)
NEW_PARTS = ('r4_windows', 'swaps_delayed', 'capped_turns', 'accelerate', 'in_course_caps2', 'full_pilot')
# brain-12 parts: likewise run only when the gate file declares their tests
PARTS12 = ('capped_turns_right', 'accelerate_right', 'r4b_windows', 'hairpins')
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


def _window_inputs(flight, k0, H):
    """Logged world request, yaw stick and request feedforward for H ticks from k0. Past the log's last row the last
    logged request and yaw stick are held and the feedforward low-pass decays (a held request has no derivative)."""
    n = len(flight.d)
    request, yaw, ff = np.zeros((H, 3)), np.zeros(H), np.zeros((H, 3))
    alpha = 1-np.exp(-DT/.05)
    for j in range(H):
        k = k0+j
        if k < n:
            request[j], yaw[j], ff[j] = flight.req[k], flight.cmds[k, 3], flight.ff[k]
        else:
            request[j], yaw[j] = request[j-1], yaw[j-1]
            ff[j] = ff[j-1]*(1-alpha)
    return request, yaw, ff


def replay_window(ctl, profile, flight, k0, H, mode, *, warm_s=8., extra_delay=0):
    """Closed loop of one drone from the logged state at k0 for H ticks, with the logged world request (all three
    axes; held past the log's last row) and the logged yaw stick clamped for throttle priority as the runtime does.
    mode 'controller': this controller (a brain warmed for warm_s on the recorded inputs; the PD as 'pd'); 'pd':
    FastMotorPD with its stick filter warmed on the logged trajectory and `extra_delay` ticks added to the 3-tick command
    delay; 'logged': the logged commands open loop (only within the log). The delay queue is primed with the logged
    commands. Returns position and velocity after each tick (H, 3)."""
    if mode == 'controller':
        mode = 'pd' if ctl.kind == 'pd' else 'brain'
    if mode == 'logged':
        H = min(H, len(flight.d)-k0)
    elif mode != 'pd' and extra_delay:
        raise ValueError('An added delay is declared for the PD reference only')
    request, yaw, ff = _window_inputs(flight, k0, H)
    sim = IdentifiedSim(profile, ctl.calibration, 'cpu', DT)
    sim.randomize(1, 0.)
    s = flight.state_fn(k0)(sim, 1)
    delay = 3+int(extra_delay)
    q = deque(torch.tensor(flight.cmds[k0-delay+i], dtype=torch.float32)[None] for i in range(delay))
    state = flight.warm(ctl, k0, warm_s) if mode == 'brain' else None
    pd = None
    if mode == 'pd':
        pd = FastMotorPD(profile, ctl.calibration)
        pd.reset()
        flight.pd_warm(profile, ctl, k0)(pd)
    cal = ctl.calibration
    pos, vel = np.zeros((H, 3)), np.zeros((H, 3))
    with torch.no_grad():
        for j in range(H):
            if mode == 'logged':
                command = torch.tensor(flight.cmds[k0+j], dtype=torch.float32)[None]
            else:
                senses = sim.sensors(s)
                r = torch.tensor(request[j:j+1], dtype=torch.float32)
                if mode == 'pd':
                    a = pd.command(senses, r, torch.tensor(ff[j:j+1], dtype=torch.float32), dt=DT)
                else:
                    a, state = ctl.act(senses, s.quad.motor.mean(-1, keepdim=True), r, state)
                command = a.clone()
                thr = cal['hover_processed']+cal['throttle_scale']*(float(command[0, 0])-cal['hover_stick_sim'])
                room = float(np.sqrt(max(.97**2-min(thr*thr, .97**2), 0.)))
                command[0, 3] = float(np.clip(yaw[j], -room, room))
            q.append(command)
            s = sim.step(s, q.popleft())
            v = s.quad.vel
            s.quad.vel = v-.0075*v.norm(dim=-1, keepdim=True)*v*DT
            s.quad.crashed[:] = False
            pos[j], vel[j] = s.quad.pos[0].numpy(), s.quad.vel[0].numpy()
    return pos, vel


def _window_metrics(pos, vel, request, z0, span, skip):
    """Speed excess |v_h| - |request_h| (mean over [skip, span) ticks, at the last tick of the span, and its maximum
    there), and heights: loss z0 - min z over all ticks, min z over the span, z at the end."""
    vh = np.linalg.norm(vel[:, :2], axis=-1)
    rh = np.linalg.norm(request[:len(vel), :2], axis=-1)
    excess = vh-rh
    s0, s1 = min(skip, span-1), min(span, len(vel))
    return dict(speed_excess_mean=round(float(excess[s0:s1].mean()), 3),
                speed_excess_end=round(float(excess[s1-1]), 3), speed_excess_max=round(float(excess[s0:s1].max()), 3),
                vh_end=round(float(vh[s1-1]), 3), rh_end=round(float(rh[s1-1]), 3),
                vh_final=round(float(vh[-1]), 3), rh_final=round(float(rh[-1]), 3),
                height_loss=round(float(z0-pos[:, 2].min()), 3), min_z_span=round(float(pos[:s1, 2].min()), 3),
                min_z=round(float(pos[:, 2].min()), 3), z_end=round(float(pos[-1, 2]), 3))


def r4_window_tests(ctl, profile, spec, reference_contract):
    """Logged round-4 live windows replayed from their logged states (see replay_window): this controller, the
    FastMotorPD and the logged commands open loop, against the live flight. The logged span ends at the log's last row
    (the impact); a longer horizon holds the last logged request and yaw stick."""
    _check_recorded_contract(ctl, reference_contract)
    rows, flights = [], {}
    for w in spec['windows']:
        name, t0 = w['window'].rsplit(':', 1)
        flight = flights.setdefault(name, Flight(spec['flights_dir'], name))
        k0 = flight.index(float(t0))
        H = int(round(w['horizon_s']/DT))
        span = min(H, len(flight.d)-1-k0)
        skip = int(round(w.get('skip_s', 0.)/DT))
        request, _, _ = _window_inputs(flight, k0, H)
        z0 = float(flight.d.z.iloc[k0])
        live_pos = flight.d[['x', 'y', 'z']].to_numpy(float)[k0+1:k0+1+span]
        row = dict(name=w['name'], window=w['window'], t0=round(float(flight.t[k0]), 3), z0=round(z0, 3),
                   span_s=round(span*DT, 2), horizon_s=round(H*DT, 2),
                   live=_window_metrics(live_pos, flight.vel[k0+1:k0+1+span], request, z0, span, skip))
        for tag, mode in (('controller', 'controller'), ('pd', 'pd'), ('logged', 'logged')):
            pos, vel = replay_window(ctl, profile, flight, k0, H if mode != 'logged' else span, mode,
                                     warm_s=spec['warm_s'])
            row[tag] = _window_metrics(pos, vel, request, z0, span, skip)
        for key in ('speed_excess_mean', 'min_z_span'):
            row[f'logged_minus_live_{key}'] = round(row['logged'][key]-row['live'][key], 3)
        for tag in ('controller', 'pd', 'live'):
            row.update({f'{tag}_{key}': value for key, value in row[tag].items()})   # flat fields for the verdict
        rows.append(row)
    return rows


def swap_delay_tests(ctl, profile, spec, reference_contract):
    """G3's logged windows (swap_tests) with the PD reference given `extra_delay_ticks` of added command delay (the
    brain's measured output lag behind its teacher), beside the undelayed PD and the logged commands open loop; the
    controller's distance at +1 s from the band between the undelayed and the delayed PD speed (0 inside it)."""
    _check_recorded_contract(ctl, reference_contract)
    rows, flights = [], {}
    for window in spec['windows']:
        name, t0 = window.rsplit(':', 1)
        flight = flights.setdefault(name, Flight(spec['flights_dir'], name))
        k0 = flight.index(float(t0))
        H = min(int(round(spec['horizon_s']/DT)), flight.impact_after(k0)-k0-1)
        k1 = min(99, H-1)
        out = dict(window=window, horizon_s=round(H*DT, 2),
                   live_vh_1s=round(float(np.linalg.norm(flight.vel[k0+1+k1, :2])), 3),
                   request_h_1s=round(float(np.linalg.norm(flight.req[k0+k1, :2])), 3))
        for tag, mode, extra in (('controller', 'controller', 0), ('pd', 'pd', 0),
                                 ('pd_delayed', 'pd', spec['extra_delay_ticks']), ('logged', 'logged', 0)):
            _, vel = replay_window(ctl, profile, flight, k0, H, mode, warm_s=spec['warm_s'], extra_delay=extra)
            out[f'{tag}_vh_1s'] = round(float(np.linalg.norm(vel[k1, :2])), 3)
        out['controller_minus_pd_delayed_1s'] = round(out['controller_vh_1s']-out['pd_delayed_vh_1s'], 3)
        low, high = sorted((out['pd_vh_1s'], out['pd_delayed_vh_1s']))
        # distance from the band between the undelayed and the latency-matched PD (0 inside it)
        out['controller_band_distance_1s'] = round(max(0., low-out['controller_vh_1s'], out['controller_vh_1s']-high), 3)
        out['controller_minus_pd_1s'] = round(out['controller_vh_1s']-out['pd_vh_1s'], 3)
        out['pd_delayed_minus_pd_1s'] = round(out['pd_delayed_vh_1s']-out['pd_vh_1s'], 3)
        out['logged_minus_live_1s'] = round(out['logged_vh_1s']-out['live_vh_1s'], 3)
        rows.append(out)
    return rows


def _slew_vec(start, goal, n, acceleration=10., tc=.25):
    """The pilot's straight-line slew of a horizontal request vector (its 0.25 s taper), n ticks: (n, 2)."""
    out, cur, goal = np.zeros((n, 2)), np.asarray(start, float).copy(), np.asarray(goal, float)
    for k in range(n):
        chord = goal-cur
        norm = float(np.linalg.norm(chord))
        if norm > 0:
            cur = cur+chord*min(1., acceleration*DT/norm, DT/tc)
        out[k] = cur
    return out


def _capped_turn(start_speed, heading, target, turn_deg, n, *, brake_slew=15., rate=1.):
    """A governor cap during a turn, n ticks (n, 2): the request magnitude falls from start_speed to `target` at
    brake_slew (no taper) while its direction turns from `heading` (rad) by turn_deg at `rate` rad/s."""
    magnitude = _drop(start_speed, target, brake_slew, n)
    turn = np.radians(turn_deg)
    angle = heading+np.sign(turn)*np.minimum(abs(turn), rate*DT*np.arange(1, n+1))
    return magnitude[:, None]*np.stack((np.cos(angle), np.sin(angle)), -1)


def _fly3(ctl, profile, sim_state_fn, horizontal, vertical, height_ref, *, brain_state=None, queue=None, vz_gain=.5,
          vertical_slew=(10., 5.)):
    """Closed loop of one batch with a planned horizontal request (T, B, 2) and a planned vertical request (T, B): an
    explicit value, or NaN for the height loop vz = vz_gain (reference - z) clipped to +-1 m/s, whose reference (per
    drone) is the height when an explicit segment ends (the altitude reached is held). The flown vertical request
    moves toward its target at the pilot's vertical command slew (up, down m/s^2); feedforward is the pilot's 0.05 s
    low-pass of the horizontal request derivative; yaw stick 0. Returns velocity (T, B, 3), height (T, B) and the flown
    request (T, B, 3)."""
    T, B = horizontal.shape[:2]
    request3 = np.concatenate((horizontal, np.zeros((T, B, 1))), -1)
    feedforward = _feedforward(request3)
    sim = IdentifiedSim(profile, ctl.calibration, 'cpu', DT)
    sim.randomize(B, 0.)
    s = sim_state_fn(sim, B)
    pd = FastMotorPD(profile, ctl.calibration)
    pd.reset()
    idle = torch.tensor([[-1., 0., 0., 0.]]).repeat(B, 1)
    queue = deque(queue) if queue is not None else deque(idle.clone() for _ in range(3))
    reference = np.broadcast_to(np.asarray(height_ref, float), (B,)).copy()
    flown = np.zeros((T, B, 3))
    flown[..., :2] = horizontal
    velocities, heights = np.zeros((T, B, 3)), np.zeros((T, B))
    vz_request = np.zeros(B)
    state = brain_state
    with torch.no_grad():
        for k in range(T):
            senses = sim.sensors(s)
            z = s.quad.pos[:, 2].numpy().astype(float)
            explicit = np.isfinite(vertical[k])
            reference = np.where(explicit, z, reference)
            target = np.where(explicit, np.nan_to_num(vertical[k]), np.clip(vz_gain*(reference-z), -1., 1.))
            vz_request = vz_request+np.clip(target-vz_request, -vertical_slew[1]*DT, vertical_slew[0]*DT)
            flown[k, :, 2] = vz_request
            r = torch.tensor(flown[k], dtype=torch.float32)
            if ctl.kind == 'pd':
                command = pd.command(senses, r, torch.tensor(feedforward[k], dtype=torch.float32), dt=DT).clone()
            else:
                if state is None:
                    state = ctl.brain.init_state(B)
                    obs = brain_observation(ctl.meta, senses, s.quad.motor.mean(-1, keepdim=True), ctl.cfg.task,
                                            torch.zeros(B, RETINA_DIM), r, ctl.contract)
                    for _ in range(50):
                        _, state, _ = ctl.brain(obs, state, ctl.W)
                command, state = ctl.act(senses, s.quad.motor.mean(-1, keepdim=True), r, state)
                command = command.clone()
            command[:, 3] = 0.
            queue.append(command)
            s = sim.step(s, queue.popleft())
            v = s.quad.vel
            s.quad.vel = v-.0075*v.norm(dim=-1, keepdim=True)*v*DT
            s.quad.crashed[:] = False
            velocities[k] = s.quad.vel.numpy()
            heights[k] = s.quad.pos[:, 2].numpy()
    return velocities, heights, flown


def _turn_cases(spec):
    """(target, turn_deg, climb) cases of the capped-turn tests: the product of the listed values, without the
    straight level cap (G1's own test)."""
    c = spec['cases']
    return [(float(x), float(a), float(v)) for x in c['targets'] for a in c['turns_deg'] for v in c['climbs']
            if not (a == 0 and v == 0)]


def _turn_rows(spec, cases, vel, z, onset, extra=None):
    vh = np.linalg.norm(vel[..., :2], axis=-1)
    e0, e1 = (onset+int(round(a/DT)) for a in spec['excess_window_s'])
    s0, s1 = (onset+int(round(a/DT)) for a in spec['settle_s'])
    rows = []
    for i, (x, turn, climb) in enumerate(cases):
        reach = np.flatnonzero(vh[onset:, i] <= x+spec['within'])
        rows.append(dict(**(extra or {}), target=x, turn_deg=turn, climb=climb, pre_cap=round(float(vh[onset-1, i]), 3),
                         t_within=round(float(reach[0]*DT), 2) if len(reach) else None,
                         excess_mean=round(float(vh[e0:e1, i].mean()-x), 3),
                         settled_excess=round(float(vh[s0:s1, i].mean()-x), 3),
                         z_range=[round(float(z[onset:, i].min()), 2), round(float(z[onset:, i].max()), 2)]))
    return rows


def capped_turn_tests(ctl, profile, spec, reference_contract):
    """Governor caps during turns and climbs (brain-11): the request magnitude falls to X at brake_slew while its
    direction turns by turn_deg at turn_rate rad/s, optionally with a climb request for climb_s (the terrain climb
    beside a cap); from a hover (cruise along +x first) and from the logged cruise states of `live.windows` (brain
    warmed on the recorded inputs; pre_s at the logged request first). Metrics per case: time until |v_h| <= X + within,
    mean |v_h| - X over excess_window_s and over settle_s after the cap onset."""
    cases = _turn_cases(spec)
    B = len(cases)
    out = {}
    h = spec['hover']
    T = int(round(h['end_s']/DT))
    kc, k1 = int(round(h['cap_at_s']/DT)), int(round(1./DT))
    horizontal, vertical = np.zeros((T, B, 2)), np.full((T, B), np.nan)
    climb_n = int(round(spec['climb_s']/DT))
    for i, (x, turn, climb) in enumerate(cases):
        horizontal[k1:kc, i, 0] = _slew(0., h['cruise'], kc-k1)
        horizontal[kc:, i] = _capped_turn(horizontal[kc-1, i, 0], 0., x, turn, T-kc, brake_slew=spec['brake_slew'],
                                          rate=spec['turn_rate'])
        if climb > 0:
            vertical[kc:kc+climb_n, i] = climb

    def start(sim, batch):
        s = sim.hover(batch, h['height'])
        s.quad.pos[:, :2] = 0.
        s.quad.vel[:] = 0.
        return s
    vel, z, _ = _fly3(ctl, profile, start, horizontal, vertical, h['height'], vz_gain=h['vz_gain'],
                      vertical_slew=spec['vertical_slew'])
    out['hover'] = _turn_rows(spec, cases, vel, z, kc)
    live = spec.get('live')
    if live:
        _check_recorded_contract(ctl, reference_contract)
        rows, flights = [], {}
        for window in live['windows']:
            name, t0 = window.rsplit(':', 1)
            flight = flights.setdefault(name, Flight(live['flights_dir'], name))
            k0 = flight.index(float(t0))
            pre, hold = int(round(live['pre_s']/DT)), int(round(live['hold_s']/DT))
            req0 = flight.req[k0, :2]
            cruise = float(np.linalg.norm(req0))
            heading = float(np.arctan2(req0[1], req0[0]))
            horizontal, vertical = np.zeros((pre+hold, B, 2)), np.full((pre+hold, B), np.nan)
            for i, (x, turn, climb) in enumerate(cases):
                horizontal[:pre, i] = req0
                horizontal[pre:, i] = _capped_turn(cruise, heading, x, turn, hold, brake_slew=spec['brake_slew'],
                                                   rate=spec['turn_rate'])
                if climb > 0:
                    vertical[pre:pre+climb_n, i] = climb
            z0 = float(flight.d.z.iloc[k0])
            queue = [torch.tensor(flight.cmds[k0-3+i], dtype=torch.float32)[None].repeat(B, 1) for i in range(3)]
            state = ctl.expand(flight.warm(ctl, k0, live['warm_s']), B) if ctl.kind == 'brain' else None
            vel, z, _ = _fly3(ctl, profile, flight.state_fn(k0), horizontal, vertical, z0, brain_state=state,
                              queue=queue, vz_gain=live['vz_gain'], vertical_slew=spec['vertical_slew'])
            rows += _turn_rows(spec, cases, vel, z, pre, dict(window=window, live_request=round(cruise, 3)))
        out['live'] = rows
    return out


def accelerate_tests(ctl, profile, spec):
    """Height kept while accelerating hard from low speed (brain-11), from a hover at `height` (height loop, yaw 0):
    'creep': the request creeps at `creep` m/s along +x from 1 s, then at go_s slews (the pilot's straight-line slew)
    to `cruise` along bearing b; 'stop': it slews to `cruise` along +x from 1 s, a cap drops it to `creep` at
    brake_slew at brake_s, and at go_s it slews to `cruise` along bearing b. Metrics per case: height loss (the
    reference height minus the lowest height from the brake or go onset to go_s + horizon_s), time from go_s until
    |v_h| >= reach, and |v_h| at go_s + 1.5 s."""
    cases = [(variant, float(b)) for variant in spec['variants'] for b in spec['bearings_deg']]
    B = len(cases)
    go, end = int(round(spec['go_s']/DT)), int(round((spec['go_s']+spec['horizon_s'])/DT))
    kb, k1 = int(round(spec['brake_s']/DT)), int(round(1./DT))
    horizontal = np.zeros((end, B, 2))
    for i, (variant, bearing) in enumerate(cases):
        if variant == 'creep':
            horizontal[k1:go, i, 0] = _slew(0., spec['creep'], go-k1)
        else:
            horizontal[k1:kb, i, 0] = _slew(0., spec['cruise'], kb-k1)
            horizontal[kb:go, i, 0] = _drop(horizontal[kb-1, i, 0], spec['creep'], spec['brake_slew'], go-kb)
        b = np.radians(bearing)
        horizontal[go:, i] = _slew_vec(horizontal[go-1, i], spec['cruise']*np.array([np.cos(b), np.sin(b)]), end-go)

    def start(sim, batch):
        s = sim.hover(batch, spec['height'])
        s.quad.pos[:, :2] = 0.
        s.quad.vel[:] = 0.
        return s
    vel, z, _ = _fly3(ctl, profile, start, horizontal, np.full((end, B), np.nan), spec['height'],
                      vz_gain=spec['vz_gain'], vertical_slew=spec['vertical_slew'])
    vh = np.linalg.norm(vel[..., :2], axis=-1)
    rows = []
    k15 = go+int(round(1.5/DT))
    for i, (variant, bearing) in enumerate(cases):
        first = kb if variant == 'stop' else go
        reach = np.flatnonzero(vh[go:, i] >= spec['reach'])
        rows.append(dict(variant=variant, bearing_deg=bearing,
                         height_loss=round(float(spec['height']-z[first:, i].min()), 3),
                         height_loss_go=round(float(z[go-1, i]-z[go:, i].min()), 3),
                         t_reach=round(float(reach[0]*DT), 2) if len(reach) else None,
                         speed_1p5s=round(float(vh[k15, i]), 3), z_min=round(float(z[first:, i].min()), 3)))
    return rows


def full_pilot_tests(ctl, profile, spec):
    """The deployed pilot (haltere.train.deployed_pilot: --obstacle-stack on --descent-view on for the motor
    contract) in the descent surrogate (haltere.liftoff.descent_rehearsal.run_batch: scoring-only hills, contacts,
    passes high above a checkpoint), on the declared course sets. Per set: its summary; overall: finishes and crashes
    of all courses, contacts on the terrain sets, high passes, stick chatter over the flat and steep courses and the
    mean finish time of the steep and hill sets. With `motor_assist: true` (brain-12) the pilot also carries the motor
    assist of the contract (--motor-assist on; the fast PD has no entry)."""
    from ..liftoff import descent_rehearsal as dr
    from .deployed_pilot import deployed_pilot_kwargs
    assist = dict(motor_assist=True) if spec.get('motor_assist') else {}
    kwargs, record = deployed_pilot_kwargs(spec['contract_brain'] if ctl.kind == 'brain' else spec['contract_pd'],
                                           **assist)
    controller = dict(kind=ctl.kind, meta=ctl.meta, cfg=ctl.cfg, brain=ctl.brain if ctl.kind == 'brain' else None,
                      contract=ctl.contract if ctl.kind == 'brain' else None)
    out = dict(declarations=record, sets={})
    rows_all = {}
    for set_spec in spec['sets']:
        kind, seeds = dr.parse_set(set_spec)
        courses, terrains = dr.course_set(kind, seeds)
        rows, _ = dr.run_batch(controller, profile, courses, terrains, pilot_kwargs=kwargs, seconds=spec['seconds'],
                               seed=spec['sim_seed'])
        for seed, row in zip(seeds, rows):
            row['seed'] = seed
        rows_all[set_spec] = rows
        out['sets'][set_spec] = dict(summary=dr._summary(rows), courses=[
            {k: r.get(k) for k in ('seed', 'finished', 'crashed', 'finish_s', 'gates', 'mean_speed', 'stick_chatter',
                                   'contacts', 'contact_s', 'high_passes', 'support_climbs')} for r in rows])
        time.sleep(spec.get('rest_s', 0.))
    every = [r for rows in rows_all.values() for r in rows]
    terrain = [r for s, rows in rows_all.items() if not s.startswith('flat') for r in rows]
    smooth = [r for s, rows in rows_all.items() if not s.startswith('hill') for r in rows]
    timed = [r for s, rows in rows_all.items() if not s.startswith('flat') for r in rows if r['finished']]
    out['overall'] = dict(courses=len(every), finished=sum(r['finished'] for r in every),
                          crashed=sum(r['crashed'] for r in every),
                          contacts=sum(r.get('contacts') or 0 for r in terrain),
                          contact_s=round(sum(r.get('contact_s') or 0 for r in terrain), 2),
                          high_passes=sum(r.get('high_passes') or 0 for r in every),
                          stick_chatter_16=round(float(np.mean([r['stick_chatter'] for r in smooth])), 5),
                          mean_finish_terrain_s=round(float(np.mean([r['finish_s'] for r in timed])), 2) if timed else None)
    return out


def _controller_dict(ctl):
    return dict(kind=ctl.kind, meta=ctl.meta, cfg=ctl.cfg, brain=ctl.brain if ctl.kind == 'brain' else None,
                contract=ctl.contract if ctl.kind == 'brain' else None)


def hairpin_tests(ctl, profile, spec):
    """brain-12: the motor-assist harness's hairpin scenarios (haltere.liftoff.motor_assist_eval.hairpin_set of the
    declared parameter set, sim seed, seconds) under the deployed pilot for the motor contract, per declared variant:
    'assist' (--motor-assist on: the contract's motor assist) or 'no_assist'. Per variant: the scenario counts of
    haltere.liftoff.motor_assist_gates.scenario_counts (clean = finished without a wall, floor, ceiling contact or a crash)
    and per-scenario rows."""
    from ..liftoff import motor_assist_eval as mae
    from ..liftoff.motor_assist_gates import scenario_counts
    from .deployed_pilot import deployed_pilot_kwargs
    contract = spec['contract_brain'] if ctl.kind == 'brain' else spec['contract_pd']
    scenarios = mae.hairpin_set(spec['set'])
    out = {}
    for variant in spec['variants']:
        if variant not in ('assist', 'no_assist'):
            raise ValueError(f'Unknown hairpin variant {variant}')
        kwargs, record = deployed_pilot_kwargs(contract, motor_assist=variant == 'assist')
        rows, _ = mae.run_scenarios(_controller_dict(ctl), profile, scenarios, pilot_kwargs=kwargs,
                                    seconds=spec['seconds'], seed=spec['sim_seed'])
        keep = ('params', 'finished', 'crashed', 'finish_s', 'wall_contact', 'wall_contact_speed', 'floor_contact',
                'ceiling_contact', 'min_height_m', 'max_height_m', 'min_wall_gap_m', 'stick_chatter')
        out[variant] = dict(declarations=record, assist_applied='motor_assist' in kwargs, counts=scenario_counts(rows),
                            rows=[{k: r.get(k) for k in keep} for r in rows])
        time.sleep(spec.get('rest_s', 0.))
    return out


def _nearest_along(path_xy, points_xy):
    """Index of the nearest point of `path_xy` (N, 2) for each of `points_xy` (M, 2)."""
    d = ((points_xy[:, None, :]-path_xy[None, :, :])**2).sum(-1)
    return d.argmin(1)


def r4b_window_tests(ctl, profile, spec, reference_contract):
    """brain-12: round-4b live windows replayed from the logged state (brain warmed on the recorded inputs; the PD's
    stick filter on the logged trajectory), with the logged world request (all three axes) and the logged yaw stick
    (replay_window), for this controller, FastMotorPD, FastMotorPD with `extra_delay_ticks` of added command delay and
    the logged commands open loop (plant validity), against the live flight. Offline scoring only: the logged positions
    locate the live contact.

    - kind 'hairpin' (t0 .. t_end, the tick 0.05 s before the logged impact): |v_h| at t_end and the mean speed excess
      |v_h| - |request_h| over [cap_t + skip_s, t_end]. With `assist` the same window is also flown with the motor
      assist of the contract applied to the logged pilot's own request with the binding caps of an open-loop replay of
      the logged pilot through the flown stack (haltere.liftoff.motor_assist_eval.live_window); the PD has no entry.
    - kind 'downhill' (t0 .. t_end, the live contact starting at contact_t0): for each replayed tick the nearest live
      position (horizontal) of the window; the height margin over the live contact onset = the lowest replayed height
      minus the live height at the nearest live point, over the ticks whose nearest live point lies within `onset_s` of
      contact_t0 (the live drone touched the ground there, before the ground held it up: > 0 passes above it; None
      when the replay never comes near it); and the mean horizontal speed shortfall
      |request_h| - |v_h| and sink excess request_z - v_z over [t0 + skip_s, contact_t0) (keeping speed on the
      descent)."""
    _check_recorded_contract(ctl, reference_contract)
    rows, flights = [], {}
    modes = (('controller', 'controller', 0), ('pd', 'pd', 0), ('pd_delayed', 'pd', spec.get('extra_delay_ticks', 0)),
             ('logged', 'logged', 0))
    for w in spec['windows']:
        name = w['flight']
        flight = flights.setdefault(name, Flight(spec['flights_dir'], name))
        k0, k_end = flight.index(float(w['t0'])), flight.index(float(w['t_end']))
        H = k_end-k0+1
        request, _, _ = _window_inputs(flight, k0, H)
        live_pos = flight.d[['x', 'y', 'z']].to_numpy(float)[k0+1:k0+1+H]
        live_vel = flight.vel[k0+1:k0+1+H]
        row = dict(name=w['name'], kind=w['kind'], flight=name, t0=round(float(flight.t[k0]), 3),
                   t_end=round(float(flight.t[k_end]), 3), span_s=round(H*DT, 2))
        tracks = {'live': (live_pos, live_vel)}
        for tag, mode, extra in modes:
            tracks[tag] = replay_window(ctl, profile, flight, k0, H, mode, warm_s=spec['warm_s'], extra_delay=extra)
        if w['kind'] == 'hairpin' and w.get('assist') and ctl.kind == 'brain':
            from ..liftoff import motor_assist_eval as mae
            from ..liftoff import visual_brain as vb
            from ..liftoff.fast_race_cue import motor_assist_for_contract
            declaration, digest = vb.load_motor_assist()
            assist = motor_assist_for_contract(declaration, spec['contract_brain'])
            arrays, sources, _ = mae.window_sources(name, str(Path(__file__).resolve().parents[2]), declaration,
                                                    spec['flights_dir'])
            res = mae.live_window(ctl, profile, flight, float(flight.t[k0]), H*DT+1e-6, assist=assist, arrays=arrays,
                                  sources=sources, warm_s=spec['warm_s'])
            row['assist_declaration'] = dict(version=declaration.get('version'), sha256=digest)
            tracks['controller_assist'] = (res['pos'][:H], res['vel'][:H])
        t = flight.t[k0+1:k0+1+H]
        for tag, (pos, vel) in tracks.items():
            n = min(len(vel), H)
            vh = np.linalg.norm(vel[:n, :2], axis=-1)
            rh = np.linalg.norm(request[:n, :2], axis=-1)
            if w['kind'] == 'hairpin':
                c0 = int(np.searchsorted(t[:n], float(w['cap_t'])+float(w.get('skip_s', 0.))))
                row[f'{tag}_speed_end'] = round(float(vh[n-1]), 3)
                row[f'{tag}_capped_excess'] = round(float((vh[c0:n]-rh[c0:n]).mean()), 3) if c0 < n else None
                row[f'{tag}_min_z'] = round(float(pos[:n, 2].min()), 3)
            else:
                # the live contact onset: live points within onset_s of contact_t0 (before the ground held the drone up)
                lo, hi = (float(w['contact_t0'])+float(a) for a in w['onset_s'])
                c0 = int(np.searchsorted(flight.t[k0+1:k0+1+H], lo))
                c1 = int(np.searchsorted(flight.t[k0+1:k0+1+H], hi))
                nearest = _nearest_along(live_pos[:, :2], pos[:n, :2])
                touch = (nearest >= c0) & (nearest <= c1)
                margin = pos[:n, 2]-live_pos[nearest, 2]
                c0 = int(np.searchsorted(flight.t[k0+1:k0+1+H], float(w['contact_t0'])))
                s0 = int(np.searchsorted(t[:n], float(flight.t[k0])+float(w.get('skip_s', 0.))))
                s1 = min(c0, n)
                row[f'{tag}_contact_margin'] = round(float(margin[touch].min()), 3) if touch.any() else None
                row[f'{tag}_reached_contact'] = bool(touch.any())
                row[f'{tag}_speed_shortfall'] = round(float((rh[s0:s1]-vh[s0:s1]).mean()), 3) if s0 < s1 else None
                row[f'{tag}_sink_excess'] = (round(float((request[s0:s1, 2]-vel[s0:s1, 2]).mean()), 3)
                                             if s0 < s1 else None)
        for key in ('speed_end', 'contact_margin'):
            if row.get(f'logged_{key}') is not None and row.get(f'live_{key}') is not None:
                row[f'logged_minus_live_{key}'] = round(row[f'logged_{key}']-row[f'live_{key}'], 3)
        rows.append(row)
    return rows


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
    # brain-12: declared course seeds, groups and sim seed (absent: the harness defaults, as every earlier gate file)
    if 'seeds' in spec:
        cmd += ['--seeds', *(str(int(s)) for s in spec['seeds'])]
    if 'groups' in spec:
        cmd += ['--groups', ','.join(f'{name}:{steep}' for name, steep in spec['groups'])]
    if 'sim_seed' in spec:
        cmd += ['--sim-seed', str(int(spec['sim_seed']))]
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
    parser.add_argument('--parts', default='', help='comma-separated parts (default: every part the file declares)')
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
            report.update({k: v for k, v in previous.items() if k in PARTS+NEW_PARTS+PARTS12})
    parts = [p for p in args.parts.split(',') if p] or [*PARTS, *(p for p in NEW_PARTS+PARTS12 if p in tests)]
    for part in parts:
        if part not in PARTS+NEW_PARTS+PARTS12:
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
        elif part == 'r4_windows':
            report['r4_windows'] = r4_window_tests(ctl, profile, tests['r4_windows'], reference_contract)
        elif part == 'swaps_delayed':
            report['swaps_delayed'] = swap_delay_tests(ctl, profile, tests['swaps_delayed'], reference_contract)
        elif part == 'capped_turns':
            report['capped_turns'] = capped_turn_tests(ctl, profile, tests['capped_turns'], reference_contract)
        elif part == 'accelerate':
            report['accelerate'] = accelerate_tests(ctl, profile, tests['accelerate'])
        elif part == 'in_course_caps2':
            report['in_course_caps2'] = in_course_caps(ctl, profile, tests['in_course_caps2'])
        elif part == 'full_pilot':
            report['full_pilot'] = full_pilot_tests(ctl, profile, tests['full_pilot'])
        elif part == 'capped_turns_right':
            report['capped_turns_right'] = capped_turn_tests(ctl, profile, tests['capped_turns_right'],
                                                             reference_contract)
        elif part == 'accelerate_right':
            report['accelerate_right'] = accelerate_tests(ctl, profile, tests['accelerate_right'])
        elif part == 'r4b_windows':
            report['r4b_windows'] = r4b_window_tests(ctl, profile, tests['r4b_windows'], reference_contract)
        elif part == 'hairpins':
            report['hairpins'] = hairpin_tests(ctl, profile, tests['hairpins'])
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
