"""A stand-in for Liftoff, for rehearsing the whole telemetry -> brain -> sticks loop without the game.

It simulates the drone with the same physics as training (but different, "unknown" parameters), streams
Liftoff-format binary telemetry over UDP at 100 Hz, and accepts stick commands over a second UDP port
(4 little-endian float32 in Liftoff's Input order: throttle, yaw, pitch, roll). Like the real game it
uses its own sign conventions, so ``haltere liftoff fit`` has something to discover:
yaw and pitch are flipped relative to the simulator and the gyro is reported as (pitch, -roll, yaw).
When nobody sends sticks it flies itself with a crude angle-mode autopilot plus excitation, which
makes a usable recording for ``haltere liftoff fit``.
"""
from __future__ import annotations

import socket
import struct
import time

import numpy as np
import torch

from ..sim.controller import RateControllerParams
from ..sim.quad import QuadParams, QuadState
from ..sim.vehicle import RatesConfig, Vehicle
from .frames import sim_quat_to_unity, sim_vec_to_unity
from .telemetry import DEFAULT_PORT, DEFAULT_STREAM, FIELD_SIZES

STICK_PORT = 9002
MAX_RPM = 30000.0


class FakeLiftoff:
    def __init__(self, telemetry_port: int = DEFAULT_PORT, stick_port: int = STICK_PORT, rate: float = 100.0,
                 quad: QuadParams | None = None, autopilot: bool = True, seed: int = 0, verbose: bool = True):
        self.rate = rate
        self.dt = 1.0 / rate
        self.quad = quad or QuadParams(twr=5.5, thrust_exp=1.3, motor_tau=0.04, drag_lin=(0.08, 0.08, 0.12))
        self.veh = Vehicle(self.quad, RateControllerParams(), RatesConfig(), 'cpu', dt=self.dt, substeps=2)
        self.autopilot = autopilot
        self.verbose = verbose
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tx_addr = ('127.0.0.1', telemetry_port)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(('127.0.0.1', stick_port))
        self.rx.setblocking(False)
        self.rng = np.random.default_rng(seed)
        self.freqs = self.rng.uniform(0.2, 0.8, size=(3, 2))
        self.phases = self.rng.uniform(0, 2 * np.pi, size=(3, 2))
        self.reset()

    def reset(self) -> None:
        self.vs = self.veh.wrap(QuadState.hover(1, 'cpu', height=0.5))
        self.t = 0.0
        self.last_ext = -1e9
        self.resets = getattr(self, 'resets', -1) + 1

    # ------------------------------------------------------------------ conventions
    @staticmethod
    def liftoff_to_sim_sticks(inp: np.ndarray) -> np.ndarray:
        """Liftoff Input (throttle, yaw, pitch, roll) -> simulator sticks (throttle, roll, pitch, yaw)."""
        return np.array([inp[0], inp[3], -inp[2], -inp[1]])

    @staticmethod
    def sim_to_liftoff_input(sticks: np.ndarray) -> np.ndarray:
        return np.array([sticks[0], -sticks[3], -sticks[2], sticks[1]])

    def autopilot_sticks(self) -> np.ndarray:
        st = self.vs.quad
        sens = self.veh.sim.sensors(st, noise=False)
        gb = sens['gravity_body'][0].numpy()
        exc = np.array([np.sum(0.12 * np.sin(2 * np.pi * self.freqs[i] * self.t + self.phases[i])) for i in range(3)])
        hover = 2 * self.veh.sim.hover_command() - 1
        z, vz = float(st.pos[0, 2]), float(st.vel[0, 2])
        thr = hover + 0.25 * (3.0 - z) - 0.15 * vz
        return np.clip(np.array([thr, 1.5 * gb[1] + exc[0], -1.5 * gb[0] + exc[1], exc[2]]), -1, 1)

    def frame_bytes(self, sim_sticks: np.ndarray) -> bytes:
        st = self.vs.quad
        pos_u = sim_vec_to_unity(st.pos[0].numpy())
        vel_u = sim_vec_to_unity(st.vel[0].numpy())
        q_u = sim_quat_to_unity(st.quat[0].numpy())
        om = st.omega[0].numpy()
        gyro = np.rad2deg([om[1], -om[0], om[2]])
        inp = self.sim_to_liftoff_input(sim_sticks)
        rpm = st.motor[0].numpy() * MAX_RPM
        parts = {'Timestamp': [self.t], 'Position': pos_u, 'Attitude': q_u, 'Velocity': vel_u, 'Gyro': gyro,
                 'Input': inp, 'Battery': [15.8, 0.85]}
        out = b''
        for f in DEFAULT_STREAM:
            if f == 'MotorRPM':
                out += struct.pack('<B4f', 4, *rpm)
            else:
                out += struct.pack(f'<{FIELD_SIZES[f]}f', *[float(x) for x in parts[f]])
        return out

    def poll_sticks(self) -> np.ndarray | None:
        latest = None
        while True:
            try:
                data, _ = self.rx.recvfrom(64)
            except (BlockingIOError, OSError):
                break
            if len(data) >= 16:
                latest = np.array(struct.unpack_from('<4f', data, 0))
        return latest

    # ------------------------------------------------------------------ main loop
    def run(self, seconds: float = 60.0) -> None:
        t_wall0 = time.perf_counter()
        next_t = t_wall0
        n = 0
        ext = None
        sim_sticks = np.array([-1.0, 0.0, 0.0, 0.0])
        if self.verbose:
            print(f'fake Liftoff: telemetry -> udp://{self.tx_addr[0]}:{self.tx_addr[1]}, sticks <- udp port '
                  f'{self.rx.getsockname()[1]}, autopilot {"on" if self.autopilot else "off"}', flush=True)
        while time.perf_counter() - t_wall0 < seconds:
            got = self.poll_sticks()
            if got is not None:
                ext = got
                self.last_ext = self.t
            if ext is not None and self.t - self.last_ext < 0.5:
                sim_sticks = self.liftoff_to_sim_sticks(np.clip(ext, -1, 1))
            elif self.autopilot:
                sim_sticks = self.autopilot_sticks()
            else:
                sim_sticks = np.array([-1.0, 0.0, 0.0, 0.0])
            self.tx.sendto(self.frame_bytes(sim_sticks), self.tx_addr)
            self.vs = self.veh.step(self.vs, torch.tensor(sim_sticks, dtype=torch.float32)[None])
            self.t += self.dt
            n += 1
            if bool(self.vs.quad.crashed.any()):
                if self.verbose:
                    print(f'  crash at t={self.t:.2f}s -> reset', flush=True)
                self.reset()
            if self.verbose and n % int(self.rate) == 0:
                p = self.vs.quad.pos[0].numpy()
                src = 'external' if (ext is not None and self.t - self.last_ext < 0.5) else 'autopilot'
                print(f'  t={self.t:6.2f}s pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) sticks={np.round(sim_sticks, 2)} '
                      f'[{src}] resets={self.resets}', flush=True)
            next_t += self.dt
            while True:
                rem = next_t - time.perf_counter()
                if rem <= 0:
                    break
                if rem > 0.002:
                    time.sleep(rem - 0.0015)
        self.tx.close()
        self.rx.close()
