"""Liftoff drone-telemetry: configuration file and UDP frame receiver.

Liftoff (LuGus Studios) streams a header-less binary frame per UDP datagram when
``TelemetryConfiguration.json`` exists in its LocalLow folder. Fields are little-endian float32,
concatenated in the order listed in ``StreamFormat``; ``MotorRPM`` is a uint8 motor count followed
by that many floats. Frames arrive at about 100 Hz while a drone is being simulated.

Field semantics (official guide, Steam community id 3160488434):
Position (m, Unity world frame: x right, y up, z forward), Attitude (quaternion x,y,z,w),
Velocity (m/s, world), Gyro (deg/s: pitch, roll, yaw), Input (throttle, yaw, pitch, roll),
Battery (voltage, percentage), MotorRPM (left front, right front, left back, right back).
"""
from __future__ import annotations

import json
import os
import select
import shutil
import socket
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LIFTOFF_LOCALLOW = Path(os.path.expanduser('~')) / 'AppData' / 'LocalLow' / 'LuGus Studios' / 'Liftoff'
CONFIG_NAME = 'TelemetryConfiguration.json'

FIELD_SIZES = {
    'Timestamp': 1,
    'Position': 3, 'PositionX': 1, 'PositionY': 1, 'PositionZ': 1,
    'Attitude': 4, 'AttitudeX': 1, 'AttitudeY': 1, 'AttitudeZ': 1, 'AttitudeW': 1,
    'Velocity': 3, 'SpeedX': 1, 'SpeedY': 1, 'SpeedZ': 1,
    'Gyro': 3, 'GyroPitch': 1, 'GyroRoll': 1, 'GyroYaw': 1,
    'Input': 4, 'InputThrottle': 1, 'InputYaw': 1, 'InputPitch': 1, 'InputRoll': 1,
    'Battery': 2, 'BatteryVoltage': 1, 'BatteryPercentage': 1,
    'MotorRPM': None,  # uint8 count + count floats
}
DEFAULT_STREAM = ['Timestamp', 'Position', 'Attitude', 'Velocity', 'Gyro', 'Input', 'Battery', 'MotorRPM']
DEFAULT_PORT = 9001


@dataclass
class TelemetryFrame:
    timestamp: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))   # Unity world, m
    attitude: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))  # x,y,z,w
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))   # Unity world, m/s
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))       # deg/s: pitch, roll, yaw
    input: np.ndarray = field(default_factory=lambda: np.zeros(4))      # throttle, yaw, pitch, roll
    battery: np.ndarray = field(default_factory=lambda: np.zeros(2))    # voltage, percentage
    motor_rpm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    recv_time: float = 0.0

    def as_row(self) -> list[float]:
        rpm = list(self.motor_rpm[:4]) + [0.0] * (4 - min(4, len(self.motor_rpm)))
        return ([self.recv_time, self.timestamp] + list(self.position) + list(self.attitude) + list(self.velocity)
                + list(self.gyro) + list(self.input) + list(self.battery) + rpm)

    @staticmethod
    def columns() -> list[str]:
        return (['recv_time', 'timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz',
                 'gyro_pitch', 'gyro_roll', 'gyro_yaw', 'in_throttle', 'in_yaw', 'in_pitch', 'in_roll',
                 'batt_v', 'batt_pct', 'rpm_lf', 'rpm_rf', 'rpm_lb', 'rpm_rb'])


class FrameParser:
    """Decode datagrams laid out according to a ``StreamFormat`` list."""

    _target = {  # field name -> (frame attribute, slice start)
        'Position': ('position', 0), 'PositionX': ('position', 0), 'PositionY': ('position', 1), 'PositionZ': ('position', 2),
        'Attitude': ('attitude', 0), 'AttitudeX': ('attitude', 0), 'AttitudeY': ('attitude', 1),
        'AttitudeZ': ('attitude', 2), 'AttitudeW': ('attitude', 3),
        'Velocity': ('velocity', 0), 'SpeedX': ('velocity', 0), 'SpeedY': ('velocity', 1), 'SpeedZ': ('velocity', 2),
        'Gyro': ('gyro', 0), 'GyroPitch': ('gyro', 0), 'GyroRoll': ('gyro', 1), 'GyroYaw': ('gyro', 2),
        'Input': ('input', 0), 'InputThrottle': ('input', 0), 'InputYaw': ('input', 1), 'InputPitch': ('input', 2),
        'InputRoll': ('input', 3),
        'Battery': ('battery', 0), 'BatteryVoltage': ('battery', 0), 'BatteryPercentage': ('battery', 1),
    }

    def __init__(self, stream: list[str] | None = None):
        self.stream = list(stream or DEFAULT_STREAM)
        for f in self.stream:
            if f not in FIELD_SIZES:
                raise ValueError(f'unknown telemetry field {f!r}')
        self.fixed_size = sum(FIELD_SIZES[f] or 0 for f in self.stream) * 4
        self.has_rpm = 'MotorRPM' in self.stream

    def expected_size(self, n_motors: int = 4) -> int:
        return self.fixed_size + ((1 + 4 * n_motors) if self.has_rpm else 0)

    def parse(self, data: bytes, recv_time: float | None = None) -> TelemetryFrame:
        fr = TelemetryFrame(recv_time=time.time() if recv_time is None else recv_time)
        off = 0
        for f in self.stream:
            if f == 'MotorRPM':
                n = data[off]
                off += 1
                fr.motor_rpm = np.array(struct.unpack_from(f'<{n}f', data, off), dtype=np.float64)
                off += 4 * n
                continue
            k = FIELD_SIZES[f]
            vals = struct.unpack_from(f'<{k}f', data, off)
            off += 4 * k
            if f == 'Timestamp':
                fr.timestamp = vals[0]
            else:
                attr, start = self._target[f]
                arr = getattr(fr, attr)
                arr[start:start + k] = vals
        return fr


class TelemetryReceiver:
    """Non-blocking UDP receiver that always exposes the newest frame."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = '127.0.0.1', stream: list[str] | None = None,
                 forward_port: int | None = None):
        if forward_port is not None and (not 1 <= forward_port <= 65535 or forward_port == port):
            raise ValueError('Forward telemetry to a different valid local UDP port')
        self.forward_port = forward_port
        self.parser = FrameParser(stream)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.frames = 0
        self.bad = 0
        self.last: TelemetryFrame | None = None

    def poll(self) -> TelemetryFrame | None:
        """Drain the socket; return the newest frame received (or None)."""
        newest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                newest = self.parser.parse(data)
            except (struct.error, IndexError):
                self.bad += 1
                continue
            self.frames += 1
            if self.forward_port is not None:
                self.sock.sendto(data, ('127.0.0.1', self.forward_port))
        if newest is not None:
            self.last = newest
        return newest

    def wait(self, timeout: float = 0.1) -> TelemetryFrame | None:
        """Block up to ``timeout`` seconds for a frame; return the newest one."""
        fr = self.poll()
        if fr is not None:
            return fr
        r, _, _ = select.select([self.sock], [], [], timeout)
        if r:
            return self.poll()
        return None

    def close(self) -> None:
        self.sock.close()


# ----------------------------------------------------------------------------- configuration file

def config_path(root: Path | None = None) -> Path:
    return (root or LIFTOFF_LOCALLOW) / CONFIG_NAME


def read_config(root: Path | None = None) -> dict | None:
    p = config_path(root)
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def write_config(port: int = DEFAULT_PORT, host: str = '127.0.0.1', stream: list[str] | None = None,
                 root: Path | None = None, backup: bool = True) -> Path:
    """Create/replace Liftoff's TelemetryConfiguration.json (Liftoff re-reads it whenever the drone is reset)."""
    p = config_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    if backup and p.exists():
        shutil.copy2(p, p.with_suffix('.json.bak'))
    cfg = {'EndPoint': f'{host}:{port}', 'StreamFormat': list(stream or DEFAULT_STREAM)}
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    return p
