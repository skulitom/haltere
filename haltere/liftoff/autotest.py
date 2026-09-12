"""Automated axis check and system-identification flight in Liftoff, flown by the virtual pad.

Sequence (all recorded to CSV, all sticks in [-1, 1]):
  0. rest: throttle low for 2 s; if the drone rises anyway the throttle axis is inverted -> abort
  1. lift probe: ramp the throttle up slowly until the drone climbs; remember the stick at lift-off
  2. altitude hold at +2.5 m above the start with a simple PD loop on the throttle
  3. pulses: +/- roll, +/- pitch, +/- yaw (net-zero pairs) while holding altitude
  4. landing: throttle ramped down

Then the recording is analysed exactly like a manual one (``haltere liftoff fit``): stick and gyro
sign conventions by correlation, optionally the physics fit.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from .frames import unity_quat_to_sim, unity_vec_to_sim, omega_from_quats
from .telemetry import DEFAULT_STREAM, TelemetryFrame, TelemetryReceiver, read_config


class AutoTest:
    def __init__(self, port: int, out: str, max_throttle: float = 0.25, climb: float = 2.5, verbose: bool = True):
        from .gamepad import VirtualPad
        cfg = read_config()
        self.rx = TelemetryReceiver(port=port, stream=(cfg or {}).get('StreamFormat', DEFAULT_STREAM))
        self.pad = VirtualPad()
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.max_throttle = max_throttle
        self.climb = climb
        self.verbose = verbose
        self.rows = []
        self.sticks = np.array([-1.0, 0.0, 0.0, 0.0])   # throttle, roll, pitch, yaw
        self.z0 = None
        self.last_ts = None
        self.aborted = None

    # ------------------------------------------------------------------ helpers
    def send(self, thr=None, roll=None, pitch=None, yaw=None):
        for i, v in enumerate((thr, roll, pitch, yaw)):
            if v is not None:
                self.sticks[i] = float(np.clip(v, -1.0, 1.0))
        self.sticks[0] = min(self.sticks[0], self.max_throttle)
        self.pad.send(self.sticks[0], self.sticks[1], self.sticks[2], self.sticks[3])

    def frame(self, timeout: float = 0.5) -> TelemetryFrame | None:
        fr = self.rx.wait(timeout)
        if fr is None:
            return None
        if self.last_ts is not None and fr.timestamp < self.last_ts - 0.5:
            self.aborted = 'drone was reset in Liftoff'
        self.last_ts = fr.timestamp
        self.rows.append(fr.as_row())
        return fr

    def alt(self, fr: TelemetryFrame) -> float:
        return float(fr.position[1] - self.z0)

    def vz(self, fr: TelemetryFrame) -> float:
        return float(fr.velocity[1])

    def say(self, msg: str):
        if self.verbose:
            print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

    # ------------------------------------------------------------------ phases
    def run(self) -> dict:
        self.say('waiting for telemetry (start a flight in Liftoff with the drone on the ground) ...')
        fr = None
        t0 = time.time()
        while fr is None and time.time() - t0 < 30:
            fr = self.frame(1.0)
        if fr is None:
            return self.finish('no telemetry received in 30 s')
        self.z0 = float(fr.position[1])
        self.say(f'telemetry OK: t={fr.timestamp:.1f}s, ground altitude {self.z0:.2f} m (Unity y)')

        # 0. rest check
        self.send(thr=-1.0)
        t_end = time.time() + 2.0
        while time.time() < t_end:
            fr = self.frame() or fr
        if self.alt(fr) > 0.5:
            return self.finish(f'drone rose {self.alt(fr):.2f} m with the throttle stick at -1: the throttle axis looks inverted')

        # 1. lift probe: ramp throttle slowly
        self.say('lift probe: ramping throttle up slowly')
        thr = -1.0
        lift_stick = None
        while thr < self.max_throttle and lift_stick is None:
            thr += 0.02
            self.send(thr=thr)
            t_end = time.time() + 0.15
            while time.time() < t_end:
                fr = self.frame() or fr
            if self.aborted:
                return self.finish(self.aborted)
            if self.vz(fr) > 0.4 and self.alt(fr) > 0.15:
                lift_stick = thr
        if lift_stick is None:
            return self.finish(f'no lift-off with throttle stick up to {self.max_throttle:+.2f}; raise --max-throttle or check the mapping')
        hover = lift_stick - 0.04
        self.say(f'lift-off at throttle stick {lift_stick:+.2f}; hover estimate {hover:+.2f}')

        # 2. altitude hold
        target = self.climb
        self.say(f'altitude hold at +{target:.1f} m for 8 s')
        t_end = time.time() + 8.0
        integ = 0.0
        while time.time() < t_end:
            fr = self.frame() or fr
            if self.aborted:
                return self.finish(self.aborted)
            err = target - self.alt(fr)
            integ = float(np.clip(integ + 0.02 * err * 0.01, -0.2, 0.2))
            self.send(thr=hover + integ + 0.18 * err - 0.10 * self.vz(fr))
            if self.alt(fr) > target + 5.0:
                return self.finish('altitude runaway (climbed more than 5 m above target); landing')
        hover = hover + integ
        self.say(f'hover stick refined to {hover:+.2f}; altitude {self.alt(fr):.2f} m')

        # 3. pulses
        for axis, name, amp, dur in ((1, 'roll', 0.35, 0.12), (2, 'pitch', 0.35, 0.12), (3, 'yaw', 0.5, 0.3)):
            for sign in (+1.0, -1.0):
                self.say(f'pulse {name} {sign * amp:+.2f} for {dur:.2f}s')
                t_end = time.time() + dur
                while time.time() < t_end:
                    fr = self.frame() or fr
                    err = target - self.alt(fr)
                    kw = {name: sign * amp}
                    self.send(thr=hover + 0.18 * err - 0.10 * self.vz(fr), **kw)
                # settle with the axis neutral
                t_end = time.time() + 1.5
                while time.time() < t_end:
                    fr = self.frame() or fr
                    err = target - self.alt(fr)
                    self.send(thr=hover + 0.18 * err - 0.10 * self.vz(fr), **{name: 0.0})
                if self.aborted:
                    return self.finish(self.aborted)

        # 4. land
        self.say('landing')
        t_end = time.time() + 6.0
        while time.time() < t_end and self.alt(fr) > 0.15:
            fr = self.frame() or fr
            self.send(thr=hover - 0.06 - 0.10 * (self.vz(fr) + 0.6))
        self.send(thr=-1.0)
        return self.finish(None, hover=hover, lift_stick=lift_stick)

    def finish(self, error: str | None, **info) -> dict:
        self.send(thr=-1.0, roll=0.0, pitch=0.0, yaw=0.0)
        time.sleep(0.3)
        self.pad.close()
        self.rx.close()
        with open(self.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(TelemetryFrame.columns())
            w.writerows(self.rows)
        self.say(f'{len(self.rows)} frames written to {self.out}')
        if error:
            self.say('ABORTED: ' + error)
        return {'error': error, 'frames': len(self.rows), 'csv': str(self.out), **info}
