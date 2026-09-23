"""Read-only progress/stall monitoring of a growing flight CSV, without screenshots.

Telemetry cannot prove a race finish. Result-screen adjudication remains separate.
The monitor never touches the gamepad, pauses the game, or changes the controller.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import math
from pathlib import Path
import time


class ProgressMonitor:
    def __init__(self, window_s=60., radius_m=1., max_gap_s=.5):
        if any(not math.isfinite(v) or v <= 0 for v in (window_s, radius_m, max_gap_s)):
            raise ValueError('Monitoring limits must be finite and positive')
        self.window_s, self.radius_m, self.max_gap_s = window_s, radius_m, max_gap_s
        self.samples = deque()
        self.rows = self.invalid_rows = self.resets = self.gaps = 0
        self.start = self.last = None

    def update(self, row):
        try:
            values = [float(row[k]) for k in ('wall', 'ts', 'x', 'y', 'z', 'vx', 'vy', 'vz')]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Nonfinite telemetry')
        except (KeyError, TypeError, ValueError):
            self.invalid_rows += 1
            return
        wall, ts, *motion = values
        if self.last is not None:
            if ts < self.last['ts'] or wall < self.last['wall']:
                self.samples.clear()
                self.resets += 1
                self.start = wall
            elif max(ts-self.last['ts'], wall-self.last['wall']) > self.max_gap_s:
                self.samples.clear()
                self.gaps += 1
        self.start = wall if self.start is None else self.start
        self.last = dict(wall=wall, ts=ts, position=motion[:3], velocity=motion[3:],
                         controller_status=row.get('geometry_control_status'))
        self.samples.append((ts, motion[:3]))
        # Keep one sample just before the boundary, so a complete window is required.
        while len(self.samples) > 1 and self.samples[1][0] <= ts-self.window_s:
            self.samples.popleft()
        self.rows += 1

    def snapshot(self, now=None):
        result = dict(rows=self.rows, invalid_rows=self.invalid_rows, resets=self.resets,
                      telemetry_gaps=self.gaps, finish_confirmed=None,
                      finish_evidence='Requires separate game result evidence',
                      stall_window_s=self.window_s, stall_radius_m=self.radius_m)
        if self.last is None:
            return dict(result, state='waiting_for_telemetry', stall_detected=False)
        duration = self.samples[-1][0]-self.samples[0][0]
        displacement = max(math.dist(self.samples[0][1], p) for _, p in self.samples)
        age = max(0., (time.time() if now is None else now)-self.last['wall'])
        stalled = duration >= self.window_s-1e-6 and displacement < self.radius_m
        return dict(result, **self.last, elapsed_s=self.last['wall']-self.start,
                    speed_mps=math.sqrt(sum(v*v for v in self.last['velocity'])),
                    log_age_s=age, recent_duration_s=duration,
                    recent_max_displacement_m=displacement, stall_detected=stalled,
                    state='telemetry_stale' if age > 2 else 'stalled' if stalled else 'active')


class CsvTail:
    """Consume only new complete lines; a partial write is held until the next read."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b''
        self.header = None

    def read(self):
        if not self.path.exists():
            return []
        if self.path.stat().st_size < self.offset:
            raise RuntimeError('Flight log was truncated; start a new monitor for a new attempt')
        with self.path.open('rb') as source:
            source.seek(self.offset)
            data = self.pending+source.read()
            self.offset = source.tell()
        lines = data.split(b'\n')
        self.pending = lines.pop()
        result = []
        for raw in lines:
            values = next(csv.reader([raw.decode('utf-8-sig').rstrip('\r')]))
            if self.header is None:
                self.header = values
            else:
                result.append(dict(zip(self.header, values)) if len(values) == len(self.header) else {})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--out', type=Path, help='New evidence directory; live status and change events')
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--stall-seconds', type=float, default=60.)
    parser.add_argument('--stall-radius', type=float, default=1.)
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.interval)):
        parser.error('Use finite positive duration and interval')
    monitor = ProgressMonitor(args.stall_seconds, args.stall_radius)
    tail = CsvTail(args.log)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
    begin, previous = time.monotonic(), None
    while True:
        for row in tail.read():
            monitor.update(row)
        result = monitor.snapshot()
        metadata_path = args.log.with_suffix('.json')
        metadata = None
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError:
                pass  # Controller may still be writing its terminal report.
        if metadata is not None:
            result['state'] = 'controller_stopped'
            result['stop_reason'] = metadata.get('stop_reason')
            result['failures'] = {k: metadata.get(k) for k in
                                  ('impact', 'camera_failure', 'controller_deadline_failure')}
        encoded = json.dumps(result, allow_nan=False)
        if args.out:
            temporary = args.out/'status.tmp'
            temporary.write_text(encoded+'\n')
            temporary.replace(args.out/'status.json')
        event = (result['state'], result['stall_detected'])
        if event != previous or not args.watch:
            print(encoded, flush=True)
            if args.out:
                with (args.out/'events.jsonl').open('a') as output:
                    output.write(encoded+'\n')
            previous = event
        if not args.watch or metadata is not None or time.monotonic()-begin >= args.seconds:
            break
        time.sleep(min(args.interval, max(0., args.seconds-(time.monotonic()-begin))))


if __name__ == '__main__':
    main()
