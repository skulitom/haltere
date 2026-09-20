"""Passive, bounded collection of FPV frames and the player's telemetry.

This module never constructs a brain or a gamepad. A recording has one coordinate
origin and ends on a game reset. Replay/spectator video is not paired with player
telemetry: Liftoff does not export replay-drone telemetry.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .frames import unity_quat_to_sim, unity_vec_to_sim
from .telemetry import DEFAULT_STREAM, TelemetryFrame, TelemetryReceiver, read_config


class DatasetWriter:
    def __init__(self, out, *, course, camera, max_age=0.05, teacher_route=None):
        if not course.strip() or not np.isfinite(max_age) or max_age <= 0:
            raise ValueError('Provide a course name and a positive telemetry age limit')
        camera = Path(camera)
        camera_bytes = camera.read_bytes()
        teacher = None
        if teacher_route:
            from .collection_route import CollectionRoute
            route = CollectionRoute(teacher_route)
            teacher = {'path': str(route.path.resolve()),
                       'sha256': hashlib.sha256(route.path.read_bytes()).hexdigest(),
                       'race_id': route.data['race_id']}
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / 'frames').mkdir()
        (self.out / 'camera.yaml').write_bytes(camera_bytes)
        self.max_age, self.origin, self.start_ts = max_age, None, None
        self.frames, self.rejected, self.ages = 0, 0, []
        self.last_ts = None
        self.meta = {
            'schema': 1, 'course': course, 'source': 'route_teacher_live' if teacher else 'player_live',
            'oracle_route': teacher is not None, 'teacher_route': teacher,
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'camera_sha256': hashlib.sha256(camera_bytes).hexdigest(),
            'frame': 'sim xyz relative to first accepted image pose; quaternion wxyz',
            'input_order': ['throttle', 'yaw', 'pitch', 'roll'],
            'input_meaning': 'Liftoff telemetry Input, not raw radio/gamepad values',
            'timing': 'screen grab start/end and latest prior UDP receive; physical display latency unmeasured',
            'max_telemetry_age_s': max_age, 'labels_reviewed': False,
            'course_complete': False,
        }
        self.index = (self.out / 'index.csv').open('w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.index)
        self.writer.writerow(['file', 'wall_time', 'capture_end', 'telemetry_age_s', 't',
                              'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'ts',
                              'in_throttle', 'in_yaw', 'in_pitch', 'in_roll'])
        self._manifest('recording')

    def _manifest(self, reason):
        self.meta.update(frames=self.frames, rejected_frames=self.rejected, stop_reason=reason,
                         origin_sim=None if self.origin is None else self.origin.tolist(),
                         telemetry_age_p95_s=float(np.quantile(self.ages, .95)) if self.ages else None)
        (self.out / 'capture.json').write_text(json.dumps(self.meta, indent=2), encoding='utf-8')

    def add(self, image, frame: TelemetryFrame, grab_start, grab_end):
        """Save a fresh, advancing observation. Return False for stale/invalid pairs."""
        import cv2

        age = grab_end - frame.recv_time
        values = [frame.timestamp, frame.recv_time, grab_start, grab_end, *frame.position,
                  *frame.attitude, *frame.input]
        if (not np.isfinite(values).all() or grab_end < grab_start or frame.recv_time > grab_start
                or not 0 <= age <= self.max_age or abs(np.linalg.norm(frame.attitude) - 1) > .01
                or (self.last_ts is not None and frame.timestamp <= self.last_ts)):
            self.rejected += 1
            return False
        pos = unity_vec_to_sim(frame.position)
        if self.origin is None:
            self.origin, self.start_ts = pos.copy(), frame.timestamp
        q = unity_quat_to_sim(frame.attitude)
        filename = f'{self.frames:06d}.jpg'
        small = cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)
        if not cv2.imwrite(str(self.out / 'frames' / filename), small, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError('Could not write captured image')
        self.writer.writerow([filename, grab_start, grab_end, age, frame.timestamp - self.start_ts,
                              *(pos - self.origin), *q, frame.timestamp, *frame.input])
        self.last_ts = frame.timestamp
        self.frames += 1
        self.ages.append(age)
        self.index.flush()
        return True

    def close(self, reason):
        self.index.close()
        self._manifest(reason)


def capture_dataset(out, *, course, camera, seconds=180., fps=10., port=9001,
                    window='Liftoff', max_age=.05, teacher_route=None):
    """Capture only the foreground game window, with no input or focus changes."""
    if not np.isfinite([seconds, fps]).all() or seconds <= 0 or not 0 < fps <= 60:
        raise ValueError('Use a positive duration and a capture rate in (0, 60]')
    if not window.strip():
        raise ValueError('A specific game window title is required')
    import mss
    from .commands import find_game_window, game_window_active
    from .recorder import find_window_rect

    stream = (read_config() or {}).get('StreamFormat', DEFAULT_STREAM)
    required = {'Timestamp', 'Position', 'Attitude', 'Input'}
    if not required.issubset(stream):
        raise ValueError(f'Capture requires full telemetry fields: {sorted(required)}')
    writer = DatasetWriter(out, course=course, camera=camera, max_age=max_age, teacher_route=teacher_route)
    rx = None
    reason = 'error'
    try:
        rx = TelemetryReceiver(port=port, stream=stream)
        with (writer.out / 'telemetry.csv').open('w', newline='', encoding='utf-8') as raw, mss.mss() as screen:
            log = csv.writer(raw)
            log.writerow(TelemetryFrame.columns())
            deadline, next_image, last_ts = time.monotonic() + seconds, 0., None
            print(f'Passive capture to {out}. Fly the player drone; keep {window} in front. '
                  'A reset ends this recording. Start a new directory for each flight.', flush=True)
            while time.monotonic() < deadline:
                fr = rx.wait(.1)
                if fr is None:
                    continue
                log.writerow(fr.as_row())
                if last_ts is not None and fr.timestamp < last_ts - .5:
                    reason = 'game_reset'
                    break
                last_ts = fr.timestamp
                now = time.monotonic()
                if now < next_image:
                    continue
                next_image = now + 1 / fps
                hwnd = find_game_window(window)
                rect = find_window_rect(window)
                if not game_window_active(hwnd) or rect is None or min(rect[2:]) < 64:
                    writer.rejected += 1
                    continue
                start = time.time()
                shot = np.asarray(screen.grab(dict(zip(('left', 'top', 'width', 'height'), rect))))[:, :, :3]
                end = time.time()
                if not game_window_active(hwnd):
                    writer.rejected += 1
                    continue
                writer.add(shot, fr, start, end)
            else:
                reason = 'duration'
    except KeyboardInterrupt:
        reason = 'interrupted'
    finally:
        if rx is not None:
            rx.close()
        writer.close(reason)
    if writer.frames == 0:
        raise RuntimeError(f'No aligned frames captured; see {writer.out / "capture.json"}')
    return writer.meta
