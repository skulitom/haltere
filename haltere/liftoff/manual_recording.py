"""Passive, user-operated recording of Liftoff images, controls and telemetry.

Run with ``python -m haltere.liftoff.manual_recording`` on the player's desktop.
This module does not construct a pilot, send input, or change the game settings.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .dataset import DatasetWriter
from .telemetry import DEFAULT_STREAM, TelemetryFrame, TelemetryReceiver, read_config


PROFILES = [
    ('straw_bale_race', 'Straw Bale — race, 3 laps'),
    ('pine_valley_race', 'Pine Valley — race, 3 laps'),
    ('minus_two_race', 'Minus Two — race, 3 laps'),
    ('straw_bale_fence', 'Straw Bale — fence freestyle'),
]


def live_pose(frame):
    """Results/menu telemetry has an all-zero position despite an advancing clock."""
    return (np.isfinite([frame.timestamp, *frame.position, *frame.attitude, *frame.input]).all()
            and abs(np.linalg.norm(frame.attitude) - 1) < .01
            and np.any(frame.position != 0))


def discontinuity(previous, current):
    if previous is None:
        return None
    dt = current.timestamp - previous.timestamp
    if dt < -.5:
        return 'game_reset'
    if np.linalg.norm(current.position - previous.position) > max(10., 50. * max(dt, 0.)):
        return 'position_jump'
    return None


class ManualTake:
    """One flight with its own origin; immutable directory, explicit provenance."""

    def __init__(self, root, profile, camera, fps):
        import cv2
        key, label = PROFILES[profile]
        stamp = time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns() % 1000000:06d}'
        self.path = Path(root) / key / stamp
        self.writer = DatasetWriter(self.path, course=label, camera=camera)
        self.writer.meta.update(controller='Stark (user stated)', pilot='human',
                                profile=key, expected_laps=3 if key.endswith('_race') else None,
                                camera_alignment_verified=False, map_label_verified=False,
                                video_timing='Preview uses a fixed frame rate; index.csv timestamps are authoritative.')
        self.raw = (self.path / 'telemetry.csv').open('w', newline='', encoding='utf-8')
        self.log = csv.writer(self.raw)
        self.log.writerow(TelemetryFrame.columns())
        self.video = cv2.VideoWriter(str(self.path / 'preview.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps, (640, 360))
        if not self.video.isOpened():
            self.raw.close()
            self.writer.close('video_error')
            raise RuntimeError('Could not open the preview video encoder')
        self.writer._manifest('recording')
        self.started = time.monotonic()
        self.last_flush = self.started

    def telemetry(self, frame):
        self.log.writerow(frame.as_row())
        if time.monotonic() - self.last_flush >= 1:
            self.raw.flush()
            self.writer._manifest('recording')
            self.last_flush = time.monotonic()

    def image(self, image, frame, start, end):
        import cv2
        if self.writer.add(image, frame, start, end):
            self.video.write(cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR))

    def close(self, reason):
        self.video.release()
        self.raw.close()
        self.writer.close(reason)


class RecordingWorker(threading.Thread):
    def __init__(self, root, camera, fps=20., port=9001):
        super().__init__(daemon=False)
        self.root, self.camera, self.fps, self.port = Path(root), camera, fps, port
        self.commands = queue.Queue()
        self.snapshot = {'state': 'Starting recorder', 'armed': False, 'frames': 0, 'path': ''}
        self.done = threading.Event()

    def run(self):
        import mss
        from .commands import find_game_window, game_window_active
        from .recorder import find_window_rect
        take = rx = None
        armed, profile, previous, next_image = False, 0, None, 0.
        last_packet, last_publish = 0., 0.
        state, last_path, last_frames = 'Idle — select a take, then start', '', 0

        def close(reason):
            nonlocal take, last_path, last_frames
            if take is not None:
                last_path, last_frames = str(take.path.resolve()), take.writer.frames
                take.close(reason)
                take = None

        self.root.mkdir(parents=True, exist_ok=True)
        status_path = self.root / 'recorder-status.json'
        try:
            stream = (read_config() or {}).get('StreamFormat', DEFAULT_STREAM)
            if not {'Timestamp', 'Position', 'Attitude', 'Input'}.issubset(stream):
                raise ValueError('Liftoff telemetry configuration is missing required fields')
            rx = TelemetryReceiver(port=self.port, stream=stream)
            with mss.mss() as screen:
                while not self.done.is_set():
                    while True:
                        try:
                            command, value = self.commands.get_nowait()
                        except queue.Empty:
                            break
                        if command == 'start':
                            close('new_take')
                            profile, armed, previous = int(value), True, None
                            state = 'Armed — return to Liftoff and fly'
                        elif command == 'stop':
                            close('user_stop')
                            armed, previous = False, None
                            state = 'Saved — ready for the next take'
                        elif command == 'quit':
                            self.done.set()
                    if self.done.is_set():
                        break
                    now = time.monotonic()
                    fr = rx.wait(.02)
                    if fr is not None:
                        last_packet = now
                    if armed and fr is not None:
                        hwnd = find_game_window('Liftoff')
                        active = bool(hwnd and game_window_active(hwnd))
                        if not live_pose(fr):
                            close('menu_or_race_end')
                            previous = None
                            state = 'Armed — waiting for a live flight'
                        elif previous is not None and fr.timestamp == previous.timestamp:
                            state = 'Armed — game paused' if take is None else 'Recording — game paused'
                        else:
                            reason = discontinuity(previous, fr)
                            if reason:
                                close(reason)
                            previous = fr
                            if take is None and active:
                                take = ManualTake(self.root, profile, self.camera, self.fps)
                                next_image = 0.
                            if take is not None:
                                take.telemetry(fr)
                                state = 'Recording' if active else 'Recording telemetry — game view hidden'
                                if active and now >= next_image:
                                    next_image = now + 1. / self.fps
                                    rect = find_window_rect('Liftoff')
                                    if rect is not None and min(rect[2:]) >= 64:
                                        start = time.time()
                                        shot = np.asarray(screen.grab(dict(zip(
                                            ('left', 'top', 'width', 'height'), rect))))[:, :, :3]
                                        end = time.time()
                                        if game_window_active(hwnd):
                                            take.image(shot, fr, start, end)
                    if armed and now - last_packet > 2:
                        state = 'Armed — waiting for Liftoff telemetry'
                    if now - last_publish >= .25:
                        self.snapshot = dict(state=state, armed=armed, profile=PROFILES[profile][0],
                                             frames=take.writer.frames if take else last_frames,
                                             path=str(take.path.resolve()) if take else last_path,
                                             telemetry_age_s=None if not last_packet else now-last_packet)
                        temporary = status_path.with_suffix('.tmp')
                        temporary.write_text(json.dumps(self.snapshot, indent=2), encoding='utf-8')
                        temporary.replace(status_path)
                        last_publish = now
        except Exception as exc:
            self.snapshot = dict(state=f'Error: {exc}', armed=False, frames=last_frames, path=last_path)
        finally:
            close('recorder_closed')
            if rx is not None:
                rx.close()
            self.snapshot['armed'] = False
            if not self.snapshot['state'].startswith('Error:'):
                self.snapshot['state'] = 'Recorder closed'
            status_path.write_text(json.dumps(self.snapshot, indent=2), encoding='utf-8')
            self.done.set()


def main():
    import ctypes
    import ctypes.wintypes as wt
    import tkinter as tk
    from tkinter import ttk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/vision/manual_sessions')
    parser.add_argument('--camera', default='configs/camera_seat.yaml')
    parser.add_argument('--fps', type=float, default=20.)
    parser.add_argument('--port', type=int, default=9001)
    args = parser.parse_args()
    if not 0 < args.fps <= 60:
        parser.error('--fps must be between 0 and 60')
    worker = RecordingWorker(args.out, args.camera, args.fps, args.port)
    app = tk.Tk()
    app.title('Haltere — gameplay recorder')
    app.geometry('580x320')
    app.resizable(False, False)
    body = ttk.Frame(app, padding=20)
    body.pack(fill='both', expand=True)
    ttk.Label(body, text='Record your flights', font=('Segoe UI', 17, 'bold')).pack(anchor='w')
    ttk.Label(body, text='Choose the matching map and mode in Liftoff.').pack(anchor='w', pady=(6, 12))
    selected = tk.StringVar(value=PROFILES[0][1])
    choices = ttk.Combobox(body, textvariable=selected, values=[p[1] for p in PROFILES], state='readonly')
    choices.pack(fill='x')
    state = tk.StringVar(value='Idle — plug in your Stark controller when ready')
    count = tk.StringVar(value='No take recorded yet')
    ttk.Label(body, textvariable=state, wraplength=535).pack(anchor='w', pady=(16, 4))
    ttk.Label(body, textvariable=count, wraplength=535).pack(anchor='w')
    desired_armed = False

    def toggle():
        nonlocal desired_armed
        if worker.done.is_set():
            return
        desired_armed = not desired_armed
        worker.commands.put(('start', choices.current()) if desired_armed else ('stop', None))
        choices.configure(state='disabled' if desired_armed else 'readonly')
        button.configure(text='Stop and save' if desired_armed else 'Start recording')
        import winsound
        winsound.Beep(1200 if desired_armed else 600, 100)

    button = ttk.Button(body, text='Start recording', command=toggle)
    button.pack(anchor='w', pady=(16, 10))
    ttk.Label(body, text='Ctrl+Alt+F9: start / stop     Ctrl+Alt+F10: next take (when stopped)').pack(anchor='w')
    ttk.Label(body, text='Resets start a new file. Only the active Liftoff view is captured.').pack(anchor='w', pady=4)
    hotkeys = []
    user32 = ctypes.windll.user32
    for ident, vk in [(101, 0x78), (102, 0x79)]:
        if user32.RegisterHotKey(None, ident, 0x4003, vk):
            hotkeys.append(ident)
    shortcut_warning = '' if len(hotkeys) == 2 else ' — use the button; a shortcut is unavailable'

    def poll():
        nonlocal desired_armed
        msg = wt.MSG()
        while user32.PeekMessageW(ctypes.byref(msg), None, 0x0312, 0x0312, 1):
            if msg.wParam == 101:
                toggle()
            elif msg.wParam == 102 and not desired_armed:
                choices.current((choices.current()+1) % len(PROFILES))
        snap = worker.snapshot
        state.set(snap['state'] + shortcut_warning)
        count.set(f"{snap['frames']:,} frames saved" + (f"  •  {Path(snap['path']).parent.name}" if snap['path'] else ''))
        if worker.done.is_set():
            desired_armed = False
            button.configure(state='disabled')
        app.after(100, poll)

    def shutdown():
        worker.commands.put(('quit', None))
        for ident in hotkeys:
            user32.UnregisterHotKey(None, ident)
        def finish():
            if worker.done.is_set():
                app.destroy()
            else:
                app.after(100, finish)
        finish()

    app.protocol('WM_DELETE_WINDOW', shutdown)
    worker.start()
    app.after(100, poll)
    app.mainloop()
    worker.join(timeout=5)


if __name__ == '__main__':
    main()
