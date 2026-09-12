"""Record the fly brain flying in Liftoff: the game window captured from the screen, composed with a
live panel of the brain's activity, written to an MP4 (and/or shown in a window with ``--show``).

The pilot loop (``haltere liftoff fly``) publishes the neurons' rates and the flight state into
shared memory at 100 Hz; a separate process captures the screen, renders the brain panel and
encodes video, so the control loop is never slowed down by drawing.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

STATE_FIELDS = ['t', 'dist', 'thr', 'roll', 'pitch', 'yaw', 'px', 'py', 'pz', 'tx', 'ty', 'tz', 'waypoint', 'n_waypoints']


def find_window_rect(title_substring: str) -> tuple[int, int, int, int] | None:
    """Screen rectangle (left, top, width, height) of the client area of the first visible window whose
    title contains the text."""
    user32 = ctypes.windll.user32
    found = []

    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if title_substring.lower() in buf.value.lower():
                    found.append(hwnd)
        return True

    proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(cb)
    user32.EnumWindows(proc, 0)
    if not found:
        return None
    r = wt.RECT()
    user32.GetClientRect(found[0], ctypes.byref(r))
    pt = wt.POINT(0, 0)
    user32.ClientToScreen(found[0], ctypes.byref(pt))
    w, h = r.right - r.left, r.bottom - r.top
    if w < 64 or h < 64:
        user32.GetWindowRect(found[0], ctypes.byref(r))
        return (r.left, r.top, r.right - r.left, r.bottom - r.top)
    return (pt.x, pt.y, w, h)


class SharedFlightState:
    """Shared memory written by the pilot process and read by the recorder."""

    def __init__(self, n_neurons: int):
        self.rates = mp.Array('f', n_neurons, lock=False)
        self.state = mp.Array('d', len(STATE_FIELDS), lock=False)
        self.stop = mp.Value('i', 0)
        self.n = n_neurons

    def publish(self, rates: np.ndarray | None, **kw) -> None:
        if rates is not None:
            np.frombuffer(self.rates, dtype=np.float32)[:] = rates
        st = np.frombuffer(self.state, dtype=np.float64)
        for k, v in kw.items():
            st[STATE_FIELDS.index(k)] = float(v)


def _resize_rows_cols(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    ys = (np.arange(new_h) * img.shape[0] / new_h).astype(int).clip(0, img.shape[0] - 1)
    xs = (np.arange(new_w) * img.shape[1] / new_w).astype(int).clip(0, img.shape[1] - 1)
    return img[ys][:, xs]


def _recorder_main(shared: SharedFlightState, graph_path: str, out: str | None, capture: str | None,
                   rect: tuple | None, fps: int, show: bool, panel_height: int):
    from ..connectome.graph import BrainGraph
    from ..viz.fastpanel import FastBrainPanel
    from ..viz.render import neuron_layout
    graph = BrainGraph.load(Path(graph_path))
    layout, colors = neuron_layout(graph)
    panel = FastBrainPanel(layout, colors, width=int(panel_height * 0.9), height=panel_height)
    rates_view = np.frombuffer(shared.rates, dtype=np.float32)
    state_view = np.frombuffer(shared.state, dtype=np.float64)
    mu = var = None

    sct = region = None
    if capture or rect:
        import mss
        sct = mss.mss()
        if rect:
            region = {'left': rect[0], 'top': rect[1], 'width': rect[2], 'height': rect[3]}
    viewer = None
    if show:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.ion()
        vfig = plt.figure('Haltere: fly brain', figsize=(panel.W / 100, panel.H / 100), dpi=100, facecolor='black')
        vax = vfig.add_axes([0, 0, 1, 1])
        vax.axis('off')
        viewer = (plt, vfig, vax.imshow(np.zeros((panel.H, panel.W, 3), dtype=np.uint8)))
    proc = None
    frame_h = frame_w = None
    ffmpeg = shutil.which('ffmpeg')
    t_next = time.perf_counter()
    n_frames = 0
    t_report = time.perf_counter()
    while not shared.stop.value:
        r = rates_view.copy()
        if mu is None:
            mu, var = r.copy(), np.full_like(r, 0.01)
        else:
            mu += 0.02 * (r - mu)
            var += 0.02 * ((r - mu) ** 2 - var)
        st = dict(zip(STATE_FIELDS, state_view.copy()))
        img_panel = panel.render(r, mu, np.sqrt(var + 1e-4), st)
        frame = img_panel
        if sct is not None:
            if region is None:
                rr = find_window_rect(capture)
                if rr is not None:
                    region = {'left': rr[0], 'top': rr[1], 'width': rr[2], 'height': rr[3]}
                else:
                    mon = sct.monitors[1]
                    region = {'left': mon['left'], 'top': mon['top'], 'width': mon['width'], 'height': mon['height']}
                    print(f'recorder: window "{capture}" not found, capturing the primary monitor', file=sys.stderr, flush=True)
            shot = np.asarray(sct.grab(region))[:, :, :3][:, :, ::-1]   # BGRA -> RGB
            new_w = max(2, int(shot.shape[1] * panel.H / shot.shape[0]) // 2 * 2)
            shot = _resize_rows_cols(shot, panel.H, new_w)
            frame = np.concatenate([img_panel, shot], axis=1)
        if frame.shape[1] % 2:
            frame = frame[:, :-1]
        if out and ffmpeg:
            if proc is None:
                frame_h, frame_w = frame.shape[:2]
                cmd = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{frame_w}x{frame_h}',
                       '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'veryfast', out]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                print(f'recorder: writing {frame_w}x{frame_h} @ {fps} fps to {out}', file=sys.stderr, flush=True)
            if frame.shape[:2] != (frame_h, frame_w):
                fixed = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                h, w = min(frame_h, frame.shape[0]), min(frame_w, frame.shape[1])
                fixed[:h, :w] = frame[:h, :w]
                frame = fixed
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            n_frames += 1
        if viewer is not None:
            plt, vfig, im = viewer
            im.set_data(img_panel)
            vfig.canvas.draw_idle()
            vfig.canvas.flush_events()
            if not plt.fignum_exists(vfig.number):
                break
        now = time.perf_counter()
        if now - t_report > 5.0:
            n_interval = n_frames - getattr(_recorder_main, '_last_n', 0)
            _recorder_main._last_n = n_frames
            print(f'recorder: {n_frames} frames written, {n_interval / (now - t_report):.0f} fps' if out else 'recorder: live',
                  file=sys.stderr, flush=True)
            t_report = now
        t_next += 1.0 / fps
        rem = t_next - time.perf_counter()
        if rem > 0:
            time.sleep(rem)
        else:
            t_next = time.perf_counter()
    if proc is not None:
        proc.stdin.close()
        proc.wait()
        print(f'recorder: finished {out}', file=sys.stderr, flush=True)


class FlightRecorder:
    """Starts/stops the recorder process; the pilot calls ``shared.publish`` every step."""

    def __init__(self, shared: SharedFlightState, graph_path: str, out: str | None = None, capture: str | None = 'Liftoff',
                 rect: tuple | None = None, fps: int = 20, show: bool = False, panel_height: int = 720):
        self.shared = shared
        self.proc = mp.Process(target=_recorder_main, args=(shared, graph_path, out, capture, rect, fps, show, panel_height),
                               daemon=True)

    def start(self) -> None:
        self.proc.start()

    def stop(self) -> None:
        self.shared.stop.value = 1
        self.proc.join(timeout=15)
        if self.proc.is_alive():
            self.proc.terminate()
