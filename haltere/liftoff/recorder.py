"""Record the fly brain flying in Liftoff: the game window captured from the screen, composed with a
live panel of the brain's activity, written to an MP4 (and/or shown in a window with ``--show``).

The pilot loop (``haltere liftoff fly``) publishes the neurons' rates and the flight state into
shared memory at 100 Hz; a separate process captures the screen, renders the brain panel and
encodes video, so the control loop is never slowed down by drawing.
"""
from __future__ import annotations

import os
import sys


def _scrub_cv2_from_sys_path() -> None:
    """OpenCV's loader puts its own package directory on sys.path while it imports; a process spawned at that
    moment inherits it, and then the standard library's ``typing`` resolves to ``cv2/typing`` and numpy fails
    to import. Runs at import time of this module (first thing a spawned recorder child imports)."""
    sys.path[:] = [q for q in sys.path if os.path.basename(os.path.normpath(q)).lower() != 'cv2']


_scrub_cv2_from_sys_path()

import ctypes
import ctypes.wintypes as wt
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

STATE_FIELDS = ['t', 'dist', 'thr', 'roll', 'pitch', 'yaw', 'px', 'py', 'pz', 'tx', 'ty', 'tz', 'waypoint', 'n_waypoints',
                'qw', 'qx', 'qy', 'qz', 'ts']   # attitude quaternion (sim frame) and the game timestamp, for datasets


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


def _capture_game_frame(sct, capture, rect=None):
    """Capture only the foreground game; never fall back to the user's desktop."""
    from .commands import find_game_window, game_window_active
    hwnd = find_game_window(capture)
    if not hwnd or not game_window_active(hwnd):
        return None
    region = rect or find_window_rect(capture)
    if region is None or min(region[2:]) < 64:
        return None
    shot = np.asarray(sct.grab(dict(zip(('left', 'top', 'width', 'height'), region))))[:, :, :3][:, :, ::-1]
    return shot if game_window_active(hwnd) else None


def _recorder_main(shared: SharedFlightState, graph_path: str, out: str | None, capture: str | None,
                   rect: tuple | None, fps: int, show: bool, panel_height: int, dataset: str | None = None,
                   dataset_every: int = 2, dataset_size: tuple[int, int] = (640, 360)):
    from ..connectome.graph import BrainGraph
    from ..viz.fastpanel import FastBrainPanel
    from ..viz.render import neuron_layout
    graph = BrainGraph.load(Path(graph_path))
    layout, colors = neuron_layout(graph)
    panel = FastBrainPanel(layout, colors, width=int(panel_height * 0.9), height=panel_height)
    rates_view = np.frombuffer(shared.rates, dtype=np.float32)
    state_view = np.frombuffer(shared.state, dtype=np.float64)
    mu = var = None

    sct = None
    if capture or rect:
        import mss
        sct = mss.mss()
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
    ds_dir = ds_index = None
    n_ds = 0
    if dataset:
        # vision dataset: the game view (no brain panel) downscaled to JPEG + the pose and target for each frame
        from PIL import Image
        ds_dir = Path(dataset)
        (ds_dir / 'frames').mkdir(parents=True, exist_ok=True)
        new = not (ds_dir / 'index.csv').exists()
        ds_index = open(ds_dir / 'index.csv', 'a', encoding='utf-8')
        if new:
            ds_index.write('file,wall_time,' + ','.join(STATE_FIELDS) + '\n')
        print(f'recorder: saving every {dataset_every}th captured frame to {ds_dir} ({dataset_size[0]}x{dataset_size[1]})',
              file=sys.stderr, flush=True)
    t_next = t0 = time.perf_counter()
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
            shot = _capture_game_frame(sct, capture or 'Liftoff', rect)
            if shot is not None and ds_dir is not None and st['ts'] > 0 and n_frames % dataset_every == 0:
                name = f'{n_ds:06d}.jpg'
                Image.fromarray(np.ascontiguousarray(shot)).resize(dataset_size, Image.BILINEAR).save(
                    ds_dir / 'frames' / name, quality=90)
                ds_index.write(f'{name},{time.time():.4f},' + ','.join(f'{st[k]:.5f}' for k in STATE_FIELDS) + '\n')
                n_ds += 1
                if n_ds % 50 == 0:
                    ds_index.flush()
            if shot is None:
                from PIL import Image, ImageDraw
                blank = Image.new('RGB', (frame_w-panel.W if frame_w else int(panel.H*16/9)//2*2, panel.H))
                ImageDraw.Draw(blank).text((20, 20), 'Waiting for foreground Liftoff view', fill='white')
                shot = np.asarray(blank)
            else:
                new_w = max(2, int(shot.shape[1] * panel.H / shot.shape[0]) // 2 * 2)
                shot = _resize_rows_cols(shot, panel.H, new_w)
            frame = np.concatenate([img_panel, shot], axis=1)
        if frame.shape[1] % 2:
            frame = frame[:, :-1]
        if not (out and ffmpeg):
            n_frames += 1
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
            # keep the video in real time: when capture + rendering are slower than `fps` (a busy machine),
            # repeat the current frame for the missed ticks instead of letting the video play fast
            due = int((time.perf_counter() - t0) * fps) - n_frames
            buf = np.ascontiguousarray(frame).tobytes()
            for _ in range(max(1, min(due, 8))):
                proc.stdin.write(buf)
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
    if ds_index is not None:
        ds_index.close()
        print(f'recorder: dataset {ds_dir}: {n_ds} frames', file=sys.stderr, flush=True)


class FlightRecorder:
    """Starts/stops the recorder process; the pilot calls ``shared.publish`` every step."""

    def __init__(self, shared: SharedFlightState, graph_path: str, out: str | None = None, capture: str | None = 'Liftoff',
                 rect: tuple | None = None, fps: int = 20, show: bool = False, panel_height: int = 720,
                 dataset: str | None = None, dataset_every: int = 2):
        self.shared = shared
        self.proc = mp.Process(target=_recorder_main, args=(shared, graph_path, out, capture, rect, fps, show, panel_height,
                                                            dataset, dataset_every), daemon=True)

    def start(self) -> None:
        _scrub_cv2_from_sys_path()          # the child copies sys.path at spawn time
        self.proc.start()

    def stop(self) -> None:
        self.shared.stop.value = 1
        self.proc.join(timeout=15)
        if self.proc.is_alive():
            self.proc.terminate()
