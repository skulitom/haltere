"""Live gate vision for the pilot: capture the game window, run GateNet, hand the pilot a goal direction.

A background thread grabs the Liftoff window with mss at up to ``fps`` frames per second, resizes the
game view to the network's input size and runs GateNet on the GPU. The latest detection is converted
into a direction (unit vector in the drone's body frame) and a distance estimate (from the gate's
apparent width, the camera's focal length and the nominal gate width). The pilot builds the brain's
goal vector from this instead of from telemetry positions.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .camera import Camera
from .gates import GATE_WIDTH_M
from .model import IN_H, IN_W, decode
from .train import load_gatenet


@dataclass
class Detection:
    t: float = 0.0                 # wall time of the frame
    p_visible: float = 0.0
    u: float = 0.0                 # pixel coordinates in the network's input frame
    v: float = 0.0
    width_px: float = 0.0
    direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    dist_m: float = 0.0
    frames: int = 0


class GateVision:
    def __init__(self, ckpt: str, cam: Camera, window_title: str = 'Liftoff', fps: float = 15.0,
                 device: str = 'cuda', p_thresh: float = 0.5):
        self.net = load_gatenet(ckpt, device)
        self.device = next(self.net.parameters()).device
        self.cam = cam.scaled(IN_W, IN_H)          # focal length at the network's input resolution
        self.title = window_title
        self.fps = fps
        self.p_thresh = p_thresh
        self.latest = Detection()
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "GateVision":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop = True

    def get(self) -> Detection:
        with self._lock:
            return self.latest

    def _run(self) -> None:
        import cv2
        import mss
        from ..liftoff.recorder import find_window_rect
        sct = mss.mss()
        region = None
        n = 0
        t_next = time.perf_counter()
        while not self._stop:
            if region is None:
                rr = find_window_rect(self.title)
                if rr is None or rr[2] < 64:
                    time.sleep(0.5)
                    continue
                region = {'left': rr[0], 'top': rr[1], 'width': rr[2], 'height': rr[3]}
            shot = np.asarray(sct.grab(region))[:, :, :3][:, :, ::-1]
            if shot.shape[0] < 64:                      # minimized window: re-find it
                region = None
                time.sleep(0.5)
                continue
            # the same path the training frames took: 640x360 bilinear, JPEG at quality 90, then the network's
            # input size with area resampling (a direct 1920 -> 320 resize looks different to the network)
            small = cv2.resize(np.ascontiguousarray(shot), (640, 360), interpolation=cv2.INTER_LINEAR)
            ok, enc = cv2.imencode('.jpg', cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            small = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else small
            img = cv2.resize(small, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)[None].to(self.device)
            with torch.no_grad():
                d = decode(self.net(x))[0].cpu().numpy()
            p, u_n, v_n, width_px = float(d[0]), float(d[1]), float(d[2]), float(d[3])
            u = (u_n + 1) / 2 * IN_W
            v = (v_n + 1) / 2 * IN_H
            direction = self.cam.unproject_body(np.array([[u, v]]))[0]
            dist = self.cam.f * GATE_WIDTH_M / max(width_px, 4.0)
            n += 1
            det = Detection(time.time(), p, u, v, width_px, direction, float(dist), n)
            with self._lock:
                self.latest = det
            t_next += 1.0 / self.fps
            rem = t_next - time.perf_counter()
            if rem > 0:
                time.sleep(rem)
            else:
                t_next = time.perf_counter()
