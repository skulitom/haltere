"""Live gate vision for the pilot: capture the game window, run GateNet, hand the pilot a goal direction.

A background thread grabs the Liftoff window with mss at up to ``fps`` frames per second, resizes the
game view to the network's input size and runs GateNet on the GPU. The latest detection is converted
into a direction (unit vector in the drone's body frame) and a distance estimate (from the gate's
apparent width, the camera's focal length and a NOMINAL gate width, which the detection carries so the
pilot can divide it back out). The pilot builds the brain's goal vector from this instead of from
telemetry positions.
"""
from __future__ import annotations

import sys
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
    t: float = 0.0                 # wall time of the screen grab
    p_visible: float = 0.0
    u: float = 0.0                 # pixel coordinates in the network's input frame
    v: float = 0.0
    width_px: float = 0.0
    direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    dist_m: float = 0.0
    frames: int = 0
    # the arch width ``dist_m`` was computed with. The detector cannot know how wide the arches of an unseen course
    # are, so it reports a range for a nominal one; ``dist_m / width_m`` is the range per metre of gate width, the
    # part of the measurement that IS course-independent, and the pilot multiplies it by its own online estimate.
    width_m: float = GATE_WIDTH_M


def detection_geometry(cam: Camera, u: float, v: float, width_px: float,
                       width_m: float = GATE_WIDTH_M) -> tuple[np.ndarray, float]:
    """Body-frame unit direction through the detected centre (u, v) and the range from the apparent width, for a
    camera at the network's input resolution. Shared by the live detector and the simulator rehearsal.

    The range is strictly proportional to ``width_m``: apparent size gives f * stretch / width_px, the range per
    metre of gate width, and nothing in an image says how wide the arch is. The default is the nominal width of the
    course the detector was trained on; the by-sight pilot divides it back out and scales by the width it has
    triangulated for the course it is on (``SightPilot`` / ``SightParams.width_est``).

    Off the optical axis a rectilinear image stretches things: a gate seen at horizontal offset du and radial offset
    (du, dv) appears wider by sqrt(f^2 + du^2) * sqrt(f^2 + du^2 + dv^2) / f^2, a factor 2.5 at the edge of a
    116 deg view; the range divides that out."""
    direction = cam.unproject_body(np.array([[u, v]]))[0]
    f = cam.f
    du, dv = u - cam.width / 2, v - cam.height / 2
    stretch = np.sqrt(f * f + du * du) * np.sqrt(f * f + du * du + dv * dv) / (f * f)
    dist = f * width_m * stretch / max(width_px, 4.0)
    return direction, float(dist)


class GateVision:
    def __init__(self, ckpt: str, cam: Camera, window_title: str = 'Liftoff', fps: float = 15.0,
                 device: str = 'cuda', p_thresh: float = 0.5, gate_width_m: float = GATE_WIDTH_M):
        self.net = load_gatenet(ckpt, device)
        self.device = next(self.net.parameters()).device
        self.cam = cam.scaled(IN_W, IN_H)          # focal length at the network's input resolution
        self.gate_width_m = float(gate_width_m)    # the width its ranges assume; the pilot re-scales them
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
        errors = 0
        t_next = time.perf_counter()
        while not self._stop:
            try:
                if region is None:
                    rr = find_window_rect(self.title)
                    if rr is None or rr[2] < 64:
                        time.sleep(0.5)
                        continue
                    region = {'left': rr[0], 'top': rr[1], 'width': rr[2], 'height': rr[3]}
                t_grab = time.time()                    # the detection describes this moment, not the end of inference
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
                direction, dist = detection_geometry(self.cam, u, v, width_px, self.gate_width_m)
                n += 1
                det = Detection(t_grab, p, u, v, width_px, direction, dist, n, self.gate_width_m)
                with self._lock:
                    self.latest = det
                t_next += 1.0 / self.fps
                rem = t_next - time.perf_counter()
                if rem > 0:
                    time.sleep(rem)
                else:
                    t_next = time.perf_counter()
            except Exception as e:                   # noqa: BLE001 - a dead thread would freeze the detections
                errors += 1
                if errors == 1 or errors % 50 == 0:
                    print(f'VISION: capture/inference error #{errors} ({type(e).__name__}: {e}); retrying',
                          file=sys.stderr, flush=True)
                region = None
                try:
                    sct = mss.mss()
                except Exception:                    # noqa: BLE001
                    pass
                time.sleep(0.5)
                t_next = time.perf_counter()
