"""Replay capture for offline runtime benches: a recorded flight video served as live camera frames.

Never used in flight (the flight CLI offers only the mss and dxgi backends). `RetinaCamera` selects it with
``backend='replay:<spec>'``, where spec is ``<video path>?start=S&pad=MS&anchor=FILE`` (all optional but the
path):

- the video is a flight recording (1928 x 720 composite: the gameplay crop x >= 648 is used) or a 1280 x 720
  gameplay video, decoded forward in real time: a read returns the newest frame for
  ``start + (now - anchor)`` seconds of video (the anchor is the float monotonic time in FILE, polled until
  it exists; until then the first frame is served);
- ``pad`` sleeps until the read has taken at least MS milliseconds, standing in for the screen-capture cost
  (live mss capture: 18-22 ms median on the fast-stack flights).

Reads return (capture time = monotonic time at the start of the read, RGB uint8 frame), like DxGameCapture.
"""
from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np

COMPOSITE_GAMEPLAY_X = 648


class ReplayCapture:
    def __init__(self, path, start=0., pad_ms=0., anchor=None):
        import cv2
        self.path = str(path)
        self.video = cv2.VideoCapture(self.path)
        if not self.video.isOpened():
            raise FileNotFoundError(self.path)
        self.fps = float(self.video.get(cv2.CAP_PROP_FPS)) or 18.
        self.count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start, self.pad = float(start), float(pad_ms)/1000.
        self.anchor_file = Path(anchor) if anchor else None
        self.anchor = None if self.anchor_file else time.monotonic()
        if self.start > 0:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, int(self.start*self.fps))
        self.index = int(self.start*self.fps)-1
        self.frame = None
        self.reads = 0

    @classmethod
    def from_spec(cls, spec):
        path, _, query = str(spec).partition('?')
        q = {k: v[-1] for k, v in parse_qs(query).items()}
        return cls(path, start=float(q.get('start', 0.)), pad_ms=float(q.get('pad', 0.)), anchor=q.get('anchor'))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _anchor(self):
        if self.anchor is None and self.anchor_file is not None and self.anchor_file.exists():
            try:
                self.anchor = float(self.anchor_file.read_text().strip())
            except ValueError:
                pass
        return self.anchor

    def video_time(self, now):
        anchor = self._anchor()
        return self.start if anchor is None else self.start+max(0., now-anchor)

    def _decode_to(self, index):
        import cv2
        while self.index < index or self.frame is None:
            ok = self.video.grab()
            if not ok:
                return False
            self.index += 1
            if self.index >= index or self.frame is None:
                ok, frame = self.video.retrieve()
                if not ok:
                    return False
                if frame.shape[1] == 1928:
                    frame = frame[:, COMPOSITE_GAMEPLAY_X:]
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return True

    def read(self):
        begin = time.monotonic()
        index = int(self.video_time(begin)*self.fps)
        if not self._decode_to(index):
            return None
        self.reads += 1
        rest = self.pad-(time.monotonic()-begin)
        if rest > 0:
            time.sleep(rest)
        return begin, np.ascontiguousarray(self.frame)

    def finished(self):
        return self.count and self.index >= self.count-1

    def close(self):
        if self.video is not None:
            self.video.release()
            self.video = None
