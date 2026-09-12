"""A fast pixel renderer for the live brain-activity panel (tens of frames per second for 30k neurons).

Neurons are splatted into an RGB buffer with numpy (bincount per colour channel), text and stick
bars are drawn with Pillow. Used by the Liftoff recorder; the offline video uses matplotlib.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .render import LEGEND

FONT_CANDIDATES = [r'C:\Windows\Fonts\consola.ttf', r'C:\Windows\Fonts\segoeui.ttf', r'C:\Windows\Fonts\arial.ttf']


def _font(size: int):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class FastBrainPanel:
    def __init__(self, layout: np.ndarray, colors: np.ndarray, width: int = 640, height: int = 720,
                 title: str = 'fly brain (male CNS v1.0), live on Liftoff telemetry'):
        self.W, self.H = width, height
        self.colors = colors.astype(np.float32)
        top, bottom, side = 46, 150, 14
        area_h = height - top - bottom
        area_w = width - 2 * side
        px = (side + layout[:, 0] * (area_w - 1)).astype(np.int64)
        py = (top + (1.0 - layout[:, 1]) * (area_h - 1)).astype(np.int64)
        self.base = py * width + px
        self.offsets3 = [(dy * width + dx, 1.0 if (dx == 0 and dy == 0) else 0.45) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        self.offsets5 = [(dy * width + dx, 0.30) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2) if max(abs(dx), abs(dy)) == 2]
        self.n_pix = width * height
        off3 = np.array([o for o, _ in self.offsets3]); w3 = np.array([w for _, w in self.offsets3], dtype=np.float32)
        self.idx3 = (self.base[:, None] + off3[None, :]).ravel()
        self.w3 = np.tile(w3, (len(self.base), 1))
        self.off5 = np.array([o for o, _ in self.offsets5])
        self.font = _font(15)
        self.font_small = _font(12)
        # static background: title + legend
        img = Image.new('RGB', (width, height), (0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((side, 10), title, fill=(235, 235, 235), font=self.font)
        y = height - bottom + 8
        short = ['senses (haltere, wing, eyes, antennae)', 'central complex (compass, goal)', 'descending neurons',
                 'premotor (readout)', 'wing motor neurons (readout)']
        for (label, rgb), text in zip(LEGEND, short):
            c = tuple(int(255 * v) for v in rgb)
            d.ellipse((side, y + 4, side + 8, y + 12), fill=c)
            d.text((side + 14, y), text, fill=(210, 210, 210), font=self.font_small)
            y += 16
        self.static = np.asarray(img).astype(np.float32)
        self.stick_box = (int(width * 0.64), height - bottom + 12, width - side, height - 14)

    def render(self, rates: np.ndarray, mu: np.ndarray, sd: np.ndarray, state: dict) -> np.ndarray:
        act = np.clip((rates - mu) / sd, -1.0, 4.0).astype(np.float32)
        bright = np.clip(0.12 + 0.32 * np.clip(act, 0, 4), 0.0, 1.0)
        rgb = np.clip(self.colors * (0.25 + 1.1 * bright[:, None]), 0, 1)
        rgb = np.clip(rgb + 0.35 * np.clip(act - 2.0, 0, 2)[:, None] / 2.0, 0, 1)
        scale = (0.45 + 0.55 * bright)[:, None] * rgb
        idx = self.idx3                                              # [N*9] pixel indices, precomputed
        w = self.w3                                                  # [N*9] kernel weights
        buf = np.stack([np.bincount(idx, weights=(scale[:, c][:, None] * w).ravel(), minlength=self.n_pix)
                        for c in range(3)], axis=1)
        hot = act > 1.8
        if hot.any():
            hb = (self.base[hot][:, None] + self.off5[None, :]).ravel()
            hs = scale[hot] * np.clip(act[hot] - 1.8, 0, 2.2)[:, None] / 2.2
            for c in range(3):
                buf[:, c] += np.bincount(hb, weights=np.repeat(hs[:, c], len(self.off5)) * 0.30, minlength=self.n_pix)
        frame = np.clip(self.static + 255.0 * np.clip(buf, 0, 1).reshape(self.H, self.W, 3), 0, 255).astype(np.uint8)
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        wp = f'   waypoint {int(state.get("waypoint", 0)) + 1}/{int(state.get("n_waypoints", 0))}' if state.get('n_waypoints', 0) else ''
        d.text((14, 26), f't = {state.get("t", 0.0):6.1f} s   distance to target {state.get("dist", 0.0):4.2f} m{wp}',
               fill=(255, 255, 255), font=self.font_small)
        x0, y0, x1, y1 = self.stick_box
        labels = ['throttle', 'roll', 'pitch', 'yaw']
        keys = ['thr', 'roll', 'pitch', 'yaw']
        row_h = (y1 - y0) / 4
        mid = (x0 + 60 + x1) / 2
        half = (x1 - x0 - 60) / 2
        for i, (lab, key) in enumerate(zip(labels, keys)):
            yy = y0 + i * row_h
            d.text((x0, yy + 2), lab, fill=(200, 200, 200), font=self.font_small)
            v = float(np.clip(state.get(key, 0.0), -1, 1))
            d.line((mid, yy + 2, mid, yy + row_h - 4), fill=(80, 80, 80))
            col = (250, 77, 90) if i == 0 else (140, 153, 242)
            xa, xb = (mid, mid + v * half) if v >= 0 else (mid + v * half, mid)
            d.rectangle((xa, yy + 5, xb, yy + row_h - 7), fill=col)
        return np.asarray(img)
