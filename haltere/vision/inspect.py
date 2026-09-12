"""Overlay projected gate labels (and GateNet predictions) on dataset frames to check them by eye."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def overlay(dataset: str | Path, out: str | Path, every: int = 25, count: int = 12, cols: int = 3,
            ckpt: str | None = None, device: str = 'cuda') -> Path:
    from PIL import Image, ImageDraw
    d = Path(dataset)
    labels = json.loads((d / 'labels.json').read_text(encoding='utf-8'))
    picks = labels[::every][:count]
    net = None
    if ckpt:
        import torch
        from .model import IN_H, IN_W, decode
        from .train import load_gatenet
        net = load_gatenet(ckpt, device)
        dev = next(net.parameters()).device
    ims = []
    for lab in picks:
        im = Image.open(d / 'frames' / lab['file']).convert('RGB')
        w, h = im.size
        clean = np.asarray(im).copy()          # the network must see the frame without the overlay drawings
        dr = ImageDraw.Draw(im)
        if lab['visible']:
            u, v, s = lab['u'], lab['v'], lab['width_px'] / 2
            dr.rectangle([(u - s, v - s), (u + s, v + s)], outline=(0, 255, 0), width=2)
            dr.text((u - s, v - s - 12), f"gate {lab.get('gate')} {lab['dist_m']:.1f} m", fill=(0, 255, 0))
        else:
            dr.text((5, 5), 'no gate label', fill=(255, 120, 0))
        if net is not None:
            import cv2
            arr = cv2.resize(clean, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)[None].to(dev)
            with torch.no_grad():
                p, un, vn, wpx = decode(net(x))[0].cpu().numpy()
            if p > 0.5:
                u, v, s = (un + 1) / 2 * w, (vn + 1) / 2 * h, wpx * w / IN_W / 2
                dr.rectangle([(u - s, v - s), (u + s, v + s)], outline=(255, 0, 0), width=2)
                dr.text((u - s, v + s + 2), f'net p={p:.2f}', fill=(255, 0, 0))
            else:
                dr.text((5, 18), f'net: no gate (p={p:.2f})', fill=(255, 0, 0))
        dr.text((5, h - 14), lab['file'], fill=(255, 255, 0))
        ims.append(im)
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * w, rows * h), (20, 20, 20))
    for i, im in enumerate(ims):
        canvas.paste(im, ((i % cols) * w, (i // cols) * h))
    canvas.save(out)
    return Path(out)
