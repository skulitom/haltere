"""Measure a gate detector on labelled datasets, and check which gates a recorded flight passed.

`evaluate` runs GateNet on every frame of the datasets with the runtime's preprocessing and reports, per dataset,
the visibility accuracy, false-positive rate on gate-less frames and the centre/width errors per distance bin: the
numbers that decide whether a self-improvement round (fly by sight, label the recorded frames, retrain) helped.
`gate_passes` replays a dataset's poses through the gate list and tells whether the drone flew THROUGH each gate
it reached or beside it (the race only counts gates flown through, in order).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibrate import load_index
from .gates import load_gate_file
from .model import IN_H, IN_W, decode


def _predict(net, frames: list[Path], batch: int = 128) -> np.ndarray:
    import cv2
    import torch
    dev = next(net.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            xs = []
            for f in frames[i:i + batch]:
                img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
                xs.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
            out.append(decode(net(torch.stack(xs).to(dev))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 4))


def evaluate(ckpt: str | Path, datasets: list[str | Path], device: str = 'cuda',
             bins=((0, 6), (6, 10), (10, 15), (15, 22), (22, 30), (30, 45))) -> dict:
    import torch
    from .train import load_gatenet
    net = load_gatenet(ckpt, device)
    # A number measured on a dataset the checkpoint trained on says nothing about a track it has not seen, and
    # the two are easy to mix up once there are several environments, so every line says which kind it is.
    try:
        trained_on = {Path(x).resolve() for x in torch.load(ckpt, map_location='cpu', weights_only=False).get('datasets', [])}
    except Exception:
        trained_on = set()
    report = {}
    for d in datasets:
        d = Path(d)
        seen = d.resolve() in trained_on
        tag = 'TRAINED ON' if seen else ('held out' if trained_on else 'provenance unknown')
        labels = json.loads((d / 'labels.json').read_text(encoding='utf-8'))
        P = _predict(net, [d / 'frames' / lab['file'] for lab in labels])
        vis = np.array([lab['visible'] for lab in labels], dtype=bool)
        dist = np.array([lab.get('dist_m', np.nan) for lab in labels])
        lu = np.array([lab.get('u', np.nan) for lab in labels])
        lv = np.array([lab.get('v', np.nan) for lab in labels])
        lw = np.array([lab.get('width_px', np.nan) for lab in labels])
        pu, pv, pw = (P[:, 1] + 1) / 2 * 640, (P[:, 2] + 1) / 2 * 360, P[:, 3] * 640 / IN_W
        det = P[:, 0] > 0.5
        r = {'frames': len(labels), 'visible': int(vis.sum()),
             'accuracy': float((det == vis).mean()),
             'false_pos': float((det & ~vis).sum() / max((~vis).sum(), 1)),
             'false_neg': float((~det & vis).sum() / max(vis.sum(), 1)), 'bins': []}
        r['trained_on'] = bool(seen)
        print(f'{d.name} [{tag}]: {len(labels)} frames, {vis.sum()} with a gate; visibility accuracy {100 * r["accuracy"]:.1f}%, '
              f'false positives {100 * r["false_pos"]:.1f}% of gate-less frames, misses {100 * r["false_neg"]:.1f}%')
        for lo, hi in bins:
            m = vis & (dist >= lo) & (dist < hi)
            if not m.any():
                continue
            k = m & det
            row = {'range': (lo, hi), 'n': int(m.sum()), 'detected': float(det[m].mean()),
                   'width_ratio': float(np.median(pw[k] / lw[k])) if k.any() else float('nan'),
                   'du_px': float(np.median(pu[k] - lu[k])) if k.any() else float('nan'),
                   'dv_px': float(np.median(pv[k] - lv[k])) if k.any() else float('nan')}
            r['bins'].append(row)
            print(f'  {lo:2d}-{hi:2d} m: {row["n"]:4d} frames, detected {100 * row["detected"]:5.1f}%, '
                  f'width pred/label {row["width_ratio"]:.2f}, centre offset du {row["du_px"]:+6.1f} dv {row["dv_px"]:+6.1f} px (640x360)')
        report[d.name] = r
    return report


def gate_passes(dataset: str | Path, gates_path: str | Path, through_m: float = 2.0, verbose: bool = True) -> list[dict]:
    """Which gates the recorded flight crossed, and whether it went through them (|lateral| < through_m and
    between the passage height and 3 m above it) or beside/over them."""
    rows = load_index(dataset)
    gates, width_m, _ = load_gate_file(gates_path)
    t = np.array([r['t'] for r in rows])
    P = np.array([[r['px'], r['py'], r['pz']] for r in rows])
    out = []
    for i, g in enumerate(gates):
        gp = np.asarray(g['pos'], dtype=np.float64)
        h = g['heading']
        n = np.array([np.cos(h), np.sin(h), 0.0])
        s = np.array([-np.sin(h), np.cos(h), 0.0])
        along = (P - gp) @ n                                  # < 0 before the gate plane, > 0 past it
        for c in np.where((along[:-1] <= 0) & (along[1:] > 0))[0]:
            lat, dz = float((P[c] - gp) @ s), float(P[c, 2] - gp[2])
            through = abs(lat) < through_m and -0.5 < dz < 3.0
            out.append({'gate': i, 't': float(t[c]), 'lateral_m': lat, 'dz_m': dz, 'through': through})
            if verbose:
                print(f'gate {i} crossed at t={t[c]:6.1f} s: {lat:+5.1f} m sideways, {dz:+4.1f} m above the passage point'
                      f' -> {"THROUGH" if through else "beside / over"}')
    if verbose and not out:
        print('no gate plane crossed')
    return out
