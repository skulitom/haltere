"""Three measurements that say whether a gate detector will work anywhere but the course it was trained on.

`vision eval` answers "how good is it here", on labelled frames of a course whose gates are known. None of
that is available on a track the drone has never flown, and the numbers it gives are flattering in a
specific way: a detector that has learned one course's palette scores well on that course right up until
the light changes.

  probe          - on an UNLABELLED dataset (a new track, before anyone has built its gate list): how often
                   does it fire at all, how confident is it, and what does it fire at. Compare against its
                   false-positive rate at home: below that, the detector is seeing nothing.
  colour_stress  - on labelled held-out frames: how much recall survives grayscale, desaturation, hue
                   rotation, darkness, gamma and blur. A detector keying on shape barely moves; one keying
                   on a palette falls apart. This is the cheapest proxy for a new environment there is,
                   because it needs no new data at all.
  fit_range_corr - the per-detector range table the pilot corrects its ranges with. It has to be refitted
                   whenever the detector changes; on labelled frames the label IS the matched arch, so it
                   falls out directly instead of through noisy ray matching.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .camera import Camera
from .gates import GATE_WIDTH_M
from .model import IN_H, IN_W, decode

PERTURBATIONS = ('identity', 'grayscale', 'desat', 'hue60', 'hue180', 'dark', 'gamma2.2', 'blur')
RANGE_BINS = (0.0, 4.5, 7.5, 11.0, 15.0, 21.0, 27.0, 35.0, 60.0)


def perturb(img: np.ndarray, kind: str) -> np.ndarray:
    """img is RGB uint8. Everything here changes only appearance, never where the gate is."""
    import cv2
    if kind == 'identity':
        return img
    if kind == 'grayscale':
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    if kind.startswith('hue'):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(kind[3:]) // 2) % 180          # OpenCV hue is degrees / 2
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if kind == 'desat':
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= 0.25
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if kind == 'dark':
        return np.clip(img.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    if kind.startswith('gamma'):
        return (255.0 * np.power(img / 255.0, float(kind[5:]))).astype(np.uint8)
    if kind == 'blur':
        return cv2.GaussianBlur(img, (5, 5), 1.4)
    raise SystemExit(f'unknown perturbation {kind!r}; known: {", ".join(PERTURBATIONS)}')


def _load(paths: list[Path]) -> dict:
    import cv2
    out = {}
    for p in paths:
        out[p] = cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (IN_W, IN_H),
                            interpolation=cv2.INTER_AREA)
    return out


def _predict(net, imgs: list[np.ndarray], batch: int = 128) -> np.ndarray:
    import torch
    dev = next(net.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            xs = [torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0
                  for im in imgs[i:i + batch]]
            out.append(decode(net(torch.stack(xs).to(dev))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 4))


def _labelled(datasets, want_visible: bool | None = None, limit: int = 0, seed: int = 0):
    items = []
    for d in datasets:
        d = Path(d)
        for lab in json.loads((d / 'labels.json').read_text(encoding='utf-8')):
            if want_visible is None or bool(lab['visible']) == want_visible:
                items.append((d / 'frames' / lab['file'], lab))
    if limit and len(items) > limit:
        idx = np.random.default_rng(seed).permutation(len(items))[:limit]
        items = [items[i] for i in idx]
    return items


def probe(ckpt: str | Path, dataset: str | Path, out: str | Path = '', device: str = 'cuda',
          tiles: int = 9, verbose: bool = True) -> dict:
    """Detection statistics on a dataset with no labels, plus a sheet of what the detector was surest about."""
    import cv2

    from .train import load_gatenet
    net = load_gatenet(ckpt, device)
    frames = sorted((Path(dataset) / 'frames').glob('*.jpg'))
    if not frames:
        raise SystemExit(f'{dataset} has no frames')
    imgs = _load(frames)
    P = _predict(net, [imgs[f] for f in frames])
    p = P[:, 0]
    r = {'frames': len(frames), 'fires': float((p > 0.5).mean()), 'confident': float((p > 0.8).mean()),
         'p_median': float(np.median(p)), 'p90': float(np.quantile(p, 0.9)), 'p_max': float(p.max())}
    if verbose:
        print(f'{Path(ckpt).parent.name} on {Path(dataset).name}: {len(frames)} frames, no labels; '
              f'fires (p>0.5) {100 * r["fires"]:.1f}%, p>0.8 {100 * r["confident"]:.1f}%, '
              f'median p {r["p_median"]:.3f}, p90 {r["p90"]:.3f}, max {r["p_max"]:.3f}')
        print('  compare with the false-positive rate on gate-less frames at home: at or below it, '
              'this is noise, not detection')
    if out:
        ims = []
        for i in sorted(np.argsort(-p)[:tiles]):
            img = cv2.imread(str(frames[i]))
            u, v = (P[i, 1] + 1) / 2 * img.shape[1], (P[i, 2] + 1) / 2 * img.shape[0]
            w = P[i, 3] * img.shape[1] / IN_W
            col = (0, 255, 0) if p[i] > 0.5 else (0, 165, 255)
            cv2.line(img, (int(u - w / 2), int(v)), (int(u + w / 2), int(v)), col, 2)
            cv2.circle(img, (int(u), int(v)), 5, (0, 0, 255), -1)
            cv2.putText(img, f'{frames[i].name} p={p[i]:.2f}', (6, img.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            ims.append(img)
        cols = 3
        n = len(ims) - len(ims) % cols
        if n:
            cv2.imwrite(str(out), np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, n, cols)]))
            if verbose:
                print(f'  the {n} most confident frames -> {out}')
    return r


def colour_stress(ckpt: str | Path, datasets: list[str | Path], device: str = 'cuda', limit: int = 900,
                  kinds: tuple = PERTURBATIONS, verbose: bool = True) -> dict:
    """Recall, centre error and false positives under each perturbation, on labelled (ideally held-out) frames."""
    from .train import load_gatenet
    net = load_gatenet(ckpt, device)
    vis = _labelled(datasets, True, limit)
    neg = _labelled(datasets, False, max(limit // 2, 1))
    imgs = _load([p for p, _ in vis + neg])
    lu = np.array([lab['u'] * IN_W / 640 for _, lab in vis])
    lv = np.array([lab['v'] * IN_H / 360 for _, lab in vis])
    if verbose:
        print(f'{Path(ckpt).parent.name}: {len(vis)} frames with a gate, {len(neg)} without')
        print(f'{"perturbation":14s} {"recall":>7s} {"centre px":>10s} {"false pos":>10s}')
    report = {}
    for kind in kinds:
        Pv = _predict(net, [perturb(imgs[p], kind) for p, _ in vis])
        Pn = _predict(net, [perturb(imgs[p], kind) for p, _ in neg])
        det = Pv[:, 0] > 0.5
        pu, pv = (Pv[:, 1] + 1) / 2 * IN_W, (Pv[:, 2] + 1) / 2 * IN_H
        err = float(np.median(np.hypot(pu[det] - lu[det], pv[det] - lv[det]))) if det.any() else float('nan')
        fp = float((Pn[:, 0] > 0.5).mean()) if len(neg) else float('nan')
        report[kind] = {'recall': float(det.mean()), 'centre_px': err, 'false_pos': fp}
        if verbose:
            print(f'{kind:14s} {100 * det.mean():6.1f}% {err:9.1f} {100 * fp:9.1f}%')
    return report


def fit_range_corr(ckpt: str | Path, datasets: list[str | Path], camera: str | Path,
                   device: str = 'cuda', limit: int = 2500, verbose: bool = True) -> tuple:
    """True range / range implied by the predicted width, per range bin: the pilot's RANGE_CORR for this detector."""
    import yaml

    from .runtime import detection_geometry
    from .train import load_gatenet
    cy = yaml.safe_load(open(camera, encoding='utf-8'))
    cam = Camera(IN_W, IN_H, float(cy['f']) * IN_W / float(cy.get('width', 640)), float(cy['tilt_deg']))
    items = [it for it in _labelled(datasets, True, limit) if np.isfinite(it[1].get('dist_m', np.nan))]
    net = load_gatenet(ckpt, device)
    imgs = _load([p for p, _ in items])
    P = _predict(net, [imgs[p] for p, _ in items])
    true_d = np.array([lab['dist_m'] for _, lab in items])
    det = P[:, 0] > 0.5
    est = np.full(len(items), np.nan)
    for i in np.nonzero(det)[0]:
        est[i] = detection_geometry(cam, (P[i, 1] + 1) / 2 * IN_W, (P[i, 2] + 1) / 2 * IN_H, max(P[i, 3], 4.0))[1]
    if verbose:
        print(f'{Path(ckpt).parent.name}: {int(det.sum())} detections of {len(items)} labelled frames '
              f'(the range assumes a {GATE_WIDTH_M} m gate)')
        print(f'{"range bin":>12s} {"n":>6s} {"factor":>8s} {"quartiles":>12s}')
    table = []
    for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        m = det & (true_d >= lo) & (true_d < hi)
        if m.sum() < 15:
            continue
        ratio = true_d[m] / est[m]
        table.append((round(float(np.median(true_d[m])), 1), round(float(np.median(ratio)), 3)))
        if verbose:
            print(f'{lo:5.1f}-{hi:4.1f} {int(m.sum()):6d} {np.median(ratio):8.3f} '
                  f'{np.quantile(ratio, 0.25):6.2f}-{np.quantile(ratio, 0.75):.2f}')
    table = tuple(table) + ((45.0, 1.0),)
    if verbose:
        print('\nRANGE_CORR = (' + ', '.join(f'({d}, {k})' for d, k in table) + ')')
    return table
