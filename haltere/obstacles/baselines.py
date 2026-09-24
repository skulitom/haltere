"""Pretrained-depth baselines B2-B4 and model inference for the obstacle harness (offline only).

B2  Depth-Anything-V2 Metric-Indoor-Small (runs/dense-depth-probe-20260923/metric-indoor, revision
    8078d68a, the weights behind metric-indoor-336.ts) run at the store's native 252 x 448 (18 x 32
    patches), fp16, unmasked frames. Output = optical z-depth (m); range = depth x |ray| / z for the
    store camera; grid value = minimum range over each 14 x 14 cell, clipped to [0.3, 60] m.
    (The fixed-shape TorchScript export only accepts 336 x 602, so the HF weights are used directly.)
B3  Relative DA-V2-Small disparity per grid cell (maximum over the cell = its nearest surface) from the
    labels teacher cache (36 x 64 -> 2 x 2 max -> 18 x 32; the teacher is the same pretrained model and
    reads only the frame), else a run of runs/dense-depth-probe-20260923/model at 252 x 448. ONE
    monotone (isotonic, non-increasing) map disparity -> ln range fitted on EXACT label cells of the
    fold's TRAINING environments only.
B4  NON-CAUSAL ceiling: per frame, 1/range = a * disparity + b fitted (least squares, one robust
    refit) on that frame's own EXACT label cells.

Every heavy loop runs in chunks behind a ``thermal.ChunkGuard`` (flight lock; GPU temperature pause
at 75 C, resume at 65 C, hard stop at 80 C) and caches its chunks, so it resumes after a stop.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import contract
from .contract import GRID_H, GRID_W, IMAGE_H, IMAGE_W, PATCH_PX, RANGE_MAX_M, RANGE_MIN_M

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPTH_PROBE = REPO_ROOT / 'runs' / 'dense-depth-probe-20260923'
MAIN_REPO_DEPTH_PROBE = Path('C:/DEV/Haltere/runs/dense-depth-probe-20260923')
WEIGHT_DIRS = {'metric': 'metric-indoor', 'relative': 'model'}
MODEL_IDS = {'metric': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
             'relative': 'depth-anything/Depth-Anything-V2-Small-hf'}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
DISPARITY_SOURCE = ('DA-V2-Small relative disparity: labels teacher cache (36x64) max-pooled 2x2 to 18x32, '
                    'else a 252x448 run max-pooled per 14x14 cell')
CACHE_CHUNK = 512


def _probe_dir() -> Path:
    return DEPTH_PROBE if DEPTH_PROBE.exists() else MAIN_REPO_DEPTH_PROBE


def ray_norm() -> np.ndarray:
    """(252, 448) |ray| / z for every store pixel centre (optical depth -> Euclidean range)."""
    v, u = np.mgrid[:IMAGE_H, :IMAGE_W].astype(np.float64)
    x = (u + 0.5 - IMAGE_W / 2.0) / contract.FOCAL_PX
    y = (v + 0.5 - IMAGE_H / 2.0) / contract.FOCAL_PX
    return np.sqrt(1.0 + x * x + y * y)


def pool_cells(img, how: str = 'min', valid=None) -> np.ndarray:
    """(n, 252, 448) -> (n, 18, 32) min or max over each 14 x 14 cell; cells with no valid pixel are NaN."""
    a = np.asarray(img, np.float64)
    fill = np.inf if how == 'min' else -np.inf
    bad = ~np.isfinite(a)
    if valid is not None:
        bad |= ~np.asarray(valid, bool)
    a = np.where(bad, fill, a)
    blocks = a.reshape(len(a), GRID_H, PATCH_PX, GRID_W, PATCH_PX)
    out = blocks.min(axis=(2, 4)) if how == 'min' else blocks.max(axis=(2, 4))
    return np.where(np.isfinite(out), out, np.nan)


def teacher_to_cells(teacher) -> np.ndarray:
    """Teacher disparity (n, 36, 64) -> (n, 18, 32) cell maximum (the nearest surface of each cell)."""
    t = np.asarray(teacher, np.float64)
    t = np.where(np.isfinite(t), t, -np.inf).reshape(len(t), GRID_H, 2, GRID_W, 2).max(axis=(2, 4))
    return np.where(np.isfinite(t), t, np.nan)


# ----------------------------------------------------------------------------- pretrained runner

class PretrainedDepth:
    """DA-V2-Small (metric-indoor or relative) at 252 x 448 on CUDA, fp16 by default. Frames: uint8 RGB."""

    def __init__(self, kind: str = 'metric', *, weights_dir=None, deps_dir=None, device: str = 'cuda',
                 fp16: bool = True):
        if kind not in WEIGHT_DIRS:
            raise ValueError(f'kind must be one of {tuple(WEIGHT_DIRS)}')
        import torch
        probe = _probe_dir()
        self.kind = kind
        self.weights_dir = Path(weights_dir) if weights_dir else probe / WEIGHT_DIRS[kind]
        deps = Path(deps_dir) if deps_dir else probe / 'dependencies'
        if deps.exists() and str(deps) not in sys.path:
            sys.path.insert(0, str(deps))
        from transformers import AutoModelForDepthEstimation
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        self.fp16 = bool(fp16) and self.device.type == 'cuda'
        model = AutoModelForDepthEstimation.from_pretrained(str(self.weights_dir), local_files_only=True).eval()
        self.model = (model.half() if self.fp16 else model).to(self.device)
        self.weights_sha256 = _sha256_file(self.weights_dir / 'model.safetensors')
        self._mean = torch.tensor(IMAGENET_MEAN * 255.0, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD * 255.0, device=self.device).view(1, 3, 1, 1)

    def config(self) -> dict:
        return dict(kind=self.kind, model=MODEL_IDS[self.kind], weights_sha256=self.weights_sha256,
                    input=[IMAGE_H, IMAGE_W], precision='fp16' if self.fp16 else 'fp32',
                    preprocessing='store frame uint8 RGB (INTER_AREA 448x252), ImageNet mean/std, no resize, '
                                  'no overlay masking',
                    output='metric optical z-depth (m)' if self.kind == 'metric' else 'relative disparity')

    def predict(self, frames) -> np.ndarray:
        """(n, 252, 448, 3) uint8 -> (n, 252, 448) float32 (depth in m, or relative disparity)."""
        import torch
        import torch.nn.functional as F
        frames = np.asarray(frames)
        if frames.dtype != np.uint8 or frames.shape[1:] != (IMAGE_H, IMAGE_W, 3):
            raise ValueError('PretrainedDepth needs uint8 (n, 252, 448, 3) RGB store frames')
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device).permute(0, 3, 1, 2).float()
            x = (x - self._mean) / self._std
            out = self.model(pixel_values=x.half() if self.fp16 else x).predicted_depth
            if out.shape[-2:] != (IMAGE_H, IMAGE_W):
                out = F.interpolate(out[:, None].float(), size=(IMAGE_H, IMAGE_W), mode='bilinear',
                                    align_corners=False)[:, 0]
            return out.float().cpu().numpy()


def run_grid(runner, store, rows, guard, *, batch: int = 16, cache_dir=None, mask_fn=None, log=None,
             reduce: str = 'min_range', chunk: int = CACHE_CHUNK) -> np.ndarray:
    """Run ``runner`` over store rows in cached chunks; returns (n, 18, 32) grids aligned with sorted rows.

    reduce='min_range': metric depth -> range -> cell minimum, clipped to [0.3, 60] m;
    reduce='max_disp': relative disparity -> cell maximum.
    ``mask_fn(frame) -> overlays.OverlayMasks``: HUD, ring stroke and ghost-trail pixels are replaced by the
    ImageNet mean in the input (overlays.masked_input, as the model sees them) and excluded from the cell
    reduction (cells with no scene pixel are NaN). Without it the raw frame is used (the prior study's setting).
    """
    rows = np.unique(np.asarray(rows, np.int64))
    out = np.full((len(rows), GRID_H, GRID_W), np.nan, np.float32)
    norm = ray_norm()
    tag = hashlib.sha256(json.dumps(dict(cfg=runner.config(), reduce=reduce, masked=mask_fn is not None),
                                    sort_keys=True).encode()).hexdigest()[:12]
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for k, a in enumerate(range(0, len(rows), chunk)):
        part = rows[a:a + chunk]
        path = cache_dir / f'{reduce}_{tag}_{int(part[0])}_{int(part[-1])}_{len(part)}.npz' if cache_dir else None
        if path is not None and path.exists():
            z = np.load(path)
            if np.array_equal(z['rows'], part):
                out[a:a + len(part)] = z['grid']
                continue
        if guard is not None:
            guard.before_chunk()
        g = np.empty((len(part), GRID_H, GRID_W), np.float32)
        for b in range(0, len(part), batch):
            rr = part[b:b + batch]
            frames = store.frames(rr)
            valid = None
            if mask_fn is not None:
                from . import overlays
                masks = [mask_fn(f) for f in frames]
                frames = np.stack([overlays.masked_input(f, m) for f, m in zip(frames, masks)])
                valid = np.stack([~m.masked for m in masks])
            pred = runner.predict(frames)
            if reduce == 'min_range':
                cells = pool_cells(pred * norm[None], 'min', valid)
                cells = np.clip(cells, RANGE_MIN_M, RANGE_MAX_M)
            else:
                cells = pool_cells(pred, 'max', valid)
            g[b:b + len(rr)] = cells
        out[a:a + len(part)] = g
        if path is not None:
            np.savez_compressed(path, rows=part, grid=g)
        if log:
            log(f'{reduce} chunk {k}: rows {int(part[0])}..{int(part[-1])} ({a + len(part)}/{len(rows)})')
    return out


def cell_disparity(store, rows, *, labels=None, runner=None, guard=None, batch: int = 16, cache_dir=None,
                   mask_fn=None, log=None) -> np.ndarray:
    """(n, 18, 32) relative disparity per cell (maximum) for sorted unique rows."""
    rows = np.unique(np.asarray(rows, np.int64))
    if labels is not None and runner is None:
        try:
            return teacher_to_cells(labels.teacher(rows)).astype(np.float32)
        except FileNotFoundError:
            if log:
                log('teacher cache missing: running the relative model')
    runner = runner or PretrainedDepth('relative')
    return run_grid(runner, store, rows, guard, batch=batch, cache_dir=cache_dir, mask_fn=mask_fn, log=log,
                    reduce='max_disp')


# ----------------------------------------------------------------------------- B3 calibration

def isotonic_nonincreasing(y, w=None) -> np.ndarray:
    """Pool-adjacent-violators fit of a non-increasing sequence (weighted least squares)."""
    y = np.asarray(y, np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, np.float64)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:          # violation of non-increasing
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            wt = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wt)
            wts.append(wt)
            cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


@dataclass
class MonotoneCalibration:
    """Non-increasing map relative disparity -> ln(range); flat beyond the knots; range clipped to [0.3, 60]."""
    x: np.ndarray
    y: np.ndarray
    fold: str
    n: int = 0
    envs: list = field(default_factory=list)
    source: str = DISPARITY_SOURCE

    def __call__(self, disparity) -> np.ndarray:
        d = np.asarray(disparity, np.float64)
        lr = np.interp(np.nan_to_num(d, nan=0.0), self.x, self.y)
        r = np.clip(np.exp(lr), RANGE_MIN_M, RANGE_MAX_M)
        return np.where(np.isfinite(d), r, np.nan)

    def to_json(self) -> dict:
        return dict(schema='haltere.obstacles.b3_calibration.v1', fold=self.fold, n=self.n, envs=list(self.envs),
                    source=self.source, x=[float(v) for v in self.x], y=[float(v) for v in self.y],
                    rule='isotonic non-increasing ln(range) on disparity, fitted on fold TRAINING environments only')

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_json(cls, obj: dict) -> 'MonotoneCalibration':
        return cls(np.asarray(obj['x'], np.float64), np.asarray(obj['y'], np.float64), obj['fold'], obj.get('n', 0),
                   obj.get('envs', []), obj.get('source', DISPARITY_SOURCE))


def fit_monotone(disparity, true_range, *, fold: str, n_bins: int = 200, envs=()) -> MonotoneCalibration:
    """Fit the B3 map on paired samples (cell disparity, true cell range)."""
    d = np.asarray(disparity, np.float64).ravel()
    r = np.asarray(true_range, np.float64).ravel()
    ok = np.isfinite(d) & np.isfinite(r) & (r > 0)
    d, lr = d[ok], np.log(r[ok])
    if len(d) < 10:
        raise ValueError('too few calibration samples')
    order = np.argsort(d, kind='stable')
    d, lr = d[order], lr[order]
    edges = np.unique(np.quantile(np.arange(len(d)), np.linspace(0, 1, min(n_bins, len(d)) + 1)).astype(int))
    xs, ys, ws = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        xs.append(float(np.median(d[a:b])))
        ys.append(float(np.mean(lr[a:b])))
        ws.append(float(b - a))
    xs, ys, ws = np.asarray(xs), np.asarray(ys), np.asarray(ws)
    yf = isotonic_nonincreasing(ys, ws)
    ux, inv = np.unique(xs, return_inverse=True)
    uy = np.array([np.average(yf[inv == k], weights=ws[inv == k]) for k in range(len(ux))])
    return MonotoneCalibration(ux, uy, fold, int(len(d)), sorted(envs))


def fit_b3_calibration(store, labels, fold: str, *, guard=None, max_frames: int = 20000, seed: int = 0,
                       runner=None, cache_dir=None, out_path=None, log=None) -> MonotoneCalibration:
    """B3 map fitted on EXACT label cells (hindsight/impact/collider) of the fold's TRAINING side only."""
    from .labels import LabelKind
    from .splits import FOLDS
    rows = store.rows(fold=fold, side='train')
    if len(rows) > max_frames:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_frames, replace=False))
    env_codes = np.unique(np.asarray(store.index['env'])[rows])
    from .splits import ENV_BY_CODE
    envs = [ENV_BY_CODE[int(e)].name for e in env_codes]
    if set(envs) & set(FOLDS[fold].test):
        raise RuntimeError('B3 calibration rows include held-out environments')
    gv, gk, _ = labels.grid(rows)
    disp = cell_disparity(store, rows, labels=labels, runner=runner, guard=guard, cache_dir=cache_dir, log=log)
    m = (gk == LabelKind.EXACT) & np.isfinite(gv) & np.isfinite(disp)
    cal = fit_monotone(disp[m], gv[m].astype(np.float64), fold=fold, envs=envs)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(cal.to_json(), indent=1) + '\n', encoding='utf-8')
    if log:
        log(f'B3 calibration {fold}: {cal.n} cells from {len(rows)} training frames ({", ".join(envs)})')
    return cal


# ----------------------------------------------------------------------------- B4 ceiling

def per_frame_affine(disparity, grid_value, grid_kind, *, min_anchors: int = 8) -> np.ndarray:
    """B4: per frame 1/range = a * disparity + b on its EXACT label cells (NON-CAUSAL); NaN without anchors."""
    from .labels import LabelKind
    d = np.asarray(disparity, np.float64)
    gv = np.asarray(grid_value, np.float64)
    gk = np.asarray(grid_kind)
    out = np.full(d.shape, np.nan)
    for i in range(len(d)):
        m = (gk[i] == LabelKind.EXACT) & np.isfinite(gv[i]) & np.isfinite(d[i]) & (gv[i] > 0)
        if m.sum() < min_anchors:
            continue
        x, y = d[i][m], 1.0 / gv[i][m]
        A = np.stack([x, np.ones_like(x)], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        if mad > 0:
            keep = np.abs(res) <= 3 * mad
            if keep.sum() >= min_anchors:
                coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        inv = coef[0] * d[i] + coef[1]
        r = np.where(inv > 1.0 / RANGE_MAX_M, 1.0 / np.maximum(inv, 1e-9), RANGE_MAX_M)
        out[i] = np.where(np.isfinite(d[i]), np.clip(r, RANGE_MIN_M, RANGE_MAX_M), np.nan)
    return out


# ----------------------------------------------------------------------------- model inference

def model_inputs(frames, gravity, masks_fn=None):
    """(n, 4, 252, 448) float32 model input + (n, 3) gravity from uint8 store frames (overlays contract)."""
    from . import overlays
    masks_fn = masks_fn or overlays.overlay_masks
    x = np.empty((len(frames), 4, IMAGE_H, IMAGE_W), np.float32)
    for i, f in enumerate(frames):
        mk = masks_fn(f)
        rgb = overlays.masked_input(f, mk).astype(np.float32) / 255.0
        x[i, :3] = ((rgb - IMAGE_NET_MEAN_F) / IMAGE_NET_STD_F).transpose(2, 0, 1)
        x[i, 3] = overlays.validity_channel(mk)
    return x, np.asarray(gravity, np.float32)


IMAGE_NET_MEAN_F = IMAGENET_MEAN.reshape(1, 1, 3)
IMAGE_NET_STD_F = IMAGENET_STD.reshape(1, 1, 3)


def model_predictions(model_dir, store, rows, guard, *, batch: int = 16, device: str = 'cuda', masks_fn=None,
                      net=None, card=None, log=None):
    """Eager fp32 ClearanceNet outputs on store rows -> evaluate.PredictionSet (kind 'model')."""
    import torch
    from .evaluate import PredictionSet
    from .model import outputs_to_numpy
    model_dir = Path(model_dir)
    if net is None:
        net, card = load_model(model_dir, device)
    if card is None:
        raise ValueError('model_predictions needs the model card with an explicit net')
    net = net.eval().to(device)
    rows = np.unique(np.asarray(rows, np.int64))
    keys = ('grid_q20', 'grid_q50', 'fan_q20', 'fan_q50', 'fan_p4', 'fan_p8')
    acc = {k: [] for k in keys}
    for a in range(0, len(rows), CACHE_CHUNK):
        if guard is not None:
            guard.before_chunk()
        part = rows[a:a + CACHE_CHUNK]
        for b in range(0, len(part), batch):
            rr = part[b:b + batch]
            x, g = model_inputs(store.frames(rr), store.gravity_camera(rr), masks_fn)
            with torch.inference_mode():
                out = net(torch.from_numpy(x).to(device), torch.from_numpy(g).to(device))
            o = outputs_to_numpy(out)
            for k in keys:
                acc[k].append(o[k].astype(np.float32))
        if log:
            log(f'model rows {a + len(part)}/{len(rows)}')
    arrays = {k: np.concatenate(v) if v else np.zeros((0,) + ((18, 32) if 'grid' in k else (4, 9)), np.float32)
              for k, v in acc.items()}
    return PredictionSet(f"{card['fold']}-{model_dir.name}", 'model', True, rows, fold=card['fold'],
                         sha256=card['model_sha256'], arrays=arrays).validate()


def model_predict_fn(net, device: str = 'cuda', masks_fn=None):
    """E8 callable: (frames uint8 (n, 252, 448, 3), quat (n, 4)) -> {'grid_q50': (n, 18, 32)} for a ClearanceNet."""
    import torch
    from .model import outputs_to_numpy
    net = net.eval().to(device)

    def fn(frames, quat):
        x, g = model_inputs(frames, contract.gravity_camera(np.asarray(quat, np.float64)), masks_fn)
        with torch.inference_mode():
            out = net(torch.from_numpy(x).to(device), torch.from_numpy(g).to(device))
        return outputs_to_numpy(out)
    return fn


def depth_predict_fn(runner, mask_fn=None):
    """E8 callable for B2: metric depth -> minimum range per cell (same masking as ``run_grid``)."""
    norm = ray_norm()

    def fn(frames, quat):
        frames = np.asarray(frames)
        valid = None
        if mask_fn is not None:
            from . import overlays
            masks = [mask_fn(f) for f in frames]
            frames = np.stack([overlays.masked_input(f, m) for f, m in zip(frames, masks)])
            valid = np.stack([~m.masked for m in masks])
        d = runner.predict(frames)
        return dict(grid_q50=np.clip(pool_cells(d * norm[None], 'min', valid), RANGE_MIN_M, RANGE_MAX_M))
    return fn


def load_model(model_dir, device: str = 'cuda'):
    """(net, card) of a frozen model directory (model.pt + model.json, sha256 verified)."""
    import torch
    from .model import build_model, read_model_card
    model_dir = Path(model_dir)
    card = read_model_card(model_dir / 'model.pt')
    net = build_model(card['arch'])
    net.load_state_dict(torch.load(model_dir / 'model.pt', map_location='cpu'))
    return net.eval().to(device), card


def overlay_mask_fn():
    """overlays.overlay_masks when the overlays module is delivered, else None."""
    from . import overlays
    try:
        overlays.overlay_masks(np.zeros((IMAGE_H, IMAGE_W, 3), np.uint8))
    except NotImplementedError:
        return None
    return overlays.overlay_masks


def ring_mask_fn():
    """The overlays ring-stroke detector when the overlays module is delivered, else None (E8 ring_removed skipped)."""
    fn = overlay_mask_fn()
    return None if fn is None else (lambda f: fn(f).ring)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
