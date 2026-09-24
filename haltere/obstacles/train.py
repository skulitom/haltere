"""ClearanceNet training (offline).

``python -m haltere.obstacles.train --fold F12 --arch dav2s --gpu-budget-min 40 --chunk-min 10
--max-temp 75 --flight-lock PATH [--bench]``

Data: ``FrameStore.rows(fold=..., side='inner_train')`` for fitting and ``side='inner_val'`` for early
stopping and selection (F12 inner validation = the F5 group; ALL = held-back flights). Never touch the
fold's test side or the sealed part. Per-epoch environment shares from ``splits.sampling_weights``
(sqrt balancing, Straw Bale <= 30 %, Drawing Board <= 20 %; when every training asset is capped, as in F12
whose inner-train side is only Straw Bale + Drawing Board, the caps cannot be met and the shares are
renormalised, i.e. plain sqrt balancing); rows flagged DENSE_EXTRA are sampled at the stride-grid rate
(weight 1/3) so event windows are not over-represented. Frames are masked with ``overlays.masked_input`` +
``validity_channel``; grid cells more than half covered by ghost-trail pixels carry no loss.

Augmentation (v0): JPEG re-encode (p 0.3, quality 30-90) before the overlay detectors run; synthetic
masked ring strokes (p 0.3; half of them centred on a labelled obstacle cell <= 8 m) so a masked ring shape
carries no free-space information; on the GPU colour/gamma/brightness/contrast/saturation incl. a low-light
mode (p 0.2), blur (p 0.2), noise (p 0.3) on scene pixels only; horizontal flip (p 0.5) with
``contract.mirror_grid``/``mirror_fan`` label mirroring and gravity x -> -x. Not in v0 (M2): H.264
re-encode and random HUD glyphs. No crops, zooms or rotations (the fan geometry depends on f = 140 px and
the 30 deg tilt). Hindsight index fields (tti_s, event_id, IN_CLEAN/PRE_EVENT flags) are never inputs.

Losses (``laplace_censored_nll`` and ``fan_bce`` below are the numpy references; the torch versions
match them to 1e-5, tests/test_obstacle_model.py):

- grid and fan distance: censored Laplace NLL on ln(metres) with mu = q50 and
  b = max((q50 - q20) / ln 2.5, 0.01) (the Laplace q20 is mu - b ln 2.5). EXACT: ln(2b) + |y - mu|/b;
  LOWER s: -ln P(Y >= ln s); UPPER u: -ln P(Y <= ln u); UNKNOWN: 0.
- fan probabilities: BCE with ``labels.fan_blocked_targets`` for 4 m and 8 m (weight 0 = ignored).
- weight x4 on grid cells whose centre ray is within 30 deg of the heading and whose label value is
  <= 8 m, and on fan directions with |yaw| <= 30 deg. Each term is a weighted mean over its labelled entries.
- teacher shape loss (weight 0.1): scale/shift-invariant L1 between -grid q50 (log) and the teacher
  log-disparity pooled to 18 x 32 (2 x 2 max), after a per-frame least-squares affine alignment of the
  teacher to the (detached) prediction, on valid cells.

Selection (declared before training): the EMA weights are evaluated on a fixed, environment-balanced subset
of the inner-validation rows (<= 400 per environment) every ``--eval-every-s`` GPU seconds; the checkpoint
with the lowest inner-validation objective (grid NLL + fan NLL + BCE4 + BCE8, same weights, no augmentation,
no teacher term) becomes model.pt. Held-out (test-side) rows are never read here.

Outputs in ``runs/obstacle-train/<FOLD>-v<k>/``: ckpt_<chunk>.pt (model + EMA + optimiser + progress, every
chunk, resumable; the two newest are kept), best_ema.pt, train_log.jsonl (per evaluation and chunk: losses,
inner-val metrics, GPU temperature, flight-lock waits), model.pt + model.json (model.MODEL_CARD_KEYS). GPU
work only in <= 10 min chunks behind ``thermal.ChunkGuard(lock, gpu=True)`` (pause > 75 C until <= 65 C,
hard stop 80 C); inside a chunk the GPU temperature is read every 30 s (>= 80 C: checkpoint and stop;
> --max-temp: end the chunk early) and the flight lock every ~5 s (present: checkpoint and end the chunk, the
next ``before_chunk`` waits). ``thermal.limit_threads(2, torch=True, cv2=True)``; two data threads.
``--bench`` measures step throughput in < 2 min before any budgeted training.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .labels import LabelKind, fan_blocked_targets

LN_2P5 = float(np.log(2.5))
MIN_SCALE = 0.01
NEAR_WEIGHT = 4.0
NEAR_RANGE_M = 8.0
NEAR_YAW_DEG = 30.0
TEACHER_WEIGHT = 0.1
GHOST_CELL_MAX = 0.5
DENSE_EXTRA_WEIGHT = 1.0 / 3.0
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / 'runs' / 'obstacle-train'
DEFAULT_STORE = REPO_ROOT / 'runs' / 'obstacle-store-v1'
MAIN_STORE = Path('C:/DEV/Haltere/runs/obstacle-store-v1')

RECIPES = {
    'dav2s': dict(batch=16, lr_backbone=5e-5, layer_decay=0.8, lr_patch=5e-5, lr_dpt=1e-4, lr_heads=5e-4,
                  weight_decay=0.01, warmup_frac=0.03, final_lr_frac=0.05, ema=0.998, grad_clip=1.0,
                  p_jpeg=0.3, p_ring=0.3, p_flip=0.5, p_lowlight=0.2, p_blur=0.2, p_noise=0.3,
                  val_per_env=400, eval_every_s=240.0, amp='bf16'),
    # run B of v0: same architecture, pretrained DA-V2 backbone + DPT neck/head frozen (only the new heads learn)
    'dav2s_frozen': dict(batch=16, lr_backbone=0.0, layer_decay=0.8, lr_patch=0.0, lr_dpt=0.0, lr_heads=5e-4,
                         weight_decay=0.01, warmup_frac=0.03, final_lr_frac=0.05, ema=0.998, grad_clip=1.0,
                         p_jpeg=0.3, p_ring=0.3, p_flip=0.5, p_lowlight=0.2, p_blur=0.2, p_noise=0.3,
                         val_per_env=400, eval_every_s=120.0, amp='bf16', freeze_pretrained=True),
    'resnet18fpn': dict(batch=16, lr_backbone=1e-3, layer_decay=1.0, lr_patch=1e-3, lr_dpt=1e-3, lr_heads=1e-3,
                        weight_decay=0.01, warmup_frac=0.03, final_lr_frac=0.05, ema=0.995, grad_clip=1.0,
                        p_jpeg=0.3, p_ring=0.3, p_flip=0.5, p_lowlight=0.2, p_blur=0.2, p_noise=0.3,
                        val_per_env=400, eval_every_s=75.0, amp='bf16'),
}
RECIPES_ARCH = {'dav2s_frozen': 'dav2s'}
SELECTION_RULE = ('lowest inner-validation objective (grid NLL + fan NLL + BCE4 + BCE8, near weights, EMA weights, '
                  'no augmentation, no teacher term) on <= 400 rows per inner-validation environment')


# ----------------------------------------------------------------------------- numpy references

def laplace_scale(q20, q50):
    return np.maximum((np.asarray(q50, np.float64) - np.asarray(q20, np.float64)) / LN_2P5, MIN_SCALE)


def laplace_censored_nll(q20_log, q50_log, value_m, kind):
    """Per-element censored Laplace NLL on ln(metres); 0 where UNKNOWN (or value not finite)."""
    mu = np.asarray(q50_log, np.float64)
    b = laplace_scale(q20_log, q50_log)
    v = np.asarray(value_m, np.float64)
    k = np.asarray(kind)
    ok = np.isfinite(v) & (v > 0)
    y = np.log(np.where(ok, v, 1.0))
    z = (y - mu) / b
    exact = np.log(2 * b) + np.abs(z)
    # P(Y >= y) = 1 - 0.5 e^z (z < 0) or 0.5 e^-z (z >= 0)
    lower = np.where(z >= 0, np.log(2.0) + z, -np.log1p(-0.5 * np.exp(np.minimum(z, 0.0))))
    # P(Y <= y) = 0.5 e^z (z < 0) or 1 - 0.5 e^-z (z >= 0)
    upper = np.where(z < 0, np.log(2.0) - z, -np.log1p(-0.5 * np.exp(-np.maximum(z, 0.0))))
    out = np.where(k == LabelKind.EXACT, exact,
                   np.where(k == LabelKind.LOWER, lower, np.where(k == LabelKind.UPPER, upper, 0.0)))
    return np.where(ok & (k != LabelKind.UNKNOWN), out, 0.0)


def fan_bce(prob, value_m, kind, within_m):
    """Per-element BCE of P(blocked <= within_m) against labels.fan_blocked_targets (0 where ignored)."""
    t, w = fan_blocked_targets(value_m, kind, within_m)
    p = np.clip(np.asarray(prob, np.float64), 1e-7, 1 - 1e-7)
    return w * -(t * np.log(p) + (1 - t) * np.log(1 - p))


# ----------------------------------------------------------------------------- torch losses

def laplace_censored_nll_torch(q20_log, q50_log, value_m, kind):
    """Torch twin of ``laplace_censored_nll`` (value_m NaN where unknown; kind uint8/int tensor)."""
    import torch
    mu = q50_log
    b = ((q50_log - q20_log) / LN_2P5).clamp_min(MIN_SCALE)
    ok = torch.isfinite(value_m) & (value_m > 0) & (kind != int(LabelKind.UNKNOWN))
    y = torch.log(torch.where(ok, value_m, torch.ones_like(value_m)))
    z = (y - mu) / b
    ln2 = math.log(2.0)
    exact = torch.log(2 * b) + z.abs()
    lower = torch.where(z >= 0, ln2 + z, -torch.log1p(-0.5 * torch.exp(z.clamp_max(0.0))))
    upper = torch.where(z < 0, ln2 - z, -torch.log1p(-0.5 * torch.exp(-z.clamp_min(0.0))))
    k = kind.long()
    out = torch.where(k == int(LabelKind.EXACT), exact,
                      torch.where(k == int(LabelKind.LOWER), lower,
                                  torch.where(k == int(LabelKind.UPPER), upper, torch.zeros_like(z))))
    return torch.where(ok, out, torch.zeros_like(out))


def fan_bce_torch(logit, target, weight):
    """Per-element BCE from logits with fan_blocked_targets (target, weight); equals ``fan_bce`` on sigmoid(logit)."""
    import torch.nn.functional as F
    return weight * F.binary_cross_entropy_with_logits(logit, target, reduction='none')


def teacher_shape_loss(q50_log, teacher, valid=None):
    """Scale/shift-invariant L1 between -q50 (B, 18, 32) and log teacher disparity (B, 36, 64 -> 2x2 max).

    The teacher is aligned to the detached prediction per frame (least squares a * t + b); returns the mean
    |(-q50) - (a t + b)| over valid cells of frames with >= 16 valid cells (0 when none).
    """
    import torch
    import torch.nn.functional as F
    t = teacher.float()
    t = torch.where(torch.isfinite(t), t, torch.full_like(t, -1.0))
    t = F.max_pool2d(t[:, None], 2)[:, 0]
    ok = t > 0
    if valid is not None:
        ok = ok & valid
    lt = torch.log(t.clamp_min(1e-3))
    p = -q50_log.float()
    w = ok.float()
    n = w.sum((1, 2))
    use = n >= 16
    if not bool(use.any()):
        return q50_log.sum() * 0.0
    nn_ = n.clamp_min(1.0)[:, None, None]
    pd = p.detach()
    mt = (lt * w).sum((1, 2), keepdim=True) / nn_
    mp = (pd * w).sum((1, 2), keepdim=True) / nn_
    cov = ((lt - mt) * (pd - mp) * w).sum((1, 2), keepdim=True)
    var = ((lt - mt) ** 2 * w).sum((1, 2), keepdim=True).clamp_min(1e-6)
    a = cov / var
    tgt = a * (lt - mt) + mp
    err = ((p - tgt).abs() * w).sum((1, 2)) / n.clamp_min(1.0)
    return err[use].mean()


# ----------------------------------------------------------------------------- data

def _cv2():
    import cv2
    return cv2


class TrainData:
    """Store rows + labels + per-row sampling weights for one fold side (training or inner validation)."""

    def __init__(self, store, labels, rows, *, teacher=True):
        from . import contract
        from .store import Flag
        self.store, self.labels = store, labels
        self.rows = np.asarray(rows, np.int64)
        idx = store.index
        self.env = np.asarray(idx['env'])[self.rows]
        self.flags = np.asarray(idx['flags'])[self.rows]
        quat = np.asarray(idx['quat'])[self.rows].astype(np.float64)
        self.grav = contract.gravity_camera(quat).astype(np.float32)
        self.dense = (self.flags & int(Flag.DENSE_EXTRA)) > 0
        self.teacher = teacher
        self._teacher_arr = None
        if teacher:
            path = labels.root / 'teacher' / 'disparity.npy'
            tm = json.loads((path.parent / 'manifest.json').read_text(encoding='utf-8'))
            if tm.get('status') != 'complete' or tm.get('store_index_sha256') != labels.manifest['store_index_sha256']:
                raise ValueError('teacher cache incomplete or for another store index')
            self._teacher_arr = np.load(path, mmap_mode='r')

    def env_counts(self) -> dict:
        from .splits import ENV_BY_CODE
        return {ENV_BY_CODE[int(e)].name: int((self.env == e).sum()) for e in np.unique(self.env)}

    def sampling_probs(self) -> tuple[np.ndarray, dict]:
        from .splits import ENV_BY_CODE, sampling_weights
        rw = np.where(self.dense, DENSE_EXTRA_WEIGHT, 1.0)
        counts = self.env_counts()
        shares = sampling_weights(counts)
        tot = sum(shares.values())
        shares = {k: v / tot for k, v in shares.items()}
        p = np.zeros(len(self.rows))
        for code in np.unique(self.env):
            name = ENV_BY_CODE[int(code)].name
            m = self.env == code
            p[m] = shares[name] * rw[m] / rw[m].sum()
        return p / p.sum(), shares

    def prepare(self, positions, rng: np.random.Generator | None, recipe: dict):
        """Batch dict of numpy arrays for positions into self.rows (augment when rng is given)."""
        from . import overlays
        from .model import cell_heading_angle_deg
        cv2 = _cv2()
        positions = np.asarray(positions, np.int64)
        rows = self.rows[positions]
        n = len(rows)
        img = np.empty((n, 252, 448, 3), np.uint8)
        valid = np.empty((n, 252, 448), np.uint8)
        ghost = np.empty((n, 18, 32), np.float32)
        gv, gk, _ = self.labels.grid(rows)
        fv, fk, _ = self.labels.fan(rows)
        for i, r in enumerate(rows):
            f = np.array(self.store.frame(int(r)))
            if rng is not None and rng.random() < recipe['p_jpeg']:
                q = int(rng.integers(30, 91))
                ok, enc = cv2.imencode('.jpg', f[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok:
                    f = np.ascontiguousarray(cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1])
            mk = overlays.overlay_masks(f)
            masked = mk.masked
            if rng is not None and rng.random() < recipe['p_ring']:
                masked = masked | self._ring_hole(rng, gv[i], gk[i])
            out = f.copy()
            out[masked] = overlays.IMAGENET_MEAN_U8
            img[i] = out
            v = np.full((252, 448), 2, np.uint8)
            v[mk.propeller] = 1
            v[masked] = 0
            valid[i] = v
            ghost[i] = mk.ghost.reshape(18, 14, 32, 14).mean(axis=(1, 3))
        batch = dict(img=img, valid=valid, ghost=ghost, grav=self.grav[positions],
                     angle=cell_heading_angle_deg(self.grav[positions]).astype(np.float32),
                     gv=gv, gk=gk, fv=fv, fk=fk, env=self.env[positions], rows=rows)
        for d, key in ((4.0, 'b4'), (8.0, 'b8')):
            t, w = fan_blocked_targets(fv, fk, d)
            batch[key + 't'], batch[key + 'w'] = t, w
        if self.teacher:
            batch['teacher'] = self._teacher_arr[rows].astype(np.float32)
        return batch

    @staticmethod
    def _ring_hole(rng, gv, gk):
        """A synthetic masked ring stroke (the shape a masked checkpoint ring leaves), half on a near obstacle."""
        cv2 = _cv2()
        hole = np.zeros((252, 448), np.uint8)
        near = np.argwhere(((gk == LabelKind.EXACT) | (gk == LabelKind.UPPER)) & np.isfinite(gv) & (gv <= 8.0))
        if len(near) and rng.random() < 0.5:
            r, c = near[rng.integers(len(near))]
            cx, cy = c * 14 + 7 + rng.normal(0, 5), r * 14 + 7 + rng.normal(0, 5)
        else:
            cx, cy = rng.uniform(20, 428), rng.uniform(20, 232)
        a = rng.uniform(8, 60)
        b = a * rng.uniform(0.6, 1.0)
        cv2.ellipse(hole, (int(cx), int(cy)), (int(a), int(b)), float(rng.uniform(0, 180)), 0, 360, 1,
                    int(rng.integers(2, 6)))
        return hole.astype(bool)


class BatchStream:
    """Deterministic, resumable prefetching batch stream: batch k uses rng seeded by (seed, k)."""

    def __init__(self, data: TrainData, probs: np.ndarray, recipe: dict, seed: int, start_step: int,
                 threads: int = 2, depth: int = 4):
        self.data, self.probs, self.recipe, self.seed = data, probs, recipe, seed
        self.next_step = start_step
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.queue = deque()
        self.depth = depth
        self._fill()

    def _make(self, step: int):
        rng = np.random.default_rng([self.seed, step])
        pos = rng.choice(len(self.probs), size=self.recipe['batch'], p=self.probs)
        return self.data.prepare(np.sort(pos), rng, self.recipe)

    def _fill(self):
        while len(self.queue) < self.depth:
            self.queue.append(self.pool.submit(self._make, self.next_step))
            self.next_step += 1

    def get(self):
        fut = self.queue.popleft()
        b = fut.result()
        self._fill()
        return b

    def close(self):
        for f in self.queue:
            f.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)


def val_positions(data: TrainData, per_env: int, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for code in np.unique(data.env):
        idx = np.flatnonzero(data.env == code)
        out.append(rng.choice(idx, min(per_env, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


# ----------------------------------------------------------------------------- GPU side

class Trainer:
    def __init__(self, net, recipe: dict, device, *, arch: str):
        import torch
        self.torch = torch
        self.net = net.to(device)
        self.device = device
        self.recipe = recipe
        self.arch = arch
        if recipe.get('freeze_pretrained'):
            if arch != 'dav2s':
                raise ValueError('freeze_pretrained applies to the dav2s arch')
            for p in self.net.da.parameters():
                p.requires_grad_(False)
        self.ema = self._ema_copy()
        self.opt = torch.optim.AdamW(self._param_groups(), lr=1.0, betas=(0.9, 0.999))
        for g in self.opt.param_groups:
            g['base_lr'] = g['lr']
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        yaw = np.deg2rad(np.array([-40, -30, -20, -10, 0, 10, 20, 30, 40], np.float32))
        near_yaw = np.abs(np.rad2deg(yaw)) <= NEAR_YAW_DEG + 1e-6
        self.fan_near = torch.tensor(np.broadcast_to(near_yaw, (4, 9)).astype(np.float32), device=device)

    def _ema_copy(self):
        import copy
        e = copy.deepcopy(self.net).eval()
        for p in e.parameters():
            p.requires_grad_(False)
        return e

    def _param_groups(self):
        r = self.recipe
        groups = []
        net = self.net
        seen = set()

        def add(params, lr, name):
            params = [p for p in params if p.requires_grad and id(p) not in seen]
            if lr <= 0:
                return
            if not params:
                return
            for p in params:
                seen.add(id(p))
            decay = [p for p in params if p.ndim > 1]
            no_decay = [p for p in params if p.ndim <= 1]
            if decay:
                groups.append(dict(params=decay, lr=lr, weight_decay=r['weight_decay'], name=name))
            if no_decay:
                groups.append(dict(params=no_decay, lr=lr, weight_decay=0.0, name=name + '_nd'))

        if self.arch == 'dav2s':
            add([net.da.backbone.embeddings.patch_embeddings.projection.weight,
                 net.da.backbone.embeddings.patch_embeddings.projection.bias], r['lr_patch'], 'patch')
            n_layers = len(net.da.backbone.encoder.layer)
            for name, params, depth in net.layer_groups():
                if name == 'dpt':
                    add(params, r['lr_dpt'], name)
                elif name == 'heads':
                    add(params, r['lr_heads'], name)
                else:
                    add(params, r['lr_backbone'] * r['layer_decay'] ** (n_layers - depth), name)
        else:
            for name, params, _ in net.layer_groups():
                add(params, r['lr_heads'] if name == 'heads' else r['lr_backbone'], name)
        return groups

    def set_lr(self, progress: float):
        r = self.recipe
        if progress < r['warmup_frac']:
            f = max(progress / r['warmup_frac'], 0.01)
        else:
            x = min((progress - r['warmup_frac']) / max(1 - r['warmup_frac'], 1e-6), 1.0)
            f = r['final_lr_frac'] + (1 - r['final_lr_frac']) * 0.5 * (1 + math.cos(math.pi * x))
        for g in self.opt.param_groups:
            g['lr'] = g['base_lr'] * f
        return f

    def to_device(self, b: dict, *, augment: bool, step: int, seed: int):
        """Tensors on the GPU; GPU augmentation and horizontal flip when ``augment``."""
        torch = self.torch
        dev = self.device
        img = torch.from_numpy(b['img']).to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        valid = torch.from_numpy(b['valid']).to(dev).float() * 0.5
        t = {k: torch.from_numpy(np.ascontiguousarray(b[k])).to(dev)
             for k in ('grav', 'angle', 'ghost', 'b4t', 'b4w', 'b8t', 'b8w')}
        t['gv'] = torch.from_numpy(b['gv']).to(dev).float()
        t['gk'] = torch.from_numpy(b['gk']).to(dev)
        t['fv'] = torch.from_numpy(b['fv']).to(dev).float()
        t['fk'] = torch.from_numpy(b['fk']).to(dev)
        if 'teacher' in b:
            t['teacher'] = torch.from_numpy(b['teacher']).to(dev)
        if augment:
            gen = torch.Generator(device=dev)
            gen.manual_seed(int(seed) * 1_000_003 + int(step))
            img = self._augment(img, valid > 0, gen)
            flip = torch.rand(img.shape[0], generator=gen, device=dev) < self.recipe['p_flip']
            if bool(flip.any()):
                fi = flip.view(-1, 1, 1, 1)
                img = torch.where(fi, img.flip(-1), img)
                valid = torch.where(flip.view(-1, 1, 1), valid.flip(-1), valid)
                for k in ('angle', 'ghost', 'gv', 'gk', 'fv', 'fk', 'b4t', 'b4w', 'b8t', 'b8w', 'teacher'):
                    if k in t:
                        t[k] = torch.where(flip.view(-1, *([1] * (t[k].ndim - 1))), t[k].flip(-1), t[k])
                g = t['grav'].clone()
                g[flip, 0] = -g[flip, 0]
                t['grav'] = g
        x = (img - self.mean) / self.std
        x = torch.where(valid[:, None] > 0, x, torch.zeros_like(x))
        t['x'] = torch.cat([x, valid[:, None]], 1)
        return t

    def _augment(self, img, keep, gen):
        torch = self.torch
        F = torch.nn.functional
        r = self.recipe
        B = img.shape[0]
        dev = img.device

        def U(lo, hi):
            return lo + (hi - lo) * torch.rand(B, 1, 1, 1, generator=gen, device=dev)

        def P(p):
            return torch.rand(B, 1, 1, 1, generator=gen, device=dev) < p
        low = P(r['p_lowlight'])
        bright = torch.where(low, U(0.15, 0.5), U(0.75, 1.25))
        gamma = torch.where(low, U(1.0, 1.6), U(0.75, 1.3))
        contrast, sat = U(0.75, 1.25), U(0.6, 1.3)
        x = img.clamp(1e-4, 1.0) ** gamma
        k3 = keep[:, None].float()
        m = (x * k3).sum((1, 2, 3), keepdim=True) / (3 * k3.sum((1, 2, 3), keepdim=True)).clamp_min(1.0)
        x = (x - m) * contrast + m
        grey = x.mean(1, keepdim=True)
        x = ((x - grey) * sat + grey) * bright
        blur = P(r['p_blur'])
        if bool(blur.any()):
            s = 1.0
            ax = torch.arange(-2, 3, device=dev, dtype=x.dtype)
            k1 = torch.exp(-ax ** 2 / (2 * s * s))
            k1 = k1 / k1.sum()
            k2 = (k1[:, None] * k1[None, :]).expand(3, 1, 5, 5)
            xb = F.conv2d(F.pad(x, (2, 2, 2, 2), mode='replicate'), k2, groups=3)
            x = torch.where(blur, xb, x)
        noise = P(r['p_noise'])
        if bool(noise.any()):
            x = torch.where(noise, x + torch.randn(x.shape, generator=gen, device=dev) * U(0.005, 0.03), x)
        return x.clamp(0.0, 1.0)

    def losses(self, out, t, *, teacher: bool):
        torch = self.torch
        g = out['grid_log_range'].float()
        f = out['fan_log_free'].float()
        lg = out['fan_logit'].float()
        gk, gv = t['gk'], t['gv']
        known = (gk != int(LabelKind.UNKNOWN)).float()
        near = ((t['angle'] <= NEAR_YAW_DEG) & torch.isfinite(gv) & (gv <= NEAR_RANGE_M)).float()
        wg = known * (1.0 + (NEAR_WEIGHT - 1.0) * near) * (t['ghost'] <= GHOST_CELL_MAX).float()
        nll_g = laplace_censored_nll_torch(g[:, 0], g[:, 1], gv, gk)
        L = {}
        L['grid_nll'] = (nll_g * wg).sum() / wg.sum().clamp_min(1.0)
        fk = t['fk']
        wf_base = 1.0 + (NEAR_WEIGHT - 1.0) * self.fan_near[None]
        wf = (fk != int(LabelKind.UNKNOWN)).float() * wf_base
        nll_f = laplace_censored_nll_torch(f[:, 0], f[:, 1], t['fv'], fk)
        L['fan_nll'] = (nll_f * wf).sum() / wf.sum().clamp_min(1.0)
        for j, key in enumerate(('b4', 'b8')):
            w = t[key + 'w'] * wf_base
            L['bce' + key[1:]] = (fan_bce_torch(lg[:, j], t[key + 't'], w)).sum() / w.sum().clamp_min(1.0)
        L['objective'] = L['grid_nll'] + L['fan_nll'] + L['bce4'] + L['bce8']
        if teacher and 'teacher' in t:
            L['teacher'] = teacher_shape_loss(g[:, 1], t['teacher'])
            L['total'] = L['objective'] + TEACHER_WEIGHT * L['teacher']
        else:
            L['total'] = L['objective']
        return L

    def autocast(self):
        torch = self.torch
        if self.recipe.get('amp') == 'bf16':
            return torch.autocast('cuda', dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def step(self, t):
        torch = self.torch
        self.net.train()
        if self.recipe.get('freeze_pretrained'):
            self.net.da.eval()
        with self.autocast():
            out = self.net(t['x'], t['grav'])
        L = self.losses(out, t, teacher=True)
        self.opt.zero_grad(set_to_none=True)
        L['total'].backward()
        gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.recipe['grad_clip'])
        self.opt.step()
        return {k: float(v.detach()) for k, v in L.items()}, float(gn)

    def update_ema(self, step: int):
        torch = self.torch
        d = min(self.recipe['ema'], (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            for pe, p in zip(self.ema.parameters(), self.net.parameters()):
                pe.mul_(d).add_(p.detach(), alpha=1 - d)
            for be, b in zip(self.ema.buffers(), self.net.buffers()):
                be.copy_(b)

    def evaluate(self, batches, net=None) -> dict:
        """Inner-validation objective and diagnostics (EMA weights by default, no augmentation)."""
        torch = self.torch
        from .splits import ENV_BY_CODE
        net = net or self.ema
        net.eval()
        sums, n = {}, 0
        per_env: dict = {}
        ratios = {'2-4': [], '4-7': []}
        p8, y8, w8 = [], [], []
        with torch.no_grad():
            for b in batches:
                t = self.to_device(b, augment=False, step=0, seed=0)
                with self.autocast():
                    out = net(t['x'], t['grav'])
                # per-environment objective (frame groups)
                for code in np.unique(b['env']):
                    sel = torch.from_numpy(b['env'] == code).to(self.device)
                    ts = {k: v[sel] for k, v in t.items()}
                    os_ = {k: v[sel] for k, v in out.items()}
                    Ls = self.losses(os_, ts, teacher=False)
                    e = per_env.setdefault(ENV_BY_CODE[int(code)].name, {'objective': 0.0, 'n': 0})
                    e['objective'] += float(Ls['objective']) * int(sel.sum())
                    e['n'] += int(sel.sum())
                L = self.losses(out, t, teacher=False)
                bs = len(b['rows'])
                for k, v in L.items():
                    sums[k] = sums.get(k, 0.0) + float(v) * bs
                n += bs
                q50 = torch.exp(out['grid_log_range'][:, 1].float()).cpu().numpy()
                ex = (b['gk'] == LabelKind.EXACT) & np.isfinite(b['gv'])
                gvv = b['gv'].astype(np.float64)
                for lo, hi, key in ((2, 4, '2-4'), (4, 7, '4-7')):
                    m = ex & (gvv >= lo) & (gvv < hi)
                    ratios[key].append(q50[m] / gvv[m])
                p = torch.sigmoid(out['fan_logit'][:, 1].float()).cpu().numpy()
                p8.append(p.ravel())
                y8.append(b['b8t'].ravel())
                w8.append(b['b8w'].ravel())
        res = {k: v / max(n, 1) for k, v in sums.items()}
        res['per_env_objective'] = {k: v['objective'] / max(v['n'], 1) for k, v in per_env.items()}
        for key, arr in ratios.items():
            a = np.concatenate(arr) if arr else np.zeros(0)
            res[f'ratio_{key}_median'] = float(np.median(a)) if len(a) else None
            res[f'ratio_{key}_n'] = int(len(a))
        p8, y8, w8 = np.concatenate(p8), np.concatenate(y8), np.concatenate(w8)
        m = w8 > 0
        res['fan_p8_auroc'] = _auroc(p8[m], y8[m])
        res['fan_p8_ece'] = _ece(p8[m], y8[m])
        res['fan_p8_n'] = int(m.sum())
        res['n_frames'] = n
        return res


def _auroc(score, y) -> float | None:
    y = np.asarray(y) > 0.5
    if y.all() or (~y).all():
        return None
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s = np.asarray(score)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    npos = y.sum()
    nneg = len(y) - npos
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _ece(p, y, bins: int = 10) -> float | None:
    if len(p) == 0:
        return None
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - (np.asarray(y)[m] > 0.5).mean())
    return float(e)


# ----------------------------------------------------------------------------- orchestration

def _code_commit() -> str | None:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO_ROOT,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _log(out_dir: Path, rec: dict):
    rec = dict(rec, time=_dt.datetime.now().isoformat(timespec='seconds'))
    with open(out_dir / 'train_log.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, default=float) + '\n')


def _store_root(args) -> Path:
    if args.store:
        return Path(args.store)
    return DEFAULT_STORE if (DEFAULT_STORE / 'manifest.json').exists() else MAIN_STORE


def _open(args):
    from .labels import LabelSet
    from .store import FrameStore
    store = FrameStore(_store_root(args))
    labels = LabelSet(store.root, store.manifest['index_sha256'])
    return store, labels


def _checkpoints(out_dir: Path):
    return sorted(out_dir.glob('ckpt_*.pt'))


def train(args):
    import torch
    from . import thermal
    from .model import build_model
    from .splits import FOLDS
    thermal.limit_threads(2, torch=True, cv2=True)
    lock = thermal.require_flight_lock_path(args.flight_lock)
    arch = args.arch
    recipe_name = args.recipe_name or arch
    if RECIPES_ARCH.get(recipe_name, recipe_name) != arch:
        raise SystemExit(f'recipe {recipe_name} is for arch {RECIPES_ARCH.get(recipe_name, recipe_name)}')
    recipe = dict(RECIPES[recipe_name], name=recipe_name)
    if args.batch:
        recipe['batch'] = int(args.batch)
    if args.eval_every_s:
        recipe['eval_every_s'] = float(args.eval_every_s)
    out_dir = Path(args.out) if args.out else DEFAULT_OUT / f'{args.fold}-v0'
    out_dir.mkdir(parents=True, exist_ok=True)
    store, labels = _open(args)
    if args.fold == 'ALL':
        raise SystemExit('ALL training is M7')
    fold = FOLDS[args.fold]
    tr = TrainData(store, labels, store.rows(fold=args.fold, side='inner_train'))
    va = TrainData(store, labels, store.rows(fold=args.fold, side='inner_val'))
    test_envs = set(fold.test)
    for d in (tr, va):
        if set(d.env_counts()) & test_envs:
            raise RuntimeError('held-out environments in the training/selection rows')
    probs, shares = tr.sampling_probs()
    vpos = val_positions(va, recipe['val_per_env'])
    budget_s = float(args.gpu_budget_min) * 60.0
    chunk_s = min(float(args.chunk_min) * 60.0, thermal.GPU_CHUNK_MAX_S)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    guard = thermal.ChunkGuard(lock, gpu=True, pause_c=args.max_temp, chunk_max_s=chunk_s)

    if args.bench:
        return bench(args, tr, va, probs, recipe, guard, dev)
    if args.finalize_only:
        return finalize(args, out_dir, store, labels, tr, va, recipe, shares)

    state = None
    ck = _checkpoints(out_dir)
    if ck:
        state = torch.load(ck[-1], map_location='cpu', weights_only=False)
        print(f'resuming from {ck[-1].name}: step {state["step"]}, gpu {state["gpu_s"]:.0f} s', flush=True)
        if state['arch'] != arch or state['fold'] != args.fold:
            raise SystemExit('checkpoint belongs to another arch/fold')
    else:
        meta = dict(event='start', arch=arch, fold=args.fold, recipe=recipe, selection=SELECTION_RULE,
                    train_rows=len(tr.rows), train_env_counts=tr.env_counts(), sampling_shares=shares,
                    val_rows=len(vpos), val_env_counts=va.env_counts(), budget_min=args.gpu_budget_min,
                    chunk_min=chunk_s / 60, seed=args.seed, code_commit=_code_commit(),
                    store=str(store.root), store_index_sha256=store.manifest['index_sha256'])
        _log(out_dir, meta)
        (out_dir / 'selection_rule.json').write_text(json.dumps(meta, indent=1, default=float) + '\n',
                                                     encoding='utf-8')
    net = build_model(arch, pretrained=(state is None and arch == 'dav2s'))
    trainer = Trainer(net, recipe, dev, arch=arch)
    step, gpu_s, chunk, last_eval_s = 0, 0.0, 0, -1e9
    best = dict(objective=float('inf'), step=None, gpu_s=None)
    max_temp_seen = 0.0
    if state is not None:
        trainer.net.load_state_dict(state['model'])
        trainer.ema.load_state_dict(state['ema'])
        trainer.opt.load_state_dict(state['opt'])
        step, gpu_s, chunk = state['step'], state['gpu_s'], state['chunk']
        last_eval_s, best = state['last_eval_s'], state['best']
        max_temp_seen = state.get('max_temp_seen', 0.0)
    if gpu_s >= budget_s:
        print('budget already used: finalising', flush=True)
        return finalize(args, out_dir, store, labels, tr, va, recipe, shares)

    def val_batches():
        bs = recipe['batch']
        pool = ThreadPoolExecutor(max_workers=2)
        futs = [pool.submit(va.prepare, vpos[a:a + bs], None, recipe) for a in range(0, len(vpos), bs)]
        for f in futs:
            yield f.result()
        pool.shutdown()

    chunks_run = 0
    stop_reason = None
    while gpu_s < budget_s and (args.max_chunks is None or chunks_run < args.max_chunks):
        torch.cuda.empty_cache()
        guard.before_chunk()
        t_chunk = time.monotonic()
        stream = BatchStream(tr, probs, recipe, args.seed, step)
        chunk += 1
        chunks_run += 1
        run = {'loss': [], 'grad': [], 'data_s': 0.0, 'steps': 0}
        last_temp_check = time.monotonic()
        last_lock_check = time.monotonic()
        end_reason = 'chunk_time'
        while True:
            now = time.monotonic()
            elapsed = gpu_s + (now - t_chunk)
            if elapsed >= budget_s:
                end_reason = 'budget'
                break
            if now - t_chunk >= chunk_s - 15:
                break
            if now - last_lock_check >= 5.0:
                last_lock_check = now
                if lock is not None and Path(lock).exists():
                    end_reason = 'flight_lock'
                    break
            if now - last_temp_check >= 30.0:
                last_temp_check = now
                tc = thermal.gpu_temperature()
                if tc is not None:
                    max_temp_seen = max(max_temp_seen, tc)
                    if tc >= thermal.GPU_HARD_STOP_C:
                        end_reason = 'hard_stop'
                        break
                    if tc > args.max_temp:
                        end_reason = 'hot'
                        break
            if elapsed - last_eval_s >= recipe['eval_every_s'] and step > 0:
                ev = trainer.evaluate(val_batches())
                last_eval_s = gpu_s + (time.monotonic() - t_chunk)
                improved = ev['objective'] < best['objective']
                if improved:
                    best = dict(objective=ev['objective'], step=step, gpu_s=last_eval_s)
                    torch.save(trainer.ema.state_dict(), out_dir / 'best_ema.pt')
                tc = thermal.gpu_temperature()
                max_temp_seen = max(max_temp_seen, tc or 0.0)
                rec = dict(event='eval', step=step, gpu_s=round(last_eval_s, 1), temp_c=tc, improved=improved,
                           train_loss=float(np.mean(run['loss'][-200:])) if run['loss'] else None, **ev)
                _log(out_dir, rec)
                print(f"eval step {step} gpu {last_eval_s / 60:.1f} min: obj {ev['objective']:.4f} "
                      f"grid {ev['grid_nll']:.3f} fan {ev['fan_nll']:.3f} bce4 {ev['bce4']:.3f} bce8 {ev['bce8']:.3f} "
                      f"r24 {ev['ratio_2-4_median']} r47 {ev['ratio_4-7_median']} auroc8 {ev['fan_p8_auroc']} "
                      f"T {tc} {'*' if improved else ''}", flush=True)
                continue
            td = time.monotonic()
            b = stream.get()
            run['data_s'] += time.monotonic() - td
            progress = (gpu_s + (time.monotonic() - t_chunk)) / budget_s
            trainer.set_lr(progress)
            t = trainer.to_device(b, augment=True, step=step, seed=args.seed)
            L, gn = trainer.step(t)
            if not math.isfinite(L['total']):
                stream.close()
                raise FloatingPointError(f'non-finite loss at step {step}: {L}')
            trainer.update_ema(step)
            run['loss'].append(L['total'])
            run['grad'].append(gn)
            run['steps'] += 1
            step += 1
            if step % 200 == 0:
                print(f"step {step} gpu {(gpu_s + time.monotonic() - t_chunk) / 60:.1f} min loss "
                      f"{np.mean(run['loss'][-200:]):.4f} (grid {L['grid_nll']:.3f} fan {L['fan_nll']:.3f} "
                      f"b4 {L['bce4']:.3f} b8 {L['bce8']:.3f} teach {L.get('teacher', 0):.3f}) "
                      f"data_wait {run['data_s']:.0f} s", flush=True)
        stream.close()
        chunk_wall = time.monotonic() - t_chunk
        gpu_s += chunk_wall
        tc = thermal.gpu_temperature()
        max_temp_seen = max(max_temp_seen, tc or 0.0)
        ckpt = dict(model=trainer.net.state_dict(), ema=trainer.ema.state_dict(), opt=trainer.opt.state_dict(),
                    step=step, gpu_s=gpu_s, chunk=chunk, last_eval_s=last_eval_s, best=best, arch=arch,
                    fold=args.fold, recipe=recipe, max_temp_seen=max_temp_seen)
        path = out_dir / f'ckpt_{chunk:03d}.pt'
        torch.save(ckpt, str(path) + '.tmp')
        os.replace(str(path) + '.tmp', path)
        for old in _checkpoints(out_dir)[:-2]:
            old.unlink()
        rec = dict(event='chunk', chunk=chunk, end=end_reason, steps=run['steps'], step=step,
                   chunk_s=round(chunk_wall, 1), gpu_s=round(gpu_s, 1), temp_c=tc, max_temp_seen=max_temp_seen,
                   mean_loss=float(np.mean(run['loss'])) if run['loss'] else None,
                   mean_grad=float(np.mean(run['grad'])) if run['grad'] else None,
                   data_wait_s=round(run['data_s'], 1), guard=guard.summary())
        _log(out_dir, rec)
        print(f"chunk {chunk} done ({end_reason}): {run['steps']} steps, {chunk_wall / 60:.1f} min, "
              f"gpu total {gpu_s / 60:.1f}/{budget_s / 60:.0f} min, T {tc} C (max {max_temp_seen})", flush=True)
        if end_reason == 'hard_stop':
            stop_reason = 'hard_stop'
            raise thermal.GpuTooHot(f'GPU reached {thermal.GPU_HARD_STOP_C} C; checkpointed at step {step}')
    if gpu_s >= budget_s:
        if gpu_s - last_eval_s > 30:
            # final evaluation (counted as GPU time)
            guard.before_chunk()
            t0 = time.monotonic()
            ev = trainer.evaluate(val_batches())
            gpu_s += time.monotonic() - t0
            improved = ev['objective'] < best['objective']
            if improved:
                best = dict(objective=ev['objective'], step=step, gpu_s=gpu_s)
                torch.save(trainer.ema.state_dict(), out_dir / 'best_ema.pt')
            _log(out_dir, dict(event='eval', step=step, gpu_s=round(gpu_s, 1), final=True, improved=improved, **ev))
            print(f"final eval step {step}: obj {ev['objective']:.4f} {'*' if improved else ''}", flush=True)
            ckpt = torch.load(_checkpoints(out_dir)[-1], map_location='cpu', weights_only=False)
            ckpt.update(gpu_s=gpu_s, best=best, last_eval_s=gpu_s)
            torch.save(ckpt, _checkpoints(out_dir)[-1])
        return finalize(args, out_dir, store, labels, tr, va, recipe, shares)
    return stop_reason


def finalize(args, out_dir: Path, store, labels, tr, va, recipe, shares):
    """model.pt (the selected EMA weights) + model.json."""
    import torch
    from .model import DAV2S_LICENCE, DAV2S_MODEL_ID, dav2s_weights_dir, model_sha256
    from .splits import FOLDS
    ck = torch.load(_checkpoints(out_dir)[-1], map_location='cpu', weights_only=False)
    recipe = ck.get('recipe', recipe)
    sd = torch.load(out_dir / 'best_ema.pt', map_location='cpu', weights_only=True)
    torch.save(sd, out_dir / 'model.pt')
    sha = model_sha256(out_dir / 'model.pt')
    fold = FOLDS[args.fold]
    wd = dav2s_weights_dir()
    pretrained = (dict(model=DAV2S_MODEL_ID, licence=DAV2S_LICENCE, weights_dir=str(wd),
                       weights_sha256=_sha256_file(wd / 'model.safetensors') if wd else None,
                       readme_licence='apache-2.0 (model/README.md front matter)')
                  if args.arch == 'dav2s' else
                  dict(model='torchvision resnet18 architecture', weights='random initialisation (no ImageNet '
                       'weights on this machine; none downloaded)'))
    samples = ck['step'] * recipe['batch']
    lm = labels.root / 'manifest.json'
    card = dict(schema='haltere.obstacles.model.v1', arch=args.arch, fold=args.fold,
                train_envs=sorted(tr.env_counts()), inner_val_envs=sorted(va.env_counts()),
                held_out_envs=list(fold.test), seen_environment_note='fold model: held-out environments unseen',
                store_index_sha256=store.manifest['index_sha256'], labels_manifest_sha256=_sha256_file(lm),
                recipe=dict(recipe, sampling_shares=shares, selection=SELECTION_RULE, near_weight=NEAR_WEIGHT,
                            teacher_weight=TEACHER_WEIGHT, losses='censored Laplace NLL (grid, fan) + BCE4 + BCE8 '
                            '+ 0.1 teacher shape'),
                epochs=dict(steps=ck['step'], samples=samples, epochs=round(samples / len(tr.rows), 3),
                            selected_step=ck['best']['step'], selected_gpu_s=ck['best']['gpu_s'],
                            selected_inner_val_objective=ck['best']['objective']),
                gpu_minutes=round(ck['gpu_s'] / 60.0, 2), max_temp_c=ck.get('max_temp_seen'),
                code_commit=_code_commit(), model_sha256=sha, pretrained=pretrained,
                inputs='masked RGB + validity + gravity_camera (no speed, route, ring or labels)',
                created=_dt.datetime.now().isoformat(timespec='seconds'))
    (out_dir / 'model.json').write_text(json.dumps(card, indent=1, default=float) + '\n', encoding='utf-8')
    _log(out_dir, dict(event='finalize', model_sha256=sha, selected=ck['best']))
    print(f'model.pt {sha[:12]} selected step {ck["best"]["step"]} (inner-val objective '
          f'{ck["best"]["objective"]:.4f}); gpu {ck["gpu_s"] / 60:.1f} min', flush=True)
    return card


def bench(args, tr, va, probs, recipe, guard, dev):
    """< 2 min: data throughput, training step time and eval forward time (no checkpoint, no budget)."""
    import torch
    from .model import build_model
    guard.before_chunk()
    t_start = time.monotonic()
    res = {}
    t0 = time.monotonic()
    rng = np.random.default_rng(0)
    pos = np.sort(rng.choice(len(probs), recipe['batch'], p=probs))
    tr.prepare(pos, rng, recipe)
    res['prepare_batch_s_1thread'] = round(time.monotonic() - t0, 3)
    net = build_model(args.arch, pretrained=(args.arch == 'dav2s'))
    trainer = Trainer(net, recipe, dev, arch=args.arch)
    if args.arch == 'dav2s':
        # the pretrained head must reproduce the reference runner on a real frame (zero validity weights)
        from .baselines import PretrainedDepth
        ref = PretrainedDepth('relative', fp16=False)
        f = np.array(tr.store.frame(int(tr.rows[0])))[None]
        want = ref.predict(f)[0]
        x = torch.from_numpy(f).to(dev).permute(0, 3, 1, 2).float()
        x = (x / 255.0 - trainer.mean) / trainer.std
        x = torch.cat([x, torch.ones_like(x[:, :1])], 1)
        with torch.no_grad():
            out = trainer.net(x, torch.tensor([[0.0, 1.0, 0.0]], device=dev), aux=True)
        got = out['disparity'][0].float().cpu().numpy()
        res['pretrained_disparity_rel_err'] = float(np.median(np.abs(got - want) / (np.abs(want) + 1e-3)))
        del ref
    stream = BatchStream(tr, probs, recipe, args.seed, 0)
    times, data = [], []
    step = 0
    while time.monotonic() - t_start < 80 and step < 400:
        td = time.monotonic()
        b = stream.get()
        data.append(time.monotonic() - td)
        ts = time.monotonic()
        t = trainer.to_device(b, augment=True, step=step, seed=0)
        L, gn = trainer.step(t)
        trainer.update_ema(step)
        torch.cuda.synchronize()
        times.append(time.monotonic() - ts)
        step += 1
    stream.close()
    w = times[5:] or times
    res.update(steps=step, step_s_median=round(float(np.median(w)), 4), data_wait_s_median=round(float(np.median(data[5:] or data)), 4),
               wall_per_step_s=round(float(np.mean([a + b for a, b in zip(times[5:], data[5:])] or [0])), 4),
               last_loss=L, grad_norm=gn, batch=recipe['batch'],
               max_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    t0 = time.monotonic()
    vb = [va.prepare(val_positions(va, 8)[:recipe['batch']], None, recipe)]
    ev = trainer.evaluate(vb, net=trainer.net)
    res['eval_one_batch_s'] = round(time.monotonic() - t0, 2)
    res['eval_objective_one_batch'] = ev['objective']
    res['total_s'] = round(time.monotonic() - t_start, 1)
    from . import thermal
    res['temp_c'] = thermal.gpu_temperature()
    print(json.dumps(res, indent=1, default=float), flush=True)
    out = Path(args.out) if args.out else DEFAULT_OUT / f'{args.fold}-v0'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'bench_{args.arch}.json').write_text(json.dumps(res, indent=1, default=float) + '\n', encoding='utf-8')
    return res


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m haltere.obstacles.train')
    p.add_argument('--fold', default='F12')
    p.add_argument('--arch', default='dav2s', choices=('dav2s', 'resnet18fpn'))
    p.add_argument('--store', type=Path, default=None)
    p.add_argument('--out', type=Path, default=None, help='default runs/obstacle-train/<FOLD>-v0')
    p.add_argument('--recipe', type=Path, default=None, help='(M2) recipe JSON; v0 uses RECIPES[arch]')
    p.add_argument('--gpu-budget-min', type=float, default=40)
    p.add_argument('--chunk-min', type=float, default=10)
    p.add_argument('--max-temp', type=float, default=75)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--eval-every-s', type=float, default=None)
    p.add_argument('--max-chunks', type=int, default=None, help='run at most this many chunks in this process')
    p.add_argument('--recipe-name', default=None, choices=sorted(RECIPES), help='built-in v0 recipe (default: the arch)')
    p.add_argument('--finalize-only', action='store_true', help='write model.pt/model.json from the run directory')
    p.add_argument('--bench', action='store_true')
    p.add_argument('--flight-lock', default=None)
    args = p.parse_args(argv)
    if args.chunk_min > 10:
        raise SystemExit('GPU chunks are limited to 10 minutes')
    if args.fold not in ('F12', 'F3', 'F4', 'F5', 'ALL'):
        raise SystemExit('unknown fold')
    if args.recipe is not None:
        raise SystemExit('--recipe is an M2 option; v0 uses the built-in recipe')
    return train(args)


if __name__ == '__main__':
    main()
