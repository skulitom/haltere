"""ClearanceNet training (offline). DELIVERED BY THE MODEL AGENT; this file fixes the contract.

``python -m haltere.obstacles.train --fold F12 --arch dav2s --gpu-budget-min 40 --chunk-min 10
--max-temp 75 --flight-lock PATH [--bench]``

Data: ``FrameStore.rows(fold=..., side='inner_train')`` for fitting and ``side='inner_val'`` for early
stopping and selection (F12 inner validation = the F5 group; ALL = held-back flights). Never touch the
fold's test side or the sealed part. Per-epoch environment shares from ``splits.sampling_weights``
(sqrt balancing, Straw Bale <= 30 %, Drawing Board <= 20 %); rows flagged DENSE_EXTRA are sampled at
the stride-grid rate so event windows are not over-represented. Frames are masked with
``overlays.masked_input`` + ``validity_channel``; ghost-trail pixels carry no loss. Augmentation:
colour/gamma/brightness incl. a low-light mode, blur, noise, JPEG/H.264 re-encode, horizontal flip
with ``contract.mirror_grid``/``mirror_fan`` and gravity x -> -x, synthetic ring strokes (half of them
on labelled obstacles), random HUD glyphs. No crops, zooms or rotations (the fan geometry depends on
f = 140 px and the 30 deg tilt). Hindsight index fields (tti_s, event_id, IN_CLEAN/PRE_EVENT flags)
are never inputs.

Losses (``laplace_censored_nll`` and ``fan_bce`` below are the numpy references; the torch versions
must match them to 1e-5):

- grid and fan distance: censored Laplace NLL on ln(metres) with mu = q50 and
  b = max((q50 - q20) / ln 2.5, 0.01) (the Laplace q20 is mu - b ln 2.5). EXACT: ln(2b) + |y - mu|/b;
  LOWER s: -ln P(Y >= ln s); UPPER u: -ln P(Y <= ln u); UNKNOWN: 0.
- fan probabilities: BCE with ``labels.fan_blocked_targets`` for 4 m and 8 m (weight 0 = ignored).
- weight x4 on grid cells whose centre ray is within 30 deg of the heading and whose label value is
  <= 8 m, and on fan directions with |yaw| <= 30 deg.
- teacher shape loss (weight 0.1): scale/shift-invariant L1 between -grid q50 (log) and the teacher
  log-disparity pooled to 18 x 32, after a per-frame least-squares affine alignment, on valid cells.

Outputs in ``runs/obstacle-train/<FOLD>-v<k>/``: ckpt_<chunk>.pt (model + optimiser + sampler state,
every chunk, resumable), train_log.jsonl (per chunk: losses, inner-val metrics, GPU temperature,
flight-lock waits), model.pt + model.json (model.MODEL_CARD_KEYS). GPU work only in <= 10 min chunks
behind ``thermal.ChunkGuard(lock, gpu=True)`` (pause > 75 C until <= 65 C, hard stop 80 C);
``thermal.limit_threads(2, torch=True, cv2=True)``. ``--bench`` measures step throughput in < 2 min
before any budgeted training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .labels import LabelKind, fan_blocked_targets

LN_2P5 = float(np.log(2.5))
MIN_SCALE = 0.01
NEAR_WEIGHT = 4.0
NEAR_RANGE_M = 8.0
NEAR_YAW_DEG = 30.0
TEACHER_WEIGHT = 0.1
DEFAULT_OUT = Path(__file__).resolve().parents[2] / 'runs' / 'obstacle-train'


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


def train(args):
    raise NotImplementedError('ClearanceNet training: delivered by the model build agent')


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m haltere.obstacles.train')
    p.add_argument('--fold', default='F12')
    p.add_argument('--arch', default='dav2s', choices=('dav2s', 'resnet18fpn'))
    p.add_argument('--store', type=Path, default=None)
    p.add_argument('--out', type=Path, default=None, help='default runs/obstacle-train/<FOLD>-v0')
    p.add_argument('--recipe', type=Path, default=None)
    p.add_argument('--gpu-budget-min', type=float, default=40)
    p.add_argument('--chunk-min', type=float, default=10)
    p.add_argument('--max-temp', type=float, default=75)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bench', action='store_true')
    p.add_argument('--flight-lock', default=None)
    args = p.parse_args(argv)
    if args.chunk_min > 10:
        raise SystemExit('GPU chunks are limited to 10 minutes')
    if args.fold not in ('F12', 'F3', 'F4', 'F5', 'ALL'):
        raise SystemExit('unknown fold')
    return train(args)


if __name__ == '__main__':
    main()
