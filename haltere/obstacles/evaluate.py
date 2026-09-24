"""Held-out obstacle harness E1-E8, baselines B0-B4, thresholds freeze and scoring ledger (offline only).

The metric and baseline functions are DELIVERED BY THE EVALUATION AGENT; this module fixes their
interfaces, the prediction format, deployed timing, provenance records, the freeze and the
score-once ledger (those parts are implemented here).

Predictions (``PredictionSet``): one entry per store row the predictor ran on. Distances are metres,
probabilities 0..1, float32. A frame's output becomes usable at ``t_wall + latency_s`` (deployed 65 ms
at ~16 Hz; stress 100 and 150 ms). Decisions at time t use the latest prediction with
t_available <= t (``latest_available``); nothing may use a frame captured after t.

Records: every metric value is one JSON record (``make_record``) carrying the metric, fold,
environment, held_out / seen_environment / sealed flags, predictor name/kind/causal/sha256,
thresholds sha256, store index sha256, labels manifest sha256, latency and counts. Records of a held-out
score are appended to ``runs/obstacle-train/eval_ledger.jsonl``; ``--once`` refuses a second held-out
score for the same (predictor sha256, fold, thresholds sha256). Every version is reported; no silent
best-of.

Thresholds: ``configs/obstacles/thresholds.json`` is frozen (``freeze``) before any held-out score.
The hash covers everything except the meta keys (frozen, frozen_at, sha256). Decision parameters are
chosen on fold training environments only.

Metric definitions (thresholds.json holds the numbers):

E1  Range at impact points. Frames T-3 .. T-0.15 s before each held-out event where the labelled
    point projects into the image; true = |point - camera| (m); predicted = minimum grid_q50 over the
    3 x 3 cells around the projected cell. Median predicted/true (and IQR) in the 0.5-2, 2-4, 4-7,
    7-16 m bins; P(pred > 1.5 true | true < 6 m). Per environment and pooled; per unique obstacle too.
E2  Box-collider cells (F4 held-out). Grid cells with COLLIDER EXACT labels: AbsRel, delta < 1.25,
    near-cell (< 6 m) overestimation rate (> 1.5x); fan blocked-within-6 m (fan_q20 <= 6) recall and
    precision for |yaw| <= 20 deg against collider fan labels.
E3  Fan quality vs held-out hindsight labels: P(blocked <= 8 m) AUROC and ECE (10 bins) on
    fan_blocked_targets; blocked-within-6 m precision/recall within +-20 deg of the ring bearing
    (heading when no cue); flown-tube false blocks: share of TUBE-LOWER grid cells with
    grid_q20 < label; gate-opening blocked share before gate passes (when gate-pass times exist).
E4  Lateral decision replay on the lateral-study events (Minus 4, Pine 4, Straw 3) and new blind
    lateral events: a frozen decision rule maps each prediction to a Side; correct free side held
    >= 1.0 s before impact (0.15 s end grace), first wrong-side lead, wrong side held >= 0.3 s.
    Reuses the lateral study's scoring (scratchpad/lateral/cand-split-looming/evalcore.py).
E5  Out-of-view and vertical events reported separately (impact point out of view in > 1/3 of the
    T-2 .. T-1 s window, or free side up/down): climb/brake indication lead where applicable.
E6  Clean windows (store clean_windows.json, plus held-out clean laps): activation episodes
    (>= 0.2 s) per minute overall and per environment, % time active, the minus-fast6-01 PD pillar
    near pass (12.2-13.0 s) must not swerve toward the pillar.
E7  Per-unique-obstacle success with Wilson 95 % intervals (``wilson_interval``); pooled rates too.
E8  Leak tests: median relative change of predicted range at test pixels under ring paint on an
    obstacle / on free space, ring removal, HUD-glyph scramble, inserted ghost trails (and 0.5-1.5x
    translation scaling for a temporal variant). Pass < 5 %.

Baselines (PredictionSet with kind='baseline'):
B0  constant RIGHT and constant LEFT side decision (E4/E6).
B1  split looming side decision from the lateral study (config_selected.json) (E4/E6).
B2  pretrained Depth-Anything-V2 Metric-Indoor-Small at 252 x 448 fp16 -> grid (E1-E3; fan via
    contract.grid_to_fan; E4 via the same frozen decision rule as the model).
B3  pretrained relative DA-V2-Small + ONE monotone log-range calibration fitted on the fold's
    training environments only (tests whether the scale compression is environment-consistent).
B4  per-frame affine fit of relative disparity to the evaluated frame's own labels: NON-CAUSAL
    ceiling, causal=False, never a candidate.
Reproduction targets (plan M1): B1 1/11 lateral, constant right 7/10, B2 E1 medians within +-15 % of
2.62 (2-4 m) and 1.72 (4-7 m).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

from .contract import FAN_SHAPE, GRID_SHAPE
from .splits import FOLDS, SEALED_ENVS, env_side

EVAL_SCHEMA = 'haltere.obstacles.eval.v1'
REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = REPO_ROOT / 'configs' / 'obstacles' / 'thresholds.json'
LEDGER_PATH = REPO_ROOT / 'runs' / 'obstacle-train' / 'eval_ledger.jsonl'
DEPLOYED_LATENCY_S = 0.065
STRESS_LATENCY_S = (0.100, 0.150)
DEPLOYED_RATE_HZ = 16.0
THRESHOLD_META_KEYS = ('frozen', 'frozen_at', 'sha256')

METRICS = ('E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8')
BASELINES = {
    'B0': dict(name='constant side (right / left)', causal=True, outputs=('side',)),
    'B1': dict(name='split looming (lateral study)', causal=True, outputs=('side',)),
    'B2': dict(name='pretrained DA-V2 metric-indoor 252x448 fp16', causal=True, outputs=('grid',)),
    'B3': dict(name='pretrained relative + monotone log-range calibration (fold train envs)', causal=True,
               outputs=('grid',)),
    'B4': dict(name='per-frame affine fit to the frame labels (NON-CAUSAL ceiling)', causal=False,
               outputs=('grid',)),
}


class Side(IntEnum):
    """Lateral decision codes (same as the lateral study's SIDE_CODE)."""
    NONE = 0
    LEFT = 1
    RIGHT = 2
    CENTRE = 3


# ----------------------------------------------------------------------------- predictions

PRED_FIELDS = {
    'grid_q20': GRID_SHAPE, 'grid_q50': GRID_SHAPE,
    'fan_q20': FAN_SHAPE, 'fan_q50': FAN_SHAPE, 'fan_p4': FAN_SHAPE, 'fan_p8': FAN_SHAPE,
    'side': (),
}


@dataclass
class PredictionSet:
    """Per-frame outputs of one predictor over store rows (metres, probabilities, Side codes)."""
    name: str
    kind: str                         # 'model' | 'baseline'
    causal: bool
    rows: np.ndarray                  # (N,) int64 store rows, strictly increasing
    fold: str | None = None
    sha256: str | None = None         # model.pt sha256 or a hash of the baseline code/config
    baseline_id: str | None = None
    latency_s: float = DEPLOYED_LATENCY_S
    arrays: dict = field(default_factory=dict)   # subset of PRED_FIELDS -> (N, *shape) arrays

    def validate(self, n_store: int | None = None) -> 'PredictionSet':
        if self.kind not in ('model', 'baseline'):
            raise ValueError('kind must be model or baseline')
        rows = np.asarray(self.rows)
        if rows.ndim != 1 or (len(rows) > 1 and not (np.diff(rows) > 0).all()):
            raise ValueError('rows must be strictly increasing store rows')
        if n_store is not None and len(rows) and (rows[0] < 0 or rows[-1] >= n_store):
            raise ValueError('rows outside the store')
        for k, a in self.arrays.items():
            if k not in PRED_FIELDS:
                raise ValueError(f'unknown prediction field {k!r}')
            if a.shape != (len(rows),) + PRED_FIELDS[k]:
                raise ValueError(f'{k}: shape {a.shape} != {(len(rows),) + PRED_FIELDS[k]}')
        if self.kind == 'baseline' and self.baseline_id not in BASELINES:
            raise ValueError('baseline_id must be one of B0-B4')
        if self.baseline_id == 'B4' and self.causal:
            raise ValueError('B4 is a non-causal ceiling')
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        meta = dict(schema=EVAL_SCHEMA, name=self.name, kind=self.kind, causal=self.causal, fold=self.fold,
                    sha256=self.sha256, baseline_id=self.baseline_id, latency_s=self.latency_s)
        np.savez_compressed(path, rows=np.asarray(self.rows, np.int64), meta=json.dumps(meta),
                            **{k: np.asarray(v) for k, v in self.arrays.items()})

    @classmethod
    def load(cls, path: str | Path) -> 'PredictionSet':
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z['meta']))
        meta.pop('schema', None)
        arrays = {k: z[k] for k in z.files if k in PRED_FIELDS}
        return cls(rows=z['rows'], arrays=arrays, **meta).validate()


def available_times(t_wall, latency_s: float = DEPLOYED_LATENCY_S) -> np.ndarray:
    return np.asarray(t_wall, np.float64) + latency_s


def latest_available(t_available, t_query) -> np.ndarray:
    """Index of the latest prediction with t_available <= t_query for each query (-1 if none).

    ``t_available`` must be sorted ascending (store rows of one run are in capture order).
    """
    t_available = np.asarray(t_available, np.float64)
    if len(t_available) > 1 and (np.diff(t_available) < 0).any():
        raise ValueError('t_available must be sorted')
    return np.searchsorted(t_available, np.asarray(t_query, np.float64), side='right') - 1


# ----------------------------------------------------------------------------- thresholds

def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')


def thresholds_sha256(obj: dict) -> str:
    body = {k: v for k, v in obj.items() if k not in THRESHOLD_META_KEYS}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def freeze_thresholds(path: str | Path = THRESHOLDS_PATH) -> dict:
    """Mark thresholds.json frozen and record its hash. Refuses to re-freeze changed content silently:
    a frozen file whose content changed must get a new ``version`` first."""
    path = Path(path)
    obj = json.loads(path.read_text(encoding='utf-8'))
    sha = thresholds_sha256(obj)
    if obj.get('frozen') and obj.get('sha256') != sha:
        raise ValueError(f'{path} was frozen and has since changed: bump "version" and freeze again')
    if not obj.get('frozen'):
        obj.update(frozen=True, frozen_at=_dt.datetime.now().isoformat(timespec='seconds'), sha256=sha)
        path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return obj


def load_thresholds(path: str | Path = THRESHOLDS_PATH, *, require_frozen: bool = True) -> tuple[dict, str]:
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    sha = thresholds_sha256(obj)
    if require_frozen:
        if not obj.get('frozen'):
            raise RuntimeError(f'{path} is not frozen: run `python -m haltere.obstacles.evaluate freeze` '
                               'before any held-out score')
        if obj.get('sha256') != sha:
            raise RuntimeError(f'{path} changed after freezing (sha256 mismatch)')
    return obj, sha


# ----------------------------------------------------------------------------- records and ledger

def environment_flags(fold: str, env: str) -> dict:
    """held_out (fold test side), seen_environment (fold training side) and sealed flags for a record."""
    if env in SEALED_ENVS:
        return dict(held_out=True, seen_environment=False, sealed=True)
    side = env_side(fold, env)
    return dict(held_out=side == 'test', seen_environment=side != 'test', sealed=False)


def make_record(metric: str, *, pred: PredictionSet, fold: str, env: str, value, n: int,
                thresholds_sha256: str, store_index_sha256: str, labels_manifest_sha256: str | None,
                variant: str | None = None, n_events: int | None = None, ci95=None, passed: bool | None = None,
                latency_s: float | None = None, code_commit: str | None = None, extra: dict | None = None) -> dict:
    if metric not in METRICS:
        raise ValueError(f'unknown metric {metric!r}')
    if fold not in FOLDS:
        raise ValueError(f'unknown fold {fold!r}')
    flags = environment_flags(fold, env) if env != 'pooled' else dict(held_out=None, seen_environment=None,
                                                                       sealed=False)
    return dict(schema=EVAL_SCHEMA, metric=metric, variant=variant, fold=fold, env=env, **flags,
                predictor=dict(name=pred.name, kind=pred.kind, causal=pred.causal, sha256=pred.sha256,
                               baseline_id=pred.baseline_id, fold=pred.fold),
                thresholds_sha256=thresholds_sha256, store_index_sha256=store_index_sha256,
                labels_manifest_sha256=labels_manifest_sha256,
                latency_s=pred.latency_s if latency_s is None else latency_s,
                n=int(n), n_events=n_events, value=value, ci95=ci95, passed=passed,
                code_commit=code_commit, created=_dt.datetime.now().isoformat(timespec='seconds'),
                **(extra or {}))


class Ledger:
    """Append-only score ledger enforcing one held-out score per (predictor sha256, fold, thresholds sha256)."""

    def __init__(self, path: str | Path = LEDGER_PATH):
        self.path = Path(path)

    @staticmethod
    def key(predictor_sha256: str, fold: str, thresholds_sha: str) -> str:
        return f'{predictor_sha256}|{fold}|{thresholds_sha}'

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def scored(self, predictor_sha256: str, fold: str, thresholds_sha: str) -> bool:
        k = self.key(predictor_sha256, fold, thresholds_sha)
        return any(e.get('key') == k for e in self.entries())

    def append(self, predictor_sha256: str, fold: str, thresholds_sha: str, records: list[dict], *,
               once: bool = True) -> None:
        if not predictor_sha256:
            raise ValueError('a held-out score needs the predictor sha256')
        if once and self.scored(predictor_sha256, fold, thresholds_sha):
            raise RuntimeError(f'{predictor_sha256[:12]} was already scored on {fold} held-out data with these '
                               'thresholds; report that result instead of re-scoring')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(key=self.key(predictor_sha256, fold, thresholds_sha), predictor_sha256=predictor_sha256,
                     fold=fold, thresholds_sha256=thresholds_sha,
                     created=_dt.datetime.now().isoformat(timespec='seconds'), records=records)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (E7)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


# ----------------------------------------------------------------------------- metrics (evaluation agent)

def e1_range_at_impacts(pred, store, labels, events, thresholds) -> list[dict]:
    raise NotImplementedError('E1: evaluation build agent')


def e2_collider_cells(pred, store, labels, thresholds) -> list[dict]:
    raise NotImplementedError('E2: evaluation build agent')


def e3_fan_quality(pred, store, labels, thresholds) -> list[dict]:
    raise NotImplementedError('E3: evaluation build agent')


def e4_lateral_replay(pred, store, events, thresholds, *, latency_s: float = DEPLOYED_LATENCY_S) -> list[dict]:
    raise NotImplementedError('E4: evaluation build agent')


def e5_out_of_view(pred, store, events, thresholds) -> list[dict]:
    raise NotImplementedError('E5: evaluation build agent')


def e6_clean_windows(pred, store, thresholds, *, latency_s: float = DEPLOYED_LATENCY_S) -> list[dict]:
    raise NotImplementedError('E6: evaluation build agent')


def e7_unique_obstacles(records: list[dict], events) -> list[dict]:
    raise NotImplementedError('E7: evaluation build agent')


def e8_leak_tests(predict_fn, store, rows, thresholds) -> list[dict]:
    raise NotImplementedError('E8: evaluation build agent')


def decide_side(pred: PredictionSet, store, thresholds) -> np.ndarray:
    """Frozen decision rule (thresholds.json 'decision') mapping fan/grid outputs to Side codes per row."""
    raise NotImplementedError('decision rule: evaluation build agent')


# ----------------------------------------------------------------------------- baselines (evaluation agent)

def baseline_b0(store, rows, side: Side) -> PredictionSet:
    raise NotImplementedError('B0: evaluation build agent')


def baseline_b1(store, rows) -> PredictionSet:
    raise NotImplementedError('B1: evaluation build agent')


def baseline_b2(store, rows, guard) -> PredictionSet:
    raise NotImplementedError('B2: evaluation build agent')


def baseline_b3(store, rows, fold: str, guard) -> PredictionSet:
    raise NotImplementedError('B3: evaluation build agent')


def baseline_b4(store, rows, labels, guard) -> PredictionSet:
    raise NotImplementedError('B4: evaluation build agent')


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m haltere.obstacles.evaluate')
    sub = p.add_subparsers(dest='cmd', required=True)
    fz = sub.add_parser('freeze', help='freeze configs/obstacles/thresholds.json and print its sha256')
    fz.add_argument('--thresholds', type=Path, default=THRESHOLDS_PATH)
    b = sub.add_parser('baselines', help='score baselines B0-B4')
    b.add_argument('--set', default='B0,B1,B2,B3,B4')
    b.add_argument('--fold', default='F12')
    b.add_argument('--once', action='store_true')
    b.add_argument('--flight-lock', default=None)
    m = sub.add_parser('model', help='score a frozen model directory once on its fold')
    m.add_argument('model_dir', type=Path)
    m.add_argument('--fold', default='F12')
    m.add_argument('--once', action='store_true')
    m.add_argument('--sealed-final', action='store_true')
    m.add_argument('--flight-lock', default=None)
    args = p.parse_args(argv)
    if args.cmd == 'freeze':
        obj = freeze_thresholds(args.thresholds)
        print(f'{args.thresholds}: frozen {obj["frozen_at"]} sha256 {obj["sha256"]}')
        return
    raise NotImplementedError(f'evaluate {args.cmd}: evaluation build agent')


if __name__ == '__main__':
    main()
