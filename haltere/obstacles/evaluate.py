"""Held-out obstacle harness E1-E8, baselines B0-B4, thresholds freeze and scoring ledger (offline only).

Predictions (``PredictionSet``): one entry per store row the predictor ran on. Distances are metres,
probabilities 0..1, float32. A frame's output becomes usable at ``t_wall + latency_s`` (deployed 65 ms
at ~16 Hz; stress 100 and 150 ms). Decisions at time t use the latest prediction with
t_available <= t (``latest_available``); nothing may use a frame captured after t. Frames denser than
the deployed rate are thinned with ``deployed_schedule`` (a skipped frame produces no new output, so the
previous decision stays in force).

Records: every metric value is one JSON record (``make_record``) carrying the metric, fold,
environment, held_out / seen_environment / sealed flags, predictor name/kind/causal/sha256,
thresholds sha256, store index sha256, labels manifest sha256, latency and counts. Records of a held-out
score are appended to ``runs/obstacle-train/eval_ledger.jsonl``; ``--once`` refuses a second held-out
score for the same (predictor sha256, fold, thresholds sha256). Every version is reported; no silent
best-of.

Thresholds: ``configs/obstacles/thresholds.json`` is frozen (``freeze``) before any held-out score.
The hash covers everything except the meta keys (frozen, frozen_at, sha256). Decision parameters are
chosen on fold training environments only (version 1: a-priori values from the M3 governor design,
fitted to no data).

Time axis: events and clean windows use the CSV ``phase`` clock (store ``t_phase``); runs without it
fall back to ``t_wall`` (events then need ``t_wall``). Labels, events and clean windows are offline
evaluation data only; predictors never see them (B4 is the declared non-causal exception).

Metric definitions (thresholds.json holds the numbers):

E1  Range at impact points. Frames T-3 .. T-0.15 s before each event where the labelled point
    (``point_w``, else ``drone_pos_w``) projects into the image; true = |point - camera| (m);
    predicted = minimum grid_q50 over the ``neighbourhood_cells`` x ``neighbourhood_cells`` cells around the
    projected cell (1 in thresholds v1: the study's 3 x 3 pixels at 160 x 90 are ~8 px at 448, and a 3 x 3-cell
    window biases the ratio down by ~30 % on identical depth maps). Median predicted/true
    (IQR, obstacle-cluster bootstrap CI) in the 0.5-2, 2-4, 4-7, 7-16 m bins; P(pred > 1.5 true |
    true < 6 m); pillar approaches (range within 1.3x from 6 m to 1.5 m). Per environment, pooled over
    the fold's held-out environments and pooled over all; per unique obstacle too. The optional
    ``prior_filter`` (chord within 15 deg of the velocity, 0.5-20 m, speed >= 1 m/s) reproduces the
    lateral study's scale check.
E2  Box-collider cells. Grid cells with COLLIDER EXACT labels: AbsRel, delta < 1.25, near-cell (< 6 m)
    overestimation rate (> 1.5x); fan blocked-within-6 m (fan_q20 <= 6) recall and precision for
    |yaw| <= 20 deg against COLLIDER fan labels.
E3  Fan quality vs hindsight labels: P(blocked <= 8 m) and P(blocked <= 4 m) AUROC, ECE (10 bins) and
    Brier on labels.fan_blocked_targets; blocked-within-6 m precision/recall within +-20 deg yaw and
    +-10 deg elevation of the ring bearing (velocity when no cue); flown-tube false blocks: share of
    TUBE-LOWER grid cells with grid_q20 < 0.9 x label; gate opening: share of frames 2.0-0.5 s before a
    gate pass whose grid_q20 at the pass point is < 0.9 x its distance (passes from logged cue switches
    or given explicitly).
E4  Lateral decision replay on lateral events (``lateral`` true, terminal impacts): the frozen decision
    rule (``decide_side``) or the predictor's own side output; correct free side sustained >= 1.0 s
    before impact through T - 0.15 s (the lateral study's scoring, evalcore.impact_metrics), first
    wrong-side lead, longest wrong-side hold (>= 0.3 s counts as wrong). Strict (primary side;
    'either' accepts both) and permissive (accepted sides) variants.
E5  Out-of-view and vertical events (point out of view in > 1/3 of T-2 .. T-1 s, or free side up/down or
    unknown): in-view fraction and warning lead (any non-NONE decision, incl. CENTRE = brake/climb).
E6  Clean windows (store clean_windows.json): side episodes (>= 0.2 s) per minute overall and per
    environment, % time with a side, % time CENTRE, plus 'any activation' (side or centre); near passes
    (events kind near_pass and thresholds E6.pd_pillar_near_pass): time and longest episode steering
    toward the obstacle.
E7  Per-unique-obstacle success with Wilson 95 % intervals; pooled event rates with Wilson and an
    obstacle-cluster bootstrap interval (repeated hits on one pillar are not independent).
E8  Leak tests (haltere.obstacles.leaks): median relative change of predicted range at test cells under
    ring paint on an obstacle / on free space, ring removal, HUD-glyph scramble and inserted ghost
    trails. Pass < 5 %. (Translation scaling applies to a temporal variant only.)

Baselines (PredictionSet with kind='baseline'):
B0  constant RIGHT and constant LEFT side decision (E4/E6).
B1  split looming side decision from the lateral study (haltere.obstacles.split_looming, selected config).
B2  pretrained Depth-Anything-V2 Metric-Indoor-Small at 252 x 448 fp16 -> range grid (min over each cell)
    (E1-E3; fan via contract.grid_to_fan; E4 via the same frozen decision rule as the model).
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
import subprocess
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

from . import contract
from .contract import FAN_ELEV_DEG, FAN_MAX_M, FAN_SHAPE, FAN_YAW_DEG, GRID_H, GRID_SHAPE, GRID_W, PATCH_PX
from .splits import ENV_BY_CODE, FOLDS, SEALED_ENVS, env_side

EVAL_SCHEMA = 'haltere.obstacles.eval.v1'
REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = REPO_ROOT / 'configs' / 'obstacles' / 'thresholds.json'
LEDGER_PATH = REPO_ROOT / 'runs' / 'obstacle-train' / 'eval_ledger.jsonl'
REPRO_LEDGER_PATH = REPO_ROOT / 'runs' / 'obstacle-train' / 'eval_ledger_reproduction.jsonl'
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
    """Lateral decision codes (same as the lateral study's SIDE_CODE). LEFT/RIGHT = the side to steer toward."""
    NONE = 0
    LEFT = 1
    RIGHT = 2
    CENTRE = 3


SIDE_NAME = {Side.NONE: 'none', Side.LEFT: 'left', Side.RIGHT: 'right', Side.CENTRE: 'centre'}
SIDE_OF_NAME = {'left': Side.LEFT, 'right': Side.RIGHT}


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

    def positions(self, rows) -> np.ndarray:
        """Index into this set's arrays for each store row (-1 where the predictor did not run)."""
        rows = np.asarray(rows, dtype=np.int64)
        mine = np.asarray(self.rows, dtype=np.int64)
        if len(mine) == 0:
            return np.full(len(rows), -1, np.int64)
        pos = np.clip(np.searchsorted(mine, rows), 0, len(mine) - 1)
        return np.where(mine[pos] == rows, pos, -1)

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


def deployed_schedule(t, rate_hz: float | None = DEPLOYED_RATE_HZ, jitter: float = 0.25) -> np.ndarray:
    """Boolean mask of the frames a ``rate_hz`` runtime would process (sorted capture times of one run).

    A frame is processed when at least (1 - jitter) / rate_hz s passed since the last processed frame,
    so a ~17 Hz stream is kept whole and a 30 Hz stream is halved. ``rate_hz=None`` keeps everything.
    """
    t = np.asarray(t, np.float64)
    keep = np.ones(len(t), bool)
    if rate_hz is None or len(t) == 0:
        return keep
    min_dt = (1.0 - jitter) / float(rate_hz)
    last = -np.inf
    for i, ti in enumerate(t):
        if ti - last >= min_dt - 1e-9:
            last = ti
        else:
            keep[i] = False
    return keep


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
        missing = unset_decision_parameters(obj)
        if missing:
            raise ValueError(f'{path}: decision parameters not set: {missing}')
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


def unset_decision_parameters(obj: dict) -> list[str]:
    """Dotted names of decision parameters that are still null (a draft cannot be frozen with them)."""
    out = []

    def walk(d, prefix):
        for k, v in d.items():
            if k == 'note':
                continue
            if isinstance(v, dict):
                walk(v, f'{prefix}{k}.')
            elif v is None:
                out.append(prefix + k)
    walk(obj.get('decision', {}), 'decision.')
    return out


# ----------------------------------------------------------------------------- records and ledger

def environment_flags(fold: str, env: str) -> dict:
    """held_out (fold test side), seen_environment (fold training side) and sealed flags for a record."""
    if env in SEALED_ENVS:
        return dict(held_out=True, seen_environment=False, sealed=True)
    side = env_side(fold, env)
    return dict(held_out=side == 'test', seen_environment=side != 'test', sealed=False)


def pooled_flags(fold: str, envs) -> dict:
    """Flags of a pooled record: True/False when every pooled environment agrees, else None (mixed)."""
    flags = [environment_flags(fold, e) for e in envs if e in ENV_NAMES_ALL]
    if not flags:
        return dict(held_out=None, seen_environment=None, sealed=False)
    out = {}
    for k in ('held_out', 'seen_environment', 'sealed'):
        vals = {f[k] for f in flags}
        out[k] = vals.pop() if len(vals) == 1 else None
    if out['sealed'] is None:
        out['sealed'] = True       # any sealed environment makes the pooled record sealed
    return out


ENV_NAMES_ALL = tuple(e.name for e in ENV_BY_CODE.values())


def make_record(metric: str, *, pred: PredictionSet, fold: str, env: str, value, n: int,
                thresholds_sha256: str, store_index_sha256: str, labels_manifest_sha256: str | None,
                variant: str | None = None, n_events: int | None = None, ci95=None, passed: bool | None = None,
                latency_s: float | None = None, code_commit: str | None = None, extra: dict | None = None,
                envs=None) -> dict:
    """One metric record. ``env='pooled'`` with ``envs`` = the pooled environments sets the flags when
    they agree (held-out only / training only)."""
    if metric not in METRICS:
        raise ValueError(f'unknown metric {metric!r}')
    if fold not in FOLDS:
        raise ValueError(f'unknown fold {fold!r}')
    if env != 'pooled':
        flags = environment_flags(fold, env)
    elif envs is not None:
        flags = pooled_flags(fold, envs)
    else:
        flags = dict(held_out=None, seen_environment=None, sealed=False)
    rec = dict(schema=EVAL_SCHEMA, metric=metric, variant=variant, fold=fold, env=env, **flags,
               predictor=dict(name=pred.name, kind=pred.kind, causal=pred.causal, sha256=pred.sha256,
                              baseline_id=pred.baseline_id, fold=pred.fold),
               thresholds_sha256=thresholds_sha256, store_index_sha256=store_index_sha256,
               labels_manifest_sha256=labels_manifest_sha256,
               latency_s=pred.latency_s if latency_s is None else latency_s,
               n=int(n), n_events=n_events, value=value, ci95=ci95, passed=passed,
               code_commit=code_commit, created=_dt.datetime.now().isoformat(timespec='seconds'),
               **(extra or {}))
    if envs is not None:
        rec['envs'] = sorted(envs)
    return rec


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
               once: bool = True, note: str | None = None) -> None:
        if not predictor_sha256:
            raise ValueError('a held-out score needs the predictor sha256')
        already = self.scored(predictor_sha256, fold, thresholds_sha)
        if once and already:
            raise RuntimeError(f'{predictor_sha256[:12]} was already scored on {fold} held-out data with these '
                               'thresholds; report that result instead of re-scoring')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(key=self.key(predictor_sha256, fold, thresholds_sha), predictor_sha256=predictor_sha256,
                     fold=fold, thresholds_sha256=thresholds_sha,
                     created=_dt.datetime.now().isoformat(timespec='seconds'), rescore=already, note=note,
                     records=records)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, default=_json_default) + '\n')


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f'not JSON serialisable: {type(o)}')


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (E7)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def cluster_bootstrap(values, groups, stat=np.median, *, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float] | None:
    """Percentile interval of ``stat`` resampling whole groups (unique obstacles) with replacement."""
    values = np.asarray(values, np.float64)
    groups = np.asarray(groups)
    keys = np.unique(groups)
    if len(values) == 0 or len(keys) == 0:
        return None
    members = [values[groups == k] for k in keys]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        out[b] = stat(np.concatenate([members[i] for i in pick]))
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def auroc(score, target) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties); NaN without both classes."""
    from scipy.stats import rankdata
    score = np.asarray(score, np.float64)
    target = np.asarray(target).astype(bool)
    n1, n0 = int(target.sum()), int((~target).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(score)
    return float((r[target].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def expected_calibration_error(p, y, bins: int = 10) -> float:
    """ECE with ``bins`` equal-width probability bins."""
    p = np.clip(np.asarray(p, np.float64), 0.0, 1.0)
    y = np.asarray(y, np.float64)
    if len(p) == 0:
        return float('nan')
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def _code_commit() -> str | None:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- store / event plumbing

class _StoreCache:
    """Per-store numpy copies of the index fields the harness needs, and per-run row lists in time order."""

    def __init__(self, store):
        ix = store.index
        self.n = len(ix)
        self.run_id = np.asarray(ix['run_id'], np.int64)
        self.env = np.asarray(ix['env'], np.int64)
        self.t_wall = np.asarray(ix['t_wall'], np.float64)
        self.t_phase = np.asarray(ix['t_phase'], np.float64)
        self.pos = np.asarray(ix['pos'], np.float64)
        self.quat = np.asarray(ix['quat'], np.float64)
        self.vel = np.asarray(ix['vel'], np.float64)
        self.cue_uv = np.asarray(ix['cue_uv'], np.float64)
        self.cue_src = np.asarray(ix['cue_src'], np.int64)
        self.rows_by_run: dict[int, np.ndarray] = {}
        self.axis_by_run: dict[int, str] = {}
        order = np.lexsort((np.nan_to_num(self.t_wall, nan=np.inf), self.run_id))
        rid = self.run_id[order]
        starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
        ends = np.r_[starts[1:], len(order)]
        for a, b in zip(starts, ends):
            rows = order[a:b]
            r = int(rid[a])
            self.rows_by_run[r] = rows
            self.axis_by_run[r] = 'phase' if np.isfinite(self.t_phase[rows]).all() else 'wall'

    def times(self, rows, axis: str) -> np.ndarray:
        return (self.t_phase if axis == 'phase' else self.t_wall)[np.asarray(rows, np.int64)]

    def env_name(self, row) -> str:
        e = ENV_BY_CODE.get(int(self.env[row]))
        return e.name if e is not None else 'unknown'


def store_cache(store) -> _StoreCache:
    c = getattr(store, '_eval_cache', None)
    if c is None or c.n != len(store.index):
        c = _StoreCache(store)
        try:
            store._eval_cache = c
        except AttributeError:
            pass
    return c


def _side_table(store, name: str):
    if hasattr(store, 'side_table'):
        return store.side_table(name)
    root = getattr(store, 'root', None)
    if root is None:
        return None
    path = Path(root) / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def store_events(store) -> list[dict]:
    obj = _side_table(store, 'events.json')
    if obj is None:
        return []
    return list(obj['events'] if isinstance(obj, dict) else obj)


def clean_windows(store) -> list[dict]:
    """Store clean windows: {run_id, start_phase_s, end_phase_s, ...}."""
    obj = _side_table(store, 'clean_windows.json')
    if obj is None:
        return []
    return list(obj['windows'] if isinstance(obj, dict) else obj)


def store_index_sha(store) -> str:
    if hasattr(store, 'manifest') and isinstance(store.manifest, dict) and store.manifest.get('index_sha256'):
        return store.manifest['index_sha256']
    return store.index_sha256()


def labels_manifest_sha(labels) -> str | None:
    if labels is None:
        return None
    root = getattr(labels, 'root', None)
    if root is not None and (Path(root) / 'manifest.json').exists():
        return _sha256_file(Path(root) / 'manifest.json')
    m = getattr(labels, 'manifest', None)
    return hashlib.sha256(canonical_json(m)).hexdigest() if m is not None else None


def resolve_event_run(store, event: dict) -> int | None:
    """Store run_id of a labelled event: via store_event_id, else the run's aliases / source id / flight."""
    sid = event.get('store_event_id')
    if sid is not None and int(sid) >= 0:
        for e in store_events(store):
            if int(e.get('store_event_id', -2)) == int(sid):
                return int(e['run_id'])
    names = {event.get('run'), event.get('flight')} - {None}
    cands = []
    for r in store.runs:
        keys = {r.get('source_id'), r.get('flight')} | set(r.get('aliases') or [])
        if names & keys:
            cands.append(r)
    if not cands:
        return None
    cands.sort(key=lambda r: (bool(r.get('secondary')), int(r['run_id'])))
    return int(cands[0]['run_id'])


def _event_time(event: dict, axis: str):
    v = event.get('t_phase') if axis == 'phase' else event.get('t_wall')
    return None if v is None else float(v)


def _event_point(event: dict, point: str = 'point_w'):
    p = event.get(point)
    if p is None and point == 'point_w':
        p = event.get('drone_pos_w')
    return None if p is None else np.asarray(p, np.float64)


def _free_sides(event: dict) -> tuple[set, set] | None:
    """(strict, permissive) lateral free sides of an event, or None when it is not laterally scorable."""
    prim = event.get('primary_free_side')
    if prim == 'either':
        strict = {'left', 'right'}
    elif prim in ('left', 'right'):
        strict = {prim}
    else:
        return None
    acc = {s for s in (event.get('accepted_free_sides') or []) if s in ('left', 'right')} | strict
    return strict, acc


def _is_scored_event(event: dict) -> bool:
    return not event.get('oracle_route', False)


def project_points(P, pos, quat):
    """World point(s) P (3,) or (n, 3) seen from camera poses (n, 3)/(n, 4) -> (d_w, d_c, uv, in_front)."""
    d_w = np.asarray(P, np.float64) - np.asarray(pos, np.float64)
    R_wc = contract.camera_to_world(np.asarray(quat, np.float64))
    d_c = np.einsum('...ji,...j->...i', R_wc, d_w)
    uv, ok = contract.project_camera(d_c)
    return d_w, d_c, uv, ok


def cell_of_uv(uv) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uv, np.float64)
    c = np.clip(np.floor(uv[..., 0] / PATCH_PX).astype(np.int64), 0, GRID_W - 1)
    r = np.clip(np.floor(uv[..., 1] / PATCH_PX).astype(np.int64), 0, GRID_H - 1)
    return r, c


def window_min(grid, r, c, size: int = 3) -> np.ndarray:
    """NaN-aware minimum of (n, 18, 32) grids over the size x size cells centred on (r, c) (NaN if none)."""
    grid = np.asarray(grid, np.float64)
    h = size // 2
    pad = np.pad(grid, ((0, 0), (h, h), (h, h)), constant_values=np.nan)
    n = len(grid)
    vals = np.stack([pad[np.arange(n), r + dr, c + dc] for dr in range(size) for dc in range(size)], axis=1)
    finite = np.isfinite(vals)
    out = np.where(finite, vals, np.inf).min(axis=1)
    return np.where(finite.any(axis=1), out, np.nan)


def reference_bearing(cue_uv, quat, vel, min_speed: float = 1.0):
    """Per-frame (ref_yaw, ref_elev, travel_yaw, travel_elev, speed) in the heading frame (deg).

    Reference = the logged HUD ring bearing (a runtime observation) when present, else the travel
    (telemetry velocity) direction, else straight ahead. Travel elevation is 0 below ``min_speed``.
    """
    cue_uv = np.asarray(cue_uv, np.float64)
    quat = np.asarray(quat, np.float64)
    vel = np.asarray(vel, np.float64)
    speed = np.linalg.norm(vel, axis=-1)
    with np.errstate(invalid='ignore'):
        yaw_c, el_c = contract.image_bearing_heading(cue_uv, quat)
        yaw_v, el_v = contract.world_bearing_heading(vel, quat)
    moving = speed >= min_speed
    travel_yaw = np.where(moving & np.isfinite(yaw_v), yaw_v, 0.0)
    travel_el = np.where(moving & np.isfinite(el_v), el_v, 0.0)
    has_cue = np.isfinite(yaw_c) & np.isfinite(el_c)
    ref_yaw = np.where(has_cue, yaw_c, travel_yaw)
    ref_el = np.where(has_cue, el_c, travel_el)
    return ref_yaw, ref_el, travel_yaw, travel_el, speed


# ----------------------------------------------------------------------------- fans of grid-only predictors

def prediction_fans(pred: PredictionSet, positions, quat) -> dict:
    """fan_q20/fan_q50/fan_p4/fan_p8 at the given positions; grid-only predictors get a geometric fan.

    Geometric fan (contract.grid_to_fan on grid_q20, else grid_q50): +inf (no grid point in the
    corridor) becomes FAN_MAX_M; probabilities are hard 0/1 (fan <= 4 m / 8 m). ``derived`` says which.
    """
    a = pred.arrays
    positions = np.asarray(positions, np.int64)
    if 'fan_q20' in a or 'fan_q50' in a:
        q20 = np.asarray(a.get('fan_q20', a.get('fan_q50')), np.float64)[positions]
        q50 = np.asarray(a.get('fan_q50', a.get('fan_q20')), np.float64)[positions]
        out = dict(fan_q20=q20, fan_q50=q50, derived=False)
        for k, d in (('fan_p4', 4.0), ('fan_p8', 8.0)):
            out[k] = (np.asarray(a[k], np.float64)[positions] if k in a else (q50 <= d).astype(np.float64))
        return out
    g = a.get('grid_q20', a.get('grid_q50'))
    if g is None:
        raise ValueError(f'{pred.name}: no fan or grid outputs')
    fan = contract.grid_to_fan(np.asarray(g, np.float64)[positions], np.asarray(quat, np.float64))
    fan = np.where(np.isinf(fan), FAN_MAX_M, np.minimum(fan, FAN_MAX_M))
    return dict(fan_q20=fan, fan_q50=fan, fan_p4=(fan <= 4.0).astype(np.float64),
                fan_p8=(fan <= 8.0).astype(np.float64), derived=True)


# ----------------------------------------------------------------------------- decision rule

@dataclass(frozen=True)
class DecisionParams:
    """The frozen side rule (thresholds.json 'decision'); see ``decide_side``."""
    quantile: str
    reaction_s: float
    margin_m: float
    brake_mps2: float
    hold_s: float
    release_s: float
    switch_margin_deg: float
    on_frames: int
    off_frames: int
    elev_window_deg: float = 5.0
    min_speed_mps: float = 1.0
    in_view_margin_px: float = 0.0
    max_gap_s: float = 0.4

    @classmethod
    def from_thresholds(cls, thresholds: dict) -> 'DecisionParams':
        d = thresholds['decision']
        missing = unset_decision_parameters(thresholds)
        if missing:
            raise ValueError(f'decision parameters not set: {missing}')
        rd, hy = d['required_distance'], d['hysteresis']
        return cls(quantile=d['quantile'], reaction_s=float(rd['reaction_s']), margin_m=float(rd['margin_m']),
                   brake_mps2=float(rd['brake_mps2']), hold_s=float(hy['hold_s']), release_s=float(hy['release_s']),
                   switch_margin_deg=float(hy['switch_margin_deg']), on_frames=int(d['on_frames']),
                   off_frames=int(d['off_frames']), elev_window_deg=float(d.get('elev_window_deg', 5.0)),
                   min_speed_mps=float(d.get('min_speed_mps', 1.0)),
                   in_view_margin_px=float(d.get('in_view_margin_px', 0.0)),
                   max_gap_s=float(d.get('max_gap_s', 0.4)))

    def required_distance(self, speed, latency_s: float) -> np.ndarray:
        v = np.asarray(speed, np.float64)
        return v * (latency_s + self.reaction_s) + v * v / (2.0 * self.brake_mps2) + self.margin_m


def raw_side_decisions(fan, in_view, speed, ref_yaw, travel_elev, params: DecisionParams, latency_s: float,
                       travel_yaw=None):
    """Per-frame raw decisions before hysteresis.

    Per yaw column the free distance is the minimum fan value over the elevation rows within
    ``elev_window_deg`` of the travel elevation (at least the nearest row); a row out of view or a NaN
    value counts as blocked (no evidence is not free space). Column clear = free distance >= D(v).
    Reference column clear -> NONE; else the nearest clear column left (positive yaw) or right of the
    reference -> LEFT / RIGHT (ties: larger free distance, then the travel side, then RIGHT); nothing
    clear -> CENTRE. Below ``min_speed_mps`` -> NONE. Returns (codes int8, dist_left, dist_right).
    """
    fan = np.asarray(fan, np.float64)
    n = len(fan)
    speed = np.asarray(speed, np.float64)
    D = params.required_distance(speed, latency_s)
    te = np.clip(np.asarray(travel_elev, np.float64), FAN_ELEV_DEG[0], FAN_ELEV_DEG[-1])
    rows = np.abs(FAN_ELEV_DEG[None, :] - te[:, None]) <= params.elev_window_deg + 1e-9
    nearest = np.argmin(np.abs(FAN_ELEV_DEG[None, :] - te[:, None]), axis=1)
    rows[np.arange(n), nearest] = True
    val = np.where(np.isfinite(fan) & np.asarray(in_view, bool), fan, -np.inf)
    val = np.where(rows[:, :, None], val, np.inf)
    free = val.min(axis=1)                                       # (n, 9)
    clear = free >= D[:, None]
    ry = np.clip(np.asarray(ref_yaw, np.float64), FAN_YAW_DEG[0], FAN_YAW_DEG[-1])
    j_ref = np.argmin(np.abs(FAN_YAW_DEG[None, :] - ry[:, None]), axis=1)
    ang = FAN_YAW_DEG[None, :] - ry[:, None]                     # positive = left of the reference
    col = np.arange(len(FAN_YAW_DEG))[None, :]
    left_ok = clear & (col > j_ref[:, None])
    right_ok = clear & (col < j_ref[:, None])
    dl = np.where(left_ok, ang, np.inf).min(axis=1)
    dr = np.where(right_ok, -ang, np.inf).min(axis=1)
    jl = np.where(left_ok, ang, np.inf).argmin(axis=1)
    jr = np.where(right_ok, -ang, np.inf).argmin(axis=1)
    fl = free[np.arange(n), jl]
    fr = free[np.arange(n), jr]
    codes = np.full(n, int(Side.CENTRE), np.int8)
    codes[dl < dr] = Side.LEFT
    codes[dr < dl] = Side.RIGHT
    tie = np.isfinite(dl) & (dl == dr)
    ty = ry if travel_yaw is None else np.asarray(travel_yaw, np.float64)
    even = tie & (fl == fr)
    codes[even] = np.where(ty[even] > ry[even], int(Side.LEFT), int(Side.RIGHT))
    codes[tie & (fl > fr)] = Side.LEFT
    codes[tie & (fr > fl)] = Side.RIGHT
    codes[clear[np.arange(n), j_ref]] = Side.NONE
    codes[speed < params.min_speed_mps] = Side.NONE
    return codes, dl, dr


def apply_hysteresis(t, raw, dist_left, dist_right, params: DecisionParams) -> np.ndarray:
    """Causal temporal filter of raw decisions over one contiguous segment (times ascending).

    Activation: ``on_frames`` consecutive identical non-NONE raw decisions. A LEFT/RIGHT side is held at
    least ``hold_s``; switching to the opposite side also needs ``on_frames`` consecutive opposite raw
    decisions and an angular advantage >= ``switch_margin_deg`` (the active side's nearest clear column
    farther from the reference by that much, or gone). CENTRE (nothing clear) is entered after
    ``on_frames`` without a hold. Release: raw NONE for >= ``release_s`` and >= ``off_frames`` frames
    after the hold.
    """
    t = np.asarray(t, np.float64)
    out = np.zeros(len(t), np.int8)
    state, since = int(Side.NONE), -np.inf
    cand, cand_n = None, 0
    none_t0, none_n = None, 0
    for i in range(len(t)):
        r = int(raw[i])
        if state == Side.NONE:
            if r != Side.NONE:
                cand_n = cand_n + 1 if cand == r else 1
                cand = r
                if cand_n >= params.on_frames:
                    state, since = r, t[i]
                    cand, cand_n, none_t0, none_n = None, 0, None, 0
            else:
                cand, cand_n = None, 0
        elif r == Side.NONE:
            cand, cand_n = None, 0
            if none_t0 is None:
                none_t0, none_n = t[i], 1
            else:
                none_n += 1
            held = state == Side.CENTRE or t[i] - since >= params.hold_s
            if held and t[i] - none_t0 >= params.release_s - 1e-9 and none_n >= params.off_frames:
                state, since = int(Side.NONE), -np.inf
                none_t0, none_n = None, 0
        else:
            none_t0, none_n = None, 0
            if r == state:
                cand, cand_n = None, 0
            else:
                cand_n = cand_n + 1 if cand == r else 1
                cand = r
                ok = cand_n >= params.on_frames
                if ok and state in (Side.LEFT, Side.RIGHT) and r in (Side.LEFT, Side.RIGHT):
                    d_act = dist_left[i] if state == Side.LEFT else dist_right[i]
                    d_new = dist_left[i] if r == Side.LEFT else dist_right[i]
                    ok = (t[i] - since >= params.hold_s) and (d_act - d_new >= params.switch_margin_deg - 1e-9)
                elif ok and state in (Side.LEFT, Side.RIGHT) and r == Side.CENTRE:
                    ok = True
                elif ok and state == Side.CENTRE:
                    ok = True
                if ok:
                    state, since = r, t[i]
                    cand, cand_n = None, 0
        out[i] = state
    return out


def segments(t, max_gap_s: float) -> list[tuple[int, int]]:
    """[a, b) index ranges of a sorted time series split where consecutive frames are > max_gap_s apart."""
    t = np.asarray(t, np.float64)
    if len(t) == 0:
        return []
    cut = np.flatnonzero(~(np.diff(t) <= max_gap_s)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(t)]
    return list(zip(starts.tolist(), ends.tolist()))


def decide_side(pred: PredictionSet, store, thresholds, *, latency_s: float | None = None,
                rate_hz: float | None = DEPLOYED_RATE_HZ, return_raw: bool = False):
    """Frozen decision rule (thresholds.json 'decision') mapping fan/grid outputs to Side codes per row.

    Runs causally per run over contiguous segments (gap > max_gap_s restarts the filter) at the deployed
    rate: frames the rate schedule skips carry the previous decision (no new output). A predictor with
    its own ``side`` output (B0, B1) is only rate-scheduled. Returns int8 codes aligned with
    ``pred.rows`` (and the raw per-frame codes with ``return_raw``).
    """
    params = DecisionParams.from_thresholds(thresholds) if 'side' not in pred.arrays else None
    latency = pred.latency_s if latency_s is None else float(latency_s)
    c = store_cache(store)
    rows = np.asarray(pred.rows, np.int64)
    out = np.zeros(len(rows), np.int8)
    raw_all = np.zeros(len(rows), np.int8)
    run_of = c.run_id[rows]
    for rid in np.unique(run_of):
        sel = np.flatnonzero(run_of == rid)
        axis = c.axis_by_run.get(int(rid), 'wall')
        t = c.times(rows[sel], axis)
        order = np.argsort(t, kind='stable')
        sel, t = sel[order], t[order]
        gap = params.max_gap_s if params is not None else 0.4
        for a, b in segments(t, gap):
            idx = sel[a:b]
            keep = deployed_schedule(t[a:b], rate_hz)
            if params is None:
                codes = np.asarray(pred.arrays['side'], np.int8)[idx].copy()
                raw = codes.copy()
            else:
                r = rows[idx]
                fans = prediction_fans(pred, idx, c.quat[r])
                fan = fans[params.quantile] if params.quantile in fans else fans['fan_q20']
                inview = contract.fan_in_view(c.quat[r], margin_px=params.in_view_margin_px)
                ref_yaw, _, travel_yaw, travel_el, speed = reference_bearing(c.cue_uv[r], c.quat[r], c.vel[r],
                                                                             params.min_speed_mps)
                raw, dl, dr = raw_side_decisions(fan, inview, speed, ref_yaw, travel_el, params, latency,
                                                 travel_yaw=travel_yaw)
                k = np.flatnonzero(keep)
                codes = np.zeros(len(idx), np.int8)
                codes[k] = apply_hysteresis(t[a:b][k], raw[k], dl[k], dr[k], params)
            # carry the last processed decision through skipped frames
            last = 0
            for j in range(len(idx)):
                if keep[j]:
                    last = codes[j]
                else:
                    codes[j] = last
            out[idx] = codes
            raw_all[idx] = raw
    return (out, raw_all) if return_raw else out


# ----------------------------------------------------------------------------- timeline scoring (pure)

def lateral_event_metrics(t_usable, codes, T: float, *, strict, permissive, window_s: float = 3.0,
                          end_grace_s: float = 0.15, correct_hold_s: float = 1.0,
                          wrong_hold_s: float = 0.3) -> dict:
    """The lateral study's impact scoring (evalcore.impact_metrics) on a decision timeline.

    ``t_usable``: when each decision became usable (capture + latency, same clock as T), ascending;
    ``codes``: Side codes (a decision holds until the next one). Leads are seconds before T. A lead is
    the start of the final unbroken run of the side through the last decision usable at or before
    T - end_grace_s, within the window_s analysis window (so at most ~window_s).
    """
    t_usable = np.asarray(t_usable, np.float64)
    codes = np.asarray(codes, np.int64)
    tti = T - t_usable
    m = (tti <= window_s) & (tti >= 0)
    t, s = tti[m], codes[m]
    order = np.argsort(-t, kind='stable')
    t, s = t[order], s[order]
    dur = -np.diff(np.r_[t, 0.0])
    tot = dur.sum() if len(dur) else 1.0
    frac = {SIDE_NAME[Side(k)]: float(dur[s == k].sum() / tot) if tot > 0 else 0.0 for k in range(4)}
    last1 = t <= 1.0
    tot1 = max(dur[last1].sum(), 1e-9)
    frac1 = {SIDE_NAME[Side(k)]: float(dur[last1 & (s == k)].sum() / tot1) for k in range(4)}

    def sustained(code, grace):
        keep = t >= grace
        tt, ss = t[keep], s[keep]
        if not len(tt) or ss[-1] != code:
            return 0.0
        j = len(ss) - 1
        while j > 0 and ss[j - 1] == code:
            j -= 1
        return float(tt[j])

    def first_on(code):
        hit = t[s == code]
        return float(hit.max()) if len(hit) else 0.0

    def longest_hold(code):
        # interval of decision i: [t_i, t_{i+1}] (seconds before T), clipped to [end_grace_s, window_s]
        hi = np.minimum(t, window_s)
        lo = np.maximum(np.r_[t[1:], 0.0], end_grace_s)
        d = np.maximum(hi - lo, 0.0)
        best = cur = 0.0
        for k in range(len(t)):
            if s[k] == code:
                cur += d[k]
                best = max(best, cur)
            else:
                cur = 0.0
        return float(best)

    out = dict(frac_last3=frac, frac_last1=frac1, n_decisions=int(m.sum()))
    for nm, code in (('left', Side.LEFT), ('right', Side.RIGHT)):
        out[f'sustained_{nm}'] = sustained(int(code), end_grace_s)
        out[f'sustained_{nm}_nograce'] = sustained(int(code), 0.0)
        out[f'first_{nm}'] = first_on(int(code))
        out[f'longest_hold_{nm}'] = longest_hold(int(code))
    strict, permissive = set(strict), set(permissive)
    out['correct_lead'] = max(out[f'sustained_{x}'] for x in strict)
    out['correct_lead_nograce'] = max(out[f'sustained_{x}_nograce'] for x in strict)
    out['correct_lead_permissive'] = max(out[f'sustained_{x}'] for x in permissive)
    wrong = {'left', 'right'} - strict
    wrongp = {'left', 'right'} - permissive
    out['first_wrong_lead'] = max([out[f'first_{x}'] for x in wrong] + [0.0])
    out['first_wrong_lead_permissive'] = max([out[f'first_{x}'] for x in wrongp] + [0.0])
    out['wrong_hold_max_s'] = max([out[f'longest_hold_{x}'] for x in wrong] + [0.0])
    out['wrong_hold_max_s_permissive'] = max([out[f'longest_hold_{x}'] for x in wrongp] + [0.0])
    out['success'] = bool(out['correct_lead'] >= correct_hold_s)
    out['success_permissive'] = bool(out['correct_lead_permissive'] >= correct_hold_s)
    out['wrong'] = bool(out['wrong_hold_max_s'] >= wrong_hold_s)
    out['wrong_any'] = bool(out['first_wrong_lead'] > 0)
    return out


def warning_lead(t_usable, codes, T: float, *, window_s: float = 3.0, end_grace_s: float = 0.15) -> dict:
    """E5: lead of any non-NONE decision sustained through T - grace, and the first warning lead."""
    t_usable = np.asarray(t_usable, np.float64)
    active = np.asarray(codes) != Side.NONE
    tti = T - t_usable
    m = (tti <= window_s) & (tti >= 0)
    t, a = tti[m], active[m]
    order = np.argsort(-t, kind='stable')
    t, a = t[order], a[order]
    keep = t >= end_grace_s
    tt, aa = t[keep], a[keep]
    lead = 0.0
    if len(tt) and aa[-1]:
        j = len(aa) - 1
        while j > 0 and aa[j - 1]:
            j -= 1
        lead = float(tt[j])
    first = float(t[a].max()) if a.any() else 0.0
    return dict(warning_lead=lead, first_warning_lead=first, n_decisions=int(m.sum()))


def clean_window_metrics(t_usable, codes, start: float, end: float, *, min_episode_s: float = 0.2,
                         max_dt: float = 0.25) -> dict:
    """E6 on one clean window (the lateral study's clean_metrics, plus 'any activation' incl. CENTRE)."""
    t_usable = np.asarray(t_usable, np.float64)
    codes = np.asarray(codes, np.int64)
    sel = (t_usable >= start) & (t_usable <= end)
    ph, side = t_usable[sel], codes[sel]
    n = len(ph)
    if n == 0:
        return dict(total_s=0.0, side_s=0.0, centre_s=0.0, any_s=0.0, episodes=[], any_episodes=[])
    dt = np.diff(ph)
    dt = np.r_[dt, np.median(dt) if len(dt) else 0.06]
    dt = np.minimum(dt, max_dt)
    lateral = (side == Side.LEFT) | (side == Side.RIGHT)

    def runs(active, same_code):
        eps = []
        i = 0
        while i < n:
            if active[i]:
                j = i
                while j + 1 < n and active[j + 1] and (not same_code or side[j + 1] == side[i]):
                    j += 1
                d = float(ph[j] - ph[i] + dt[j])
                if d >= min_episode_s:
                    eps.append((float(ph[i]), round(d, 3), int(side[i])))
                i = j + 1
            else:
                i += 1
        return eps

    return dict(total_s=float(dt.sum()), side_s=float(dt[lateral].sum()), centre_s=float(dt[side == Side.CENTRE].sum()),
                any_s=float(dt[side != Side.NONE].sum()), episodes=runs(lateral, True),
                any_episodes=runs(side != Side.NONE, False))


# ----------------------------------------------------------------------------- scoring helpers

@dataclass
class _Ctx:
    pred: PredictionSet
    fold: str
    thresholds_sha: str
    store_sha: str
    labels_sha: str | None
    code_commit: str | None = None

    def record(self, metric, env, value, n, **kw):
        return make_record(metric, pred=self.pred, fold=self.fold, env=env, value=value, n=n,
                           thresholds_sha256=self.thresholds_sha, store_index_sha256=self.store_sha,
                           labels_manifest_sha256=self.labels_sha, code_commit=self.code_commit, **kw)


def _ctx(pred, store, labels, thresholds, fold) -> _Ctx:
    return _Ctx(pred=pred, fold=fold or pred.fold or 'F12', thresholds_sha=thresholds_sha256(thresholds),
                store_sha=store_index_sha(store), labels_sha=labels_manifest_sha(labels))


def _scopes(fold: str, envs) -> list[tuple[str, list[str], str]]:
    """(env or 'pooled', pooled envs, variant) scopes: every environment, pooled test envs, pooled all."""
    envs = sorted(set(envs))
    test = [e for e in envs if e in FOLDS[fold].test]
    out = [(e, [e], f'env {e}') for e in envs]
    if test:
        out.append(('pooled', test, 'pooled held-out envs'))
    if len(envs) > 1:
        out.append(('pooled', envs, 'pooled all envs'))
    return out


def _timeline(pred, store, codes_by_row, run_id, latency, *, event_axis=None):
    """(t_usable, codes, axis) of a predictor's decisions on one run (rows where it ran)."""
    c = store_cache(store)
    rows = c.rows_by_run.get(int(run_id))
    if rows is None:
        return None
    pos = pred.positions(rows)
    ok = pos >= 0
    rows, pos = rows[ok], pos[ok]
    axis = event_axis or c.axis_by_run[int(run_id)]
    t = c.times(rows, axis)
    fin = np.isfinite(t)
    order = np.argsort(t[fin], kind='stable')
    return t[fin][order] + latency, codes_by_row[pos[fin]][order], axis


# ----------------------------------------------------------------------------- E1

def e1_samples(pred, store, events, thresholds, *, point: str = 'point_w', kinds=None,
               prior_filter: bool = False, field: str | None = None, neighbourhood: int | None = None) -> list[dict]:
    """Per-frame E1 samples: {event_id, env, unique_obstacle, obstacle, row, tti, true, pred}."""
    th = thresholds['E1']
    kinds = tuple(th.get('event_kinds', ['terminal_impact'])) if kinds is None else tuple(kinds)
    lo, hi = th['window_s_before_event']
    field = field or th.get('prediction', 'grid_q50')
    size = int(neighbourhood or th.get('neighbourhood_cells', 1))
    pf = th.get('prior_filter', dict(max_chord_angle_deg=15.0, range_m=[0.5, 20.0], min_speed_mps=1.0,
                                     min_optical_depth_m=0.3))
    grid = pred.arrays.get(field)
    if grid is None:
        raise ValueError(f'E1 needs {field} predictions')
    c = store_cache(store)
    out = []
    for ev in events:
        if ev.get('kind') not in kinds or not _is_scored_event(ev):
            continue
        P = _event_point(ev, point)
        rid = resolve_event_run(store, ev)
        if P is None or rid is None or rid not in c.rows_by_run:
            continue
        rows = c.rows_by_run[rid]
        axis = c.axis_by_run[rid]
        T = _event_time(ev, axis)
        if T is None:
            continue
        tti = T - c.times(rows, axis)
        rows = rows[(tti >= lo) & (tti <= hi)]
        pos = pred.positions(rows)
        rows, pos = rows[pos >= 0], pos[pos >= 0]
        if not len(rows):
            continue
        d_w, d_c, uv, ok = project_points(P, c.pos[rows], c.quat[rows])
        true = np.linalg.norm(d_w, axis=-1)
        keep = contract.in_image(uv, ok)
        if prior_filter:
            v = c.vel[rows]
            sp = np.linalg.norm(v, axis=-1)
            with np.errstate(invalid='ignore', divide='ignore'):
                cosang = np.einsum('ij,ij->i', d_w, v) / (true * sp)
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            keep &= (ang <= pf['max_chord_angle_deg']) & (true >= pf['range_m'][0]) & (true <= pf['range_m'][1])
            keep &= (sp >= pf['min_speed_mps']) & (d_c[:, 2] >= pf['min_optical_depth_m'])
        if not keep.any():
            continue
        rows, pos, uv, true = rows[keep], pos[keep], uv[keep], true[keep]
        r, cc = cell_of_uv(uv)
        pv = window_min(np.asarray(grid, np.float64)[pos], r, cc, size)
        tt = T - c.times(rows, axis)
        env = ev.get('env') or c.env_name(rows[0])
        for k in range(len(rows)):
            if np.isfinite(pv[k]):
                out.append(dict(event_id=ev['event_id'], env=env, unique_obstacle=ev.get('unique_obstacle'),
                                obstacle=ev.get('obstacle'), row=int(rows[k]), tti=float(tt[k]),
                                true=float(true[k]), pred=float(pv[k])))
    return out


def e1_summary(samples: list[dict], thresholds: dict, *, n_boot: int = 2000, seed: int = 0) -> dict:
    th = thresholds['E1']
    true = np.array([s['true'] for s in samples], np.float64)
    pred = np.array([s['pred'] for s in samples], np.float64)
    grp = np.array([str(s.get('unique_obstacle') or s['event_id']) for s in samples])
    ev = np.array([s['event_id'] for s in samples])
    ratio = pred / np.maximum(true, 1e-6)
    out = dict(n=len(true), n_events=int(len(np.unique(ev))), n_obstacles=int(len(np.unique(grp))), bins={})
    for lo, hi in th['bins_m']:
        m = (true >= lo) & (true < hi)
        key = f'{lo:g}-{hi:g}'
        if not m.any():
            out['bins'][key] = dict(n=0)
            continue
        q = ratio[m]
        out['bins'][key] = dict(n=int(m.sum()), median=float(np.median(q)), q25=float(np.percentile(q, 25)),
                                q75=float(np.percentile(q, 75)), n_events=int(len(np.unique(ev[m]))),
                                n_obstacles=int(len(np.unique(grp[m]))),
                                ci95_obstacle_bootstrap=cluster_bootstrap(q, grp[m], n_boot=n_boot, seed=seed))
    near = true < th['overshoot_true_below_m']
    out['overshoot'] = dict(n=int(near.sum()), ratio=th['overshoot_ratio'], below_m=th['overshoot_true_below_m'],
                            fraction=float((ratio[near] > th['overshoot_ratio']).mean()) if near.any() else None)
    pa = th.get('m2_pass', {}).get('pillar_approaches', {})
    rng = pa.get('range_m', [1.5, 6.0])
    mx = pa.get('max_ratio', 1.3)
    pillars = {}
    for s in samples:
        if 'pillar' in str(s.get('obstacle') or '').lower() and rng[0] <= s['true'] <= rng[1]:
            pillars.setdefault(s['event_id'], []).append(s['pred'] / s['true'])
    out['pillar_approaches'] = {str(k): dict(n=len(v), within_frac=float(np.mean([(1 / mx) <= x <= mx for x in v])),
                                             stays=bool(all((1 / mx) <= x <= mx for x in v)),
                                             max_ratio=float(max(v)), min_ratio=float(min(v)))
                                for k, v in pillars.items()}
    per_obs = {}
    for g in np.unique(grp):
        m = grp == g
        per_obs[g] = {f'{lo:g}-{hi:g}': (float(np.median(ratio[m & (true >= lo) & (true < hi)]))
                                          if (m & (true >= lo) & (true < hi)).any() else None)
                      for lo, hi in th['bins_m']}
        per_obs[g]['n'] = int(m.sum())
    out['per_obstacle'] = per_obs
    return out


def e1_m2_checks(summ: dict, per_env: dict, thresholds: dict) -> dict:
    """The M2 E1 pass criteria evaluated on a pooled held-out summary (informational at M1)."""
    mp = thresholds['E1'].get('m2_pass', {})
    checks = {}
    for key, (a, b) in mp.get('median_ratio_bins', {}).items():
        med = summ['bins'].get(key, {}).get('median')
        checks[f'median {key} in [{a}, {b}]'] = None if med is None else bool(a <= med <= b)
    lo, hi = mp.get('per_env_median_ratio', [0.75, 1.33])
    for env, s in per_env.items():
        meds = [s['bins'][k].get('median') for k in ('2-4', '4-7') if s['bins'].get(k, {}).get('median') is not None]
        checks[f'{env} median in [{lo}, {hi}]'] = None if not meds else bool(all(lo <= x <= hi for x in meds))
    ov = summ['overshoot']['fraction']
    checks['overshoot <= max'] = None if ov is None else bool(ov <= mp.get('max_overshoot_fraction', 0.15))
    pa = mp.get('pillar_approaches', {})
    stays = [v['stays'] for v in summ['pillar_approaches'].values()]
    checks['pillar approaches'] = None if not stays else bool(sum(stays) >= pa.get('min_approaches', 2))
    return checks


def e1_range_at_impacts(pred, store, labels, events, thresholds, *, fold: str | None = None,
                        point: str = 'point_w', kinds=None, prior_filter: bool = False,
                        n_boot: int = 2000) -> list[dict]:
    """E1 records: one per environment, pooled held-out and pooled all (bins, overshoot, pillars, per obstacle)."""
    ctx = _ctx(pred, store, labels, thresholds, fold)
    kinds = tuple(thresholds['E1'].get('event_kinds', ['terminal_impact'])) if kinds is None else tuple(kinds)
    samples = e1_samples(pred, store, events, thresholds, point=point, kinds=kinds, prior_filter=prior_filter)
    opts = dict(point=point, kinds=list(kinds), prior_filter=prior_filter,
                prediction=thresholds['E1'].get('prediction', 'grid_q50'),
                neighbourhood_cells=thresholds['E1'].get('neighbourhood_cells', 1))
    recs = []
    per_env = {}
    for env, envs, variant in _scopes(ctx.fold, {s['env'] for s in samples}):
        sub = [s for s in samples if s['env'] in envs]
        summ = e1_summary(sub, thresholds, n_boot=n_boot)
        if env != 'pooled':
            per_env[env] = summ
        value = dict(summ, options=opts)
        if variant == 'pooled held-out envs':
            value['m2_checks'] = e1_m2_checks(summ, {e: per_env[e] for e in envs if e in per_env}, thresholds)
        recs.append(ctx.record('E1', env, value, summ['n'], variant=variant, n_events=summ['n_events'],
                               envs=envs if env == 'pooled' else None))
    return recs


# ----------------------------------------------------------------------------- E2

def _iter_chunks(rows, size):
    rows = np.asarray(rows, np.int64)
    for a in range(0, len(rows), size):
        yield rows[a:a + size]


def e2_collider_cells(pred, store, labels, thresholds, *, fold: str | None = None, rows=None,
                      chunk: int = 4096) -> list[dict]:
    """E2 records per environment and pooled: collider-cell range accuracy and blocked-within-6 m fan scores."""
    from .labels import LabelKind, LabelSource, fan_blocked_targets
    th = thresholds['E2']
    ctx = _ctx(pred, store, labels, thresholds, fold)
    c = store_cache(store)
    rows = np.asarray(pred.rows if rows is None else rows, np.int64)
    field = th.get('prediction', 'grid_q50')
    acc: dict[str, dict] = {}
    for part in _iter_chunks(rows, chunk):
        pos = pred.positions(part)
        part, pos = part[pos >= 0], pos[pos >= 0]
        if not len(part):
            continue
        gv, gk, gs = labels.grid(part)
        fv, fk, fs = labels.fan(part)
        gcol = ((gs & int(LabelSource.COLLIDER)) > 0) & (gk == LabelKind.EXACT) & np.isfinite(gv)
        fcol = (fs & int(LabelSource.COLLIDER)) > 0
        if not gcol.any() and not fcol.any():
            continue
        pg = np.asarray(pred.arrays[field], np.float64)[pos]
        fans = prediction_fans(pred, pos, c.quat[part])
        fq = fans[th.get('fan_quantile', 'fan_q20')]
        tgt, w = fan_blocked_targets(fv, fk, th['fan_blocked_within_m'])
        yaw_ok = (np.abs(FAN_YAW_DEG) <= th['fan_max_abs_yaw_deg'])[None, None, :]
        envs = np.array([c.env_name(r) for r in part])
        for e in np.unique(envs):
            m = envs == e
            a = acc.setdefault(e, dict(true=[], pred=[], frames=0, tp=0, fp=0, fn=0, fan_n=0))
            gm = gcol[m]
            a['true'].append(np.asarray(gv[m][gm], np.float64))
            a['pred'].append(pg[m][gm])
            a['frames'] += int(gm.any(axis=(1, 2)).sum())
            sel = fcol[m] & (w[m] > 0) & yaw_ok
            pb = fq[m] <= th['fan_blocked_within_m']
            tb = tgt[m] > 0.5
            a['tp'] += int((sel & pb & tb).sum())
            a['fp'] += int((sel & pb & ~tb).sum())
            a['fn'] += int((sel & ~pb & tb).sum())
            a['fan_n'] += int(sel.sum())

    def summarise(names):
        true = np.concatenate([x for e in names for x in acc[e]['true']] or [np.zeros(0)])
        pred_ = np.concatenate([x for e in names for x in acc[e]['pred']] or [np.zeros(0)])
        ok = np.isfinite(pred_)
        true, pred_ = true[ok], pred_[ok]
        tp = sum(acc[e]['tp'] for e in names)
        fp = sum(acc[e]['fp'] for e in names)
        fn = sum(acc[e]['fn'] for e in names)
        ratio = pred_ / np.maximum(true, 1e-6)
        near = true < th['near_m']
        v = dict(n_cells=int(len(true)), n_frames=int(sum(acc[e]['frames'] for e in names)))
        if len(true):
            rel = np.abs(pred_ - true) / true
            v.update(absrel_median=float(np.median(rel)), absrel_mean=float(np.mean(rel)),
                     delta_1p25=float(np.mean(np.maximum(ratio, 1 / np.maximum(ratio, 1e-9)) < 1.25)),
                     median_ratio=float(np.median(ratio)),
                     near_cells=int(near.sum()),
                     near_overshoot_fraction=float(np.mean(ratio[near] > th['overshoot_ratio'])) if near.any() else None)
        v['fan'] = dict(n=int(sum(acc[e]['fan_n'] for e in names)), tp=tp, fp=fp, fn=fn,
                        precision=float(tp / (tp + fp)) if tp + fp else None,
                        recall=float(tp / (tp + fn)) if tp + fn else None,
                        blocked_within_m=th['fan_blocked_within_m'], max_abs_yaw_deg=th['fan_max_abs_yaw_deg'],
                        fan_derived_from_grid=bool('fan_q20' not in pred.arrays and 'fan_q50' not in pred.arrays))
        return v

    recs = []
    for env, envs, variant in _scopes(ctx.fold, acc.keys()):
        v = summarise(envs)
        recs.append(ctx.record('E2', env, v, v['n_cells'], variant=variant, envs=envs if env == 'pooled' else None))
    return recs


# ----------------------------------------------------------------------------- E3

def detect_gate_passes(t, cue_uv, cue_src=None, *, jump_min: float = 0.25, centre_u: float = 0.35,
                       centre_v: float = 0.40, max_dt: float = 0.3) -> np.ndarray:
    """Gate-pass times from logged ring-cue switches (heuristic, evaluation only).

    A pass is placed midway between two consecutive frames (<= max_dt apart) whose logged cue jumps by
    more than ``jump_min`` (normalised image units) while the earlier cue was near the image centre.
    """
    t = np.asarray(t, np.float64)
    cue = np.asarray(cue_uv, np.float64)
    ok = np.isfinite(cue).all(axis=1)
    if cue_src is not None:
        ok &= np.asarray(cue_src) == 1
    out = []
    for i in range(1, len(t)):
        if not (ok[i] and ok[i - 1]) or t[i] - t[i - 1] > max_dt:
            continue
        if np.linalg.norm(cue[i] - cue[i - 1]) > jump_min and abs(cue[i - 1, 0] - 0.5) < centre_u \
                and abs(cue[i - 1, 1] - 0.5) < centre_v:
            out.append(0.5 * (t[i] + t[i - 1]))
    return np.asarray(out, np.float64)


def _roc_block(p, y):
    return dict(n=int(len(y)), n_pos=int(np.sum(y > 0.5)), auroc=auroc(p, y > 0.5),
                ece=expected_calibration_error(p, y), brier=float(np.mean((p - y) ** 2)) if len(y) else None)


def e3_fan_quality(pred, store, labels, thresholds, *, fold: str | None = None, rows=None, gate_passes=None,
                   chunk: int = 4096) -> list[dict]:
    """E3 records per environment and pooled: calibration/ranking of P(blocked), blocked-within-6 m near the
    bearing, flown-tube false blocks and gate-opening blocks."""
    from .labels import LabelKind, LabelSource, fan_blocked_targets
    th = thresholds['E3']
    ctx = _ctx(pred, store, labels, thresholds, fold)
    c = store_cache(store)
    rows = np.asarray(pred.rows if rows is None else rows, np.int64)
    tube_q = th.get('tube_false_block_quantile', 'grid_q20')
    tube_field = tube_q if tube_q in pred.arrays else 'grid_q50'
    tol = float(th.get('tube_tolerance', 0.1))
    acc: dict[str, dict] = {}
    for part in _iter_chunks(rows, chunk):
        pos = pred.positions(part)
        part, pos = part[pos >= 0], pos[pos >= 0]
        if not len(part):
            continue
        fv, fk, _ = labels.fan(part)
        gv, gk, gs = labels.grid(part)
        q = c.quat[part]
        fans = prediction_fans(pred, pos, q)
        ref_yaw, ref_el, _, _, _ = reference_bearing(c.cue_uv[part], q, c.vel[part])
        yaw_ok = np.abs(FAN_YAW_DEG[None, :] - ref_yaw[:, None]) <= th['max_abs_yaw_from_bearing_deg'] + 1e-9
        el_ok = np.abs(FAN_ELEV_DEG[None, :] - np.clip(ref_el, FAN_ELEV_DEG[0], FAN_ELEV_DEG[-1])[:, None]) \
            <= th.get('max_abs_elev_from_bearing_deg', 10.0) + 1e-9
        near_dir = el_ok[:, :, None] & yaw_ok[:, None, :]
        t8, w8 = fan_blocked_targets(fv, fk, 8.0)
        t4, w4 = fan_blocked_targets(fv, fk, 4.0)
        t6, w6 = fan_blocked_targets(fv, fk, th['blocked_within_m'])
        pb6 = fans[th.get('fan_quantile', 'fan_q20')] <= th['blocked_within_m']
        tube = ((gs & int(LabelSource.TUBE)) > 0) & (gk == LabelKind.LOWER) & np.isfinite(gv)
        pg = np.asarray(pred.arrays[tube_field], np.float64)[pos] if tube_field in pred.arrays else None
        envs = np.array([c.env_name(r) for r in part])
        for e in np.unique(envs):
            m = envs == e
            a = acc.setdefault(e, dict(p8=[], y8=[], p4=[], y4=[], tp=0, fp=0, fn=0, n6=0, tube_n=0, tube_fb=0))
            s8 = w8[m] > 0
            a['p8'].append(fans['fan_p8'][m][s8])
            a['y8'].append(t8[m][s8])
            s4 = w4[m] > 0
            a['p4'].append(fans['fan_p4'][m][s4])
            a['y4'].append(t4[m][s4])
            s6 = (w6[m] > 0) & near_dir[m]
            tb = t6[m] > 0.5
            a['tp'] += int((s6 & pb6[m] & tb).sum())
            a['fp'] += int((s6 & pb6[m] & ~tb).sum())
            a['fn'] += int((s6 & ~pb6[m] & tb).sum())
            a['n6'] += int(s6.sum())
            if pg is not None:
                tm = tube[m]
                a['tube_n'] += int(tm.sum())
                a['tube_fb'] += int((pg[m][tm] < gv[m][tm].astype(np.float64) * (1 - tol)).sum())
    # gate openings
    gate = _gate_opening(pred, store, th, gate_passes, rows)
    recs = []
    envs_all = set(acc) | set(gate)
    for env, envs, variant in _scopes(ctx.fold, envs_all):
        a = [acc[e] for e in envs if e in acc]
        cat = lambda k: np.concatenate([x for d in a for x in d[k]] or [np.zeros(0)]).astype(np.float64)
        tp, fp, fn = (sum(d[k] for d in a) for k in ('tp', 'fp', 'fn'))
        tn, tfb = sum(d['tube_n'] for d in a), sum(d['tube_fb'] for d in a)
        g = [gate[e] for e in envs if e in gate]
        gf, gb, gp = sum(x['frames'] for x in g), sum(x['blocked'] for x in g), sum(x['passes'] for x in g)
        v = dict(p_blocked_8m=_roc_block(cat('p8'), cat('y8')), p_blocked_4m=_roc_block(cat('p4'), cat('y4')),
                 near_bearing_blocked_within=dict(
                     within_m=th['blocked_within_m'], n=sum(d['n6'] for d in a), tp=tp, fp=fp, fn=fn,
                     precision=float(tp / (tp + fp)) if tp + fp else None,
                     recall=float(tp / (tp + fn)) if tp + fn else None),
                 tube_false_block=dict(n_cells=tn, blocked=tfb, fraction=float(tfb / tn) if tn else None,
                                       field=tube_field, tolerance=tol),
                 gate_opening=dict(passes=gp, frames=gf, blocked=gb, fraction=float(gb / gf) if gf else None,
                                   passes_with_block=sum(x['passes_with_block'] for x in g),
                                   source=next((x['source'] for x in g), None)),
                 fan_derived_from_grid=bool('fan_q20' not in pred.arrays and 'fan_q50' not in pred.arrays))
        recs.append(ctx.record('E3', env, v, v['p_blocked_8m']['n'], variant=variant,
                               envs=envs if env == 'pooled' else None))
    return recs


def _gate_opening(pred, store, th, gate_passes, rows) -> dict:
    """Per environment: gate passes, frames 2.0-0.5 s before them with the pass point in view, blocked frames."""
    gcfg = th.get('gate_opening', {})
    before = gcfg.get('window_s_before_pass', [0.5, 2.0])
    ratio = float(gcfg.get('blocked_ratio', 0.9))
    field = gcfg.get('prediction', 'grid_q20')
    if field not in pred.arrays:
        field = 'grid_q50' if 'grid_q50' in pred.arrays else None
    if field is None:
        return {}
    c = store_cache(store)
    rows = np.asarray(rows, np.int64)
    runs = np.unique(c.run_id[rows]) if len(rows) else []
    out: dict[str, dict] = {}
    for rid in runs:
        rr = c.rows_by_run[int(rid)]
        axis = c.axis_by_run[int(rid)]
        t = c.times(rr, axis)
        if gate_passes is not None:
            passes = np.asarray(gate_passes.get(int(rid), []), np.float64)
            src = 'given'
        else:
            passes = detect_gate_passes(t, c.cue_uv[rr], c.cue_src[rr], **gcfg.get('detector', {}))
            src = 'detected from logged cue switches (heuristic)'
        if not len(passes):
            continue
        env = c.env_name(rr[0])
        d = out.setdefault(env, dict(passes=0, frames=0, blocked=0, passes_with_block=0, source=src))
        pos_all = pred.positions(rr)
        for tp in passes:
            j = np.searchsorted(t, tp)
            if j <= 0 or j >= len(t) or t[j] - t[j - 1] > 0.5:
                continue
            w = (tp - t[j - 1]) / max(t[j] - t[j - 1], 1e-9)
            P = (1 - w) * c.pos[rr[j - 1]] + w * c.pos[rr[j]]
            sel = np.flatnonzero((tp - t >= before[0]) & (tp - t <= before[1]) & (pos_all >= 0))
            if not len(sel):
                continue
            r = rr[sel]
            d_w, _, uv, ok = project_points(P, c.pos[r], c.quat[r])
            vis = contract.in_image(uv, ok)
            if not vis.any():
                continue
            gr, gc = cell_of_uv(uv[vis])
            pv = np.asarray(pred.arrays[field], np.float64)[pos_all[sel][vis], gr, gc]
            dist = np.linalg.norm(d_w[vis], axis=-1)
            blk = pv < ratio * dist
            d['passes'] += 1
            d['frames'] += int(vis.sum())
            d['blocked'] += int(blk.sum())
            d['passes_with_block'] += int(blk.any())
    return out


# ----------------------------------------------------------------------------- E4 / E5 / E6

def _side_codes(pred, store, thresholds, latency_s, rate_hz):
    return decide_side(pred, store, thresholds, latency_s=latency_s, rate_hz=rate_hz)


def e4_event_results(pred, store, events, thresholds, *, latency_s: float = DEPLOYED_LATENCY_S,
                     rate_hz: float | None = DEPLOYED_RATE_HZ, codes=None, kinds=None) -> list[dict]:
    """Per-event lateral results (dicts with event metadata and lateral_event_metrics)."""
    th = thresholds['E4']
    kinds = tuple(th.get('event_kinds', ['terminal_impact'])) if kinds is None else tuple(kinds)
    codes = _side_codes(pred, store, thresholds, latency_s, rate_hz) if codes is None else codes
    out = []
    for ev in events:
        if not ev.get('lateral') or ev.get('kind') not in kinds or not _is_scored_event(ev):
            continue
        sides = _free_sides(ev)
        rid = resolve_event_run(store, ev)
        if sides is None or rid is None:
            continue
        tl = _timeline(pred, store, codes, rid, latency_s)
        if tl is None:
            continue
        t_u, cd, axis = tl
        T = _event_time(ev, axis)
        if T is None:
            continue
        m = lateral_event_metrics(t_u, cd, T, strict=sides[0], permissive=sides[1], window_s=th['window_s'],
                                  end_grace_s=th['end_grace_s'], correct_hold_s=th['correct_hold_s'],
                                  wrong_hold_s=th['wrong_hold_s'])
        out.append(dict(event_id=ev['event_id'], run=ev.get('run'), env=ev.get('env'), obstacle=ev.get('obstacle'),
                        unique_obstacle=ev.get('unique_obstacle'), primary_free_side=ev.get('primary_free_side'),
                        blind=ev.get('blind'), **m))
    return out


def _e4_summary(res: list[dict]) -> dict:
    n = len(res)
    k = sum(r['success'] for r in res)
    kp = sum(r['success_permissive'] for r in res)
    pill = [r for r in res if 'pillar' in str(r.get('obstacle') or '').lower()]
    return dict(n=n, correct=k, correct_ci95=wilson_interval(k, n), correct_permissive=kp,
                correct_ge_0p5s=sum(r['correct_lead'] >= 0.5 for r in res),
                wrong_held=sum(r['wrong'] for r in res), wrong_any=sum(r['wrong_any'] for r in res),
                pillars=dict(n=len(pill), correct=sum(r['success'] for r in pill),
                             wrong_held=sum(r['wrong'] for r in pill)),
                events={str(r['event_id']): dict(run=r['run'], lead=round(r['correct_lead'], 3),
                                                 lead_permissive=round(r['correct_lead_permissive'], 3),
                                                 first_wrong=round(r['first_wrong_lead'], 3),
                                                 wrong_hold=round(r['wrong_hold_max_s'], 3),
                                                 success=r['success'], frac_last3=r['frac_last3'])
                        for r in res})


def e4_lateral_replay(pred, store, events, thresholds, *, latency_s: float = DEPLOYED_LATENCY_S,
                      fold: str | None = None, rate_hz: float | None = DEPLOYED_RATE_HZ, labels=None,
                      codes=None) -> list[dict]:
    """E4 records: per environment and pooled (counts, Wilson CI, pillars, per-event leads)."""
    ctx = _ctx(pred, store, labels, thresholds, fold)
    res = e4_event_results(pred, store, events, thresholds, latency_s=latency_s, rate_hz=rate_hz, codes=codes)
    recs = []
    for env, envs, variant in _scopes(ctx.fold, {r['env'] for r in res}):
        sub = [r for r in res if r['env'] in envs]
        v = _e4_summary(sub)
        recs.append(ctx.record('E4', env, v, v['n'], variant=f'{variant} @ {latency_s * 1e3:.0f} ms',
                               n_events=v['n'], latency_s=latency_s, ci95=v['correct_ci95'],
                               envs=envs if env == 'pooled' else None,
                               extra=dict(per_event=[dict(r) for r in sub])))
    return recs


def e5_out_of_view(pred, store, events, thresholds, *, latency_s: float = DEPLOYED_LATENCY_S,
                   fold: str | None = None, rate_hz: float | None = DEPLOYED_RATE_HZ, labels=None,
                   point: str = 'point_w', codes=None) -> list[dict]:
    """E5 records: out-of-view / vertical / unknown-side events with in-view fraction and warning leads."""
    th = thresholds['E5']
    ctx = _ctx(pred, store, labels, thresholds, fold)
    c = store_cache(store)
    codes = _side_codes(pred, store, thresholds, latency_s, rate_hz) if codes is None else codes
    w0, w1 = th['window_s_before_event']
    lead_need = thresholds['E4']['correct_hold_s']
    res = []
    for ev in events:
        if ev.get('kind') not in ('terminal_impact', 'contact') or not _is_scored_event(ev):
            continue
        rid = resolve_event_run(store, ev)
        if rid is None or rid not in c.rows_by_run:
            continue
        rows = c.rows_by_run[rid]
        axis = c.axis_by_run[rid]
        T = _event_time(ev, axis)
        if T is None:
            continue
        tti = T - c.times(rows, axis)
        P = _event_point(ev, point)
        inview = None
        if P is not None:
            r = rows[(tti >= w0) & (tti <= w1)]
            if len(r):
                _, _, uv, ok = project_points(P, c.pos[r], c.quat[r])
                inview = float(contract.in_image(uv, ok).mean())
        vertical = ev.get('primary_free_side') in ('up', 'down', 'unknown', 'none') or _free_sides(ev) is None
        out_of_view = inview is None or (1.0 - inview) > th['out_of_view_fraction']
        if not (vertical or out_of_view):
            continue
        tl = _timeline(pred, store, codes, rid, latency_s)
        wl = warning_lead(tl[0], tl[1], T, window_s=thresholds['E4']['window_s'],
                          end_grace_s=thresholds['E4']['end_grace_s']) if tl is not None else {}
        res.append(dict(event_id=ev['event_id'], run=ev.get('run'), env=ev.get('env'), kind=ev.get('kind'),
                        obstacle=ev.get('obstacle'), unique_obstacle=ev.get('unique_obstacle'),
                        primary_free_side=ev.get('primary_free_side'), in_view_frac_T2_T1=inview,
                        reason='out of view' if out_of_view else 'vertical/unknown side', **wl))
    recs = []
    for env, envs, variant in _scopes(ctx.fold, {r['env'] for r in res}):
        sub = [r for r in res if r['env'] in envs]
        v = dict(n=len(sub), warned_ge_lead=sum(r.get('warning_lead', 0) >= lead_need for r in sub),
                 lead_required_s=lead_need,
                 events={str(r['event_id']): {k: r[k] for k in ('run', 'kind', 'obstacle', 'primary_free_side',
                                                                'in_view_frac_T2_T1', 'reason', 'warning_lead',
                                                                'first_warning_lead') if k in r} for r in sub})
        recs.append(ctx.record('E5', env, v, v['n'], variant=f'{variant} @ {latency_s * 1e3:.0f} ms',
                               n_events=v['n'], latency_s=latency_s, envs=envs if env == 'pooled' else None))
    return recs


def e6_clean_windows(pred, store, thresholds, *, latency_s: float = DEPLOYED_LATENCY_S, fold: str | None = None,
                     rate_hz: float | None = DEPLOYED_RATE_HZ, windows=None, events=None, labels=None,
                     codes=None) -> list[dict]:
    """E6 records per environment and pooled: side/any activation episodes per minute, active fractions;
    plus near-pass records (events kind near_pass and the thresholds' PD pillar near pass)."""
    th = thresholds['E6']
    ctx = _ctx(pred, store, labels, thresholds, fold)
    c = store_cache(store)
    codes = _side_codes(pred, store, thresholds, latency_s, rate_hz) if codes is None else codes
    windows = clean_windows(store) if windows is None else windows
    per = []
    for w in windows:
        rid = int(w['run_id'])
        tl = _timeline(pred, store, codes, rid, latency_s)
        if tl is None or rid not in c.rows_by_run:
            continue
        t_u, cd, axis = tl
        a, b = float(w['start_phase_s']), float(w['end_phase_s'])
        if axis == 'wall':
            if w.get('start_wall_s') is None:
                continue
            a, b = float(w['start_wall_s']), float(w['end_wall_s'])
        m = clean_window_metrics(t_u, cd, a, b, min_episode_s=th['min_episode_s'])
        if m['total_s'] <= 0:
            continue
        per.append(dict(run_id=rid, alias=w.get('alias') or w.get('run'), start=a, end=b,
                        env=c.env_name(c.rows_by_run[rid][0]), **m))
    recs = []
    mp = th.get('m3_pass', {})
    for env, envs, variant in _scopes(ctx.fold, {p['env'] for p in per}):
        sub = [p for p in per if p['env'] in envs]
        tot = sum(p['total_s'] for p in sub)
        eps = sum(len(p['episodes']) for p in sub)
        aeps = sum(len(p['any_episodes']) for p in sub)
        v = dict(total_min=tot / 60.0, windows=len(sub), side_episodes=eps,
                 side_episodes_per_min=eps / (tot / 60.0) if tot else None,
                 any_episodes=aeps, any_episodes_per_min=aeps / (tot / 60.0) if tot else None,
                 side_active_fraction=sum(p['side_s'] for p in sub) / tot if tot else None,
                 centre_fraction=sum(p['centre_s'] for p in sub) / tot if tot else None,
                 any_active_fraction=sum(p['any_s'] for p in sub) / tot if tot else None,
                 episode_durations_s=sorted(e[1] for p in sub for e in p['episodes']),
                 per_window={f"{p['alias']}[{p['start']:.1f}-{p['end']:.1f}]": dict(
                     total_s=round(p['total_s'], 2), episodes=p['episodes'], any_episodes=p['any_episodes'])
                     for p in sub})
        if tot:
            lim = mp.get('max_per_min_env' if env != 'pooled' else 'max_per_min_overall')
            v['m3_checks'] = {
                'any episodes/min <= limit': bool(v['any_episodes_per_min'] <= lim) if lim is not None else None,
                'any active fraction <= max': bool(v['any_active_fraction'] <= mp.get('max_active_fraction', 1.0))}
        recs.append(ctx.record('E6', env, v, len(sub), variant=f'{variant} @ {latency_s * 1e3:.0f} ms',
                               latency_s=latency_s, envs=envs if env == 'pooled' else None))
    recs += _near_passes(ctx, pred, store, thresholds, codes, latency_s, events or [])
    return recs


def _near_passes(ctx, pred, store, thresholds, codes, latency_s, events) -> list[dict]:
    th = thresholds['E6']
    items = []
    for ev in events:
        if ev.get('kind') == 'near_pass' and _is_scored_event(ev):
            side = str(ev.get('obstacle_side') or '').lower()
            toward = 'right' if side.startswith('right') else 'left' if side.startswith('left') else None
            rid = resolve_event_run(store, ev)
            if toward and rid is not None:
                t0 = float(ev['t_phase'])
                pre, post = th.get('near_pass_window_s', [1.4, 0.4])
                items.append((f"event {ev['event_id']}", ev.get('env'), rid, toward, [t0 - pre, t0 + post]))
    pp = th.get('pd_pillar_near_pass')
    if pp:
        rid = resolve_event_run(store, dict(run=pp['run'], flight=pp.get('flight')))
        if rid is not None:
            toward = 'right' if 'right' in pp.get('must_not', '') else 'left'
            a, b = pp['phase_s']
            items.append((f"pd_pillar_near_pass {pp['run']}", store_cache(store).env_name(
                store_cache(store).rows_by_run[rid][0]), rid, toward, [a - float(pp.get('pre_s', 1.0)), b]))
    recs = []
    for name, env, rid, toward, (a, b) in items:
        tl = _timeline(pred, store, codes, rid, latency_s)
        if tl is None:
            continue
        m = clean_window_metrics(tl[0], tl[1], a, b, min_episode_s=th['min_episode_s'])
        code = int(SIDE_OF_NAME[toward])
        toward_eps = [e for e in m['episodes'] if e[2] == code]
        sel = (tl[0] >= a) & (tl[0] <= b)
        v = dict(window=[a, b], toward=toward, toward_fraction=float(np.mean(tl[1][sel] == code)) if sel.any() else None,
                 toward_longest_s=max([e[1] for e in toward_eps] + [0.0]), episodes=m['episodes'])
        recs.append(ctx.record('E6', env or 'pooled', v, int(sel.sum()), variant=f'near pass {name}',
                               latency_s=latency_s, passed=not toward_eps))
    return recs


# ----------------------------------------------------------------------------- E7

def e7_unique_obstacles(records: list[dict], events, *, z: float = 1.96, n_boot: int = 2000,
                        seed: int = 0) -> list[dict]:
    """Per-unique-obstacle success (E4 lateral events) with Wilson intervals, pooled with Wilson and an
    obstacle-cluster bootstrap. ``records``: E4 records (their per_event lists)."""
    ev_by_id = {e['event_id']: e for e in events}
    out = []
    for rec in records:
        if rec.get('metric') != 'E4' or 'per_event' not in rec:
            continue
        res = rec['per_event']
        if not res:
            continue
        groups = {}
        for r in res:
            key = r.get('unique_obstacle') or (ev_by_id.get(r['event_id'], {}).get('unique_obstacle')) \
                or f"event-{r['event_id']}"
            groups.setdefault(key, []).append(bool(r['success']))
        per = {k: dict(n=len(v), correct=int(sum(v)), ci95=wilson_interval(int(sum(v)), len(v), z))
               for k, v in groups.items()}
        vals = np.array([float(r['success']) for r in res])
        grp = np.array([r.get('unique_obstacle') or (ev_by_id.get(r['event_id'], {}).get('unique_obstacle'))
                        or f"event-{r['event_id']}" for r in res])
        k, n = int(vals.sum()), len(vals)
        value = dict(per_obstacle=per, n_events=n, n_obstacles=len(groups), correct=k,
                     event_rate=k / n, event_ci95_wilson=wilson_interval(k, n, z),
                     event_ci95_obstacle_bootstrap=cluster_bootstrap(vals, grp, np.mean, n_boot=n_boot, seed=seed),
                     obstacles_all_correct=int(sum(v['correct'] == v['n'] for v in per.values())),
                     obstacles_any_correct=int(sum(v['correct'] > 0 for v in per.values())))
        r2 = {k2: rec[k2] for k2 in rec if k2 not in ('per_event', 'value', 'metric', 'variant', 'n', 'n_events',
                                                        'ci95', 'passed', 'created')}
        r2.update(metric='E7', variant=f"unique obstacles ({rec.get('variant')})", value=value, n=len(groups),
                  n_events=n, ci95=value['event_ci95_obstacle_bootstrap'], passed=None,
                  created=_dt.datetime.now().isoformat(timespec='seconds'))
        out.append(r2)
    return out


# ----------------------------------------------------------------------------- E8

def e8_leak_tests(predict_fn, store, rows, thresholds, *, labels=None, pred: PredictionSet | None = None,
                  fold: str | None = None, seed: int = 0, batch: int = 16, guard=None,
                  ring_mask_fn=None) -> list[dict]:
    """E8 records: median relative change of predicted range at test cells under each perturbation.

    ``predict_fn(frames uint8 (n, 252, 448, 3), quat (n, 4)) -> dict with grid_q50 (n, 18, 32)``.
    Ring-on-obstacle/free-space test cells come from the frame labels when given (EXACT/UPPER <= 8 m /
    LOWER >= 20 m cells), else from the predictor's own unperturbed output (near/far cells).
    """
    from . import leaks
    th = thresholds['E8']
    pred = pred or PredictionSet('predict_fn', 'model', True, np.asarray(rows, np.int64))
    ctx = _ctx(pred, store, labels, thresholds, fold)
    c = store_cache(store)
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows, np.int64)
    changes: dict[str, list] = {}
    for part in _iter_chunks(rows, batch):
        if guard is not None:
            guard.before_chunk()
        frames = store.frames(part)
        quat = c.quat[part]
        base = np.asarray(predict_fn(frames, quat)['grid_q50'], np.float64)
        lab = labels.grid(part) if labels is not None else None
        for name, (pert, cells) in leaks.perturb_batch(frames, base, lab, rng, ring_mask_fn=ring_mask_fn).items():
            if pert is None:
                changes.setdefault(name, [])
                continue
            out = np.asarray(predict_fn(pert, quat)['grid_q50'], np.float64)
            rel = np.abs(out - base) / np.maximum(base, 1e-6)
            changes.setdefault(name, []).append(rel[cells])
    recs = []
    for name, parts in changes.items():
        vals = np.concatenate(parts) if parts else np.zeros(0)
        vals = vals[np.isfinite(vals)]
        med = float(np.median(vals)) if len(vals) else None
        v = dict(perturbation=name, n_cells=int(len(vals)), median_relative_change=med,
                 p90_relative_change=float(np.percentile(vals, 90)) if len(vals) else None,
                 max_allowed=th['max_median_relative_change'], frames=int(len(rows)))
        recs.append(ctx.record('E8', 'pooled', v, len(vals), variant=f'leak {name}',
                               passed=None if med is None else bool(med < th['max_median_relative_change']),
                               envs=sorted({c.env_name(r) for r in rows})))
    return recs


# ----------------------------------------------------------------------------- orchestration

def score(pred: PredictionSet, store, labels, events, thresholds, *, fold: str | None = None,
          metrics=('E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7'), latencies=None, rate_hz: float | None = None,
          windows=None, gate_passes=None, log=None) -> list[dict]:
    """Run the requested metrics for one predictor; E4-E6 at the deployed and stress latencies."""
    fold = fold or pred.fold or 'F12'
    timing = thresholds.get('timing', {})
    rate_hz = timing.get('rate_hz', DEPLOYED_RATE_HZ) if rate_hz is None else rate_hz
    if latencies is None:
        latencies = [timing.get('latency_s', DEPLOYED_LATENCY_S)] + list(timing.get('stress_latency_s',
                                                                                    STRESS_LATENCY_S))
    has_grid = 'grid_q50' in pred.arrays
    recs = []
    say = log or (lambda m: None)
    if 'E1' in metrics and has_grid:
        say('E1')
        recs += e1_range_at_impacts(pred, store, labels, events, thresholds, fold=fold)
    if 'E2' in metrics and has_grid and labels is not None:
        say('E2')
        recs += e2_collider_cells(pred, store, labels, thresholds, fold=fold)
    if 'E3' in metrics and has_grid and labels is not None:
        say('E3')
        recs += e3_fan_quality(pred, store, labels, thresholds, fold=fold, gate_passes=gate_passes)
    for lat in latencies:
        codes = None
        if any(m in metrics for m in ('E4', 'E5', 'E6')):
            say(f'decisions @ {lat * 1e3:.0f} ms')
            codes = decide_side(pred, store, thresholds, latency_s=lat, rate_hz=rate_hz)
        if 'E4' in metrics:
            e4 = e4_lateral_replay(pred, store, events, thresholds, latency_s=lat, fold=fold, rate_hz=rate_hz,
                                   labels=labels, codes=codes)
            recs += e4
            if 'E7' in metrics and abs(lat - latencies[0]) < 1e-9:
                recs += e7_unique_obstacles(e4, events)
        if 'E5' in metrics:
            recs += e5_out_of_view(pred, store, events, thresholds, latency_s=lat, fold=fold, rate_hz=rate_hz,
                                   labels=labels, codes=codes)
        if 'E6' in metrics:
            recs += e6_clean_windows(pred, store, thresholds, latency_s=lat, fold=fold, rate_hz=rate_hz,
                                     windows=windows, events=events, labels=labels, codes=codes)
    return recs


def any_held_out(records: list[dict]) -> bool:
    return any(r.get('held_out') is True or r.get('sealed') is True for r in records)


# ----------------------------------------------------------------------------- baselines

def baseline_sha256(baseline_id: str, config: dict, files=()) -> str:
    body = dict(baseline_id=baseline_id, config=config,
                files={str(Path(f).name): _sha256_file(f) for f in files})
    return hashlib.sha256(canonical_json(body)).hexdigest()


def baseline_b0(store, rows, side: Side) -> PredictionSet:
    side = Side(side)
    if side not in (Side.LEFT, Side.RIGHT):
        raise ValueError('B0 is constant LEFT or constant RIGHT')
    rows = np.asarray(rows, np.int64)
    return PredictionSet(f'B0-{side.name.lower()}', 'baseline', True, rows, baseline_id='B0',
                         sha256=baseline_sha256('B0', dict(side=side.name)),
                         arrays=dict(side=np.full(len(rows), int(side), np.int8))).validate()


def baseline_b1(store, rows, *, guard=None, config: dict | None = None, frame_source=None, chunk: int = 256,
                max_gap_s: float = 0.25, latency_s: float = DEPLOYED_LATENCY_S, log=None) -> PredictionSet:
    """Split looming (lateral study, selected config) streamed over contiguous segments of each run.

    ``frame_source(rows) -> list of (image, usable-or-None)``; default = store frames (448 x 252 RGB; the
    estimator resizes to 240 x 135 and applies its own HUD mask). Side codes: LEFT/RIGHT = steer toward
    (the study's ``side``; its 'centre' state is not a side and scores as NONE).
    """
    from . import split_looming as S
    cfg = dict(S.SELECTED_CONFIG if config is None else config)
    c = store_cache(store)
    rows = np.unique(np.asarray(rows, np.int64))
    side = np.zeros(len(rows), np.int8)
    if frame_source is None:
        def frame_source(rr):
            return [(f, None) for f in store.frames(rr)]
    run_of = c.run_id[rows]
    for rid in np.unique(run_of):
        sel = np.flatnonzero(run_of == rid)
        t = c.t_wall[rows[sel]]
        order = np.argsort(t, kind='stable')
        sel, t = sel[order], t[order]
        for a, b in segments(t, max_gap_s):
            est = S.SplitLoomingEstimator(S.SplitLoomingConfig(**cfg))
            seg = sel[a:b]
            for k0 in range(0, len(seg), chunk):
                if guard is not None:
                    guard.before_chunk()
                part = seg[k0:k0 + chunk]
                imgs = frame_source(rows[part])
                for j, (img, usable) in zip(part, imgs):
                    r = rows[j]
                    o = est.update(img, c.t_wall[r], c.quat[r], c.vel[r], usable=usable)
                    if o is not None:
                        side[j] = S.SIDE_CODE[o['side']]
        if log:
            log(f'B1 run {int(rid)}: {len(sel)} frames')
    src = Path(S.__file__)
    return PredictionSet('B1-split-looming', 'baseline', True, rows, baseline_id='B1',
                         sha256=baseline_sha256('B1', cfg, [src]), latency_s=latency_s,
                         arrays=dict(side=side)).validate()


def baseline_b2(store, rows, guard, *, runner=None, batch: int = 16, cache_dir=None, mask_fn=None,
                log=None) -> PredictionSet:
    """Pretrained DA-V2 metric-indoor at 252 x 448 (fp16) -> range grid (min per 14 x 14 cell), q20 = q50."""
    from . import baselines as BL
    runner = runner or BL.PretrainedDepth('metric')
    grid = BL.run_grid(runner, store, rows, guard, batch=batch, cache_dir=cache_dir, mask_fn=mask_fn, log=log,
                       reduce='min_range')
    rows = np.unique(np.asarray(rows, np.int64))
    return PredictionSet('B2-dav2-metric-indoor-252x448', 'baseline', True, rows, baseline_id='B2',
                         sha256=baseline_sha256('B2', dict(runner.config(), overlay_masked=mask_fn is not None)),
                         arrays=dict(grid_q50=grid, grid_q20=grid.copy())).validate()


def baseline_b3(store, rows, fold: str, guard, *, labels, calibration=None, disparity=None, runner=None,
                batch: int = 16, cache_dir=None, log=None) -> PredictionSet:
    """Relative DA-V2-Small disparity (labels teacher cache, else a run) + one monotone log-range map fitted
    on the fold's TRAINING environments only (``baselines.fit_b3_calibration``)."""
    from . import baselines as BL
    rows = np.unique(np.asarray(rows, np.int64))
    if calibration is None:
        calibration = BL.fit_b3_calibration(store, labels, fold, guard=guard, log=log)
    if calibration.fold != fold:
        raise ValueError(f'B3 calibration was fitted for {calibration.fold}, not {fold}')
    disp = BL.cell_disparity(store, rows, labels=labels, runner=runner, guard=guard, batch=batch,
                             cache_dir=cache_dir, log=log) if disparity is None else disparity
    grid = calibration(disp).astype(np.float32)
    return PredictionSet(f'B3-relative-monotone-{fold}', 'baseline', True, rows, fold=fold, baseline_id='B3',
                         sha256=baseline_sha256('B3', dict(calibration=calibration.sha256(),
                                                           disparity=BL.DISPARITY_SOURCE)),
                         arrays=dict(grid_q50=grid, grid_q20=grid.copy())).validate()


def baseline_b4(store, rows, labels, guard, *, disparity=None, runner=None, batch: int = 16, cache_dir=None,
                min_anchors: int = 8, log=None) -> PredictionSet:
    """NON-CAUSAL ceiling: per-frame affine fit of inverse range to relative disparity on the frame's own
    EXACT label cells; frames with fewer than ``min_anchors`` anchors get NaN (no prediction)."""
    from . import baselines as BL
    rows = np.unique(np.asarray(rows, np.int64))
    disp = BL.cell_disparity(store, rows, labels=labels, runner=runner, guard=guard, batch=batch,
                             cache_dir=cache_dir, log=log) if disparity is None else disparity
    gv, gk, _ = labels.grid(rows)
    grid = BL.per_frame_affine(disp, gv, gk, min_anchors=min_anchors).astype(np.float32)
    return PredictionSet('B4-per-frame-affine-ceiling', 'baseline', False, rows, baseline_id='B4',
                         sha256=baseline_sha256('B4', dict(min_anchors=min_anchors, disparity=BL.DISPARITY_SOURCE)),
                         arrays=dict(grid_q50=grid, grid_q20=grid.copy())).validate()


# ----------------------------------------------------------------------------- CLI

def _evaluation_rows(store, events, fold: str, *, windows=None, pre_s: float = 6.5,
                     include_test: bool = True) -> np.ndarray:
    """Rows any metric reads for one fold: every event window (pre_s before each event), every clean window
    and (``include_test``) the fold's test rows for E2/E3; the sealed part is never loaded here."""
    c = store_cache(store)
    keep = np.zeros(c.n, bool)
    for ev in events:
        rid = resolve_event_run(store, ev)
        if rid is None or rid not in c.rows_by_run:
            continue
        rr = c.rows_by_run[rid]
        axis = c.axis_by_run[rid]
        T = _event_time(ev, axis)
        if T is None:
            continue
        tti = T - c.times(rr, axis)
        keep[rr[(tti >= -0.2) & (tti <= pre_s)]] = True
    for w in (clean_windows(store) if windows is None else windows):
        rr = c.rows_by_run.get(int(w['run_id']))
        if rr is None:
            continue
        t = c.times(rr, 'phase')
        keep[rr[(t >= float(w['start_phase_s']) - 1.5) & (t <= float(w['end_phase_s']))]] = True
    if include_test and hasattr(store, 'rows'):
        keep[store.rows(fold=fold, side='test')] = True
    return np.flatnonzero(keep).astype(np.int64)


def _write_report(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default) + '\n', encoding='utf-8')
    tmp.replace(path)


def _load_eval_inputs(args):
    from .store import FrameStore
    store = FrameStore(args.store)
    labels = None
    events = []
    try:
        from .labels import LabelSet
        labels = LabelSet(args.store, store_index_sha(store))
        events = labels.events()
    except FileNotFoundError:
        pass
    if args.events:
        from .labels import load_events
        extra = load_events(args.events)
        known = {e['event_id'] for e in events}
        events += [e for e in extra if e['event_id'] not in known]
    return store, labels, events


def _finish(args, pred, recs, thresholds_sha, out_dir: Path):
    for r in recs:
        r['code_commit'] = r.get('code_commit') or _code_commit()
    _write_report(out_dir / f'{pred.name}_{args.fold}_{thresholds_sha[:12]}.json', dict(records=recs))
    held = any_held_out(recs)
    if held:
        Ledger(args.ledger).append(pred.sha256, args.fold, thresholds_sha, recs, once=args.once,
                                   note=f'{pred.kind} {pred.name}')
    print(f'{pred.name}: {len(recs)} records ({"held-out, ledgered" if held else "no held-out records"})')


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m haltere.obstacles.evaluate')
    sub = p.add_subparsers(dest='cmd', required=True)
    fz = sub.add_parser('freeze', help='freeze configs/obstacles/thresholds.json and print its sha256')
    fz.add_argument('--thresholds', type=Path, default=THRESHOLDS_PATH)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--store', type=Path, default=REPO_ROOT / 'runs' / 'obstacle-store-v1')
    common.add_argument('--events', type=Path, default=None, help='extra events (e.g. configs/obstacles/events_f12.json)')
    common.add_argument('--thresholds', type=Path, default=THRESHOLDS_PATH)
    common.add_argument('--ledger', type=Path, default=LEDGER_PATH)
    common.add_argument('--out', type=Path, default=REPO_ROOT / 'runs' / 'obstacle-train' / 'eval')
    common.add_argument('--fold', default='F12')
    common.add_argument('--once', action='store_true', help='refuse a second held-out score (default policy)')
    common.add_argument('--flight-lock', default=None)
    common.add_argument('--metrics', default='E1,E2,E3,E4,E5,E6,E7')
    common.add_argument('--leaks', type=int, default=0, help='E8 on this many held-out frames (model and B2)')
    b = sub.add_parser('baselines', parents=[common], help='score baselines B0-B4')
    b.add_argument('--set', default='B0,B1,B2,B3,B4')
    b.add_argument('--gpu-batch', type=int, default=16)
    m = sub.add_parser('model', parents=[common], help='score a frozen model directory once on its fold')
    m.add_argument('model_dir', type=Path)
    m.add_argument('--sealed-final', action='store_true')
    m.add_argument('--gpu-batch', type=int, default=16)
    s = sub.add_parser('score', parents=[common], help='score a saved PredictionSet (.npz)')
    s.add_argument('pred', type=Path)
    r = sub.add_parser('reproduce-lateral', help='reproduce the lateral study numbers on its caches (B0/B1/B2)')
    r.add_argument('--lateral-data', type=Path, required=True, help='scratchpad lateral/data directory')
    r.add_argument('--depth-cache', type=Path, default=None, help='lateral cand-mono-depth/depth (prior B2 maps)')
    r.add_argument('--b2-pred', type=Path, default=None, help='B2 PredictionSet over the adapter rows (252x448)')
    r.add_argument('--b1', action='store_true', help='stream the split-looming estimator (about 2-3 min CPU)')
    r.add_argument('--thresholds', type=Path, default=THRESHOLDS_PATH)
    r.add_argument('--out', type=Path, default=REPO_ROOT / 'runs' / 'obstacle-train' / 'reproduction' / 'lateral.json')
    r.add_argument('--flight-lock', default=None)
    args = p.parse_args(argv)
    if args.cmd == 'freeze':
        obj = freeze_thresholds(args.thresholds)
        print(f'{args.thresholds}: frozen {obj["frozen_at"]} sha256 {obj["sha256"]}')
        return
    from . import thermal
    lock = thermal.require_flight_lock_path(args.flight_lock)
    thermal.limit_threads(2, cv2=True)
    if args.cmd == 'reproduce-lateral':
        from .lateral_cache import reproduce
        thresholds, _ = load_thresholds(args.thresholds, require_frozen=False)
        res = reproduce(args.lateral_data, thresholds, depth_cache=args.depth_cache, b2_pred=args.b2_pred,
                        b1=args.b1, guard=thermal.ChunkGuard(lock), log=print)
        _write_report(args.out, res)
        print(json.dumps(res['summary'], indent=1, default=_json_default))
        return
    thresholds, tsha = load_thresholds(args.thresholds, require_frozen=True)
    store, labels, events = _load_eval_inputs(args)
    metrics = tuple(x.strip() for x in args.metrics.split(',') if x.strip())
    out_dir = Path(args.out)
    if args.cmd == 'score':
        pred = PredictionSet.load(args.pred).validate(len(store))
        recs = score(pred, store, labels, events, thresholds, fold=args.fold, metrics=metrics, log=print)
        _finish(args, pred, recs, tsha, out_dir)
        return
    rows = _evaluation_rows(store, events, args.fold)
    from . import baselines as BL
    if args.cmd == 'baselines':
        cpu = thermal.ChunkGuard(lock)
        gpu = thermal.ChunkGuard(lock, gpu=True)
        dense = _evaluation_rows(store, events, args.fold, include_test=False)   # side-only baselines
        for bid in [x.strip() for x in args.set.split(',') if x.strip()]:
            leak_fn = None
            if bid == 'B0':
                preds = [baseline_b0(store, dense, Side.RIGHT), baseline_b0(store, dense, Side.LEFT)]
            elif bid == 'B1':
                preds = [baseline_b1(store, dense, guard=cpu, log=print)]
            elif bid == 'B2':
                gpu.before_chunk()
                runner = BL.PretrainedDepth('metric')
                masks = BL.overlay_mask_fn()          # ghost trails / HUD / ring stroke never reach the input
                preds = [baseline_b2(store, rows, gpu, runner=runner, batch=args.gpu_batch, mask_fn=masks,
                                     cache_dir=out_dir / 'cache' / 'B2', log=print)]
                leak_fn = BL.depth_predict_fn(runner, masks)
            elif bid == 'B3':
                cal = BL.fit_b3_calibration(store, labels, args.fold, guard=gpu, log=print,
                                            cache_dir=out_dir / 'cache' / 'B3',
                                            out_path=out_dir / f'B3_calibration_{args.fold}.json')
                preds = [baseline_b3(store, rows, args.fold, gpu, labels=labels, calibration=cal,
                                     batch=args.gpu_batch, cache_dir=out_dir / 'cache' / 'B3', log=print)]
            elif bid == 'B4':
                preds = [baseline_b4(store, rows, labels, gpu, batch=args.gpu_batch,
                                     cache_dir=out_dir / 'cache' / 'B3', log=print)]
            else:
                raise SystemExit(f'unknown baseline {bid}')
            for pred in preds:
                pred.fold = pred.fold or args.fold
                pred.save(out_dir / f'{pred.name}_{args.fold}.npz')
                recs = score(pred, store, labels, events, thresholds, fold=args.fold, metrics=metrics, log=print)
                if leak_fn is not None and args.leaks > 0:
                    recs += e8_leak_tests(leak_fn, store, _leak_rows(store, args.fold, args.leaks), thresholds,
                                          labels=labels, pred=pred, fold=args.fold, guard=gpu,
                                          ring_mask_fn=BL.ring_mask_fn())
                _finish(args, pred, recs, tsha, out_dir)
        return
    if args.cmd == 'model':
        if args.sealed_final:
            raise SystemExit('--sealed-final scoring is M7 only: load the sealed store part explicitly there')
        gpu = thermal.ChunkGuard(lock, gpu=True)
        gpu.before_chunk()
        net, card = BL.load_model(args.model_dir)
        pred = BL.model_predictions(args.model_dir, store, rows, gpu, batch=args.gpu_batch, net=net, card=card,
                                    log=print)
        if pred.fold != args.fold:
            raise SystemExit(f'model fold {pred.fold} != --fold {args.fold}')
        pred.save(out_dir / f'{pred.name}_{args.fold}.npz')
        recs = score(pred, store, labels, events, thresholds, fold=args.fold, metrics=metrics, log=print)
        if args.leaks > 0:
            recs += e8_leak_tests(BL.model_predict_fn(net), store, _leak_rows(store, args.fold, args.leaks),
                                  thresholds, labels=labels, pred=pred, fold=args.fold, guard=gpu,
                                  ring_mask_fn=BL.ring_mask_fn())
        _finish(args, pred, recs, tsha, out_dir)
        return


def _leak_rows(store, fold: str, n: int, seed: int = 0) -> np.ndarray:
    """E8 frames: n test rows of the fold, spread evenly over its environments (fixed seed)."""
    c = store_cache(store)
    test = store.rows(fold=fold, side='test')
    rng = np.random.default_rng(seed)
    envs = np.unique(c.env[test])
    per = max(1, n // max(len(envs), 1))
    out = [rng.choice(test[c.env[test] == e], min(per, int((c.env[test] == e).sum())), replace=False) for e in envs]
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)


if __name__ == '__main__':
    main()
