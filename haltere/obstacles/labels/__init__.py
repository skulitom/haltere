"""Offline label schema: range grid, clearance fan, events and provenance (TRAINING/EVALUATION ONLY).

Labels use hindsight: future frames, whole-flight triangulation, impact annotations and box
colliders. Runtime modules must never import this package
(tests/test_obstacle_label_isolation.py), and it refuses to import in a process that sets
``HALTERE_RUNTIME_PROCESS=1``.

Layout under ``<store>/labels/`` (row i of every array is store index row i)::

    manifest.json               LABEL_MANIFEST_KEYS; status building|complete; store index sha256
    progress.json               builder resumability: {stage: [run_id, ...]}
    grid_value.npy              (N, 18, 32) float16  metres; NaN where UNKNOWN
    grid_kind.npy               (N, 18, 32) uint8    LabelKind
    grid_src.npy                (N, 18, 32) uint8    LabelSource bits that support the value
    fan_value.npy               (N, 4, 9)   float16  metres; NaN where UNKNOWN
    fan_kind.npy                (N, 4, 9)   uint8    LabelKind
    fan_src.npy                 (N, 4, 9)   uint8    LabelSource bits
    events.json                 labelled events (EVENT_FIELDS), superset of <store>/events.json
    teacher/disparity.npy       (N, 36, 64) float16  DA-V2-Small relative disparity (affine-invariant)
    teacher/manifest.json       model id, revision, weights sha256, licence (Apache-2.0), input size
    parts/<stage>/rNNNNN.npz    per-run intermediate outputs (voxel maps, raw per-source labels)

Grid value: for cell (r, c) of contract.GRID_SHAPE, the MINIMUM Euclidean range (m) from the camera
centre over the cell's frustum (sampled with at least 3 x 3 sub-rays), clipped to
[RANGE_MIN_M, RANGE_MAX_M]; a frustum observed free to RANGE_MAX_M is LOWER RANGE_MAX_M.

Fan value: for direction (e, j) of contract.FAN_SHAPE in the gravity-levelled heading frame at the
frame's capture pose, the along-ray distance s >= 0 (m) to the first occupied point within
FAN_CORRIDOR_RADIUS_M (0.5 m) of the ray; observed free to FAN_MAX_M = LOWER FAN_MAX_M. A
corridor that leaves observed space (unobserved volume, outside the image, below the image
edge) before any occupied point is LOWER at that distance (right-censored).

Kinds (``LabelKind``): the true value y relates to the stored value v as
    EXACT  y = v (within EXACT_TOL)   LOWER  y >= v   UPPER  y <= v   UNKNOWN  no information.

Sources (``LabelSource`` bits): L1 TUBE flown tube (next 3 s, 0.35 m radius, before any contact)
= free-space lower bounds; L2 HINDSIGHT bidirectional multi-baseline triangulation fused into
0.2 m voxels with visibility carving; L3 IMPACT contact points (curated + blind-labelled); L4
TEACHER relative depth affine-fitted to L1-L3 anchors (M2); L5 COLLIDER box-collider ray-casts
(Drawing Board box course only, absolute simulator positions).

Combination: each source first reduces its sub-rays/points to one constraint per cell or fan
direction (``min_over``), then sources are folded with ``intersect`` in COMBINE_ORDER. Per-source
raw outputs stay in parts/ so the combination can be recomputed.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from pathlib import Path

import numpy as np

from .. import RUNTIME_ENV_FLAG
from ..contract import FAN_MAX_M, FAN_SHAPE, GRID_SHAPE, RANGE_MAX_M, RANGE_MIN_M

if os.environ.get(RUNTIME_ENV_FLAG) == '1':
    raise ImportError('haltere.obstacles.labels holds offline hindsight labels and must not be imported '
                      'by a runtime process')

LABELS_SCHEMA = 'haltere.obstacles.labels.v1'
EXACT_TOL = 0.10          # relative tolerance under which two constraints count as the same value
TEACHER_SHAPE = (36, 64)  # 7 x 7 px blocks of the 448 x 252 frame


class LabelKind(IntEnum):
    UNKNOWN = 0
    EXACT = 1
    LOWER = 2    # true value >= stored value (free at least this far)
    UPPER = 3    # true value <= stored value (something occupied at or before this distance)


class LabelSource(IntFlag):
    NONE = 0
    TUBE = 1 << 0        # L1 flown tube
    HINDSIGHT = 1 << 1   # L2 hindsight triangulation / voxel carving
    IMPACT = 1 << 2      # L3 impact/contact points
    TEACHER = 1 << 3     # L4 teacher fitted to anchors (M2)
    COLLIDER = 1 << 4    # L5 box colliders


COMBINE_ORDER = (LabelSource.COLLIDER, LabelSource.IMPACT, LabelSource.HINDSIGHT, LabelSource.TUBE,
                 LabelSource.TEACHER)


@dataclass(frozen=True)
class ArraySpec:
    file: str
    dtype: str
    shape: tuple      # per-frame shape
    fill: float


ARRAYS = {
    'grid_value': ArraySpec('grid_value.npy', 'float16', GRID_SHAPE, np.nan),
    'grid_kind': ArraySpec('grid_kind.npy', 'uint8', GRID_SHAPE, int(LabelKind.UNKNOWN)),
    'grid_src': ArraySpec('grid_src.npy', 'uint8', GRID_SHAPE, 0),
    'fan_value': ArraySpec('fan_value.npy', 'float16', FAN_SHAPE, np.nan),
    'fan_kind': ArraySpec('fan_kind.npy', 'uint8', FAN_SHAPE, int(LabelKind.UNKNOWN)),
    'fan_src': ArraySpec('fan_src.npy', 'uint8', FAN_SHAPE, 0),
}
TEACHER = ArraySpec('teacher/disparity.npy', 'float16', TEACHER_SHAPE, np.nan)

LABEL_MANIFEST_KEYS = {
    'schema': LABELS_SCHEMA,
    'status': 'building | complete',
    'created': 'ISO time',
    'code_commit': 'git commit of the label code',
    'store_index_sha256': 'sha256 of the store index.npy the labels are aligned with',
    'n_frames': 'rows (== store rows)',
    'arrays': '{name: {file, dtype, shape, sha256}}',
    'sources': '{TUBE|HINDSIGHT|IMPACT|TEACHER|COLLIDER: {parameters, frames_labelled, cells_by_kind}}',
    'combine': '{order, exact_tol, min_known_frac, conflicts, interval_to_lower}',
    'inputs': '{colliders, lateral_manifest, events_f12, folds, inventory: {path, sha256}}',
    'quality': 'K0b results: L2 vs colliders, L2 vs L3, uncensored near-travel fan share per environment',
    'offline_only': 'true: hindsight labels, never runtime inputs',
}

# ----------------------------------------------------------------------------- events

EVENT_KINDS = ('terminal_impact', 'contact', 'near_pass')
SIDES = ('left', 'right', 'up', 'down', 'either', 'none', 'unknown')
EVENT_FIELDS = {
    'event_id': 'int, unique in the file',
    'store_event_id': 'int id in <store>/events.json, or -1 for label-only events (near passes)',
    'run': 'run name used by the manifests (e.g. "minus-brain03-01"); runs.json aliases',
    'flight': 'inventory physical-flight key',
    'env': 'environment name (splits)',
    'kind': 'terminal_impact | contact | near_pass',
    't_phase': 'run control clock (s) of the contact / closest approach',
    't_wall': 'UNIX epoch s or null',
    'point_w': '[x, y, z] launch-relative simulator FLU contact point on the obstacle surface, or null',
    'normal_w': 'unit surface normal (world FLU) or null',
    'drone_pos_w': '[x, y, z] launch-relative drone position at contact, or null',
    'speed_mps': 'speed just before contact, or null',
    'obstacle': 'pillar | wall | boulder | tree | terrain | gate structure | flag | roof | truss | unknown | ...',
    'unique_obstacle': 'stable key grouping repeated hits on one physical obstacle (E7), e.g. "minus-two/pillar-54.8-3.9"',
    'obstacle_side': 'side of the obstacle relative to travel (free text as in the lateral manifest)',
    'primary_free_side': 'left | right | up | down | either | none | unknown',
    'accepted_free_sides': 'list of sides judged passable',
    'lateral': 'bool: scored by E4',
    'in_view_frac_T2_T1': 'fraction of frames in [T-2, T-1] s where point_w projects into the image, or null',
    'oracle_route': 'bool: privileged oracle-route flight (excluded from evaluation sets)',
    'source': 'lateral_manifest | blind_label | csv_contact | sidecar',
    'blind': 'bool: labelled before any model or baseline output on this event was seen',
    'labeller': 'who labelled it (agent/session id or person)',
    'labelled_at': 'ISO time',
    'confidence': 'high | medium | low (object / side may be given separately in notes)',
    'notes': 'free text',
    'evidence': 'list of image/plot paths',
}
REQUIRED_EVENT_FIELDS = ('event_id', 'run', 'env', 'kind', 't_phase', 'obstacle', 'unique_obstacle',
                         'primary_free_side', 'lateral', 'source', 'blind')


def validate_event(e: dict) -> dict:
    missing = [k for k in REQUIRED_EVENT_FIELDS if k not in e]
    if missing:
        raise ValueError(f'event {e.get("event_id")}: missing {missing}')
    unknown = [k for k in e if k not in EVENT_FIELDS]
    if unknown:
        raise ValueError(f'event {e["event_id"]}: unknown fields {unknown}')
    if e['kind'] not in EVENT_KINDS:
        raise ValueError(f'event {e["event_id"]}: kind {e["kind"]!r} not in {EVENT_KINDS}')
    if e['primary_free_side'] not in SIDES:
        raise ValueError(f'event {e["event_id"]}: primary_free_side {e["primary_free_side"]!r} not in {SIDES}')
    for k in ('point_w', 'normal_w', 'drone_pos_w'):
        v = e.get(k)
        if v is not None and (len(v) != 3 or not np.all(np.isfinite(v))):
            raise ValueError(f'event {e["event_id"]}: {k} must be three finite numbers or null')
    return e


def load_events(path: str | os.PathLike) -> list[dict]:
    """Events from labels/events.json or configs/obstacles/events_f12.json ({'events': [...]}), validated."""
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    events = [validate_event(e) for e in obj['events']]
    ids = [e['event_id'] for e in events]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{path}: duplicate event_id')
    return events


# ----------------------------------------------------------------------------- semantics

def _intervals(values, kinds):
    v = np.asarray(values, dtype=np.float64)
    k = np.asarray(kinds)
    lo = np.where((k == LabelKind.EXACT) | (k == LabelKind.LOWER), v, 0.0)
    hi = np.where((k == LabelKind.EXACT) | (k == LabelKind.UPPER), v, np.inf)
    return np.nan_to_num(lo, nan=0.0), np.where(np.isnan(hi), np.inf, hi)


WIDE_POLICIES = ('lower', 'upper')


def _from_interval(lo, hi, tol, wide: str = 'lower'):
    """Interval [lo, hi] -> (value, kind): exact when hi <= lo * (1 + tol) (value = lo, the nearer end);
    upper when lo == 0; lower when hi is infinite. A wide finite interval keeps the free-space bound
    (LOWER lo) with ``wide='lower'`` (default) or the occupied-evidence bound (UPPER hi) with
    ``wide='upper'``."""
    if wide not in WIDE_POLICIES:
        raise ValueError(f'wide must be one of {WIDE_POLICIES}')
    exact = np.isfinite(hi) & (lo > 0) & (hi <= lo * (1 + tol))
    upper = np.isfinite(hi) & (lo <= 0)
    if wide == 'upper':
        upper = upper | (np.isfinite(hi) & (lo > 0) & ~exact)
    lower = (lo > 0) & ~exact & ~upper
    kind = np.full(np.shape(lo), LabelKind.UNKNOWN, np.uint8)
    kind[lower] = LabelKind.LOWER
    kind[upper] = LabelKind.UPPER
    kind[exact] = LabelKind.EXACT
    value = np.where(exact | lower, lo, np.where(upper, hi, np.nan))
    return value, kind


def min_over(values, kinds, axis: int = -1, *, tol: float = EXACT_TOL, min_known_frac: float = 1.0):
    """Constraint on min_k(y_k) from constraints on each y_k (sub-rays of a cell, points of a corridor).

    Sub-constraints that are UNKNOWN are ignored when at least ``min_known_frac`` of them are known;
    otherwise the minimum is at most the smallest upper value (UPPER) or UNKNOWN.
    Returns (value float64, kind uint8) with ``axis`` removed.
    """
    v = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    k = np.moveaxis(np.asarray(kinds), axis, -1)
    if k.shape[-1] == 0:
        return np.full(k.shape[:-1], np.nan), np.zeros(k.shape[:-1], np.uint8)
    lo, hi = _intervals(v, k)
    known = k != LabelKind.UNKNOWN
    ignore_unknown = known.mean(axis=-1) >= min_known_frac
    ub = hi.min(axis=-1)                       # the minimum is at most any upper value
    # Lower bound of the minimum: every considered element must be at least its lower value.
    # UPPER and (considered) UNKNOWN elements have lower value 0.
    considered = known | ~ignore_unknown[..., None]
    lb = np.where(considered, lo, np.inf).min(axis=-1)
    lb = np.where(np.isinf(lb), 0.0, lb)
    return _from_interval(lb, ub, tol)


def intersect(v1, k1, v2, k2, *, tol: float = EXACT_TOL, wide: str = 'lower'):
    """Combine two constraints on the SAME quantity. Returns (value, kind, conflict bool).

    Intervals are intersected; a contradiction beyond ``tol`` gives UNKNOWN and conflict=True.
    ``wide`` chooses which end of a wide finite interval is kept (see ``_from_interval``).
    """
    lo1, hi1 = _intervals(v1, k1)
    lo2, hi2 = _intervals(v2, k2)
    lo, hi = np.maximum(lo1, lo2), np.minimum(hi1, hi2)
    conflict = lo > hi * (1 + tol)
    # Overlap within tolerance collapses to the nearer value (conservative).
    lo = np.where(conflict, 0.0, np.minimum(lo, hi))
    hi = np.where(conflict, np.inf, hi)
    value, kind = _from_interval(lo, hi, tol, wide)
    return value, kind, conflict


def fan_blocked_targets(values, kinds, within_m: float):
    """BCE targets for P(first blocked <= within_m) and their weights (0 = no information).

    EXACT v -> (v <= d, 1); LOWER s -> (0, 1) if s >= d else ignored; UPPER u -> (1, 1) if u <= d
    else ignored; UNKNOWN -> ignored.
    """
    v = np.asarray(values, dtype=np.float64)
    k = np.asarray(kinds)
    target = np.zeros(v.shape, np.float32)
    weight = np.zeros(v.shape, np.float32)
    ex = k == LabelKind.EXACT
    target[ex] = (v[ex] <= within_m)
    weight[ex] = 1
    lw = (k == LabelKind.LOWER) & (v >= within_m)
    weight[lw] = 1
    up = (k == LabelKind.UPPER) & (v <= within_m)
    target[up] = 1
    weight[up] = 1
    return target, weight


def clip_range(values, kinds, max_m: float):
    """Clip label values to [RANGE_MIN_M, max_m]; EXACT/LOWER values beyond max_m become LOWER max_m."""
    v = np.asarray(values, dtype=np.float64).copy()
    k = np.asarray(kinds).astype(np.uint8).copy()
    far = (v > max_m) & ((k == LabelKind.EXACT) | (k == LabelKind.LOWER))
    v[far] = max_m
    k[far] = LabelKind.LOWER
    far_up = (v > max_m) & (k == LabelKind.UPPER)
    k[far_up] = LabelKind.UNKNOWN
    v[far_up] = np.nan
    v = np.where(k == LabelKind.UNKNOWN, np.nan, np.maximum(v, RANGE_MIN_M))
    return v, k


def clip_grid(values, kinds):
    return clip_range(values, kinds, RANGE_MAX_M)


def clip_fan(values, kinds):
    return clip_range(values, kinds, FAN_MAX_M)


# ----------------------------------------------------------------------------- files

def labels_root(store_root, *, part: str = 'main', sealed_final: bool = False) -> Path:
    root = Path(store_root)
    if part == 'sealed':
        from ..splits import require_sealed
        require_sealed(sealed_final, 'sealed labels')
        root = root / 'sealed'
    return root / 'labels'


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class LabelWriter:
    """Creates/opens the N-aligned label arrays (filled UNKNOWN) and tracks per-stage progress."""

    def __init__(self, store_root, n_frames: int, store_index_sha256: str, *, part: str = 'main',
                 sealed_final: bool = False, teacher: bool = False):
        self.root = labels_root(store_root, part=part, sealed_final=sealed_final)
        self.root.mkdir(parents=True, exist_ok=True)
        self.n = int(n_frames)
        self.index_sha256 = store_index_sha256
        mpath = self.root / 'manifest.json'
        if mpath.exists():
            m = json.loads(mpath.read_text(encoding='utf-8'))
            if m.get('store_index_sha256') != store_index_sha256 or m.get('n_frames') != self.n:
                raise ValueError(f'{self.root}: labels belong to another store index; move them aside first')
        else:
            self._write_manifest(dict(schema=LABELS_SCHEMA, status='building', store_index_sha256=store_index_sha256,
                                      n_frames=self.n, offline_only=True))
        specs = dict(ARRAYS, **({'teacher': TEACHER} if teacher else {}))
        self.arrays = {}
        for name, spec in specs.items():
            path = self.root / spec.file
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                a = np.load(path, mmap_mode='r+')
            else:
                a = np.lib.format.open_memmap(path, mode='w+', dtype=spec.dtype, shape=(self.n,) + tuple(spec.shape))
                a[:] = spec.fill
            if a.shape != (self.n,) + tuple(spec.shape) or a.dtype != np.dtype(spec.dtype):
                raise ValueError(f'{path}: shape/dtype {a.shape} {a.dtype} does not match the schema')
            self.arrays[name] = a
        ppath = self.root / 'progress.json'
        self.progress = json.loads(ppath.read_text(encoding='utf-8')) if ppath.exists() else {}

    def write(self, name: str, rows, values) -> None:
        self.arrays[name][np.asarray(rows, dtype=np.int64)] = values

    def is_done(self, stage: str, run_id: int) -> bool:
        return int(run_id) in self.progress.get(stage, [])

    def mark_done(self, stage: str, run_id: int) -> None:
        for a in self.arrays.values():
            a.flush()
        self.progress.setdefault(stage, []).append(int(run_id))
        tmp = self.root / 'progress.json.tmp'
        tmp.write_text(json.dumps(self.progress), encoding='utf-8')
        os.replace(tmp, self.root / 'progress.json')

    def finalize(self, **manifest) -> dict:
        for a in self.arrays.values():
            a.flush()
        arrays = {}
        for name, a in self.arrays.items():
            spec = TEACHER if name == 'teacher' else ARRAYS[name]
            arrays[name] = dict(file=spec.file, dtype=spec.dtype, shape=list(a.shape),
                                sha256=_sha256_file(self.root / spec.file))
        m = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        m.update(manifest, arrays=arrays, status='complete', schema=LABELS_SCHEMA,
                 store_index_sha256=self.index_sha256, n_frames=self.n, offline_only=True)
        self._write_manifest(m)
        return m

    def _write_manifest(self, m: dict) -> None:
        tmp = self.root / 'manifest.json.tmp'
        tmp.write_text(json.dumps(m, indent=1) + '\n', encoding='utf-8')
        os.replace(tmp, self.root / 'manifest.json')


class LabelSet:
    """Read-only labels aligned with a store index (checked by sha256)."""

    def __init__(self, store_root, store_index_sha256: str, *, part: str = 'main', sealed_final: bool = False,
                 require_complete: bool = True):
        self.root = labels_root(store_root, part=part, sealed_final=sealed_final)
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('schema') != LABELS_SCHEMA:
            raise ValueError(f'{self.root}: schema {self.manifest.get("schema")!r} != {LABELS_SCHEMA!r}')
        if self.manifest.get('store_index_sha256') != store_index_sha256:
            raise ValueError(f'{self.root}: labels were built for another store index (re-posed store?)')
        if require_complete and self.manifest.get('status') != 'complete':
            raise RuntimeError(f'{self.root}: labels are not complete')
        self.n = int(self.manifest['n_frames'])
        self.arrays = {}
        for name, spec in ARRAYS.items():
            a = np.load(self.root / spec.file, mmap_mode='r')
            if a.shape != (self.n,) + tuple(spec.shape) or a.dtype != np.dtype(spec.dtype):
                raise ValueError(f'{self.root / spec.file}: shape/dtype does not match the schema')
            self.arrays[name] = a

    def grid(self, rows):
        """(value float32 m with NaN, kind uint8, src uint8), each (n, 18, 32)."""
        rows = np.asarray(rows, dtype=np.int64)
        a = self.arrays
        return a['grid_value'][rows].astype(np.float32), a['grid_kind'][rows], a['grid_src'][rows]

    def fan(self, rows):
        """(value float32 m with NaN, kind uint8, src uint8), each (n, 4, 9)."""
        rows = np.asarray(rows, dtype=np.int64)
        a = self.arrays
        return a['fan_value'][rows].astype(np.float32), a['fan_kind'][rows], a['fan_src'][rows]

    def teacher(self, rows):
        path = self.root / TEACHER.file
        if not path.exists():
            raise FileNotFoundError(f'{path}: teacher cache not built')
        return np.load(path, mmap_mode='r')[np.asarray(rows, dtype=np.int64)].astype(np.float32)

    def events(self) -> list[dict]:
        return load_events(self.root / 'events.json')
