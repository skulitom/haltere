"""Frame store v1: memory-mapped 448 x 252 RGB frames plus a per-frame index (offline only).

Layout under ``runs/obstacle-store-v1`` (gitignored; never commit it)::

    manifest.json          schema, status (building|complete), camera, build parameters, counts,
                           index sha256, inventory/folds sha256, code commit
    plan.json              ordered run plan written first; run_id = position (stable across resumes)
    runs.json              final run table (list; entry run_id == list position), see RUN_KEYS
    index.npy              (N,) INDEX_DTYPE, rows sorted by (run_id, slot); row i is "frame i"
    events.json            store-level events: terminal impacts and telemetry-detected contacts
    clean_windows.json     curated clean windows (lateral manifest + additions with their criterion)
    frames/r00012.u8       raw uint8 frames of run 12, C-order (n, 252, 448, 3) RGB, no header
    parts/r00012.index.npy per-run index part (builder resumability)
    parts/r00012.json      per-run record; written last, so its presence marks the run as done
    timing/                timing.py outputs (refined per-video offsets, residuals)
    labels/                offline labels (haltere.obstacles.labels schema)
    sealed/                same layout for The Green + Hall 26, only built with --sealed-final

Frames: 448 x 252 RGB uint8, resized with cv2.INTER_AREA from the 1280 x 720 gameplay crop
(x >= 648 of the 1928 x 720 run videos) or from the 640 x 360 PNG/JPG. Nothing is masked in
the store: overlays (HUD glyphs, ring stroke, ghost trails, propeller zone) are computed at
load time by haltere.obstacles.overlays so masks can change without a rebuild. Camera:
haltere.obstacles.contract.store_camera() (f = 140 px, principal point at the centre, 30 deg
uptilt, pixel centres at integer + 0.5).

Index fields (INDEX_DTYPE, little-endian, packed, 120 bytes per row):

==================  ==========  ===============================================================
field               dtype       meaning
==================  ==========  ===============================================================
run_id              <i4         entry in runs.json
slot                <i4         row inside frames/r{run_id:05d}.u8
env                 u1          splits.ENV_CODE (Drawing Board layouts are separate codes)
source              u1          Source: 0 geometry PNG, 1 capture dataset, 2 run video
grade               u1          Grade: 0 exact, 1 capture, 2 good, 3 fair, 4 unreliable
pose_method         u1          PoseMethod (how pos/quat/vel were obtained at t_wall)
flags               <u2         Flag bits (see Flag)
cue_src             u1          CueSource of cue_uv
luma                u1          mean grey level 0-255 of the stored frame
t_wall              <f8         image capture time, UNIX epoch seconds (time.time() domain)
t_phase             <f8         run control clock: the CSV ``phase`` column at t_wall (fallback
                                t_wall - csv.wall[0]); the manifests' impact_phase_s and clean
                                windows use this clock. NaN without telemetry CSV
t_game              <f8         Liftoff telemetry timestamp ``ts`` at t_wall (NaN unknown)
pose_lag_s          <f4         t_wall minus the time of the source pose before compensation
                                (0 when interpolated at t_wall; NaN unknown)
align_offset_s      <f4         video offset used: alignment.used_offset_s if present, else
                                alignment.offset_after_first_row_s (t_wall = csv.wall[0] +
                                t_video + offset); 0 for PNG/capture sources
timing_delta_s      <f4         refinement from timing.py already applied on top of
                                align_offset_s (0 = none)
pos                 <f4 (3,)    camera = body origin, simulator world FLU, LAUNCH-RELATIVE metres
                                as logged; absolute = pos + runs[run_id]['origin_sim']
quat                <f4 (4,)    world-from-body attitude, wxyz, unit norm
vel                 <f4 (3,)    world FLU velocity, m/s
omega               <f4 (3,)    body FLU angular rate, rad/s (NaN unknown)
cue_uv              <f4 (2,)    next-checkpoint ring centre in normalised image coordinates
                                (u right, v down, 0..1 over the 448 x 252 frame); NaN if none
tti_s               <f4         HINDSIGHT: seconds from t_wall to the next terminal impact or
                                detected contact of this run (+inf if none)
event_id            <i4         HINDSIGHT: that event's store_event_id in events.json (-1 none)
==================  ==========  ===============================================================

Hindsight fields (tti_s, event_id, the IN_CLEAN/PRE_EVENT/POST_EVENT flags) are for sampling
and evaluation only and must never be a model input. The model's per-frame inputs are the
frame and the gravity direction derived from ``quat`` (contract.gravity_camera); speed,
position and cue_uv are not model inputs by default.

Selection defaults (FrameStore.rows): grades EXACT/CAPTURE/GOOD/FAIR; POST_EVENT and
SECONDARY_SOURCE rows excluded. UNRELIABLE rows are stored only with --include-unreliable
and never receive geometry labels. Fold membership is not stored: it is derived from ``env``
(and, for ALL, the run's flight key) by haltere.obstacles.splits via FrameStore.rows(fold, side).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from enum import IntEnum, IntFlag
from pathlib import Path

import numpy as np

from . import contract
from .splits import (ENV_BY_CODE, ENV_CODE, FOLDS, SEALED_ENVS, SealedAccessError, env_code_mask,
                     heldback_flight, require_sealed)

STORE_SCHEMA = 'haltere.obstacles.store.v1'
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = REPO_ROOT / 'runs' / 'obstacle-store-v1'
FRAME_SHAPE = contract.FRAME_SHAPE                 # (252, 448, 3)
FRAME_BYTES = int(np.prod(FRAME_SHAPE))            # 338,688 bytes per frame
SEALED_CODES = tuple(ENV_CODE[n] for n in SEALED_ENVS)

DEFAULT_STRIDE = 3            # keep every 3rd new (non-repeated) frame ...
PRE_EVENT_DENSE_S = 6.0       # ... plus every new frame within 6 s before an event (3 s analysis + 3 s warm-up)
POST_CONTACT_DROP_S = 1.0     # frames within 1 s after a non-terminal contact (reset fade) are dropped
HIGH_YAW_RATE = 2.0           # rad/s, Flag.HIGH_YAW_RATE threshold on |omega_z|


class Source(IntEnum):
    GEOMETRY_PNG = 0      # geometry-worker PNG + pose interpolated to the exact capture time
    CAPTURE_DATASET = 1   # data/vision JPG + latest-prior telemetry pose
    RUN_VIDEO = 2         # recorded mp4 + 100 Hz CSV aligned offline


class Grade(IntEnum):
    EXACT = 0         # geometry PNG
    CAPTURE = 1       # capture dataset
    GOOD = 2          # video alignment graded good
    FAIR = 3          # video alignment graded fair
    UNRELIABLE = 4    # video alignment failed; not used for geometry labels


class PoseMethod(IntEnum):
    WORKER_INTERP = 0       # geometry worker pose at capture time
    TELEMETRY_INTERP = 1    # 100 Hz CSV interpolated at t_wall (quaternion slerp/nlerp)
    SOURCE_COMPENSATED = 2  # latest-prior source pose, position advanced by vel * pose_lag_s
    SOURCE_RAW = 3          # latest-prior source pose, uncompensated


class CueSource(IntEnum):
    NONE = 0
    LOGGED = 1      # runtime pilot logged the ring position (CSV cue columns)
    DETECTED = 2    # detected from pixels offline by the overlay ring detector


class Flag(IntFlag):
    NONE = 0
    IN_CLEAN = 1 << 0          # HINDSIGHT: inside a curated clean window (clean_windows.json)
    PRE_EVENT = 1 << 1         # HINDSIGHT: tti_s <= PRE_EVENT_DENSE_S
    POST_EVENT = 1 << 2        # HINDSIGHT: after the terminal impact or < 1 s after a contact (dropped by default)
    ORACLE_ROUTE = 1 << 3      # privileged oracle-route collection flight: data only, never autonomous evidence
    HUMAN = 1 << 4             # human-piloted flight
    DENSE_EXTRA = 1 << 5       # kept for event/clean density, not on the stride grid (debias when sampling)
    HIGH_YAW_RATE = 1 << 6     # |omega_z| > 2 rad/s (timing refinement down-weights these)
    SECONDARY_SOURCE = 1 << 7  # a second source of a flight already in the store (compression-gap study)
    HAS_CSV = 1 << 8           # 100 Hz telemetry CSV available for this run
    TIMING_REFINED = 1 << 9    # timing.py refinement applied (timing_delta_s)


HINDSIGHT_FIELDS = ('tti_s', 'event_id')
HINDSIGHT_FLAGS = Flag.IN_CLEAN | Flag.PRE_EVENT | Flag.POST_EVENT
DEFAULT_GRADES = (Grade.EXACT, Grade.CAPTURE, Grade.GOOD, Grade.FAIR)
DEFAULT_EXCLUDE = Flag.POST_EVENT | Flag.SECONDARY_SOURCE

INDEX_DTYPE = np.dtype([
    ('run_id', '<i4'),
    ('slot', '<i4'),
    ('env', 'u1'),
    ('source', 'u1'),
    ('grade', 'u1'),
    ('pose_method', 'u1'),
    ('flags', '<u2'),
    ('cue_src', 'u1'),
    ('luma', 'u1'),
    ('t_wall', '<f8'),
    ('t_phase', '<f8'),
    ('t_game', '<f8'),
    ('pose_lag_s', '<f4'),
    ('align_offset_s', '<f4'),
    ('timing_delta_s', '<f4'),
    ('pos', '<f4', (3,)),
    ('quat', '<f4', (4,)),
    ('vel', '<f4', (3,)),
    ('omega', '<f4', (3,)),
    ('cue_uv', '<f4', (2,)),
    ('tti_s', '<f4'),
    ('event_id', '<i4'),
])

# runs.json / plan.json entry. Keys marked (builder) are filled when the run is written.
RUN_KEYS = {
    'run_id': 'int, position in plan/runs',
    'source_id': 'inventory source id (run video id | "vision:<dataset>" | "<run>/flight:geometry")',
    'flight': 'inventory physical-flight key (flights[].flight); heldback_flight() uses it',
    'aliases': 'names used by other manifests for this run (e.g. lateral "minus-brain03-01")',
    'source': 'Source name: geometry_png | capture_dataset | run_video',
    'env': 'environment name (splits.ENVIRONMENTS)',
    'path': 'repo-relative video file, capture dataset dir or PNG dir',
    'telemetry_csv': 'repo-relative CSV or null',
    'sidecar_json': 'repo-relative runner JSON or null',
    'origin_sim': '[x, y, z] absolute simulator FLU launch position (sidecar origin_sim) or null',
    'alignment': '{grade, offset_s, basis, refine_delta_s} (offset_s = used_offset_s or offset_after_first_row_s)',
    'controller': 'inventory controller dict (control mode, pilot, motor, runtime_route_oracle, ...)',
    'oracle_route': 'bool: privileged oracle-route collection',
    'human': 'bool: human pilot',
    'secondary': 'bool: not the best source of its flight',
    'n_frames': '(builder) frames written',
    'frames_file': '(builder) "frames/rNNNNN.u8"',
    'frames_sha256': '(builder) sha256 of the raw frame bytes in write order',
    'dropped': '(builder) counts by reason: repeat, menu, countdown, finish, outside_telemetry, post_event, stride',
}
SOURCE_NAMES = {Source.GEOMETRY_PNG: 'geometry_png', Source.CAPTURE_DATASET: 'capture_dataset',
                Source.RUN_VIDEO: 'run_video'}


def empty_index(n: int) -> np.ndarray:
    """An index block of n rows with neutral defaults (NaN floats, -1/inf hindsight fields)."""
    rows = np.zeros(n, INDEX_DTYPE)
    for name in ('t_wall', 't_phase', 't_game', 'pose_lag_s', 'omega', 'cue_uv'):
        rows[name] = np.nan
    rows['quat'] = (1.0, 0.0, 0.0, 0.0)
    rows['tti_s'] = np.inf
    rows['event_id'] = -1
    return rows


def run_stem(run_id: int) -> str:
    return f'r{int(run_id):05d}'


def _json_dump(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def index_sha256(index: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(index).tobytes()).hexdigest()


# ----------------------------------------------------------------------------- writer

class RunWriter:
    """Streams one run's frames and index rows to disk in chunks; ``close()`` marks it done."""

    def __init__(self, store: 'StoreWriter', record: dict):
        self.store, self.record = store, dict(record)
        self.run_id = int(record['run_id'])
        self.env_code = ENV_CODE[record['env']]
        stem = run_stem(self.run_id)
        self.final = store.root / 'frames' / f'{stem}.u8'
        self.tmp = store.root / 'frames' / f'{stem}.u8.tmp'
        self._f = open(self.tmp, 'wb')
        self._sha = hashlib.sha256()
        self._rows: list[np.ndarray] = []
        self.n = 0
        self.closed = False

    def append(self, frames: np.ndarray, rows: np.ndarray) -> None:
        frames = np.asarray(frames)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != FRAME_SHAPE:
            raise ValueError(f'frames must be uint8 (k, {FRAME_SHAPE[0]}, {FRAME_SHAPE[1]}, 3) RGB')
        if rows.dtype != INDEX_DTYPE or rows.shape != (len(frames),):
            raise ValueError('rows must be an INDEX_DTYPE array with one row per frame')
        if (rows['env'] != self.env_code).any():
            raise ValueError(f'run {self.run_id}: every row must carry env code {self.env_code}')
        rows = rows.copy()
        rows['run_id'] = self.run_id
        rows['slot'] = np.arange(self.n, self.n + len(frames), dtype=np.int32)
        data = np.ascontiguousarray(frames).tobytes()
        self._f.write(data)
        self._sha.update(data)
        self._rows.append(rows)
        self.n += len(frames)

    def close(self, *, dropped: dict | None = None, extra: dict | None = None) -> dict:
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        os.replace(self.tmp, self.final)
        rows = np.concatenate(self._rows) if self._rows else np.zeros(0, INDEX_DTYPE)
        stem = run_stem(self.run_id)
        np.save(self.store.root / 'parts' / f'{stem}.index.npy', rows)
        rec = dict(self.record, n_frames=int(self.n), frames_file=f'frames/{stem}.u8',
                   frames_sha256=self._sha.hexdigest(), dropped=dict(dropped or {}), **(extra or {}))
        _json_dump(self.store.root / 'parts' / f'{stem}.json', rec)   # done marker, written last
        self.closed = True
        return rec

    def abort(self) -> None:
        if not self._f.closed:
            self._f.close()
        if self.tmp.exists():
            self.tmp.unlink()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and not self.closed:
            self.abort()
        return False


class StoreWriter:
    """Resumable store writer used by the builder (haltere.obstacles.store_build).

    Protocol: ``write_plan(records, build)`` once (idempotent when the plan is unchanged);
    then for every planned run not ``done()``: ``ChunkGuard.before_chunk()``,
    ``with begin_run(run_id) as w: w.append(...); ...; w.close(dropped=...)``; finally
    ``finalize()``. Interrupted runs leave only ``frames/*.u8.tmp`` files, which
    ``begin_run`` overwrites.
    """

    def __init__(self, root: str | os.PathLike = DEFAULT_STORE, *, part: str = 'main', sealed_final: bool = False):
        root = Path(root)
        if part == 'sealed':
            require_sealed(sealed_final, 'building the sealed store part')
            root = root / 'sealed'
        elif part != 'main':
            raise ValueError("part must be 'main' or 'sealed'")
        self.part = part
        self.root = root
        for sub in ('frames', 'parts'):
            (root / sub).mkdir(parents=True, exist_ok=True)
        self.plan = None
        if (root / 'plan.json').exists():
            self.plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))

    def write_plan(self, records: list[dict], build: dict) -> list[dict]:
        runs = []
        for i, r in enumerate(records):
            missing = [k for k in ('source_id', 'flight', 'source', 'env') if k not in r]
            if missing:
                raise ValueError(f'plan record {i} misses {missing}')
            if r['env'] not in ENV_CODE:
                raise ValueError(f'plan record {r["source_id"]}: unknown environment {r["env"]!r}')
            sealed = r['env'] in SEALED_ENVS
            if sealed != (self.part == 'sealed'):
                raise SealedAccessError(f'{r["source_id"]} ({r["env"]}) does not belong in the {self.part} store part')
            runs.append(dict(r, run_id=i))
        if self.plan is not None:
            old = [r['source_id'] for r in self.plan['runs']]
            if old != [r['source_id'] for r in runs]:
                raise ValueError('an existing plan.json lists different runs; finish or remove the old build first')
            return self.plan['runs']
        self.plan = dict(schema=STORE_SCHEMA, part=self.part, build=build, runs=runs)
        _json_dump(self.root / 'plan.json', self.plan)
        self._write_manifest(status='building')
        return runs

    def done(self, run_id: int) -> bool:
        stem = run_stem(run_id)
        part = self.root / 'parts' / f'{stem}.json'
        if not part.exists():
            return False
        rec = json.loads(part.read_text(encoding='utf-8'))
        frames = self.root / rec['frames_file']
        return frames.exists() and frames.stat().st_size == rec['n_frames'] * FRAME_BYTES

    def begin_run(self, run_id: int) -> RunWriter:
        if self.plan is None:
            raise RuntimeError('write_plan() first')
        return RunWriter(self, self.plan['runs'][int(run_id)])

    def write_json(self, name: str, obj) -> None:
        """Store-level side tables: events.json, clean_windows.json."""
        if name not in ('events.json', 'clean_windows.json'):
            raise ValueError('only events.json and clean_windows.json are side tables')
        _json_dump(self.root / name, obj)

    def finalize(self, *, allow_missing: bool = False) -> dict:
        if self.plan is None:
            raise RuntimeError('no plan')
        runs, parts = [], []
        for r in self.plan['runs']:
            stem = run_stem(r['run_id'])
            if not self.done(r['run_id']):
                if not allow_missing:
                    raise RuntimeError(f'run {r["run_id"]} ({r["source_id"]}) is not done')
                runs.append(dict(r, n_frames=0, frames_file=None, frames_sha256=None, dropped=dict(missing=True)))
                continue
            runs.append(json.loads((self.root / 'parts' / f'{stem}.json').read_text(encoding='utf-8')))
            parts.append(np.load(self.root / 'parts' / f'{stem}.index.npy'))
        index = np.concatenate(parts) if parts else np.zeros(0, INDEX_DTYPE)
        if self.part == 'main' and np.isin(index['env'], SEALED_CODES).any():
            raise SealedAccessError('sealed environment rows in the main store part')
        np.save(self.root / 'index.npy', index)
        _json_dump(self.root / 'runs.json', runs)
        return self._write_manifest(status='complete', index=index, runs=runs)

    def _write_manifest(self, status: str, index: np.ndarray | None = None, runs: list | None = None) -> dict:
        cam = contract.store_camera()
        m = dict(schema=STORE_SCHEMA, part=self.part, status=status,
                 frame_shape=list(FRAME_SHAPE), frame_dtype='uint8', color='RGB', frame_bytes=FRAME_BYTES,
                 frame_layout='frames/rNNNNN.u8: raw C-order (n, 252, 448, 3), no header',
                 resize='cv2.INTER_AREA from the 1280x720 gameplay crop (x >= 648 in 1928x720 run videos) '
                        'or the 640x360 PNG/JPG',
                 camera=dict(width=cam.width, height=cam.height, f=cam.f, cx=cam.cx, cy=cam.cy,
                             tilt_deg=cam.tilt_deg, focal_320=contract.FOCAL_320,
                             pixel_centres='integer + 0.5', frames='see haltere.obstacles.contract'),
                 index_dtype=[list(map(str, d)) for d in INDEX_DTYPE.descr],
                 build=(self.plan or {}).get('build', {}))
        if index is not None:
            counts: dict = {}
            for code in np.unique(index['env']):
                sel = index['env'] == code
                counts[ENV_BY_CODE[int(code)].name] = {
                    SOURCE_NAMES[Source(int(s))]: int((index['source'][sel] == s).sum())
                    for s in np.unique(index['source'][sel])}
            m.update(n_frames=int(len(index)), n_runs=len(runs or []), index_sha256=index_sha256(index),
                     counts=counts)
        _json_dump(self.root / 'manifest.json', m)
        return m


# ----------------------------------------------------------------------------- reader

class FrameStore:
    """Read-only access to a finished store. Sealed rows are only readable with part='sealed'."""

    def __init__(self, root: str | os.PathLike = DEFAULT_STORE, *, part: str = 'main', sealed_final: bool = False):
        root = Path(root)
        if part == 'sealed':
            require_sealed(sealed_final, 'reading the sealed store part')
            root = root / 'sealed'
        elif part != 'main':
            raise ValueError("part must be 'main' or 'sealed'")
        self.root, self.part = root, part
        self.manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('schema') != STORE_SCHEMA:
            raise ValueError(f'{root}: schema {self.manifest.get("schema")!r} != {STORE_SCHEMA!r}')
        if self.manifest.get('status') != 'complete':
            raise RuntimeError(f'{root}: store status is {self.manifest.get("status")!r}, not complete')
        self.index = np.load(root / 'index.npy', mmap_mode='r')
        if self.index.dtype != INDEX_DTYPE:
            raise ValueError(f'{root}/index.npy has dtype {self.index.dtype}, expected INDEX_DTYPE')
        if part == 'main' and np.isin(self.index['env'], SEALED_CODES).any():
            raise SealedAccessError(f'{root}: sealed rows found in the main store part')
        self.runs = json.loads((root / 'runs.json').read_text(encoding='utf-8'))
        self._mm: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.index)

    @property
    def camera(self):
        return contract.store_camera()

    def index_sha256(self) -> str:
        return index_sha256(np.asarray(self.index))

    def run(self, run_id: int) -> dict:
        return self.runs[int(run_id)]

    def _frames(self, run_id: int) -> np.memmap:
        mm = self._mm.get(run_id)
        if mm is None:
            rec = self.runs[run_id]
            path = self.root / rec['frames_file']
            if path.stat().st_size != rec['n_frames'] * FRAME_BYTES:
                raise ValueError(f'{path}: size does not match n_frames={rec["n_frames"]}')
            mm = np.memmap(path, dtype=np.uint8, mode='r', shape=(rec['n_frames'],) + FRAME_SHAPE)
            self._mm[run_id] = mm
        return mm

    def frame(self, i: int) -> np.ndarray:
        """Read-only (252, 448, 3) uint8 RGB view of frame i."""
        row = self.index[int(i)]
        return self._frames(int(row['run_id']))[int(row['slot'])]

    def frames(self, rows) -> np.ndarray:
        """(n, 252, 448, 3) uint8 copy of the given frame rows (read in storage order)."""
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((len(rows),) + FRAME_SHAPE, np.uint8)
        order = np.argsort(rows, kind='stable')
        for k in order:
            out[k] = self.frame(rows[k])
        return out

    def rows(self, *, fold: str | None = None, side: str | None = None, envs=None, sources=None,
             grades=DEFAULT_GRADES, require: int = 0, exclude: int = DEFAULT_EXCLUDE) -> np.ndarray:
        """Frame rows matching all filters, as sorted int64 indices.

        ``fold``/``side`` use splits (side in train | inner_train | inner_val | test). For ALL,
        inner_train/inner_val split flights with heldback_flight(). ``require``/``exclude`` are
        Flag masks: all required bits set, no excluded bit set.
        """
        ix = self.index
        m = np.ones(len(ix), bool)
        if fold is not None:
            if side is None:
                raise ValueError('fold needs a side')
            m &= env_code_mask(ix['env'], fold, side)
            if FOLDS[fold].flight_heldback and side in ('inner_train', 'inner_val'):
                held = np.array([heldback_flight(r['flight']) for r in self.runs], bool)
                want = held if side == 'inner_val' else ~held
                m &= want[ix['run_id']]
        if envs is not None:
            m &= np.isin(ix['env'], [ENV_CODE[e] for e in envs])
        if sources is not None:
            m &= np.isin(ix['source'], [int(s) for s in sources])
        if grades is not None:
            m &= np.isin(ix['grade'], [int(g) for g in grades])
        flags = ix['flags']
        if require:
            m &= (flags & int(require)) == int(require)
        if exclude:
            m &= (flags & int(exclude)) == 0
        return np.flatnonzero(m).astype(np.int64)

    def origin(self, rows) -> np.ndarray:
        """(n, 3) absolute launch positions (runs.json origin_sim), NaN when unknown."""
        rows = np.asarray(rows, dtype=np.int64)
        org = np.array([r.get('origin_sim') or [np.nan] * 3 for r in self.runs], dtype=np.float64)
        return org[self.index['run_id'][rows]]

    def absolute_pos(self, rows) -> np.ndarray:
        """(n, 3) absolute simulator FLU positions (needed by box colliders)."""
        rows = np.asarray(rows, dtype=np.int64)
        return self.index['pos'][rows].astype(np.float64) + self.origin(rows)

    def gravity_camera(self, rows) -> np.ndarray:
        """(n, 3) unit gravity in the camera frame: the model's only per-frame metadata input."""
        rows = np.asarray(rows, dtype=np.int64)
        return contract.gravity_camera(self.index['quat'][rows].astype(np.float64))


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m haltere.obstacles.store')
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='decode sources into the store (implemented in store_build.py)')
    b.add_argument('--inventory', required=True, type=Path)
    b.add_argument('--folds', type=Path, default=REPO_ROOT / 'configs' / 'obstacles' / 'folds.json')
    b.add_argument('--lateral-manifest', type=Path, default=None, help='clean windows and curated impacts')
    b.add_argument('--out', type=Path, default=DEFAULT_STORE)
    b.add_argument('--stride', type=int, default=DEFAULT_STRIDE)
    b.add_argument('--size', default='448x252')
    b.add_argument('--include-unreliable', action='store_true')
    b.add_argument('--secondary', nargs='*', default=[], help='flight keys stored from a second source')
    b.add_argument('--runs', nargs='*', default=None, help='restrict to these source ids (debugging)')
    b.add_argument('--flight-lock', default=None)
    b.add_argument('--sealed-final', action='store_true')
    r = sub.add_parser('repose', help='apply timing refinement to pose fields (implemented in store_build.py)')
    r.add_argument('--store', type=Path, default=DEFAULT_STORE)
    r.add_argument('--timing', type=Path, default=None)
    r.add_argument('--flight-lock', default=None)
    i = sub.add_parser('info', help='summarise a finished store')
    i.add_argument('--store', type=Path, default=DEFAULT_STORE)
    args = p.parse_args(argv)
    if args.cmd in ('build', 'repose'):
        if args.cmd == 'build' and args.size != '448x252':
            raise SystemExit('store v1 is fixed at 448x252')
        from . import store_build   # delivered by the store build agent
        return getattr(store_build, args.cmd)(args)
    s = FrameStore(args.store)
    print(json.dumps(dict(n_frames=len(s), n_runs=len(s.runs), counts=s.manifest.get('counts'),
                          index_sha256=s.manifest.get('index_sha256')), indent=1))


if __name__ == '__main__':
    main()
