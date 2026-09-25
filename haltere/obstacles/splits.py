"""Environments, leave-environment-out folds and the sealed-set guard (offline only).

Rules (docs: scratchpad plan M1; survey-data recommended_splits):

- Split by environment asset, never by frame or run. Every run, geometry archive and
  capture dataset of an environment is on the same side, whatever the layout,
  speed or controller. All Drawing Board layouts are one asset.
- F12 is the primary fold: Minus Two + Pine Valley + Autumn Fields held out, so one
  frozen model is unseen for both target environments offline and live. Its inner
  validation is the F5 group. Recipe choices (M2) are made on F12 inner validation
  only; F3/F4/F5 reuse the recipe and use their inner validation for early stopping.
- The sealed set (The Green, Hall 26) is in no fold. Any request for it raises
  ``SealedAccessError`` unless the caller passes ``sealed_final=True`` (the
  ``--sealed-final`` flag, M7 only).
- ALL is the product candidate: every non-sealed environment trains; its inner
  validation is a deterministic 10 % of flights per environment (``heldback_flight``).
  Its results on training environments are labelled seen-environment.

Fold sides: ``train`` = all non-sealed environments not in ``test``; ``inner_val`` = a
subset of ``train`` used for early stopping and selection; ``inner_train`` = ``train``
minus ``inner_val``; ``test`` = held-out environments; ``sealed`` = the sealed set.

``configs/obstacles/folds.json`` is written by ``python -m haltere.obstacles.splits assign``
from the survey inventory. It freezes these definitions plus the environment of every
source (run video, capture dataset, geometry archive) and physical flight, so the
store builder never re-derives environments. ``load_folds_config`` checks the file
against the constants here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FOLDS_SCHEMA = 'haltere.obstacles.folds.v1'
DEFAULT_FOLDS_PATH = Path(__file__).resolve().parents[2] / 'configs' / 'obstacles' / 'folds.json'


class SealedAccessError(PermissionError):
    """Raised when the sealed final-test environments are requested without sealed_final=True."""


@dataclass(frozen=True)
class Environment:
    name: str          # exact inventory environment name
    code: int          # uint8 code stored in the frame-store index (field `env`)
    asset: str         # environment asset: the unit of splitting
    domain: str        # coarse visual domain (reporting only)
    sealed: bool = False


# Codes are frozen: they are written into runs/obstacle-store-v1/index.npy. Append only.
ENVIRONMENTS = (
    Environment('Straw Bale', 0, 'Straw Bale', 'outdoor farm'),
    Environment('Drawing Board box course', 1, 'Drawing Board', 'generated grid arena'),
    Environment('Drawing Board loop v2', 2, 'Drawing Board', 'generated grid arena'),
    Environment('Drawing Board empty arena', 3, 'Drawing Board', 'generated grid arena'),
    Environment('Minus Two', 4, 'Minus Two', 'dark indoor garage'),
    Environment('Pine Valley', 5, 'Pine Valley', 'forest'),
    Environment('Autumn Fields', 6, 'Autumn Fields', 'forest'),
    Environment('Hangar C03', 7, 'Hangar C03', 'night hangar'),
    Environment('Hannover', 8, 'Hannover', 'stadium'),
    Environment('Paris', 9, 'Paris', 'neon night city'),
    Environment('The Pit', 10, 'The Pit', 'canyon'),
    Environment('The Green', 11, 'The Green', 'outdoor', sealed=True),
    Environment('Hall 26', 12, 'Hall 26', 'indoor hall', sealed=True),
)
UNKNOWN_ENV_CODE = 255
EXCLUDED_ENV_NAMES = ('desktop/other',)   # inventory bucket for clips without telemetry

ENV_BY_NAME = {e.name: e for e in ENVIRONMENTS}
ENV_BY_CODE = {e.code: e for e in ENVIRONMENTS}
ENV_CODE = {e.name: e.code for e in ENVIRONMENTS}
ENV_NAMES = tuple(e.name for e in ENVIRONMENTS)
SEALED_ENVS = tuple(e.name for e in ENVIRONMENTS if e.sealed)
OPEN_ENVS = tuple(e.name for e in ENVIRONMENTS if not e.sealed)

DRAWING_BOARD = ('Drawing Board box course', 'Drawing Board loop v2', 'Drawing Board empty arena')
F5_GROUP = ('Hangar C03', 'Hannover', 'Paris', 'The Pit')
FOREST = ('Pine Valley', 'Autumn Fields')

# Per-epoch sampling caps by environment asset (training-set share after sqrt balancing).
EPOCH_CAPS = {'Straw Bale': 0.30, 'Drawing Board': 0.20}
HELDBACK_FRACTION = 0.10

SIDES = ('train', 'inner_train', 'inner_val', 'test', 'sealed')


@dataclass(frozen=True)
class Fold:
    name: str
    test: tuple[str, ...]
    inner_val: tuple[str, ...]      # environments; empty for ALL (flight-level held-back set instead)
    purpose: str
    flight_heldback: bool = False   # ALL: inner validation by held-back flights, not environments

    @property
    def train(self) -> tuple[str, ...]:
        return tuple(n for n in OPEN_ENVS if n not in self.test)

    @property
    def inner_train(self) -> tuple[str, ...]:
        return tuple(n for n in self.train if n not in self.inner_val)


FOLDS = {
    'F12': Fold('F12', ('Minus Two', 'Pine Valley', 'Autumn Fields'), F5_GROUP,
                'Primary. Dark indoor pillars/walls and forest boulder/trees/mounds unseen; the frozen F12 model '
                'can also fly Minus Two and Pine Valley live as unseen environments.'),
    'F3': Fold('F3', ('Straw Bale',), F5_GROUP,
               'Straw Bale held out: arches, flags, bales, fences; how far the smaller domains carry.'),
    'F4': Fold('F4', DRAWING_BOARD, F5_GROUP,
               'All Drawing Board layouts held out: exact box-collider depth (E2).'),
    'F5': Fold('F5', F5_GROUP, FOREST,
               'Rare domains held out: night hangar trusses, stadium, neon city, canyon.'),
    'ALL': Fold('ALL', (), (),
                'Product candidate trained on every open environment; results on them are seen-environment. '
                'Inner validation = held-back flights. Sealed set scored once with --sealed-final (M7).',
                flight_heldback=True),
}
PRIMARY_FOLD = 'F12'
LOEO_FOLDS = ('F12', 'F3', 'F4', 'F5')   # together they hold out every open environment exactly once


def environment(name_or_code) -> Environment:
    if isinstance(name_or_code, (int, np.integer)):
        return ENV_BY_CODE[int(name_or_code)]
    return ENV_BY_NAME[name_or_code]


def require_sealed(sealed_final: bool, what: str = 'the sealed environments') -> None:
    if not sealed_final:
        raise SealedAccessError(f'{what} (The Green, Hall 26) are sealed; pass sealed_final=True '
                                '(--sealed-final) only for the single M7 evaluation')


def fold_envs(fold: str, side: str, *, sealed_final: bool = False) -> tuple[str, ...]:
    """Environment names on one side of a fold. ``side='sealed'`` requires sealed_final=True."""
    if side not in SIDES:
        raise ValueError(f'Unknown side {side!r}; use one of {SIDES}')
    if side == 'sealed':
        require_sealed(sealed_final)
        return SEALED_ENVS
    f = FOLDS[fold]
    if f.flight_heldback and side in ('inner_train', 'inner_val'):
        # Flight-level split: both sides draw from every training environment.
        return f.train
    return getattr(f, side)


def env_side(fold: str, env: str) -> str:
    """'test', 'inner_val', 'inner_train' or 'sealed' for an environment in a fold (ALL: 'train')."""
    if env in SEALED_ENVS:
        return 'sealed'
    if env not in OPEN_ENVS:
        raise KeyError(f'Unknown environment {env!r}')
    f = FOLDS[fold]
    if env in f.test:
        return 'test'
    if f.flight_heldback:
        return 'train'
    return 'inner_val' if env in f.inner_val else 'inner_train'


def heldback_flight(flight_key: str, fraction: float = HELDBACK_FRACTION) -> bool:
    """Deterministic flight-level hold-back for ALL inner validation: sha1(flight key) < fraction."""
    h = int(hashlib.sha1(flight_key.encode('utf-8')).hexdigest()[:8], 16)
    return h / 0x100000000 < fraction


def env_code_mask(env_codes, fold: str, side: str, *, sealed_final: bool = False) -> np.ndarray:
    """Boolean mask over uint8 environment codes for one fold side (environment level only)."""
    codes = np.asarray(env_codes)
    wanted = [ENV_CODE[n] for n in fold_envs(fold, side, sealed_final=sealed_final)]
    return np.isin(codes, wanted)


def sampling_weights(counts: dict[str, int], caps: dict[str, float] | None = None) -> dict[str, float]:
    """Per-environment epoch shares: proportional to sqrt(n_env), then asset caps with redistribution.

    ``counts`` maps environment name -> number of trainable frames on the training side.
    Caps apply to the summed share of an asset (e.g. all Drawing Board layouts <= 0.20);
    the excess is redistributed proportionally to uncapped assets. Shares sum to 1.
    """
    caps = EPOCH_CAPS if caps is None else caps
    envs = [e for e, n in counts.items() if n > 0]
    if not envs:
        return {}
    w = {e: float(np.sqrt(counts[e])) for e in envs}
    tot = sum(w.values())
    share = {e: v / tot for e, v in w.items()}
    capped: set[str] = set()
    for _ in range(len(caps) + 2):
        asset_share: dict[str, float] = {}
        for e, s in share.items():
            asset_share[ENV_BY_NAME[e].asset] = asset_share.get(ENV_BY_NAME[e].asset, 0.0) + s
        over = [a for a, s in asset_share.items() if a in caps and s > caps[a] + 1e-12]
        if not over:
            break
        for a in over:
            members = [e for e in share if ENV_BY_NAME[e].asset == a]
            scale = caps[a] / asset_share[a]
            for e in members:
                share[e] *= scale
            capped.add(a)
        free = [e for e in share if ENV_BY_NAME[e].asset not in capped]
        rest = 1.0 - sum(share[e] for e in share if ENV_BY_NAME[e].asset in capped)
        free_tot = sum(share[e] for e in free)
        if free_tot <= 0:
            break
        for e in free:
            share[e] *= rest / free_tot
    return share


# ----------------------------------------------------------------------------- inventory

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def assign_environments(inventory: dict) -> dict:
    """Environment of every source and physical flight in the survey inventory.

    Returns ``{'flights': {flight_key: env}, 'sources': {source_id: {...}}}`` where each
    source record is ``{kind, env, flight, usable, counted, derived_from}``. Sources of one
    physical flight must agree on the environment (raises otherwise). Clips without
    telemetry (``desktop/other``) get ``env=None`` and ``usable=False``.
    """
    flights: dict[str, str] = {}
    src_to_flight: dict[str, str] = {}
    for f in inventory['flights']:
        env = f['environment']
        if env not in ENV_BY_NAME:
            raise ValueError(f'Flight {f["flight"]}: unknown environment {env!r}')
        flights[f['flight']] = env
        for s in f.get('sources', []):
            src_to_flight[s['id']] = f['flight']
    sources: dict[str, dict] = {}

    def add(kind, rec):
        env = rec['environment']
        sid = rec['id']
        if env in EXCLUDED_ENV_NAMES:
            env = None
        elif env not in ENV_BY_NAME:
            raise ValueError(f'Source {sid}: unknown environment {env!r}')
        flight = src_to_flight.get(sid)
        if flight is not None and env is not None and flights[flight] != env:
            raise ValueError(f'Source {sid} ({env}) disagrees with its flight {flight} ({flights[flight]})')
        sources[sid] = dict(kind=kind, env=env, flight=flight, usable=bool(rec.get('usable')) and env is not None,
                            counted=bool(rec.get('counted_in_totals', True)),
                            derived_from=rec.get('derived_from'))

    for r in inventory['run_videos']:
        add('run_video', r)
    for r in inventory['capture_datasets']:
        add('capture_dataset', r)
    for r in inventory['geometry_archives']:
        add('geometry_png', r)
    return dict(flights=dict(sorted(flights.items())), sources=dict(sorted(sources.items())))


def folds_definition() -> dict:
    """The JSON-serialisable fold definitions (compared against folds.json by tests)."""
    return dict(
        environments=[dict(name=e.name, code=e.code, asset=e.asset, domain=e.domain, sealed=e.sealed)
                      for e in ENVIRONMENTS],
        unknown_env_code=UNKNOWN_ENV_CODE,
        sealed=dict(environments=list(SEALED_ENVS),
                    guard='haltere.obstacles.splits.SealedAccessError unless sealed_final=True (--sealed-final); '
                          'the store builder skips sealed sources unless --sealed-final and then writes them '
                          'only to runs/obstacle-store-v1/sealed/'),
        folds={k: dict(test=list(f.test), inner_val=list(f.inner_val), train=list(f.train),
                       inner_train=list(f.inner_train), flight_heldback=f.flight_heldback, purpose=f.purpose)
               for k, f in FOLDS.items()},
        primary_fold=PRIMARY_FOLD,
        loeo_folds=list(LOEO_FOLDS),
        epoch_caps_by_asset=dict(EPOCH_CAPS),
        sampling='per-environment share proportional to sqrt(trainable frames), asset caps applied with '
                 'proportional redistribution (splits.sampling_weights)',
        heldback=dict(fraction=HELDBACK_FRACTION, rule='int(sha1(flight_key)[:8], 16) / 2**32 < fraction'),
    )


def write_folds_config(inventory_path: Path, out_path: Path = DEFAULT_FOLDS_PATH) -> dict:
    inventory_path = Path(inventory_path)
    inv = json.loads(inventory_path.read_text(encoding='utf-8'))
    assigned = assign_environments(inv)
    cfg = dict(schema=FOLDS_SCHEMA,
               principle='Leave-environment-out by environment asset; sealed set in no fold; selection on '
                         'training environments of the fold only (F12 recipe choices on its inner validation).',
               **folds_definition(),
               inventory=dict(path=str(inventory_path).replace('\\', '/'), sha256=_sha256_file(inventory_path),
                              created=inv.get('created'), note='survey inventory (scratchpad, not tracked)'),
               **assigned)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return cfg


def load_folds_config(path: Path = DEFAULT_FOLDS_PATH, *, check: bool = True) -> dict:
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if check:
        if cfg.get('schema') != FOLDS_SCHEMA:
            raise ValueError(f'{path}: schema {cfg.get("schema")!r} != {FOLDS_SCHEMA!r}')
        want = folds_definition()
        for key, value in want.items():
            if cfg.get(key) != value:
                raise ValueError(f'{path}: {key!r} differs from haltere.obstacles.splits; regenerate with '
                                 '`python -m haltere.obstacles.splits assign`')
    return cfg


def source_environment(source_id: str, cfg: dict | None = None) -> str | None:
    """Environment of an inventory source id (run video id, 'vision:...', '.../flight:geometry')."""
    cfg = load_folds_config() if cfg is None else cfg
    return cfg['sources'][source_id]['env']


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m haltere.obstacles.splits')
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('assign', help='write configs/obstacles/folds.json from the survey inventory')
    a.add_argument('--inventory', required=True, type=Path)
    a.add_argument('--out', type=Path, default=DEFAULT_FOLDS_PATH)
    s = sub.add_parser('show', help='print fold sides')
    s.add_argument('--folds', type=Path, default=DEFAULT_FOLDS_PATH)
    args = p.parse_args(argv)
    if args.cmd == 'assign':
        cfg = write_folds_config(args.inventory, args.out)
        n_src = sum(1 for s in cfg['sources'].values() if s['usable'])
        print(f'wrote {args.out}: {len(cfg["flights"])} flights, {len(cfg["sources"])} sources ({n_src} usable)')
    else:
        cfg = load_folds_config(args.folds)
        for k, f in cfg['folds'].items():
            print(f'{k}: test={f["test"]} inner_val={f["inner_val"]}')


if __name__ == '__main__':
    main()
