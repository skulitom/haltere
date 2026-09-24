"""Label build orchestration. OFFLINE ONLY.

``python -m haltere.obstacles.labels build --store runs/obstacle-store-v1 --colliders
runs/challenge-box-calibration-20260923/offline-geometry.json --events <lateral manifest.json>
--extra-events configs/obstacles/events_f12.json --flight-lock PATH``

Stages (each resumable per run with LabelWriter.is_done/mark_done; ChunkGuard.before_chunk() per run):

  events     labels/events.json: store events + curated lateral-manifest events + blind events_f12.json,
             contact geometry from the run CSV, unique-obstacle keys, geometric near passes
  colliders  L5 box-collider ray casts (Drawing Board box course, absolute positions)
  tube       L1 flown tube (next 3 s, 0.35 m radius, stopping before contacts/impacts/resets)
  impacts    L3 contact points in frames [T - 3 s, T - 0.15 s]
  hindsight  L2 per-flight triangulation, voxel fusion, projection (maps kept in parts/hindsight_map)
  combine    interval intersection of the sources (combine_sources); conflicts -> UNKNOWN (counted);
             clip; write arrays
  quality    K0b checks (label manifest 'quality' and labels/quality.json)

Per-source outputs are kept in ``labels/parts/<stage>/rNNNNN.npz`` (rows, grid/fan value+kind) so the
combination can be recomputed. Every part records a hash of its inputs (stage code version, rows, poses,
stops, event geometry, feature-mask provenance); a rerun rebuilds exactly the parts whose inputs changed
(a timing re-pose, new events, a new overlay mask) and recombines everything. Stops (the tube and the
flown-path clearing end before them) are contact/impact times, telemetry resets and, for store-pose
paths, jumps the logged velocity cannot explain. Data paths in runs.json are repo-relative; they are
resolved against $HALTERE_DATA_ROOT, then the repository root, then the store's grandparent directory.
Only grade <= FAIR rows get labels (UNRELIABLE alignments stay UNKNOWN).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .. import contract, thermal
from . import (COMBINE_ORDER, EXACT_TOL, WIDE_UPPER_MAX_M, LabelKind, LabelSource, LabelWriter, clip_fan,
               clip_grid, fan_blocked_targets, intersect, labels_root, load_events, validate_event)

STAGES = ('events', 'colliders', 'tube', 'impacts', 'hindsight', 'combine', 'quality')
SOURCE_STAGES = {'colliders': LabelSource.COLLIDER, 'impacts': LabelSource.IMPACT,
                 'hindsight': LabelSource.HINDSIGHT, 'tube': LabelSource.TUBE}
BOX_ENV = 'Drawing Board box course'
TEL_COLS = ('wall', 'ts', 'phase', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz')
TEL_CONSISTENT_M = 0.15       # CSV path used for the tube when it matches the index poses this well (median)
RESET_JUMP_M = 1.0            # telemetry position jump between consecutive rows = reset
NEAR_TRAVEL_YAW = 20.0        # K0b fan density: |yaw| <= 20 deg ...
NEAR_TRAVEL_M = 8.0           # ... and the P(blocked <= 8 m) question
K0B = dict(l2_vs_colliders_median_rel=0.10, l2_vs_l3_within=0.15, l2_vs_l3_frac=0.80, fan_near_uncensored=0.30)


def _log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- data access

def data_path(rel, store_root) -> Path | None:
    if rel is None:
        return None
    p = Path(rel)
    if p.is_absolute():
        return p if p.exists() else None
    from ..store import REPO_ROOT
    roots = [os.environ.get('HALTERE_DATA_ROOT'), REPO_ROOT, Path(store_root).resolve().parents[1]]
    for r in roots:
        if r and (Path(r) / p).exists():
            return Path(r) / p
    return None


def load_telemetry(path) -> dict | None:
    """100 Hz run CSV -> dict of float arrays (wall, ts, phase, pos, vel, quat), or None."""
    import csv
    if path is None or not Path(path).exists():
        return None
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        alias = {} if 'x' in (rd.fieldnames or []) else {'x': 'px', 'y': 'py', 'z': 'pz'}   # older recorder CSVs
        cols = {k: [] for k in TEL_COLS}
        for r in rd:
            for k in TEL_COLS:
                v = r.get(alias.get(k, k))
                try:
                    cols[k].append(float(v) if v not in (None, '', 'nan') else np.nan)
                except ValueError:
                    cols[k].append(np.nan)
    c = {k: np.asarray(v, np.float64) for k, v in cols.items()}
    tel = dict(wall=c['wall'], ts=c['ts'], phase=c['phase'], pos=np.c_[c['x'], c['y'], c['z']],
               vel=np.c_[c['vx'], c['vy'], c['vz']], quat=np.c_[c['qw'], c['qx'], c['qy'], c['qz']])
    ok = np.isfinite(tel['wall']) & np.isfinite(tel['pos']).all(1)
    if ok.sum() < 5:
        return None
    tel = {k: v[ok] for k, v in tel.items()}
    o = np.argsort(tel['wall'], kind='stable')
    return {k: v[o] for k, v in tel.items()}


def index_telemetry(store, run_id: int) -> dict | None:
    """Coarse telemetry from the store index rows of a run (capture datasets without a flight CSV)."""
    ix = store.index
    rows = np.flatnonzero(ix['run_id'] == run_id)
    if len(rows) < 5:
        return None
    r = ix[rows[np.argsort(ix['t_wall'][rows], kind='stable')]]
    return dict(wall=r['t_wall'].astype(np.float64), ts=r['t_game'].astype(np.float64),
                phase=r['t_phase'].astype(np.float64), pos=r['pos'].astype(np.float64),
                vel=r['vel'].astype(np.float64), quat=r['quat'].astype(np.float64), coarse=True)


def reset_times(tel) -> np.ndarray:
    """Wall times of telemetry discontinuities (respawn/reset): position jumps or game-clock drops."""
    if tel is None:
        return np.zeros(0)
    jump = np.linalg.norm(np.diff(tel['pos'], axis=0), axis=1) > RESET_JUMP_M
    back = np.diff(tel['ts']) < -0.05 if np.isfinite(tel['ts']).all() else np.zeros(len(jump), bool)
    return tel['wall'][1:][jump | back]


class RunData:
    """Everything the stages need about one store run."""

    def __init__(self, store, run_id: int, events: list[dict], store_root):
        from ..store import Grade
        self.run_id = int(run_id)
        self.rec = store.runs[self.run_id]
        self.env = self.rec['env']
        ix = store.index
        rows = np.flatnonzero(ix['run_id'] == run_id)
        rows = rows[ix['grade'][rows] <= int(Grade.FAIR)]
        self.rows = rows[np.argsort(ix['t_wall'][rows], kind='stable')]
        self.ix = ix[self.rows]
        self.t = self.ix['t_wall'].astype(np.float64)
        self.pos = self.ix['pos'].astype(np.float64)
        self.quat = self.ix['quat'].astype(np.float64)
        self.events = [e for e in events if e.get('_run_id') == self.run_id]
        self.tel = load_telemetry(data_path(self.rec.get('telemetry_csv'), store_root))
        self.path_t, self.path_pos, self.path_source = self._path()
        stops = [e['t_wall'] for e in self.events if e['kind'] in ('terminal_impact', 'contact') and e.get('t_wall')]
        self.stops = np.sort(np.r_[np.asarray(stops, float), reset_times(self.tel), self._path_jumps()])

    def _path_jumps(self) -> np.ndarray:
        """Times of path samples reached by a jump the logged velocity cannot explain (respawn/reset between
        attempts of a capture set; the store-pose path has no CSV reset detection). The tube stops before them."""
        if self.path_source != 'store index poses' or len(self.t) < 2:
            return np.zeros(0)
        dt = np.diff(self.t)
        v = self.ix['vel'].astype(np.float64)
        pred = 0.5 * (v[1:] + v[:-1]) * dt[:, None]
        err = np.linalg.norm(np.diff(self.pos, axis=0) - pred, axis=1)
        jump = err > RESET_JUMP_M + 0.25 * np.linalg.norm(pred, axis=1)
        return self.t[1:][jump]

    def _path(self):
        if self.tel is not None and len(self.t):
            tw = self.tel['wall']
            inside = (self.t >= tw[0]) & (self.t <= tw[-1])
            if inside.sum() >= 3:
                p = np.stack([np.interp(self.t[inside], tw, self.tel['pos'][:, k]) for k in range(3)], 1)
                err = np.median(np.linalg.norm(p - self.pos[inside], axis=1))
                if err <= TEL_CONSISTENT_M:
                    return tw, self.tel['pos'], f'telemetry csv (median index mismatch {err:.3f} m)'
        return self.t, self.pos, 'store index poses'

    def stop_after(self, t) -> float | None:
        j = np.searchsorted(self.stops, t, side='right')
        return float(self.stops[j]) if j < len(self.stops) else None


# ----------------------------------------------------------------------------- events

def _aliases(rec) -> set:
    a = set(rec.get('aliases') or [])
    for k in ('source_id', 'flight', 'linked_video'):
        if rec.get(k):
            a.add(rec[k])
            a.add(str(rec[k]).rsplit('/', 1)[-1])
    return a


def _match_run(store, e) -> int | None:
    cands = [r['run_id'] for r in store.runs if e['run'] in _aliases(r) or (e.get('flight') and e['flight'] == r.get('flight'))]
    prim = [c for c in cands if not store.runs[c].get('secondary')]
    return (prim or cands or [None])[0]


def build_events(store, store_root, lateral_path, extra_path, log=_log) -> list[dict]:
    """Labelled events (EVENT_FIELDS + private '_run_id') for the whole store."""
    from . import impacts as I
    sev = json.loads((Path(store_root) / 'events.json').read_text(encoding='utf-8'))['events'] \
        if (Path(store_root) / 'events.json').exists() else []
    labelled = []
    if lateral_path:
        labelled += I.lateral_manifest_events(json.loads(Path(lateral_path).read_text(encoding='utf-8')))
    if extra_path and Path(extra_path).exists():
        labelled += [dict(e) for e in load_events(extra_path)]
    tel_cache = {}

    def tel_of(run_id):
        if run_id not in tel_cache:
            tel_cache[run_id] = (load_telemetry(data_path(store.runs[run_id].get('telemetry_csv'), store_root))
                                 or index_telemetry(store, run_id))
        return tel_cache[run_id]

    out, used_store = [], set()
    for e in labelled:
        rid = _match_run(store, e)
        e = dict(e)
        e['_run_id'] = rid
        if rid is not None:
            rec = store.runs[rid]
            e['flight'] = e.get('flight') or rec.get('flight')
            tel = tel_of(rid)
            if (tel is not None and tel.get('coarse') and e['kind'] != 'near_pass' and e.get('point_w') is None
                    and e.get('drone_pos_w') is not None and e.get('t_wall') is not None):
                e = I.contact_point_from_position(e, e['drone_pos_w'], tel['wall'], tel['vel'])
            elif tel is not None and e['kind'] != 'near_pass':
                e = I.attach_contact_geometry(e, tel['wall'], tel['phase'], tel['pos'], tel['vel'], tel['quat'])
            elif tel is not None and e.get('t_wall') is None:
                ok = np.isfinite(tel['phase'])
                e['t_wall'] = round(float(np.interp(e['t_phase'], tel['phase'][ok], tel['wall'][ok])), 3)
            # link to the store event of the same run and time
            best, dbest = None, 0.5
            for s in sev:
                if s['run_id'] != rid or s.get('t_phase') is None:
                    continue
                d = abs(float(s['t_phase']) - float(e['t_phase']))
                if d < dbest and s['store_event_id'] not in used_store:
                    best, dbest = s, d
            if best is not None and e['kind'] != 'near_pass':
                e['store_event_id'] = int(best['store_event_id'])
                used_store.add(best['store_event_id'])
        out.append(e)
    # store events nobody labelled: keep them (tube stops, L3 points) as unlabelled events
    for s in sev:
        if s['store_event_id'] in used_store:
            continue
        rid = int(s['run_id'])
        rec = store.runs[rid]
        e = dict(event_id=-1, store_event_id=int(s['store_event_id']), run=rec['source_id'], flight=rec.get('flight'),
                 env=rec['env'], kind=s['kind'], t_phase=float(s['t_phase']) if s.get('t_phase') is not None else float('nan'),
                 t_wall=float(s['t_wall']), point_w=None, normal_w=None, drone_pos_w=s.get('drone_pos_w'),
                 speed_mps=s.get('speed_mps'), obstacle='unknown', unique_obstacle=None, obstacle_side=None,
                 primary_free_side='unknown', accepted_free_sides=[], lateral=False, in_view_frac_T2_T1=None,
                 oracle_route=bool(rec.get('oracle_route')), source=s.get('source', 'store'), blind=False,
                 labeller='store builder (telemetry)', labelled_at=None, confidence='low',
                 notes=f"unlabelled store event ({s.get('source')}); obstacle not identified", evidence=[], _run_id=rid)
        tel = tel_of(rid)
        if tel is not None and tel.get('coarse') and s.get('drone_pos_w') is not None:
            # no flight CSV: the store poses (frame rate) cannot time the contact; the store event's position comes
            # from the ~35 Hz UDP log. Point = that position - CONTACT_OFFSET_M * (minus the pre-contact velocity).
            e = I.contact_point_from_position(e, s['drone_pos_w'], tel['wall'], tel['vel'])
        elif tel is not None:
            e['drone_pos_w'] = None
            e = I.attach_contact_geometry(e, tel['wall'], tel['phase'], tel['pos'], tel['vel'], tel['quat'])
        out.append(e)
    for e in out:
        if e.get('_run_id') is not None:
            e['oracle_route'] = bool(e.get('oracle_route')) or bool(store.runs[e['_run_id']].get('oracle_route'))
    nps = [e for e in out if e['kind'] == 'near_pass' and e.get('point_w') is None]
    out = [e for e in out if not (e['kind'] == 'near_pass' and e.get('point_w') is None)]
    out = I.assign_unique_obstacles(out)
    out += [resolve_near_pass(e, out, tel_of) for e in nps]
    out += near_passes(store, store_root, out, tel_of)
    for i, e in enumerate(out):
        e['event_id'] = i
        if e.get('in_view_frac_T2_T1') is None and e.get('_run_id') is not None:
            e['in_view_frac_T2_T1'] = in_view_fraction(e, tel_of(e['_run_id']))
    missing = [e['run'] for e in out if e.get('_run_id') is None]
    if missing:
        log(f'events: {len(missing)} labelled events have no store run: {sorted(set(missing))[:10]}')
    return out


def in_view_fraction(e, tel, window=(2.0, 1.0)) -> float | None:
    """Fraction of telemetry samples in [T - 2, T - 1] s where the event point projects into the store image."""
    from .corridors import world_to_pixels
    if tel is None or e.get('point_w') is None or not e.get('t_wall'):
        return None
    T = float(e['t_wall'])
    m = (tel['wall'] >= T - window[0]) & (tel['wall'] <= T - window[1])
    if not m.any():
        return None
    p = np.asarray(e['point_w'], float)[None]
    ok = []
    for k in np.flatnonzero(m):
        q = tel['quat'][k]
        if not np.isfinite(q).all():
            continue
        uv, _, front = world_to_pixels(p, tel['pos'][k], q / np.linalg.norm(q))
        ok.append(bool(contract.in_image(uv, front)[0]))
    return round(float(np.mean(ok)), 3) if ok else None


def resolve_near_pass(e, events, tel_of, window_s: float = 1.0, max_m: float = 2.0) -> dict:
    """A curated near pass without geometry: the nearest located obstacle event (same environment and obstacle
    word) to the flown path within +-window_s of its time gives point_w, normal_w and unique_obstacle."""
    e = dict(e)
    tel = tel_of(e['_run_id']) if e.get('_run_id') is not None else None
    word = (e.get('obstacle') or '').split('(')[0].split()[0].lower() if e.get('obstacle') else ''
    if tel is None or not e.get('t_wall'):
        e['unique_obstacle'] = e.get('unique_obstacle') or f"{e['env'].lower().replace(' ', '-')}/{word or 'unknown'}/unlocated"
        return e
    m = np.abs(tel['wall'] - float(e['t_wall'])) <= window_s
    best, dbest, kbest = None, max_m, None
    for o in events:
        if o['env'] != e['env'] or o.get('point_w') is None or o['kind'] == 'near_pass':
            continue
        if word and not (o.get('obstacle') or '').lower().startswith(word):
            continue
        d = np.linalg.norm(tel['pos'][m] - np.asarray(o['point_w'])[None], axis=1)
        if len(d) and d.min() < dbest:
            best, dbest, kbest = o, float(d.min()), np.flatnonzero(m)[int(np.argmin(d))]
    if best is None:
        e['unique_obstacle'] = f"{e['env'].lower().replace(' ', '-')}/{word or 'unknown'}/unlocated"
        return e
    e.update(point_w=list(best['point_w']), normal_w=best.get('normal_w'), unique_obstacle=best['unique_obstacle'],
             drone_pos_w=[round(float(x), 3) for x in tel['pos'][kbest]], t_wall=round(float(tel['wall'][kbest]), 3),
             speed_mps=round(float(np.linalg.norm(tel['vel'][kbest])), 2))
    e['notes'] = (e.get('notes') or '') + f' | located at the obstacle point of event {best["run"]} ({dbest:.2f} m closest approach)'
    return e


def near_passes(store, store_root, events, tel_of) -> list[dict]:
    """Geometric near passes: other flights of the same environment whose path comes within NEAR_PASS_M of a
    labelled obstacle point (curated or blind events), before any contact of that flight."""
    from . import impacts as I
    pts = [e for e in events if e.get('point_w') is not None and e['kind'] in ('terminal_impact', 'contact')
           and e['source'] in ('lateral_manifest', 'blind_label')]
    out = []
    by_env = {}
    for e in pts:
        by_env.setdefault(e['env'], []).append(e)
    seen = {(e.get('_run_id'), e.get('unique_obstacle')) for e in events if e['kind'] == 'near_pass'}
    seen |= {(e.get('_run_id'), e.get('unique_obstacle')) for e in events if e['kind'] != 'near_pass'}
    for r in store.runs:
        if r['env'] not in by_env or r.get('secondary'):
            continue
        tel = tel_of(r['run_id'])
        if tel is None:
            continue
        stops = [e['t_wall'] for e in events if e.get('_run_id') == r['run_id'] and e.get('t_wall')
                 and e['kind'] in ('terminal_impact', 'contact')]
        for ob in by_env[r['env']]:
            key = (r['run_id'], ob['unique_obstacle'])
            if key in seen:
                continue
            d = np.linalg.norm(tel['pos'] - np.asarray(ob['point_w'])[None], axis=1)
            ok = np.ones(len(d), bool)
            for s in stops:
                ok &= np.abs(tel['wall'] - s) > 1.0
            if not (ok & (d <= I.NEAR_PASS_M)).any():
                continue
            k = int(np.argmin(np.where(ok, d, np.inf)))
            speed = float(np.linalg.norm(tel['vel'][k]))
            if speed < 1.0:
                continue
            seen.add(key)
            out.append(dict(event_id=-1, store_event_id=-1, run=r['source_id'], flight=r.get('flight'), env=r['env'],
                            kind='near_pass', t_phase=float(tel['phase'][k]), t_wall=float(tel['wall'][k]),
                            point_w=list(ob['point_w']), normal_w=ob.get('normal_w'),
                            drone_pos_w=[round(float(x), 3) for x in tel['pos'][k]], speed_mps=round(speed, 2),
                            obstacle=ob['obstacle'], unique_obstacle=ob['unique_obstacle'], obstacle_side=None,
                            primary_free_side='none', accepted_free_sides=[], lateral=False, in_view_frac_T2_T1=None,
                            oracle_route=bool(r.get('oracle_route')), source='near_pass_geometry', blind=True,
                            labeller='label builder (geometry: flown path vs labelled obstacle point)',
                            labelled_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), confidence='medium',
                            notes=(f'closest approach {float(d[k]):.2f} m (drone centre to the labelled contact point of '
                                   f'event {ob["run"]} at t_phase {ob["t_phase"]:.2f})'), evidence=[], _run_id=r['run_id']))
    return out


def public_event(e) -> dict:
    d = {k: v for k, v in e.items() if not k.startswith('_')}
    for k in ('t_phase', 't_wall', 'speed_mps'):
        if d.get(k) is not None and not np.isfinite(d[k]):
            d[k] = None if k != 't_phase' else -1.0
    validate_event(d)
    return d


# ----------------------------------------------------------------------------- per-source stages

def _alloc(n):
    return (np.full((n,) + contract.GRID_SHAPE, np.nan, np.float32), np.zeros((n,) + contract.GRID_SHAPE, np.uint8),
            np.full((n,) + contract.FAN_SHAPE, np.nan, np.float32), np.zeros((n,) + contract.FAN_SHAPE, np.uint8))


def stage_colliders(rd: RunData, scene) -> dict | None:
    from .colliders import collider_labels
    if rd.env != BOX_ENV or scene is None or not rd.rec.get('origin_sim'):
        return None
    org = np.asarray(rd.rec['origin_sim'], np.float64)
    gv, gk, fv, fk = _alloc(len(rd.rows))
    for i in range(len(rd.rows)):
        g = collider_labels(scene, rd.pos[i] + org, rd.quat[i] / np.linalg.norm(rd.quat[i]))
        gv[i], gk[i], fv[i], fk[i] = g
    return dict(grid_v=gv, grid_k=gk, fan_v=fv, fan_k=fk)


def stage_tube(rd: RunData) -> dict | None:
    from .tube import tube_labels
    if len(rd.path_t) < 2:
        return None
    gv, gk, fv, fk = _alloc(len(rd.rows))
    for i in range(len(rd.rows)):
        stop = rd.stop_after(rd.t[i])
        g = tube_labels(rd.path_t, rd.path_pos, rd.t[i], rd.pos[i], rd.quat[i] / np.linalg.norm(rd.quat[i]), stop)
        gv[i], gk[i], fv[i], fk[i] = g
    return dict(grid_v=gv, grid_k=gk, fan_v=fv, fan_k=fk)


def stage_impacts(rd: RunData) -> dict | None:
    from .impacts import IMPACT_WINDOW_S, impact_labels
    evs = [e for e in rd.events if e.get('point_w') is not None and e['kind'] in ('terminal_impact', 'contact')
           and e.get('t_wall')]
    if not evs:
        return None
    gv, gk, fv, fk = _alloc(len(rd.rows))
    touched = np.zeros(len(rd.rows), bool)
    for e in evs:
        T = float(e['t_wall'])
        sel = np.flatnonzero((rd.t >= T - IMPACT_WINDOW_S[1]) & (rd.t <= T - IMPACT_WINDOW_S[0]))
        for i in sel:
            m = (rd.path_t >= rd.t[i]) & (rd.path_t <= T)
            path = np.vstack([rd.pos[i][None], rd.path_pos[m]])
            res = impact_labels(e, rd.pos[i], rd.quat[i] / np.linalg.norm(rd.quat[i]), flown_path=path)
            if res is None:
                continue
            a, b, c, d = res
            if touched[i]:
                a, b, _ = intersect(gv[i], gk[i], a, b)
                c, d, _ = intersect(fv[i], fk[i], c, d)
            gv[i], gk[i], fv[i], fk[i] = a, b, c, d
            touched[i] = True
    if not touched.any():
        return None
    return dict(grid_v=gv, grid_k=gk, fan_v=fv, fan_k=fk)


def stage_hindsight(rd: RunData, store, guard, parts_dir: Path) -> dict | None:
    from . import hindsight as H
    if len(rd.rows) < 3:
        return None
    mp = parts_dir / 'hindsight_map' / f'r{rd.run_id:05d}.npz'
    ep = parts_dir / 'hindsight_map' / f'r{rd.run_id:05d}.est.npz'
    mi = stage_inputs(rd, 'hindsight_map')
    vmap = est = None
    if mp.exists() and ep.exists():
        # the flight map (tracking, triangulation, fusion, path clearing: ~70 % of the stage) is reused when only
        # the projection into frames changed
        old = H.VoxelMap.load(mp)
        if old.meta.get('map_inputs') == mi:
            d = np.load(ep)
            vmap, est = old, H.Estimates(**{k: d[k] for k in d.files})
    if vmap is None:
        rows, vmap, est, tracks = H.build_flight_map(store, rd.run_id, guard, rows=rd.rows)
        assert np.array_equal(rows, rd.rows)
        vmap = H.clear_flown_path(vmap, rd.path_t, rd.path_pos, list(rd.stops))
        vmap.meta['map_inputs'] = mi
        vmap.save(mp)
        np.savez_compressed(ep, **{k: getattr(est, k) for k in ('frame', 'track', 'uv', 'point', 'range_m', 'sigma_m',
                                                                 'offset_s')})
    gv, gk, fv, fk = _alloc(len(rd.rows))
    for i in range(len(rd.rows)):
        if guard is not None and i and i % H.CHUNK_FRAMES == 0:
            guard.before_chunk()
        uv, rr, pp = H.own_frame_points(est, vmap, i)
        g = H.hindsight_labels(vmap, rd.pos[i], rd.quat[i] / np.linalg.norm(rd.quat[i]), own_uv=uv, own_range=rr,
                               own_points=pp)
        gv[i], gk[i], fv[i], fk[i] = g
    return dict(grid_v=gv, grid_k=gk, fan_v=fv, fan_k=fk, meta=vmap.meta)


def save_part(path: Path, rows, out: dict | None, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    if out is None:
        np.savez_compressed(tmp, rows=np.asarray(rows, np.int64), empty=np.array(True),
                            meta=np.array(json.dumps(extra or {})))
    else:
        np.savez_compressed(tmp, rows=np.asarray(rows, np.int64), empty=np.array(False),
                            grid_v=out['grid_v'].astype(np.float16), grid_k=out['grid_k'],
                            fan_v=out['fan_v'].astype(np.float16), fan_k=out['fan_k'],
                            meta=np.array(json.dumps(dict(out.get('meta') or {}, **(extra or {})), default=str)))
    os.replace(tmp, path)


def load_part(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    if bool(d['empty']):
        return dict(rows=d['rows'], empty=True, meta=json.loads(str(d['meta'])))
    return dict(rows=d['rows'], empty=False, grid_v=d['grid_v'].astype(np.float32), grid_k=d['grid_k'],
                fan_v=d['fan_v'].astype(np.float32), fan_k=d['fan_k'], meta=json.loads(str(d['meta'])))


# ----------------------------------------------------------------------------- combination

def combine_sources(parts: dict) -> tuple:
    """{LabelSource: (value, kind) arrays} -> (value, kind, src, conflict).

    Every source's constraint is an interval [lo, hi] on the true value; the sources are intersected
    (labels.intersect semantics, applied to the full intervals so the result does not depend on the order
    COMBINE_ORDER in which they are listed): lo = max lo_s, hi = min hi_s. lo > hi (1 + EXACT_TOL) is a
    conflict -> UNKNOWN (counted). hi <= lo (1 + EXACT_TOL) -> EXACT lo. A one-sided result keeps its side.
    A wide finite interval (free-space evidence lo AND occupied evidence hi, e.g. flown tube and a hit
    beyond it) keeps UPPER hi when hi <= WIDE_UPPER_MAX_M (the near field where the clearance questions
    P(blocked <= 4 / 8 m) and avoidance decisions live: the occupied bound decides them, the free one does
    not) and LOWER lo otherwise (far hits bound little; the free extent is the useful part).
    ``src`` holds the bits of every source that constrains a non-conflicting cell.
    """
    from . import _intervals, resolve_interval
    shape = next(iter(parts.values()))[0].shape
    lo = np.zeros(shape)
    hi = np.full(shape, np.inf)
    src = np.zeros(shape, np.uint8)
    for s in COMBINE_ORDER:
        if s not in parts:
            continue
        sv, sk = parts[s]
        lo_s, hi_s = _intervals(sv, sk)
        lo, hi = np.maximum(lo, lo_s), np.minimum(hi, hi_s)
        src |= np.where(np.asarray(sk) != LabelKind.UNKNOWN, np.uint8(int(s)), np.uint8(0))
    conflict = lo > hi * (1 + EXACT_TOL)
    lo = np.where(conflict, 0.0, np.minimum(lo, hi))
    hi = np.where(conflict, np.inf, hi)
    v, k, wide, near = resolve_interval(lo, hi, EXACT_TOL, WIDE_UPPER_MAX_M)
    src = np.where(conflict | (k == LabelKind.UNKNOWN), 0, src).astype(np.uint8)
    combine_sources.last_wide = dict(to_upper=int(near.sum()), to_lower=int((wide & ~near).sum()))
    return v, k, src, conflict


COMBINE_STATS = dict(conflicts_grid=0, conflicts_fan=0, wide_to_upper_grid=0, wide_to_lower_grid=0,
                     wide_to_upper_fan=0, wide_to_lower_fan=0, cells_grid=0, cells_fan=0)


def combine_run(writer: LabelWriter, parts_dir: Path, run_id: int) -> dict:
    stats = dict(COMBINE_STATS)
    loaded = {}
    rows = None
    for stage, s in SOURCE_STAGES.items():
        p = load_part(parts_dir / stage / f'r{run_id:05d}.npz')
        if p is None or p['empty']:
            continue
        if rows is None:
            rows = p['rows']
        elif not np.array_equal(rows, p['rows']):
            raise ValueError(f'run {run_id}: {stage} rows differ from the other sources')
        loaded[s] = p
    if not loaded:
        return stats
    gv, gk, gs, gc = combine_sources({s: (p['grid_v'], p['grid_k']) for s, p in loaded.items()})
    wide_g = combine_sources.last_wide
    fv, fk, fs, fc = combine_sources({s: (p['fan_v'], p['fan_k']) for s, p in loaded.items()})
    wide_f = combine_sources.last_wide
    stats.update(wide_to_upper_grid=wide_g['to_upper'], wide_to_lower_grid=wide_g['to_lower'],
                 wide_to_upper_fan=wide_f['to_upper'], wide_to_lower_fan=wide_f['to_lower'],
                 cells_grid=int(gk.size), cells_fan=int(fk.size))
    gv, gk2 = clip_grid(gv, gk)
    fv, fk2 = clip_fan(fv, fk)
    gs = np.where(gk2 == LabelKind.UNKNOWN, 0, gs).astype(np.uint8)
    fs = np.where(fk2 == LabelKind.UNKNOWN, 0, fs).astype(np.uint8)
    writer.write('grid_value', rows, gv.astype(np.float16))
    writer.write('grid_kind', rows, gk2)
    writer.write('grid_src', rows, gs)
    writer.write('fan_value', rows, fv.astype(np.float16))
    writer.write('fan_kind', rows, fk2)
    writer.write('fan_src', rows, fs)
    stats.update(conflicts_grid=int(gc.sum()), conflicts_fan=int(fc.sum()))
    return stats


# ----------------------------------------------------------------------------- K0b quality

def _cell_of(uv):
    return (np.clip((uv[..., 1] // contract.PATCH_PX).astype(int), 0, contract.GRID_H - 1),
            np.clip((uv[..., 0] // contract.PATCH_PX).astype(int), 0, contract.GRID_W - 1))


L23_LEVELS = ('point', 'cell', 'disc')


def _l23_level(value, kind, true, visible, tol) -> dict:
    """One comparison of an L2 constraint (value, kind) with the true range of the contact point."""
    both = kind in (LabelKind.EXACT, LabelKind.UPPER) and np.isfinite(value)
    ratio = float(value / true) if both else None
    # free space claimed beyond a visible contact point (LOWER or EXACT more than tol behind it)
    beyond = bool(visible and kind in (LabelKind.LOWER, LabelKind.EXACT) and value > true * (1 + tol))
    return dict(kind=int(kind), ratio=ratio, beyond=beyond)


def l2_vs_l3_frames(store, parts_dir: Path, e: dict, store_root, window=(2.0, 0.5)) -> list[dict] | None:
    """Per frame in [T - window[0], T - window[1]] where the contact point projects into the image: the L2
    constraint at the contact point against its true range |point - camera|, at three levels.

    point  the L2 sub-ray nearest to the projected point (recomputed from the saved flight map): the triangulated
           range at the point itself (the K0b definition used for pass/fail)
    cell   the L2 grid cell holding the projected point (the E1 cell definition; its value is the minimum over the
           cell's 14 x 14 px frustum, below the point's range on grazing surfaces and at nearer occluders)
    disc   median over the L3 disc sub-rays (labels.impacts: disc on the point) with L2 EXACT or UPPER
    Each level gives ``ratio`` (L2 value / true when EXACT or UPPER) and ``beyond`` (a VISIBLE point, i.e. the L3
    disc is EXACT because the camera-to-point segment was flown, with L2 LOWER or EXACT more than 15 % behind it:
    free space claimed through the obstacle). None when the run has no L2 map.
    """
    from . import hindsight as H
    from .corridors import subray_pixels
    from .impacts import impact_labels
    rid = int(e['_run_id'])
    mp = parts_dir / 'hindsight_map' / f'r{rid:05d}.npz'
    ep = parts_dir / 'hindsight_map' / f'r{rid:05d}.est.npz'
    part = load_part(parts_dir / 'hindsight' / f'r{rid:05d}.npz')
    if not mp.exists() or not ep.exists() or part is None or part['empty']:
        return None
    vmap = H.VoxelMap.load(mp)
    d = np.load(ep)
    est = H.Estimates(**{k: d[k] for k in d.files})
    rd = RunData(store, rid, [], store_root)
    if not np.array_equal(part['rows'], rd.rows):
        return None
    T = float(e['t_wall'])
    p = np.asarray(e['point_w'], float)
    tol = K0B['l2_vs_l3_within']
    px = subray_pixels().reshape(-1, 2)
    out = []
    for i in np.flatnonzero((rd.t >= T - window[0]) & (rd.t <= T - window[1])):
        q = rd.quat[i] / np.linalg.norm(rd.quat[i])
        uv, ok = contract.project_camera(((p - rd.pos[i]) @ contract.camera_to_world(q))[None])
        if not contract.in_image(uv, ok)[0]:
            continue
        true = float(np.linalg.norm(p - rd.pos[i]))
        m = (rd.path_t >= rd.t[i]) & (rd.path_t <= T)
        res = impact_labels(e, rd.pos[i], q, flown_path=np.vstack([rd.pos[i][None], rd.path_pos[m]]),
                            return_subrays=True)
        s3v, s3k = res[4].reshape(-1), res[5].reshape(-1)
        disc = s3k != LabelKind.UNKNOWN
        visible = bool(disc.any() and (s3k[disc] == LabelKind.EXACT).all())
        uvo, rro, ppo = H.own_frame_points(est, vmap, int(i))
        s2v, s2k = H.hindsight_labels(vmap, rd.pos[i], q, own_uv=uvo, own_range=rro, own_points=ppo,
                                      return_subrays=True)[4:]
        s2v, s2k = s2v.reshape(-1), s2k.reshape(-1)
        j = int(np.argmin(np.linalg.norm(px - uv[0][None], axis=1)))
        r, c = _cell_of(uv[0])
        rec = dict(i=int(i), true_m=true, visible=visible,
                   point=_l23_level(float(s2v[j]), int(s2k[j]), true, visible, tol),
                   cell=_l23_level(float(part['grid_v'][i, r, c]), int(part['grid_k'][i, r, c]), true, visible, tol))
        both = disc & np.isin(s2k, (LabelKind.EXACT, LabelKind.UPPER))
        beyond = disc & (s3k == LabelKind.EXACT) & np.isin(s2k, (LabelKind.LOWER, LabelKind.EXACT)) & (
            s2v > s3v * (1 + tol))
        rec['disc'] = dict(kind=-1, ratio=float(np.median(s2v[both] / s3v[both])) if both.any() else None,
                           beyond=bool(beyond.any()))
        out.append(rec)
    return out


def quality(store, parts_dir: Path, events: list[dict], run_ids, writer: LabelWriter | None = None) -> dict:
    """K0b: L2 vs colliders, L2 vs L3, near-travel fan density (Minus Two / Pine Valley)."""
    q = dict(thresholds=K0B)
    # (a) L2 vs colliders on the box course
    rel = {1: [], 3: []}
    ratio_all, low = [], [0, 0, 0, 0]          # LOWER cells vs collider EXACT: n, violations; fan LOWER: n, violations
    fan_rel = []
    for rid in run_ids:
        if store.runs[rid]['env'] != BOX_ENV:
            continue
        a = load_part(parts_dir / 'hindsight' / f'r{rid:05d}.npz')
        b = load_part(parts_dir / 'colliders' / f'r{rid:05d}.npz')
        if a is None or b is None or a['empty'] or b['empty']:
            continue
        m = (b['grid_k'] == LabelKind.EXACT) & (b['grid_v'] >= 2) & (b['grid_v'] <= 10)
        for kind in (LabelKind.EXACT, LabelKind.UPPER):
            mm = m & (a['grid_k'] == kind)
            rel[int(kind)].append(np.abs(a['grid_v'][mm] - b['grid_v'][mm]) / b['grid_v'][mm])
            ratio_all.append(a['grid_v'][mm] / b['grid_v'][mm])
        mm = (a['grid_k'] == LabelKind.LOWER) & (b['grid_k'] == LabelKind.EXACT)
        low[0] += int(mm.sum())
        low[1] += int((a['grid_v'][mm] > b['grid_v'][mm] * (1 + EXACT_TOL)).sum())
        truth = np.where(b['fan_k'] == LabelKind.EXACT, b['fan_v'], contract.FAN_MAX_M)
        mm = a['fan_k'] == LabelKind.LOWER
        low[2] += int(mm.sum())
        low[3] += int((a['fan_v'][mm] > truth[mm] * (1 + EXACT_TOL)).sum())
        mm = np.isin(a['fan_k'], (LabelKind.EXACT, LabelKind.UPPER)) & (b['fan_k'] == LabelKind.EXACT)
        fan_rel.append(np.abs(a['fan_v'][mm] - b['fan_v'][mm]) / b['fan_v'][mm])
    r1 = np.concatenate(rel[1]) if rel[1] else np.zeros(0)
    r3 = np.concatenate(rel[3]) if rel[3] else np.zeros(0)
    rall = np.r_[r1, r3]
    ratio_all = np.concatenate(ratio_all) if ratio_all else np.zeros(0)
    fan_rel = np.concatenate(fan_rel) if fan_rel else np.zeros(0)
    q['l2_vs_colliders'] = dict(
        cells_exact=int(len(r1)), cells_upper=int(len(r3)),
        median_rel_err_exact=float(np.median(r1)) if len(r1) else None,
        median_rel_err_upper=float(np.median(r3)) if len(r3) else None,
        median_rel_err_all=float(np.median(rall)) if len(rall) else None,
        median_ratio_all=float(np.median(ratio_all)) if len(ratio_all) else None,
        frac_within_10pct=float(np.mean(rall <= 0.10)) if len(rall) else None,
        grid_lower_cells=low[0], grid_lower_violation_frac=low[1] / low[0] if low[0] else None,
        fan_lower_cells=low[2], fan_lower_violation_frac=low[3] / low[2] if low[2] else None,
        fan_hit_cells=int(len(fan_rel)), fan_hit_median_rel_err=float(np.median(fan_rel)) if len(fan_rel) else None,
        passed=bool(len(rall) and np.median(rall) <= K0B['l2_vs_colliders_median_rel']),
        definition=('box-course grid cells with collider EXACT range in [2, 10] m and an L2 EXACT or UPPER value; '
                    'violations: L2 LOWER beyond the collider range (grid: collider EXACT; fan: collider hit or '
                    'FAN_MAX_M) by more than 10 %'))
    # (b) L2 vs L3 at the contact point, sub-ray level, in [T - 2.0, T - 0.5] s
    store_root = Path(parts_dir).resolve().parents[1]
    tol = K0B['l2_vs_l3_within']
    in_scope = {int(r) for r in run_ids}
    cnt_keys = ('both', 'within', 'exact', 'exact_within', 'beyond')

    def counts(recs, level):
        c = dict.fromkeys(cnt_keys, 0)
        for r in recs:
            x = r[level]
            c['beyond'] += x['beyond']
            if x['ratio'] is None:
                continue
            ok = abs(x['ratio'] - 1) <= tol
            c['both'] += 1
            c['within'] += ok
            c['exact'] += x['kind'] == LabelKind.EXACT
            c['exact_within'] += ok and x['kind'] == LabelKind.EXACT
        return c

    def summary(recs):
        s = dict(frames_in_view=len(recs), frames_visible=int(sum(r['visible'] for r in recs)))
        for level in L23_LEVELS:
            c = counts(recs, level)
            s[level] = dict(c, frac_within_15pct=c['within'] / c['both'] if c['both'] else None,
                            coverage=c['both'] / len(recs) if recs else None,
                            median_ratio=(float(np.median([r[level]['ratio'] for r in recs
                                                           if r[level]['ratio'] is not None])) if c['both'] else None))
        return s

    groups = {}          # 'all', 'labelled' (curated + blind), per environment -> frame records
    per_event = []
    for e in events:
        if (e.get('_run_id') is None or e['_run_id'] not in in_scope or e.get('point_w') is None
                or e['kind'] not in ('terminal_impact', 'contact') or not e.get('t_wall')):
            continue
        recs = l2_vs_l3_frames(store, parts_dir, e, store_root)
        if not recs:
            continue
        ranges = [r['true_m'] for r in recs]
        per_event.append(dict(event_id=e['event_id'], run=e['run'], env=e['env'], kind=e['kind'], source=e['source'],
                              blind=e.get('blind'), obstacle=e.get('obstacle'),
                              range_m=[round(float(min(ranges)), 2), round(float(max(ranges)), 2)], **summary(recs)))
        keys = ['all', f'env:{e["env"]}'] + (['labelled'] if e['source'] in ('lateral_manifest', 'blind_label') else [])
        for g in keys:
            groups.setdefault(g, []).extend(recs)
    groups = {g: summary(recs) for g, recs in groups.items()}
    head = groups.get('all', {}).get('cell', {})
    frac = head.get('frac_within_15pct')
    q['l2_vs_l3'] = dict(
        groups=groups, per_event=per_event, frac_within_15pct=frac, coverage=head.get('coverage'),
        passed=bool(frac is not None and frac >= K0B['l2_vs_l3_frac']),
        definition=('frames in [T-2.0, T-0.5] s before located terminal impacts and contacts where the contact point '
                    'projects into the image. Pass/fail on level "cell" over all such events (the E1 definition of '
                    'haltere.obstacles.evaluate, i.e. the target the model is trained and scored on): the L2 grid cell '
                    'holding the projected point; "both" = that cell is EXACT or UPPER; ratio = cell value / |point - '
                    'camera|; within = |ratio - 1| <= 0.15; frac_within_15pct = within / both; coverage = both / in '
                    'view. The cell value is the minimum over its 14 x 14 px frustum, so it sits below the point range '
                    'on grazing surfaces (terrain) and with nearer occluders in the cell. Also reported: level "point" '
                    '(the L2 sub-ray nearest to the projected point) and "disc" (median over the L3 disc sub-rays); '
                    '"exact" counts EXACT only; '
                    '"beyond" = frames where a VISIBLE point (L3 disc EXACT: the camera-to-point segment was flown) '
                    'has L2 LOWER or EXACT more than 15 % behind it (free space claimed through the obstacle). Groups: '
                    'all events, curated + blind labelled events, per environment'))
    # (c) near-travel fan density on Minus Two and Pine Valley (combined labels)
    q['fan_near_travel'] = {}
    if writer is not None:
        yaw_ok = np.abs(contract.FAN_YAW_DEG) <= NEAR_TRAVEL_YAW
        from ..splits import ENV_CODE
        from ..store import Grade
        # rows the builder labels (grade <= FAIR) of the runs in this build
        in_build = np.isin(store.index['run_id'], np.asarray(list(run_ids), np.int64)) & (
            store.index['grade'] <= int(Grade.FAIR))
        for env in ('Minus Two', 'Pine Valley', 'Autumn Fields', 'Straw Bale', BOX_ENV, 'Hangar C03', 'Hannover',
                    'Paris', 'The Pit', 'Drawing Board loop v2'):
            rows = np.flatnonzero((store.index['env'] == ENV_CODE[env]) & in_build)
            if not len(rows):
                continue
            fv = writer.arrays['fan_value'][rows][..., yaw_ok].astype(np.float32)
            fk = writer.arrays['fan_kind'][rows][..., yaw_ok]
            _, w8 = fan_blocked_targets(fv, fk, NEAR_TRAVEL_M)
            t8, _ = fan_blocked_targets(fv, fk, NEAR_TRAVEL_M)
            q['fan_near_travel'][env] = dict(
                frames=int(len(rows)), cells=int(fk.size), uncensored_8m=float(w8.mean()),
                blocked_8m=float((t8 * w8).sum() / fk.size), free_8m=float(((1 - t8) * w8).sum() / fk.size),
                exact=float((fk == LabelKind.EXACT).mean()), lower=float((fk == LabelKind.LOWER).mean()),
                upper=float((fk == LabelKind.UPPER).mean()),
                passed=bool(w8.mean() >= K0B['fan_near_uncensored']))
        q['fan_near_travel']['definition'] = ('fan cells with |yaw| <= 20 deg (all 4 elevations) whose combined label '
                                              'decides P(first blocked <= 8 m) (labels.fan_blocked_targets weight > 0)')
    return q



# ----------------------------------------------------------------------------- resumability

LABEL_CODE_VERSION = 'labels-m1-1'     # bump when every stage's output for the same inputs changes
# per-stage versions: bump one when that stage's output for the same inputs changes
STAGE_CODE_VERSION = dict(colliders='1', tube='1', impacts='2',   # impacts 2: 0.1 m disc
                          hindsight='3',       # 2: adjacent partners, epipolar check, track support, per-stage hash;
                                               # 3: bracketed cell minima keep UPPER hit <= 8 m
                          hindsight_map='2')   # the flight map alone (tracking, triangulation, fusion, path clearing)


def _hash(*arrays, stage: str = '') -> str:
    import hashlib
    h = hashlib.sha256(LABEL_CODE_VERSION.encode())
    if STAGE_CODE_VERSION.get(stage, '1') != '1':
        h.update(f'{stage}:{STAGE_CODE_VERSION[stage]}'.encode())
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()[:24]


def stage_inputs(rd: 'RunData', stage: str) -> str:
    """Hash of everything a stage's output for one run depends on (code version, poses, rows, stops, events)."""
    base = [rd.rows, rd.t, rd.pos, rd.quat]
    if stage == 'colliders':
        return _hash(*base, np.asarray(rd.rec.get('origin_sim') or [np.nan] * 3, float), stage=stage)
    if stage == 'tube':
        return _hash(*base, rd.stops, rd.path_t[:: max(1, len(rd.path_t) // 500)], stage=stage)
    if stage in ('hindsight', 'hindsight_map'):
        from .hindsight import mask_provenance
        mask = np.frombuffer(mask_provenance().encode(), np.uint8)
        return _hash(*base, rd.stops, rd.path_t[:: max(1, len(rd.path_t) // 500)], mask, stage=stage)
    ev = [np.r_[e.get('t_wall') or np.nan, e.get('point_w') or [np.nan] * 3, e.get('normal_w') or [np.nan] * 3]
          for e in rd.events if e['kind'] in ('terminal_impact', 'contact')]
    return _hash(*base, np.asarray(ev, float).reshape(-1, 7), rd.path_t[:: max(1, len(rd.path_t) // 500)], stage=stage)


def part_current(parts_dir: Path, stage: str, run_id: int, inputs: str | None) -> bool:
    """A stage part exists (and, when ``inputs`` is given, was built from these inputs)."""
    p = parts_dir / stage / f'r{run_id:05d}.npz'
    if not p.exists():
        return False
    if stage == 'hindsight' and not (parts_dir / 'hindsight_map' / f'r{run_id:05d}.npz').exists():
        d = np.load(p, allow_pickle=False)
        if not bool(d['empty']):
            return False
    if inputs is None:
        return True
    try:
        meta = json.loads(str(np.load(p, allow_pickle=False)['meta']))
    except Exception:
        return False
    return meta.get('inputs') == inputs


def move_stale_arrays(root: Path, sha: str, n: int, log=_log) -> None:
    """Labels built on another store index (e.g. before a timing re-pose) are moved to stale-<sha>/; the
    per-run parts stay and are reused where their inputs did not change."""
    mpath = root / 'manifest.json'
    if not mpath.exists():
        return
    m = json.loads(mpath.read_text(encoding='utf-8'))
    if m.get('store_index_sha256') == sha and m.get('n_frames') == n:
        return
    from . import ARRAYS
    if m.get('status') != 'complete' and m.get('n_frames') == n:
        # an unfinished build on the old index: reset its arrays in place (nothing final to keep)
        for spec in ARRAYS.values():
            if (root / spec.file).exists():
                a = np.load(root / spec.file, mmap_mode='r+')
                a[:] = spec.fill
                a.flush()
                del a
        m.update(store_index_sha256=sha, status='building')
        tmp = root / 'manifest.json.tmp'
        tmp.write_text(json.dumps(m, indent=1) + '\n', encoding='utf-8')
        os.replace(tmp, mpath)
        (root / 'progress.json').write_text('{}', encoding='utf-8')
        log('unfinished labels of another store index: arrays reset; parts reused where their inputs are unchanged')
        return
    stale = root / f"stale-{str(m.get('store_index_sha256'))[:12]}"
    stale.mkdir(parents=True, exist_ok=True)
    for name in ['manifest.json', 'progress.json', 'events.json', 'quality.json'] + [a.file for a in ARRAYS.values()]:
        if (root / name).exists():
            os.replace(root / name, stale / name)
    log(f'labels were built on store index {m.get("store_index_sha256")}: arrays moved to {stale}; parts reused '
        f'where their inputs are unchanged')


# ----------------------------------------------------------------------------- CLI entry points

def _sha(path) -> str | None:
    import hashlib
    if not path or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def build(args):
    from ..store import FrameStore
    lock = thermal.require_flight_lock_path(args.flight_lock)
    thermal.limit_threads(2, cv2=True)
    guard = thermal.ChunkGuard(lock)
    if getattr(args, 'overlays', None):
        from .hindsight import use_overlays_file
        _log('feature mask:', use_overlays_file(args.overlays), 'from', args.overlays)
    store_root = Path(args.store)
    store = FrameStore(store_root)
    # Hold the index in memory, not memory-mapped: a timing re-pose must be able to replace index.npy
    # (Windows refuses to replace a mapped file). Labels built on the old index are moved aside next time.
    store.index = np.array(store.index)
    sha = store.manifest.get('index_sha256') or store.index_sha256()
    root = labels_root(store_root)
    move_stale_arrays(root, sha, len(store))
    writer = LabelWriter(store_root, len(store), sha)
    parts_dir = root / 'parts'
    stages = tuple(args.stages or STAGES)
    run_ids = [r['run_id'] for r in store.runs if r.get('n_frames')]
    if getattr(args, 'runs', None):
        run_ids = [int(x) for x in args.runs]
    t0 = time.monotonic()
    ev_path = root / 'events.json'
    guard.before_chunk()
    events = build_events(store, store_root, args.events, args.extra_events)
    pub = [public_event(e) for e in events]
    from .impacts import write_events
    write_events(ev_path, pub, note='labelled events: lateral manifest (curated), events_f12.json (blind), store '
                                    'events (unlabelled) and geometric near passes. OFFLINE ONLY.')
    _log(f'events: {len(events)} ({sum(e["source"] == "blind_label" for e in events)} blind, '
         f'{sum(e["source"] == "lateral_manifest" for e in events)} curated, '
         f'{sum(e["kind"] == "near_pass" for e in events)} near passes)')
    scene = None
    if 'colliders' in stages and args.colliders:
        from .colliders import BoxScene
        scene = BoxScene(json.loads(Path(args.colliders).read_text(encoding='utf-8')))
    per_stage = {s: {} for s in SOURCE_STAGES}
    order = sorted(run_ids, key=lambda r: (store.runs[r].get('source') == 'run_video', r)) \
        if getattr(args, 'videos_last', True) else run_ids
    for rid in order:
        guard.before_chunk()
        # every stage part is checked against its inputs (code version, poses, stops, events, feature mask):
        # a part built from other inputs is rebuilt even when progress.json lists the run as done
        rd = RunData(store, rid, events, store_root)
        todo = [s for s in SOURCE_STAGES if s in stages
                and not part_current(parts_dir, s, rid, stage_inputs(rd, s))]
        for s in todo:
            t1 = time.monotonic()
            if s == 'colliders':
                out = stage_colliders(rd, scene)
            elif s == 'tube':
                out = stage_tube(rd)
            elif s == 'impacts':
                out = stage_impacts(rd)
            else:
                out = stage_hindsight(rd, store, guard, parts_dir)
            save_part(parts_dir / s / f'r{rid:05d}.npz', rd.rows, out,
                      dict(seconds=round(time.monotonic() - t1, 1), path=rd.path_source, inputs=stage_inputs(rd, s)))
            writer.mark_done(s, rid)
        for s in SOURCE_STAGES:
            if s in stages and not writer.is_done(s, rid):
                writer.mark_done(s, rid)
        if todo:
            _log(f'run {rid:4d} {rd.env[:22]:22s} {rd.rec["source_id"][:50]:50s} rows {len(rd.rows):6d} '
                 f'stages {",".join(todo)} ({time.monotonic() - t0:.0f} s)')
    combine_stats = dict(COMBINE_STATS)
    if 'combine' in stages:
        for rid in run_ids:
            st = combine_run(writer, parts_dir, rid)
            for k in combine_stats:
                combine_stats[k] += st[k]
            writer.mark_done('combine', rid)
    q = quality(store, parts_dir, events, run_ids, writer) if 'quality' in stages else {}
    from . import colliders as Cm, hindsight as Hm, impacts as Im, tube as Tm
    sources = dict(
        TUBE=dict(parameters=dict(horizon_s=Tm.TUBE_HORIZON_S, radius_m=Tm.TUBE_RADIUS_M, grid_rule=Tm.GRID_RULE,
                                  fan_rule=Tm.FAN_RULE)),
        HINDSIGHT=dict(parameters=dict(voxel_m=Hm.VOXEL_M, keyframe_offsets_s=Hm.KEYFRAME_OFFSETS_S,
                                       max_sigma_fraction=Hm.MAX_SIGMA_FRACTION, rules=Hm.RULES,
                                       feature_mask=Hm.mask_provenance(),
                                       feature_mask_file=Hm._OVERLAYS_FILE or 'haltere/obstacles/overlays.py')),
        IMPACT=dict(parameters=dict(window_s=Im.IMPACT_WINDOW_S, disc_radius_m=Im.IMPACT_DISC_RADIUS_M,
                                    visible_tube_m=Im.VISIBLE_TUBE_M, contact_offset_m=Im.CONTACT_OFFSET_M)),
        COLLIDER=dict(parameters=dict(assumption=Cm.ASSUMPTION, bundle_rings=Cm.BUNDLE_RINGS)),
        TEACHER=dict(parameters='M2: affine fit to L1-L3 anchors; not a label source in M1'))
    all_runs = [r['run_id'] for r in store.runs if r.get('n_frames')]
    complete = (not getattr(args, 'runs', None) and set(STAGES) <= set(stages)
                and all(writer.is_done(s, r) for s in SOURCE_STAGES for r in all_runs))
    m = writer.finalize(status='complete' if complete else 'building',
                        created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), code_commit=_git_commit(), sources=sources,
                        combine=dict(order=[s.name for s in COMBINE_ORDER], exact_tol=EXACT_TOL, min_known_frac=1.0,
                                     rule='interval intersection over all sources (order-independent)',
                                     wide_interval=f'UPPER hi if hi <= {WIDE_UPPER_MAX_M} m else LOWER lo',
                                     **combine_stats),
                        inputs=dict(colliders=dict(path=str(args.colliders), sha256=_sha(args.colliders)),
                                    lateral_manifest=dict(path=str(args.events), sha256=_sha(args.events)),
                                    events_f12=dict(path=str(args.extra_events), sha256=_sha(args.extra_events))),
                        quality=q, runs=len(run_ids),
                        build_scope='all runs' if complete else f'partial: runs {sorted(run_ids)[:50]}')
    (root / 'quality.json').write_text(json.dumps(q, indent=1, default=str) + '\n', encoding='utf-8')
    _log(json.dumps({k: v for k, v in q.items() if k != 'l2_vs_l3'} | {'l2_vs_l3_all': q.get('l2_vs_l3', {}).get('groups', {}).get('all')},
                    indent=1, default=str)[:4000])
    return m


def teacher(args):
    from ..store import FrameStore
    from .teacher import build_teacher_cache
    lock = thermal.require_flight_lock_path(args.flight_lock)
    thermal.limit_threads(2, torch=True, cv2=True)
    guard = thermal.ChunkGuard(lock, gpu=True)
    store = FrameStore(args.store)
    return build_teacher_cache(store, labels_root(args.store), guard)
