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
  combine    labels.intersect in COMBINE_ORDER; conflicts -> UNKNOWN (counted); clip; write arrays
  quality    K0b checks (label manifest 'quality')

Per-source outputs are kept in ``labels/parts/<stage>/rNNNNN.npz`` (rows, grid/fan value+kind) so the
combination can be recomputed. Data paths in runs.json are repo-relative; they are resolved against
$HALTERE_DATA_ROOT, then the repository root, then the store's grandparent directory.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .. import contract, thermal
from . import (COMBINE_ORDER, EXACT_TOL, LabelKind, LabelSource, LabelWriter, clip_fan, clip_grid,
               fan_blocked_targets, intersect, labels_root, load_events, validate_event)

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
        cols = {k: [] for k in TEL_COLS}
        for r in rd:
            for k in TEL_COLS:
                v = r.get(k)
                try:
                    cols[k].append(float(v) if v not in (None, '', 'nan') else np.nan)
                except ValueError:
                    cols[k].append(np.nan)
    c = {k: np.asarray(v, np.float64) for k, v in cols.items()}
    if len(c['wall']) < 5:
        return None
    tel = dict(wall=c['wall'], ts=c['ts'], phase=c['phase'], pos=np.c_[c['x'], c['y'], c['z']],
               vel=np.c_[c['vx'], c['vy'], c['vz']], quat=np.c_[c['qw'], c['qx'], c['qy'], c['qz']])
    ok = np.isfinite(tel['wall']) & np.isfinite(tel['pos']).all(1)
    tel = {k: v[ok] for k, v in tel.items()}
    o = np.argsort(tel['wall'], kind='stable')
    return {k: v[o] for k, v in tel.items()}


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
        self.stops = np.sort(np.r_[np.asarray(stops, float), reset_times(self.tel)])

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
            tel_cache[run_id] = load_telemetry(data_path(store.runs[run_id].get('telemetry_csv'), store_root))
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
            if tel is not None and e['kind'] != 'near_pass':
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
        if tel is not None:
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
    rows, vmap, est, tracks = H.build_flight_map(store, rd.run_id, guard, rows=rd.rows)
    assert np.array_equal(rows, rd.rows)
    stops = [s for s in rd.stops]
    vmap = H.clear_flown_path(vmap, rd.path_t, rd.path_pos, stops)
    mp = parts_dir / 'hindsight_map' / f'r{rd.run_id:05d}.npz'
    vmap.save(mp)
    np.savez_compressed(parts_dir / 'hindsight_map' / f'r{rd.run_id:05d}.est.npz',
                        **{k: getattr(est, k) for k in ('frame', 'track', 'uv', 'point', 'range_m', 'sigma_m', 'offset_s')})
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
    """{LabelSource: (value, kind) arrays} -> (value, kind, src, conflict) folded in COMBINE_ORDER."""
    shape = next(iter(parts.values()))[0].shape
    v = np.full(shape, np.nan)
    k = np.zeros(shape, np.uint8)
    src = np.zeros(shape, np.uint8)
    conflict = np.zeros(shape, bool)
    for s in COMBINE_ORDER:
        if s not in parts:
            continue
        sv, sk = parts[s]
        v, k, c = intersect(v, k, sv, sk, tol=EXACT_TOL)
        conflict |= c
        src |= np.where((sk != LabelKind.UNKNOWN) & ~c, np.uint8(int(s)), np.uint8(0))
    k = np.where(conflict, LabelKind.UNKNOWN, k).astype(np.uint8)
    v = np.where(conflict, np.nan, v)
    src = np.where(conflict | (k == LabelKind.UNKNOWN), 0, src).astype(np.uint8)
    return v, k, src, conflict


def combine_run(writer: LabelWriter, parts_dir: Path, run_id: int) -> dict:
    stats = dict(conflicts_grid=0, conflicts_fan=0)
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
    fv, fk, fs, fc = combine_sources({s: (p['fan_v'], p['fan_k']) for s, p in loaded.items()})
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


def quality(store, parts_dir: Path, events: list[dict], run_ids, writer: LabelWriter | None = None) -> dict:
    """K0b: L2 vs colliders, L2 vs L3, near-travel fan density (Minus Two / Pine Valley)."""
    from .corridors import world_to_pixels
    q = dict(thresholds=K0B)
    # (a) L2 vs colliders on the box course
    rel = {1: [], 3: []}
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
    r1 = np.concatenate(rel[1]) if rel[1] else np.zeros(0)
    r3 = np.concatenate(rel[3]) if rel[3] else np.zeros(0)
    rall = np.r_[r1, r3]
    q['l2_vs_colliders'] = dict(
        cells_exact=int(len(r1)), cells_upper=int(len(r3)),
        median_rel_err_exact=float(np.median(r1)) if len(r1) else None,
        median_rel_err_upper=float(np.median(r3)) if len(r3) else None,
        median_rel_err_all=float(np.median(rall)) if len(rall) else None,
        frac_within_10pct=float(np.mean(rall <= 0.10)) if len(rall) else None,
        passed=bool(len(rall) and np.median(rall) <= K0B['l2_vs_colliders_median_rel']),
        definition='box-course grid cells with collider EXACT range in [2, 10] m and an L2 EXACT or UPPER value')
    # (b) L2 vs L3 in [T - 2.0, T - 0.5] s where the impact point projects into the image and L2 labels its cell
    per_env, per_event = {}, []
    for e in events:
        rid = e.get('_run_id')
        if rid is None or e.get('point_w') is None or e['kind'] not in ('terminal_impact', 'contact') or not e.get('t_wall'):
            continue
        a = load_part(parts_dir / 'hindsight' / f'r{rid:05d}.npz')
        if a is None or a['empty']:
            continue
        rows = a['rows']
        ix = store.index[rows]
        T = float(e['t_wall'])
        sel = np.flatnonzero((ix['t_wall'] >= T - 2.0) & (ix['t_wall'] <= T - 0.5))
        n_view, n_both, n_ok = 0, 0, 0
        ratios = []
        for i in sel:
            p = np.asarray(e['point_w'], float)
            uv, rng, ok = world_to_pixels(p[None], ix['pos'][i].astype(float), ix['quat'][i].astype(float))
            if not contract.in_image(uv, ok)[0]:
                continue
            n_view += 1
            r, c = _cell_of(uv[0])
            if a['grid_k'][i, r, c] in (LabelKind.EXACT, LabelKind.UPPER):
                n_both += 1
                ratio = float(a['grid_v'][i, r, c]) / float(rng[0])
                ratios.append(ratio)
                n_ok += abs(ratio - 1) <= K0B['l2_vs_l3_within']
        per_event.append(dict(event_id=e['event_id'], run=e['run'], env=e['env'], kind=e['kind'], frames_in_view=n_view,
                              frames_both=n_both, frames_within_15pct=n_ok,
                              median_ratio=float(np.median(ratios)) if ratios else None))
        d = per_env.setdefault(e['env'], dict(frames_in_view=0, frames_both=0, frames_within_15pct=0))
        d['frames_in_view'] += n_view
        d['frames_both'] += n_both
        d['frames_within_15pct'] += n_ok
    for d in per_env.values():
        d['frac_within_15pct'] = d['frames_within_15pct'] / d['frames_both'] if d['frames_both'] else None
        d['coverage'] = d['frames_both'] / d['frames_in_view'] if d['frames_in_view'] else None
    tot = {k: sum(d[k] for d in per_env.values()) for k in ('frames_in_view', 'frames_both', 'frames_within_15pct')}
    frac = tot['frames_within_15pct'] / tot['frames_both'] if tot['frames_both'] else None
    q['l2_vs_l3'] = dict(per_env=per_env, per_event=per_event, total=tot, frac_within_15pct=frac,
                         passed=bool(frac is not None and frac >= K0B['l2_vs_l3_frac']),
                         definition=('frames in [T-2.0, T-0.5] s where the contact point projects into the image; '
                                     '"both" = the L2 cell holding it is EXACT/UPPER; ratio = L2 value / true range'))
    # (c) near-travel fan density on Minus Two and Pine Valley (combined labels)
    q['fan_near_travel'] = {}
    if writer is not None:
        yaw_ok = np.abs(contract.FAN_YAW_DEG) <= NEAR_TRAVEL_YAW
        from ..splits import ENV_CODE
        for env in ('Minus Two', 'Pine Valley', 'Autumn Fields', 'Straw Bale', BOX_ENV):
            rows = np.flatnonzero(store.index['env'] == ENV_CODE[env])
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
    store_root = Path(args.store)
    store = FrameStore(store_root)
    sha = store.manifest.get('index_sha256') or store.index_sha256()
    writer = LabelWriter(store_root, len(store), sha)
    root = labels_root(store_root)
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
    for rid in run_ids:
        todo = [s for s in SOURCE_STAGES if s in stages and not writer.is_done(s, rid)]
        if not todo:
            continue
        guard.before_chunk()
        rd = RunData(store, rid, events, store_root)
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
                      dict(seconds=round(time.monotonic() - t1, 1), path=rd.path_source))
            writer.mark_done(s, rid)
        _log(f'run {rid:4d} {rd.env[:22]:22s} {rd.rec["source_id"][:50]:50s} rows {len(rd.rows):6d} '
             f'stages {",".join(todo)} ({time.monotonic() - t0:.0f} s)')
    combine_stats = dict(conflicts_grid=0, conflicts_fan=0)
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
                                       max_sigma_fraction=Hm.MAX_SIGMA_FRACTION, rules=Hm.RULES)),
        IMPACT=dict(parameters=dict(window_s=Im.IMPACT_WINDOW_S, disc_radius_m=Im.IMPACT_DISC_RADIUS_M,
                                    visible_tube_m=Im.VISIBLE_TUBE_M, contact_offset_m=Im.CONTACT_OFFSET_M)),
        COLLIDER=dict(parameters=dict(assumption=Cm.ASSUMPTION, bundle_rings=Cm.BUNDLE_RINGS)),
        TEACHER=dict(parameters='M2: affine fit to L1-L3 anchors; not a label source in M1'))
    m = writer.finalize(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), code_commit=_git_commit(), sources=sources,
                        combine=dict(order=[s.name for s in COMBINE_ORDER], exact_tol=EXACT_TOL, min_known_frac=1.0,
                                     wide_interval='lower (labels.intersect default)', **combine_stats),
                        inputs=dict(colliders=dict(path=str(args.colliders), sha256=_sha(args.colliders)),
                                    lateral_manifest=dict(path=str(args.events), sha256=_sha(args.events)),
                                    events_f12=dict(path=str(args.extra_events), sha256=_sha(args.extra_events))),
                        quality=q, runs=len(run_ids))
    (root / 'quality.json').write_text(json.dumps(q, indent=1, default=str) + '\n', encoding='utf-8')
    _log(json.dumps({k: v for k, v in q.items() if k != 'l2_vs_l3'} | {'l2_vs_l3_total': q.get('l2_vs_l3', {}).get('total')},
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
