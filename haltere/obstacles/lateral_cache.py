"""The lateral study's run caches as a pseudo frame store, to reproduce its numbers through the harness.

OFFLINE REPRODUCTION ONLY. The lateral study (session scratchpad ``lateral/data``: manifest.json and
cache/<run>.npz with 240 x 135 grey frames, aligned telemetry, the HUD-free mask and the logged cue)
scored split looming (1/11 lateral impacts with a >= 1 s correct lead), constant answers (right 7/10
on the mono-depth study's 10 scored impacts) and the pretrained metric depth scale (pred/true 2.62 at
2-4 m, 1.72 at 4-7 m). ``LateralCacheStore`` exposes those caches through the attributes the harness
reads (index with INDEX_DTYPE fields, runs, side tables) so ``reproduce`` can score them with the same
functions that score the real store (``evaluate.e4_event_results``, ``clean_window_metrics``,
``e1_samples``). Frames are grey 240 x 135 only (``frames`` raises); B2 at 252 x 448 needs RGB frames
decoded from the run videos (see ``b2_pred``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .splits import ENV_CODE
from .store import INDEX_DTYPE, Grade, Source, empty_index

# The lateral study's scoring labels (scratchpad lateral/cand-split-looming/evalcore.py LAT): primary free
# side ('either' = both count) and accepted sides.
LATERAL_SCORING = {
    'minus-brain03-01': ('right', ['right', 'left']),
    'minus-brain04-02': ('right', ['right', 'left']),
    'minus-brain05-01': ('either', ['right', 'left']),
    'minus-fast6-01': ('left', ['left']),
    'pine-fast6-ttc-01': ('right', ['right']),
    'pine-brain05-02': ('right', ['right']),
    'pine-fast6-01': ('right', ['right']),
    'pine-brain03-01': ('left', ['left', 'right']),
    'straw-brain05-01': ('right', ['right']),
    'straw-brain05-trim-02': ('left', ['left']),
    'straw-brain05-gov-01': ('right', ['right']),
}
NOT_LATERAL = {'pine-fast6-loom-01': 'up', 'pine-brain04-01': 'unknown', 'straw-fast6-01': 'up',
               'straw-brain05-trim-01': 'up'}
# The mono-depth study scored 10 of these (pine-fast6-01 counted as 'up'): the constant-side reference set.
MONO_DEPTH_UNSCORED = ('pine-fast6-01',)
RUN_PREFIX_ENV = (('minus', 'Minus Two'), ('pine', 'Pine Valley'), ('straw', 'Straw Bale'),
                  ('loop', 'Drawing Board loop v2'))
FLIGHT_PREFIX = 'fast-stack-20260923/'


def run_env(run: str) -> str:
    for p, env in RUN_PREFIX_ENV:
        if run.startswith(p):
            return env
    raise KeyError(f'no environment for lateral run {run!r}')


def _unique_keys(items, radius_m: float = 3.0) -> list[str]:
    """Greedy clustering of (env, obstacle word, position) into stable unique-obstacle keys."""
    keys, seen = [], []
    for env, obstacle, pos in items:
        word = (obstacle or 'unknown').split(' ')[0].split('(')[0].lower()
        found = None
        for k, e, w, p in seen:
            if e == env and w == word and pos is not None and p is not None and np.linalg.norm(pos - p) <= radius_m:
                found = k
                break
        if found is None:
            found = f"{env.lower().replace(' ', '-')}/{word}-" + (
                f'{pos[0]:.1f}-{pos[1]:.1f}' if pos is not None else str(len(seen)))
            seen.append((found, env, word, pos))
        keys.append(found)
    return keys


class LateralCacheStore:
    """Read-only pseudo store over the lateral study's caches (one run per cache file)."""

    def __init__(self, data_dir):
        self.root = Path(data_dir)
        self.manifest_lateral = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        man = self.manifest_lateral
        names = sorted({i['run'] for i in man['impacts']} | {c['run'] for c in man['clean_windows']})
        grade = {i['run']: i.get('alignment', {}).get('grade') for i in man['impacts']}
        grade.update({c['run']: c.get('alignment_grade') for c in man['clean_windows'] if c['run'] not in grade})
        parts, self.runs, self._grey, self._usable = [], [], [], []
        for k, run in enumerate(names):
            z = np.load(self.root / 'cache' / f'{run}.npz')
            n = len(z['t_wall'])
            ix = empty_index(n)
            ix['run_id'] = k
            ix['slot'] = np.arange(n)
            ix['env'] = ENV_CODE[run_env(run)]
            ix['source'] = Source.RUN_VIDEO
            ix['grade'] = Grade.FAIR if grade.get(run) == 'fair' else Grade.GOOD
            ix['t_wall'] = z['t_wall']
            ix['t_phase'] = z['phase_ctrl']
            ix['pos'] = z['pos']
            ix['quat'] = z['quat']
            ix['vel'] = z['v_world']
            ix['omega'] = z['omega_b']
            cue = np.asarray(z['cue_uv'], np.float64)
            cue = np.where((cue < 0).any(axis=1, keepdims=True), np.nan, cue)
            ix['cue_uv'] = cue
            ix['cue_src'] = np.where(np.isfinite(cue).all(axis=1), 1, 0)
            parts.append(ix)
            self._grey.append(z['grey240'])
            self._usable.append(z['usable240'])
            self.runs.append(dict(run_id=k, source_id=FLIGHT_PREFIX + run, flight=FLIGHT_PREFIX + run, aliases=[run],
                                  env=run_env(run), source='run_video', secondary=False, n_frames=n))
        self.index = np.concatenate(parts) if parts else np.zeros(0, INDEX_DTYPE)
        self._offset = np.r_[0, np.cumsum([len(p) for p in parts])]
        self.manifest = dict(index_sha256=self.index_sha256(), note='lateral study caches (reproduction adapter)')
        self._events = self._build_events()
        self._windows = [dict(run_id=next(r['run_id'] for r in self.runs if r['aliases'][0] == c['run']),
                              alias=c['run'], start_phase_s=float(c['start_s']), end_phase_s=float(c['end_s']),
                              criterion=c.get('criterion'), origin='lateral manifest')
                         for c in man['clean_windows']]

    def __len__(self):
        return len(self.index)

    def index_sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.index).tobytes()).hexdigest()

    def side_table(self, name: str):
        if name == 'clean_windows.json':
            return dict(windows=self._windows)
        return None

    def events(self) -> list[dict]:
        return [dict(e) for e in self._events]

    def frames(self, rows):
        raise NotImplementedError('the lateral caches hold 240x135 grey frames only')

    def grey_frames(self, rows):
        """[(grey 135 x 240 uint8, usable bool mask)] for store rows (the study's exact B1 input)."""
        out = []
        for r in np.asarray(rows, np.int64):
            k = int(self.index['run_id'][r])
            s = int(self.index['slot'][r])
            use = np.unpackbits(self._usable[k][s], axis=-1)[..., :240].astype(bool)
            out.append((self._grey[k][s], use))
        return out

    def _build_events(self) -> list[dict]:
        man = self.manifest_lateral
        items, raw = [], []
        for imp in man['impacts']:
            run = imp['run']
            prim, acc = LATERAL_SCORING.get(run, (NOT_LATERAL.get(run, 'unknown'), []))
            raw.append(dict(run=run, kind='terminal_impact', t_phase=float(imp['impact_phase_s']),
                            drone_pos_w=[float(v) for v in imp['impact_pos']], speed_mps=imp.get('speed_pre_mps'),
                            obstacle=imp['obstacle'], obstacle_side=imp.get('obstacle_side'), primary_free_side=prim,
                            accepted_free_sides=list(acc), lateral=run in LATERAL_SCORING,
                            confidence=imp.get('confidence')))
            items.append((run_env(run), imp['obstacle'], np.asarray(imp['impact_pos'], float)))
        for c in man.get('non_fatal_contacts', []):
            raw.append(dict(run=c['run'], kind='contact', t_phase=float(c['phase_s']), drone_pos_w=None,
                            obstacle=c['obstacle'], obstacle_side=c.get('obstacle_side'), primary_free_side='unknown',
                            accepted_free_sides=[], lateral=False))
            items.append((run_env(c['run']), c['obstacle'], None))
        for c in man.get('near_passes', []):
            a, b = c['phase_s']
            raw.append(dict(run=c['run'], kind='near_pass', t_phase=float(b) - 0.4, drone_pos_w=None,
                            obstacle=c['obstacle'], obstacle_side=c.get('obstacle_side'), primary_free_side='unknown',
                            accepted_free_sides=[], lateral=False))
            items.append((run_env(c['run']), c['obstacle'], None))
        keys = _unique_keys(items)
        # the PD near pass is the Minus pillar of the three brain impacts
        pillar = next((k for k, e in zip(keys, raw) if e['kind'] == 'terminal_impact' and e['obstacle'] == 'pillar'),
                      None)
        out = []
        for i, (e, key) in enumerate(zip(raw, keys)):
            if e['kind'] == 'near_pass' and 'pillar' in e['obstacle'] and pillar:
                key = pillar
            out.append(dict(event_id=i, store_event_id=-1, flight=FLIGHT_PREFIX + e['run'], env=run_env(e['run']),
                            point_w=None, normal_w=None, unique_obstacle=key, oracle_route=False,
                            source='lateral_manifest', blind=False, labeller='lateral study', **e))
        return out


# ----------------------------------------------------------------------------- reproduction

def _counts(res, runs=None):
    sel = [r for r in res if runs is None or r['run'] in runs]
    return dict(n=len(sel), correct_ge_1s=sum(r['success'] for r in sel),
                correct_ge_0p5s=sum(r['correct_lead'] >= 0.5 for r in sel),
                correct_permissive=sum(r['success_permissive'] for r in sel),
                wrong_any=sum(r['wrong_any'] for r in sel), wrong_held_0p3s=sum(r['wrong'] for r in sel),
                leads={r['run']: round(r['correct_lead'], 3) for r in sel})


def prior_depth_prediction(store: LateralCacheStore, depth_dir) -> tuple:
    """PredictionSet (grid_q50 from the prior 160 x 90 optical-depth maps, 5 x 5 min -> 18 x 32 ranges) and the
    dense maps {row: (90, 160) optical depth} for the study's own 3 x 3-pixel scale check."""
    from .evaluate import PredictionSet
    depth_dir = Path(depth_dir)
    W, H, F = 160, 90, 400.0 * 160 / 1280.0
    v, u = np.mgrid[:H, :W].astype(np.float64)
    norm = np.sqrt(1 + ((u + 0.5 - W / 2) / F) ** 2 + ((v + 0.5 - H / 2) / F) ** 2)
    rows, grids, dense = [], [], {}
    for run in store.runs:
        p = depth_dir / f"{run['aliases'][0]}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        base = store._offset[run['run_id']]
        d = z['depth'].astype(np.float64)
        rng = d * norm[None]
        g = rng.reshape(len(d), 18, 5, 32, 5).min(axis=(2, 4))
        for k, ci in enumerate(z['cache_idx']):
            if np.isfinite(d[k]).any():
                rows.append(base + int(ci))
                grids.append(g[k])
                dense[base + int(ci)] = d[k]
    order = np.argsort(rows)
    rows = np.asarray(rows, np.int64)[order]
    grid = np.asarray(grids, np.float32)[order]
    pred = PredictionSet('B2-prior-336x602-maps', 'baseline', True, rows, baseline_id='B2',
                         sha256=hashlib.sha256(b'lateral cand-mono-depth depth/ (metric-indoor-336.ts)').hexdigest(),
                         arrays=dict(grid_q50=grid, grid_q20=grid.copy())).validate()
    return pred, dense


def prior_scale_check(store: LateralCacheStore, dense: dict, thresholds: dict, *, point='drone_pos_w') -> dict:
    """The study's scale_check.py on the adapter: optical depth, 3 x 3 window at 160 x 90, chord filter."""
    from . import contract
    W, H, F = 160, 90, 400.0 * 160 / 1280.0
    c_idx = store.index
    pred, true = [], []
    for ev in store.events():
        if ev['kind'] != 'terminal_impact' or ev.get(point) is None:
            continue
        rid = next(r['run_id'] for r in store.runs if r['aliases'][0] == ev['run'])
        rows = np.flatnonzero(c_idx['run_id'] == rid)
        P = np.asarray(ev[point], np.float64)
        for r in rows:
            if r not in dense:
                continue
            tti = ev['t_phase'] - float(c_idx['t_phase'][r])
            if not (0.15 <= tti <= 3.0):
                continue
            chord = P - c_idx['pos'][r].astype(np.float64)
            dist = np.linalg.norm(chord)
            vel = c_idx['vel'][r].astype(np.float64)
            sp = np.linalg.norm(vel)
            if not (0.5 <= dist <= 20) or sp < 1:
                continue
            if np.degrees(np.arccos(np.clip(chord @ vel / (dist * sp), -1, 1))) > 15:
                continue
            pc = np.einsum('ji,j->i', contract.camera_to_world(c_idx['quat'][r].astype(np.float64)), chord)
            if pc[2] < 0.3:
                continue
            u, vv = W / 2 + F * pc[0] / pc[2], H / 2 + F * pc[1] / pc[2]
            if not (1 <= u < W - 1 and 1 <= vv < H - 1):
                continue
            pred.append(float(dense[r][int(vv) - 1:int(vv) + 2, int(u) - 1:int(u) + 2].min()))
            true.append(float(pc[2]))
    pred, true = np.asarray(pred), np.asarray(true)
    out = dict(n=len(pred))
    for lo, hi in thresholds['E1']['bins_m']:
        m = (true >= lo) & (true < hi)
        out[f'{lo:g}-{hi:g}'] = dict(n=int(m.sum()), median=float(np.median(pred[m] / true[m])) if m.any() else None)
    return out


def reproduce(data_dir, thresholds: dict, *, depth_cache=None, b2_pred=None, b1: bool = False, guard=None,
              log=None) -> dict:
    """Score B0, B1 and B2 on the lateral caches with the harness and compare with the study's numbers."""
    from . import evaluate as ev
    say = log or (lambda m: None)
    store = LateralCacheStore(data_dir)
    events = store.events()
    rows = np.arange(len(store), dtype=np.int64)
    lat_runs = set(LATERAL_SCORING)
    mono_runs = lat_runs - set(MONO_DEPTH_UNSCORED)
    out = dict(data=str(data_dir), store_rows=len(store), events=len(events), results={}, summary={})
    timings = dict(study=(0.0, None), deployed=(thresholds['timing']['latency_s'], thresholds['timing']['rate_hz']))
    for side in (ev.Side.RIGHT, ev.Side.LEFT):
        pred = ev.baseline_b0(store, rows, side)
        for tname, (lat, rate) in timings.items():
            codes = ev.decide_side(pred, store, thresholds, latency_s=lat, rate_hz=rate)
            res = ev.e4_event_results(pred, store, events, thresholds, latency_s=lat, rate_hz=rate, codes=codes)
            out['results'][f'{pred.name} {tname}'] = dict(split_looming_set_11=_counts(res, lat_runs),
                                                          mono_depth_set_10=_counts(res, mono_runs))
    s = out['results']
    out['summary']['B0_right_mono_depth_set'] = (f"{s['B0-right study']['mono_depth_set_10']['correct_ge_1s']}/"
                                                 f"{s['B0-right study']['mono_depth_set_10']['n']} (target 7/10)")
    out['summary']['B0_right_split_looming_set'] = (f"{s['B0-right study']['split_looming_set_11']['correct_ge_1s']}/"
                                                    f"{s['B0-right study']['split_looming_set_11']['n']}")
    out['summary']['B0_left_mono_depth_set'] = (f"{s['B0-left study']['mono_depth_set_10']['correct_ge_1s']}/"
                                                f"{s['B0-left study']['mono_depth_set_10']['n']} (study: 4/10)")
    out['summary']['B0_right_deployed'] = (f"{s['B0-right deployed']['mono_depth_set_10']['correct_ge_1s']}/"
                                           f"{s['B0-right deployed']['mono_depth_set_10']['n']}")
    if b1:
        say('B1: streaming split looming over every cached frame')
        pred = ev.baseline_b1(store, rows, guard=guard, frame_source=store.grey_frames, max_gap_s=0.25, log=say)
        for tname, (lat, rate) in timings.items():
            codes = ev.decide_side(pred, store, thresholds, latency_s=lat, rate_hz=rate)
            res = ev.e4_event_results(pred, store, events, thresholds, latency_s=lat, rate_hz=rate, codes=codes)
            e6 = ev.e6_clean_windows(pred, store, thresholds, latency_s=lat, fold='F12', rate_hz=rate,
                                     events=events, codes=codes)
            pooled = next(r for r in e6 if r['variant'].startswith('pooled all envs'))['value']
            near = {r['variant']: dict(toward_longest_s=r['value']['toward_longest_s'],
                                       toward_fraction=r['value']['toward_fraction'], passed=r['passed'])
                    for r in e6 if r['variant'].startswith('near pass')}
            out['results'][f'B1 {tname}'] = dict(split_looming_set_11=_counts(res, lat_runs),
                                                 clean=dict(total_min=pooled['total_min'],
                                                            episodes=pooled['side_episodes'],
                                                            episodes_per_min=pooled['side_episodes_per_min'],
                                                            active_fraction=pooled['side_active_fraction'],
                                                            any_episodes_per_min=pooled['any_episodes_per_min']),
                                                 near_passes=near)
        r = out['results']['B1 study']
        out['summary']['B1_study_timing'] = (f"{r['split_looming_set_11']['correct_ge_1s']}/"
                                             f"{r['split_looming_set_11']['n']} lateral, wrong {r['split_looming_set_11']['wrong_any']}, "
                                             f"clean {r['clean']['episodes_per_min']:.2f}/min (study: 1/11, 0 wrong, 0.31/min)")
        r = out['results']['B1 deployed']
        out['summary']['B1_deployed'] = (f"{r['split_looming_set_11']['correct_ge_1s']}/"
                                         f"{r['split_looming_set_11']['n']} lateral, clean "
                                         f"{r['clean']['episodes_per_min']:.2f}/min")
    b2_sets = []
    if depth_cache is not None:
        pred, dense = prior_depth_prediction(store, depth_cache)
        out['results']['B2 prior maps, study scale check'] = prior_scale_check(store, dense, thresholds)
        b2_sets.append(('B2 prior 336x602 maps', pred))
        sc = out['results']['B2 prior maps, study scale check']
        out['summary']['B2_prior_study_method'] = (f"2-4 m {sc['2-4']['median']:.2f} (n={sc['2-4']['n']}), "
                                                   f"4-7 m {sc['4-7']['median']:.2f} (n={sc['4-7']['n']}) "
                                                   '(study: 2.62 n=72, 1.72 n=98)')
    if b2_pred is not None:
        b2_sets.append(('B2 252x448 fp16', ev.PredictionSet.load(b2_pred)))
    nb = int(thresholds['E1'].get('neighbourhood_cells', 1))
    main_v = f'contract E1 ({nb}x{nb} cells, point=drone_pos_w, all in-view frames)'
    for name, pred in b2_sets:
        for variant, kw in ((main_v, dict(point='drone_pos_w', prior_filter=False)),
                            ('contract E1 + study chord filter', dict(point='drone_pos_w', prior_filter=True)),
                            ('3x3-cell window (plan text, not used)',
                             dict(point='drone_pos_w', prior_filter=False, neighbourhood=3))):
            samples = ev.e1_samples(pred, store, events, thresholds, **kw)
            summ = ev.e1_summary(samples, thresholds, n_boot=500)
            out['results'][f'{name}: {variant}'] = summ
        b = out['results'][f'{name}: contract E1 + study chord filter']['bins']
        a = out['results'][f'{name}: {main_v}']['bins']
        c3 = out['results'][f'{name}: 3x3-cell window (plan text, not used)']['bins']
        tgt = thresholds['reproduce_prior']['B2_E1_median_ratio']
        tol = thresholds['reproduce_prior']['tolerance']
        fmt = lambda d, k: f"{d[k]['median']:.2f} (n={d[k]['n']})" if d.get(k, {}).get('median') is not None else 'n/a'
        out['summary'][f'{name} E1'] = (f"contract: 2-4 m {fmt(a, '2-4')}, 4-7 m {fmt(a, '4-7')}; with chord filter: "
                                        f"2-4 m {fmt(b, '2-4')}, 4-7 m {fmt(b, '4-7')}; 3x3-cell window: "
                                        f"2-4 m {fmt(c3, '2-4')}, 4-7 m {fmt(c3, '4-7')}")
        within = {k: (a.get(k, {}).get('median') is not None
                      and abs(a[k]['median'] / tgt[k] - 1) <= tol) for k in tgt}
        out['summary'][f'{name} E1 within +-{tol:.0%} of 2.62/1.72 (contract)'] = within
    return out
