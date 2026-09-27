"""Gap pilot side commitment (configs/obstacles/gap_pilot.json version 3): pilot-level offline evaluation.

OFFLINE ONLY. The runtime rule is haltere.liftoff.gap_aim (the pilot's confirmation of the per-frame gap-cue shift);
the per-frame cue (haltere.vision.gap_cue, configs/obstacles/gap_cue.json) is unchanged. Nothing here is flight
evidence: every replay is open loop (the recorded motion does not respond to the replayed aim).

Two replays feed the pilot's gap aim with the gap-cue samples a flight saw, or would have seen:

- Live logs of obstacle-stack flights: every logged controller tick through FastRaceCue with the deployed stack
  (`haltere.obstacles.vertical_replay` ``--stack on --near-on-path --gap-pilot <declaration>``): the logged gap
  samples (now with their near_on_path), looming samples and ring cues, as the runner fed them.
- Flights without live samples: the gap-cue evaluation's per-frame decisions (`gap_cue_eval.flight_sequence` /
  ``store_sequence``, teacher basis = the runtime 336 x 602 input) become samples received ``gap_latency_s`` after
  capture; each frame's ring bearing reaches the pilot ``cue_latency_s`` after capture (ring reconciliation: a
  checkpoint switch releases the shift). `GapAim` alone runs at 100 Hz: no flag clearance, no terrain votes (no
  governor), no lag turn.

Hindsight quantities (impact times, pillar boxes from the scene colliders / impact anatomy, checkpoint-switch times
from the logged cue) select and score ticks only; the pilot sees none of them. Gates: configs/obstacles/
gap_commit_gates.json, frozen with the version 3 declaration before any gate is scored (``score`` refuses unfrozen
or edited files and records both hashes).

Commands (``python -m haltere.obstacles.gap_commit_eval <cmd>``):
  seq     per-frame gap-cue decisions of the offline sequences -> OUT/seq/<name>.npz (CPU)
  score   the frozen gates from OUT/seq and the live replay files -> OUT/results.json
  dev     report-only statistics on development runs (no gate)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GATES_PATH = REPO_ROOT/'configs'/'obstacles'/'gap_commit_gates.json'
OB = REPO_ROOT/'configs'/'obstacles'
EPS = 1e-9


def _log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- declarations and gates

def load_json_frozen(path, *, require_frozen=True):
    from ..liftoff.gap_stack import config_sha256
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = config_sha256(obj)
    if obj.get('frozen') and obj.get('sha256') != digest:
        raise RuntimeError(f'{path} changed after freezing (sha256 mismatch)')
    if require_frozen and obj.get('frozen') is not True:
        raise RuntimeError(f'{path} is not frozen')
    return obj, digest


def aim_config(path):
    from ..liftoff.gap_aim import GapAimConfig
    obj, digest = load_json_frozen(path)
    return GapAimConfig.from_dict(obj['pilot']), obj, digest


# ----------------------------------------------------------------------------- offline sequences

SEQ_KEYS = ('t', 'raw', 'valid', 'kind', 'ring', 'r_peak', 'r_ring', 'near_on_path', 'lr', 'pos', 'vel', 'tti')


def build_sequences(specs, cache_dir, out_dir, gates):
    """Per-frame decisions of each offline sequence (teacher basis) -> out_dir/seq/<name>.npz (resumable)."""
    from ..vision import gap_cue as gc
    from . import gap_cue_eval as ge
    params, cfg, _ = gc.load_config(require_frozen=True)
    params = replace(params, enabled=True)
    layers = tuple(cfg['mask_layers'])
    models = gc.load_response_models()
    lat = float(gates['cue_age_s'])
    cache_dir, out = Path(cache_dir), Path(out_dir)/'seq'
    out.mkdir(parents=True, exist_ok=True)
    store = None
    for spec in specs:
        name = spec['name']
        path = out/f'{name}.npz'
        if path.exists():
            continue
        motor = models[gates['motors'][spec['motor']]]
        if 'store_run_id' in spec:
            if store is None:
                S = ge.data_root()/'runs'/'obstacle-store-v1'
                store = dict(index=np.load(S/'index.npy', mmap_mode='r'),
                             teacher=np.load(S/'labels'/'teacher'/'disparity.npy', mmap_mode='r'),
                             masks=ge._npz(cache_dir/'store_masks.npz'))
            seq = ge.store_sequence(int(spec['store_run_id']), 'teacher', params, layers, motor, lat, store)
        else:
            seq = ge.flight_sequence(spec['flight'], 'teacher', params, layers, motor, lat, cache_dir)
        arrays = dict(t=np.asarray(seq['t'], float), raw=np.asarray(seq['raw'], float),
                      valid=np.asarray(seq['valid'], bool), kind=np.asarray(seq['kind'], dtype='U12'),
                      ring=np.asarray(seq['ring'], float), r_peak=np.asarray(seq['r_peak'], float),
                      r_ring=np.asarray(seq['r_ring'], float), near_on_path=np.asarray(seq['near_on_path'], bool),
                      lr=np.asarray(seq['lr'], float), pos=np.asarray(seq['pos'], float),
                      vel=np.asarray(seq['vel'], float), tti=np.asarray(seq['tti'], float))
        np.savez_compressed(path, **arrays)
        _log(f'{name}: {len(arrays["t"])} frames')


def load_seq(out_dir, name):
    with np.load(Path(out_dir)/'seq'/f'{name}.npz') as z:
        return {k: z[k] for k in z.files}


def pilot_replay(seq, config, *, gap_latency_s, cue_latency_s, dt=.01):
    """GapAim alone over an offline sequence at 1/dt Hz (see the module docstring). Per-tick arrays: now, target,
    applied, mode ('obstacle', 'terrain', ''), commit (-1, 0, 1), ring (latest ring bearing reaching the pilot),
    pos (the recorded position interpolated at the tick)."""
    from ..liftoff.gap_aim import GapAim
    t = np.asarray(seq['t'], float)
    aim = GapAim(config)
    ring_events = [(float(t[k])+cue_latency_s, float(seq['ring'][k])) for k in range(len(t))]
    samples = []
    for k in range(len(t)):
        valid = bool(seq['valid'][k])
        raw = float(seq['raw'][k])
        samples.append((float(t[k])+gap_latency_s,
                        dict(time=float(t[k]), shift=raw if valid and np.isfinite(raw) else 0., valid=valid,
                             kind=str(seq['kind'][k]), ring_deg=float(seq['ring'][k]) if np.isfinite(seq['ring'][k])
                             else None, lr=float(seq['lr'][k]) if np.isfinite(seq['lr'][k]) else None,
                             near_on_path=bool(seq['near_on_path'][k]))))
    start, end = float(t[0]), float(t[-1])+max(gap_latency_s, cue_latency_s)+.5
    n = int(np.ceil((end-start)/dt))+1
    out = dict(now=np.empty(n), target=np.empty(n), applied=np.empty(n), mode=np.empty(n, dtype='U8'),
               commit=np.zeros(n), ring=np.full(n, np.nan), conflict=np.zeros(n, bool))
    si = ri = 0
    latest = None
    ring = np.nan
    for i in range(n):
        now = start+i*dt
        while si < len(samples) and samples[si][0] <= now+EPS:
            latest = samples[si][1]
            si += 1
        aim.ingest(latest, now)
        aim.step(now, dt)
        conflict = False
        while ri < len(ring_events) and ring_events[ri][0] <= now+EPS:
            value = ring_events[ri][1]
            ri += 1
            if np.isfinite(value):
                ring = value
                conflict |= aim.reconcile_ring(value, now)
        out['now'][i], out['target'][i], out['applied'][i] = now, aim.target, aim.applied
        out['mode'][i], out['commit'][i], out['ring'][i] = aim.mode or '', float(aim.commit_side), ring
        out['conflict'][i] = conflict
    pos = np.asarray(seq['pos'], float)
    out['pos'] = np.stack([np.interp(out['now'], t, pos[:, j]) for j in range(3)], 1)
    out['counts'] = aim.metadata()['counts']
    return out


def live_ticks(arrays):
    """Per-tick arrays of a live-log replay (vertical_replay npz) in the pilot_replay layout."""
    mode = np.asarray(arrays['gap_mode']).astype(str)
    return dict(now=np.asarray(arrays['now'], float), t=np.asarray(arrays['t'], float),
                target=np.asarray(arrays['gap_target'], float), applied=np.asarray(arrays['gap_applied'], float),
                offset=np.asarray(arrays['gap_offset'], float), mode=mode,
                commit=np.asarray(arrays['gap_commit'], float), ring=np.asarray(arrays['gap_ring'], float),
                pos=np.stack([np.asarray(arrays[k], float) for k in ('x', 'y', 'z')], 1))


# ----------------------------------------------------------------------------- scoring helpers

def wrap_deg(a):
    return (np.asarray(a, np.float64)+180.)%360.-180.


def box_extent(pos, box):
    """(lo, hi) world azimuths (deg) of a footprint box seen from each position."""
    az = np.stack([np.degrees(np.arctan2(y-pos[:, 1], x-pos[:, 0])) for x in box['x'] for y in box['y']], 1)
    ref = az[:, :1]
    rel = wrap_deg(az-ref)
    return ref[:, 0]+rel.min(1), ref[:, 0]+rel.max(1)


def inside(a, lo, hi):
    return (wrap_deg(a-lo) >= 0) & (wrap_deg(hi-a) >= 0)


def in_region(pos, region):
    return ((pos[:, 0] > region['x'][0]) & (pos[:, 0] < region['x'][1])
            & (pos[:, 1] > region['y'][0]) & (pos[:, 1] < region['y'][1]))


def approach_metrics(ticks, g):
    """Pillar A on one approach at the pilot level: the first tick with a confirmed obstacle target on the ring side
    (LEFT) in the approach region and its distance to the pillar; confirmed RIGHT obstacle ticks; ticks whose flown
    aim (ring + applied) lies inside the pillar's azimuth extent while the ring does not; the share of region ticks
    after the first left confirmation that keep a left target."""
    pos = ticks['pos']
    reg = in_region(pos, g['approach_region'])
    d = np.hypot(g['distance_to'][0]-pos[:, 0], g['distance_to'][1]-pos[:, 1])
    obstacle = ticks['mode'] == 'obstacle'
    left = np.flatnonzero(reg & obstacle & (ticks['target'] > 0))
    right = reg & obstacle & (ticks['target'] < 0)
    lo, hi = box_extent(pos, g['pillar_box_xy'])
    aim = ticks['ring']+ticks['applied']
    aim_in = reg & (np.abs(ticks['applied']) > 1e-6) & np.isfinite(aim) & inside(aim, lo, hi) & ~inside(ticks['ring'], lo, hi)
    first = int(left[0]) if len(left) else None
    after = reg & (np.arange(len(reg)) >= (first if first is not None else len(reg)))
    return dict(region_ticks=int(reg.sum()),
                first_left_d_m=None if first is None else round(float(d[first]), 3),
                right_ticks=int(right.sum()),
                right_d_m=[round(float(v), 2) for v in d[right][:: max(1, int(right.sum())//8 or 1)]],
                aim_into_pillar_ticks=int(aim_in.sum()),
                left_share_after_first=None if first is None else round(float(np.mean(ticks['target'][after] > 0)), 4),
                closest_d_m=round(float(d[reg].min()), 2) if reg.any() else None)


def pillar_b_metrics(ticks, impact_t, g):
    """G2 on the pilot's target: the first tick in the TTI window whose target points to world -x."""
    tti = impact_t-ticks['now']
    w = (tti >= g['window_tti_s'][0]) & (tti <= g['window_tti_s'][1])
    xdir = ticks['target']*(-np.sin(np.radians(ticks['ring'])))
    mx = np.flatnonzero(w & (ticks['mode'] == 'obstacle') & (xdir < 0))
    px = np.flatnonzero(w & (ticks['mode'] == 'obstacle') & (xdir > 0))
    first = float(tti[mx[0]]) if len(mx) else None
    return dict(first_minus_x_tti_s=None if first is None else round(first, 3),
                plus_x_ticks_before=int(np.sum(px < (mx[0] if len(mx) else len(tti)))),
                passes=bool(first is not None and first >= g['min_lead_s']))


def straw_metrics(ticks, seq, g, cue_latency_s):
    """Clean-lap statistics at the pilot level: episodes (runs of a nonzero target, split at side changes) per
    minute of flight (G3's minutes: frame gaps < 0.3 s), p90 |applied| over engaged ticks, engaged share, and the
    checkpoint switches whose last `switch_window_s` before the pilot saw the switch kept |applied| <= the limit."""
    from . import gap_cue_eval as ge
    t = np.asarray(seq['t'], float)
    dt = np.diff(t)
    minutes = float(np.sum(dt[dt < .3]))/60
    covered = np.zeros(len(ticks['now']), bool)
    for a, b in zip(t[:-1][dt < .3], t[1:][dt < .3]):
        covered |= (ticks['now'] >= a) & (ticks['now'] < b)
    side = np.sign(ticks['target'])
    onset = (side != 0) & (np.r_[0., side[:-1]] != side)
    episodes = int((onset & covered).sum())
    engaged = side != 0
    sp = np.hypot(seq['vel'][:, 0], seq['vel'][:, 1])
    switches = ge.checkpoint_switches(t, seq['ring'], sp)
    ok, worst = [], []
    for i in switches:
        seen = t[i]+cue_latency_s
        m = (ticks['now'] >= seen-g['switch_window_s']) & (ticks['now'] < seen)
        mx = float(np.max(np.abs(ticks['applied'][m]))) if m.any() else 0.
        ok.append(mx <= g['switch_abs_shift_max_deg']+EPS)
        worst.append(round(mx, 2))
    app = np.abs(ticks['applied'][engaged & covered])
    return dict(minutes=round(minutes, 3), episodes=episodes, episodes_per_min=round(episodes/max(minutes, 1e-9), 3),
                engaged_share=round(float(np.mean(engaged[covered])), 4) if covered.any() else 0.,
                applied_p90=round(float(np.percentile(app, 90)), 2) if len(app) else 0.,
                switches=len(switches), switches_ok=int(sum(ok)), switch_worst_deg=worst)


def pillar_c_pass_metrics(ticks, g):
    """A run that passed pillar C (on its free, right side): ticks in the pillar C approach region whose applied
    shift points LEFT (toward the pillar) or whose flown aim lies inside the pillar's azimuth extent while the ring
    does not."""
    pos = ticks['pos']
    reg = in_region(pos, g['approach_region'])
    left = reg & (ticks['applied'] > 1e-6)
    lo, hi = box_extent(pos, g['pillar_box_xy'])
    aim = ticks['ring']+ticks['applied']
    aim_in = reg & (np.abs(ticks['applied']) > 1e-6) & np.isfinite(aim) & inside(aim, lo, hi) & ~inside(ticks['ring'], lo, hi)
    right = reg & (ticks['applied'] < -1e-6)
    return dict(region_ticks=int(reg.sum()), left_ticks=int(left.sum()), right_ticks=int(right.sum()),
                aim_into_pillar_ticks=int(aim_in.sum()),
                max_left_deg=round(float(ticks['applied'][reg].max()), 2) if reg.any() else None)


def pillar_c_metrics(ticks, impact_t, g):
    """minus-fast6-vg-02 at pillar C (live-log replay of the deployed stack): LEFT (toward the pillar) applied ticks
    in the approach window; the first tick with a RIGHT (free side) obstacle target and its lead before the impact;
    whether the target stays right at every tick from then to the impact, and never shrinks."""
    t = ticks['t']
    w = (t >= g['window_start_t']) & (t <= impact_t)
    left = w & (ticks['applied'] > 1e-6)
    right = np.flatnonzero(w & (ticks['mode'] == 'obstacle') & (ticks['target'] < 0))
    res = dict(window=[g['window_start_t'], round(float(impact_t), 3)], left_applied_ticks=int(left.sum()),
               max_left_applied_deg=round(float(ticks['applied'][w].max()), 2) if w.any() else None)
    if not len(right):
        res.update(first_right_t=None, lead_s=None, held=False, non_decreasing=False)
        return res
    k = int(right[0])
    span = np.arange(k, int(np.flatnonzero(w)[-1])+1)
    target = ticks['target'][span]
    res.update(first_right_t=round(float(t[k]), 3), lead_s=round(float(impact_t-t[k]), 3),
               held=bool(np.all(target < 0)), non_decreasing=bool(np.all(np.diff(np.abs(target)) >= -EPS)),
               min_abs_target_after=round(float(np.min(np.abs(target))), 2),
               applied_at_impact=round(float(ticks['applied'][span[-1]]), 2),
               zero_target_ticks=int(np.sum(target == 0)), left_target_ticks=int(np.sum(target > 0)))
    return res


def response_crossing(ticks, t0, impact_t, model, ring_xy, face_y, box_x):
    """REPORT ONLY: where the declared response model would cross the pillar face if, from the logged state at t0,
    the aim were the bearing to the (hindsight, triangulated) ring plus the replayed applied shift."""
    import math
    k0 = int(np.argmin(np.abs(ticks['t']-t0)))
    x, y = float(ticks['pos'][k0, 0]), float(ticks['pos'][k0, 1])
    vel = ticks['vel'][k0]
    v = math.hypot(vel[0], vel[1])
    h = math.atan2(vel[1], vel[0])
    omega = model.a_lat_mps2/max(v, .1)
    tt, dt = 0., .005
    while y < face_y and tt < 3.:
        if tt > model.delay_s:
            shift = float(np.interp(t0+tt-model.delay_s, ticks['t'], ticks['applied']))
            aim = math.atan2(ring_xy[1]-y, ring_xy[0]-x)+math.radians(shift)
            err = (aim-h+math.pi) % (2*math.pi)-math.pi
            h += max(-omega*dt, min(omega*dt, err))
        x += v*dt*math.cos(h)
        y += v*dt*math.sin(h)
        tt += dt
    return dict(x_at_face=round(x, 3), clearance_m=round(max(box_x[0]-x, x-box_x[1]), 3), t_to_face=round(tt, 3))


def triangulate_ring(ticks, window):
    """Hindsight ring position (scoring only): least-squares intersection of the pilot's ring-centre azimuths."""
    t = ticks['t']
    m = (t >= window[0]) & (t <= window[1]) & np.isfinite(ticks['ring'])
    az = np.radians(ticks['ring'][m])
    p = ticks['pos'][m]
    A = np.stack([-np.sin(az), np.cos(az)], 1)
    b = A[:, 0]*p[:, 0]+A[:, 1]*p[:, 1]
    ring, *_ = np.linalg.lstsq(A, b, rcond=None)
    return ring


# ----------------------------------------------------------------------------- identity

COMMAND_KEYS = ('cvx', 'cvy', 'cvz', 'yaw_cmd', 'state', 'cap', 'climb', 'descent_scale')
GAP_KEYS = ('gap_target', 'gap_applied', 'gap_offset')


def identical(a, b, keys):
    out = {}
    for k in keys:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.dtype.kind in 'fc' and y.dtype.kind in 'fc':
            out[k] = bool(x.shape == y.shape and np.array_equal(x, y, equal_nan=True))
        else:
            out[k] = bool(x.shape == y.shape and np.all(x == y))
    return all(out.values()), out


def _npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- score

def score(out_dir, live_prefix, base_prefix, gates_path=GATES_PATH):
    """All frozen gates (see configs/obstacles/gap_commit_gates.json)."""
    from ..vision import gap_cue as gc
    gates, gates_sha = load_json_frozen(gates_path)
    new, new_obj, new_sha = aim_config(REPO_ROOT/gates['declarations']['candidate'])
    old, old_obj, old_sha = aim_config(REPO_ROOT/gates['declarations']['current'])
    if new_obj.get('version') != gates['declarations']['candidate_version']:
        raise RuntimeError('the candidate declaration is not the version the gates were frozen for')
    runs = Path(gates['runs_dir'])
    lat = gates['offline_replay']
    res = dict(schema='haltere.obstacles.gap_commit_results.v1',
               scored_at=_dt.datetime.now().isoformat(timespec='seconds'), gates_sha256=gates_sha,
               gates_version=gates['version'], candidate=dict(path=gates['declarations']['candidate'],
                                                             version=new_obj['version'], sha256=new_sha),
               current=dict(path=gates['declarations']['current'], version=old_obj['version'], sha256=old_sha))
    configs = dict(current=old, candidate=new)
    replays = {}

    def offline(name):
        if name not in replays:
            seq = load_seq(out_dir, name)
            replays[name] = (seq, {k: pilot_replay(seq, c, gap_latency_s=lat['gap_latency_s'],
                                                   cue_latency_s=lat['cue_latency_s']) for k, c in configs.items()})
        return replays[name]

    def live(name, which):
        tag = gates['live_replay']['tags'][which]
        arrays = _npz(f'{live_prefix}_{tag}_{name}.npz')
        ticks = live_ticks(arrays)
        ticks['vel'] = np.stack([np.asarray(arrays[k], float) for k in ('vx', 'vy', 'vz')], 1)
        return ticks

    verdict = {}
    # ---- C: pillar C on minus-fast6-vg-02
    g = gates['C_pillar_c']
    side = json.loads((runs/f"{g['flight']}.json").read_text(encoding='utf-8'))
    ticks = {k: live(g['flight'], k) for k in configs}
    impact_t = float(ticks['candidate']['t'][-1])
    rows = {k: pillar_c_metrics(v, impact_t, g) for k, v in ticks.items()}
    c = rows['candidate']
    checks = dict(no_left=c['left_applied_ticks'] == 0,
                  early=c['lead_s'] is not None and c['lead_s'] >= g['min_lead_s']-EPS,
                  held=bool(c.get('held')), non_decreasing=bool(c.get('non_decreasing')))
    ring = triangulate_ring(ticks['current'], g['ring_window_t'])
    models = gc.load_response_models()
    report = {k: {t0: response_crossing(v, t0, impact_t, models['fast_pd'], ring, g['pillar_box_xy']['y'][0],
                                        g['pillar_box_xy']['x']) for t0 in g['report_response_t0']}
              for k, v in ticks.items()}
    res['C_pillar_c'] = dict(rows=rows, checks=checks, passed=all(checks.values()), stop_reason=side.get('stop_reason'),
                             report_only=dict(ring_xy_hindsight=[round(float(v), 3) for v in ring],
                                              response_model_crossing=report))
    verdict['C_pillar_c'] = res['C_pillar_c']['passed']
    # ---- A: pillar A on every Minus log that reached it
    g = gates['A_pillar_a']
    per = {}
    for spec in g['approaches']:
        name = spec['name']
        if spec['source'] == 'live':
            rows = {k: approach_metrics(live(name, k), g) for k in configs}
        else:
            _, rp = offline(name)
            rows = {k: approach_metrics(v, g) for k, v in rp.items()}
        a, b = rows['current'], rows['candidate']
        ok = dict(first_left=(a['first_left_d_m'] is None and b['first_left_d_m'] is None)
                  or (b['first_left_d_m'] is not None
                      and (a['first_left_d_m'] is None or b['first_left_d_m'] >= a['first_left_d_m']-g['tolerance_m'])),
                  right=b['right_ticks'] <= a['right_ticks'], into=b['aim_into_pillar_ticks'] <= a['aim_into_pillar_ticks'])
        per[name] = dict(source=spec['source'], role=spec.get('role'), current=a, candidate=b, checks=ok,
                         passed=all(ok.values()))
    res['A_pillar_a'] = dict(approaches=per, passing=sum(v['passed'] for v in per.values()), total=len(per),
                             passed=all(v['passed'] for v in per.values()))
    verdict['A_pillar_a'] = res['A_pillar_a']['passed']
    # ---- B: pillar B
    g = gates['B_pillar_b']
    per = {}
    for name in g['runs']:
        seq, rp = offline(name)
        impact_t = float(seq['t'][0]+seq['tti'][0])
        per[name] = {k: pillar_b_metrics(v, impact_t, g) for k, v in rp.items()}
    res['B_pillar_b'] = dict(runs=per, passed=all(v['candidate']['passes'] for v in per.values()))
    verdict['B_pillar_b'] = res['B_pillar_b']['passed']
    # ---- P: runs that passed pillar C on its free side
    g = gates['P_pillar_c_passes']
    per = {}
    for name in g['runs']:
        _, rp = offline(name)
        rows = {k: pillar_c_pass_metrics(v, g) for k, v in rp.items()}
        ok = dict(left=rows['candidate']['left_ticks'] <= rows['current']['left_ticks'],
                  into=rows['candidate']['aim_into_pillar_ticks'] <= rows['current']['aim_into_pillar_ticks'])
        per[name] = dict(rows, checks=ok, passed=all(ok.values()))
    res['P_pillar_c_passes'] = dict(runs=per, passed=all(v['passed'] for v in per.values()))
    verdict['P_pillar_c_passes'] = res['P_pillar_c_passes']['passed']
    # ---- S: clean Straw Bale laps (G3 flights)
    g = gates['S_straw']
    laps = {}
    tot = {k: dict(minutes=0., episodes=0, switches=0, switches_ok=0) for k in configs}
    for name in g['flights']:
        seq, rp = offline(name)
        laps[name] = {k: straw_metrics(v, seq, g, lat['cue_latency_s']) for k, v in rp.items()}
        for k in configs:
            for key in tot[k]:
                tot[k][key] += laps[name][k][key]
    for k in configs:
        tot[k]['episodes_per_min'] = round(tot[k]['episodes']/max(tot[k]['minutes'], 1e-9), 3)
        tot[k]['switch_ok_fraction'] = round(tot[k]['switches_ok']/max(tot[k]['switches'], 1), 4)
        app = np.concatenate([np.abs(rp[k]['applied'][rp[k]['target'] != 0]) for _, rp in
                              (replays[n] for n in g['flights'])])
        tot[k]['applied_p90'] = round(float(np.percentile(app, 90)), 2) if len(app) else 0.
    a, b = tot['current'], tot['candidate']
    checks = dict(episodes=b['episodes_per_min'] <= a['episodes_per_min']+EPS,
                  switches=b['switch_ok_fraction'] >= min(a['switch_ok_fraction'], g['switch_min_fraction'])-EPS)
    res['S_straw'] = dict(laps=laps, total=tot, checks=checks, passed=all(checks.values()),
                          cue_level_g3_note=g['cue_level_note'])
    verdict['S_straw'] = res['S_straw']['passed']
    # ---- L: leak test (the per-frame cue is unchanged)
    g = gates['L_leaks']
    params, cfg, cue_sha = gc.load_config(require_frozen=True)
    ref = json.loads((REPO_ROOT/g['reference_results']).read_text(encoding='utf-8'))
    rescored = Path(out_dir)/'cue_rescore'/'results.json'
    same = None
    if rescored.exists():
        mine = json.loads(rescored.read_text(encoding='utf-8'))
        same = all(mine['bases'][b]['G4'][k] == ref['bases'][b]['G4'][k] for b in ('teacher', 'runtime')
                   for k in ('ring_painted', 'ring_removed', 'hud_only'))
    checks = dict(cue_config_unchanged=cue_sha == g['gap_cue_sha256'], g4_rescored_identical=bool(same))
    res['L_leaks'] = dict(checks=checks, passed=all(checks.values()),
                          g4_teacher=ref['bases']['teacher']['G4'] if 'bases' in ref else None)
    verdict['L_leaks'] = res['L_leaks']['passed']
    # ---- O: stack off and shadow; the version 2 declaration through this code (default off)
    g = gates['O_identity']
    rows = []
    for flight in g['flights']:
        for mine, base, keys in g['pairs']:
            try:
                a = _npz(f'{live_prefix}_{mine}_{flight}.npz')
                b = _npz(f'{base_prefix if base.startswith("BASE:") else live_prefix}_{base.removeprefix("BASE:")}_{flight}.npz')
                ok, detail = identical(a, b, COMMAND_KEYS+(GAP_KEYS if keys == 'gap' else ()))
            except FileNotFoundError as exc:
                ok, detail = None, str(exc)
            rows.append(dict(flight=flight, variant=mine, against=base, identical=ok, keys=detail))
    res['O_identity'] = dict(pairs=rows, passed=bool(rows) and all(r['identical'] for r in rows))
    verdict['O_identity'] = res['O_identity']['passed']
    res['verdict'] = verdict
    # ---- report only: Pine trunk, counts
    g = gates.get('report_only', {})
    extra = {}
    for name in g.get('pine_trunk', []):
        seq, rp = offline(name)
        impact_t = float(seq['t'][0]+seq['tti'][0])
        extra[name] = {}
        for k, v in rp.items():
            tti = impact_t-v['now']
            w = tti <= 2.
            first = np.flatnonzero(w & (v['target'] != 0))
            extra[name][k] = dict(first_target_tti_s=None if not len(first) else round(float(tti[first[0]]), 3),
                                  sides=sorted({'left' if x > 0 else 'right' for x in v['target'][w] if x != 0}),
                                  applied_at_impact=round(float(np.interp(impact_t, v['now'], v['applied'])), 2),
                                  counts=v['counts'])
    res['report_only'] = dict(pine_trunk=extra)
    Path(out_dir, 'results.json').write_text(json.dumps(res, indent=1, default=str)+'\n', encoding='utf-8')
    _log(json.dumps(verdict))
    return res


# ----------------------------------------------------------------------------- dev (report only)

def dev_report(out_dir, candidate_path, current_path, names, straw_gate, replay):
    """Report-only statistics of a candidate declaration against the current one on development sequences."""
    from ..liftoff.gap_aim import GapAimConfig
    load = lambda p: GapAimConfig.from_dict(json.loads(Path(p).read_text(encoding='utf-8'))['pilot'])
    configs = dict(current=load(current_path), candidate=load(candidate_path))
    rows = {}
    for name in names:
        seq = load_seq(out_dir, name)
        rows[name] = {k: straw_metrics(pilot_replay(seq, c, gap_latency_s=replay['gap_latency_s'],
                                                    cue_latency_s=replay['cue_latency_s']), seq, straw_gate,
                                       replay['cue_latency_s'])
                      for k, c in configs.items()}
        _log(name, json.dumps(rows[name]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m haltere.obstacles.gap_commit_eval')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('seq')
    a.add_argument('--out', required=True)
    a.add_argument('--cache', required=True, help='gap-cue evaluation output with flights/, depth/, store_masks.npz')
    a.add_argument('--gates', default=str(GATES_PATH))
    a.add_argument('--names', default=None, help='comma-separated subset of the gate sequences')
    a.add_argument('--extra', default=None,
                   help='JSON file with a list of extra specs (name, motor, store_run_id | flight)')
    a.add_argument('--flight-lock', default=None)
    a = sub.add_parser('score')
    a.add_argument('--out', required=True)
    a.add_argument('--live', required=True, help='live replay prefix (this tree)')
    a.add_argument('--base', required=True, help='live replay prefix of the baseline tree')
    a.add_argument('--gates', default=str(GATES_PATH))
    a = sub.add_parser('dev')
    a.add_argument('--out', required=True)
    a.add_argument('--candidate', required=True)
    a.add_argument('--current', default=str(OB/'gap_pilot_v2.json'))
    a.add_argument('--names', required=True)
    a.add_argument('--gates', default=str(GATES_PATH))
    args = ap.parse_args(argv)
    from .thermal import limit_threads
    limit_threads(2, cv2=True)
    if args.cmd == 'seq':
        from .thermal import ChunkGuard, require_flight_lock_path
        ChunkGuard(require_flight_lock_path(args.flight_lock), gpu=False).before_chunk()
        gates = json.loads(Path(args.gates).read_text(encoding='utf-8'))
        specs = list(gates['sequences'])
        if args.extra:
            specs += json.loads(Path(args.extra).read_text(encoding='utf-8'))
        if args.names:
            keep = set(args.names.split(','))
            specs = [s for s in specs if s['name'] in keep]
        build_sequences(specs, args.cache, args.out, gates)
    elif args.cmd == 'score':
        score(args.out, args.live, args.base, args.gates)
    else:
        gates = json.loads(Path(args.gates).read_text(encoding='utf-8'))
        rows = dev_report(args.out, args.candidate, args.current, args.names.split(','), gates['S_straw'],
                          gates['offline_replay'])
        Path(args.out, 'dev_report.json').write_text(json.dumps(rows, indent=1)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
