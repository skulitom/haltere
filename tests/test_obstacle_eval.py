"""Obstacle harness E1-E8, decision rule, baselines and reproduction adapter on small synthetic fixtures."""
import json
from pathlib import Path

import numpy as np
import pytest

from haltere.obstacles import baselines as BL
from haltere.obstacles import contract, leaks
from haltere.obstacles import evaluate as ev
from haltere.obstacles.labels import LabelKind as K
from haltere.obstacles.labels import LabelSource as LS
from haltere.obstacles.splits import ENV_CODE, env_code_mask
from haltere.obstacles.store import empty_index

LATERAL_DATA = Path('C:/Users/artem/AppData/Local/Temp/claude/C--DEV-Haltere/6f308e08-af33-4f7e-b2e8-e1cad1c82b5f/'
                    'scratchpad/lateral/data')
DT = 0.06            # ~16.7 Hz, like the run videos
SPEED = 6.0


def _thresholds():
    obj, _ = ev.load_thresholds(require_frozen=False)
    return obj


# ----------------------------------------------------------------------------- fixtures

class FakeStore:
    """Minimal store: index (INDEX_DTYPE), runs, side tables in ``root``, deterministic frames."""

    def __init__(self, root, runs):
        self.root = Path(root)
        parts = []
        self.runs = []
        for k, r in enumerate(runs):
            n = len(r['t'])
            ix = empty_index(n)
            ix['run_id'] = k
            ix['slot'] = np.arange(n)
            ix['env'] = ENV_CODE[r['env']]
            ix['t_phase'] = r['t']
            ix['t_wall'] = 1.7e9 + 100 * k + r['t']
            ix['pos'] = r['pos']
            ix['quat'] = r.get('quat', np.tile([1.0, 0, 0, 0], (n, 1)))
            ix['vel'] = r['vel']
            ix['cue_uv'] = r.get('cue_uv', np.full((n, 2), np.nan))
            ix['cue_src'] = np.where(np.isfinite(ix['cue_uv']).all(axis=1), 1, 0)
            parts.append(ix)
            self.runs.append(dict(run_id=k, source_id=r['name'], flight=r['name'], aliases=[r['name']], env=r['env']))
        self.index = np.concatenate(parts)
        self.manifest = dict(index_sha256=self.index_sha256())

    def __len__(self):
        return len(self.index)

    def index_sha256(self):
        import hashlib
        return hashlib.sha256(self.index.tobytes()).hexdigest()

    def rows(self, fold=None, side=None, **kw):
        m = np.ones(len(self.index), bool) if fold is None else env_code_mask(self.index['env'], fold, side)
        return np.flatnonzero(m).astype(np.int64)

    def gravity_camera(self, rows):
        return contract.gravity_camera(self.index['quat'][np.asarray(rows)].astype(np.float64))

    def frames(self, rows):
        out = []
        for r in np.asarray(rows):
            rng = np.random.default_rng(int(r))
            f = np.full((252, 448, 3), 90, np.uint8)
            f[:126] = (120, 160, 210)                      # sky
            f += rng.integers(0, 20, f.shape, dtype=np.uint8)
            out.append(f)
        return np.stack(out)


class FakeLabels:
    def __init__(self, n):
        self.gv = np.full((n, 18, 32), np.nan, np.float32)
        self.gk = np.zeros((n, 18, 32), np.uint8)
        self.gs = np.zeros((n, 18, 32), np.uint8)
        self.fv = np.full((n, 4, 9), np.nan, np.float32)
        self.fk = np.zeros((n, 4, 9), np.uint8)
        self.fs = np.zeros((n, 4, 9), np.uint8)
        self.manifest = dict(schema='test labels')

    def grid(self, rows):
        return self.gv[rows], self.gk[rows], self.gs[rows]

    def fan(self, rows):
        return self.fv[rows], self.fk[rows], self.fs[rows]

    def teacher(self, rows):
        raise FileNotFoundError


def straight_run(name, env, T=5.0, n=None, speed=SPEED, y=0.0):
    n = n or int(round(T / DT)) + 1
    t = np.arange(n) * DT
    pos = np.stack([speed * t, np.full(n, y), np.zeros(n)], 1)
    vel = np.tile([speed, 0.0, 0.0], (n, 1))
    return dict(name=name, env=env, t=t, pos=pos, vel=vel)


@pytest.fixture
def scene(tmp_path):
    """Run 0: Minus Two, flies +x into a pillar face at x = 30.15 m (impact at t = 5 s, free side right).
    Run 1: Straw Bale clean flight (clean window 0.5-4.5 s). Run 2: Pine Valley impact, free side up."""
    runs = [straight_run('minus-a', 'Minus Two'), straight_run('straw-clean', 'Straw Bale'),
            straight_run('pine-up', 'Pine Valley')]
    s = FakeStore(tmp_path, runs)
    (tmp_path / 'clean_windows.json').write_text(json.dumps(dict(windows=[
        dict(run_id=1, alias='straw-clean', start_phase_s=0.5, end_phase_s=4.5)])))
    events = [
        dict(event_id=0, store_event_id=-1, run='minus-a', env='Minus Two', kind='terminal_impact', t_phase=5.0,
             point_w=[30.15, 0.0, 0.0], drone_pos_w=[30.0, 0.0, 0.0], obstacle='pillar',
             unique_obstacle='minus/pillar-1', primary_free_side='right', accepted_free_sides=['right'],
             lateral=True, source='test', blind=False),
        dict(event_id=1, store_event_id=-1, run='pine-up', env='Pine Valley', kind='terminal_impact', t_phase=5.0,
             point_w=[30.0, 0.0, -2.5], drone_pos_w=[30.0, 0.0, 0.0], obstacle='terrain',
             unique_obstacle='pine/mound-1', primary_free_side='up', accepted_free_sides=['up'], lateral=False,
             source='test', blind=True),
    ]
    return s, events


def true_range_grid(store, rows, point):
    c = ev.store_cache(store)
    d = np.linalg.norm(np.asarray(point) - c.pos[rows], axis=1)
    return np.repeat(d[:, None, None], 18, 1).repeat(32, 2).astype(np.float32)


def fan_pred(store, rows, fan_q20, **extra):
    n = len(rows)
    arrays = dict(fan_q20=np.broadcast_to(fan_q20, (n, 4, 9)).astype(np.float32).copy())
    arrays['fan_q50'] = arrays['fan_q20'].copy()
    arrays.update(extra)
    return ev.PredictionSet('fan-test', 'model', True, np.asarray(rows), fold='F12', sha256='abc',
                            arrays=arrays).validate(len(store))


# ----------------------------------------------------------------------------- contract geometry

def test_grid_to_fan_wall_distance_corridor_and_mirror():
    q = np.array([1.0, 0, 0, 0])
    rays = contract.cell_rays_body()
    grid = np.full((18, 32), np.inf)
    # a fronto-parallel wall 5 m ahead (x = 5): every cell whose ray hits it gets its range
    with np.errstate(divide='ignore'):
        r = np.where(rays[..., 0] > 0.05, 5.0 / rays[..., 0], np.inf)
    grid[:] = r
    fan = contract.grid_to_fan(grid, q)
    assert fan.shape == (4, 9)
    # straight ahead (level) the first blocked point is the wall at s ~ 5 m (points within 0.5 m of the ray)
    assert abs(fan[1, 4] - 5.0) < 0.3
    # at yaw 40 the wall is at 5 / cos 40 along the ray
    assert abs(fan[1, 0] - 5.0 / np.cos(np.deg2rad(40))) < 0.5
    # unknown cells add no points
    assert np.isinf(contract.grid_to_fan(np.full((18, 32), np.nan), q)).all()
    # a horizontal image flip of the grid mirrors the fan (level attitude)
    g = np.full((18, 32), np.inf)
    g[14, 20] = 4.0
    f1 = contract.grid_to_fan(g, q)
    f2 = contract.grid_to_fan(contract.mirror_grid(g), q)
    assert np.allclose(np.nan_to_num(contract.mirror_fan(f1), posinf=99), np.nan_to_num(f2, posinf=99))
    # batch broadcasting and undefined heading (nose straight up) -> NaN
    up = np.array([np.cos(np.deg2rad(-45)), 0, np.sin(np.deg2rad(-45)), 0])     # pitch up 90 deg
    b = contract.grid_to_fan(np.stack([grid, grid]), np.stack([q, up]))
    assert b.shape == (2, 4, 9) and np.isfinite(b[0]).any()


def test_fan_in_view_and_bearings():
    q = np.array([1.0, 0, 0, 0])
    v = contract.fan_in_view(q)
    assert v[1:, :].all() and v[0, 1:-1].all() and not v[0, 0]
    pitched = np.array([np.cos(np.deg2rad(-20)), 0, np.sin(np.deg2rad(-20)), 0])    # nose up 40 deg
    assert not contract.fan_in_view(pitched).any()
    yaw, el = contract.image_bearing_heading(np.array([0.5, 0.5]), q)
    assert abs(yaw) < 1e-9 and abs(el - 30.0) < 1e-6
    yaw, _ = contract.image_bearing_heading(np.array([0.1, 0.5]), q)
    assert yaw > 0                                   # left of centre = positive yaw (FLU)
    yaw, el = contract.world_bearing_heading(np.array([1.0, -1.0, 0.0]), q)
    assert abs(yaw + 45) < 1e-9 and abs(el) < 1e-9


# ----------------------------------------------------------------------------- timing

def test_deployed_schedule_and_positions():
    t17 = np.arange(40) / 17.3
    assert ev.deployed_schedule(t17).all()
    t30 = np.arange(60) / 30.0
    keep = ev.deployed_schedule(t30)
    assert 25 <= keep.sum() <= 35
    assert ev.deployed_schedule(t30, None).all()
    p = ev.PredictionSet('x', 'model', True, np.array([2, 5, 9]))
    assert p.positions([5, 3, 9, 0]).tolist() == [1, -1, 2, -1]


def test_lateral_event_metrics_reproduce_the_study_scoring():
    t = np.arange(0.0, 5.0, 0.06)
    T = 5.0
    codes = np.where(t >= 3.0, int(ev.Side.RIGHT), 0)
    m = ev.lateral_event_metrics(t, codes, T, strict={'right'}, permissive={'right'})
    assert abs(m['correct_lead'] - (T - t[t >= 3.0][0])) < 1e-9 and m['success']
    assert m['first_wrong_lead'] == 0 and not m['wrong']
    # the same timeline 65 ms later loses 65 ms of lead
    m2 = ev.lateral_event_metrics(t + 0.065, codes, T, strict={'right'}, permissive={'right'})
    assert abs(m['correct_lead'] - m2['correct_lead'] - 0.065) < 1e-9
    # a left indication 2.5-2.0 s before impact is a wrong side held ~0.5 s
    codes3 = np.where((T - t <= 2.5) & (T - t > 2.0), int(ev.Side.LEFT), codes)
    m3 = ev.lateral_event_metrics(t, codes3, T, strict={'right'}, permissive={'right'})
    assert m3['wrong'] and 0.4 < m3['wrong_hold_max_s'] < 0.6 and m3['first_wrong_lead'] > 2.4
    # an indication that ends before T - grace gives no lead; 'either' accepts both sides
    codes4 = np.where((T - t > 0.5) & (T - t < 2.5), int(ev.Side.LEFT), 0)
    assert ev.lateral_event_metrics(t, codes4, T, strict={'left'}, permissive={'left'})['correct_lead'] == 0
    m5 = ev.lateral_event_metrics(t, np.full(len(t), int(ev.Side.LEFT)), T, strict={'left', 'right'},
                                  permissive={'left', 'right'})
    assert m5['success'] and m5['correct_lead'] <= 3.0 and not m5['wrong_any']


def test_clean_window_metrics_episodes():
    t = np.arange(0.0, 60.0, 0.06)
    codes = np.zeros(len(t), int)
    codes[(t > 10) & (t < 11)] = ev.Side.LEFT          # 1 s episode
    codes[(t > 20) & (t < 20.1)] = ev.Side.RIGHT       # too short (< 0.2 s)
    codes[(t > 30) & (t < 31)] = ev.Side.CENTRE        # centre: not a side episode, an 'any' episode
    m = ev.clean_window_metrics(t, codes, 0.0, 60.0)
    assert len(m['episodes']) == 1 and m['episodes'][0][2] == ev.Side.LEFT
    assert len(m['any_episodes']) == 2
    assert abs(m['total_s'] - 60.0) < 0.1 and 0.9 < m['side_s'] < 1.2 and 0.9 < m['centre_s'] < 1.1


# ----------------------------------------------------------------------------- decision rule

def test_raw_decisions_nearest_clear_side_centre_and_none():
    p = ev.DecisionParams.from_thresholds(_thresholds())
    D6 = p.required_distance(6.0, 0.065)
    assert 5.5 < D6 < 6.5                                     # ~6 m at 6 m/s (M3 design)
    inview = np.ones((4, 4, 9), bool)
    fan = np.full((4, 4, 9), 20.0)
    fan[0, :, 3:6] = 3.0                                      # blocked ahead, clear left and right; equal distance
    fan[0, :, 6] = 3.0                                        # ... one more column blocked on the left -> right
    fan[1, :, :] = 3.0                                        # everything blocked -> centre
    fan[2, :, 4] = 30.0                                       # ahead clear -> none
    fan[3, 1, :] = np.nan                                     # level row unknown -> blocked; rows +-10 still clear?
    speed = np.full(4, 6.0)
    zero = np.zeros(4)
    codes, dl, dr = ev.raw_side_decisions(fan, inview, speed, zero, zero, p, 0.065)
    assert codes[0] == ev.Side.RIGHT and dr[0] == 20 and dl[0] == 30
    assert codes[1] == ev.Side.CENTRE and codes[2] == ev.Side.NONE
    assert codes[3] == ev.Side.CENTRE                         # a NaN level row blocks every column (min over rows)
    # out of view = no evidence = blocked
    iv = inview.copy()
    iv[0, :, :4] = False
    c2, _, _ = ev.raw_side_decisions(fan, iv, speed, zero, zero, p, 0.065)
    assert c2[0] == ev.Side.LEFT
    # slow: no decision
    c3, _, _ = ev.raw_side_decisions(fan, inview, np.full(4, 0.5), zero, zero, p, 0.065)
    assert (c3 == ev.Side.NONE).all()


def test_hysteresis_onset_hold_switch_and_release():
    p = ev.DecisionParams.from_thresholds(_thresholds())
    t = np.arange(30) * 0.0625
    raw = np.zeros(30, int)
    raw[2:10] = ev.Side.RIGHT
    raw[10:14] = ev.Side.LEFT          # opposite side too early (hold 0.6 s) -> kept right
    dl = np.full(30, 10.0)
    dr = np.full(30, 30.0)             # left is nearer by 20 deg once the hold has passed
    out = ev.apply_hysteresis(t, raw, dl, dr, p)
    assert out[2] == 0 and out[3] == ev.Side.RIGHT            # onset after 2 frames
    assert (out[10:12] == ev.Side.RIGHT).all()                # hold + 2-frame confirmation
    assert out[13] == ev.Side.LEFT                            # switched (advantage 20 deg >= 10)
    # release: raw NONE for >= 0.3 s and >= 3 frames, and not before the new side's 0.6 s hold
    assert out[14] == ev.Side.LEFT and out[29] == 0
    switched = np.flatnonzero(out == ev.Side.LEFT)[0]
    first_none = np.flatnonzero(out[14:] == 0)[0] + 14
    assert t[first_none] - t[switched] >= 0.6 - 1e-9 and t[first_none] - t[14] >= 0.3 - 1e-9
    assert t[first_none - 1] - t[switched] < 0.6
    # a short raw gap inside the hold does not release; CENTRE is entered without a hold
    raw2 = np.array([2, 2, 0, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0])
    o2 = ev.apply_hysteresis(np.arange(14) * 0.0625, raw2, np.full(14, 20.0), np.full(14, 10.0), p)
    assert (o2[1:5] == ev.Side.RIGHT).all() and o2[6] == ev.Side.CENTRE


def test_decide_side_on_a_store_uses_ring_bearing_rate_and_latency(scene):
    s, _ = scene
    rows = s.rows()
    fan = np.full((4, 9), 20.0)
    fan[:, 3:6] = 3.0                                         # blocked ahead, both sides clear -> tie
    fan[:, 6] = 3.0                                           # left one more blocked -> right
    pred = fan_pred(s, rows, fan)
    codes = ev.decide_side(pred, s, _thresholds())
    run0 = ev.store_cache(s).run_id[rows] == 0
    assert codes[run0][0] == 0 and (codes[run0][2:] == ev.Side.RIGHT).all()
    # the ring far to the right moves the reference onto a clear column -> NONE
    s.index['cue_uv'][:] = (0.95, 0.6)
    s._eval_cache = None
    codes = ev.decide_side(pred, s, _thresholds())
    assert (codes == 0).all()
    # the ring just left of the blocked columns: the nearest clear column is further left -> LEFT
    s.index['cue_uv'][:] = (0.45, 0.6)
    s._eval_cache = None
    yaw, _ = contract.image_bearing_heading(np.array([0.45, 0.6]), np.array([1.0, 0, 0, 0]))
    assert 5 < yaw < 15
    codes = ev.decide_side(pred, s, _thresholds())
    assert (codes[run0][2:] == ev.Side.LEFT).all()
    # latency does not change the decisions of a static scene, only when they become usable
    assert (ev.decide_side(pred, s, _thresholds(), latency_s=0.15) == codes).all()


# ----------------------------------------------------------------------------- metrics

def test_e1_ratio_bins_flags_and_prior_filter(scene):
    s, events = scene
    th = _thresholds()
    rows = s.rows()
    grid = np.concatenate([true_range_grid(s, rows[ev.store_cache(s).run_id[rows] == k],
                                           events[0]['point_w'] if k == 0 else events[1]['point_w'])
                           for k in range(3)])
    pred = ev.PredictionSet('F12-v0', 'model', True, rows, fold='F12', sha256='m', arrays=dict(grid_q50=grid))
    recs = ev.e1_range_at_impacts(pred, s, None, events, th, fold='F12', n_boot=50)
    by = {r['variant']: r for r in recs}
    minus = by['env Minus Two']
    assert minus['held_out'] is True and minus['seen_environment'] is False and minus['n'] > 20
    for b in ('2-4', '4-7', '7-16'):
        assert abs(minus['value']['bins'][b]['median'] - 1.0) < 1e-3
    assert minus['value']['overshoot']['fraction'] == 0.0
    assert by['pooled held-out envs']['held_out'] is True and 'm2_checks' in by['pooled held-out envs']['value']
    stays = minus['value']['pillar_approaches']['0']
    assert stays['stays'] and stays['n'] > 5
    pred2 = ev.PredictionSet('x2', 'model', True, rows, fold='F12', sha256='m2', arrays=dict(grid_q50=2 * grid))
    r2 = ev.e1_range_at_impacts(pred2, s, None, events, th, fold='F12', n_boot=50)
    m2 = next(r for r in r2 if r['variant'] == 'env Minus Two')
    assert abs(m2['value']['bins']['2-4']['median'] - 2.0) < 1e-3 and m2['value']['overshoot']['fraction'] == 1.0
    # window: only frames 0.15-3 s before T (never after the impact)
    smp = ev.e1_samples(pred, s, events, th)
    assert all(0.15 <= x['tti'] <= 3.0 for x in smp)
    # the pine point is 2.5 m below the travel line: the study's chord filter (<= 15 deg) drops the near frames
    n_all = sum(x['event_id'] == 1 for x in smp)
    kept = [x for x in ev.e1_samples(pred, s, events, th, prior_filter=True) if x['event_id'] == 1]
    assert 0 < len(kept) <= n_all and min(x['true'] for x in kept) >= 2.5 / np.sin(np.deg2rad(15)) - 1e-6


def test_e4_constant_sides_and_e7(scene):
    s, events = scene
    th = _thresholds()
    rows = s.rows()
    right = ev.baseline_b0(s, rows, ev.Side.RIGHT)
    left = ev.baseline_b0(s, rows, ev.Side.LEFT)
    rr = ev.e4_lateral_replay(right, s, events, th, fold='F12')
    rl = ev.e4_lateral_replay(left, s, events, th, fold='F12')
    mr = next(r for r in rr if r['variant'].startswith('env Minus Two'))
    ml = next(r for r in rl if r['variant'].startswith('env Minus Two'))
    assert mr['value']['correct'] == 1 and mr['value']['wrong_held'] == 0 and mr['held_out']
    lead = mr['per_event'][0]['correct_lead']
    assert 2.9 < lead <= 3.0                                  # the whole 3 s window (usable times)
    assert ml['value']['correct'] == 0 and ml['value']['wrong_held'] == 1
    assert mr['predictor']['baseline_id'] == 'B0' and mr['latency_s'] == 0.065
    e7 = ev.e7_unique_obstacles(rr, events)
    assert e7 and e7[0]['metric'] == 'E7' and e7[0]['value']['per_obstacle']['minus/pillar-1']['correct'] == 1
    assert e7[0]['value']['event_ci95_wilson'][1] == 1.0


def test_e5_vertical_event_and_e6_clean(scene):
    s, events = scene
    th = _thresholds()
    rows = s.rows()
    right = ev.baseline_b0(s, rows, ev.Side.RIGHT)
    e5 = ev.e5_out_of_view(right, s, events, th, fold='F12')
    pine = next(r for r in e5 if r['env'] == 'Pine Valley')
    # 2.5 m below the path, 6-12 m out: below the image bottom (-12 deg) for part of T-2..T-1 s
    assert pine['value']['n'] == 1 and pine['value']['events']['1']['reason'] == 'out of view'
    assert 0 < pine['value']['events']['1']['in_view_frac_T2_T1'] < 2 / 3
    assert pine['value']['events']['1']['warning_lead'] > 2.5
    e6 = ev.e6_clean_windows(right, s, th, fold='F12')
    straw = next(r for r in e6 if r['env'] == 'Straw Bale')
    assert straw['seen_environment'] is True and straw['value']['side_episodes'] == 1
    assert straw['value']['side_active_fraction'] > 0.95
    clear = fan_pred(s, rows, np.full((4, 9), 20.0))
    e6c = ev.e6_clean_windows(clear, s, th, fold='F12')
    straw = next(r for r in e6c if r['env'] == 'Straw Bale')
    assert straw['value']['any_episodes'] == 0 and straw['value']['m3_checks']['any active fraction <= max']


def test_e2_e3_against_labels(scene):
    s, events = scene
    th = _thresholds()
    rows = s.rows()
    n = len(rows)
    lab = FakeLabels(n)
    rng = np.random.default_rng(0)
    true = rng.uniform(1.0, 15.0, (n, 18, 32)).astype(np.float32)
    lab.gv[:] = true
    lab.gk[:] = K.EXACT
    lab.gs[:] = LS.COLLIDER
    fan_true = rng.uniform(1.0, 19.0, (n, 4, 9)).astype(np.float32)
    lab.fv[:] = fan_true
    lab.fk[:] = K.EXACT
    lab.fs[:] = LS.COLLIDER | LS.HINDSIGHT
    lab.gk[:, 0, 0] = K.LOWER                      # a flown-tube lower bound per frame
    lab.gs[:, 0, 0] = LS.TUBE
    perfect = dict(grid_q50=true.copy(), grid_q20=true.copy(), fan_q20=fan_true.copy(), fan_q50=fan_true.copy(),
                   fan_p4=(fan_true <= 4).astype(np.float32), fan_p8=(fan_true <= 8).astype(np.float32))
    pred = ev.PredictionSet('perfect', 'model', True, rows, fold='F4', sha256='p', arrays=perfect)
    e2 = ev.e2_collider_cells(pred, s, lab, th, fold='F4')
    pooled = next(r for r in e2 if r['variant'] == 'pooled all envs')
    assert pooled['value']['absrel_median'] < 1e-6 and pooled['value']['delta_1p25'] == 1.0
    assert pooled['value']['fan']['precision'] == 1.0 and pooled['value']['fan']['recall'] == 1.0
    e3 = ev.e3_fan_quality(pred, s, lab, th, fold='F12')
    p3 = next(r for r in e3 if r['variant'] == 'pooled all envs')
    assert p3['value']['p_blocked_8m']['auroc'] == 1.0 and p3['value']['p_blocked_8m']['ece'] == 0.0
    assert p3['value']['near_bearing_blocked_within']['precision'] == 1.0
    assert p3['value']['tube_false_block']['fraction'] == 0.0
    # a predictor that is 2x too far: overshoots near cells, misses blocked cells
    far = {k: (v * 2 if not k.startswith('fan_p') else np.zeros_like(v)) for k, v in perfect.items()}
    pf = ev.PredictionSet('far', 'model', True, rows, fold='F4', sha256='f', arrays=far)
    v = next(r for r in ev.e2_collider_cells(pf, s, lab, th, fold='F4') if r['variant'] == 'pooled all envs')['value']
    assert v['near_overshoot_fraction'] == 1.0 and v['fan']['recall'] < 0.5
    near = {k: (v * 0.5 if not k.startswith('fan_p') else v) for k, v in perfect.items()}
    pn = ev.PredictionSet('near', 'model', True, rows, fold='F4', sha256='n', arrays=near)
    v3 = next(r for r in ev.e3_fan_quality(pn, s, lab, th, fold='F12') if r['variant'] == 'pooled all envs')['value']
    assert v3['tube_false_block']['fraction'] == 1.0
    # grid-only predictors get a geometric fan (hard probabilities)
    g_only = ev.PredictionSet('grid', 'baseline', True, rows, baseline_id='B2', sha256='g',
                              arrays=dict(grid_q50=true.copy()))
    v4 = next(r for r in ev.e3_fan_quality(g_only, s, lab, th, fold='F12') if r['variant'] == 'pooled all envs')
    assert v4['value']['fan_derived_from_grid']


def test_gate_pass_detection_and_gate_opening(scene):
    s, _ = scene
    t = np.arange(0, 3, 0.06)
    cue = np.tile([0.5, 0.55], (len(t), 1))
    cue[t > 1.5] = (0.1, 0.3)
    passes = ev.detect_gate_passes(t, cue)
    assert len(passes) == 1 and 1.4 < passes[0] < 1.6
    th = _thresholds()
    rows = s.rows()
    c = ev.store_cache(s)
    lab = FakeLabels(len(rows))
    # an open gate: the predictor sees 40 m through it; a smeared gate: 0.5 x the distance
    open_ = ev.PredictionSet('open', 'model', True, rows, fold='F12', sha256='o',
                             arrays=dict(grid_q20=np.full((len(rows), 18, 32), 40.0, np.float32),
                                         grid_q50=np.full((len(rows), 18, 32), 40.0, np.float32)))
    g = ev._gate_opening(open_, s, th['E3'], {0: [4.0]}, rows)
    assert g['Minus Two']['passes'] == 1 and g['Minus Two']['frames'] > 10 and g['Minus Two']['blocked'] == 0
    smear = ev.PredictionSet('smear', 'model', True, rows, fold='F12', sha256='s',
                             arrays=dict(grid_q20=np.ones((len(rows), 18, 32), np.float32),
                                         grid_q50=np.ones((len(rows), 18, 32), np.float32)))
    g = ev._gate_opening(smear, s, th['E3'], {0: [4.0]}, rows)
    assert g['Minus Two']['blocked'] == g['Minus Two']['frames']
    del c, lab


def test_e8_leaks_detect_a_ring_shortcut(scene):
    s, _ = scene
    th = _thresholds()
    rows = s.rows()[:6]

    def invariant(frames, quat):
        return dict(grid_q50=np.full((len(frames), 18, 32), 7.0))

    def ring_shortcut(frames, quat):
        # reads "far" wherever the frame is green-cyan (the painted ring): a route shortcut
        f = frames.astype(np.float64)
        green = ((f[..., 1] > 200) & (f[..., 0] < 100)).reshape(len(f), 18, 14, 32, 14).mean(axis=(2, 4))
        return dict(grid_q50=5.0 + 50.0 * green)

    ok = ev.e8_leak_tests(invariant, s, rows, th, fold='F12')
    assert {r['value']['perturbation'] for r in ok} >= {'ring_on_obstacle', 'ring_on_free_space',
                                                         'hud_glyph_scramble', 'ghost_trails_inserted'}
    assert all(r['passed'] for r in ok if r['passed'] is not None)
    bad = {r['value']['perturbation']: r for r in ev.e8_leak_tests(ring_shortcut, s, rows, th, fold='F12')}
    assert bad['ring_on_obstacle']['passed'] is False
    assert bad['ring_removed']['passed'] is None           # no ring mask available: reported as skipped


def test_leak_perturbations_are_local():
    rng = np.random.default_rng(1)
    f = np.full((252, 448, 3), 80, np.uint8)
    x0, y0, x1, y1 = leaks.zone_px('centre_line_marker')
    f[y0 + 10:y0 + 14, x0 + 5:x0 + 9] = 250                # a white glyph
    g, zones = leaks.scramble_hud(f, rng)
    assert (g[~zones] == f[~zones]).all() and zones.any()
    p, stroke, disc = leaks.paint_ring(f, (224, 126), 20)
    assert stroke.sum() > 50 and (p[stroke] == leaks.RING_RGB).all() and disc.sum() > stroke.sum()
    import cv2
    hsv = cv2.cvtColor(np.array([[leaks.RING_RGB]], np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
    assert 35 <= hsv[0] <= 100 and hsv[1] > 75 and hsv[2] > 75      # the overlays' ring colour rule
    r = leaks.remove_ring(p, stroke)
    assert np.abs(r.astype(int) - f.astype(int))[stroke].mean() < 20
    q, lines = leaks.insert_ghost_trails(f, rng)
    assert lines.any() and (q[~lines] == f[~lines]).mean() > 0.98


# ----------------------------------------------------------------------------- baselines

def test_b3_isotonic_calibration_and_b4_affine_ceiling():
    rng = np.random.default_rng(0)
    r = rng.uniform(1.0, 30.0, 5000)
    disp = 3.0 / r + 0.2 + rng.normal(0, 0.005, r.shape)       # affine inverse depth
    cal = BL.fit_monotone(disp, r, fold='F12', envs=['Straw Bale'])
    assert (np.diff(cal.y) <= 1e-12).all()                       # non-increasing ln range
    test = np.array([3.0 / 2 + 0.2, 3.0 / 10 + 0.2])
    assert np.allclose(cal(test), [2.0, 10.0], rtol=0.1)
    again = BL.MonotoneCalibration.from_json(json.loads(json.dumps(cal.to_json())))
    assert again.sha256() == cal.sha256() and np.isnan(cal(np.array([np.nan]))[0])
    gv = rng.uniform(1.0, 20.0, (2, 18, 32))
    d = 2.0 / gv + 0.1
    gk = np.full((2, 18, 32), K.EXACT, np.uint8)
    gk[1] = K.UNKNOWN
    gk[1, 0, :4] = K.EXACT                                        # only 4 anchors: no prediction
    out = BL.per_frame_affine(d, gv, gk, min_anchors=8)
    assert np.allclose(out[0], gv[0], rtol=1e-6) and np.isnan(out[1]).all()
    assert list(BL.isotonic_nonincreasing([3, 1, 2, 0])) == [3, 1.5, 1.5, 0]


def test_pool_cells_and_ray_norm():
    img = np.full((1, 252, 448), 10.0)
    img[0, 20, 30] = 2.0
    g = BL.pool_cells(img, 'min')
    assert g.shape == (1, 18, 32) and g[0, 1, 2] == 2.0 and g[0, 0, 0] == 10.0
    valid = np.ones((1, 252, 448), bool)
    valid[0, :14, :14] = False
    assert np.isnan(BL.pool_cells(img, 'min', valid)[0, 0, 0])
    n = BL.ray_norm()
    assert n.shape == (252, 448) and abs(n[126, 224] - 1.0) < 1e-4 and n[0, 0] > 1.9
    t = np.zeros((1, 36, 64))
    t[0, 3, 5] = 7.0
    assert BL.teacher_to_cells(t)[0, 1, 2] == 7.0


def test_b1_split_looming_runs_on_store_frames(scene):
    s, _ = scene
    rows = s.rows()[:12]
    p = ev.baseline_b1(s, rows)
    assert p.baseline_id == 'B1' and p.causal and p.arrays['side'].shape == (12,)
    assert set(np.unique(p.arrays['side'])) <= {0, 1, 2}


def test_model_inference_path_builds_a_prediction_set(scene):
    torch = pytest.importorskip('torch')
    from haltere.obstacles.overlays import OverlayMasks
    s, _ = scene
    rows = s.rows()[:5]

    class Tiny(torch.nn.Module):
        def forward(self, x, g):
            assert x.shape[1:] == (4, 252, 448) and g.shape[1:] == (3,)
            b = x.shape[0]
            grid = torch.full((b, 2, 18, 32), float(np.log(5.0)))
            grid[:, 0] -= 0.2
            return dict(grid_log_range=grid, fan_log_free=torch.zeros(b, 2, 4, 9), fan_logit=torch.zeros(b, 2, 4, 9))

    z = np.zeros((252, 448), bool)
    masks = lambda f: OverlayMasks(hud=z, ring=z, ghost=z, propeller=z)
    p = BL.model_predictions('unused', s, rows, None, device='cpu', masks_fn=masks, net=Tiny(),
                             card=dict(fold='F12', model_sha256='tiny'))
    assert p.kind == 'model' and p.sha256 == 'tiny' and p.fold == 'F12'
    assert np.allclose(p.arrays['grid_q50'], 5.0) and (p.arrays['grid_q20'] < p.arrays['grid_q50']).all()
    assert np.allclose(p.arrays['fan_p8'], 0.5)
    fn = BL.model_predict_fn(Tiny(), device='cpu', masks_fn=masks)
    assert np.allclose(fn(s.frames(rows[:2]), s.index['quat'][rows[:2]])['grid_q50'], 5.0)


def test_thresholds_have_decision_parameters_and_hash_is_stable():
    obj, sha = ev.load_thresholds(require_frozen=False)
    assert ev.unset_decision_parameters(obj) == []
    assert ev.thresholds_sha256(obj) == sha
    p = ev.DecisionParams.from_thresholds(obj)
    assert p.quantile == 'fan_q20' and p.on_frames >= 1
    draft = json.loads(json.dumps(obj))
    draft['decision']['on_frames'] = None
    assert ev.unset_decision_parameters(draft) == ['decision.on_frames']
    with pytest.raises(ValueError):
        ev.DecisionParams.from_thresholds(draft)


def test_freeze_refuses_unset_decision_parameters(tmp_path):
    obj, _ = ev.load_thresholds(require_frozen=False)
    obj = dict(obj, frozen=False, frozen_at=None, sha256=None)
    obj['decision'] = dict(obj['decision'], quantile=None)
    path = tmp_path / 't.json'
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        ev.freeze_thresholds(path)


def test_ledger_marks_rescores_and_pooled_flags(tmp_path):
    led = ev.Ledger(tmp_path / 'l.jsonl')
    led.append('abc', 'F12', 't1', [dict(metric='E1')])
    led.append('abc', 'F12', 't1', [dict(metric='E1')], once=False, note='explicit re-score')
    e = led.entries()
    assert e[0]['rescore'] is False and e[1]['rescore'] is True
    assert ev.pooled_flags('F12', ['Minus Two', 'Pine Valley'])['held_out'] is True
    assert ev.pooled_flags('F12', ['Minus Two', 'Straw Bale'])['held_out'] is None
    assert ev.pooled_flags('F12', ['Straw Bale'])['seen_environment'] is True


def test_score_orchestrates_and_flags_held_out(scene):
    s, events = scene
    th = _thresholds()
    rows = s.rows()
    grid = np.full((len(rows), 18, 32), 20.0, np.float32)
    pred = ev.PredictionSet('F12-test', 'model', True, rows, fold='F12', sha256='z',
                            arrays=dict(grid_q50=grid, grid_q20=grid.copy()))
    recs = ev.score(pred, s, None, events, th, fold='F12', metrics=('E1', 'E4', 'E5', 'E6', 'E7'))
    metrics = {r['metric'] for r in recs}
    assert metrics >= {'E1', 'E4', 'E5', 'E6', 'E7'}
    lat = {r['latency_s'] for r in recs if r['metric'] == 'E4'}
    assert lat == {0.065, 0.1, 0.15}
    assert ev.any_held_out(recs)
    json.dumps(recs, default=ev._json_default)


def test_offline_evaluation_modules_stay_out_of_runtime_imports():
    from haltere.obstacles import OFFLINE_MODULES
    for m in ('haltere.obstacles.baselines', 'haltere.obstacles.leaks', 'haltere.obstacles.lateral_cache',
              'haltere.obstacles.split_looming'):
        assert m in OFFLINE_MODULES
    import ast
    root = Path(ev.__file__).resolve().parent
    for name in ('contract.py', 'overlays.py', 'model.py'):
        tree = ast.parse((root / name).read_text(encoding='utf-8'))
        mods = {getattr(n, 'module', None) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not mods & {'evaluate', 'baselines', 'leaks', 'lateral_cache', 'labels', '.labels'}


# ----------------------------------------------------------------------------- CLI end to end on a real store

def test_cli_scores_a_prediction_set_once_on_a_frame_store(tmp_path, monkeypatch):
    from haltere.obstacles import thermal
    monkeypatch.delenv(thermal.FLIGHT_LOCK_ENV, raising=False)
    from haltere.obstacles import labels as L
    from haltere.obstacles import store as ST
    root = tmp_path / 'store'
    w = ST.StoreWriter(root)
    runs = [straight_run('minus-a', 'Minus Two'), straight_run('straw-clean', 'Straw Bale')]
    w.write_plan([dict(source_id=r['name'], flight=r['name'], source='run_video', env=r['env'], aliases=[r['name']])
                  for r in runs], {})
    rng = np.random.default_rng(0)
    for k, r in enumerate(runs):
        n = len(r['t'])
        rows = ST.empty_index(n)
        rows['env'] = ENV_CODE[r['env']]
        rows['grade'] = ST.Grade.GOOD
        rows['source'] = ST.Source.RUN_VIDEO
        rows['t_phase'] = r['t']
        rows['t_wall'] = 1.7e9 + r['t']
        rows['pos'], rows['vel'] = r['pos'], r['vel']
        with w.begin_run(k) as rw:
            rw.append(rng.integers(0, 255, (n,) + ST.FRAME_SHAPE, dtype=np.uint8), rows)
            rw.close()
    w.write_json('clean_windows.json', dict(windows=[dict(run_id=1, alias='straw-clean', start_phase_s=0.5,
                                                          end_phase_s=4.5)]))
    man = w.finalize()
    s = ST.FrameStore(root)
    lw = L.LabelWriter(root, len(s), man['index_sha256'])
    lw.finalize()
    ev_list = [dict(event_id=0, store_event_id=-1, run='minus-a', flight='minus-a', env='Minus Two',
                    kind='terminal_impact', t_phase=5.0, point_w=[30.15, 0.0, 0.0], obstacle='pillar',
                    unique_obstacle='minus/pillar-1', primary_free_side='right', accepted_free_sides=['right'],
                    lateral=True, source='test', blind=False)]
    (root / 'labels' / 'events.json').write_text(json.dumps(dict(events=ev_list)))
    rows = np.arange(len(s))
    grid = true_range_grid(s, rows, ev_list[0]['point_w'])
    pred = ev.PredictionSet('F12-cli', 'model', True, rows, fold='F12', sha256='cli-sha',
                            arrays=dict(grid_q50=grid, grid_q20=grid.copy()))
    pred.save(tmp_path / 'pred.npz')
    th = tmp_path / 'thresholds.json'
    obj, _ = ev.load_thresholds(require_frozen=False)
    th.write_text(json.dumps(dict(obj, frozen=False, frozen_at=None, sha256=None)))
    args = ['score', str(tmp_path / 'pred.npz'), '--store', str(root), '--thresholds', str(th),
            '--ledger', str(tmp_path / 'ledger.jsonl'), '--out', str(tmp_path / 'out'), '--fold', 'F12',
            '--once', '--flight-lock', str(tmp_path / 'FLIGHT_LOCK'), '--metrics', 'E1,E4,E6,E7']
    with pytest.raises(RuntimeError):
        ev.main(args)                                   # thresholds not frozen: no held-out score
    ev.main(['freeze', '--thresholds', str(th)])
    ev.main(args)
    entries = ev.Ledger(tmp_path / 'ledger.jsonl').entries()
    assert len(entries) == 1 and entries[0]['predictor_sha256'] == 'cli-sha'
    recs = entries[0]['records']
    e1 = next(r for r in recs if r['metric'] == 'E1' and r['env'] == 'Minus Two')
    assert e1['held_out'] and abs(e1['value']['bins']['2-4']['median'] - 1.0) < 1e-3
    assert e1['thresholds_sha256'] == ev.load_thresholds(th)[1] and e1['store_index_sha256'] == man['index_sha256']
    assert e1['labels_manifest_sha256'] is not None
    assert any(r['metric'] == 'E6' and r['env'] == 'Straw Bale' and r['seen_environment'] for r in recs)
    with pytest.raises(RuntimeError):
        ev.main(args)                                   # --once: the same frozen version is scored once
    with pytest.raises(SystemExit):
        ev.main([a for a in args if a not in ('--flight-lock', str(tmp_path / 'FLIGHT_LOCK'))])


# ----------------------------------------------------------------------------- reproduction (session data)

@pytest.mark.skipif(not (LATERAL_DATA / 'manifest.json').exists(), reason='lateral study caches not present')
def test_reproduce_constant_side_on_the_lateral_study_caches():
    from haltere.obstacles.lateral_cache import LateralCacheStore, MONO_DEPTH_UNSCORED, LATERAL_SCORING
    th = _thresholds()
    store = LateralCacheStore(LATERAL_DATA)
    events = store.events()
    rows = np.arange(len(store))
    res = ev.e4_event_results(ev.baseline_b0(store, rows, ev.Side.RIGHT), store, events, th, latency_s=0.0,
                              rate_hz=None)
    mono = [r for r in res if r['run'] not in MONO_DEPTH_UNSCORED]
    assert len(res) == len(LATERAL_SCORING) == 11 and len(mono) == 10
    assert sum(r['success'] for r in mono) == 7                    # the mono-depth study's 7/10
    assert sum(r['success'] for r in res) == 8
    left = ev.e4_event_results(ev.baseline_b0(store, rows, ev.Side.LEFT), store, events, th, latency_s=0.0,
                               rate_hz=None)
    assert sum(r['success'] for r in left if r['run'] not in MONO_DEPTH_UNSCORED) == 4
