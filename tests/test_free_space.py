"""FreeSpace corridor planner v1, perception half (haltere.vision.free_space): synthetic scenes, tracks, scale,
interface round trip and the depth-process stage."""
import json
import math
import os

import numpy as np
import pytest

from haltere.vision import free_space as fs
from haltere.vision.camera import quat_wxyz_to_mat

CFG, RAW, SHA = fs.load_config(version=fs.VERSION)
RGB0 = np.zeros((252, 448, 3), np.uint8)
ALL_VALID = np.ones((36, 64), bool)
V6 = np.array([6., 0., 0.])


def quat_pitch_down(deg, yaw_deg=0.):
    """World-from-body quaternion: yaw about +z, then nose down by ``deg`` (positive rotation about body +y)."""
    ay, ap = math.radians(yaw_deg) / 2, math.radians(deg) / 2
    qy = np.array([math.cos(ay), 0, 0, math.sin(ay)])
    qp = np.array([math.cos(ap), 0, math.sin(ap), 0])
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = qp
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def render(geo, R, boxes=(), floor=None, ceil=None, ramp=None):
    """Per-block inverse optical depth (1/Z, 0 = nothing) of a scene seen from a camera at the origin: axis-aligned
    boxes (lo, hi), a floor / ceiling plane at z, a ramp (x0, z0, slope) rising from x0 over the floor."""
    rw = geo.rays_body @ R.T
    t = np.full(len(rw), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        if floor is not None:
            tf = floor / rw[:, 2]
            t = np.where((rw[:, 2] < 0) & (tf > 0), np.minimum(t, tf), t)
        if ceil is not None:
            tc = ceil / rw[:, 2]
            t = np.where((rw[:, 2] > 0) & (tc > 0), np.minimum(t, tc), t)
        if ramp is not None:
            x0, z0, s = ramp
            tr = (z0 - x0 * s) / (rw[:, 2] - rw[:, 0] * s)
            ok = (tr > 0) & (tr * rw[:, 0] >= x0)
            t = np.where(ok, np.minimum(t, tr), t)
        for lo, hi in boxes:
            lo, hi = np.asarray(lo, float), np.asarray(hi, float)
            t1, t2 = lo[None] / rw, hi[None] / rw
            tmin = np.nanmax(np.minimum(t1, t2), 1)
            tmax = np.nanmin(np.maximum(t1, t2), 1)
            hit = (tmax >= tmin) & (tmax > 0)
            t = np.where(hit, np.minimum(t, np.maximum(tmin, 0)), t)
    return np.where(np.isfinite(t), 1 / t, 0.).reshape(36, 64)


def cue_of(geo, R, point):
    c = geo.M @ (R.T @ np.asarray(point, float))
    return ((geo.cx + geo.f * c[0] / c[2]) / geo.W, (geo.cy + geo.f * c[1] / c[2]) / geo.H)


def plan(scene, motor='brain08', pitch=25., valid=None, v=V6, ring=(15., 0., 0.), scale=(1., 0.)):
    pl = fs.FreeSpacePlanner(CFG, motor, enabled=True)
    q = quat_pitch_down(pitch)
    R = quat_wxyz_to_mat(q)
    d = render(pl.geo, R, **scene)
    out = pl.process(RGB0, d, ALL_VALID if valid is None else valid, q, v, cue_of(pl.geo, R, ring), 1.0,
                     scale_override=scale)
    return out, pl


FLOOR = dict(floor=-0.8)


# ----------------------------------------------------------------------------- config

def test_config_is_the_declared_v1():
    assert CFG.version == 1 and RAW['schema'] == fs.SCHEMA and RAW['enabled'] is False
    assert fs.horizon_s(fs.load_response_models(CFG.response_models)['brain08'], CFG.plan) == pytest.approx(1.53, abs=.01)
    assert fs.horizon_s(fs.load_response_models(CFG.response_models)['fast_pd'], CFG.plan) == pytest.approx(1.21, abs=.01)
    pl = fs.FreeSpacePlanner(CFG, 'brain08', enabled=True)
    assert len(pl.cand_az) == 155 and (pl.cand_el >= 0).all()


def test_config_refuses_edits_after_freeze_and_other_versions(tmp_path):
    obj = dict(RAW, frozen=True, frozen_at='x', sha256=fs.config_sha256(RAW))
    p = tmp_path / 'fs.json'
    p.write_text(json.dumps(obj), encoding='utf-8')
    fs.load_config(p, require_frozen=True)
    obj['plan'] = dict(obj['plan'], r_lat_m=0.5)
    p.write_text(json.dumps(obj), encoding='utf-8')
    with pytest.raises(ValueError, match='changed after the freeze'):
        fs.load_config(p)
    draft = dict(RAW)
    draft.pop('frozen', None)
    draft.pop('sha256', None)
    p.write_text(json.dumps(draft), encoding='utf-8')
    with pytest.raises(ValueError, match='not frozen'):
        fs.load_config(p, require_frozen=True)
    p.write_text(json.dumps(dict(draft, version=2)), encoding='utf-8')
    with pytest.raises(ValueError, match='version'):
        fs.load_config(p)


def test_stop_speed_formula():
    for delay in (0.3, 0.55):
        for D in (0.3, 0.5, 2.0, 5.0, 9.0):
            want = 0. if D <= 0.5 else 8 * (math.sqrt(delay ** 2 + (D - 0.5) / 4) - delay)
            assert fs.stop_speed(D, delay) == pytest.approx(want)
            v = fs.stop_speed(D, delay)
            if D > .5:
                assert v * delay + v * v / 16 + .5 == pytest.approx(D)


# ----------------------------------------------------------------------------- synthetic scenes

def test_flat_floor_is_clear_with_the_floor_height():
    for motor in fs.MOTORS:
        out, pl = plan(FLOOR, motor)
        assert pl.last['kind'] == 'clear' and out['valid'] == 1.
        assert out['az'] == 0. and out['el'] == 0. and out['cls'] == 0.
        assert abs(out['h_floor'] - 0.8) <= 0.04
        assert np.isnan(out['rise']) and np.isnan(out['h_ceil']) and out['n_block_ring'] == 0


def test_pillar_right_of_the_line_shifts_left():
    pillar = dict(floor=-0.8, boxes=[((6., -1.22, -0.8), (6.8, -0.42, 1.4))])
    for motor in fs.MOTORS:
        out, pl = plan(pillar, motor)
        assert pl.last['kind'] == 'shift'
        assert out['cls'] == fs.CLS_LEFT and out['az'] > 0 and out['feasible'] == 1.
        assert out['l_ok'] == 1. and out['l_az'] == out['az']
        assert 0 < out['az'] <= CFG.plan.max_az_deg and 0 <= out['el'] <= CFG.plan.max_el_deg


def test_face_on_wall_blocks_with_the_stopping_speed():
    wall = dict(floor=-0.8, boxes=[((5., -20., -0.8), (5.3, 20., 3.))])
    for motor in fs.MOTORS:
        out, pl = plan(wall, motor)
        assert pl.last['kind'] == 'blocked' and out['cls'] == 0. and out['az'] == 0. and out['el'] == 0.
        assert np.isnan(out['l_az']) and np.isnan(out['r_az']) and np.isnan(out['v_el'])
        L = pl.last
        D = float(L['free'][len(pl.cand_az):][L['lag_elig']].max())
        assert D == pytest.approx(5.0, abs=.3)
        assert out['v_cap'] == pytest.approx(fs.stop_speed(D, pl.model.delay_s))
        assert out['d_free_ring'] == pytest.approx(5.0, abs=.3)


def test_ceiling_gives_its_height_and_blocks_the_vertical_escape():
    low = [((6., -10., -0.8), (6.3, 10., 0.3))]
    out, pl = plan(dict(floor=-0.8, boxes=low), 'brain08')
    assert pl.last['kind'] == 'shift' and out['cls'] == fs.CLS_VERTICAL and np.isfinite(out['v_el'])
    out, pl = plan(dict(floor=-0.8, ceil=1.0, boxes=low), 'brain08')
    assert abs(out['h_ceil'] - 1.0) <= 0.05
    assert np.isnan(out['v_el']) and out['cls'] != fs.CLS_VERTICAL
    # the region above unobserved (masked): an up-escape needs observed free space
    rw = pl.geo.rays_body @ quat_wxyz_to_mat(quat_pitch_down(25)).T
    above = np.degrees(np.arctan2(rw[:, 2], np.hypot(rw[:, 0], rw[:, 1]))).reshape(36, 64) > 2.
    out, pl = plan(dict(floor=-0.8, boxes=low), 'brain08', valid=ALL_VALID & ~above)
    assert np.isnan(out['v_el']) and out['cls'] != fs.CLS_VERTICAL
    assert not pl.last['static_elig'][(pl.cand_az == 0) & (pl.cand_el > 0)].any()


def test_rising_ground_but_not_a_pillar_or_a_masked_base_wall():
    out, _ = plan(dict(floor=-0.8, ramp=(4., -0.8, math.tan(math.radians(20)))), 'brain08')
    assert np.isfinite(out['rise']) and out['rise'] > 0.3 and 4.0 < out['rise_x'] < 8.0
    out, _ = plan(dict(floor=-0.8, boxes=[((6., -0.2, -0.8), (6.4, 0.2, 2.0))]), 'brain08')
    assert np.isnan(out['rise'])
    # a wall whose base is masked: the visible floor ends, then the wall face starts well above it
    wall = dict(floor=-0.8, boxes=[((6., -20., -0.8), (6.3, 20., 3.))])
    pl = fs.FreeSpacePlanner(CFG, 'brain08', enabled=True)
    q = quat_pitch_down(25)
    R = quat_wxyz_to_mat(q)
    rw = (pl.geo.rays_body @ R.T).reshape(36, 64, 3)
    d = render(pl.geo, R, **wall)
    base = (np.abs(1 / np.maximum(d, 1e-9) * rw[..., 0] - 6.) < .2) & (1 / np.maximum(d, 1e-9) * rw[..., 2] < 0.)
    valid = ALL_VALID & ~base
    out = pl.process(RGB0, d, valid, q, V6, cue_of(pl.geo, R, (15, 0, 0)), 1., scale_override=(1., 0.))
    assert np.isnan(out['rise'])


def test_hud_only_disparity_gives_no_blocking_points():
    hud = np.zeros((252, 448), bool)
    hud[200:240, 150:300] = True            # a HUD panel
    hud[20:40, 10:120] = True
    frac = hud.reshape(36, 7, 64, 7).mean(axis=(1, 3))
    d = 1. + 9. * frac                      # 10 on HUD pixels, 1 elsewhere (block means)
    valid = frac <= CFG.masks['block_mask_max_fraction']
    pl = fs.FreeSpacePlanner(CFG, 'brain08', enabled=True)
    q = quat_pitch_down(25)
    R = quat_wxyz_to_mat(q)
    # scale: d = 1 lies beyond the 30 m range, d = 10 at 6 m (inside the horizon if it leaked)
    out = pl.process(RGB0, d, valid, q, V6, cue_of(pl.geo, R, (15, 0, 0)), 1., scale_override=(1 / 60., 0.))
    assert pl.last['kind'] == 'clear' and out['n_block_ring'] == 0 and len(pl.last['cloud']) == 0
    assert not pl.last['cnt'].any()
    leaked = pl.process(RGB0, d, ALL_VALID, q, V6, cue_of(pl.geo, R, (15, 0, 0)), 2., scale_override=(1 / 60., 0.))
    assert len(pl.last['cloud']) > 0        # the same disparity without the mask would have produced points
    assert leaked['valid'] == 1.


def test_kinds_without_evidence():
    pl = fs.FreeSpacePlanner(CFG, 'brain08', enabled=False)
    assert fs.kind_name(pl.process(RGB0, np.ones((36, 64)), ALL_VALID, [1, 0, 0, 0], V6, (.5, .5), 1.)['kind']) == 'off'
    pl = fs.FreeSpacePlanner(CFG, 'brain08', enabled=True)
    q = quat_pitch_down(25)
    R = quat_wxyz_to_mat(q)
    d = render(pl.geo, R, **FLOOR)
    cue = cue_of(pl.geo, R, (15, 0, 0))
    run = lambda **kw: fs.kind_name(pl.process(RGB0, d, ALL_VALID, kw.get('q', q), kw.get('v', V6), kw.get('cue', cue),
                                               1., stale=kw.get('stale', False),
                                               scale_override=kw.get('scale', (1., 0.)))['kind'])
    assert run(q=None) == 'no_pose'
    assert run(stale=True) == 'stale'
    assert run(cue=None) == 'no_ring'
    assert run(cue=dict(u=cue[0], v=cue[1], edge=True)) == 'no_ring'
    assert run(cue=(0.001, 0.5)) == 'no_ring'
    assert run(v=np.array([1.2, 0, 0])) == 'slow'
    assert run(scale=None) == 'no_scale'
    assert run() == 'clear'
    for k in fs.PLAN_KINDS:
        assert fs.kind_name(fs.kind_index(k)) == k


# ----------------------------------------------------------------------------- tracks and scale

def _project(geo, R, points_w, pos):
    c = (np.asarray(points_w) - pos) @ R @ geo.M.T
    return np.stack([geo.cx + geo.f * c[:, 0] / c[:, 2], geo.cy + geo.f * c[:, 1] / c[:, 2]], 1), c[:, 2]


def _scene_points(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(3, 25, n), rng.uniform(-6, 6, n), rng.uniform(-1, 3, n)], 1)


def test_hud_usable_matches_looming2_hud_mask():
    pytest.importorskip('cv2')
    from haltere.vision.looming2 import hud_mask
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (252, 448, 3), dtype=np.uint8)
    img[100:120, 50:200] = [230, 235, 228]           # white HUD text
    img[60:70, 300:320] = [200, 190, 120]            # bright but coloured: usable
    np.testing.assert_array_equal(fs.hud_usable(img), hud_mask(img))


def test_pure_rotation_gives_zero_derotated_flow():
    geo = fs.Geometry(CFG.camera)
    Rp = quat_wxyz_to_mat(quat_pitch_down(20, 0.))
    Rc = quat_wxyz_to_mat(quat_pitch_down(23, 4.))
    Pw = _scene_points()
    P, zp = _project(geo, Rp, Pw, np.zeros(3))
    Q, zc = _project(geo, Rc, Pw, np.zeros(3))
    ok = (zp > 0) & (zc > 0)
    H = fs.derotation_homography(geo, Rp, Rc)
    ph = np.c_[P[ok], np.ones(ok.sum())] @ H.T
    ph = ph[:, :2] / ph[:, 2:]
    assert np.abs(Q[ok] - ph).max() < 1e-6
    # no translation: no anchor survives the inverse-depth range
    a = fs.parallax_anchors(P[ok], Q[ok], Rp, Rc, np.array([6., 0, 0]), .055, geo, CFG.tracks)
    assert a is None or np.all(np.abs(a['inv_z']) < 1e-9) or len(a['inv_z']) == 0


def test_synthetic_translation_recovers_inverse_depth():
    geo = fs.Geometry(CFG.camera)
    Rp = quat_wxyz_to_mat(quat_pitch_down(22, 0.))
    Rc = quat_wxyz_to_mat(quat_pitch_down(24, 2.))
    v = np.array([6., .4, .2])
    dt = .055
    Pw = _scene_points(800)
    P, zp = _project(geo, Rp, Pw, np.zeros(3))
    Q, zc = _project(geo, Rc, Pw, v * dt)
    ok = (zp > 0) & (zc > 0) & (P[:, 0] > 0) & (P[:, 0] < 448) & (P[:, 1] > 0) & (P[:, 1] < 252)
    a = fs.parallax_anchors(P[ok], Q[ok], Rp, Rc, v, dt, geo, CFG.tracks)
    assert a is not None and len(a['inv_z']) > 100
    # the true inverse optical depth of each kept anchor in the current frame
    truth = {tuple(np.round(q, 6)): 1 / z for q, z in zip(Q[ok], zc[ok])}
    t = np.array([truth[tuple(np.round(q, 6))] for q in a['Q']])
    assert np.median(np.abs(a['inv_z'] / t - 1)) < 0.01
    assert np.percentile(np.abs(a['inv_z'] / t - 1), 95) < 0.05


def test_lk_tracker_recovers_a_textured_wall_depth():
    cv2 = pytest.importorskip('cv2')
    cfg = CFG
    geo = fs.Geometry(cfg.camera)
    q = quat_pitch_down(30.)                # camera axis horizontal, along the velocity: FOE at the centre
    R = quat_wxyz_to_mat(q)
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur(rng.uniform(0, 255, (252, 448)).astype(np.float32), (0, 0), 1.2)
    tex = cv2.resize(cv2.resize(tex, (224, 126)), (448, 252))
    img = np.clip(np.stack([tex] * 3, -1), 0, 255).astype(np.uint8)
    Z, v, dt = 8.0, np.array([6., 0, 0]), .055
    s = Z / (Z - v[0] * dt)
    A = np.array([[s, 0, geo.cx * (1 - s)], [0, s, geo.cy * (1 - s)]])
    img2 = cv2.warpAffine(img, A, (448, 252), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    tr = fs.ScaleTracker(cfg, geo)
    valid = ALL_VALID
    disp = np.full((36, 64), 1.)
    tr.update(0., img, valid, R, v, disp, usable=np.ones((252, 448), bool))
    out = tr.update(dt, img2, valid, R, v, disp, usable=np.ones((252, 448), bool))
    a = out['anchors']
    assert a is not None and len(a['inv_z']) > 50
    assert np.median(a['inv_z']) == pytest.approx(1 / (Z - v[0] * dt), rel=0.05)


def test_scale_fit_pools_and_the_guard_trips_on_low_spread():
    tr = fs.ScaleTracker(CFG)
    rng = np.random.default_rng(0)
    d = rng.uniform(0.2, 2.0, 200)
    y = 0.25 * d + 0.02 + rng.normal(0, 0.003, 200)
    y[:10] += 0.5                           # outliers
    fit = fs.huber_fit(d, y, np.ones(200), 4, 1.345, 0.01)
    assert fit[0][0] == pytest.approx(0.25, abs=0.01) and fit[0][1] == pytest.approx(0.02, abs=0.01)
    tr.N, tr.rhs, tr.n_eff = fit[1], fit[2], 200.
    wide = np.linspace(0.2, 2.0, 36 * 64).reshape(36, 64)
    st = tr.state(0., wide, ALL_VALID)
    assert st['scale_ok'] and not st['held'] and st['spread'] > 0.05
    narrow = np.linspace(1.0, 1.15, 36 * 64).reshape(36, 64)     # 0.25 * 0.12 < 0.05
    st = tr.state(1.0, narrow, ALL_VALID)
    assert not st['scale_ok'] and st['spread'] < 0.05
    st = tr.state(1.2, narrow, ALL_VALID)
    assert not st['scale_ok']
    tr.state(2.0, wide, ALL_VALID)
    st = tr.state(2.2, narrow, ALL_VALID)                          # within the hold: the last valid fit is held
    assert st['scale_ok'] and st['held']
    tr.n_eff = 30.
    st = tr.state(2.9, wide, ALL_VALID)                            # too few pooled anchors, hold expired
    assert not st['scale_ok']


# ----------------------------------------------------------------------------- interface

def test_plan_sample_round_trip():
    from haltere.liftoff import camera_process as cp
    assert cp.PLAN_FIELDS == fs.PLAN_FIELDS and cp.PLAN_KINDS == fs.PLAN_KINDS
    assert cp.plan_sample([0.] * len(fs.PLAN_FIELDS)) is None
    out, _ = plan(dict(floor=-0.8, boxes=[((6., -1.22, -0.8), (6.8, -0.42, 1.4))]))
    out = dict(out, seq=7., age=.1)
    values = fs.plan_values(out)
    s = cp.plan_sample(values)
    assert s['kind'] == 'shift' and s['valid'] is True and s['seq'] == 7.
    assert s['cls'] == 1. and s['v_el'] is None and s['v_ok'] is None
    back = cp.plan_values_of(s)
    np.testing.assert_array_equal(np.asarray(back), np.asarray(values))


def test_plan_stage_and_pose_lag(tmp_path):
    from haltere.liftoff import gap_stack
    p = tmp_path / 'free_space.json'
    p.write_text(json.dumps(dict(RAW, frozen=True, frozen_at='t', sha256=fs.config_sha256(RAW))), encoding='utf-8')
    spec = gap_stack.plan_spec('fast_velocity_brain_v1', mode='shadow', path=p)
    assert spec['motor'] == 'brain08' and spec['motor_index'] == 1 and spec['version'] == 1
    with pytest.raises(ValueError):
        gap_stack.plan_spec('fast_velocity_brain_v1', mode='on', stack_mode='shadow', path=p)
    with pytest.raises(OSError):
        gap_stack.plan_spec('fast_velocity_brain_v1', mode='shadow', path=tmp_path / 'missing.json')
    with pytest.raises(ValueError):
        gap_stack.plan_spec('fast_velocity_brain_v1', mode='off', path=p)
    draft = tmp_path / 'draft.json'
    draft.write_text(json.dumps(RAW), encoding='utf-8')
    with pytest.raises(ValueError, match='not frozen'):
        gap_stack.plan_spec('fast_velocity_pd_v1', mode='shadow', path=draft)
    stage = gap_stack.PlanStage(spec)
    geo = stage.planner.geo
    q = quat_pitch_down(25)
    R = quat_wxyz_to_mat(q)
    d = render(geo, R, floor=-0.8)
    cue = cue_of(geo, R, (15, 0, 0))
    perceived = dict(disparity=d, plan_valid=ALL_VALID, overlay_ms=1., depth_ms=9.)
    vals = stage.process(RGB0, 10., dict(u=cue[0], v=cue[1], edge=False), (q, V6), perceived, stale=False, seq=3,
                         frame_ms=.5, clock=lambda: 10.1)
    s = dict(zip(fs.PLAN_FIELDS, vals))
    assert fs.kind_name(s['kind']) == 'no_scale' and s['age'] == pytest.approx(.1) and s['seq'] == 3.
    assert s['depth_ms'] == 9. and s['overlay_ms'] == 1. and s['frame_ms'] == .5
    vals = stage.process(RGB0, 10.05, None, (q, V6), perceived, stale=True, seq=4, frame_ms=.5, clock=lambda: 10.1)
    assert fs.kind_name(vals[fs.PLAN_FIELDS.index('kind')]) == 'stale'
    st = stage.status()
    assert st['frames'] == 2 and st['counts']['stale'] == 1 and 'percentiles' in st
    rows = np.zeros((5, 14))
    rows[:, 1] = [9.9, 9.92, 9.94, 9.96, 9.98]
    rows[:, 0] = rows[:, 1]
    rows[:, 6] = 1.
    assert gap_stack.pose_for_plan(rows, 9.95, 9.99, .05) is not None
    assert gap_stack.pose_for_plan(rows, 10.1, 10.11, .05) is None        # nearest pose 0.12 s away


def test_depth_worker_publishes_plan_samples_to_their_own_array(monkeypatch, tmp_path):
    """gap_process_worker with the planner: the gap sample is still written, and a plan sample (same frame, its own
    array) follows; a cue that never arrives gives kind stale."""
    import multiprocessing as mp
    import queue
    import threading
    import time as _time
    cv2 = pytest.importorskip('cv2')
    torch = pytest.importorskip('torch')
    from haltere.liftoff import camera_process as cp
    from haltere.liftoff import gap_stack, scheduling
    from haltere.liftoff.geometry_shadow import MotionBuffer
    monkeypatch.setattr(scheduling, 'flight_process_priority', lambda: dict(applied=True))

    class FakeWorker:
        def __init__(self, spec):
            self.plan_masks = None

        def perceive(self, frame):
            assert self.plan_masks is not None and 'propeller' in self.plan_masks[0]
            return dict(overlay_ms=1., depth_ms=2., disparity=np.ones((36, 64), np.float32),
                        plan_valid=np.ones((36, 64), bool))

        def process(self, frame, capture_time, cue, pose, frame_ms, seq, perceived):
            return [capture_time] + [0.] * (len(cp.GAP_FIELDS) - 1)

        def status(self):
            return dict(ready=True)
    monkeypatch.setattr(gap_stack, 'GapFrameWorker', FakeWorker)

    class Status(queue.Queue):
        def cancel_join_thread(self):
            pass
    p = tmp_path / 'free_space.json'
    p.write_text(json.dumps(dict(RAW, frozen=True, frozen_at='t', sha256=fs.config_sha256(RAW))), encoding='utf-8')
    spec = dict(stride=1, plan=gap_stack.plan_spec('fast_velocity_pd_v1', mode='shadow', path=p))
    threads, cv2_threads = torch.get_num_threads(), cv2.getNumThreads()
    ctx = mp.get_context('spawn')
    slot, motion, status = gap_stack.FrameSlot(), MotionBuffer(), Status(maxsize=2)
    gap_out = ctx.Array('d', len(cp.GAP_FIELDS), lock=True)
    plan_out = ctx.Array('d', len(cp.PLAN_FIELDS), lock=True)
    stop = ctx.Event()
    now = _time.monotonic()
    for k in range(3):
        motion.publish(now + 60., now - .02 + .01 * k, 0., [0., 0., 5.], [1., 0., 0., 0.], [6., 0., 0.], [0., 0., 0.])
    worker = threading.Thread(target=gap_stack.gap_process_worker,
                              args=(slot, gap_out, motion, stop, status, spec, plan_out))
    worker.start()
    shared = np.frombuffer(plan_out.get_obj(), dtype=np.float64)
    try:
        slot.write(np.zeros((252, 448, 3), np.uint8), now - .01)
        slot.publish_cue(now - .01, dict(u=.5, v=.5, edge=False, aim_u=.5))
        deadline = _time.monotonic() + 5.
        while not shared[0] and _time.monotonic() < deadline:
            _time.sleep(.01)
        first = cp.plan_sample(shared.copy())
        slot.write(np.zeros((252, 448, 3), np.uint8), now - .005)        # no cue published for this frame
        deadline = _time.monotonic() + 5.
        while shared[0] == first['time'] and _time.monotonic() < deadline:
            _time.sleep(.01)
        second = cp.plan_sample(shared.copy())
    finally:
        stop.set()
        worker.join(5.)
        torch.set_num_threads(threads)
        cv2.setNumThreads(cv2_threads)
    assert not worker.is_alive()
    assert np.frombuffer(gap_out.get_obj(), dtype=np.float64)[0] > 0
    assert first['time'] == pytest.approx(now - .01) and first['seq'] == 1. and first['motor'] == 0.
    assert first['kind'] == 'no_scale' and first['depth_ms'] == 2. and first['age'] > 0
    assert second['kind'] == 'stale' and second['seq'] == 2.
    packets = []
    while not status.empty():
        packets.append(status.get_nowait())
    plan = packets[-1]['gap']['plan']
    assert plan['frames'] == 2 and plan['counts']['stale'] == 1 and plan['mode'] == 'shadow'


def test_process_camera_refuses_the_planner_in_the_camera_placement():
    from haltere.liftoff.camera_process import ProcessRetinaCamera
    with pytest.raises(ValueError, match='depth process'):
        ProcessRetinaCamera(gate_sensor=dict(focal_320=100., tilt_deg=30.), race_cues=True,
                            gap=dict(placement='camera', stride=1, plan=dict(mode='shadow')))


@pytest.mark.skipif(os.environ.get('HALTERE_SLOW_TESTS') is None, reason='timing check (set HALTERE_SLOW_TESTS)')
def test_plan_time_budget():
    import time
    pl = fs.FreeSpacePlanner(CFG, 'brain08', enabled=True)
    q = quat_pitch_down(25)
    R = quat_wxyz_to_mat(q)
    d = render(pl.geo, R, floor=-0.8, ceil=1.2, boxes=[((6., -1.22, -0.8), (6.8, -0.42, 1.4))])
    ms = []
    for i in range(50):
        t0 = time.perf_counter()
        pl.process(RGB0, d, ALL_VALID, q, V6, cue_of(pl.geo, R, (15, 0, 0)), float(i), scale_override=(1., 0.))
        ms.append(pl.last and (time.perf_counter() - t0) * 1e3)
    assert np.percentile(ms, 95) < 12
