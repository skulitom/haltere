"""Offline label builders: colliders (L5), flown tube (L1), impacts (L3), hindsight (L2), events, build."""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from haltere.obstacles import contract, store as S
from haltere.obstacles.labels import LabelKind as K, LabelSource, load_events, validate_event
from haltere.obstacles.labels import build as B
from haltere.obstacles.labels import colliders as C
from haltere.obstacles.labels import corridors as R
from haltere.obstacles.labels import hindsight as H
from haltere.obstacles.labels import impacts as I
from haltere.obstacles.labels import teacher as T
from haltere.obstacles.labels import tube as U

REPO = Path(__file__).resolve().parents[1]
LEVEL = np.array([1.0, 0.0, 0.0, 0.0])


def _quat_axis(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.deg2rad(deg) / 2
    return np.r_[np.cos(h), np.sin(h) * axis]


def _geometry(prims, ground=0.0):
    return dict(runtime_geometry_allowed=False, unknown_geometry=[], ground_plane_y=ground,
                primitives=[dict(instance_id=i + 1, kind='box', center=c, size=s, yaw_deg=y) for i, (c, s, y) in enumerate(prims)])


# ----------------------------------------------------------------------------- L5 colliders

def test_box_scene_matches_section_geometry():
    from haltere.liftoff.section_geometry import collision_depth
    rng = np.random.default_rng(0)
    geo = _geometry([([rng.uniform(-8, 8), rng.uniform(0.5, 4), rng.uniform(5, 25)], list(rng.uniform(0.3, 3, 3)),
                      float(rng.uniform(-90, 90))) for _ in range(25)])
    origin = np.array([0.3, 1.2, -0.5])
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ref = collision_depth(geo, origin, d, max_range=60.0)['range_m']
    got = C.BoxScene(geo).cast(origin, d, 60.0)
    assert np.array_equal(np.isfinite(ref), np.isfinite(got))
    assert np.allclose(ref[np.isfinite(ref)], got[np.isfinite(got)], atol=1e-9)


def test_collider_labels_wall_ahead_and_fan_bundle():
    # Unity frame: a wall 6 m ahead of a level camera at 1.5 m height (sim x forward = unity z).
    geo = _geometry([([0.0, 5.0, 6.5], [40.0, 10.0, 1.0], 0.0)], ground=-50.0)
    gv, gk, fv, fk = C.collider_labels(geo, np.array([0.0, 0.0, 1.5]), LEVEL)
    assert gv.shape == contract.GRID_SHAPE and fv.shape == contract.FAN_SHAPE
    cam_r = contract.cell_rays_body()
    centre = np.unravel_index(np.argmax(cam_r[..., 0]), cam_r.shape[:2])
    assert gk[centre] == K.EXACT and gv[centre] == pytest.approx(6.0 / cam_r[centre][0], rel=0.03)
    # level straight-ahead corridor: the wall face is at 6 m along the axis
    assert fk[1, 4] == K.EXACT and fv[1, 4] == pytest.approx(6.0, abs=0.05)
    # a thin pole 0.4 m beside the axis is inside the 0.5 m corridor; at 0.7 m it is not
    for off, hit in ((0.4, True), (0.7, False)):
        pole = _geometry([([-off, 0.0, 8.0], [0.05, 10.0, 0.05], 0.0)], ground=-50.0)   # sim y = -unity x
        h = C.fan_hits(pole, np.array([0.0, 0.0, 1.5]), LEVEL)
        assert np.isfinite(h[1, 4]) == hit


def test_collider_scene_refuses_unknown_geometry():
    geo = _geometry([([0, 1, 5], [1, 1, 1], 0.0)])
    geo['unknown_geometry'] = [dict(item='flag')]
    with pytest.raises(ValueError):
        C.BoxScene(geo)


# ----------------------------------------------------------------------------- L1 tube

def test_ray_tube_exit_straight_path():
    centres = np.c_[np.linspace(0, 10, 201), np.zeros(201), np.zeros(201)]
    d = np.array([[1.0, 0, 0], [0, 1.0, 0], [np.cos(0.1), np.sin(0.1), 0]])
    s = U.ray_tube_exit(np.zeros(3), d, centres, radius=0.35, s_max=60)
    assert s[0] == pytest.approx(10.35, abs=1e-6)
    assert s[1] == pytest.approx(0.35, abs=1e-6)
    assert s[2] == pytest.approx(0.35 / np.sin(0.1), rel=0.02)


def test_tube_labels_are_lower_bounds_and_stop_before_contacts():
    t = np.linspace(0, 5, 501)
    pos = np.c_[4.0 * t, np.zeros_like(t), np.full_like(t, 2.0)]
    # camera pitched so the optical axis is level: body pitched down 30 deg (y axis, positive = nose down in FLU)
    q = _quat_axis([0, 1, 0], 30.0)
    gv, gk, fv, fk = U.tube_labels(t, pos, 1.0, pos[100], q)
    assert set(np.unique(gk)) <= {K.UNKNOWN, K.LOWER} and set(np.unique(fk)) <= {K.UNKNOWN, K.LOWER}
    assert fk[1, 4] == K.LOWER and fv[1, 4] == pytest.approx(12.0, abs=0.3)      # 3 s at 4 m/s
    assert fk[1, 0] == K.UNKNOWN                                                   # 40 deg off the path
    _, _, fv2, fk2 = U.tube_labels(t, pos, 1.0, pos[100], q, stop_time=2.0)      # contact 1 s later
    expect = 4.0 * (1.0 - U.TUBE_STOP_MARGIN_S) + U.TUBE_FAN_AXIS_M
    assert fk2[1, 4] == K.LOWER and fv2[1, 4] == pytest.approx(expect, abs=0.1)
    gv3, gk3, _, _ = U.tube_labels(t, pos, 4.9, pos[490], q, stop_time=5.0)
    assert (gk3 == K.UNKNOWN).all()                                               # nothing flown yet: no bound


# ----------------------------------------------------------------------------- L3 impacts

def _telemetry_with_contact(terminal: bool):
    t = np.arange(0, 3.0, 0.01)
    v = np.tile([5.0, 0.0, 0.0], (len(t), 1))
    k = 200
    if not terminal:
        v[k:] = [1.0, 0.0, 0.0]            # 4 m/s lost in one tick: an external push along -x
    pos = np.cumsum(v * 0.01, axis=0)
    q = np.tile(LEVEL, (len(t), 1))
    if terminal:                            # the runner stops logging at the impact
        return t[:k + 1], pos[:k + 1], v[:k + 1], q[:k + 1], t[k] + 0.01
    return t, pos, v, q, t[k]


def test_contact_from_telemetry_contact_and_terminal_modes():
    t, p, v, q, tl = _telemetry_with_contact(terminal=False)
    c = I.contact_from_telemetry(t, p, v, q, tl)
    assert c['mode'] == 'contact' and c['normal_w'][0] < -0.9
    assert np.allclose(c['point_w'], np.array(c['drone_pos_w']) - I.CONTACT_OFFSET_M * np.array(c['normal_w']))
    t, p, v, q, tl = _telemetry_with_contact(terminal=True)
    c = I.contact_from_telemetry(t, p, v, q, tl)
    assert c['mode'] == 'terminal' and c['drone_pos_w'][0] == pytest.approx(p[-1, 0] + 0.05, abs=1e-6)
    assert c['point_w'][0] == pytest.approx(c['drone_pos_w'][0] + I.CONTACT_OFFSET_M, abs=1e-6)


def test_impact_labels_exact_when_the_line_of_sight_was_flown():
    q = _quat_axis([0, 1, 0], 30.0)                      # level optical axis
    ev = dict(point_w=[6.0, 0.0, 1.0], normal_w=[-1.0, 0.0, 0.0])
    cam = np.array([1.0, 0.0, 1.0])
    path = np.c_[np.linspace(1, 5.9, 50), np.zeros(50), np.ones(50)]
    gv, gk, fv, fk, sv, sk = I.impact_labels(ev, cam, q, flown_path=path, return_subrays=True)
    assert (sk == K.EXACT).any() and np.nanmin(sv[sk == K.EXACT]) == pytest.approx(5.0, rel=0.02)
    assert (gk == K.UPPER).any() and np.nanmin(gv[gk == K.UPPER]) == pytest.approx(5.0, rel=0.02)   # cell = min
    assert fk[1, 4] == K.UPPER and fv[1, 4] == pytest.approx(5.0, abs=0.05)
    *_, sv, sk = I.impact_labels(ev, cam, q, flown_path=path + [0, 1.0, 0], return_subrays=True)   # flown 1 m aside
    assert (sk == K.UPPER).any() and not (sk == K.EXACT).any()
    behind = I.impact_labels(dict(point_w=[-3.0, 0.0, 1.0], normal_w=None), cam, q)
    assert (behind[1] == K.UNKNOWN).all()                # out of view: no grid label


def test_events_f12_blind_labels_validate():
    evs = load_events(REPO / 'configs' / 'obstacles' / 'events_f12.json')
    assert len(evs) == 8
    assert {e['env'] for e in evs} == {'Minus Two', 'Pine Valley'}
    assert sum(e['env'] == 'Minus Two' for e in evs) == 6
    for e in evs:
        assert e['blind'] and e['source'] == 'blind_label' and e['labelled_at'] and e['labeller']
        assert e['point_w'] is not None and e['unique_obstacle'] and not e['oracle_route']


def test_lateral_manifest_events_and_unique_obstacles():
    man = dict(created='2026-09-24', impacts=[
        dict(run='minus-brain03-01', environment='Minus Two (dark)', impact_phase_s=13.5, impact_pos=[54.5, 3.6, 0.6],
             obstacle='pillar', obstacle_side='centre-left', free_side='right', confidence='high')],
        non_fatal_contacts=[dict(run='pine-fast6-ttc-01', phase_s=16.6, obstacle='tree branches', obstacle_side='left')],
        near_passes=[dict(run='minus-fast6-01', phase_s=[12.2, 13.0], obstacle='pillar', obstacle_side='right')])
    evs = I.lateral_manifest_events(man)
    for e in evs:
        validate_event(e)
    assert [e['kind'] for e in evs] == ['terminal_impact', 'contact', 'near_pass']
    assert evs[0]['primary_free_side'] == 'right' and evs[0]['lateral'] and not evs[0]['blind']
    evs[0]['point_w'] = [54.7, 3.6, 0.6]
    other = dict(evs[0], event_id=9, run='x', unique_obstacle='minus-two/pillar-A', point_w=[54.6, 3.7, 0.7])
    out = I.assign_unique_obstacles([other, evs[0], dict(evs[0], point_w=[80.0, 3.6, 0.6], event_id=10)])
    assert out[1]['unique_obstacle'] == 'minus-two/pillar-A'
    assert out[2]['unique_obstacle'] != 'minus-two/pillar-A'


# ----------------------------------------------------------------------------- corridors

def test_first_blocked_kinds():
    q = _quat_axis([0, 1, 0], 30.0)
    pts = np.array([[5.0, 0.2, 0.0]])
    v, k = R.first_blocked(pts, 3.0, np.zeros(3), q)             # observed free only to 3 m: hit is an upper bound
    assert k[1, 4] == K.UPPER and v[1, 4] == pytest.approx(5.0, abs=0.01)
    v, k = R.first_blocked(pts, 4.8, np.zeros(3), q)             # observed free to (almost) the hit: exact
    assert k[1, 4] == K.EXACT
    v, k = R.first_blocked(np.zeros((0, 3)), 7.0, np.zeros(3), q)
    assert k[1, 4] == K.LOWER and v[1, 4] == pytest.approx(7.0)
    v, k = R.first_blocked(np.zeros((0, 3)), None, np.zeros(3), q)
    assert (k == K.UNKNOWN).all()


# ----------------------------------------------------------------------------- L2 hindsight (synthetic scene)

WALL_X = 12.0


def _texture(u, v, cell=0.35, seed=3):
    i, j = np.floor(u / cell).astype(np.int64), np.floor(v / cell).astype(np.int64)
    h = (i * 73856093) ^ (j * 19349663) ^ seed
    return (np.abs(h) % 200 + 30).astype(np.float64)


def _render(pos, quat):
    """A textured wall at x = WALL_X (y in [-10, 10], z in [0, 12]) and a textured ground z = 0."""
    import cv2
    uv = np.stack(np.meshgrid(np.arange(contract.IMAGE_W) + 0.5, np.arange(contract.IMAGE_H) + 0.5), -1)
    d = R.pixel_dirs_camera(uv) @ contract.camera_to_world(quat).T
    img = np.full(uv.shape[:2], 200.0)
    rng = np.full(uv.shape[:2], np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        sw = (WALL_X - pos[0]) / d[..., 0]
        hw = pos[None, None] + sw[..., None] * d
        okw = (sw > 0) & (np.abs(hw[..., 1]) <= 10) & (hw[..., 2] >= 0) & (hw[..., 2] <= 12)
        sg = -pos[2] / d[..., 2]
        hg = pos[None, None] + sg[..., None] * d
        okg = (sg > 0) & (hg[..., 0] < WALL_X)
    img[okg] = _texture(hg[..., 0], hg[..., 1], seed=11)[okg]
    rng[okg] = sg[okg]
    w = okw & (sw < rng)
    img[w] = _texture(hw[..., 1], hw[..., 2])[w]
    rng[w] = sw[w]
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.8)
    return np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, axis=-1), rng


def _flight(n=30, speed=3.0, rate=10.0):
    t = 1000.0 + np.arange(n) / rate
    pos = np.c_[speed * (t - t[0]), 0.15 * np.sin(t - t[0]), 1.5 + 0.1 * (t - t[0])]
    quat = np.tile(_quat_axis([0, 1, 0], 20.0), (n, 1))
    return t, pos, quat


def _no_mask(rgb):
    return np.zeros(rgb.shape[:2], bool), 'test: no overlays'


def test_hindsight_triangulates_a_wall_and_labels_frames():
    t, pos, quat = _flight()
    frames = np.stack([_render(p, q)[0] for p, q in zip(pos, quat)])
    tracks = H.track_run(frames, t, pos, quat, mask_fn=_no_mask)
    est = H.triangulate(tracks, t, pos, quat)
    assert len(est.frame) > 200
    rel = np.abs(np.linalg.norm(est.point - pos[est.frame], axis=1) - est.range_m) / est.range_m
    assert rel.max() < 1e-6                                       # range is from the reference camera
    vmap = H.fuse(est, pos)
    on_wall = np.abs(vmap.occ_pos[:, 0] - WALL_X) < 0.3
    assert vmap.occ_pos.shape[0] > 50 and on_wall.mean() > 0.6
    k = 10
    uv, rr, pp = H.own_frame_points(est, vmap, k)
    gv, gk, fv, fk, sv, sk = H.hindsight_labels(vmap, pos[k], quat[k], own_uv=uv, own_range=rr, own_points=pp,
                                                return_subrays=True)
    _, true = _render(pos[k], quat[k])
    step = contract.PATCH_PX / R.SUB
    px = R.subray_pixels().reshape(-1, 2)
    truth = true[np.clip((px[:, 1]).astype(int), 0, 251), np.clip((px[:, 0]).astype(int), 0, 447)].reshape(R.SUB_SHAPE)
    ex = (sk == K.EXACT) & np.isfinite(truth)
    assert ex.sum() > 20 and np.median(np.abs(sv[ex] - truth[ex]) / truth[ex]) < 0.08
    lo = (sk == K.LOWER) & np.isfinite(truth)
    assert lo.sum() > 0 and np.mean(sv[lo] <= truth[lo] * 1.1) > 0.95        # free-space bounds hold
    # the flown path is free: an occupied voxel placed on it (a ghost-trail point) is removed
    extra = H.voxel_key(H.voxel_index(pos[5:6]))
    assert not np.isin(extra, vmap.occ_key).any()
    key = np.r_[vmap.occ_key, extra]
    o = np.argsort(key)
    fake = H.VoxelMap(key[o], np.r_[vmap.occ_pos, pos[5:6]][o], np.r_[vmap.occ_n, 3][o], np.r_[vmap.occ_frames, 3][o],
                      np.r_[vmap.occ_sigma, 0.1][o], vmap.free_key, {})
    assert fake.lookup(pos[5:6])[0] == 2
    cleared = H.clear_flown_path(fake, t, pos)
    assert cleared.lookup(pos[5:6])[0] == 1 and len(cleared.occ_key) < len(fake.occ_key)
    del step


def test_voxel_map_lookup_and_save(tmp_path):
    rng = np.random.default_rng(2)
    est = H.Estimates(np.repeat(np.arange(4, dtype=np.int32), 10), np.arange(40, dtype=np.int64),
                      rng.uniform(0, 400, (40, 2)).astype(np.float32),
                      np.tile(rng.uniform(-5, 5, (10, 3)) + [10, 0, 0], (4, 1)), np.full(40, 10.0), np.full(40, 0.3),
                      np.zeros(40, np.float32))
    cams = np.zeros((4, 3))
    vm = H.fuse(est, cams)
    assert len(vm.occ_key) == 10 and (vm.occ_frames == 4).all()
    assert (vm.lookup(vm.occ_pos) == 2).all()
    mid = 0.5 * vm.occ_pos[0]
    assert vm.lookup(mid[None])[0] == 1                                        # carved line of sight
    vm.save(tmp_path / 'm.npz')
    vm2 = H.VoxelMap.load(tmp_path / 'm.npz')
    assert np.array_equal(vm2.occ_key, vm.occ_key) and (vm2.lookup(vm.occ_pos) == 2).all()


# ----------------------------------------------------------------------------- combination and build

def test_combine_sources_order_and_conflicts():
    v = lambda *a: np.array(a, float)
    k = lambda *a: np.array(a, np.uint8)
    parts = {LabelSource.COLLIDER: (v(5, 5, np.nan), k(K.EXACT, K.EXACT, K.UNKNOWN)),
             LabelSource.HINDSIGHT: (v(5.2, 2.0, 7.0), k(K.UPPER, K.UPPER, K.UPPER)),
             LabelSource.TUBE: (v(1.5, 1.0, 3.0), k(K.LOWER, K.LOWER, K.LOWER))}
    val, kind, src, conflict = B.combine_sources(parts)
    assert kind.tolist() == [K.EXACT, K.UNKNOWN, K.LOWER] and conflict.tolist() == [False, True, False]
    assert src[0] & int(LabelSource.COLLIDER) and src[0] & int(LabelSource.TUBE) and src[1] == 0
    assert val[0] == pytest.approx(5.0) and val[2] == pytest.approx(3.0)


def _synthetic_store(root):
    t, pos, quat = _flight(n=30)
    w = S.StoreWriter(root)
    w.write_plan([dict(source_id='synthetic/run', flight='synthetic/run', aliases=['run'], source='run_video',
                       env='Straw Bale', origin_sim=None, telemetry_csv=None)], {})
    rows = S.empty_index(len(t))
    rows['env'] = 0
    rows['source'] = int(S.Source.RUN_VIDEO)
    rows['grade'] = int(S.Grade.GOOD)
    rows['t_wall'] = t
    rows['t_phase'] = t - t[0]
    rows['pos'] = pos
    rows['quat'] = quat
    with w.begin_run(0) as rw:
        rw.append(np.stack([_render(p, q)[0] for p, q in zip(pos, quat)]), rows)
        rw.close()
    w.write_json('events.json', dict(events=[]))
    return w.finalize()


def test_label_build_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(H, 'feature_mask', _no_mask)
    _synthetic_store(tmp_path)
    args = argparse.Namespace(store=tmp_path, colliders=None, events=None, extra_events=None, stages=None,
                              flight_lock=str(tmp_path / 'NO_FLIGHT_LOCK'), runs=None)
    m = B.build(args)
    assert m['status'] == 'complete' and m['n_frames'] == 30
    from haltere.obstacles.labels import LabelSet
    ls = LabelSet(tmp_path, S.FrameStore(tmp_path).index_sha256())
    gv, gk, gs = ls.grid(np.arange(30))
    assert (gk != K.UNKNOWN).mean() > 0.05
    assert (gs[gk != K.UNKNOWN] != 0).all()
    assert set(np.unique(gs[gk != K.UNKNOWN])) <= {int(LabelSource.HINDSIGHT), int(LabelSource.TUBE),
                                                   int(LabelSource.HINDSIGHT | LabelSource.TUBE)}
    assert 'l2_vs_colliders' in m['quality'] and (tmp_path / 'labels' / 'events.json').exists()
    # resumable: a second run does no per-run work and yields identical arrays
    before = np.array(ls.arrays['grid_value'])
    B.build(args)
    after = LabelSet(tmp_path, S.FrameStore(tmp_path).index_sha256()).arrays['grid_value']
    assert np.array_equal(np.nan_to_num(before, nan=-1), np.nan_to_num(np.array(after), nan=-1))


def test_teacher_block_average_and_skip(tmp_path, monkeypatch):
    d = np.arange(252 * 448, dtype=np.float32).reshape(252, 448)
    b = T.block_average(d)
    assert b.shape == (36, 64) and b[0, 0] == pytest.approx(d[:7, :7].mean())
    monkeypatch.setattr(T, 'find_model_dir', lambda: None)
    m = T.build_teacher_cache(store=None, labels_root=tmp_path, guard=None, log=lambda *a: None)
    assert m['status'] == 'skipped' and 'no download' in m['reason']
    assert json.loads((tmp_path / 'teacher' / 'manifest.json').read_text())['status'] == 'skipped'
