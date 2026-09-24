import json

import numpy as np
import pytest

from haltere.obstacles import contract, labels, model, store, train
from haltere.obstacles.labels import LabelKind as K
from haltere.obstacles.splits import ENV_CODE, SealedAccessError


def _quat_axis(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.deg2rad(deg) / 2
    return np.r_[np.cos(h), np.sin(h) * axis]


# ----------------------------------------------------------------------------- contract geometry

def test_store_camera_and_grid_geometry():
    cam = contract.store_camera()
    assert (cam.width, cam.height, cam.f, cam.tilt_deg) == (448, 252, 140.0, 30.0)
    assert abs(cam.hfov_deg - 116.0) < 0.05
    assert contract.GRID_SHAPE == (18, 32) and contract.PATCH_PX * 32 == 448 and contract.PATCH_PX * 18 == 252
    rays = contract.cell_rays_body()
    assert rays.shape == (18, 32, 3)
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1)
    px, ok = cam.project_body(rays.reshape(-1, 3))
    assert ok.all()
    assert np.allclose(px.reshape(18, 32, 2), contract.cell_centres_px(), atol=1e-6)
    assert np.allclose(contract.cell_centres_px()[0, 0], [7, 7])


def test_fan_layout_signs_and_mirror():
    assert contract.FAN_SHAPE == (4, 9) and contract.FAN_N == 36
    assert contract.FAN_YAW_DEG.tolist() == [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    assert contract.FAN_ELEV_DEG.tolist() == [-10, 0, 10, 20]
    d = contract.fan_directions_world(np.array([1.0, 0, 0, 0]))
    assert d.shape == (4, 9, 3)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1)
    assert np.allclose(d[1, 4], [1, 0, 0])                 # level, straight ahead
    assert d[1, 5, 1] > 0 and d[1, 3, 1] < 0               # positive yaw = LEFT (+y)
    assert d[3, 4, 2] > 0 and d[0, 4, 2] < 0               # positive elevation = up
    assert np.allclose(d[:, ::-1] * [1, -1, 1], d)         # reversing the yaw axis = yaw sign flip
    grid = np.arange(36).reshape(4, 9)
    assert (contract.mirror_fan(grid)[:, 0] == grid[:, 8]).all()
    # Level attitude: straight ahead is 30 deg below the optical axis (v = 126 + 140 tan 30).
    uv, ok = contract.project_camera(contract.fan_directions_camera(np.array([1.0, 0, 0, 0]))[1, 4])
    assert ok and np.allclose(uv, [224, 126 + 140 * np.tan(np.deg2rad(30))])


def test_heading_frame_is_gravity_levelled():
    yaw = _quat_axis([0, 0, 1], 37)
    roll_pitch = _quat_axis([1, 0.3, 0], 25)
    q = np.array([yaw, roll_pitch])
    H = contract.heading_frame(q)
    assert H.shape == (2, 3, 3)
    assert np.allclose(H[..., :, 2], [0, 0, 1])
    assert np.allclose(H[..., 2, 0], 0) and np.allclose(H[..., 2, 1], 0)
    assert np.allclose(H[0, :, 0], [np.cos(np.deg2rad(37)), np.sin(np.deg2rad(37)), 0])
    for h in H:
        assert np.allclose(h.T @ h, np.eye(3)) and np.isclose(np.linalg.det(h), 1)
    # Nose up 60 deg: the optical axis is vertical, the heading falls back to body x.
    up60 = _quat_axis([0, 1, 0], -60)     # rotation about +y by -60 deg raises body x (FLU)
    assert np.allclose(contract.heading_frame(up60)[:, 0], [1, 0, 0], atol=1e-6)


def test_gravity_camera():
    g = contract.gravity_camera(np.array([1.0, 0, 0, 0]))
    assert np.allclose(g, [0, np.cos(np.deg2rad(30)), -0.5])
    g = contract.gravity_camera(np.stack([_quat_axis([1, 0, 0], 20), _quat_axis([0, 0, 1], 90)]))
    assert g.shape == (2, 3) and np.allclose(np.linalg.norm(g, axis=-1), 1)
    assert np.allclose(g[1], [0, np.cos(np.deg2rad(30)), -0.5])   # pure yaw does not change gravity


# ----------------------------------------------------------------------------- store

def test_index_dtype_contract():
    names = ['run_id', 'slot', 'env', 'source', 'grade', 'pose_method', 'flags', 'cue_src', 'luma', 't_wall',
             't_phase', 't_game', 'pose_lag_s', 'align_offset_s', 'timing_delta_s', 'pos', 'quat', 'vel', 'omega',
             'cue_uv', 'tti_s', 'event_id']
    assert list(store.INDEX_DTYPE.names) == names
    assert store.INDEX_DTYPE.itemsize == 120
    assert store.INDEX_DTYPE['t_wall'] == np.dtype('<f8')
    assert store.INDEX_DTYPE['quat'].shape == (4,)
    assert store.FRAME_BYTES == 252 * 448 * 3
    e = store.empty_index(3)
    assert np.isinf(e['tti_s']).all() and (e['event_id'] == -1).all() and np.isnan(e['cue_uv']).all()
    assert np.allclose(e['quat'], [1, 0, 0, 0])


def _records():
    return [dict(source_id='minus/run-a', flight='minus/run-a', source='run_video', env='Minus Two',
                 origin_sim=[10.0, 20.0, 1.0]),
            dict(source_id='vision:straw_1', flight='straw/run-b', source='capture_dataset', env='Straw Bale',
                 origin_sim=None)]


def _rows(n, env, grade=store.Grade.GOOD, flags=0):
    r = store.empty_index(n)
    r['env'] = ENV_CODE[env]
    r['grade'] = grade
    r['flags'] = flags
    r['t_wall'] = 1.79e9 + np.arange(n) * 0.06
    r['pos'] = np.arange(n)[:, None] * [1.0, 0, 0]
    return r


def _build(root, rng):
    w = store.StoreWriter(root)
    plan = w.write_plan(_records(), dict(stride=3))
    assert [r['run_id'] for r in plan] == [0, 1]
    frames = rng.integers(0, 256, (5,) + store.FRAME_SHAPE, dtype=np.uint8)
    assert not w.done(0)
    with w.begin_run(0) as rw:
        rw.append(frames[:3], _rows(3, 'Minus Two'))
        rows = _rows(2, 'Minus Two', flags=int(store.Flag.POST_EVENT))
        rw.append(frames[3:], rows)
        rw.close(dropped=dict(repeat=4))
    assert w.done(0)
    with w.begin_run(1) as rw:
        rw.append(frames[:2], _rows(2, 'Straw Bale', grade=store.Grade.UNRELIABLE))
        rw.close()
    w.write_json('events.json', dict(events=[]))
    return w.finalize(), frames


def test_store_writer_reader_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    manifest, frames = _build(tmp_path, rng)
    assert manifest['status'] == 'complete' and manifest['n_frames'] == 7
    s = store.FrameStore(tmp_path)
    assert len(s) == 7 and s.index_sha256() == manifest['index_sha256']
    assert (s.index['run_id'] == [0, 0, 0, 0, 0, 1, 1]).all() and (s.index['slot'] == [0, 1, 2, 3, 4, 0, 1]).all()
    assert np.array_equal(s.frame(4), frames[4])
    assert np.array_equal(s.frames([6, 1]), frames[[1, 1]])
    assert s.run(0)['dropped'] == dict(repeat=4)
    # Defaults drop POST_EVENT and UNRELIABLE rows.
    assert s.rows().tolist() == [0, 1, 2]
    assert s.rows(fold='F12', side='test', grades=None, exclude=0).tolist() == [0, 1, 2, 3, 4]
    assert s.rows(fold='F12', side='inner_train', grades=None).tolist() == [5, 6]
    held = s.rows(fold='ALL', side='inner_val', grades=None, exclude=0)
    kept = s.rows(fold='ALL', side='inner_train', grades=None, exclude=0)
    assert sorted(held.tolist() + kept.tolist()) == list(range(7))
    assert np.allclose(s.absolute_pos([1]), [[11.0, 20.0, 1.0]])
    assert np.isnan(s.absolute_pos([5])).all()
    assert s.gravity_camera([0, 5]).shape == (2, 3)


def test_store_writer_validates_and_resumes(tmp_path):
    rng = np.random.default_rng(1)
    w = store.StoreWriter(tmp_path)
    w.write_plan(_records(), {})
    with pytest.raises(ValueError):
        store.StoreWriter(tmp_path).write_plan(_records()[::-1], {})   # a different plan cannot replace it
    assert len(store.StoreWriter(tmp_path).write_plan(_records(), {})) == 2   # same plan: resume
    rw = w.begin_run(0)
    with pytest.raises(ValueError):
        rw.append(np.zeros((1, 10, 10, 3), np.uint8), _rows(1, 'Minus Two'))
    with pytest.raises(ValueError):
        rw.append(np.zeros((1,) + store.FRAME_SHAPE, np.uint8), _rows(1, 'Straw Bale'))   # wrong env code
    rw.abort()
    assert not (tmp_path / 'frames' / 'r00000.u8.tmp').exists()
    with pytest.raises(RuntimeError):
        w.finalize()                                   # runs not done
    with pytest.raises(RuntimeError):
        store.FrameStore(tmp_path)                     # status is still 'building'
    frames = rng.integers(0, 256, (1,) + store.FRAME_SHAPE, dtype=np.uint8)
    with pytest.raises(KeyError):
        with w.begin_run(1) as bad:
            bad.append(frames, _rows(1, 'Straw Bale'))
            raise KeyError('crash mid-run')
    assert not w.done(1) and not (tmp_path / 'frames' / 'r00001.u8.tmp').exists()


def test_sealed_environments_stay_out_of_the_main_store(tmp_path):
    rec = [dict(source_id='green/run', flight='green/run', source='run_video', env='The Green')]
    with pytest.raises(SealedAccessError):
        store.StoreWriter(tmp_path).write_plan(rec, {})
    with pytest.raises(SealedAccessError):
        store.StoreWriter(tmp_path, part='sealed')
    with pytest.raises(SealedAccessError):
        store.FrameStore(tmp_path, part='sealed')
    store.StoreWriter(tmp_path, part='sealed', sealed_final=True).write_plan(rec, {})
    assert (tmp_path / 'sealed' / 'plan.json').exists()
    other = store.StoreWriter(tmp_path / 'x', part='sealed', sealed_final=True)
    with pytest.raises(SealedAccessError):
        other.write_plan(_records(), {})              # open environments never go into the sealed part


# ----------------------------------------------------------------------------- labels

def test_label_schema_constants():
    assert labels.ARRAYS['grid_value'].shape == (18, 32) and labels.ARRAYS['grid_value'].dtype == 'float16'
    assert labels.ARRAYS['fan_kind'].shape == (4, 9) and labels.ARRAYS['fan_kind'].dtype == 'uint8'
    assert [int(k) for k in K] == [0, 1, 2, 3]
    assert labels.LabelSource.COLLIDER == 16


def test_min_over_reduces_subrays_to_a_cell_constraint():
    v, k = labels.min_over([[5.0, 6.0, 7.0]], [[K.EXACT] * 3])
    assert k[0] == K.EXACT and v[0] == 5.0
    v, k = labels.min_over([[5.0, 3.0]], [[K.EXACT, K.LOWER]])        # min in [3, 5]: wide -> lower 3
    assert (k[0], v[0]) == (K.LOWER, 3.0)
    v, k = labels.min_over([[5.0, 4.8]], [[K.EXACT, K.LOWER]])        # within 10 %: exact at the nearer end
    assert (k[0], v[0]) == (K.EXACT, 4.8)
    v, k = labels.min_over([[5.0, np.nan]], [[K.EXACT, K.UNKNOWN]])   # strict: something may be nearer
    assert (k[0], v[0]) == (K.UPPER, 5.0)
    v, k = labels.min_over([[5.0, np.nan]], [[K.EXACT, K.UNKNOWN]], min_known_frac=0.5)
    assert (k[0], v[0]) == (K.EXACT, 5.0)
    v, k = labels.min_over([[4.0, 9.0]], [[K.LOWER, K.LOWER]])
    assert (k[0], v[0]) == (K.LOWER, 4.0)
    v, k = labels.min_over([[np.nan, np.nan]], [[K.UNKNOWN, K.UNKNOWN]])
    assert k[0] == K.UNKNOWN and np.isnan(v[0])
    v, k = labels.min_over(np.full((2, 3, 9), 7.0), np.full((2, 3, 9), K.EXACT), axis=-1)
    assert v.shape == (2, 3) and (k == K.EXACT).all()


def test_intersect_combines_sources_and_flags_conflicts():
    cases = [((5, K.EXACT, 3, K.LOWER), (5, K.EXACT, False)),
             ((5, K.EXACT, 5.3, K.LOWER), (5, K.EXACT, False)),
             ((5, K.EXACT, 7, K.LOWER), (np.nan, K.UNKNOWN, True)),
             ((5, K.EXACT, 4, K.UPPER), (np.nan, K.UNKNOWN, True)),
             ((3, K.LOWER, 10, K.UPPER), (3, K.LOWER, False)),
             ((4, K.UPPER, 6, K.UPPER), (4, K.UPPER, False)),
             ((np.nan, K.UNKNOWN, 6, K.LOWER), (6, K.LOWER, False)),
             ((5.2, K.EXACT, 5.0, K.EXACT), (5.0, K.EXACT, False))]
    for (v1, k1, v2, k2), (ve, ke, ce) in cases:
        v, k, c = labels.intersect(np.array(v1, float), np.array(k1), np.array(v2, float), np.array(k2))
        assert int(k) == ke and bool(c) == ce, (v1, k1, v2, k2)
        assert (np.isnan(v) and np.isnan(ve)) or np.isclose(v, ve)


def test_fan_blocked_targets_and_clipping():
    v = np.array([3.0, 5.0, 6.0, 2.0, 9.0, np.nan])
    k = np.array([K.EXACT, K.EXACT, K.LOWER, K.LOWER, K.UPPER, K.UNKNOWN])
    t, w = labels.fan_blocked_targets(v, k, 4.0)
    assert t.tolist() == [1, 0, 0, 0, 0, 0] and w.tolist() == [1, 1, 1, 0, 0, 0]
    t, w = labels.fan_blocked_targets(np.array([3.5]), np.array([K.UPPER]), 4.0)
    assert (t[0], w[0]) == (1, 1)
    cv, ck = labels.clip_fan(np.array([30.0, 25.0, 25.0, 0.1]), np.array([K.EXACT, K.LOWER, K.UPPER, K.EXACT]))
    assert ck.tolist() == [K.LOWER, K.LOWER, K.UNKNOWN, K.EXACT]
    assert cv[0] == contract.FAN_MAX_M and np.isnan(cv[2]) and cv[3] == contract.RANGE_MIN_M


def test_label_writer_and_reader(tmp_path):
    w = labels.LabelWriter(tmp_path, 4, 'abc')
    assert np.isnan(w.arrays['grid_value']).all() and (w.arrays['fan_kind'] == K.UNKNOWN).all()
    w.write('grid_value', [1], np.full((1, 18, 32), 7.5))
    w.write('grid_kind', [1], np.full((1, 18, 32), K.EXACT))
    w.mark_done('collider', 0)
    w2 = labels.LabelWriter(tmp_path, 4, 'abc')          # resume
    assert w2.is_done('collider', 0) and not w2.is_done('tube', 0)
    with pytest.raises(ValueError):
        labels.LabelWriter(tmp_path, 4, 'other-index')
    with pytest.raises(RuntimeError):
        labels.LabelSet(tmp_path, 'abc')                 # not complete yet
    m = w2.finalize(sources=dict(COLLIDER=dict(frames_labelled=1)))
    assert m['status'] == 'complete' and set(m['arrays']) == set(labels.ARRAYS)
    ls = labels.LabelSet(tmp_path, 'abc')
    gv, gk, gs = ls.grid([1, 2])
    assert gv.dtype == np.float32 and np.allclose(gv[0], 7.5) and np.isnan(gv[1]).all() and (gk[0] == K.EXACT).all()
    with pytest.raises(ValueError):
        labels.LabelSet(tmp_path, 'stale-index')
    (tmp_path / 'labels' / 'events.json').write_text(json.dumps(dict(events=[])), encoding='utf-8')
    assert ls.events() == []


def test_event_schema():
    e = dict(event_id=1, run='minus-brain03-01', env='Minus Two', kind='terminal_impact', t_phase=13.56,
             obstacle='pillar', unique_obstacle='minus-two/pillar-54.8-3.9', primary_free_side='right',
             lateral=True, source='lateral_manifest', blind=False, point_w=[54.56, 3.62, 0.63])
    assert labels.validate_event(dict(e)) == e
    with pytest.raises(ValueError):
        labels.validate_event(dict(e, kind='crash'))
    with pytest.raises(ValueError):
        labels.validate_event(dict(e, colour='red'))
    with pytest.raises(ValueError):
        labels.validate_event({k: v for k, v in e.items() if k != 'blind'})
    from haltere.obstacles.store import REPO_ROOT
    assert labels.load_events(REPO_ROOT / 'configs' / 'obstacles' / 'events_f12.json') == []


# ----------------------------------------------------------------------------- model and loss references

def test_model_output_contract():
    assert model.INPUT_SHAPE == (4, 252, 448)
    assert model.OUTPUT_SPEC == {'grid_log_range': (2, 18, 32), 'fan_log_free': (2, 4, 9), 'fan_logit': (2, 4, 9)}
    out = {'grid_log_range': np.log(np.full((1, 2, 18, 32), 5.0)), 'fan_log_free': np.zeros((1, 2, 4, 9)),
           'fan_logit': np.zeros((1, 2, 4, 9))}
    p = model.outputs_to_numpy(out)
    assert np.allclose(p['grid_q50'], 5.0) and np.allclose(p['fan_q20'], 1.0) and np.allclose(p['fan_p8'], 0.5)
    assert p['grid_q20'].shape == (1, 18, 32) and p['fan_p4'].dtype == np.float32
    with pytest.raises(NotImplementedError):
        model.build_model('dav2s')


def test_censored_laplace_reference():
    q50 = np.log(5.0)
    q20 = q50 - 0.3 * train.LN_2P5            # b = 0.3
    nll = lambda v, k: float(train.laplace_censored_nll(q20, q50, v, k))
    assert np.isclose(nll(5.0, K.EXACT), np.log(0.6))
    assert nll(5.0, K.EXACT) < nll(7.0, K.EXACT)
    assert nll(1.0, K.LOWER) < 0.01 < nll(8.0, K.LOWER)            # bound far below the prediction: ~free
    assert nll(25.0, K.UPPER) < 0.01 < nll(3.0, K.UPPER)
    assert np.isclose(nll(5.0, K.LOWER), np.log(2)) and np.isclose(nll(5.0, K.UPPER), np.log(2))
    assert nll(np.nan, K.UNKNOWN) == 0 and nll(3.0, K.UNKNOWN) == 0
    assert np.isfinite(nll(1e4, K.LOWER)) and np.isfinite(nll(1e-3, K.UPPER))
    b = train.fan_bce(np.array([0.9, 0.9]), np.array([3.0, 3.0]), np.array([K.EXACT, K.UNKNOWN]), 4.0)
    assert np.isclose(b[0], -np.log(0.9)) and b[1] == 0
