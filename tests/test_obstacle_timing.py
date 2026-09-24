"""Timing refinement: camera conventions, triangulation residual, delta recovery and K0a bookkeeping."""
import numpy as np
import pytest

from haltere.obstacles import contract, timing
from haltere.obstacles.store_build import Telemetry


def _yaw_quat(yaw):
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], -1)


def _synthetic_flight(seconds=6.0):
    t = 2000.0 + np.arange(0, seconds, 0.01)
    rel = t - t[0]
    yaw = 0.6 * np.sin(1.3 * rel) + 0.2 * rel               # changing yaw rate: a clock shift is observable
    pos = np.stack([3.0 * rel, 0.5 * np.sin(0.7 * rel), 1.5 + 0.2 * np.sin(rel)], 1)
    vel = np.gradient(pos, rel, axis=0)
    return Telemetry(t, rel + 10.0, pos, _yaw_quat(yaw), vel, None, rel)


def _project(X, C, q, cam):
    R = contract.camera_to_world(q)
    Xc = np.einsum('ji,nj->ni', R, X - C)
    ok = Xc[:, 2] > 0.5
    u = cam.cx + cam.f * Xc[:, 0] / Xc[:, 2] - 0.5            # OpenCV pixel coordinates
    v = cam.cy + cam.f * Xc[:, 1] / Xc[:, 2] - 0.5
    return np.stack([u, v], 1), ok


def _tracks(tel, cam, delta_true, n_windows=12, window=5, rng=None):
    rng = rng or np.random.default_rng(1)
    frames = tel.wall[0] + 0.5 + np.arange(n_windows * window) / 15.0
    t_frames = frames.reshape(n_windows, window)
    tracks, win_of = [], []
    for w in range(n_windows):
        s = tel.sample(t_frames[w] + delta_true)
        s0 = tel.sample(t_frames[w, :1])
        fwd = contract.camera_to_world(s0['quat'])[0][:, 2]
        X = s0['pos'][0] + fwd * rng.uniform(4, 30, (200, 1)) + rng.normal(0, 4, (200, 3))
        uv = np.stack([_project(X, s['pos'][k], s['quat'][k], cam)[0] for k in range(window)], 1)
        ok = np.stack([_project(X, s['pos'][k], s['quat'][k], cam)[1] for k in range(window)], 1).all(1)
        ok &= ((uv[..., 0] > 0) & (uv[..., 0] < 640) & (uv[..., 1] > 0) & (uv[..., 1] < 360)).all(1)
        tracks.append(uv[ok])
        win_of.append(np.full(ok.sum(), w))
    return np.concatenate(tracks), np.concatenate(win_of), t_frames


def test_camera_640_matches_the_store_camera():
    cam = timing.camera_640()
    store_cam = contract.store_camera()
    assert (cam.width, cam.height, cam.f, cam.tilt_deg) == (640, 360, 200.0, 30.0)
    assert cam.f / cam.width == pytest.approx(store_cam.f / store_cam.width)


def test_rays_and_reprojection_are_consistent_with_the_contract():
    cam = timing.camera_640()
    q = _yaw_quat(np.array([0.3]))
    R = contract.camera_to_world(q)
    X = np.array([[10.0, 2.0, 3.0]])
    uv, ok = _project(X, np.zeros(3), q[0], cam)
    d = timing.rays_world(uv, R, cam)
    assert np.allclose(d[0], X[0] / np.linalg.norm(X[0]), atol=1e-9)
    # the store image convention: the contract projection at 448 px is the same ray scaled by 0.7
    uv448, _ = contract.project_camera(np.einsum('ji,j->i', R[0], X[0]))
    assert np.allclose((uv[0] + 0.5) * 448 / 640, uv448, atol=1e-9)


def test_triangulation_residual_is_zero_at_the_true_clock_and_finds_the_shift():
    tel = _synthetic_flight()
    cam = timing.camera_640()
    tracks, win_of, t_frames = _tracks(tel, cam, delta_true=0.025)
    deltas = np.arange(-0.06, 0.0601, 0.005)
    curve, per_track = timing.score_curve(tracks, win_of, t_frames, np.ones(len(tracks)), tel, deltas, cam)
    k = int(np.argmin(curve))
    assert deltas[k] == pytest.approx(0.025, abs=1e-9)
    assert curve[k] < 1e-3
    assert curve[np.argmin(np.abs(deltas))] > 0.05                 # a 25 ms clock error is visible
    boot = timing.bootstrap_delta(per_track, win_of, np.ones(len(tracks)), deltas, t_frames.shape[0],
                                  np.random.default_rng(0), n=16)
    assert boot < 5.0


def test_scan_accepts_an_interior_minimum_and_rejects_the_edge():
    tel = _synthetic_flight()
    cam = timing.camera_640()
    tracks, win_of, t_frames = _tracks(tel, cam, delta_true=-0.02)
    res = timing._scan(tracks, win_of, t_frames, np.ones(len(tracks)), tel, 60, 5, cam, t_frames.shape[0])
    assert res['accepted'] and res['delta_s'] == pytest.approx(-0.02)
    assert res['residual_px_after'] < res['residual_px_before']
    tracks, win_of, t_frames = _tracks(tel, cam, delta_true=0.09)   # outside the +-60 ms search
    res = timing._scan(tracks, win_of, t_frames, np.ones(len(tracks)), tel, 60, 5, cam, t_frames.shape[0])
    assert not res['accepted'] and res['reason'] == 'minimum at the search edge' and res['delta_s'] == 0.0


def test_track_window_follows_a_known_image_shift():
    import cv2
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (400, 700), dtype=np.uint8), (0, 0), 2.0)
    shifts = [(0, 0), (3, 1), (6, 2), (9, 3), (12, 4)]
    greys = [np.ascontiguousarray(tex[20 - dy:380 - dy, 30 - dx:670 - dx]) for dx, dy in shifts]
    masks = [np.ones((360, 640), bool)] * len(greys)
    T = timing.track_window(greys, masks)
    assert len(T) > 100
    motion = T - T[:, :1]
    assert np.allclose(np.median(motion, axis=0), shifts, atol=0.1)


def test_choose_windows_requires_consecutive_valid_moving_frames():
    t = np.arange(40) / 15.0
    valid = np.ones(40, bool)
    valid[10] = False
    speed = np.full(40, 2.0)
    speed[30:] = 0.0
    starts = timing.choose_windows(t, valid, speed, max_windows=100)
    assert all(not (s <= 10 < s + timing.WINDOW) for s in starts)
    assert all(s + timing.WINDOW <= 32 or speed[s:s + timing.WINDOW].mean() >= 1.0 for s in starts)
    assert len(timing.choose_windows(t, valid, speed, max_windows=3)) == 3


def test_k0a_summary():
    runs = {'1': dict(grade='good', residual_px_after=0.6, residual_px_before=0.7, accepted=True),
            '2': dict(grade='fair', residual_px_after=1.4, residual_px_before=1.4, accepted=False),
            '3': dict(grade='good', residual_px_after=None, residual_px_before=None, accepted=False)}
    k = timing.k0a_summary(runs)
    assert (k['n_runs'], k['n_scored'], k['n_within']) == (3, 2, 1)
    assert k['fraction'] == pytest.approx(1 / 3, abs=1e-3) and not k['passed']


def test_weighted_median():
    assert timing.weighted_median([1, 2, 3], [1, 1, 1]) == 2
    assert timing.weighted_median([1, 2, 10], [0.25, 0.25, 1]) == 10
