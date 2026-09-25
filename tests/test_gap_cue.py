"""Gap cue (haltere.vision.gap_cue) on synthetic relative-disparity maps, plus its configs and the depth wrapper."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from haltere.vision import gap_cue as gc
from haltere.vision.camera import quat_wxyz_to_mat

CAM = gc.DEFAULT_CAMERA
SHAPE = (36, 64)
P = replace(gc.GapCueParams(), enabled=True)
BRAIN = gc.ResponseModel('brain08', 0.55, 6.5)


def quat_yaw_pitch(yaw_deg, pitch_down_deg):
    """World-from-body wxyz for a yaw about z then a nose-down pitch about body y."""
    y, p = math.radians(yaw_deg) / 2, math.radians(pitch_down_deg) / 2
    qz = np.array([math.cos(y), 0, 0, math.sin(y)])
    qy = np.array([math.cos(p), 0, math.sin(p), 0])      # +rotation about y tips the nose down (FLU)
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qy
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


Q = quat_yaw_pitch(20.0, 15.0)
YAW = 20.0


def block_az_el(q=Q):
    w = gc.block_rays_body(SHAPE, CAM) @ quat_wxyz_to_mat(q).T
    az = np.degrees(np.arctan2(w[:, 1], w[:, 0])).reshape(SHAPE)
    el = np.degrees(np.arcsin(np.clip(w[:, 2], -1, 1))).reshape(SHAPE)
    return az, el


def cue_for(az_world, el=0.0, q=Q):
    """Normalised image position of the world direction (azimuth, elevation) in the camera."""
    a, e = math.radians(az_world), math.radians(el)
    w = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    b = quat_wxyz_to_mat(q).T @ w
    px, ok = CAM.project_body(b[None])
    assert ok[0]
    return np.array([px[0, 0] / CAM.width, px[0, 1] / CAM.height])


RING = YAW + 3.0          # ring bearing (world azimuth), inside the view
CUE = cue_for(RING)


def scene(objects=(), base=1.0, ring=RING, q=Q):
    """Disparity: ``base`` everywhere; each (az_lo, az_hi, value) object (relative to the ring) fills the band
    elevations -10..+15 deg between those azimuths."""
    az, el = block_az_el(q)
    rel = gc.wrap_deg(az - ring)
    d = np.full(SHAPE, base, np.float32)
    for lo, hi, val in objects:
        d[(rel >= lo) & (rel <= hi) & (el > -10) & (el < 15)] = val
    return d


def test_no_near_object_is_clear():
    d = gc.decide(scene(), Q, CUE, params=P)
    assert d.valid and d.kind == 'clear' and d.shift_deg == 0.0
    assert abs(d.ring_bearing_deg - RING) < 0.2


def test_pillar_right_of_ring_shifts_left_with_margin():
    d = gc.decide(scene([(-7.0, -2.0, 3.0)]), Q, CUE, params=P)
    assert d.kind == 'gap'
    m = P.mu0_deg * min(max(3.0 / P.kappa, 1.0), P.margin_ratio_cap)
    assert d.margins_deg[0] == pytest.approx(m, abs=1e-6) and d.margins_deg[1] == 0.0
    assert 0 < d.shift_deg <= P.shift_clip_deg
    assert d.shift_deg == pytest.approx(d.interval_deg[0] + m, abs=1e-6)
    assert d.shift_deg > -2.0 + m - 1.0          # the aim keeps the margin from the pillar's left edge


def test_pillar_left_of_ring_shifts_right():
    d = gc.decide(scene([(2.0, 7.0, 3.0)]), Q, CUE, params=P)
    assert d.kind == 'gap' and d.shift_deg < 0


def test_far_side_object_does_not_move_the_aim():
    d = gc.decide(scene([(18.0, 24.0, 3.0)]), Q, CUE, params=P)
    assert d.kind == 'gap' and d.shift_deg == 0.0


def test_centred_gate_keeps_the_aim_on_the_ring():
    d = gc.decide(scene([(-12.0, -8.0, 3.0), (8.0, 12.0, 3.0)]), Q, CUE, params=P)
    assert d.kind == 'gap'
    assert abs(d.shift_deg) <= 1.0


def test_occluded_ring_moves_to_the_nearest_free_side():
    d = gc.decide(scene([(-3.0, 12.0, 3.0)]), Q, CUE, params=P)
    assert d.kind == 'occluded'
    assert d.shift_deg == -P.shift_clip_deg          # right side is nearer; clipped to 12 deg
    assert d.r_ring > P.kappa


def test_blocked_when_no_free_column_in_the_search_window():
    d = gc.decide(scene([(-26.0, 26.0, 3.0)], base=1.0), Q, CUE, params=replace(P, background_halfwidth_deg=60.0))
    # background over +-60 is mostly the free flanks, so +-26 reads near and nothing within +-25 is free
    assert d.kind == 'blocked' and d.shift_deg == 0.0


def test_all_terrain_band_is_not_an_obstacle():
    d = gc.decide(scene(base=5.0), Q, CUE, params=P)
    assert d.kind == 'clear' and d.shift_deg == 0.0
    az, el = block_az_el()
    slope = (1.0 + 0.01 * gc.wrap_deg(az - RING)).astype(np.float32)    # left side nearer (higher disparity)
    d = gc.decide(slope, Q, CUE, params=P)
    assert d.kind == 'clear' and d.shift_deg == 0.0 and d.lr > 0


def test_hud_only_artefact_never_makes_a_near_column():
    d_img = scene()
    hud_px = np.zeros((252, 448), bool)
    az, el = block_az_el()
    rel = gc.wrap_deg(az - RING)
    blocks = (np.abs(rel + 5) < 3) & (el > -4) & (el < 6)         # a HUD glyph right of the ring, in the band
    assert blocks.any()
    hud_px[np.repeat(np.repeat(blocks, 7, 0), 7, 1)] = True
    d_img[blocks] = 10.0                                           # the depth model reads the glyph as near
    raw = gc.decide(d_img, Q, CUE, params=P)
    assert raw.kind == 'gap' and raw.shift_deg != 0.0              # unmasked: a false near column
    valid = gc.block_validity(hud_px, SHAPE, P.block_mask_max_fraction)
    assert not valid[blocks].any()
    d = gc.decide(d_img, Q, CUE, valid=valid, params=P)
    assert d.kind == 'clear' and d.shift_deg == 0.0


def test_masked_ring_column_is_unknown_not_an_obstacle():
    az, el = block_az_el()
    valid = ~(np.abs(gc.wrap_deg(az - RING)) < 3)
    d = gc.decide(scene(), Q, CUE, valid=valid, params=P)
    assert d.kind == 'clear' and d.shift_deg == 0.0


def test_missing_or_edge_clamped_ring_cue():
    assert gc.decide(scene(), Q, None, params=P).kind == 'no_ring'
    assert gc.decide(scene(), Q, np.array([np.nan, 0.5]), params=P).kind == 'no_ring'
    assert gc.decide(scene(), Q, np.array([0.01, 0.5]), params=P).kind == 'no_ring'


def test_band_outside_the_view_is_invalid():
    q_up = quat_yaw_pitch(YAW, -40.0)                               # nose far up: the horizon band is below view
    d = gc.decide(scene(q=q_up), q_up, cue_for(RING, 60.0, q_up), params=P)
    assert not d.valid and d.kind == 'invalid'


def velocity(course_world, speed=6.0):
    a = math.radians(course_world)
    return np.array([speed * math.cos(a), speed * math.sin(a), 0.0])


def test_near_on_path_with_a_lagged_course():
    d_img = scene([(-12.0, -8.0, 3.0)])
    on_line = gc.decide(d_img, Q, CUE, velocity=velocity(RING), params=P, response=BRAIN)
    lagged = gc.decide(d_img, Q, CUE, velocity=velocity(RING - 15.0), params=P, response=BRAIN)
    assert on_line.near_on_path is False and lagged.near_on_path is True
    assert on_line.shift_deg == lagged.shift_deg                  # the flag does not change the aim
    assert lagged.path_deg[0] <= -14.0 and lagged.course_rel_deg == pytest.approx(-15.0, abs=1e-6)
    pd = gc.ResponseModel('fast_pd', 0.30, 8.0)
    fast = gc.decide(d_img, Q, CUE, velocity=velocity(RING - 15.0), params=P, response=pd)
    assert fast.path_deg[1] > lagged.path_deg[1]                  # the PD turns towards the aim sooner
    slow = gc.decide(d_img, Q, CUE, velocity=velocity(RING - 15.0, 0.5), params=P, response=BRAIN)
    assert slow.near_on_path is False and slow.path_deg is None


def test_predicted_path_turns_after_the_delay():
    lo, hi = gc.predicted_path(-20.0, 6.0, 0.0, BRAIN, 0.5)
    assert lo == pytest.approx(-20.0, abs=1e-6) and hi == pytest.approx(-20.0, abs=1e-6)
    lo, hi = gc.predicted_path(-20.0, 6.0, 0.0, BRAIN, 1.5)
    assert lo == pytest.approx(-20.0, abs=1e-6) and -20.0 < hi <= 0.0


def test_confirmation_release_and_repeated_frames():
    cue = gc.GapCue(P, response=BRAIN)
    pil = scene([(-7.0, -2.0, 3.0)])
    o = cue.update(pil, Q, CUE, 0.00, 0.05)
    assert not o.confirmed and o.shift_deg == 0.0 and o.kind == 'gap'
    o = cue.update(pil, Q, CUE, 0.00, 0.10)                        # the same frame again: no new evidence
    assert not o.confirmed
    o = cue.update(pil, Q, CUE, 0.06, 0.11)
    assert o.confirmed and o.shift_deg > 0 and o.episode == 1
    clear = scene()
    o = cue.update(clear, Q, CUE, 0.12, 0.17)
    assert o.confirmed and o.shift_deg > 0                          # held for one clear frame
    o = cue.update(clear, Q, CUE, 0.18, 0.23)
    assert not o.confirmed and o.shift_deg == 0.0
    cue.update(pil, Q, CUE, 0.24, 0.29)
    o = cue.update(pil, Q, CUE, 0.30, 0.35)
    assert o.confirmed and o.episode == 2
    o = cue.update(scene([(2.0, 7.0, 3.0)]), Q, CUE, 0.36, 0.41)    # side change releases at once
    assert not o.confirmed and o.shift_deg == 0.0
    o = cue.update(pil, Q, None, 0.42, 0.47)                        # ring lost: invalid, released
    assert o.kind == 'no_ring' and not o.confirmed


def test_stale_frames_and_gaps():
    cue = gc.GapCue(P, response=BRAIN)
    pil = scene([(-7.0, -2.0, 3.0)])
    cue.update(pil, Q, CUE, 1.00, 1.05)
    assert cue.update(pil, Q, CUE, 1.06, 1.11).confirmed
    o = cue.update(pil, Q, CUE, 1.06, 1.06 + P.max_age_s + 0.01)     # nothing newer arrived: too old
    assert o.kind == 'stale' and not o.confirmed and o.shift_deg == 0.0
    o = cue.update(pil, Q, CUE, 1.10, 1.10 + P.max_age_s + 0.2)       # a new frame, but delivered late
    assert o.kind == 'stale' and o.shift_deg == 0.0
    cue.update(pil, Q, CUE, 2.00, 2.05)
    o = cue.update(pil, Q, CUE, 2.00 + P.max_gap_s + 0.05, 2.40)     # a long gap restarts confirmation
    assert not o.confirmed
    assert cue.update(pil, Q, CUE, 2.45, 2.50).confirmed


def test_disabled_by_default():
    params, raw, sha = gc.load_config()
    assert params.enabled is False and raw['params']['enabled'] is False
    assert len(sha) == 64
    cue = gc.GapCue(params)
    o = cue.update(scene([(-7.0, -2.0, 3.0)]), Q, CUE, 0.0, 0.0)
    assert o.kind == 'off' and o.shift_deg == 0.0 and o.decision is None
    assert gc.from_config().params.enabled is False


def test_repository_configs_match_the_code():
    params, raw, sha = gc.load_config()
    assert gc.config_sha256(raw) == sha
    if raw.get('frozen'):
        assert raw['sha256'] == sha
    assert params.kappa == 1.8 and params.mu0_deg == 5.0 and params.shift_clip_deg == 12.0
    assert tuple(params.band_deg) == (-4.0, 6.0) and params.margin_ratio_cap == 2.5
    models = gc.load_response_models()
    assert models['fast_pd'].delay_s == pytest.approx(0.30) and models['fast_pd'].a_lat_mps2 == pytest.approx(8.0)
    assert models['brain08'].delay_s == pytest.approx(0.55) and models['brain08'].a_lat_mps2 == pytest.approx(6.5)
    assert params.motor in models
    with pytest.raises(ValueError):
        gc.GapCueParams.from_dict({'kappa': 2.0, 'not_a_parameter': 1})


def test_block_validity_needs_a_tiling_grid():
    with pytest.raises(ValueError):
        gc.block_validity(np.zeros((252, 448), bool), (35, 64))
    m = np.zeros((252, 448), bool)
    m[:7, :7] = True
    v = gc.block_validity(m, SHAPE, 0.3)
    assert not v[0, 0] and v.sum() == v.size - 1


def test_relative_depth_wrapper_cpu_smoke():
    from haltere.vision import relative_depth as rd
    if rd.find_model_dir() is None:
        pytest.skip('DA-V2-Small relative weights are not on this machine')
    try:
        model = rd.RelativeDepth(device='cpu')
    except ImportError:
        pytest.skip('transformers is not importable')
    frame = np.zeros((252, 448, 3), np.uint8)
    frame[:126] = (120, 170, 230)          # sky
    frame[126:] = (90, 110, 60)            # ground
    frame[60:200, 200:240] = (40, 30, 20)  # a dark trunk
    out = model(frame)
    assert out.shape == (1, 36, 64) and np.isfinite(out).all()
    assert model.provenance['weights_sha256'] == rd.EXPECTED_WEIGHTS_SHA256
    assert model.provenance['precision'] == 'fp32' and model.provenance['input_hw'] == [252, 448]
    assert out[0, 25:28, 30:33].mean() > out[0, 2:5, 2:5].mean()   # trunk nearer than the sky
