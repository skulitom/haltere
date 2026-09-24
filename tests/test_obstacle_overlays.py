"""Overlay masks for the obstacle model input (haltere.obstacles.overlays): contract, detectors, budget.

Synthetic frames are drawn at the 1280 x 720 gameplay resolution and area-resized to 448 x 252, like the
store frames. The real-frame validation (hand-checked ground truth across environments) is offline.
"""
import ast
import time
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from haltere.obstacles import overlays as ov  # noqa: E402

W, H = ov.FRAME_W, ov.FRAME_H


def down(img1280):
    return cv2.resize(img1280, (W, H), interpolation=cv2.INTER_AREA)


def textured(seed=0, base=(90, 80, 60), amp=25):
    """A 1280x720 textured, unsaturated scene without HUD."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, amp, (90, 160, 1))
    noise = cv2.resize(noise.astype(np.float32), (1280, 720), interpolation=cv2.INTER_CUBIC)[..., None]
    img = np.clip(np.array(base, np.float32) + noise, 0, 255)
    return img.astype(np.uint8)


def to448(img):
    return down(np.ascontiguousarray(img))


def test_assets_present_and_sized():
    for name in (ov.STATIC_HUD_ASSET, ov.PROPELLER_ASSET) + ov.SURVEY_ASSETS:
        img = cv2.imread(str(ov.ASSET_DIR / name), cv2.IMREAD_GRAYSCALE)
        assert img is not None, name
        assert img.shape == (720, 1280), name
        assert set(np.unique(img)) <= {0, 255}, name
    lay = ov.layout()
    assert lay['static'].shape == (H, W) and lay['static'].dtype == bool
    # the fixed glyph layer is a small, pixel-level mask, not boxes
    assert 0.005 < lay['static'].mean() < 0.03
    # propeller wedges: lower image sides only, much tighter than the survey rectangles (24 %)
    prop = lay['propeller']
    assert 0.03 < prop.mean() < 0.15
    assert not prop[:100].any()
    assert not prop[:, 150:300].any()
    # the travel/centre-line column is not covered by static glyphs
    x0, y0, x1, y1 = ov._box((616, 400, 664, 720))
    assert lay['static'][y0:y1, x0:x1].mean() < 0.02


def test_contract_shapes_validity_and_masked_input():
    rgb = to448(textured(1))
    m = ov.overlay_masks(rgb)
    for a in (m.hud, m.ring, m.ghost, m.propeller):
        assert a.shape == (H, W) and a.dtype == bool
    v = ov.validity_channel(m)
    assert v.dtype == np.float32 and v.shape == (H, W)
    assert set(np.unique(v)) <= {ov.VALID_SCENE, ov.VALID_PROPELLER, ov.VALID_MASKED}
    assert np.all(v[m.masked] == ov.VALID_MASKED)
    assert np.all(v[m.propeller & ~m.masked] == ov.VALID_PROPELLER)
    out = ov.masked_input(rgb, m)
    assert np.all(out[m.masked] == np.array(ov.IMAGENET_MEAN_U8, np.uint8))
    assert np.array_equal(out[~m.masked], rgb[~m.masked])
    with pytest.raises(ValueError):
        ov.overlay_masks(np.zeros((360, 640, 3), np.uint8))


@pytest.mark.parametrize('seed,base', [(2, (90, 80, 60)), (3, (30, 30, 35)), (4, (170, 150, 110))])
def test_plain_scene_is_kept(seed, base):
    """A scene without overlays loses only the fixed glyph layer (+1 px) and keeps the travel column."""
    rgb = to448(textured(seed, base))
    m = ov.overlay_masks(rgb)
    lay = ov.layout()
    extra = m.masked & ~cv2.dilate(lay['static'].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert extra.mean() < 0.005
    x0, y0, x1, y1 = ov._box((616, 400, 664, 720))
    assert (~m.masked[y0:y1, x0:x1]).mean() > 0.9


def test_hud_text_digits_detected_on_dark_and_mid_backgrounds():
    for base in ((20, 20, 25), (90, 110, 150)):
        img = textured(5, base, amp=8)
        cv2.putText(img, '01:23:456', (575, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, '42', (1190, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        glyph = down((np.abs(img.astype(int) - textured(5, base, amp=8).astype(int)).max(2) > 60)
                     .astype(np.float32)) >= 0.25
        m = ov.overlay_masks(to448(img))
        assert (m.hud & glyph).sum() >= 0.95 * glyph.sum(), base


def test_ranking_icons_masked_as_squares_when_row_text_present():
    img = textured(6, (25, 25, 30), amp=6)
    yc = int(ov.RANK_ROW_Y0 + ov.RANK_ROW_DY)
    cv2.putText(img, 'Falcon - Tour 1', (1110, yc + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(img, (1241, yc - 13), (1267, yc + 13), (40, 120, 230), 3)
    m = ov.overlay_masks(to448(img))
    x0, y0, x1, y1 = ov._box((1244, yc - 10, 1264, yc + 10))
    assert m.hud[y0:y1, x0:x1].all()
    # an empty row keeps its icon slot
    yc2 = int(ov.RANK_ROW_Y0 + 5 * ov.RANK_ROW_DY)
    x0, y0, x1, y1 = ov._box((1244, yc2 - 10, 1264, yc2 + 10))
    assert not m.hud[y0:y1, x0:x1].any()


def annulus(img, centre, r_out=10, r_in=5):
    yy, xx = np.mgrid[:720, :1280]
    d = np.hypot(xx - centre[0], yy - centre[1])
    img[(d <= r_out) & (d >= r_in)] = 255
    return img


@pytest.mark.parametrize('centre', [(300, 500), (900, 250), (1270, 12), (640, 570)])
def test_white_checkpoint_ring_detected_anywhere(centre):
    img = annulus(textured(7, (80, 70, 60)), centre)
    m = ov.overlay_masks(to448(img))
    truth = down((np.hypot(*np.meshgrid(np.arange(1280) - centre[0], np.arange(720) - centre[1])) <= 10)
                 .astype(np.float32)) >= 0.25
    assert (m.masked & truth).sum() >= 0.95 * truth.sum()
    assert any(abs(u * 1280 - centre[0]) < 8 and abs(v * 720 - centre[1]) < 8 for u, v in m.rings_uv) or \
        ov.layout()['static'][int(centre[1] * H / 720), int(centre[0] * W / 1280)]


def test_mint_checkpoint_disc_and_cyan_stripes_masked_not_the_glow():
    img = textured(8, (120, 100, 70))
    cv2.ellipse(img, (300, 450), (60, 25), 0, 0, 360, (115, 231, 189), -1)
    for y in range(300, 400, 8):
        cv2.line(img, (800, y), (1000, y), (40, 220, 220), 2)
    img[600:640, 700:1100] = (200, 255, 255)          # near-white cyan horizon glow (scene)
    m = ov.overlay_masks(to448(img))
    x0, y0, x1, y1 = ov._box((260, 440, 340, 460))
    assert m.ring[y0:y1, x0:x1].all()
    x0, y0, x1, y1 = ov._box((820, 305, 980, 395))
    assert m.ring[y0:y1, x0:x1].mean() > 0.9
    x0, y0, x1, y1 = ov._box((720, 605, 1080, 635))
    assert not m.ring[y0:y1, x0:x1].any()


@pytest.mark.parametrize('colour', [(240, 120, 170), (240, 140, 40), (60, 200, 60), (60, 120, 240), (150, 70, 230)])
def test_ghost_trail_detected(colour):
    base = textured(9, (150, 125, 80), amp=18)     # straw-like ground
    img = base.copy()
    pts = np.array([[100, 650], [400, 560], [700, 520], [1000, 540], [1250, 600]], np.int32)
    cv2.polylines(img, [pts], False, colour, 5, cv2.LINE_AA)
    truth = down((np.abs(img.astype(int) - base.astype(int)).max(2) > 40).astype(np.float32)) >= 0.25
    truth &= ~ov.layout()['static']
    m = ov.overlay_masks(to448(img))
    assert (m.masked & truth).sum() >= 0.9 * truth.sum(), colour
    assert (m.ghost & truth).sum() >= 0.8 * truth.sum(), colour


def test_saturated_blob_is_not_a_trail():
    img = textured(10, (90, 80, 60))
    cv2.circle(img, (400, 400), 60, (220, 40, 40), -1)      # a red scene object, not thin
    m = ov.overlay_masks(to448(img))
    x0, y0, x1, y1 = ov._box((360, 360, 440, 440))
    assert not m.ghost[y0:y1, x0:x1].any()


def test_to_model_frame_crops_composite_and_resizes():
    comp = np.zeros((720, 1928, 3), np.uint8)
    comp[:, 648:] = textured(11)
    out = ov.to_model_frame(comp)
    assert out.shape == (H, W, 3)
    assert np.array_equal(out, down(np.ascontiguousarray(comp[:, 648:])))
    small = cv2.resize(textured(11), (640, 360), interpolation=cv2.INTER_AREA)
    assert ov.to_model_frame(small).shape == (H, W, 3)
    with pytest.raises(ValueError):
        ov.to_model_frame(np.zeros((480, 640, 3), np.uint8))


def test_budget_two_ms_per_frame():
    cv2.setNumThreads(2)
    img = textured(12, (120, 100, 70))
    annulus(img, (640, 560))
    cv2.polylines(img, [np.array([[100, 650], [700, 520], [1250, 600]], np.int32)], False, (240, 120, 170), 5)
    cv2.putText(img, '01:23:456', (575, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    rgb = to448(img)
    ov.overlay_masks(rgb)
    medians = []
    for _ in range(3):
        ts = []
        for _ in range(40):
            t0 = time.perf_counter()
            ov.overlay_masks(rgb)
            ts.append(time.perf_counter() - t0)
        medians.append(float(np.median(ts)))
    assert min(medians) < 0.002, medians


def test_runtime_module_imports_only_numpy_and_cv2():
    src = Path(ov.__file__).read_text(encoding='utf-8')
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add(node.module.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.level:
            mods.add('.' + (node.module or ''))
    assert mods <= {'__future__', 'dataclasses', 'functools', 'pathlib', 'numpy', 'cv2'}, mods
