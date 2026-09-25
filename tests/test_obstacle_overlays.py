"""Overlay masks for the obstacle model input (haltere.obstacles.overlays): contract, detectors, budget.

Synthetic frames are drawn at the 1280 x 720 gameplay resolution and area-resized to 448 x 252, like the
store frames. The real-frame validation (hand-checked ground truth across environments) is offline; its
numbers are in the overlays commit message.
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
    return cv2.resize(np.ascontiguousarray(img1280), (W, H), interpolation=cv2.INTER_AREA)


def textured(seed=0, base=(90, 80, 60), amp=25):
    """A 1280x720 textured, unsaturated scene without HUD."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, amp, (90, 160, 1))
    noise = cv2.resize(noise.astype(np.float32), (1280, 720), interpolation=cv2.INTER_CUBIC)[..., None]
    img = np.clip(np.array(base, np.float32) + noise, 0, 255)
    return img.astype(np.uint8)


def rank_rows_y(n):
    """Row centres (1280 px) of an n-row ghost-racer ranking list (centred on the screen centre)."""
    return [ov.RANK_CENTRE_Y + ov.RANK_HALF_STEP * m for m in range(-(n - 1), n, 2)]


def with_ranking(img, n, colours=((40, 120, 230), (230, 90, 160), (240, 160, 40), (80, 200, 90))):
    """Draw an n-row ranking list: right-aligned name text ending at x ~1234 and a coloured icon square."""
    for k, yc in enumerate(rank_rows_y(n)):
        yc = int(round(yc))
        cv2.putText(img, f'Racer {k} - Tour 1', (1090, yc + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1,
                    cv2.LINE_AA)
        cv2.rectangle(img, (1241, yc - 12), (1265, yc + 12), colours[k % len(colours)], 3)
    return img


def box448(b):
    x0, y0, x1, y1 = ov._box(b)
    return slice(y0, y1), slice(x0, x1)


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
    assert not prop[:, 160:288].any()                  # the centre band (x 457-823 at 1280) is clear
    # the travel/centre-line column is not covered by static glyphs
    assert lay['static'][box448((616, 400, 664, 720))].mean() < 0.03    # only the stick-box dashes cross it


def test_contract_shapes_validity_and_masked_input():
    rgb = down(with_ranking(textured(1), 3))
    m = ov.overlay_masks(rgb)
    for a in (m.hud, m.ring, m.ghost, m.propeller, m.volume):
        assert a.shape == (H, W) and a.dtype == bool
    assert not (m.volume & ~m.ring).any()
    v = ov.validity_channel(m)
    assert v.dtype == np.float32 and v.shape == (H, W)
    assert set(np.unique(v)) <= {ov.VALID_SCENE, ov.VALID_PROPELLER, ov.VALID_MASKED}
    assert np.all(v[m.masked] == ov.VALID_MASKED)
    assert np.all(v[m.propeller & ~m.masked] == ov.VALID_PROPELLER)
    out = ov.masked_input(rgb, m)
    assert np.all(out[m.masked] == np.array(ov.IMAGENET_MEAN_U8, np.uint8))
    assert np.array_equal(out[~m.masked], rgb[~m.masked])
    assert np.array_equal(ov.masked_input(rgb), out)                  # masks computed when not given
    with pytest.raises(ValueError):
        ov.overlay_masks(np.zeros((360, 640, 3), np.uint8))


@pytest.mark.parametrize('seed,base', [(2, (90, 80, 60)), (3, (30, 30, 35)), (4, (170, 150, 110))])
def test_plain_scene_is_kept(seed, base):
    """A scene without overlays loses only the fixed glyph layer (+1 px) and keeps the travel column."""
    rgb = down(textured(seed, base))
    m = ov.overlay_masks(rgb)
    lay = ov.layout()
    extra = m.masked & ~cv2.dilate(lay['static'].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert extra.mean() < 0.005
    assert (~m.masked[box448((616, 400, 664, 720))]).mean() > 0.9


def test_hud_text_digits_detected_on_dark_and_mid_backgrounds():
    for base in ((20, 20, 25), (90, 110, 150)):
        img = textured(5, base, amp=8)
        cv2.putText(img, '01:23:456', (575, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, '42', (1190, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        glyph = down((np.abs(img.astype(int) - textured(5, base, amp=8).astype(int)).max(2) > 60)
                     .astype(np.float32)) >= 0.25
        m = ov.overlay_masks(down(img))
        assert (m.hud & glyph).sum() >= 0.95 * glyph.sum(), base


@pytest.mark.parametrize('n', [2, 4, 6])
def test_ranking_list_rows_found_and_icons_masked(n):
    img = with_ranking(textured(6, (25, 25, 30), amp=6), n)
    m = ov.overlay_masks(down(img))
    assert m.rank_rows == n
    for yc in rank_rows_y(n):
        assert m.hud[box448((1245, yc - 9, 1261, yc + 9))].all()
    # the outer slots of a longer list stay scene
    for yc in (rank_rows_y(n + 2)[0], rank_rows_y(n + 2)[-1]):
        assert not m.hud[box448((1245, yc - 6, 1261, yc + 6))].any()


def annulus(img, centre, r_out=8.3, r_in=3.4):
    """White donut; the default is the size of Liftoff's next-checkpoint marker at 1280 x 720."""
    yy, xx = np.mgrid[:720, :1280]
    d = np.hypot(xx - centre[0], yy - centre[1])
    img[(d <= r_out) & (d >= r_in)] = 255
    return img


@pytest.mark.parametrize('centre', [(300, 500), (900, 250), (1270, 12), (640, 570)])
def test_white_checkpoint_ring_detected_anywhere(centre):
    img = annulus(textured(7, (80, 70, 60)), centre)
    m = ov.overlay_masks(down(img))
    truth = down((np.hypot(*np.meshgrid(np.arange(1280) - centre[0], np.arange(720) - centre[1])) <= 8.3)
                 .astype(np.float32)) >= 0.25
    assert (m.masked & truth).sum() >= 0.95 * truth.sum()
    assert any(abs(u * 1280 - centre[0]) < 8 and abs(v * 720 - centre[1]) < 8 for u, v in m.rings_uv) or \
        ov.layout()['static'][int(centre[1] * H / 720), int(centre[0] * W / 1280)]


def test_mint_checkpoint_disc_masked_not_the_pale_glow():
    img = textured(8, (120, 100, 70))
    cv2.ellipse(img, (300, 450), (60, 25), 0, 0, 360, (115, 231, 189), -1)
    img[600:640, 700:1100] = (200, 255, 255)          # near-white cyan horizon glow (scene)
    m = ov.overlay_masks(down(img))
    assert m.ring[box448((260, 440, 340, 460))].all()
    assert not m.ring[box448((720, 605, 1080, 635))].any()


def scan_lines(img, x0, y0, x1, y1, colour, alpha=0.6, step=9, slope=0.0):
    """Checkpoint-volume scan lines (1 px, every ``step`` px at 1280) blended over the scene; slope != 0 draws
    a diagonal lattice instead."""
    over = img.copy()
    for k in range(-40, 80):
        ya = y0 + k * step
        cv2.line(over, (x0, ya), (x1, int(round(ya + slope * (x1 - x0)))), colour, 1, cv2.LINE_AA)
    region = np.zeros(img.shape[:2], bool)
    region[y0:y1, x0:x1] = True
    out = img.copy()
    out[region] = (img[region] * (1 - alpha) + over[region] * alpha).astype(np.uint8)
    return out


@pytest.mark.parametrize('colour', [(185, 215, 240), (230, 90, 200), (90, 230, 120), (240, 240, 240)])
def test_scan_line_checkpoint_volume_masked(colour):
    img = scan_lines(textured(9, (70, 80, 95), amp=12), 700, 280, 1000, 460, colour)
    m = ov.overlay_masks(down(img))
    inner = box448((720, 300, 980, 440))
    assert m.volume[inner].mean() > 0.9, colour
    assert m.ring[inner].mean() > 0.9
    outside = np.ones((H, W), bool)
    outside[box448((680, 260, 1020, 480))] = False
    assert (m.volume & outside).mean() < 0.002


def test_diagonal_lattice_and_plain_texture_are_not_volumes():
    for img in (scan_lines(textured(10, (70, 80, 95), amp=12), 700, 280, 1000, 460, (200, 200, 200), slope=0.8),
                textured(11, (120, 110, 90), amp=30)):
        m = ov.overlay_masks(down(img))
        assert m.volume.mean() < 0.002


# (the ranking-list text zone at x >= 1036, y ~150-580 is excluded from trail detection, so the path ends before it)
TRAIL = np.array([[100, 650], [400, 560], [700, 520], [1000, 545]], np.int32)


def trail_frame(colour, ranking=3, seed=12):
    base = with_ranking(textured(seed, (150, 125, 80), amp=18), ranking) if ranking else \
        textured(seed, (150, 125, 80), amp=18)
    img = base.copy()
    cv2.polylines(img, [TRAIL], False, colour, 5, cv2.LINE_AA)
    truth = down((np.abs(img.astype(int) - base.astype(int)).max(2) > 40).astype(np.float32)) >= 0.25
    truth &= ~ov.layout()['static']
    return down(img), truth


@pytest.mark.parametrize('colour', [(240, 120, 170), (240, 140, 40), (60, 120, 240), (150, 70, 230), (60, 200, 60)])
def test_ghost_trail_detected_when_other_racers_listed(colour):
    rgb, truth = trail_frame(colour)
    m = ov.overlay_masks(rgb)
    assert m.rank_rows == 3
    assert (m.masked & truth).sum() >= 0.9 * truth.sum(), colour
    if colour != (60, 200, 60):     # a green trail has the colour of the mint checkpoint discs: masked as ring
        assert (m.ghost & truth).sum() >= 0.8 * truth.sum(), colour


def test_trail_detector_off_without_other_racers():
    """Trails exist only when the ranking list shows other racers (own row + >= 1); the gate is a Params value."""
    rgb, truth = trail_frame((240, 120, 170), ranking=0)
    m = ov.overlay_masks(rgb)
    assert m.rank_rows == 0 and not m.ghost.any()
    m = ov.overlay_masks(rgb, ov.Params(ghost_min_rank_rows=0))
    assert (m.ghost & truth).sum() >= 0.8 * truth.sum()


def test_saturated_blob_is_not_a_trail():
    img = with_ranking(textured(10, (90, 80, 60)), 3)
    cv2.circle(img, (400, 400), 60, (220, 40, 40), -1)      # a red scene object, not thin
    m = ov.overlay_masks(down(img))
    assert not m.ghost[box448((360, 360, 440, 440))].any()


def test_stick_circles_and_collinear_horizon_dots_masked_isolated_specks_not():
    img = textured(13, (60, 70, 90), amp=10)
    sticks = [(575, 655), (705, 690)]
    for c in sticks:
        annulus(img, c, r_out=6, r_in=3)
    dots = [(470 + 22 * k, int(300 + 0.25 * 22 * k)) for k in range(6)]
    for x, y in dots:
        cv2.rectangle(img, (x, y), (x + 8, y + 5), (255, 255, 255), -1)
    specks = [(760, 200), (800, 470), (520, 430)]           # bright specks, not on one line
    for x, y in specks:
        cv2.rectangle(img, (x, y), (x + 5, y + 5), (255, 255, 255), -1)
    m = ov.overlay_masks(down(img))
    for x, y in sticks:
        assert m.hud[box448((x - 3, y - 3, x + 3, y + 3))].all()
    for x, y in dots:
        assert m.hud[box448((x + 2, y + 1, x + 6, y + 4))].all()
    for x, y in specks:
        assert not m.masked[box448((x + 1, y + 1, x + 4, y + 4))].any()


def test_repeated_calls_are_stateless():
    """Work buffers are reused between calls; results must not depend on the previous frame."""
    a, _ = trail_frame((240, 120, 170))
    b = down(scan_lines(textured(14, (70, 80, 95), amp=12), 200, 100, 900, 600, (220, 220, 240)))
    first = ov.overlay_masks(a)
    ov.overlay_masks(b)
    again = ov.overlay_masks(a)
    for k in ('hud', 'ring', 'ghost', 'volume', 'propeller'):
        assert np.array_equal(getattr(first, k), getattr(again, k)), k
    assert first.rings_uv == again.rings_uv and first.rank_rows == again.rank_rows


def test_to_model_frame_crops_composite_and_resizes():
    comp = np.zeros((720, 1928, 3), np.uint8)
    comp[:, 648:] = textured(11)
    out = ov.to_model_frame(comp)
    assert out.shape == (H, W, 3)
    assert np.array_equal(out, down(comp[:, 648:]))
    small = cv2.resize(textured(11), (640, 360), interpolation=cv2.INTER_AREA)
    assert ov.to_model_frame(small).shape == (H, W, 3)
    with pytest.raises(ValueError):
        ov.to_model_frame(np.zeros((480, 640, 3), np.uint8))


def test_budget_regression_guard():
    """The plan budget is <= 2 ms per frame. Measured offline over the 180 hand-checked frames (2 OpenCV
    threads, this machine) the median is ~2.0 ms, p95 ~2.6 ms. This synthetic frame runs every detector at once
    (6-row ranking list, marker, trail, volume, text), which real frames rarely do, so it gets a 3 ms guard
    against regressions on a shared machine rather than the 2 ms median target."""
    cv2.setNumThreads(2)
    img = with_ranking(textured(12, (120, 100, 70)), 6)
    annulus(img, (640, 560))
    cv2.polylines(img, [TRAIL], False, (240, 120, 170), 5)
    cv2.putText(img, '01:23:456', (575, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    img = scan_lines(img, 560, 420, 760, 560, (190, 220, 240))
    rgb = down(img)
    ov.overlay_masks(rgb)
    medians = []
    for _ in range(3):
        ts = []
        for _ in range(40):
            t0 = time.perf_counter()
            ov.overlay_masks(rgb)
            ts.append(time.perf_counter() - t0)
        medians.append(float(np.median(ts)))
    assert min(medians) < 0.003, medians


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
