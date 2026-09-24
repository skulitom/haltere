"""Per-frame overlay masks and the model's validity channel (runtime-safe; imports numpy and cv2 only).

Input: one 448 x 252 RGB uint8 frame. Store frames are cv2.INTER_AREA resizes of the 1280 x 720 gameplay crop
or of the 640 x 360 capture; ``to_model_frame`` does the same resize for runtime frames. Output: ``OverlayMasks``
with four boolean (252, 448) masks. One parameter set (``Params``) for every environment; nothing here reads
course geometry, routes, labels, telemetry or any other frame.

- ``hud``: Liftoff HUD glyphs at pixel level, never whole boxes (the drone's travel line crosses the centre-line
  and stick-display boxes in 65-81 % of frames).
  (a) Fixed glyphs: pixels that are near-white in >= 90 % of 300 dark-scene frames (ACRO badge, ALT/VIT/km/h
      labels, framing-column dashes, reticle circle, stick-display crosshairs, Rec. label), stored at
      1280 x 720 in ``configs/obstacles/overlay_hud_static_1280x720.png``.
  (b) Dynamic text on fixed text lines (race timer, TOUR n/N + lap clock + delta, compass tape, ALT/VIT
      digits): thin bright strokes by a top-hat on min(R, G, B), grown by one pixel into the anti-aliased
      rim; red / green personal-best delta digits and the red Rec. dot by colour.
  (c) Ghost-racer ranking list: vertically centred on the screen centre with rows 33.5 px apart, so an
      n-row list occupies n fixed slots of a half-row grid. Each slot gets an icon-square edge score; the
      row count with the largest summed score margin wins; its icon squares are masked whole and its name
      text by the glyph rule.
  (d) Dotted horizon bars (they move with attitude): small bright dots in the centre zone that lie on one
      straight line (>= ``horizon_min_dots`` collinear dots), so isolated bright specks in clouds stay scene.
  (e) Stick-position circles: the largest small bright blob off the static crosshair in each stick box,
      masked as a disc.
- ``ring``: (a) the white next-checkpoint marker, a fixed-size donut found anywhere (it is clamped to the
  image border when the checkpoint is off-screen): dark-hole candidates verified by normalised
  cross-correlation with a donut template and a ring-versus-surround contrast, masked as a disc; the centres
  are returned in ``rings_uv``; (b) translucent mint checkpoint discs by colour; (c) checkpoint volumes of any
  colour, found by their screen-space scan lines (one line every ~3.2 rows at 448 x 252): periodicity of
  the luminance over 16 x 6 px windows (see ``_volumes``), masked whole where striped; ``vol_mode=3`` adds
  the older magenta / green stripe-row rule (``_stripes``: +~1.4 points of overlay removal on the dev frames,
  +0.35 ms, more scene loss in the travel column).
- ``ghost``: other racers' ghost trails, removed from the input AND from every loss and label; never a cue.
  Run only when the ranking list shows other racers. Union at half resolution of (a) thin, elongated
  components of an opponent-colour (red-green, yellow-blue) top-hat / black-hat response that contain a
  strong core, and (b) strong response seeds whose structure tensor is coherent (a line, not foliage),
  grown along the weaker response. See ``KNOWN_GAPS``: recall is far below the plan's 90 %.
- ``propeller``: the propeller-blur wedges of the original [Copy] New Drone (temporal static-frequency map
  of 42k frames, low frequency = spinning blur, plus a margin), stored in
  ``configs/obstacles/overlay_propeller_wedge_1280x720.png``. Not deleted: 0.5 in the validity channel.

Model input: ``masked_input`` sets hud | ring | ghost pixels to IMAGENET_MEAN_U8 (0 after normalisation) and
``validity_channel`` returns float32 (252, 448): 0 masked, 0.5 propeller zone, 1 scene.

Budget: <= 2 ms per frame at 448 x 252 with 2 OpenCV threads. Measured on the 180 hand-checked frames (this
machine, Liftoff running): median 2.0-2.06 ms (dev 1.96-2.0, held-out 2.1), p95 ~2.6 ms, i.e. at the budget,
not under it; tests/test_obstacle_overlays.py only guards a synthetic frame. Implementation notes: in this
build numpy uint8 ``|`` / ``&`` on a full frame costs ~35 us (bool ops ~2 us, cv2 bitwise ~3 us), float
cv2.blur ~90 us (cv2.filter2D ~15 us) and connectedComponentsWithStats ~50 us regardless of image size, so
masks are combined as bool, float box filters use filter2D and labelling calls are merged where possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN_U8 = (124, 116, 104)
VALID_SCENE, VALID_PROPELLER, VALID_MASKED = 1.0, 0.5, 0.0

FRAME_W, FRAME_H = 448, 252
HALF_W, HALF_H = 224, 126
SOURCE_W, SOURCE_H = 1280, 720          # the gameplay layout the assets and zones are defined in
COMPOSITE_GAMEPLAY_X = 648              # 1928 x 720 run videos: brain panel x < 648, gameplay x >= 648
ASSET_DIR = Path(__file__).resolve().parents[2] / 'configs' / 'obstacles'
STATIC_HUD_ASSET = 'overlay_hud_static_1280x720.png'        # 255 = fixed HUD glyph pixel
PROPELLER_ASSET = 'overlay_propeller_wedge_1280x720.png'    # 255 = propeller zone
SURVEY_ASSETS = ('overlay_mask_fixed_hud_1280x720.png',     # survey boxes (255 = usable), provenance only
                 'overlay_mask_propellers_1280x720.png')

# HUD element zones in 1280 x 720 gameplay pixels (x0, y0, x1, y1). Glyphs are detected per frame inside.
TEXT_LINES = {
    'timer': (575, 25, 703, 49),           # 00:00:000
    'lap': (538, 56, 745, 79),             # TOUR n/N  +-00:00:000 (white, red or green)
    'compass': (505, 114, 790, 145),
    'alt': (1195, 53, 1240, 80),
    'vit': (1195, 119, 1240, 146),
}
TEXT_ZONES = {                             # no marker / ghost / stripe detection inside (letters, icons)
    'timer': (515, 14, 785, 88),
    'compass': (495, 100, 785, 156),
    'altvit': (1125, 32, 1280, 156),
    'acro': (20, 30, 115, 75),
    'rec': (0, 672, 100, 720),
}
COLOUR_ZONES = {
    'timer': (515, 14, 785, 88),           # red / green delta digits
    'rec': (0, 672, 100, 720),             # red Rec. dot and label
}
HORIZON_ZONE = (430, 140, 850, 580)        # dotted horizon bars move with attitude inside this zone
STICK_BOXES = ((536, 616, 637, 720), (643, 616, 744, 720))   # stick displays (position circle moves inside)
# Ranking list: rows centred on y = 360 (screen centre), 33.5 px apart; n rows use slots m = -(n-1), -(n-3),
# ..., n-1 of the half-row grid y = 360 + 16.75 m. Icon squares at x 1240-1266, name text right-aligned to 1236.
RANK_CENTRE_Y, RANK_HALF_STEP, RANK_MAX_ROWS = 360.0, 16.75, 7
RANK_ICON_X = (1240, 1266)
RANK_ICON_HALF = 13.0
RANK_MASK_X = (1236, 1273)                 # masked icon square (with its anti-aliased rim)
RANK_TEXT_X = (1036, 1237)
# checkpoint-volume scan lines: 9 px apart at 1280 (3.15 rows at 448), one colour every 18 px (6.3 rows)
STRIPE_LAGS = (3, 4, 6, 7)
LAG_ANCHOR = 7

KNOWN_GAPS = (
    'ghost trails: roughly 6 in 10 trail instances are found on the hand-checked frames (plan target 9 in '
    '10). Missed: white / pale trails and the pale beam of ghosts starting under the camera, wide trail '
    'sections very close to the camera, very short pieces, and trails with a weak colour shift over '
    'foliage or other busy colour texture; foliage and neon scene lines also cause false detections',
    'ghost drones (small racer models) are not masked',
    'transient centre texts (lap-time popup, countdown, LIFTOFF logo, finish screen) and the pause menu are '
    'not masked; the frame store drops those frames',
    'HUD glyphs drawn over near-white cloud are not detected (they are also nearly invisible there)',
    'checkpoint volumes: only the striped parts are found (the 16 x 6 px windows blur their borders), faint '
    'volumes over busy texture are missed, and regular horizontal scene structure with a ~3-row period '
    '(some fences, facades, roof panels) can be taken for a volume',
    'the faint crash / reset pictogram (bottom right) is not masked',
)


@dataclass(frozen=True)
class Params:
    static_cov: float = 0.04       # 448 px pixel is static HUD when >= this fraction of it is asset glyph
    propeller_cov: float = 0.5
    # white HUD glyphs on the text lines (448 px units), on min(R, G, B) / max(R, G, B)
    glyph_tophat: int = 3          # strokes are 1-2 px wide at 448
    glyph_strong: int = 22         # top-hat response of a stroke pixel
    glyph_strong_hi: int = 110     # its max(R, G, B)
    glyph_rim_tophat: int = 7
    glyph_rim: int = 0             # > 0: the 1 px rim must also pass a 7x7 top-hat of this value (0: plain growth)
    glyph_grow: int = 0            # final dilation (px) of the glyph mask inside the text line
    glyph_chroma_max: int = 90     # max - min of white glyph / dot pixels
    # ranking list
    rank_score_min: float = 22.0   # icon-square edge score margin per row
    rank_glyph_strong: int = 22
    rank_glyph_strong_hi: int = 100
    rank_icon_pad: int = 1         # extra rows (448 px) above and below each masked icon square
    # horizon dots
    dot_tophat: int = 5
    dot_strong: int = 40
    dot_lo: int = 140
    dot_max_px: int = 4            # component bbox limit (448 px)
    horizon_min_dots: int = 5      # collinear dots needed
    horizon_tol: float = 1.3       # px from the fitted line
    horizon_max_cand: int = 36
    # stick-position circles
    stick_tophat: int = 30
    stick_max_px: int = 7          # blob bbox limit (448 px)
    stick_min_area: int = 3
    stick_radius: float = 3.5
    # white next-checkpoint marker
    marker_hole: int = 45          # closing(lo, 5x5) - lo at the hole
    marker_ring_lo: int = 150      # closing(lo, 5x5) at the hole (bright ring around it)
    marker_max_cand: int = 40
    marker_ncc: float = 0.6
    marker_contrast: int = 35      # ring mean minus surround median, min(R, G, B)
    marker_radius: int = 4         # masked disc radius (448 px)
    # mint checkpoint discs (strong / weak for hysteresis)
    mint_gr: int = 30              # G - R
    mint_gb: int = 8               # G - B
    mint_s: int = 40               # saturation (max - min) / max, 0-255
    mint_v: int = 60
    mint_weak_gr: int = 14
    mint_weak_gb: int = -12
    # striped checkpoint volumes
    stripe_t: int = 12             # vertical top-hat of the magenta / green opponent channel
    stripe_abs: int = 0            # the stripe pixel itself is magenta (min(R,B) - G) or green (G - max(R,B))
    stripe_run: int = 5            # horizontal run length (px) of a stripe row
    stripe_win: int = 9            # vertical closing that fills between stripe rows (px)
    stripe_weak: int = 8           # weaker stripe rows accepted near the seeds (any hue)
    stripe_min_px: int = 24        # fewer stripe-row pixels (at 224 x 252) than this: no volume
    stripe_grow: int = 41          # seed neighbourhood (px)
    stripe_gate_level: int = 10    # quarter-res magenta / green margin that makes a pixel count for the gate
    stripe_gate_px: int = 20       # run the colour-stripe rule only with >= this many such pixels (0: always)
    # checkpoint volumes of any colour (scan-line periodicity of the luminance, 224 x 252 working resolution)
    vol_mode: int = 2              # 1 colour stripes only, 2 scan-line periodicity only, 3 both
    vol_bw: int = 4                # correlation window in 4 x 3 px blocks (16 x 6 px)
    vol_bh: int = 2
    vol_energy: float = 6.0        # min mean squared deviation from the 3-row mean
    vol_c3: float = 0.45           # min normalised correlation 3 rows apart (one scan-line period)
    vol_c1: float = -0.15          # max normalised correlation 1 row apart
    vol_ch: float = 0.5            # min normalised correlation 4 px apart horizontally (horizontal lines)
    vol_src: int = 1               # 0 max(R, G, B), 1 grey, 2 min(R, G, B)
    vol_dil: int = 3               # final dilation (px) of the accepted blocks
    # ghost trails (half resolution, 224 x 126); detector (a): thin elongated components
    ghost_k: int = 3               # opponent top-hat / black-hat size (shared with (b) when equal)
    ghost_weak: int = 5
    ghost_strong: int = 14
    ghost_v_min: int = 20
    ghost_len_min: float = 7.0     # component bbox diagonal (half-res px)
    ghost_width_max: float = 2.5   # component area / bbox diagonal (half-res px)
    ghost_n_strong: int = 2
    ghost_grow_t: int = 4
    ghost_out_dilate: int = 3
    ghost_grow_iter: int = 1
    ghost_min_rank_rows: int = 2   # run the trail detector only when the ranking list lists other racers
    ghost_thin: int = 1            # run detector (a)
    # detector (b): coherent-line trail seeds (half resolution)
    ghost_line: int = 1            # run detector (b)
    ghost_line_k: int = 3          # top-hat / black-hat size
    ghost_line_strong: int = 12    # seed response
    ghost_line_weak: int = 7       # growth response
    ghost_line_iter: int = 5       # geodesic growth steps (half-res px)
    ghost_line_len: float = 8.0    # min component bbox diagonal (half-res px)
    ghost_coh_w: int = 7           # structure-tensor window
    ghost_coh: float = 0.5         # min squared coherence ((l1 - l2) / (l1 + l2))^2 of the structure tensor


DEFAULT_PARAMS = Params()


@dataclass
class OverlayMasks:
    hud: np.ndarray        # bool (252, 448)
    ring: np.ndarray       # bool
    ghost: np.ndarray      # bool
    propeller: np.ndarray  # bool (zone, not deleted)
    rings_uv: tuple = field(default_factory=tuple)   # detected marker centres, normalised (u, v)
    rank_rows: int = 0                               # ranking-list rows found
    volume: np.ndarray | None = None                 # bool: the checkpoint-volume part of ``ring`` (masked
    #                                                  whole, so the opening behind a volume is invalid input)
    parts: dict = field(default_factory=dict)        # per-detector masks (debug=True only)

    @property
    def masked(self) -> np.ndarray:
        return self.hud | self.ring | self.ghost


def _cv2():
    import cv2
    return cv2


def to_model_frame(img: np.ndarray) -> np.ndarray:
    """448 x 252 RGB uint8 from a 1928 x 720 run-video composite, a 1280 x 720 gameplay frame or a 640 x 360
    capture (any 16:9 frame), with cv2.INTER_AREA as in the frame store."""
    cv2 = _cv2()
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError('expected an HxWx3 uint8 RGB frame')
    h, w = img.shape[:2]
    if (w, h) == (1928, 720):
        img = img[:, COMPOSITE_GAMEPLAY_X:]
        h, w = img.shape[:2]
    if abs(w * 9 - h * 16) > 16:
        raise ValueError(f'expected a 16:9 gameplay frame, got {w}x{h}')
    if (w, h) == (FRAME_W, FRAME_H):
        return img
    return cv2.resize(np.ascontiguousarray(img), (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)


def _box(b, w=FRAME_W, h=FRAME_H):
    """1280 x 720 box -> covering integer box at (w, h)."""
    sx, sy = w / SOURCE_W, h / SOURCE_H
    x0, y0, x1, y1 = b
    return (max(0, int(np.floor(x0 * sx))), max(0, int(np.floor(y0 * sy))),
            min(w, int(np.ceil(x1 * sx))), min(h, int(np.ceil(y1 * sy))))


def _slices(b, w=FRAME_W, h=FRAME_H):
    x0, y0, x1, y1 = _box(b, w, h)
    return slice(y0, y1), slice(x0, x1)


def _load_asset(name: str) -> np.ndarray:
    cv2 = _cv2()
    path = ASSET_DIR / name
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (SOURCE_H, SOURCE_W):
        raise FileNotFoundError(f'overlay asset missing or wrong size: {path}')
    return img


def marker_template(size: int = 7, r_out: float = 2.9, r_in: float = 1.2, ss: int = 16) -> np.ndarray:
    """Zero-mean, unit-norm donut template of the next-checkpoint marker at 448 px (supersampled render; the
    radii fit the mean of 49 hand-accepted markers with correlation 0.986)."""
    n = size * ss
    yy, xx = (np.mgrid[:n, :n] + 0.5) / ss - size / 2.0
    d = np.hypot(xx, yy)
    t = ((d <= r_out) & (d >= r_in)).astype(np.float32).reshape(size, ss, size, ss).mean((1, 3))
    t -= t.mean()
    return t / np.linalg.norm(t)


def _disc(r: float) -> np.ndarray:
    n = int(np.ceil(r))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return (np.hypot(xx, yy) <= r).astype(np.uint8)


@lru_cache(maxsize=4)
def layout(width: int = FRAME_W, height: int = FRAME_H) -> dict:
    """Fixed layers for the 448 x 252 frame: static HUD glyphs, propeller zone, zone slices, ranking-list slots
    and templates (cached; arrays are read-only)."""
    if (width, height) != (FRAME_W, FRAME_H):
        raise ValueError('overlay layout is defined for the 448 x 252 model frame; use to_model_frame')
    cv2 = _cv2()
    p = DEFAULT_PARAMS
    glyph = cv2.dilate(_load_asset(STATIC_HUD_ASSET), np.ones((3, 3), np.uint8))   # + anti-aliased rim
    static = cv2.resize(glyph.astype(np.float32) / 255.0, (width, height),
                        interpolation=cv2.INTER_AREA) >= p.static_cov
    prop = cv2.resize(_load_asset(PROPELLER_ASSET).astype(np.float32) / 255.0, (width, height),
                      interpolation=cv2.INTER_AREA) >= p.propeller_cov
    sx, sy = width / SOURCE_W, height / SOURCE_H
    ms = np.arange(-(2 * RANK_MAX_ROWS - 2), 2 * RANK_MAX_ROWS - 1)
    yc = RANK_CENTRE_Y + RANK_HALF_STEP * ms
    icon_a = np.round((yc - RANK_ICON_HALF) * sy).astype(int)
    icon_b = np.round((yc + RANK_ICON_HALF) * sy).astype(int)
    mask_a = np.floor((yc - RANK_ICON_HALF - 2) * sy).astype(int) - p.rank_icon_pad
    mask_b = np.ceil((yc + RANK_ICON_HALF + 2) * sy).astype(int) + p.rank_icon_pad
    text_a = np.floor((yc - 12) * sy).astype(int)
    text_b = np.ceil((yc + 26) * sy).astype(int)          # the own row has a second line below the name
    idx = {int(m): k for k, m in enumerate(ms)}
    hyps = [np.zeros(0, int)] + [np.array([idx[m] for m in range(-(n - 1), n, 2)])
                                 for n in range(1, RANK_MAX_ROWS + 1)]
    ix0, ix1 = int(round(RANK_ICON_X[0] * sx)), int(round(RANK_ICON_X[1] * sx))
    rank_y = (max(0, int(icon_a.min()) - 3), min(height, int(icon_b.max()) + 3))
    rank_x = (ix0 - 2, min(width, ix1 + 3))
    mx = (int(np.floor(RANK_MASK_X[0] * sx)), min(width, int(np.ceil(RANK_MASK_X[1] * sx))))
    tx = (int(np.floor(RANK_TEXT_X[0] * sx)), int(np.ceil(RANK_TEXT_X[1] * sx)))
    # regions where HUD text lives: no marker / ghost / stripe detection there (letters with holes, icons)
    text_region = np.zeros((height, width), bool)
    for b in TEXT_ZONES.values():
        ys, xs = _slices(b, width, height)
        text_region[ys, xs] = True
    text_region[max(0, int(mask_a.min())):int(text_b.max()), tx[0]:width] = True
    hz = _slices(HORIZON_ZONE, width, height)
    # dynamic HUD text: one band covering every text line, glyphs kept only on the lines
    boxes = np.array([_box(b, width, height) for b in TEXT_LINES.values()])
    tb = (int(boxes[:, 0].min()), int(boxes[:, 1].min()), int(boxes[:, 2].max()), int(boxes[:, 3].max()))
    text_lines = np.zeros((tb[3] - tb[1], tb[2] - tb[0]), bool)
    for x0, y0, x1, y1 in boxes:
        text_lines[y0 - tb[1]:y1 - tb[1], x0 - tb[0]:x1 - tb[0]] = True
    yy9, xx9 = np.mgrid[-4:5, -4:5]
    d9 = np.hypot(xx9, yy9)
    k_up = np.zeros((2 * LAG_ANCHOR + 1, 1), np.uint8)   # dilation picks the stripe mask d rows above ...
    k_dn = np.zeros((2 * LAG_ANCHOR + 1, 1), np.uint8)   # ... or d rows below, d in STRIPE_LAGS
    for d in STRIPE_LAGS:
        k_up[LAG_ANCHOR - d] = 1
        k_dn[LAG_ANCHOR + d] = 1
    # borders of the translucent stick-display boxes (and the bright gap between them) look like thin lines
    stick_edges = np.zeros((height, width), bool)
    for b in STICK_BOXES:
        x0, y0, x1, y1 = _box(b, width, height)
        stick_edges[max(0, y0 - 2):y1, max(0, x0 - 2):x0 + 3] = True
        stick_edges[max(0, y0 - 2):y1, x1 - 3:x1 + 2] = True
        stick_edges[max(0, y0 - 2):y0 + 3, max(0, x0 - 2):x1 + 2] = True
    # both stick boxes in one union rectangle (one connected-components call); per column: box index or -1
    sboxes = [_box(b, width, height) for b in STICK_BOXES]
    ux0, uy0 = min(b[0] for b in sboxes), min(b[1] for b in sboxes)
    ux1, uy1 = max(b[2] for b in sboxes), max(b[3] for b in sboxes)
    union_ok = np.zeros((uy1 - uy0, ux1 - ux0), bool)
    box_of_col = np.full(ux1 - ux0, -1, np.intp)
    for k, (x0, y0, x1, y1) in enumerate(sboxes):
        union_ok[y0 - uy0:y1 - uy0, x0 - ux0:x1 - ux0] = True
        box_of_col[x0 - ux0:x1 - ux0] = k
    union_ok &= ~static[uy0:uy1, ux0:ux1]
    out = dict(
        stick_union=(slice(uy0, uy1), slice(ux0, ux1)), stick_union_ok=union_ok, stick_box_of_col=box_of_col,
        static=static, static_u8=static.astype(np.uint8), propeller=prop, text_region=text_region,
        ghost_excl_half=cv2.resize((text_region | stick_edges).astype(np.uint8), (HALF_W, HALF_H),
                                    interpolation=cv2.INTER_AREA) > 0,
        marker_ok=~(text_region | static),
        win7=np.array([dy * width + dx for dy in range(-3, 4) for dx in range(-3, 4)], np.intp),
        win9=np.array([dy * width + dx for dy in range(-4, 5) for dx in range(-4, 5)], np.intp),
        text_band=(slice(tb[1], tb[3]), slice(tb[0], tb[2])), text_lines=text_lines,
        timer=_slices(COLOUR_ZONES['timer'], width, height), rec=_slices(COLOUR_ZONES['rec'], width, height),
        horizon=hz, horizon_static=static[hz],
        stick=[_slices(b, width, height) for b in STICK_BOXES],
        stick_static=[static[_slices(b, width, height)] for b in STICK_BOXES],
        rank=dict(icon_a=icon_a - rank_y[0], icon_b=icon_b - rank_y[0], mask_a=mask_a, mask_b=mask_b,
                  text_a=text_a, text_b=text_b, hyps=hyps, y=rank_y, x=rank_x, lx=1, rx=(ix1 - ix0) + 1,
                  mask_x=mx, text_x=tx),
        marker=marker_template(), marker_flat=marker_template().ravel().astype(np.float32),
        stick_disc=_disc(p.stick_radius),
        marker_band=((d9 >= 1.5) & (d9 <= 2.6)).ravel(), marker_outer=(d9 >= 3.6).ravel(),
        stripe_lag_kernels=(k_up, k_dn),
        pairs={n: np.triu_indices(n, 1) for n in range(p.horizon_max_cand + 1)},
        offsets3=np.array([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]))
    for v in list(out.values()) + list(out['rank'].values()):
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return out


_KERNELS = {}
_SCRATCH = []


def _scratch():
    """float32 work buffers for ``_volumes``: the high-pass image (252, 224) and the product panels
    (252, VOL_NPANEL * 240 - 16); the border rows / columns and panel gaps it never writes stay 0. One set per
    process (the runtime calls ``overlay_masks`` from one thread)."""
    if not _SCRATCH:
        _SCRATCH.append((np.zeros((FRAME_H, HALF_W), np.float32),
                         np.zeros((FRAME_H, VOL_NPANEL * VOL_PANEL - 16), np.float32)))
    return _SCRATCH[0]


def _k(w, h=None):
    key = (w, h or w)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = np.ones((key[1], key[0]), np.uint8)
    return k


def _boxk(w, h):
    """Normalised float32 box kernel (cv2.filter2D is several times faster than cv2.blur on float32 here)."""
    key = ('box', w, h)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = np.full((h, w), 1.0 / (w * h), np.float32)
    return k


def _any(m) -> bool:
    """Fast any() for a contiguous 2-D bool / uint8 image."""
    return _cv2().countNonZero(m.view(np.uint8)) > 0


def _glyphs(lo, hi, strong_th, strong_hi, p, cv2):
    """Glyph pixels (bool) in a text line: thin bright strokes (3 x 3 top-hat of min(R, G, B) for white text,
    of max(R, G, B) for red / green text), grown into the anti-aliased rim that the 1280 -> 448 area resize
    spreads around every stroke."""
    k = _k(p.glyph_tophat)
    th = cv2.max(cv2.morphologyEx(lo, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(hi, cv2.MORPH_TOPHAT, k))
    strong = (th >= strong_th) & (hi >= strong_hi)
    if not strong.any():
        return None
    grown = cv2.dilate(strong.view(np.uint8), _k(3)).view(bool)
    if p.glyph_rim > 0:                                  # rim pixels must be brighter than the local background
        k = _k(p.glyph_rim_tophat)
        rim = cv2.max(cv2.morphologyEx(lo, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(hi, cv2.MORPH_TOPHAT, k))
        g = strong | ((rim >= p.glyph_rim) & grown)
    else:
        g = grown
    if p.glyph_grow:
        g = cv2.dilate(g.view(np.uint8), _k(2 * p.glyph_grow + 1)).view(bool)
    return g


def rank_slot_scores(lo, sat, L=None):
    """Icon-square edge score of every ranking-list slot (the minimum over the square's four sides of the
    mean edge energy |d min(R,G,B)| + |d (max-min)| across that side, best of +-1 px)."""
    L = L or layout()
    R = L['rank']
    (y0, y1), (x0, x1) = R['y'], R['x']
    sub = lo[y0:y1, x0:x1].astype(np.int16)
    subc = sat[y0:y1, x0:x1].astype(np.int16)
    gx = np.abs(np.diff(sub, axis=1)) + np.abs(np.diff(subc, axis=1))
    gy = np.abs(np.diff(sub, axis=0)) + np.abs(np.diff(subc, axis=0))
    lx, rx = R['lx'], R['rx']
    left = np.concatenate([[0], np.cumsum(gx[:, max(0, lx - 1):lx + 2].max(1))])
    right = np.concatenate([[0], np.cumsum(gx[:, rx - 1:rx + 2].max(1))])
    rowe = gy[:, 3:-3].mean(1)
    a, b = R['icon_a'], R['icon_b']
    n = np.maximum(b - a - 2, 1)
    lm = (left[b - 1] - left[a + 1]) / n
    rm = (right[b - 1] - right[a + 1]) / n
    top = np.maximum(np.maximum(rowe[a - 2], rowe[a - 1]), rowe[a])
    bot = np.maximum(np.maximum(rowe[b - 2], rowe[b - 1]), rowe[b])
    return np.minimum(np.minimum(lm, rm), np.minimum(top, bot))


def _rank_list(lo, sat, hud, L, p, cv2):
    """Detect the ghost-racer ranking list; mask icon squares and name text in place. Returns the row count."""
    R = L['rank']
    s = rank_slot_scores(lo, sat, L) - p.rank_score_min
    best_n, best = 0, 0.0
    for n in range(1, len(R['hyps'])):
        v = float(s[R['hyps'][n]].sum())
        if v > best:
            best_n, best = n, v
    if not best_n:
        return 0
    rows = R['hyps'][best_n]
    (mx0, mx1), (tx0, tx1) = R['mask_x'], R['text_x']
    for k in rows:
        hud[max(0, R['mask_a'][k]):R['mask_b'][k], mx0:mx1] = 1
    ya, yb = max(0, int(R['text_a'][rows[0]])), min(FRAME_H, int(R['text_b'][rows[-1]]))
    hi = cv2.add(lo[ya:yb, tx0:tx1], sat[ya:yb, tx0:tx1])
    g = _glyphs(lo[ya:yb, tx0:tx1], hi, p.rank_glyph_strong, p.rank_glyph_strong_hi, p, cv2)
    if g is not None:
        hud[ya:yb, tx0:tx1] |= g.view(np.uint8)
    return best_n


def _horizon(lo, sat, hud, L, p, cv2):
    """Dotted horizon bars: small bright dots in the centre zone that lie on one straight line."""
    ys, xs = L['horizon']
    sl = lo[ys, xs]
    th = cv2.morphologyEx(sl, cv2.MORPH_TOPHAT, _k(p.dot_tophat))
    dots = (th >= p.dot_strong) & (sl >= p.dot_lo) & (sat[ys, xs] <= p.glyph_chroma_max) & ~L['horizon_static']
    if not dots.any():
        return
    n, lab, st, cen = cv2.connectedComponentsWithStats(dots.view(np.uint8), connectivity=8)
    small = np.flatnonzero((st[1:, 2] <= p.dot_max_px) & (st[1:, 3] <= p.dot_max_px)) + 1
    if len(small) < p.horizon_min_dots:
        return
    if len(small) > p.horizon_max_cand:
        c = np.rint(cen[small]).astype(int)
        strength = th[np.clip(c[:, 1], 0, th.shape[0] - 1), np.clip(c[:, 0], 0, th.shape[1] - 1)]
        small = small[np.argsort(-strength.astype(np.int16), kind='stable')[:p.horizon_max_cand]]
    x, y = cen[small, 0], cen[small, 1]
    i, j = L['pairs'][len(small)]
    ax, ay = x[j] - x[i], y[j] - y[i]
    ln = np.hypot(ax, ay)
    ok = ln >= 6.0
    if not ok.any():
        return
    i, ax, ay, ln = i[ok], ax[ok], ay[ok], ln[ok]
    # distance of every dot to the line through each pair: |cross(a, p - p_i)| / |a|
    dist = np.abs(ax[:, None] * (y[None, :] - y[i][:, None]) - ay[:, None] * (x[None, :] - x[i][:, None]))
    inl = dist <= p.horizon_tol * ln[:, None]
    cnt = inl.sum(1)
    b = int(np.argmax(cnt))
    if cnt[b] < p.horizon_min_dots:
        return
    keep = np.zeros(n, np.uint8)
    keep[small[inl[b]]] = 1
    zone = hud[ys, xs]
    cv2.max(zone, cv2.dilate(_select(lab, n, keep, cv2), _k(3)), dst=zone)


def _sticks(lo, hud, L, p, cv2):
    """Stick-position circles: the largest small bright near-white blob (off the static crosshair) inside each
    stick box, masked as a disc. Both boxes are labelled in one call (they lie side by side)."""
    ys, xs = L['stick_union']
    sub = lo[ys, xs]
    th = cv2.morphologyEx(sub, cv2.MORPH_TOPHAT, _k(p.dot_tophat))
    blob = ((th >= p.stick_tophat) & (sub >= p.dot_lo) & L['stick_union_ok']).view(np.uint8)
    if not blob.any():
        return
    n, lab, st, cen = cv2.connectedComponentsWithStats(blob, connectivity=8)
    if n <= 1:
        return
    ok = (st[1:, 2] <= p.stick_max_px) & (st[1:, 3] <= p.stick_max_px) & (st[1:, 4] >= p.stick_min_area)
    box = L['stick_box_of_col'][np.clip(np.rint(cen[1:, 0] - 0.5).astype(int), 0, sub.shape[1] - 1)]
    for k in range(len(L['stick'])):
        okk = ok & (box == k)
        if not okk.any():
            continue
        best = int(np.argmax(np.where(okk, st[1:, 4], -1))) + 1
        cx, cy = int(round(cen[best, 0] - 0.5)) + xs.start, int(round(cen[best, 1] - 0.5)) + ys.start
        D = L['stick_disc']
        h = D.shape[0] // 2
        ya, yb, xa, xb = cy - h, cy + h + 1, cx - h, cx + h + 1
        oya, oxa = max(0, -ya), max(0, -xa)
        ya, xa, yb, xb = max(0, ya), max(0, xa), min(FRAME_H, yb), min(FRAME_W, xb)
        hud[ya:yb, xa:xb] |= D[oya:oya + yb - ya, oxa:oxa + xb - xa]


def _markers(lo, ring, p, L, cv2):
    """White next-checkpoint marker(s): dark-hole candidates verified by NCC with the donut template."""
    clo = cv2.morphologyEx(lo, cv2.MORPH_CLOSE, _k(5))
    hole = cv2.subtract(clo, lo)
    cand = (hole >= p.marker_hole) & (clo >= p.marker_ring_lo) & L['marker_ok']
    if not cand.any():
        return ()
    cand &= hole >= cv2.dilate(hole, _k(3))              # one candidate per hole (local maxima)
    idx = np.flatnonzero(cand)
    if not len(idx):
        return ()
    if len(idx) > p.marker_max_cand:
        hv = hole.ravel()[idx].astype(np.int16)
        idx = idx[np.argsort(-hv, kind='stable')[:p.marker_max_cand]]
    ys, xs = np.divmod(idx, FRAME_W)
    # NCC with the donut template at the candidate and its 8 neighbours (vectorised flat-index gathers; window
    # centres are clipped 3 px inside the border, the markers clamped to the border sit >= 3 px inside)
    off = L['offsets3']
    vy = ys[:, None] + off[None, :, 0]
    vx = xs[:, None] + off[None, :, 1]
    inside = (vx >= 3) & (vy >= 3) & (vx < FRAME_W - 3) & (vy < FRAME_H - 3)
    centre = np.clip(vy, 3, FRAME_H - 4) * FRAME_W + np.clip(vx, 3, FRAME_W - 4)
    flat = lo.ravel()
    patch = flat[centre[..., None] + L['win7']].astype(np.float32)          # (n, 9, 49)
    patch -= patch.mean(axis=2, keepdims=True)
    nrm = np.sqrt((patch * patch).sum(axis=2))
    ncc = (patch @ L['marker_flat']) / np.maximum(nrm, 1e-3)
    ncc[(nrm < 1e-3) | ~inside] = -1.0
    bo = ncc.argmax(1)
    k = np.arange(len(xs))
    c, u, v = ncc[k, bo], vx[k, bo], vy[k, bo]
    ok = c >= p.marker_ncc
    if not ok.any():
        return ()
    u, v = u[ok], v[ok]
    # the white ring must stand out from its surround (band mean minus median of the outer 9 x 9 pixels)
    w9 = flat[(np.clip(v, 4, FRAME_H - 5) * FRAME_W + np.clip(u, 4, FRAME_W - 5))[:, None] + L['win9']]
    w9 = w9.astype(np.float32)
    ring_mean = w9[:, L['marker_band']].mean(1)
    outer = np.sort(w9[:, L['marker_outer']], axis=1)
    m = outer.shape[1] // 2
    outer_med = outer[:, m] if outer.shape[1] % 2 else 0.5 * (outer[:, m - 1] + outer[:, m])
    ok = ring_mean - outer_med >= p.marker_contrast
    out = []
    for uu, vv in zip(u[ok].tolist(), v[ok].tolist()):
        if any(abs(uu - a) <= 3 and abs(vv - b_) <= 3 for a, b_ in out):
            continue
        out.append((uu, vv))
        cv2.circle(ring, (uu, vv), p.marker_radius, 1, -1)
    return tuple(((uu + 0.5) / FRAME_W, (vv + 0.5) / FRAME_H) for uu, vv in out)


def _mint(r, g, b, hi, sat, p, cv2):
    """Translucent mint checkpoint discs: green-dominant cyan (G - R and G - B margins, saturation)."""
    gr = cv2.subtract(g, r)
    weak = (gr >= p.mint_weak_gr) & (hi >= p.mint_v)
    if not _any(weak):
        return None
    gb = cv2.addWeighted(g, 1.0, b, -1.0, 128.0)                          # G - B centred on 128
    strong = weak & (gr >= p.mint_gr) & (gb >= 128 + p.mint_gb)
    if not _any(strong):
        return None
    strong &= cv2.addWeighted(sat, 1.0, hi, -p.mint_s / 255.0, 128.0) >= 128   # (max - min) / max >= mint_s
    if not _any(strong):
        return None
    weak &= gb >= 128 + p.mint_weak_gb
    return strong | (weak & cv2.dilate(strong.view(np.uint8), _k(3)).view(bool))


def _stripe_colours(r, g, b, p, cv2):
    """Cheap gate for ``_stripes``: at 1/4 resolution, enough clearly magenta (min(R, B) - G) or clearly green
    (G - max(R, B)) pixels for a striped volume to be possible."""
    rq, gq, bq = (np.ascontiguousarray(c[1::4, 1::4]) for c in (r, g, b))
    mag = cv2.subtract(cv2.min(rq, bq), gq)
    grn = cv2.subtract(gq, cv2.max(rq, bq))
    return cv2.countNonZero(cv2.compare(cv2.max(mag, grn), p.stripe_gate_level, cv2.CMP_GE)) >= p.stripe_gate_px


def _stripes(r, g, b, L, p, cv2):
    """Striped checkpoint volumes: thin horizontal magenta / green rows that repeat vertically (seeds), grown
    into weaker stripe rows near the seeds and filled between the rows. The rows are horizontal, so the work is
    done at half horizontal resolution (224 x 252)."""
    mn, mx = cv2.min(r, b), cv2.max(r, b)
    mag = cv2.subtract(cv2.add(cv2.subtract(mn, g), 128), cv2.subtract(g, mn))   # 128 + min(R,B) - G, clipped
    grn = cv2.subtract(cv2.add(cv2.subtract(g, mx), 128), cv2.subtract(mx, g))   # 128 + G - max(R,B), clipped
    mag, grn = _col_pairs(mag, cv2), _col_pairs(grn, cv2)            # 224 x 252: mean of column pairs
    kv = _k(1, 5)
    th_m = cv2.morphologyEx(mag, cv2.MORPH_TOPHAT, kv)
    th_g = cv2.morphologyEx(grn, cv2.MORPH_TOPHAT, kv)
    run = _k((p.stripe_run + 1) // 2, 1)
    weak = cv2.morphologyEx((cv2.max(th_m, th_g) >= p.stripe_weak).view(np.uint8), cv2.MORPH_OPEN, run)
    if not _any(weak):
        return None
    s = (((th_m >= p.stripe_t) & (mag >= 128 + p.stripe_abs)) |
         ((th_g >= p.stripe_t) & (grn >= 128 + p.stripe_abs))).view(np.uint8)
    s = cv2.morphologyEx(s, cv2.MORPH_OPEN, run)
    if cv2.countNonZero(s) < p.stripe_min_px:
        return None
    # periodicity: the scan lines repeat every 9 px at 1280 (3.15 rows at 448) and one colour every 18 px
    # (6.3 rows); a seed row has another stripe row 3, 4, 6 or 7 rows above AND below it
    k_up, k_dn = L['stripe_lag_kernels']
    seeds = cv2.min(s, cv2.min(cv2.dilate(s, k_up, anchor=(0, LAG_ANCHOR)), cv2.dilate(s, k_dn, anchor=(0, LAG_ANCHOR))))
    if not _any(seeds):
        return None
    # grow into weaker stripe rows near the seeds, fill between the rows, one pixel into the volume border
    vol = seeds | (weak & cv2.dilate(seeds, _k(p.stripe_grow // 2 | 1, p.stripe_grow)))
    vol = cv2.morphologyEx(vol, cv2.MORPH_CLOSE, _k(1, p.stripe_win))
    vol = cv2.dilate(cv2.resize(vol, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST), _k(3))
    return vol.view(bool) & ~L['text_region']


def _col_pairs(a, cv2):
    """uint8 (H, W) -> (H, W / 2): rounded mean of column pairs (= INTER_LINEAR 2:1, without its overhead)."""
    return cv2.addWeighted(np.ascontiguousarray(a[:, 0::2]), 0.5, np.ascontiguousarray(a[:, 1::2]), 0.5, 0.0)


VOL_GRID = (112, 84)                       # block statistics grid: 4 x 3 px blocks of the 448 x 252 frame
VOL_PANEL = HALF_W + 16                    # product panels side by side, 16 zero columns apart (8 blocks)
VOL_NPANEL = 6                             # h^2, lag 3 rows, 1 row, 2 columns (4 px), 2 rows, 1 column (2 px)


def _hp3(a, out):
    """out[1:-1] = a - mean of 3 rows (rows 0 and -1 of out stay 0)."""
    o = out[1:-1]
    np.add(a[:-2], a[2:], out=o)
    o += a[1:-1]
    o *= -1.0 / 3.0
    o += a[1:-1]
    return out


def _volumes(grey2, L, p, cv2, bufs):
    """Checkpoint volumes of any colour: screen-space horizontal scan lines, one line every ~3.2 rows at
    448 x 252 (9 px apart at 1280 x 720). Every other column is used (the lines are horizontal). With h = grey2
    minus its 3-row mean, over a window of (vol_bw x vol_bh) blocks of 4 x 3 px, a block is volume when h
    (a) correlates with itself 3 rows away (one line period) and (b) anti-correlates 1 and 2 rows away, all
    normalised by the window mean of h^2, (c) is coherent 2 and 4 px apart horizontally (horizontal lines;
    rejects diagonal lattices such as fences and roof trusses) and (d) is visible (energy). The six products sit side
    by side in one float image, so one area resize and one box filter give all the window means (float
    cv2.blur is several times slower than cv2.filter2D here). Accepted blocks are closed, upsampled, dilated."""
    h, prods = bufs
    _hp3(grey2.astype(np.float32), h)                 # grey2: (252, 224) view, every other column
    w, q = HALF_W, VOL_PANEL
    np.multiply(h, h, out=prods[:, 0:w])
    np.multiply(h[:-3], h[3:], out=prods[:-3, q:q + w])
    np.multiply(h[:-1], h[1:], out=prods[:-1, 2 * q:2 * q + w])
    np.multiply(h[:, :-2], h[:, 2:], out=prods[:, 3 * q:3 * q + w - 2])
    np.multiply(h[:-2], h[2:], out=prods[:-2, 4 * q:4 * q + w])
    np.multiply(h[:, :-1], h[:, 1:], out=prods[:, 5 * q:5 * q + w - 1])
    S = cv2.filter2D(cv2.resize(prods, (VOL_GRID[0] * VOL_NPANEL + 8 * (VOL_NPANEL - 1), VOL_GRID[1]),
                                interpolation=cv2.INTER_AREA), -1, _boxk(p.vol_bw, p.vol_bh))
    nb, pb = VOL_GRID[0], VOL_PANEL // 2
    P = lambda k: S[:, k * pb:k * pb + nb]
    E = P(0)
    ne = np.maximum(E, 1e-3)
    # (b) covers 1 and 2 rows: both negative for the lines (one bright row in three), while a smooth gradient
    # plus pixel noise (sky, flat walls) is anti-correlated at 1 row but correlated at 2 rows
    # (c) at 2 and 4 px: a diagonal lattice can be back in phase after 4 px, but then not after 2 px
    m = ((E >= p.vol_energy) & (P(1) >= p.vol_c3 * ne) & (P(2) <= p.vol_c1 * ne) & (P(4) <= p.vol_c1 * ne)
         & (P(3) >= p.vol_ch * ne) & (P(5) >= p.vol_ch * ne))
    m = m.view(np.uint8)
    if not _any(m):
        return None
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _k(3, 3))
    m = cv2.resize(m, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
    if p.vol_dil:
        m = cv2.dilate(m, _k(p.vol_dil))
    return m.view(bool) & ~L['text_region']


def _opp_hat(o1, o2, k, cv2):
    """max over the red-green and yellow-blue opponent channels of their top-hat and black-hat (k x k)."""
    kk = _k(k)
    return cv2.max(cv2.max(cv2.morphologyEx(o1, cv2.MORPH_TOPHAT, kk), cv2.morphologyEx(o1, cv2.MORPH_BLACKHAT, kk)),
                   cv2.max(cv2.morphologyEx(o2, cv2.MORPH_TOPHAT, kk), cv2.morphologyEx(o2, cv2.MORPH_BLACKHAT, kk)))


def _select(lab, n, value, cv2):
    """uint8 image value[label] (value: bool or uint8 per component; a LUT when the labels fit in 8 bits)."""
    value = np.asarray(value).view(np.uint8) if np.asarray(value).dtype == bool else np.asarray(value, np.uint8)
    if n <= 256:
        lut = np.zeros(256, np.uint8)
        lut[:n] = value
        return cv2.LUT(lab.astype(np.uint8), lut)
    return value[lab]


def _coherent(r, seeds, p, cv2):
    """seeds (bool) where the structure tensor of r over a ghost_coh_w window is coherent (one orientation):
    ((l1 - l2) / (l1 + l2))^2 >= ghost_coh. Evaluated at the seed pixels only."""
    idx = np.flatnonzero(seeds)
    if not len(idx):
        return seeds
    gx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
    kx = _KERNELS.get(('sep', p.ghost_coh_w))
    if kx is None:
        kx = _KERNELS[('sep', p.ghost_coh_w)] = np.full(p.ghost_coh_w, 1.0 / p.ghost_coh_w, np.float32)
    a = cv2.sepFilter2D(gx * gx, -1, kx, kx).ravel()[idx]
    c = cv2.sepFilter2D(gy * gy, -1, kx, kx).ravel()[idx]
    b = cv2.sepFilter2D(gx * gy, -1, kx, kx).ravel()[idx]
    d, tr = a - c, a + c
    ok = d * d + 4.0 * b * b >= p.ghost_coh * tr * tr + 1e-3
    out = np.zeros(seeds.size, bool)
    out[idx[ok]] = True
    return out.reshape(seeds.shape)


def _ghost(rgb, excl, L, p, cv2):
    """Ghost trails at half resolution (224 x 126), from the opponent-colour (red-green, yellow-blue) top-hat /
    black-hat response. Union of two detectors:
    (a) thin: connected components of the k=5 response that are elongated (bbox diagonal, area / diagonal) and
        contain a strong core, grown one step into the weaker response;
    (b) line: seeds where the k=3 response is strong AND its structure tensor is coherent (one orientation over
        a ghost_coh_w window, i.e. a line rather than foliage / texture), grown geodesically inside the weaker
        response; components shorter than ghost_line_len are dropped.
    Returns a bool (252, 448) mask, never including ``excl``."""
    h = cv2.resize(rgb, (HALF_W, HALF_H), interpolation=cv2.INTER_AREA)
    r, g, b = cv2.split(h)
    hi = cv2.max(cv2.max(r, g), b)
    o1 = cv2.addWeighted(r, 0.5, g, -0.5, 128.0)                                         # red - green
    o2 = cv2.addWeighted(cv2.addWeighted(r, 0.25, g, 0.25, 0.0), 1.0, b, -0.5, 128.0)    # yellow - blue
    ex8 = cv2.resize(excl.view(np.uint8), (HALF_W, HALF_H), interpolation=cv2.INTER_AREA)
    okb = (ex8 == 0) & ~L['ghost_excl_half'] & (hi >= p.ghost_v_min)
    ok8 = okb.view(np.uint8)
    resp = _opp_hat(o1, o2, p.ghost_k, cv2)
    W = HALF_W
    both = np.zeros((HALF_H, 2 * W + 1), np.uint8)       # (a) candidates | gap column | (b) grown line seeds
    cand = None
    if p.ghost_thin:
        cand = (resp >= p.ghost_weak) & okb
        both[:, :W] = cand
    if p.ghost_line:
        r3 = resp if p.ghost_line_k == p.ghost_k else _opp_hat(o1, o2, p.ghost_line_k, cv2)
        seeds = (r3 >= p.ghost_line_strong) & okb
        if seeds.any():
            seeds = _coherent(r3, seeds, p, cv2)
            if seeds.any():
                weak = cv2.min(cv2.compare(r3, p.ghost_line_weak, cv2.CMP_GE), ok8)
                m = seeds.view(np.uint8)
                for _ in range(p.ghost_line_iter):
                    m = cv2.min(cv2.dilate(m, _k(3)), weak)
                both[:, W + 1:] = m
    if not both.any():
        return None
    # one connected-components call for both detectors (its statistics pass dominates its cost)
    n, lab, st, _ = cv2.connectedComponentsWithStats(both, connectivity=8)
    dg = np.hypot(st[:, 2], st[:, 3]).astype(np.float32)
    left = st[:, 0] < W
    keep_a = left & (dg >= p.ghost_len_min) & (st[:, 4] / np.maximum(dg, 1.0) <= p.ghost_width_max)
    if cand is not None and keep_a.any():
        keep_a &= np.bincount(lab[:, :W][cand & (resp >= p.ghost_strong)], minlength=n) >= p.ghost_n_strong
    keep_b = ~left & (dg >= p.ghost_line_len)
    keep_a[0] = keep_b[0] = False
    if not (keep_a.any() or keep_b.any()):
        return None
    code = _select(lab, n, keep_a.astype(np.uint8) * 2 + keep_b, cv2)      # 2 thin, 1 line
    thin = cv2.compare(code[:, :W], 2, cv2.CMP_EQ) if keep_a.any() else None
    if thin is not None:
        thin = cv2.min(thin, 1)
        grow = cv2.min(cv2.compare(resp, p.ghost_grow_t, cv2.CMP_GE), ok8)
        for _ in range(p.ghost_grow_iter):                  # grow into the weaker trail response
            thin = cv2.min(cv2.dilate(thin, _k(3)), cv2.max(grow, thin))
        half = cv2.max(cv2.add(thin, thin), np.ascontiguousarray(code[:, W + 1:]))
    else:
        half = np.ascontiguousarray(code[:, W + 1:])
    # one upsample for both: thin pixels coded 2, line pixels 1; only the thin part gets the output dilation
    up = cv2.resize(half, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
    out = cv2.min(up, 1)
    if thin is not None and p.ghost_out_dilate:
        out = cv2.max(out, cv2.min(cv2.dilate(cv2.compare(up, 2, cv2.CMP_GE), _k(p.ghost_out_dilate)), 1))
    return out.view(bool) & ~excl


def overlay_masks(rgb: np.ndarray, params: Params = DEFAULT_PARAMS, debug: bool = False) -> OverlayMasks:
    """Masks for one uint8 (252, 448, 3) RGB frame. ``debug`` also returns per-detector masks in ``parts``."""
    cv2 = _cv2()
    rgb = np.asarray(rgb)
    if rgb.shape != (FRAME_H, FRAME_W, 3) or rgb.dtype != np.uint8:
        raise ValueError(f'overlay_masks expects a ({FRAME_H}, {FRAME_W}, 3) uint8 RGB frame '
                         f'(use to_model_frame), got {rgb.shape} {rgb.dtype}')
    rgb = np.ascontiguousarray(rgb)
    p = params
    L = layout()
    parts = {}
    r, g, b = cv2.split(rgb)
    hi = cv2.max(cv2.max(r, g), b)
    lo = cv2.min(cv2.min(r, g), b)
    sat = cv2.subtract(hi, lo)

    # --- HUD ------------------------------------------------------------------------------------------
    hud = L['static_u8'].copy()
    if debug:
        parts['static'] = hud.astype(bool)
        prev = hud.copy()
    ys, xs = L['text_band']
    gm = _glyphs(lo[ys, xs], hi[ys, xs], p.glyph_strong, p.glyph_strong_hi, p, cv2)
    if gm is not None:
        hud[ys, xs] |= (gm & L['text_lines']).view(np.uint8)
    for ys, xs in (L['timer'], L['rec']):
        rr, gg, bb = r[ys, xs], g[ys, xs], b[ys, xs]
        col = (((rr > 150) & (cv2.subtract(rr, gg) > 60) & (cv2.subtract(rr, bb) > 50))
               | ((gg > 140) & (cv2.subtract(gg, rr) > 50) & (cv2.subtract(gg, bb) > 20)))
        if col.any():
            hud[ys, xs] |= cv2.dilate(col.view(np.uint8), _k(3))
    if debug:
        parts['text'] = (hud != prev); prev = hud.copy()
    rank_rows = _rank_list(lo, sat, hud, L, p, cv2)
    if debug:
        parts['rank'] = (hud != prev); prev = hud.copy()
    _horizon(lo, sat, hud, L, p, cv2)
    if debug:
        parts['horizon'] = (hud != prev); prev = hud.copy()
    _sticks(lo, hud, L, p, cv2)
    if debug:
        parts['sticks'] = (hud != prev)
    hud_b = hud.view(bool)

    # --- checkpoint marker, mint discs, checkpoint volumes ---------------------------------------------
    ring8 = np.zeros((FRAME_H, FRAME_W), np.uint8)
    rings_uv = _markers(lo, ring8, p, L, cv2)
    ring = ring8.view(bool)                              # bool from here on (numpy uint8 | is slow)
    if debug:
        parts['marker'] = ring.copy()
    mint = _mint(r, g, b, hi, sat, p, cv2)
    if mint is not None:
        ring |= mint
        if debug:
            parts['mint'] = mint
    volume = np.zeros((FRAME_H, FRAME_W), bool)
    if p.vol_mode & 1 and _stripe_colours(r, g, b, p, cv2):
        vol = _stripes(r, g, b, L, p, cv2)
        if vol is not None:
            volume |= vol
            if debug:
                parts['stripes'] = vol
    if p.vol_mode & 2:
        # every other column only (the lines are horizontal); a strided 3-channel copy would cost ~170 us here
        src = (hi, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if p.vol_src == 1 else None, lo)[p.vol_src][:, ::2]
        vol = _volumes(src, L, p, cv2, _scratch())
        if vol is not None:
            volume |= vol
            if debug:
                parts['volumes'] = vol
    ring |= volume
    ring_b = ring & ~hud_b
    volume &= ~hud_b

    # --- ghost trails ---------------------------------------------------------------------------------
    # other racers' trails exist only when the ranking list shows other racers (own row + >= 1 more)
    ghost = _ghost(rgb, hud_b | ring_b, L, p, cv2) if rank_rows >= p.ghost_min_rank_rows else None
    if ghost is None:
        ghost = np.zeros((FRAME_H, FRAME_W), bool)
    if debug:
        parts['ghost'] = ghost
    return OverlayMasks(hud=hud_b, ring=ring_b, ghost=ghost, propeller=L['propeller'].copy(),
                        rings_uv=rings_uv, rank_rows=rank_rows, volume=volume, parts=parts)


def validity_channel(masks: OverlayMasks) -> np.ndarray:
    """float32 (252, 448): 0 masked (HUD, ring, ghost), 0.5 propeller zone, 1 scene."""
    v = np.full(masks.hud.shape, VALID_SCENE, np.float32)
    v[masks.propeller] = VALID_PROPELLER
    v[masks.masked] = VALID_MASKED
    return v


def masked_input(rgb: np.ndarray, masks: OverlayMasks | None = None) -> np.ndarray:
    """RGB copy with HUD, ring and ghost pixels set to IMAGENET_MEAN_U8 (masks computed when not given)."""
    if masks is None:
        masks = overlay_masks(rgb)
    out = np.array(rgb, dtype=np.uint8, copy=True)
    out[masks.masked] = IMAGENET_MEAN_U8
    return out
