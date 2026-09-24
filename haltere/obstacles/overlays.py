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
  (e) Stick-position circles: the best donut-template match inside each stick box, masked as a disc.
- ``ring``: (a) the white next-checkpoint marker, a fixed-size donut found anywhere (it is clamped to the
  image border when the checkpoint is off-screen), masked as a disc; (b) translucent mint checkpoint discs
  by colour; (c) striped checkpoint volumes (screen-space horizontal scan lines about 9 px apart at
  1280 x 720, magenta or green): rows whose colour is a thin vertical extremum, continued horizontally,
  filled where the stripe density is high. The volume is masked whole where it is striped.
- ``ghost``: other racers' ghost trails, removed from the input AND from every loss and label; never a cue.
  Thin, elongated components of an opponent-colour (red-green, yellow-blue) top-hat / black-hat response
  at half resolution, with a strong core, grown into the weaker response around them. See ``KNOWN_GAPS``.
- ``propeller``: the propeller-blur wedges of the original [Copy] New Drone (temporal static-frequency map
  of 42k frames, low frequency = spinning blur, plus a margin), stored in
  ``configs/obstacles/overlay_propeller_wedge_1280x720.png``. Not deleted: 0.5 in the validity channel.

Model input: ``masked_input`` sets hud | ring | ghost pixels to IMAGENET_MEAN_U8 (0 after normalisation) and
``validity_channel`` returns float32 (252, 448): 0 masked, 0.5 propeller zone, 1 scene.

Budget: <= 2 ms CPU per frame at 448 x 252 with 2 OpenCV threads (tests/test_obstacle_overlays.py).
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
    'ghost trails: only coloured, thin, elongated trail pieces with a clear colour contrast are found; '
    'white / pale beams of a ghost flying just ahead of the camera, wide trail sections very close to the '
    'camera, very short pieces and trails with a weak colour shift over a busy background are missed',
    'ghost drones (small racer models) are not masked',
    'transient centre texts (lap-time popup, countdown, LIFTOFF logo, finish screen) and the pause menu are '
    'not masked; the frame store drops those frames',
    'HUD glyphs drawn over near-white cloud are not detected (they are also nearly invisible there)',
    'checkpoint volumes are found only where their scan-line stripes are visible and coloured',
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
    glyph_rim: int = 1
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
    # ghost trails (half resolution, 224 x 126)
    ghost_k: int = 5
    ghost_weak: int = 7
    ghost_strong: int = 18
    ghost_v_min: int = 20
    ghost_open: int = 0            # opening removes blobs; the residue keeps thin structures
    ghost_len_min: float = 7.0     # component bbox diagonal (half-res px)
    ghost_width_max: float = 2.5   # component area / bbox diagonal (half-res px)
    ghost_n_strong: int = 2
    ghost_grow_t: int = 6
    ghost_out_dilate: int = 3
    ghost_grow_iter: int = 1
    ghost_min_rank_rows: int = 2   # run the trail detector only when the ranking list lists other racers


DEFAULT_PARAMS = Params()


@dataclass
class OverlayMasks:
    hud: np.ndarray        # bool (252, 448)
    ring: np.ndarray       # bool
    ghost: np.ndarray      # bool
    propeller: np.ndarray  # bool (zone, not deleted)
    rings_uv: tuple = field(default_factory=tuple)   # detected marker centres, normalised (u, v)
    rank_rows: int = 0                               # ranking-list rows found
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
    out = dict(
        static=static, static_u8=static.astype(np.uint8), propeller=prop, text_region=text_region,
        ghost_excl_half=cv2.resize((text_region | stick_edges).astype(np.uint8), (HALF_W, HALF_H),
                                    interpolation=cv2.INTER_AREA) > 0,
        marker_excl=text_region | static,
        text_band=(slice(tb[1], tb[3]), slice(tb[0], tb[2])), text_lines=text_lines,
        timer=_slices(COLOUR_ZONES['timer'], width, height), rec=_slices(COLOUR_ZONES['rec'], width, height),
        horizon=hz, horizon_static=static[hz],
        stick=[_slices(b, width, height) for b in STICK_BOXES],
        stick_static=[static[_slices(b, width, height)] for b in STICK_BOXES],
        rank=dict(icon_a=icon_a - rank_y[0], icon_b=icon_b - rank_y[0], mask_a=mask_a, mask_b=mask_b,
                  text_a=text_a, text_b=text_b, hyps=hyps, y=rank_y, x=rank_x, lx=1, rx=(ix1 - ix0) + 1,
                  mask_x=mx, text_x=tx),
        marker=marker_template(), stick_disc=_disc(p.stick_radius),
        marker_band=((d9 >= 1.5) & (d9 <= 2.6)).ravel(), marker_outer=(d9 >= 3.6).ravel(),
        stripe_lag_kernels=(k_up, k_dn),
        pairs={n: np.triu_indices(n, 1) for n in range(p.horizon_max_cand + 1)},
        offsets3=np.array([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]))
    for v in list(out.values()) + list(out['rank'].values()):
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return out


_KERNELS = {}


def _k(w, h=None):
    key = (w, h or w)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = np.ones((key[1], key[0]), np.uint8)
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
    k = _k(p.glyph_rim_tophat)
    rim = cv2.max(cv2.morphologyEx(lo, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(hi, cv2.MORPH_TOPHAT, k))
    g = strong | ((rim >= p.glyph_rim) & cv2.dilate(strong.view(np.uint8), _k(3)).view(bool))
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
    hud[ys, xs] |= cv2.dilate(keep[lab], _k(3))


def _sticks(lo, hud, L, p, cv2):
    """Stick-position circles: the largest small bright near-white blob (off the static crosshair) inside each
    stick box, masked as a disc."""
    for (ys, xs), stat in zip(L['stick'], L['stick_static']):
        sub = lo[ys, xs]
        th = cv2.morphologyEx(sub, cv2.MORPH_TOPHAT, _k(p.dot_tophat))
        blob = ((th >= p.stick_tophat) & (sub >= p.dot_lo) & ~stat).view(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(blob, connectivity=8)
        if n <= 1:
            continue
        ok = (st[1:, 2] <= p.stick_max_px) & (st[1:, 3] <= p.stick_max_px) & (st[1:, 4] >= p.stick_min_area)
        if not ok.any():
            continue
        best = int(np.argmax(np.where(ok, st[1:, 4], -1))) + 1
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
    cand = (hole >= p.marker_hole) & (clo >= p.marker_ring_lo) & ~L['marker_excl']
    if not _any(cand):
        return ()
    cand &= hole >= cv2.dilate(hole, _k(3))              # one candidate per hole (local maxima)
    pts = cv2.findNonZero(cand.view(np.uint8))
    if pts is None:
        return ()
    pts = pts.reshape(-1, 2)
    xs, ys = pts[:, 0].astype(np.intp), pts[:, 1].astype(np.intp)
    hv = hole[ys, xs].astype(np.int16)
    order = np.argsort(-hv, kind='stable')[:p.marker_max_cand]
    xs, ys = xs[order], ys[order]
    # NCC with the donut template at the candidate and its 8 neighbours (vectorised; window positions are
    # clipped at the image border, the markers clamped to the border sit >= 3 px inside)
    T = L['marker']
    w7 = np.lib.stride_tricks.sliding_window_view(lo, (7, 7))
    off = L['offsets3']
    vy = ys[:, None] + off[None, :, 0]
    vx = xs[:, None] + off[None, :, 1]
    inside = (vx >= 3) & (vy >= 3) & (vx < FRAME_W - 3) & (vy < FRAME_H - 3)
    patch = w7[np.clip(vy - 3, 0, FRAME_H - 7), np.clip(vx - 3, 0, FRAME_W - 7)].astype(np.float32)
    patch -= patch.mean(axis=(2, 3), keepdims=True)
    nrm = np.sqrt((patch * patch).sum(axis=(2, 3)))
    ncc = (patch * T).sum(axis=(2, 3)) / np.maximum(nrm, 1e-3)
    ncc[(nrm < 1e-3) | ~inside] = -1.0
    bo = ncc.argmax(1)
    k = np.arange(len(xs))
    c, u, v = ncc[k, bo], vx[k, bo], vy[k, bo]
    ok = c >= p.marker_ncc
    if not ok.any():
        return ()
    u, v = u[ok], v[ok]
    # the white ring must stand out from its surround
    w9 = np.lib.stride_tricks.sliding_window_view(lo, (9, 9))[np.clip(v - 4, 0, FRAME_H - 9),
                                                              np.clip(u - 4, 0, FRAME_W - 9)]
    w9 = w9.reshape(len(u), 81).astype(np.float32)
    ring_mean = w9[:, L['marker_band']].mean(1)
    outer_med = np.median(w9[:, L['marker_outer']], axis=1)
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


def _stripes(r, g, b, L, p, cv2):
    """Striped checkpoint volumes: thin horizontal magenta / green rows that repeat vertically (seeds), grown
    into weaker stripe rows near the seeds and filled between the rows. The rows are horizontal, so the work is
    done at half horizontal resolution (224 x 252)."""
    hw = (HALF_W, FRAME_H)
    mn, mx = cv2.min(r, b), cv2.max(r, b)
    mag = cv2.subtract(cv2.add(cv2.subtract(mn, g), 128), cv2.subtract(g, mn))   # 128 + min(R,B) - G, clipped
    grn = cv2.subtract(cv2.add(cv2.subtract(g, mx), 128), cv2.subtract(mx, g))   # 128 + G - max(R,B), clipped
    mag = cv2.resize(mag, hw, interpolation=cv2.INTER_LINEAR)      # exact 2:1 in x: mean of column pairs
    grn = cv2.resize(grn, hw, interpolation=cv2.INTER_LINEAR)
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


def _ghost(rgb, excl, L, p, cv2):
    """Ghost trails at half resolution: thin elongated opponent-colour structures with a strong core."""
    h = cv2.resize(rgb, (HALF_W, HALF_H), interpolation=cv2.INTER_AREA)
    r, g, b = cv2.split(h)
    hi = cv2.max(cv2.max(r, g), b)
    o1 = cv2.addWeighted(r, 0.5, g, -0.5, 128.0)                                         # red - green
    o2 = cv2.addWeighted(cv2.addWeighted(r, 0.25, g, 0.25, 0.0), 1.0, b, -0.5, 128.0)    # yellow - blue
    k = _k(p.ghost_k)
    resp = cv2.max(cv2.max(cv2.morphologyEx(o1, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(o1, cv2.MORPH_BLACKHAT, k)),
                   cv2.max(cv2.morphologyEx(o2, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(o2, cv2.MORPH_BLACKHAT, k)))
    exh = (cv2.resize(excl.view(np.uint8), (HALF_W, HALF_H), interpolation=cv2.INTER_AREA) > 0) | \
        L['ghost_excl_half']
    cand = (resp >= p.ghost_weak) & (hi >= p.ghost_v_min) & ~exh
    if not _any(cand):
        return None
    thin = cand
    if p.ghost_open:
        thin = cand & ~cv2.morphologyEx(cand.view(np.uint8), cv2.MORPH_OPEN, _k(p.ghost_open)).view(bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats(thin.view(np.uint8), connectivity=8)
    if n <= 1:
        return None
    strong = thin & (resp >= p.ghost_strong)
    has = np.bincount(lab[strong], minlength=n) >= p.ghost_n_strong
    dg = np.hypot(st[:, 2], st[:, 3]).astype(np.float32)
    wd = st[:, 4] / np.maximum(dg, 1.0)
    keep = (dg >= p.ghost_len_min) & (wd <= p.ghost_width_max) & has
    keep[0] = False
    if not keep.any():
        return None
    kept = keep[lab]
    grow = ((resp >= p.ghost_grow_t) & ~exh).view(np.uint8)
    kept = kept.view(np.uint8)
    for _ in range(p.ghost_grow_iter):                  # geodesic growth along the weaker trail response
        kept = cv2.min(cv2.dilate(kept, _k(3)), cv2.max(grow, kept))
    out = cv2.resize(kept.view(np.uint8), (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
    if p.ghost_out_dilate:
        out = cv2.dilate(out, _k(p.ghost_out_dilate))
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

    # --- checkpoint marker, mint discs, striped volumes -----------------------------------------------
    ring = np.zeros((FRAME_H, FRAME_W), np.uint8)
    rings_uv = _markers(lo, ring, p, L, cv2)
    if debug:
        parts['marker'] = ring.astype(bool)
    mint = _mint(r, g, b, hi, sat, p, cv2)
    if mint is not None:
        ring |= mint.view(np.uint8)
        if debug:
            parts['mint'] = mint
    vol = _stripes(r, g, b, L, p, cv2)
    if vol is not None:
        ring |= vol.view(np.uint8)
        if debug:
            parts['stripes'] = vol
    ring_b = ring.view(bool) & ~hud_b

    # --- ghost trails ---------------------------------------------------------------------------------
    # other racers' trails exist only when the ranking list shows other racers (own row + >= 1 more)
    ghost = _ghost(rgb, hud_b | ring_b, L, p, cv2) if rank_rows >= p.ghost_min_rank_rows else None
    if ghost is None:
        ghost = np.zeros((FRAME_H, FRAME_W), bool)
    if debug:
        parts['ghost'] = ghost
    return OverlayMasks(hud=hud_b, ring=ring_b, ghost=ghost, propeller=L['propeller'].copy(),
                        rings_uv=rings_uv, rank_rows=rank_rows, parts=parts)


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
