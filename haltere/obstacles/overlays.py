"""Per-frame overlay masks and the model's validity channel (runtime-safe; imports numpy and cv2 only).

Input: one 448 x 252 RGB uint8 frame. Store frames are cv2.INTER_AREA resizes of the 1280 x 720 gameplay crop
or of the 640 x 360 capture; ``to_model_frame`` does the same resize for runtime frames. Output: ``OverlayMasks``
with four boolean (252, 448) masks. One parameter set (``Params``) for every environment; nothing here reads
course geometry, routes, labels or telemetry.

- ``hud``: Liftoff HUD glyphs, pixel level, never whole boxes (the drone's travel line crosses the centre-line
  and stick-display boxes in 65-81 % of frames).
  (a) Fixed glyphs: pixels that are near-white in >= 90 % of 300 dark-scene frames (ACRO badge, ALT/VIT/km/h
      labels, framing-column dashes, reticle circle, stick-display crosshairs, Rec. label), stored at
      1280 x 720 in ``configs/obstacles/overlay_hud_static_1280x720.png``; a 448 px pixel is static HUD when
      its footprint touches the asset (area coverage >= ``static_cov``).
  (b) Dynamic text in fixed zones (race timer / TOUR n/N / lap clock / delta line, compass tape, ALT/VIT
      digits): white glyph pixels by a top-hat on min(R, G, B), grown by one pixel into the anti-aliased
      rim; red / green personal-best delta digits and the red Rec. dot by colour.
  (c) Ghost-racer ranking list: it is vertically centred on the screen centre with rows 33.5 px apart, so an
      n-row list occupies n fixed slots of a half-row grid. Each slot gets an icon-square score (edge energy
      on the four sides of the 26 px icon square); the row count with the largest summed score margin is
      chosen, its icon squares are masked whole and its name text by the glyph rule.
  (d) Dotted horizon bars (they move with attitude) and stick-position dots: small bright components inside
      their zones.
- ``ring``: (a) the white next-checkpoint marker, a fixed-size donut (outer radius ~2.9 px, hole ~1.2 px at
  448 px) found anywhere (it is clamped to the image border when the checkpoint is off-screen): dark-hole
  candidates (black-hat of min(R, G, B) inside a bright closing) verified by normalised cross-correlation
  with the donut template and a ring-versus-surround contrast; masked as a disc.
  (b) translucent mint checkpoint discs / volumes by colour (green-dominant cyan: G - R and G - B margins,
  saturation), grown by one pixel into weaker mint pixels. The pale cyan horizon glow of the Drawing Board
  sky has B >= G and stays scene. Only coloured pixels are masked; the inside of a volume stays visible.
- ``ghost``: other racers' ghost trails, removed from the input AND from every loss and label; never a cue.
  Elongated components of an opponent-colour (red-green, yellow-blue) top-hat / black-hat response that
  contain a strong, saturated core. See ``KNOWN_GAPS`` for what is not caught.
- ``propeller``: the propeller-blur wedges of the original [Copy] New Drone (temporal static-frequency map of
  42k frames, low frequency = spinning blur, plus a margin), stored in
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
SOURCE_W, SOURCE_H = 1280, 720          # the gameplay layout the assets and zones are defined in
COMPOSITE_GAMEPLAY_X = 648              # 1928 x 720 run videos: brain panel x < 648, gameplay x >= 648
ASSET_DIR = Path(__file__).resolve().parents[2] / 'configs' / 'obstacles'
STATIC_HUD_ASSET = 'overlay_hud_static_1280x720.png'        # 255 = fixed HUD glyph pixel
PROPELLER_ASSET = 'overlay_propeller_wedge_1280x720.png'    # 255 = propeller zone
SURVEY_ASSETS = ('overlay_mask_fixed_hud_1280x720.png',     # survey boxes (255 = usable), provenance only
                 'overlay_mask_propellers_1280x720.png')

# HUD element zones in 1280 x 720 gameplay pixels (x0, y0, x1, y1). Glyphs are detected per frame inside.
TEXT_ZONES = {
    'timer': (515, 14, 785, 88),           # race timer, TOUR n/N lap clock, red/green personal-best delta
    'compass': (495, 100, 785, 156),       # compass tape (moves with yaw)
    'altvit': (1125, 32, 1280, 156),       # ALT / VIT digits (labels are static glyphs)
}
# Text lines inside those zones (measured from the hand-checked masks): glyphs are only searched here.
TEXT_LINES = {
    'timer': (575, 25, 703, 49),           # 00:00:000
    'lap': (538, 56, 745, 79),             # TOUR n/N  +-00:00:000 (white, red or green)
    'compass': (505, 114, 790, 145),
    'alt': (1195, 53, 1240, 80),
    'vit': (1195, 119, 1240, 146),
}
COLOUR_ZONES = {
    'timer': (515, 14, 785, 88),           # red / green delta digits
    'rec': (0, 672, 100, 720),             # red Rec. dot and label
}
HORIZON_ZONE = (440, 150, 840, 570)        # dotted horizon bars: small dots only (they move with attitude)
STICK_ZONES = ((536, 616, 637, 720), (643, 616, 744, 720))   # stick-position dots
# Ranking list: rows centred on y = 360 (screen centre), 33.5 px apart; n rows use slots m = -(n-1), -(n-3),
# ..., n-1 of the half-row grid y = 360 + 16.75 m. Icon squares at x 1240-1266, name text right-aligned to 1236.
RANK_CENTRE_Y, RANK_HALF_STEP, RANK_MAX_ROWS = 360.0, 16.75, 7
RANK_ICON_X = (1240, 1266)
RANK_ICON_HALF = 13.0
RANK_MASK_X = (1236, 1272)                 # masked icon square (with its anti-aliased rim)
RANK_TEXT_X = (1036, 1237)

KNOWN_GAPS = (
    'white / grey ghost trails and the wide translucent beams of a ghost flying just ahead of the camera '
    'are only partly detected: they are colourless and low-contrast, like fence rails, cloud edges, road '
    'markings and light shafts at 448 px',
    'ghost drones (small racer models) and very short trail pieces are missed unless they form an '
    'elongated coloured component',
    'transient centre texts (lap-time popup, countdown, LIFTOFF logo, finish screen) and the pause menu are '
    'not masked; the frame store drops those frames',
    'HUD glyphs drawn over near-white cloud are not detected (they are also nearly invisible there)',
    'the faint crash / reset pictogram (bottom right) is not masked',
)


@dataclass(frozen=True)
class Params:
    static_cov: float = 0.04       # 448 px pixel is static HUD when >= this fraction of it is asset glyph
    propeller_cov: float = 0.5
    # white HUD glyphs inside the text zones (448 px units), on min(R, G, B)
    glyph_tophat: int = 3          # strokes are 1-2 px wide at 448
    glyph_strong: int = 22         # top-hat response of a stroke pixel
    glyph_strong_hi: int = 110     # its max(R, G, B)
    glyph_rim_tophat: int = 7      # rim: 8-neighbours of a stroke pixel that are brighter than the background
    glyph_rim: int = 5
    glyph_chroma_max: int = 90     # max - min of white glyph / dot pixels
    # ranking list
    rank_score_min: float = 22.0   # icon-square edge score margin per row
    rank_glyph_strong: int = 22
    rank_glyph_strong_hi: int = 100
    # horizon / stick dots
    dot_tophat: int = 7
    dot_strong: int = 40
    dot_lo: int = 140
    dot_max_px: int = 4            # component bbox limit (448 px)
    stick_dot_max_px: int = 6
    # white next-checkpoint marker
    marker_hole: int = 45          # closing(lo, 5x5) - lo at the hole
    marker_ring_lo: int = 150      # closing(lo, 5x5) at the hole (bright ring around it)
    marker_max_cand: int = 40
    marker_ncc: float = 0.6
    marker_contrast: int = 35      # ring mean minus surround median, min(R, G, B)
    marker_radius: int = 4         # masked disc radius (448 px)
    # mint checkpoint discs and volumes (strong / weak for hysteresis)
    mint_gr: int = 30              # G - R
    mint_gb: int = 8               # G - B
    mint_s: int = 40               # saturation (max - min) / max, 0-255
    mint_v: int = 60
    mint_weak_gr: int = 14
    mint_weak_gb: int = -12
    # ghost trails: opponent-colour top-hat / black-hat (448 px units)
    ghost_k: int = 13
    ghost_strong: int = 16
    ghost_weak: int = 8
    ghost_s_min: int = 45          # max - min at a strong pixel
    ghost_v_min: int = 35
    ghost_len_min: int = 14        # component bbox diagonal (px)
    ghost_width_max: float = 3.2   # component area / bbox diagonal (px)


DEFAULT_PARAMS = Params()


@dataclass
class OverlayMasks:
    hud: np.ndarray        # bool (252, 448)
    ring: np.ndarray       # bool
    ghost: np.ndarray      # bool
    propeller: np.ndarray  # bool (zone, not deleted)
    rings_uv: tuple = field(default_factory=tuple)   # detected marker centres, normalised (u, v)
    rank_rows: int = 0                               # ranking-list rows found

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


@lru_cache(maxsize=4)
def layout(width: int = FRAME_W, height: int = FRAME_H) -> dict:
    """Fixed layers for the 448 x 252 frame: static HUD glyphs, propeller zone, zone slices, ranking-list slots
    and the marker template (cached; arrays are read-only)."""
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
    mask_a = np.floor((yc - RANK_ICON_HALF - 2) * sy).astype(int)
    mask_b = np.ceil((yc + RANK_ICON_HALF + 2) * sy).astype(int)
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
    # regions where HUD text lives: no marker / ghost detection there (letters with holes, coloured icons)
    text_region = np.zeros((height, width), bool)
    for b in TEXT_ZONES.values():
        ys, xs = _slices(b, width, height)
        text_region[ys, xs] = True
    text_region[int(mask_a.min()):int(text_b.max()), tx[0]:mx[1]] = True
    out = dict(
        static=static, static_u8=static.astype(np.uint8), propeller=prop, text_region=text_region,
        marker_excl=text_region | static,
        text=[_slices(b, width, height) for b in TEXT_LINES.values()],
        timer=_slices(COLOUR_ZONES['timer'], width, height), rec=_slices(COLOUR_ZONES['rec'], width, height),
        horizon=_slices(HORIZON_ZONE, width, height), stick=[_slices(b, width, height) for b in STICK_ZONES],
        rank=dict(icon_a=icon_a - rank_y[0], icon_b=icon_b - rank_y[0], mask_a=mask_a, mask_b=mask_b,
                  text_a=text_a, text_b=text_b, hyps=hyps, y=rank_y, x=rank_x, lx=1, rx=(ix1 - ix0) + 1,
                  mask_x=mx, text_x=tx),
        marker=marker_template())
    for v in list(out.values()) + list(out['rank'].values()):
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return out


_K3 = np.ones((3, 3), np.uint8)
_KERNELS = {}


def _k(n):
    k = _KERNELS.get(n)
    if k is None:
        k = _KERNELS[n] = np.ones((n, n), np.uint8)
    return k


def _glyphs(lo, hi, strong_th, strong_hi, p, cv2):
    """Glyph pixels (bool) in a text line: thin bright strokes (3 x 3 top-hat of min(R, G, B) for white text,
    of max(R, G, B) for red / green text), grown by one pixel to cover the anti-aliased rim that the
    1280 -> 448 area resize spreads around every stroke."""
    k = _k(p.glyph_tophat)
    th = cv2.max(cv2.morphologyEx(lo, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(hi, cv2.MORPH_TOPHAT, k))
    strong = (th >= strong_th) & (hi >= strong_hi)
    if not strong.any():
        return None
    k = _k(p.glyph_rim_tophat)
    rim = cv2.max(cv2.morphologyEx(lo, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(hi, cv2.MORPH_TOPHAT, k))
    return strong | ((rim >= p.glyph_rim) & cv2.dilate(strong.view(np.uint8), _K3).view(bool))


def _small_components(mask, max_px, cv2):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8), connectivity=8)
    if n <= 1:
        return None
    keep = np.zeros(n, np.uint8)
    keep[1:] = (st[1:, 2] <= max_px) & (st[1:, 3] <= max_px)
    if not keep.any():
        return None
    return keep[lab].view(bool)


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


def _markers(lo, ring, p, L, cv2):
    """White next-checkpoint marker(s): dark-hole candidates verified by NCC with the donut template."""
    clo = cv2.morphologyEx(lo, cv2.MORPH_CLOSE, _k(5))
    hole = cv2.subtract(clo, lo)
    cand = (hole >= p.marker_hole) & (clo >= p.marker_ring_lo) & ~L['marker_excl']
    ys, xs = np.nonzero(cand)
    if not len(ys):
        return ()
    if len(ys) > p.marker_max_cand:
        top = np.argpartition(-hole[ys, xs].astype(np.int16), p.marker_max_cand)[:p.marker_max_cand]
        ys, xs = ys[top], xs[top]
    order = np.argsort(-hole[ys, xs].astype(np.int16), kind='stable')
    T = L['marker']
    h2 = T.shape[0] // 2
    out = []
    for k in order:
        cx, cy = int(xs[k]), int(ys[k])
        if any(abs(cx - u) <= 3 and abs(cy - v) <= 3 for u, v in out):
            continue
        best = (-1.0, cx, cy)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                u, v = cx + dx, cy + dy
                if u - h2 < 0 or v - h2 < 0 or u + h2 + 1 > FRAME_W or v + h2 + 1 > FRAME_H:
                    continue
                patch = lo[v - h2:v + h2 + 1, u - h2:u + h2 + 1].astype(np.float32)
                patch -= patch.mean()
                nrm = float(np.sqrt((patch * patch).sum()))
                if nrm < 1e-3:
                    continue
                c = float((patch * T).sum()) / nrm
                if c > best[0]:
                    best = (c, u, v)
        c, u, v = best
        if c < p.marker_ncc:
            continue
        ya, yb, xa, xb = max(0, v - 4), min(FRAME_H, v + 5), max(0, u - 4), min(FRAME_W, u + 5)
        win = lo[ya:yb, xa:xb].astype(np.float32)
        yy, xx = np.mgrid[ya:yb, xa:xb]
        d = np.hypot(xx - u, yy - v)
        outer = win[d >= 3.6]
        if outer.size and win[(d >= 1.5) & (d <= 2.6)].mean() - float(np.median(outer)) < p.marker_contrast:
            continue
        out.append((u, v))
        cv2.circle(ring, (u, v), p.marker_radius, 1, -1)
    return tuple(((u + 0.5) / FRAME_W, (v + 0.5) / FRAME_H) for u, v in out)


def overlay_masks(rgb: np.ndarray, params: Params = DEFAULT_PARAMS) -> OverlayMasks:
    """Masks for one uint8 (252, 448, 3) RGB frame."""
    cv2 = _cv2()
    rgb = np.asarray(rgb)
    if rgb.shape != (FRAME_H, FRAME_W, 3) or rgb.dtype != np.uint8:
        raise ValueError(f'overlay_masks expects a ({FRAME_H}, {FRAME_W}, 3) uint8 RGB frame '
                         f'(use to_model_frame), got {rgb.shape} {rgb.dtype}')
    p = params
    L = layout()
    r, g, b = cv2.split(np.ascontiguousarray(rgb))
    hi = cv2.max(cv2.max(r, g), b)
    lo = cv2.min(cv2.min(r, g), b)
    sat = cv2.subtract(hi, lo)

    # --- HUD ------------------------------------------------------------------------------------------
    hud = L['static_u8'].copy()
    for ys, xs in L['text']:
        gm = _glyphs(lo[ys, xs], hi[ys, xs], p.glyph_strong, p.glyph_strong_hi, p, cv2)
        if gm is not None:
            hud[ys, xs] |= gm.view(np.uint8)
    for ys, xs in (L['timer'], L['rec']):
        rr, gg, bb = r[ys, xs], g[ys, xs], b[ys, xs]
        col = (((rr > 150) & (cv2.subtract(rr, gg) > 60) & (cv2.subtract(rr, bb) > 50))
               | ((gg > 140) & (cv2.subtract(gg, rr) > 50) & (cv2.subtract(gg, bb) > 20)))
        if col.any():
            hud[ys, xs] |= cv2.dilate(col.view(np.uint8), _K3)
    rank_rows = _rank_list(lo, sat, hud, L, p, cv2)
    for (ys, xs), mx in [(L['horizon'], p.dot_max_px)] + [(s, p.stick_dot_max_px) for s in L['stick']]:
        sl = lo[ys, xs]
        th = cv2.morphologyEx(sl, cv2.MORPH_TOPHAT, _k(p.dot_tophat))
        dots = (th >= p.dot_strong) & (sl >= p.dot_lo) & (sat[ys, xs] <= p.glyph_chroma_max) & ~L['static'][ys, xs]
        if dots.any():
            keep = _small_components(dots, mx, cv2)
            if keep is not None:
                hud[ys, xs] |= cv2.dilate(keep.view(np.uint8), _K3)
    hud_b = hud.view(bool)

    # --- white next-checkpoint marker and mint checkpoint volumes -------------------------------------
    ring = np.zeros((FRAME_H, FRAME_W), np.uint8)
    rings_uv = _markers(lo, ring, p, L, cv2)
    gr = cv2.subtract(g, r)
    gb = g.astype(np.int16) - b
    strong = (gr >= p.mint_gr) & (gb >= p.mint_gb) & (hi >= p.mint_v)
    if strong.any():
        strong &= (sat.astype(np.uint16) * 255) >= (p.mint_s * hi.astype(np.uint16))
        if strong.any():
            weak = (gr >= p.mint_weak_gr) & (gb >= p.mint_weak_gb) & (hi >= p.mint_v)
            ring |= (strong | (weak & cv2.dilate(strong.view(np.uint8), _K3).view(bool))).view(np.uint8)
    ring_b = ring.view(bool) & ~hud_b

    # --- ghost trails ---------------------------------------------------------------------------------
    o1 = cv2.addWeighted(r, 0.5, g, -0.5, 128.0)                                        # red - green
    o2 = cv2.addWeighted(cv2.addWeighted(r, 0.25, g, 0.25, 0.0), 1.0, b, -0.5, 128.0)   # yellow - blue
    k = _k(p.ghost_k)
    resp = cv2.max(cv2.max(cv2.morphologyEx(o1, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(o1, cv2.MORPH_BLACKHAT, k)),
                   cv2.max(cv2.morphologyEx(o2, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(o2, cv2.MORPH_BLACKHAT, k)))
    ghost = np.zeros((FRAME_H, FRAME_W), bool)
    cand = (resp >= p.ghost_weak) & (hi >= p.ghost_v_min) & ~(hud_b | ring_b | L['text_region'])
    if cand.any():
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand.view(np.uint8), connectivity=8)
        if n > 1:
            strong_px = cand & (resp >= p.ghost_strong) & (sat >= p.ghost_s_min)
            has_strong = np.bincount(lab[strong_px], minlength=n) > 0
            diag = np.hypot(st[:, 2], st[:, 3]).astype(np.float32)
            width = st[:, 4] / np.maximum(diag, 1.0)
            keep = (diag >= p.ghost_len_min) & (width <= p.ghost_width_max) & has_strong
            keep[0] = False
            if keep.any():
                ghost = keep[lab]
    return OverlayMasks(hud=hud_b, ring=ring_b, ghost=ghost, propeller=L['propeller'].copy(),
                        rings_uv=rings_uv, rank_rows=rank_rows)


def validity_channel(masks: OverlayMasks) -> np.ndarray:
    v = np.full(masks.hud.shape, VALID_SCENE, np.float32)
    v[masks.propeller] = VALID_PROPELLER
    v[masks.masked] = VALID_MASKED
    return v


def masked_input(rgb: np.ndarray, masks: OverlayMasks) -> np.ndarray:
    out = np.array(rgb, dtype=np.uint8, copy=True)
    out[masks.masked] = IMAGENET_MEAN_U8
    return out
