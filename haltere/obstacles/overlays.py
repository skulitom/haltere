"""Per-frame overlay masks and the model's validity channel (runtime-safe; imports numpy and cv2 only).

Input: one 448 x 252 RGB uint8 frame (store frames are cv2.INTER_AREA resizes of the 1280 x 720 gameplay
crop or of the 640 x 360 capture; ``to_model_frame`` does that resize for runtime frames). Output:
``OverlayMasks`` with four boolean (252, 448) masks:

- ``hud``: Liftoff HUD glyphs, pixel level. (a) A fixed-glyph layer: pixels that are near-white in >= 50 %
  of 300 dark-scene frames (ACRO badge, ALT/VIT/km/h labels, framing-column dashes, reticle circle,
  stick-display crosshairs, TOUR label, Rec. dot), stored at 1280 x 720 in
  ``configs/obstacles/overlay_hud_static_1280x720.png`` and area-resized per input size. (b) Per-frame
  glyphs inside the HUD element zones (timer / lap / delta line, compass tape, ALT/VIT digits, ranking
  list text, crash pictogram, Rec.): bright top-hat pixels plus the red/green delta digits, dilated by
  one pixel; ranking-list icons are masked as whole squares on rows whose name text is present. (c) The
  dotted horizon bars (they move with pitch and roll) and the stick-position dots: small bright
  components inside their zones only. The zones are never masked as whole boxes: the travel line crosses
  the centre-line and stick zones in 65-81 % of frames.
- ``ring``: (a) the white next-checkpoint ring (a thick annulus about 7 px across at 448, drawn at the
  checkpoint or clamped to the image edge), found anywhere by a compact-bright-blob detector and masked as
  a disc; (b) translucent checkpoint discs and volume stripes by colour (mint/cyan, HSV hue 68-100,
  S >= RING_S_MIN, V >= RING_V_MIN, excluding the near-white horizon glow of the Drawing Board sky).
  Only coloured pixels are masked: the interior of a volume stays visible.
- ``ghost``: other racers' ghost trails (thin coloured curves: red, pink, orange, yellow, green, blue,
  purple) found as elongated components of a Lab colour top-hat/black-hat response. Removed from the
  input AND from every loss; never a cue. White ghost trails and ghost drones are NOT detected (see
  ``KNOWN_GAPS``).
- ``propeller``: the propeller-blur wedges of the original [Copy] New Drone, derived from the temporal
  static-frequency map of 42k frames (low static frequency = spinning blur) plus a margin, stored in
  ``configs/obstacles/overlay_propeller_wedge_1280x720.png``. Not deleted: 0.5 in the validity channel.

Model input: ``masked_input`` sets hud|ring|ghost pixels to IMAGENET_MEAN_U8 (0 after normalisation) and
``validity_channel`` returns float32 (252, 448): 0 masked, 0.5 propeller zone, 1 scene.

Budget: <= 2 ms CPU per frame at 448 x 252 with 2 OpenCV threads (``tests/test_obstacle_overlays.py``).
Validation (offline, hand-checked ground truth on frames of every non-sealed environment) is described
in the M1 overlay report; this module holds one parameter set and no per-environment tuning.
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
SOURCE_W, SOURCE_H = 1280, 720          # the gameplay layout the assets are defined in
COMPOSITE_GAMEPLAY_X = 648              # 1928 x 720 run videos: brain panel x < 648, gameplay x >= 648
ASSET_DIR = Path(__file__).resolve().parents[2] / 'configs' / 'obstacles'
STATIC_HUD_ASSET = 'overlay_hud_static_1280x720.png'        # 255 = fixed HUD glyph pixel
PROPELLER_ASSET = 'overlay_propeller_wedge_1280x720.png'    # 255 = propeller zone
SURVEY_ASSETS = ('overlay_mask_fixed_hud_1280x720.png',     # survey boxes (255 = usable), provenance only
                 'overlay_mask_propellers_1280x720.png')

# HUD element zones in 1280 x 720 gameplay pixels (x0, y0, x1, y1). Glyphs are detected per frame inside.
TEXT_ZONES = {
    'acro': (20, 32, 116, 76),
    'timer': (515, 14, 785, 88),           # race timer, TOUR n/N lap clock, red/green personal-best delta
    'compass': (495, 100, 785, 156),
    'altvit': (1125, 32, 1280, 156),
    'rank': (1040, 256, 1280, 492),        # ghost-racer ranking list (names + icons)
    'crash': (1175, 640, 1280, 720),       # crash/reset pictogram
    'rec': (0, 672, 100, 720),             # red Rec. indicator
}
DOT_ZONES = {
    'horizon': (470, 180, 810, 540),       # dotted horizon bars: small dots only (they move with attitude)
    'stick': (525, 605, 755, 720),         # stick-position dots (the crosshairs are in the static layer)
}
RANK_ICON_X = (1237, 1271)                 # icon squares right of each ranking row
RANK_ROW_Y0, RANK_ROW_DY, RANK_ICON_HALF = 276.0, 33.5, 15.0
RANK_TEXT_X = (1090, 1236)

KNOWN_GAPS = (
    'white ghost trails (Ghosty Mc Ghostface) are not detected: a thin white line is indistinguishable from '
    'fence rails, arch edges and road markings at 448 px',
    'ghost drones (small translucent racer models) are only caught when their colour forms a thin curve',
    'transient centre texts (lap-time popup, countdown ring, LIFTOFF logo) are not masked; the store drops '
    'countdown/finish frames',
    'HUD glyphs drawn over near-white cloud are not detected (they are also nearly invisible)',
)


@dataclass(frozen=True)
class Params:
    # white HUD glyphs inside the text/dot zones (448 px units)
    glyph_tophat: int = 5          # top-hat structuring element (px)
    glyph_th_min: int = 28         # top-hat response on max(R,G,B)
    glyph_lo_min: int = 110        # min(R,G,B)
    glyph_chroma_max: int = 80     # max - min
    dot_max_px: int = 5            # horizon / stick dots: component bbox limit
    rank_row_min_px: int = 4       # glyph pixels on a ranking row that switch its icon square on
    # white next-checkpoint ring
    ring_tophat: int = 9
    ring_th_min: int = 45
    ring_lo_min: int = 150
    ring_min_px: int = 3
    ring_max_px: int = 9
    ring_radius: int = 4
    # mint/cyan checkpoint discs and volume stripes (OpenCV HSV)
    ring_h_min: int = 68
    ring_h_max: int = 100
    ring_s_min: int = 60
    ring_v_min: int = 80
    glow_v_min: int = 240          # near-white cyan horizon glow (Drawing Board sky) is scene ...
    glow_s_max: int = 100          # ... when V >= glow_v_min and S <= glow_s_max
    # ghost trails
    ghost_se: int = 9              # Lab a/b top-hat and black-hat structuring element
    ghost_resp_min: int = 12
    ghost_s_min: int = 60
    ghost_v_min: int = 50
    ghost_len_min: int = 12        # component bbox long side (px)
    ghost_width_max: float = 3.5   # component area / long side (px)


DEFAULT_PARAMS = Params()


@dataclass
class OverlayMasks:
    hud: np.ndarray        # bool (252, 448)
    ring: np.ndarray       # bool
    ghost: np.ndarray      # bool
    propeller: np.ndarray  # bool (zone, not deleted)
    rings_uv: tuple = field(default_factory=tuple)   # detected white-ring centres, normalised (u, v)

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
    sx, sy = w / SOURCE_W, h / SOURCE_H
    x0, y0, x1, y1 = b
    return (int(np.floor(x0 * sx)), int(np.floor(y0 * sy)), int(np.ceil(x1 * sx)), int(np.ceil(y1 * sy)))


def _load_asset(name: str) -> np.ndarray:
    cv2 = _cv2()
    path = ASSET_DIR / name
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (SOURCE_H, SOURCE_W):
        raise FileNotFoundError(f'overlay asset missing or wrong size: {path}')
    return img


@lru_cache(maxsize=4)
def layout(width: int = FRAME_W, height: int = FRAME_H) -> dict:
    """Fixed per-size layers: static HUD glyphs, propeller zone and zone masks (cached; read-only arrays).

    The 1280 x 720 assets are area-resized; a pixel is static HUD when >= 15 % of its area is glyph, and in
    the propeller zone when >= 50 % of its area is."""
    cv2 = _cv2()
    static = cv2.resize(_load_asset(STATIC_HUD_ASSET).astype(np.float32) / 255.0, (width, height),
                        interpolation=cv2.INTER_AREA) >= 0.15
    prop = cv2.resize(_load_asset(PROPELLER_ASSET).astype(np.float32) / 255.0, (width, height),
                      interpolation=cv2.INTER_AREA) >= 0.5
    text = np.zeros((height, width), bool)
    for b in TEXT_ZONES.values():
        x0, y0, x1, y1 = _box(b, width, height)
        text[y0:y1, x0:x1] = True
    dots = np.zeros((height, width), bool)
    for b in DOT_ZONES.values():
        x0, y0, x1, y1 = _box(b, width, height)
        dots[y0:y1, x0:x1] = True
    sx, sy = width / SOURCE_W, height / SOURCE_H
    rows = []
    for k in range(7):
        yc = RANK_ROW_Y0 + RANK_ROW_DY * k
        ya, yb = int(round((yc - RANK_ICON_HALF) * sy)), int(round((yc + RANK_ICON_HALF) * sy))
        rows.append((ya, yb))
    out = dict(static=static, propeller=prop, text=text, dots=dots, rank_rows=tuple(rows),
               rank_icon_x=(int(np.floor(RANK_ICON_X[0] * sx)), int(np.ceil(RANK_ICON_X[1] * sx))),
               rank_text_x=(int(np.floor(RANK_TEXT_X[0] * sx)), int(np.ceil(RANK_TEXT_X[1] * sx))),
               timer=_box(TEXT_ZONES['timer'], width, height), rec=_box(TEXT_ZONES['rec'], width, height))
    for v in out.values():
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return out


_K3 = np.ones((3, 3), np.uint8)


def _small_components(mask_u8, max_px):
    cv2 = _cv2()
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return np.zeros(mask_u8.shape, bool)
    keep = np.zeros(n, bool)
    keep[1:] = (st[1:, 2] <= max_px) & (st[1:, 3] <= max_px)
    return keep[lab]


def overlay_masks(rgb: np.ndarray, params: Params = DEFAULT_PARAMS) -> OverlayMasks:
    """Masks for one uint8 (252, 448, 3) RGB frame."""
    cv2 = _cv2()
    rgb = np.asarray(rgb)
    if rgb.shape != (FRAME_H, FRAME_W, 3) or rgb.dtype != np.uint8:
        raise ValueError(f'overlay_masks expects a ({FRAME_H}, {FRAME_W}, 3) uint8 RGB frame '
                         f'(use to_model_frame), got {rgb.shape} {rgb.dtype}')
    rgb = np.ascontiguousarray(rgb)
    p = params
    L = layout()
    hi = rgb.max(axis=2)
    lo = rgb.min(axis=2)
    chroma = cv2.subtract(hi, lo)

    # --- HUD glyphs -------------------------------------------------------------------------------
    th = cv2.morphologyEx(hi, cv2.MORPH_TOPHAT, np.ones((p.glyph_tophat, p.glyph_tophat), np.uint8))
    white = (th >= p.glyph_th_min) & (lo >= p.glyph_lo_min) & (chroma <= p.glyph_chroma_max)
    hud = L['static'].copy()
    hud |= white & L['text']
    # red / green personal-best delta in the timer zone, red Rec. dot
    for (x0, y0, x1, y1) in (L['timer'], L['rec']):
        sub = rgb[y0:y1, x0:x1].astype(np.int16)
        r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
        coloured = ((r > 150) & (r - g > 60) & (r - b > 50)) | ((g > 140) & (g - r > 50) & (g - b > 20))
        hud[y0:y1, x0:x1] |= coloured
    # ranking icons: whole squares on rows with name text
    tx0, tx1 = L['rank_text_x']
    ix0, ix1 = L['rank_icon_x']
    for ya, yb in L['rank_rows']:
        if int(np.count_nonzero(white[ya:yb, tx0:tx1])) >= p.rank_row_min_px:
            hud[ya:yb, ix0:ix1] = True
    dots = (white & L['dots']).view(np.uint8)
    if dots.any():
        hud |= _small_components(dots, p.dot_max_px)
    hud = cv2.dilate(hud.view(np.uint8), _K3).view(bool)

    # --- white next-checkpoint ring ------------------------------------------------------------------
    ring = np.zeros((FRAME_H, FRAME_W), np.uint8)
    thr = cv2.morphologyEx(hi, cv2.MORPH_TOPHAT, np.ones((p.ring_tophat, p.ring_tophat), np.uint8))
    cand = ((thr >= p.ring_th_min) & (lo >= p.ring_lo_min) & (chroma <= p.glyph_chroma_max)).view(np.uint8)
    rings_uv = []
    n, _, st, cen = cv2.connectedComponentsWithStats(cand, connectivity=8)
    for k in range(1, n):
        w, h, a = st[k, 2], st[k, 3], st[k, 4]
        if not (p.ring_min_px <= w <= p.ring_max_px and p.ring_min_px <= h <= p.ring_max_px
                and abs(int(w) - int(h)) <= 2 and a >= 0.35 * w * h):
            continue
        cx, cy = st[k, 0] + w / 2.0, st[k, 1] + h / 2.0
        if L['static'][min(int(cy), FRAME_H - 1), min(int(cx), FRAME_W - 1)]:
            continue            # the reticle and stick crosshairs are static HUD already
        cv2.circle(ring, (int(round(cx - 0.5)), int(round(cy - 0.5))), p.ring_radius, 1, -1)
        rings_uv.append((float(cx / FRAME_W), float(cy / FRAME_H)))

    # --- mint / cyan checkpoint discs and volumes ----------------------------------------------------------
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mint = ((H >= p.ring_h_min) & (H <= p.ring_h_max) & (S >= p.ring_s_min) & (V >= p.ring_v_min)
            & ~((V >= p.glow_v_min) & (S <= p.glow_s_max)))
    ring |= cv2.dilate(mint.view(np.uint8), _K3)
    ring = ring.view(bool) & ~hud

    # --- ghost trails ------------------------------------------------------------------------------------------
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    se = np.ones((p.ghost_se, p.ghost_se), np.uint8)
    resp = None
    for ch in (1, 2):
        c = np.ascontiguousarray(lab[..., ch])
        t = cv2.max(cv2.morphologyEx(c, cv2.MORPH_TOPHAT, se), cv2.morphologyEx(c, cv2.MORPH_BLACKHAT, se))
        resp = t if resp is None else cv2.max(resp, t)
    cand = ((resp >= p.ghost_resp_min) & (S >= p.ghost_s_min) & (V >= p.ghost_v_min) & ~hud & ~ring
            & ~L['text']).view(np.uint8)
    ghost = np.zeros((FRAME_H, FRAME_W), bool)
    n, lab_ids, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n > 1:
        length = np.maximum(st[:, 2], st[:, 3]).astype(np.float32)
        width = st[:, 4] / np.maximum(length, 1.0)
        keep = (length >= p.ghost_len_min) & (width <= p.ghost_width_max)
        keep[0] = False
        if keep.any():
            ghost = cv2.dilate(keep[lab_ids].view(np.uint8), _K3).view(bool) & ~hud

    return OverlayMasks(hud=hud, ring=ring, ghost=ghost, propeller=L['propeller'].copy(),
                        rings_uv=tuple(rings_uv))


def validity_channel(masks: OverlayMasks) -> np.ndarray:
    v = np.full(masks.hud.shape, VALID_SCENE, np.float32)
    v[masks.propeller] = VALID_PROPELLER
    v[masks.masked] = VALID_MASKED
    return v


def masked_input(rgb: np.ndarray, masks: OverlayMasks) -> np.ndarray:
    out = np.array(rgb, dtype=np.uint8, copy=True)
    out[masks.masked] = IMAGENET_MEAN_U8
    return out
