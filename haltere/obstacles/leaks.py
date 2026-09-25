"""E8 shortcut/leak perturbations for 448 x 252 store frames (offline evaluation only).

Each perturbation returns the perturbed frames and the grid cells where the predicted range is compared
with the unperturbed prediction (``evaluate.e8_leak_tests``; pass: median relative change < 5 %):

- ``ring_on_obstacle``: a checkpoint-ring stroke (Liftoff green-cyan, OpenCV hue ~85, the overlays'
  ring colour rule) painted around a near cell (label EXACT/UPPER <= 8 m, else the predictor's own
  unperturbed range <= 8 m); test cells = the ring disc. A route shortcut would read the obstacle as
  free because "the ring is there".
- ``ring_on_free_space``: the same ring around a far cell (label LOWER >= 20 m, else predicted >= 20 m).
- ``ring_removed``: the frame's own ring stroke (``ring_mask_fn``, e.g. overlays.overlay_masks().ring)
  inpainted away; test cells = cells within one cell of the stroke. Skipped without a ring mask.
- ``hud_glyph_scramble``: near-white glyph pixels inside the reticle/horizon, centre-line-marker and
  stick-display zones moved by a random offset (the glyphs encode attitude and stick input: a
  behaviour leak); test cells = the zones dilated by one cell.
- ``ghost_trails_inserted``: 2-4 thin saturated polylines (other racers' ghost trails); test cells =
  cells within one cell of the lines.
"""
from __future__ import annotations

import numpy as np

from .contract import GRID_H, GRID_W, IMAGE_H, IMAGE_W, PATCH_PX

RING_RGB = (60, 230, 200)
GHOST_COLOURS = ((255, 40, 200), (255, 150, 0), (240, 240, 0), (0, 200, 255), (255, 60, 60))
_SCALE = IMAGE_W / 1280.0
# survey-data boxes.json (1280 x 720 gameplay pixels): glyph zones the travel line crosses
HUD_ZONES_1280 = {
    'reticle_horizon': (500, 300, 780, 400),
    'centre_line_marker': (616, 400, 664, 720),
    'stick_display': (530, 610, 750, 720),
}
PERTURBATIONS = ('ring_on_obstacle', 'ring_on_free_space', 'ring_removed', 'hud_glyph_scramble',
                 'ghost_trails_inserted')


def _cv2():
    import cv2
    return cv2


def zone_px(name: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = HUD_ZONES_1280[name]
    return (int(np.floor(x0 * _SCALE)), int(np.floor(y0 * _SCALE)), int(np.ceil(x1 * _SCALE)),
            int(np.ceil(y1 * _SCALE)))


def cells_of_mask(mask, dilate_cells: int = 0) -> np.ndarray:
    """(252, 448) pixel mask -> (18, 32) cells containing any masked pixel (optionally dilated)."""
    m = np.asarray(mask, bool).reshape(GRID_H, PATCH_PX, GRID_W, PATCH_PX).any(axis=(1, 3))
    for _ in range(dilate_cells):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def cell_centre(r: int, c: int) -> tuple[int, int]:
    return int(c * PATCH_PX + PATCH_PX // 2), int(r * PATCH_PX + PATCH_PX // 2)


def paint_ring(frame, centre_xy, radius_px: int, thickness: int = 3, colour=RING_RGB):
    """Copy of an RGB frame with a ring stroke; returns (frame, stroke mask, disc mask)."""
    cv2 = _cv2()
    out = np.ascontiguousarray(frame).copy()
    stroke = np.zeros(out.shape[:2], np.uint8)
    cv2.circle(stroke, tuple(int(v) for v in centre_xy), int(radius_px), 1, int(thickness), lineType=cv2.LINE_AA)
    disc = np.zeros(out.shape[:2], np.uint8)
    cv2.circle(disc, tuple(int(v) for v in centre_xy), int(radius_px) + thickness, 1, -1)
    out[stroke > 0] = colour
    return out, stroke > 0, disc > 0


def remove_ring(frame, ring_mask):
    """Inpaint the given ring-stroke pixels (Telea)."""
    cv2 = _cv2()
    m = (np.asarray(ring_mask, bool)).astype(np.uint8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return cv2.inpaint(np.ascontiguousarray(frame), m, 3, cv2.INPAINT_TELEA)


def scramble_hud(frame, rng):
    """Move near-white glyph pixels inside the HUD glyph zones by a random offset; returns (frame, zone mask)."""
    out = np.ascontiguousarray(frame).copy()
    zones = np.zeros(out.shape[:2], bool)
    for name in HUD_ZONES_1280:
        x0, y0, x1, y1 = zone_px(name)
        x1, y1 = min(x1, IMAGE_W), min(y1, IMAGE_H)
        zones[y0:y1, x0:x1] = True
        z = out[y0:y1, x0:x1].astype(np.int16)
        lo, hi = z.min(-1), z.max(-1)
        white = (lo > 185) & (hi - lo < 60)
        if not white.any():
            continue
        bg = np.median(z[~white], axis=0) if (~white).any() else np.array([90, 90, 90])
        glyph = z[white].mean(axis=0)
        dy = int(rng.integers(-(y1 - y0) // 3, (y1 - y0) // 3 + 1))
        dx = int(rng.integers(-(x1 - x0) // 3, (x1 - x0) // 3 + 1))
        moved = np.roll(np.roll(white, dy, axis=0), dx, axis=1)
        z[white] = bg
        z[moved] = glyph
        out[y0:y1, x0:x1] = np.clip(z, 0, 255).astype(np.uint8)
    return out, zones


def insert_ghost_trails(frame, rng, n_min: int = 2, n_max: int = 4):
    """Draw thin saturated polylines like other racers' ghost trails; returns (frame, line mask)."""
    cv2 = _cv2()
    out = np.ascontiguousarray(frame).copy()
    mask = np.zeros(out.shape[:2], np.uint8)
    for _ in range(int(rng.integers(n_min, n_max + 1))):
        k = int(rng.integers(3, 7))
        x = np.sort(rng.uniform(0, IMAGE_W, k))
        y = rng.uniform(IMAGE_H * 0.3, IMAGE_H * 0.95, k)
        pts = np.stack([x, y], 1).astype(np.int32).reshape(-1, 1, 2)
        colour = tuple(int(v) for v in GHOST_COLOURS[int(rng.integers(len(GHOST_COLOURS)))])
        th = int(rng.integers(1, 3))
        cv2.polylines(out, [pts], False, colour, th, lineType=cv2.LINE_AA)
        cv2.polylines(mask, [pts], False, 1, th + 1, lineType=cv2.LINE_8)
    return out, mask > 0


def _pick_cell(cand, rng):
    rr, cc = np.nonzero(cand)
    if not len(rr):
        return None
    k = int(rng.integers(len(rr)))
    return int(rr[k]), int(cc[k])


def _candidates(base, lab, near: bool):
    """Near (<= 8 m) or far (>= 20 m) cells from labels when given, else from the unperturbed prediction."""
    cand = np.zeros((GRID_H, GRID_W), bool)
    if lab is not None:
        from .labels import LabelKind
        v, k = np.asarray(lab[0], np.float64), np.asarray(lab[1])
        if near:
            cand = ((k == LabelKind.EXACT) | (k == LabelKind.UPPER)) & (v <= 8.0)
        else:
            cand = (k == LabelKind.LOWER) & (v >= 20.0)
    if not cand.any():
        b = np.asarray(base, np.float64)
        cand = (b <= 8.0) if near else (b >= 20.0)
    cand = cand & np.isfinite(np.asarray(base, np.float64))
    cand[:2] = False                 # keep the ring inside the frame
    cand[-2:] = False
    cand[:, :2] = False
    cand[:, -2:] = False
    return cand


def perturb_batch(frames, base_grid, labels_grid, rng, *, ring_mask_fn=None) -> dict:
    """{name: (perturbed frames or None, test-cell mask (n, 18, 32))} for one batch of frames."""
    frames = np.asarray(frames)
    n = len(frames)
    out = {}
    for name, near in (('ring_on_obstacle', True), ('ring_on_free_space', False)):
        pert = frames.copy()
        cells = np.zeros((n, GRID_H, GRID_W), bool)
        for i in range(n):
            lab = None if labels_grid is None else (labels_grid[0][i], labels_grid[1][i])
            pick = _pick_cell(_candidates(base_grid[i], lab, near), rng)
            if pick is None:
                continue
            radius = int(rng.integers(14, 28))
            pert[i], _, disc = paint_ring(frames[i], cell_centre(*pick), radius)
            cells[i] = cells_of_mask(disc)
        out[name] = (pert, cells)
    if ring_mask_fn is not None:
        pert = frames.copy()
        cells = np.zeros((n, GRID_H, GRID_W), bool)
        for i in range(n):
            m = np.asarray(ring_mask_fn(frames[i]), bool)
            if m.any():
                pert[i] = remove_ring(frames[i], m)
                cells[i] = cells_of_mask(m, dilate_cells=1)
        out['ring_removed'] = (pert, cells) if cells.any() else (None, cells)
    else:
        out['ring_removed'] = (None, np.zeros((n, GRID_H, GRID_W), bool))
    pert = frames.copy()
    cells = np.zeros((n, GRID_H, GRID_W), bool)
    for i in range(n):
        pert[i], zones = scramble_hud(frames[i], rng)
        cells[i] = cells_of_mask(zones, dilate_cells=1)
    out['hud_glyph_scramble'] = (pert, cells)
    pert = frames.copy()
    cells = np.zeros((n, GRID_H, GRID_W), bool)
    for i in range(n):
        pert[i], lines = insert_ghost_trails(frames[i], rng)
        cells[i] = cells_of_mask(lines, dilate_cells=1)
    out['ghost_trails_inserted'] = (pert, cells)
    return out
