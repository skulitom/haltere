"""Per-frame overlay masks and the model's validity channel (runtime-safe). DELIVERED BY THE OVERLAYS AGENT.

Contract (448 x 252 RGB uint8 store/runtime frames; runtime resizes its 640 x 360 frame with
cv2.INTER_AREA first):

- ``hud``: fixed HUD boxes (survey overlay_mask_fixed_hud_1280x720.png, scaled) MINUS the centre-line
  marker box (x 616-664, y 400-720 at 1280 x 720) and the horizon/stick boxes, which are replaced by
  per-frame glyph detectors (marker circle, reticle/horizon dots, stick dots and crosshairs; near-white
  or static pixels inside those boxes). The travel line passes through these boxes in 65-81 % of
  frames, so whole-box masking is not allowed there.
- ``ring``: the next-checkpoint ring STROKE and translucent cyan checkpoint volumes (HSV hue 35-100,
  S > 75, V > 75 rule, stroke only; the ring interior stays visible because the ring is drawn
  through obstacles).
- ``ghost``: other racers' ghost trails and ghost drones (thin saturated lines, S > 120, V > 120, plus
  dilation). Removed from input AND from every label/loss; never a cue.
- ``propeller``: the conservative propeller wedge zone (overlay_mask_propellers_1280x720.png). Not
  deleted: marked 0.5 in the validity channel so the model learns to discount it.

Model input: ``masked_input`` sets hud|ring|ghost pixels to IMAGENET_MEAN_U8 (0 after normalisation)
and ``validity_channel`` returns float32 (252, 448): 0 masked, 0.5 propeller zone, 1 scene.
Budget <= 2 ms CPU per frame. Mask assets are copied into configs/obstacles/ (tracked PNGs).

Pass criteria (plan M1, 200 hand-checked frames): >= 95 % of overlay pixels removed, <= 2 % of scene
pixels removed, ghost-trail recall >= 90 %, travel-line column kept in >= 90 % of frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN_U8 = (124, 116, 104)
VALID_SCENE, VALID_PROPELLER, VALID_MASKED = 1.0, 0.5, 0.0


@dataclass
class OverlayMasks:
    hud: np.ndarray        # bool (252, 448)
    ring: np.ndarray       # bool
    ghost: np.ndarray      # bool
    propeller: np.ndarray  # bool (zone, not deleted)

    @property
    def masked(self) -> np.ndarray:
        return self.hud | self.ring | self.ghost


def overlay_masks(rgb: np.ndarray) -> OverlayMasks:
    """Masks for one uint8 (252, 448, 3) RGB frame."""
    raise NotImplementedError('overlays: delivered by the overlays build agent')


def validity_channel(masks: OverlayMasks) -> np.ndarray:
    v = np.full(masks.hud.shape, VALID_SCENE, np.float32)
    v[masks.propeller] = VALID_PROPELLER
    v[masks.masked] = VALID_MASKED
    return v


def masked_input(rgb: np.ndarray, masks: OverlayMasks) -> np.ndarray:
    out = np.array(rgb, dtype=np.uint8, copy=True)
    out[masks.masked] = IMAGENET_MEAN_U8
    return out
