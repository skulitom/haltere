"""Image exclusions for the configured Liftoff FPV HUD and original drone.

These are image-layout masks, never course geometry. Bright neutral scene
features and green scenery may also be excluded. Missing features stay unknown.
Different HUD layouts/cameras/drone silhouettes require a new calibration.
"""
import numpy as np


def liftoff_geometry_mask(rgb):
    import cv2

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Use a uint8 RGB image')
    mask = np.full((180, 320), 255, np.uint8)
    # Text, rankings, stick display, framing marks and the visible propeller arcs.
    mask[:42] = 0
    mask[150:] = 0
    mask[80:108, 260:] = 0
    mask[:, :30] = mask[:, 290:] = 0
    mask[110:, :110] = mask[110:, 210:] = 0
    mask[43:137, 105:111] = mask[43:137, 209:215] = 0
    mask[87:94, 105:215] = 0
    height, width = rgb.shape[:2]
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colored = cv2.inRange(hsv, np.array([35, 75, 75]), np.array([100, 255, 255]))
    low, high = rgb.min(axis=2), rgb.max(axis=2)
    white = ((low > 175) & (high.astype(np.int16)-low < 55)).astype(np.uint8)*255
    for excluded, radius in [(colored, 4), (white, 11)]:
        size = 2*max(1, round(radius*width/640))+1
        mask[cv2.dilate(excluded, np.ones((size, size), np.uint8)) > 0] = 0
    return mask
