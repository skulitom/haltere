"""Read the visible next-checkpoint ring for the explicit race-cue runtime.

This is game-provided race guidance, not learned gate detection. It contains
no track coordinates or checkpoint list and is not a freestyle planner.
"""
import numpy as np


def checkpoint_ring(rgb):
    """Return normalized image position and whether the cue is edge-clamped.

    Liftoff draws a small thick white annulus. Its hole/area ratio separates
    it from the thin central reticle. Exclude timer, standings and stick HUD.
    Ambiguous frames are missing observations, never an arbitrary target.
    """
    import cv2
    frame = np.asarray(rgb)
    if frame.shape[:2] != (720, 1280):
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    lo, hi = np.minimum(r, np.minimum(g, b)), np.maximum(r, np.maximum(g, b))
    mask = ((lo > 190) & (hi.astype(np.int16)-lo < 55)).astype(np.uint8)*255
    contours, tree = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if tree is None:
        return None
    hits = []
    for i, contour in enumerate(contours):
        parent = tree[0, i, 3]
        if parent < 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # Find the hole, so a ring touching a white arch still has its centre.
        if not (7 <= w <= 11 and 7 <= h <= 11 and abs(w-h) <= 2):
            continue
        u, v = x+(w-1)/2, y+(h-1)/2
        edge = u < 30 or u > 1257 or v < 27 or v > 693
        if ((not edge and v < 165) or (1060 < u < 1257 and 230 < v < 565)
                or ((545 < u < 615 or 660 < u < 730) and 620 < v < 693)):
            continue
        hole, area = cv2.contourArea(contour), cv2.contourArea(contours[parent])
        if not (45 <= hole <= 85 and hole > .47*w*h and area > 2.1*hole):
            continue
        angles = np.arange(16)*2*np.pi/16
        xx = np.clip(np.rint(u+10*np.cos(angles)).astype(int), 0, 1279)
        yy = np.clip(np.rint(v+10*np.sin(angles)).astype(int), 0, 719)
        if np.mean(mask[yy, xx] > 0) > .5:  # a hole in a white cloud or wall
            continue
        hits.append(dict(u=u/1280, v=v/720, edge=bool(edge)))
    if len(hits) != 1:
        return None
    cue = hits[0]
    cue['aim_u'] = flag_clearance(frame, cue)
    return cue


def flag_clearance(rgb, cue):
    """Keep the approach ray beside a visible blue-tipped racing flag.

    The checkpoint ring may be drawn on a solid flag or behind it. Its centre
    alone is therefore not a collision-free goal. Use the flag's apparent size
    for a clearance margin and nearby emissive route arrows to choose the side.
    This is a local image heuristic, not general obstacle detection or a route.
    """
    import cv2
    u, v = cue['u']*1280, cue['v']*720
    if cue['edge']:
        return cue['u']
    r, g, b = np.asarray(rgb, dtype=np.int16).transpose(2, 0, 1)
    blue = ((5*b > 8*g) & (5*b > 7*r) & (b > 45)).astype(np.uint8)
    blue[:170] = 0
    blue[600:] = 0
    blue[230:565, 1060:] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(blue)
    green = (g > 140) & (g-r > 35) & (b-r > 20) & (g > b)
    green[:170] = False
    green[230:565, 1060:] = False
    neutral = ((np.maximum(r, np.maximum(g, b))-np.minimum(r, np.minimum(g, b)) < 35)
               & (r > 90) & (r-b > -3) & (r-g > -12))
    aim = u
    for x, y, w, h, area in stats[1:count]:
        if not (area >= 15 and 5 <= w <= 150 and 7 <= h <= 180
                and .35 < w/h < 1.7 and y+h < v < y+7*h+15):
            continue
        # Trace the neutral cloth below the coloured cap. Its slant can reverse
        # with viewing direction; do not assume the pole is on a fixed side.
        cursor = x+(w-1)/2
        points = []
        for row in np.linspace(y+h, min(v, y+4*h), 5).astype(int):
            x0, x1 = max(0, int(cursor-2*w)), min(1280, int(cursor+2*w)+1)
            strip = neutral[row:row+1, x0:x1].astype(np.uint8)
            strip = cv2.morphologyEx(strip, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8)).ravel()
            edges = np.diff(np.r_[0, strip, 0].astype(np.int16))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            centres = [x0+(a+b-1)/2 for a, b in zip(starts, ends)
                       if max(1, .15*w) <= b-a <= 2*w]
            if centres:
                candidate = min(centres, key=lambda c: abs(c-cursor))
                if abs(candidate-cursor) <= w:
                    cursor = candidate
                    points.append((row, cursor))
        post = cursor
        if len(points) >= 2 and points[-1][0] > points[0][0]:
            slope = np.clip((points[-1][1]-points[0][1])/(points[-1][0]-points[0][0]), -.8, .8)
            post += slope*max(0, v-points[-1][0])
        if abs(u-post) > 1.4*w+12:
            continue
        y0, y1 = max(170, int(v)-100), min(620, int(v)+100)
        x0, x1 = max(0, int(post)-400), min(1280, int(post)+400)
        split = int(np.clip(post, x0, x1))
        left = int(green[y0:y1, x0:split].sum())
        right = int(green[y0:y1, split:x1].sum())
        side = 1 if right > left else -1 if left > right else (1 if u >= post else -1)
        candidate = post+side*(2.5*w+25)
        if abs(candidate-u) > abs(aim-u):
            aim = candidate
    return float(np.clip(aim/1280, .03, .97))
