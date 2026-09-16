"""Liftoff's own next-checkpoint beacon, read out of the frames as automatic gate ground truth.

While a track is loaded the game draws an emissive marker at the checkpoint the pilot has to fly
through next: a saturated green cone under a pale shade, standing at the gate. It is the one thing in
the picture whose world position is exactly a gate's, and it is far easier to find than an arch,
because it is a colour no forest, field or sky produces. Detecting it in every frame of a posed
flight and intersecting the rays gives gate positions without a human reading pixels off contact
sheets, which is what `triangulate` otherwise needs.

Measured on Pine Valley, one command per dataset. pine1 (1392 frames): a marker in 415 of them, 702
candidate rays, 8 clusters, of which the checkpoint's is 273 rays at 0.37 m RMS. pine2 (2092 frames,
a different session, flown at right angles to pine1's approach): 575 frames, 1130 rays, and the
cluster at that same checkpoint is 42 rays at 0.29 m RMS. The two independent flights put the marker
0.38 m apart - that, not agreement with a hand reading, is the real accuracy check, because their
viewing directions are perpendicular and each one's weak axis is the other's strong one.

Against the arch centre hand-read off contact sheets it sits 1.66 m away, and the decomposition says
what that is: 1.66 m along pine1's mean viewing direction and 0.05 m across it. pine1 looked at this
gate from a 6-degree-wide corridor of bearings, so depth is the one thing its own rays cannot fix.
Whether the remaining offset is the marker standing past the gate or the hand reading being short
along its own depth axis is not resolved by this data, which is why the recovered point is a
candidate to check and not a label to train on. Run `vision label` and `vision inspect` on it first.

WHAT IT ACTUALLY RECOVERS, AND WHAT YOU STILL HAVE TO DO

The colour finds every emissive green marker on the track, not only the beacon. Pine Valley also has
green ground light-strips - flat glowing quads lying on the trail - and measurement says they are not
separable from the beacon frame by frame: same colour, overlapping sizes (beacon median 16 px, strips
5-27 px), overlapping shapes. So the detector deliberately emits ALL admissible green blobs as
candidate rays and lets `gates_from_beacons` sort them out: each static marker, beacon or strip,
collapses into its own tight cluster, and the output is a handful of 3D points with an inlier count
and an RMS each. Which of them is a gate is then one `vision label` + `vision inspect` away - an arch
either is or is not at that position - instead of an afternoon of contact sheets. That is the honest
claim: it does not remove the human, it removes the pixel reading. On pine1 the checkpoint's cluster
was the largest of the eight, which is a useful hint and not a rule.

The beacon marks only the NEXT checkpoint. A flight that never reaches checkpoint 1 - a failed
by-sight attempt, a blind spiral - only ever shows one beacon, and this tool will only ever recover
one gate from it, however many frames it has. Recovering a whole course needs a flight that actually
progresses through it: fly the lap manually and the marker steps from gate to gate, the rays separate
into one cluster per gate, and the clusters come back in course order. That is a property of the
flight, not of the detector, and no amount of frames fixes it.

It is for bootstrapping ground truth and training labels, NOT a flight cue. The by-sight pilot must
find arches in the picture; steering to the game's own HUD marker would not be flying by sight, it
would be reading the answer off the screen, and every number measured that way would be a lie about
what the detector can do. Nothing in `haltere/brain` or the pilots may import this module.

WHAT MAKES A PIXEL A MARKER PIXEL

The markers are emissive, so they keep their brightness in forest shade, and they are green in a way
nothing natural here is:

  G > `g_min` (140)     - emissive. Sunlit foliage does reach this in the green channel, but only
                          when everything else is bright too, which the next two tests catch.
  G - R > `gr_min` (55) - the vegetation in these maps is yellow-green: R sits within ~40 of G on lit
                          leaves and grass. The markers' red channel is near zero.
  G - B > `gb_min` (25) - rejects the white-cyan core of the marker's own glow and the pale sky, both
                          of which have blue as high as green.

then a dilation to merge the pieces, then a size and shape gate on each blob:

  * the merge matters. The beacon renders as a cone, a shade and a bright core, and the colour test
    cuts it into two to four pieces a pixel or two apart. Without a 5x5 dilation before the connected
    components, the size floor throws away exactly the detections you want: measured on pine1, the
    frames in which the beacon is found go 134 -> 214 at the same floor.
  * `min_px` (12) is JPEG chroma noise and single-pixel speculars, which are most of the blobs in a
    frame. The beacon's merged blobs run 16 px at the median, so the floor costs it little.
  * `max_px` (900) is the only thing the light-strips fail on outright. Standing next to one fills up
    to 3234 px, and because it is an extended object its centroid slides metres along the strip
    between frames, so it does not just add a wrong ray, it adds a wrong ray that moves. The beacon
    never exceeded 866 px on pine1 - it is drawn at a near-constant screen size, 15-18 px median from
    1 m to 20 m, so range is no help but a hard ceiling is.
  * `max_elongation` (3.0) trims the slivers a strip projects to from most angles. Beacon blobs come
    in at 1.5 median, 2.3 at the 90th percentile; the worst strip, 1.9 and 4.6.

  The HUD is masked out by position instead: the leaderboard down the right-hand side and the plate
  along the top carry saturated green UI tiles that pass the colour test perfectly and never move, so
  a colour-only detector would happily triangulate a point from a user-interface element.
  `hud_right` and `hud_top` are for the 640x360 seat capture.

The camp tents and the countdown ring are not false positives here at all - they are grey and white,
and they only fool a human scanning contact sheets at thumbnail size, which is the failure this
module exists to avoid.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibrate import load_index
from .camera import Camera
from .triangulate import intersect_rays, ray_world


@dataclass
class BeaconParams:
    """Pixel criteria, justified in the module docstring. Defaults are for the 640x360 seat capture."""
    g_min: int = 140             # emissive green channel
    gr_min: int = 55             # green above red: not foliage
    gb_min: int = 25             # green above blue: not the glow core, not sky
    merge_px: int = 5            # dilation that puts the cone, shade and core back together
    min_px: int = 12             # below this it is chroma noise
    max_px: int = 900            # above this it is a ground light-strip, whose centroid slides
    max_elongation: float = 3.0  # strips project to slivers; the beacon's bounding box is near square
    hud_right: int = 560         # leaderboard: mask x >= this
    hud_top: int = 60            # timer/name plate: mask y < this


def marker_mask(rgb: np.ndarray, p: BeaconParams = BeaconParams()) -> np.ndarray:
    """Boolean mask of marker-coloured pixels in an RGB uint8 image, with the HUD masked out."""
    a = np.asarray(rgb)
    r = a[..., 0].astype(np.int16)
    g = a[..., 1].astype(np.int16)
    b = a[..., 2].astype(np.int16)
    m = (g > p.g_min) & (g - r > p.gr_min) & (g - b > p.gb_min)
    m[:, p.hud_right:] = False
    m[: p.hud_top, :] = False
    return m


def markers_in_frame(rgb: np.ndarray, p: BeaconParams = BeaconParams()) -> list[tuple[float, float, int]]:
    """Every admissible green marker blob in an RGB uint8 image, as (u, v, pixel count), largest first.

    All of them, not just the biggest: on Pine Valley the beacon is smaller than a nearby light-strip
    about as often as it is larger, so picking one per frame throws the beacon away in half the frames
    it appears in (measured: at best 49% of the picked blobs were the beacon). Sorting the candidates
    out is the RANSAC's job, and it is much better at it.
    """
    import cv2
    m = marker_mask(rgb, p).astype(np.uint8)
    if not m.any():
        return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.merge_px, p.merge_px))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, k, iterations=1), connectivity=8)
    out = []
    for i in range(1, n_lab):
        _, _, w, h, _ = stats[i]
        sel = (lab == i) & (m > 0)
        area = int(sel.sum())
        if not (p.min_px <= area <= p.max_px):
            continue
        if max(w, h) > p.max_elongation * max(1, min(w, h)):
            continue
        ys, xs = np.nonzero(sel)
        out.append((float(xs.mean()), float(ys.mean()), area))
    out.sort(key=lambda t: -t[2])
    return out


def detect_dataset(dataset: str | Path, p: BeaconParams = BeaconParams(), every: int = 1,
                   verbose: bool = True) -> list[dict]:
    """Candidate marker rays over a posed dataset: [{'file', 'u', 'v', 'n'}], several per frame allowed."""
    import cv2
    rows = load_index(dataset)[::every]
    frames = Path(dataset) / 'frames'
    out = []
    seen = 0
    for r in rows:
        img = cv2.imread(str(frames / r['file']))
        if img is None:
            continue
        hits = markers_in_frame(img[:, :, ::-1], p)
        seen += bool(hits)
        out.extend({'file': r['file'], 'u': u, 'v': v, 'n': n} for u, v, n in hits)
    if verbose:
        print(f'{dataset}: {len(rows)} frames, a green marker in {seen} '
              f'({100 * seen / max(len(rows), 1):.0f}%), {len(out)} candidate rays')
    return out


def gates_from_beacons(dataset: str | Path, detections: list[dict], cam: Camera, tol: float = 0.8,
                       min_inliers: int = 8, max_gates: int = 8, max_range_m: float = 60.0,
                       z_range: tuple[float, float] = (-3.0, 12.0), iters: int = 4000, seed: int = 0,
                       verbose: bool = True) -> list[dict]:
    """RANSAC the candidate rays into one position per static marker.

    A single least-squares intersection of every ray would be meaningless twice over: the checkpoint
    marker steps from gate to gate as the flight progresses, and most frames contribute a ray to a
    light-strip rather than to the beacon. Instead, repeatedly take the point supported by the most
    rays within `tol`, refit on its inliers, report how many there were and at what RMS, and remove
    them before looking for the next one. A recovery with four inliers at 3 m RMS then looks like what
    it is rather than like an answer, and the clusters that are strips rather than gates are visible
    as clusters you can check instead of as silent contamination of one number.

    Clusters come back in flight order (by the first frame that saw them); for a lap flown through the
    checkpoints, that is course order. Each is {'pos', 'heading', 'rms', 'n', 'gate', 'closest_m',
    'first_frame', 'last_frame', 'range_m', 'obs'}, where `obs` is the inlier detections.
    """
    rows = {r['file']: r for r in load_index(dataset)}
    order = list(rows)
    use = [d for d in detections if d['file'] in rows]
    if len(use) < 2:
        if verbose:
            print('fewer than two usable detections, nothing to recover')
        return []
    org, dirs = [], []
    for d in use:
        o, k = ray_world(rows[d['file']], float(d['u']), float(d['v']), cam)
        org.append(o)
        dirs.append(k)
    org = np.asarray(org)
    dirs = np.asarray(dirs)
    pos_all = np.array([[rows[f]['px'], rows[f]['py'], rows[f]['pz']] for f in order])

    def residuals(pt: np.ndarray) -> np.ndarray:
        w = pt[None] - org
        return np.linalg.norm(w - (w * dirs).sum(1)[:, None] * dirs, axis=1)

    def cost(pt: np.ndarray, mask: np.ndarray) -> float:
        """MSAC: every ray pays its squared distance, capped at `tol`. Counting inliers instead lets a
        hypothesis sitting between two markers win by one ray while every ray it claims is 0.7 m out,
        which is exactly how two markers 7 m apart get merged into one wrong point."""
        r = residuals(pt)[mask]
        return float(np.minimum(r * r, tol * tol).sum() + (~mask).sum() * tol * tol)

    rng = np.random.default_rng(seed)
    live = np.ones(len(use), bool)
    gates: list[dict] = []
    while len(gates) < max_gates and live.sum() >= min_inliers:
        idx = np.nonzero(live)[0]
        best_c, best_p, best_n = np.inf, None, 0
        for _ in range(iters):
            i, j = rng.choice(idx, 2, replace=False)
            if dirs[i] @ dirs[j] > 0.9995:                 # parallel rays fix nothing
                continue
            pt, _ = intersect_rays(org[[i, j]], dirs[[i, j]])
            if np.linalg.norm(pt - org[i]) > max_range_m or not (z_range[0] < pt[2] < z_range[1]):
                continue
            c = cost(pt, live)
            if c < best_c:
                best_c, best_p = c, pt
                best_n = int(((residuals(pt) < tol) & live).sum())
        if best_p is None or best_n < min_inliers:
            break
        pt, rms = best_p, float('nan')
        for _ in range(6):
            m = (residuals(pt) < tol) & live
            pt, rms = intersect_rays(org[m], dirs[m])
        m = (residuals(pt) < tol) & live
        if m.sum() < min_inliers:
            break
        k = int(np.argmin(np.linalg.norm(pos_all - pt, axis=1)))
        k0, k1 = max(0, k - 5), min(len(order) - 1, k + 5)
        vel = pos_all[k1] - pos_all[k0]
        heading = float(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 1e-3 else 0.0
        hit = [use[t] for t in np.nonzero(m)[0]]
        dist = np.linalg.norm(pt[None] - org[m], axis=1)
        gates.append({'pos': pt.tolist(), 'heading': heading, 'rms': float(rms), 'n': int(m.sum()),
                      'closest_m': float(np.linalg.norm(pos_all[k] - pt)),
                      'first_frame': min(d['file'] for d in hit),
                      'last_frame': max(d['file'] for d in hit),
                      'range_m': [float(dist.min()), float(dist.max())],
                      'obs': hit})
        live &= ~m
    gates.sort(key=lambda g: g['first_frame'])
    for i, g in enumerate(gates):
        g['gate'] = i
    if verbose:
        for g in gates:
            print(f"cluster {g['gate']}: {g['n']} rays -> {np.round(g['pos'], 2)} (rms {g['rms']:.2f} m, "
                  f"heading {np.degrees(g['heading']):.0f} deg, seen {g['first_frame']}-{g['last_frame']} "
                  f"at {g['range_m'][0]:.1f}-{g['range_m'][1]:.1f} m)")
        print(f'{len(gates)} clusters from {len(use)} rays, {int(live.sum())} unassigned. '
              f'Not every cluster is a gate: check them with vision label + vision inspect.')
    return gates


def save_observations(gates: list[dict], path: str | Path) -> None:
    """Write the inlier rays of each cluster as a `vision triangulate` observations file, so the
    recovery can be re-run, edited or merged with hand-read observations by the same command."""
    obs = [{'file': d['file'], 'u': d['u'], 'v': d['v'], 'gate': g['gate'], 'n': d['n']}
           for g in gates for d in sorted(g['obs'], key=lambda x: x['file'])]
    Path(path).write_text(json.dumps(obs, indent=1) + '\n', encoding='utf-8')


def strip_obs(gates: list[dict]) -> list[dict]:
    """The clusters without their inlier lists (what belongs in a gates JSON)."""
    return [{k: v for k, v in g.items() if k != 'obs'} for g in gates]
