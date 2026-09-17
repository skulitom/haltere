"""Closed-loop by-sight rehearsal in the simulator: the real by-sight pilot flies the course without the game.

What runs is what flies in Liftoff, minus the game and GateNet: the trained brain, the real ``TelemetryPilot`` (the
world-anchored gate estimate of ``vision_goal``, the carrot along the approach line, the altitude band, the fly-on /
creep / search state machine and the face-travel yaw) fed with telemetry frames built from the simulator's state, a
simulated clock, and the brain's sticks going through the control latency into the simulated flight controller.

GateNet is replaced by ``SyntheticGateVision``: every arch of the course is projected into the FPV camera with the
drone's simulated attitude (the geometry of the dataset labels: visual centre 1.5 m above the passage point, 4 m
wide, 5% image margin, at most 45 m, at least 22 px wide at 640) and the detector reports the widest arch in view
(arches look the same from behind), with the failure modes of the real one (``DetectorModel``, measured on recorded
flights): centre noise, width noise with a range-dependent bias, frames at 15 Hz delivered ~80 ms after the pose they
show, single-frame misses and dropout bursts, arches seen at an angle or far away found less often, flips to the
second arch in view, the occasional phantom, and optionally arches seen at an angle looking narrower.

Not modelled: Liftoff's own physics (the simulator is the brain's training physics), the terrain (the ground is flat
at the start height, so the hill under gates 5 and 6 is missing), obstacles other than the arch posts and top bars,
and the stick mapping (the sticks go straight into the simulated flight controller in the brain's convention).
"""
from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..liftoff.frames import sim_quat_to_unity, sim_vec_to_unity
from ..liftoff.sightpilot import LOG_COLUMNS as SIGHT_LOG_COLUMNS
from ..liftoff.sightpilot import SightPilot
from ..liftoff.telemetry import TelemetryFrame
from .camera import Camera, quat_wxyz_to_mat, world_to_body
from .gates import CENTRE_UP_M, GATE_WIDTH_M, VIEW_MARGIN
from .model import IN_H, IN_W
from .runtime import Detection, detection_geometry

MAX_RPM = 30000.0
LOG_COLUMNS = ['wall', 'ts', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz',
               'in_thr', 'in_yaw', 'in_pitch', 'in_roll', 'rpm', 'b_thr', 'b_roll', 'b_pitch', 'b_yaw',
               'c_thr', 'c_roll', 'c_pitch', 'c_yaw', 's_thr', 's_roll', 's_pitch', 's_yaw',
               'gx', 'gy', 'gz', 'tx', 'ty', 'tz', 'phase', 'crashed', 'det_p', 'det_w', 'det_age',
               # rehearsal only: which arch the synthetic detector reported (-1 none, -2 loose phantom,
               # -3 a clutter object), its pixel centre, the pilot's gate estimate (world) and the yaw override of
               # the search modes
               'det_gate', 'det_u', 'det_v', 'est_x', 'est_y', 'est_z', 'yaw_ovr',
               # the rabbit pilot's columns (haltere.liftoff.sightpilot.LOG_COLUMNS; empty for the legacy pilot)
               *[c for c in SIGHT_LOG_COLUMNS if c not in ('det_u', 'det_v')], 'status']


# ----------------------------------------------------------------------------- synthetic detector

@dataclass
class ClutterModel:
    """Things beside the course that are not arches and that the detector reports as if they were. OFF by default
    (``DetectorModel.clutter`` is None): every bench number taken without it stays comparable.

    Why this exists, and why ``DetectorModel.false_pos`` is not it. ``false_pos`` rolls an INDEPENDENT box per
    frame at a uniformly random place in the image, so two of them are at unrelated bearings and unrelated ranges:
    the tracker's association gate (``SightParams.perp_gate`` / ``log_gate_tent``) never lets them meet, no track
    ever reaches ``confirm_hits``, and nothing a phantom can do to the pilot is reachable. The real detector's
    false positives are COHERENT: it fires again and again on the same object standing in the world, the sightings
    triangulate on it, a track confirms, and that track can become the target. This models the object, not the box.

    Measured by replaying the six Straw Bale game flights (w22, w23, w27, w28, w29, w30) with ``liftoff
    replay-sight`` against the true gate list:

      * of every box the detector reported at p >= 0.5, 6.7-26.9 % sat on no arch at all (w30 6.7, w29 11.1,
        w23 12.1, w22 16.9, w28 17.8, w27 26.9); per detector frame, a box appeared on 4.2-13.9 % of the frames
        with no arch in view, and on 1.8-15.9 % of the frames that did have one the box was on something else;
      * they are coherent: 62-92 % of those false boxes fall into groups of three or more whose world points sit
        within 4 m of each other, and the largest single group is 64 sightings over 164 s while the drone moved
        40 m and saw the thing from 4.8 m to 25 m away -- a fixed object, seen from many places, placed at the
        same spot every time (which also says its apparent width is about that of a 4 m arch at its true range,
        or the estimate would have smeared along the ray as the drone closed on it);
      * they confirm: 5 to 17 confirmed phantoms per flight (a median of 6 per 100 s, on the clean laps as well
        as the broken ones), built from 4-8 sightings each, living 3-9 s, sitting 12-27 m from the drone and
        7-43 m from the nearest arch;
      * the boxes are confident (median p 0.77-0.98) and 15-56 px wide (median 29), which the pilot's 4 m width
        conversion reads as 6-27 m of range (median 12-22).

    The defaults are ``n_per_100m`` and ``fire`` set together so that an open-loop pass along Straw Bale at the
    pace the pilot keeps (the scripted-path harness of ``tests/test_rehearse.py``) lands inside every one of those
    measured spreads at once: 8 % of detector frames carry a false box (games 4.2-13.9 %), 11 % of the boxes
    reported are false (games 6.7-26.9 %), 4 % of the frames where an arch WAS found hand its box to an object
    instead (games 1.8-15.9 %), 71-77 % of the false boxes are placed on the same thing as two or more others
    (games 62-92 %; ``false_pos``, at ten times its shipped rate, manages 4-6 %), and the tracker confirms 6.2
    phantoms per 100 s (games 3.3-7.1) of a median 8 sightings each, the largest 18 (games: a median of 4-8, the
    largest 13-34).
    """
    n_per_100m: float = 3.0          # objects per 100 m of course line (Straw Bale is 205 m -> 6 of them)
    lateral_m: tuple = (4.0, 22.0)   # placed this far to either side of the course line ...
    min_arch_m: float = 8.0          # ... and never within this of an arch's centre (nearer than that the tracker
                                     # folds the sightings into the arch's own track, which is range error, not a
                                     # phantom: the measured phantoms sat 7-43 m from the nearest arch)
    up_m: tuple = (-0.5, 4.0)        # height above the course's own passage height where it stands
    size_m: tuple = (3.0, 5.5)       # apparent width, which is what the pilot's 4 m conversion ranges it by
    fire: float = 0.20               # per-frame probability that one object in view is reported
    range_m: tuple = (3.0, 35.0)     # only reported within this range (and above the detector's width floor)
    beats_arch: float = 0.35         # when an arch was found in the same frame, this often the object wins the box
    conf_log10: tuple = (-1.0, 0.8)  # confidence 1 - 10 ** normal(mu, sigma), clipped to [0.5, 0.9999]
    seed: int = 12345                # the objects are drawn from their own stream: the arches' noise is unchanged


def clutter_objects(gates: list[dict], model: ClutterModel, seed: int = 0,
                    start: tuple | None = None) -> np.ndarray:
    """Where the clutter stands: (n, 4) of x, y, z and apparent width, drawn along the course line.

    The course line is the spawn (when given) and then the arches in course order; an object sits at a random
    distance along it, offset to one side, at the height the course has there. Its own stream, so switching the
    clutter on does not shift the detector's noise for a seed."""
    pts = [np.asarray(g['pos'], dtype=np.float64) for g in gates]
    if start is not None:
        pts = [np.asarray([start[0], start[1], gates[0]['pos'][2]], dtype=np.float64)] + pts
    P = np.stack(pts)
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    length = float(cum[-1])
    n = int(round(model.n_per_100m * length / 100.0))
    if n <= 0 or length < 1e-6:
        return np.zeros((0, 4))
    rng = np.random.default_rng((int(seed), int(model.seed)))
    centres = np.array([p + [0.0, 0.0, CENTRE_UP_M] for p in (np.asarray(g['pos'], dtype=np.float64) for g in gates)])
    out = []
    for _ in range(40 * n):
        if len(out) >= n:
            break
        s = rng.uniform(0.0, length)
        k = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
        f = (s - cum[k]) / max(seg[k], 1e-6)
        base = P[k] + f * (P[k + 1] - P[k])
        d = P[k + 1, :2] - P[k, :2]
        side = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-6)
        off = rng.uniform(*model.lateral_m) * (1.0 if rng.random() < 0.5 else -1.0)
        xy = base[:2] + side * off
        z = base[2] + rng.uniform(*model.up_m)
        if float(np.hypot(centres[:, 0] - xy[0], centres[:, 1] - xy[1]).min()) < model.min_arch_m:
            continue
        out.append([xy[0], xy[1], max(z, 0.3), rng.uniform(*model.size_m)])
    return np.array(out, dtype=np.float64).reshape(-1, 4)


@dataclass
class DetectorModel:
    """What the synthetic GateNet gets wrong. Pixel quantities are in the network's 320 x 180 input frame.

    The defaults were measured with runs/gatenet8 on the by-sight flights run18 and run19 (4370 frames it was not
    trained on), against this module's projection of the arches with the recorded poses: an arch in view is found in
    80% of frames seen square on within 28 m, 61% at 20-35 deg off its axis, 47% at 35-50 deg, 32% beyond, and 44% at
    30-45 m; misses come mostly singly, sometimes in runs of 0.5-1 s; with two arches in view the detection sits on
    the second one in about 20% of frames; 3% of frames without any arch hold a phantom. The reported width follows
    the labels' (a vertical edge of the arch) with a range-dependent bias (0.75 x within 4 m, 1.0 x at 12 m, 1.12 x
    at 25 m, 1.5 x beyond 30 m, so far arches look near) and 17% log-normal spread; it does NOT narrow for arches
    seen at an angle (pred/label 1.04-1.06 up to 50 deg), hence ``oblique`` = 0. Centre scatter against the recorded
    poses was 4-14 px (it includes the pose/frame timing error of the recorder); validation frames gave 4 px."""
    rate_hz: float = 15.0            # detections per second (screen capture + inference)
    rate_jitter: float = 0.15        # +- fraction of the frame interval
    latency_s: float = 0.08          # from the pose a frame shows to the moment the pilot can read its detection
    latency_jitter_s: float = 0.015
    centre_px: float = 5.0           # centre noise (std, per axis)
    width_frac: float = 0.17         # width noise (log-normal std)
    width_bias: tuple = ((3.0, 0.75), (5.0, 0.93), (8.0, 0.95), (12.0, 0.98), (18.0, 1.08), (26.0, 1.12), (37.0, 1.48))
    oblique: float = 0.0             # width shrink of an arch seen at an angle: w * (1 - oblique * (1 - cos(angle)))
    miss: float = 0.2                # probability that a frame with an arch in view (square on, near) is missed
    oblique_fade: tuple = (15.0, 75.0)   # deg off the arch's axis: the find probability falls linearly between these ...
    far_fade: tuple = (28.0, 40.0)   # m: ... and halves between these distances (both floored at 0.3)
    burst_rate: float = 0.02         # per-frame probability of starting a dropout burst ...
    burst_s: tuple = (0.3, 1.0)      # ... lasting this long
    flip: float = 0.2                # probability of reporting the second widest arch in view instead of the widest
    false_pos: float = 0.03          # per-frame probability of a LOOSE phantom when no arch is detected: an
                                     # independent box at a random place in the image, which is what the real
                                     # detector's false positives are NOT (see ClutterModel). Such a box cannot
                                     # meet another one in the tracker's association gate, so it never confirms
                                     # and never reaches the pilot's choices.
    clutter: ClutterModel | None = None   # objects the detector keeps firing on (None = off, the default)
    max_dist_m: float = 45.0         # labelling limits (min width 22 px at 640 = 11 px at 320)
    min_width_px: float = 11.0

    @staticmethod
    def clean() -> "DetectorModel":
        """A perfect detector (still 15 Hz, still late): separates pilot problems from detector problems."""
        return DetectorModel(rate_jitter=0.0, latency_jitter_s=0.0, centre_px=0.0, width_frac=0.0, width_bias=(),
                             oblique=0.0, miss=0.0, oblique_fade=(), far_fade=(), burst_rate=0.0, flip=0.0,
                             false_pos=0.0, clutter=None)

    def find_probability(self, view_deg: float, dist_m: float) -> float:
        p = 1.0 - self.miss
        if self.oblique_fade:
            a0, a1 = self.oblique_fade
            p *= float(np.clip(1.0 - (view_deg - a0) / (a1 - a0), 0.3, 1.0))
        if self.far_fade:
            d0, d1 = self.far_fade
            p *= float(np.clip(1.0 - 0.5 * (dist_m - d0) / (d1 - d0), 0.3, 1.0))
        return p

    def reported_width(self, width_px: float, view_deg: float, dist_m: float) -> float:
        """Expected width before noise: the label's width, range bias, oblique shrink."""
        w = width_px * (1.0 - self.oblique * (1.0 - np.cos(np.radians(view_deg))))
        if self.width_bias:
            d, b = zip(*self.width_bias)
            w *= float(np.interp(dist_m, d, b))
        return float(w)


def arch_views(pos: np.ndarray, quat_wxyz: np.ndarray, gates: list[dict], cam: Camera, width_m: float = GATE_WIDTH_M,
               up_m: float = CENTRE_UP_M, max_dist_m: float = 45.0, min_width_px: float = 22.0) -> list[dict]:
    """Every arch as the camera sees it, with the geometry of ``gates.gate_label`` but without its next-gate rule
    (a detector sees arches it has passed, from behind, as well). Pixel units are those of ``cam``.
    Returns [{gate, visible, u, v, width_px, dist_m, view_deg}]; view_deg is the horizontal angle between the line
    of sight and the arch's axis (0 = seen square on, from either side)."""
    out = []
    pos = np.asarray(pos, dtype=np.float64)
    for i, g in enumerate(gates):
        centre = np.asarray(g['pos'], dtype=np.float64) + np.array([0.0, 0.0, up_m])
        h = float(g['heading'])
        side = np.array([-np.sin(h), np.cos(h), 0.0]) * width_m / 2
        up = np.array([0.0, 0.0, width_m / 2])
        corners = np.stack([centre + side + up, centre - side + up, centre - side - up, centre + side - up])
        pts_b = world_to_body(np.vstack([centre[None], corners]), pos, quat_wxyz)
        px, ok = cam.project_body(pts_b)
        dist = float(np.linalg.norm(pts_b[0]))
        los = (centre - pos)[:2]
        n = np.array([np.cos(h), np.sin(h)])
        cosv = abs(float(los @ n)) / max(float(np.linalg.norm(los)), 1e-6)
        view = {'gate': i, 'visible': 0, 'dist_m': dist, 'view_deg': float(np.degrees(np.arccos(min(cosv, 1.0))))}
        if ok.all() and dist >= 0.8:
            u, v = px[0]
            m = VIEW_MARGIN            # the same margin the labeller uses, or the rehearsal's detector and
            inside = (-m * cam.width <= u <= (1 + m) * cam.width      # the trained one disagree at the edge
                      and -m * cam.height <= v <= (1 + m) * cam.height)
            width_px = float(np.linalg.norm(px[1] - px[2]))
            view.update(u=float(u), v=float(v), width_px=width_px,
                        visible=int(inside and dist <= max_dist_m and width_px >= min_width_px))
        out.append(view)
    return out


class SyntheticGateVision:
    """Stands in for ``runtime.GateVision``: same ``get()`` / ``Detection``, driven by the rehearsal loop.

    ``update(now, pos, quat)`` is called every simulation step with the drone's pose: when a frame is due the arches
    are projected with that pose, and the detection is handed out ``latency_s`` later, stamped with the capture time
    (like the live detector, which stamps the screen grab)."""

    def __init__(self, gates: list[dict], cam: Camera, model: DetectorModel | None = None,
                 rng: np.random.Generator | None = None, width_m: float = GATE_WIDTH_M, up_m: float = CENTRE_UP_M,
                 start: tuple | None = None, clutter_seed: int = 0):
        self.gates = gates
        self.cam = cam.scaled(IN_W, IN_H)
        self.model = model or DetectorModel()
        self.rng = rng or np.random.default_rng(0)
        self.width_m, self.up_m = width_m, up_m
        self.stats = {'frames': 0, 'arch_in_view': 0, 'detected': 0, 'missed': 0, 'flipped': 0, 'phantoms': 0,
                      'clutter_in_view': 0, 'clutter': 0, 'clutter_over_arch': 0}
        self.clutter = (clutter_objects(gates, self.model.clutter, clutter_seed, start)
                        if self.model.clutter is not None else np.zeros((0, 4)))
        self.reset()

    def clutter_views(self, pos: np.ndarray, quat: np.ndarray) -> list[dict]:
        """Every clutter object as the camera sees it: [{obj, visible, u, v, width_px, dist_m}]. The apparent
        width is that of an object ``size_m`` wide at its range, with the same off-axis stretch
        ``runtime.detection_geometry`` divides out, so the range the pilot reads back is the object's own range
        times 4 m / size_m -- one fixed factor per object, which is what lets its sightings triangulate."""
        c, cam = self.model.clutter, self.cam
        if c is None or not len(self.clutter):
            return []
        pts_b = world_to_body(self.clutter[:, :3], pos, quat)
        px, ok = cam.project_body(pts_b)
        dist = np.linalg.norm(pts_b, axis=1)
        du, dv = px[:, 0] - cam.width / 2, px[:, 1] - cam.height / 2
        f = cam.f
        stretch = np.sqrt(f * f + du * du) * np.sqrt(f * f + du * du + dv * dv) / (f * f)
        wpx = f * self.clutter[:, 3] * stretch / np.maximum(dist, 1e-6)
        m = VIEW_MARGIN
        out = []
        for i in range(len(self.clutter)):
            u, v = float(px[i, 0]), float(px[i, 1])
            vis = (bool(ok[i]) and -m * cam.width <= u <= (1 + m) * cam.width
                   and -m * cam.height <= v <= (1 + m) * cam.height
                   and c.range_m[0] <= dist[i] <= c.range_m[1] and wpx[i] >= self.model.min_width_px)
            out.append({'obj': i, 'visible': int(vis), 'u': u, 'v': v, 'width_px': float(wpx[i]),
                        'dist_m': float(dist[i])})
        return out

    def reset(self, now: float | None = None) -> None:
        self.latest = Detection()
        self.latest_gate = -1
        self._pending: deque = deque()
        self._next_capture = now
        self._last_delivery = -np.inf
        self._burst_until = -np.inf
        self._n = 0

    def start(self) -> "SyntheticGateVision":
        return self

    def stop(self) -> None:
        pass

    def get(self) -> Detection:
        return self.latest

    def detect(self, now: float, pos: np.ndarray, quat: np.ndarray) -> tuple[Detection, int]:
        """One frame: (detection without its delivery time, what the box is on: the arch's index, -1 nothing,
        -2 a loose phantom, -3 a clutter object)."""
        m, rng, W, H = self.model, self.rng, self.cam.width, self.cam.height
        views = [v for v in arch_views(pos, quat, self.gates, self.cam, self.width_m, self.up_m, m.max_dist_m,
                                       m.min_width_px) if v['visible']]
        views.sort(key=lambda v: -v['width_px'])
        self.stats['frames'] += 1
        if views:
            self.stats['arch_in_view'] += 1
        if now < self._burst_until:
            blind = True
        elif m.burst_rate > 0 and rng.random() < m.burst_rate:
            self._burst_until = now + rng.uniform(*m.burst_s)
            blind = True
        else:
            blind = False
        p, u, v, w, gate = float(rng.uniform(0.01, 0.3)), W / 2, H / 2, 30.0, -1
        if views and not blind:
            pick = views[0]
            if len(views) > 1 and m.flip > 0 and rng.random() < m.flip:
                pick = views[1]
                self.stats['flipped'] += 1
            w_true = m.reported_width(pick['width_px'], pick['view_deg'], pick['dist_m'])
            if rng.random() < m.find_probability(pick['view_deg'], pick['dist_m']):
                # confident as a rule (median 0.998), now and then barely over the threshold (10th percentile 0.7)
                p = float(np.clip(1.0 - 10.0 ** rng.normal(-2.5, 1.55), 0.5, 0.99999))
                u = float(np.clip(pick['u'] + m.centre_px * rng.normal(), -0.1 * W, 1.1 * W))   # decode clamps +-1.2
                v = float(np.clip(pick['v'] + m.centre_px * rng.normal(), -0.1 * H, 1.1 * H))
                w = float(max(4.0, w_true * np.exp(m.width_frac * rng.normal())))
                gate = int(pick['gate'])
                self.stats['detected'] += 1
            else:
                self.stats['missed'] += 1
        elif views:
            self.stats['missed'] += 1
        c = m.clutter
        if c is not None and c.fire > 0 and len(self.clutter):
            seen = [v for v in self.clutter_views(pos, quat) if v['visible']]
            self.stats['clutter_in_view'] += len(seen)
            fired = [v for v in seen if rng.random() < c.fire]
            if fired and (gate == -1 or rng.random() < c.beats_arch):
                # the network reports one box: the widest thing it fired on, arch or not
                pick = max(fired, key=lambda v: v['width_px'])
                if gate >= 0:
                    self.stats['clutter_over_arch'] += 1      # an arch WAS found and the object took its box
                p = float(np.clip(1.0 - 10.0 ** rng.normal(*c.conf_log10), 0.5, 0.9999))
                u = float(np.clip(pick['u'] + m.centre_px * rng.normal(), -0.1 * W, 1.1 * W))
                v = float(np.clip(pick['v'] + m.centre_px * rng.normal(), -0.1 * H, 1.1 * H))
                w = float(max(4.0, pick['width_px'] * np.exp(m.width_frac * rng.normal())))
                gate = -3
                self.stats['clutter'] += 1
        if gate == -1 and m.false_pos > 0 and rng.random() < m.false_pos:
            # a phantom: hay bales and shadows on the ground in the lower part of the view
            p, u, v, w, gate = (float(rng.uniform(0.55, 0.85)), float(rng.uniform(0, W)), float(rng.uniform(0.45 * H, H)),
                                float(rng.uniform(10.0, 50.0)), -2)
            self.stats['phantoms'] += 1
        direction, dist = detection_geometry(self.cam, u, v, w)
        self._n += 1
        return Detection(0.0, p, u, v, w, direction, dist, self._n), gate

    def update(self, now: float, pos: np.ndarray, quat: np.ndarray) -> None:
        m = self.model
        if self._next_capture is None:
            self._next_capture = now
        if now >= self._next_capture - 1e-9:
            det, gate = self.detect(now, pos, quat)
            lat = max(0.01, m.latency_s + m.latency_jitter_s * self.rng.normal())
            deliver = max(self._last_delivery, now + lat)
            self._last_delivery = deliver
            det.t = now                                      # the moment the frame shows (the screen grab)
            self._pending.append((deliver, det, gate))
            period = 1.0 / m.rate_hz
            self._next_capture += period * (1.0 + m.rate_jitter * self.rng.uniform(-1, 1))
            if self._next_capture <= now:
                self._next_capture = now + period
        while self._pending and self._pending[0][0] <= now + 1e-9:
            deliver, det, gate = self._pending.popleft()
            self.latest, self.latest_gate = det, gate


# ----------------------------------------------------------------------------- the course: ground and arches

def arch_collision(p0: np.ndarray, p1: np.ndarray, gates: list[dict], post_band: tuple[float, float] = (1.7, 2.3),
                   top_band: tuple[float, float] = (3.2, 3.8)) -> tuple[int, str] | None:
    """Did the step from p0 to p1 cross an arch's plane where its frame is? A drone clipping a post (about 2 m
    either side of the passage point, from the ground below up to the top) or the top bar (3.5 m above the
    passage point) crashes. Returns (gate, 'post' | 'top') or None."""
    for i, g in enumerate(gates):
        gp = np.asarray(g['pos'], dtype=np.float64)
        h = float(g['heading'])
        n = np.array([np.cos(h), np.sin(h), 0.0])
        a0, a1 = float((p0 - gp) @ n), float((p1 - gp) @ n)
        if (a0 <= 0) == (a1 <= 0):
            continue
        f = a0 / (a0 - a1)
        pc = p0 + f * (p1 - p0)
        lat = abs(float((pc - gp) @ np.array([-np.sin(h), np.cos(h), 0.0])))
        dz = float(pc[2] - gp[2])
        if post_band[0] < lat < post_band[1] and -1.5 < dz < top_band[1]:
            return i, 'post'
        if lat <= post_band[0] and top_band[0] < dz < top_band[1]:
            return i, 'top'
    return None


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class SimClock:
    """The pilot's wall clock during a rehearsal (starts well away from zero, like time.time())."""

    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t


def frame_from_sim(ts: float, pos: np.ndarray, vel: np.ndarray, quat: np.ndarray, omega: np.ndarray,
                   motor: np.ndarray, sticks: np.ndarray, now: float) -> TelemetryFrame:
    """A Liftoff telemetry frame (Unity frame, x,y,z,w quaternion) of the simulated drone: the pilot's step()
    converts it back exactly as it converts the game's frames. Input is written in Liftoff's order."""
    return TelemetryFrame(timestamp=float(ts), position=sim_vec_to_unity(pos), attitude=sim_quat_to_unity(quat),
                          velocity=sim_vec_to_unity(vel), gyro=np.rad2deg([omega[1], omega[0], omega[2]]),
                          input=np.array([sticks[0], sticks[3], sticks[2], sticks[1]], dtype=np.float64),
                          battery=np.array([16.0, 0.9]), motor_rpm=np.asarray(motor, dtype=np.float64) * MAX_RPM,
                          recv_time=now)


# ----------------------------------------------------------------------------- the rehearsal loop

@dataclass
class RehearsalOptions:
    seconds: float = 120.0
    seed: int = 0
    face_travel: float = 0.8          # the fly command's --face-travel
    face_max: float = 0.25
    stick_gain: float | list = 1.0
    stick_lpf: float = 0.0
    delay_steps: int | None = None    # brain -> vehicle latency in control steps (default: the checkpoint's training value)
    arm_hold: float = 0.8             # the fly command's arming: throttle low, then the brain's sticks ramped in
    arm_ramp: float = 1.2
    start: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # x, y, z (m), yaw (deg) of the reset point
    collide: bool = True              # arch posts and top bars crash the drone
    crash_hold: float = 1.5           # s on the ground after a crash before the reset (a new attempt)
    pilot_set: dict = field(default_factory=dict)   # TelemetryPilot attributes to override (e.g. vision_speed)
    physics_jitter: float = 0.0       # +- fraction of mass/thrust/drag/motor-lag randomization (0 = nominal physics)
    sight: str = 'legacy'             # by-sight pilot: 'legacy' or 'rabbit' (haltere.liftoff.sightpilot)
    track_report: bool = False        # record every track and sighting (the rabbit only), so ``phantom_report``
                                      # can say which of them sat on no arch -- the analysis `replay-sight` prints
                                      # for a game flight, which cannot read a rehearsal log (its clock is too small)
    sight_params: object = None       # SightParams for the rabbit (None: defaults with the simulator's yaw rate)
    flow_gain: float = 1.0            # the fly command's --flow-gain (the rabbit's highest speed-sense gain)
    max_gpu_temp: float = 70.0        # C; pause (and wait for it to cool) above this; 0 = never check
    burst_s: float = 100.0            # wall seconds of GPU work between cool-down pauses (0 = no pauses)
    cool_s: float = 20.0
    verbose: bool = True


def run_rehearsal(brain, cfg, gates: list[dict], cam: Camera, log_path: str | Path | None = None,
                  opts: RehearsalOptions | None = None, detector: DetectorModel | None = None,
                  width_m: float = GATE_WIDTH_M, up_m: float = CENTRE_UP_M) -> dict:
    """Fly the by-sight pilot through the course in the simulator. Returns the raw per-step arrays and events;
    ``summarize`` turns them into a score. ``brain``/``cfg`` as returned by ``train.bptt.load_checkpoint``."""
    from ..liftoff.pilot import LiftoffMapping, TelemetryPilot
    from ..sim.quad import QuadState
    from ..sim.vehicle import Vehicle
    opts = opts or RehearsalOptions()
    rng = np.random.default_rng(opts.seed)
    torch.manual_seed(opts.seed)
    dt = float(cfg.brain.dt)
    vehicle = Vehicle(cfg.quad, cfg.ctl, cfg.rates, 'cpu', dt=dt, substeps=cfg.train.substeps)
    if opts.physics_jitter > 0:
        vehicle.sim.randomize(1, opts.physics_jitter, generator=torch.Generator().manual_seed(opts.seed))
    delay_steps = cfg.train.delay_steps if opts.delay_steps is None else opts.delay_steps
    clock = SimClock()
    pilot = TelemetryPilot(brain, cfg.task, LiftoffMapping(), brain.device, stick_gain=opts.stick_gain,
                           stick_lpf=opts.stick_lpf, face_gain=opts.face_travel, face_max=opts.face_max)
    pilot.clock = clock
    pilot.flow_gain = opts.flow_gain
    pilot.sight = opts.sight
    if opts.sight == 'rabbit':
        from ..liftoff.sightpilot import SightParams
        if opts.stick_lpf > 0:
            raise ValueError('the rabbit pilot does not run with a stick low-pass (stick_lpf)')
        sp_params = opts.sight_params or SightParams(yaw_rate=3.8)
        sp_params.flow_max = opts.flow_gain
        pilot.sight_params = sp_params
    vision = SyntheticGateVision(gates, cam, detector, rng, width_m, up_m, start=tuple(opts.start[:3]),
                                 clutter_seed=opts.seed)
    pilot.vision = vision
    rec = None
    if opts.track_report:
        if opts.sight != 'rabbit':
            raise ValueError("track_report records the rabbit pilot's tracks (sight='rabbit')")
        from ..liftoff.sightreplay import RecordingSightPilot
        rec = {'row': -1, 'epoch': 0, 'events': [], 'ends': [], 'sightings': [], 'cols': [], 'epochs': [],
               'snap': {k: [] for k in ('row', 'epoch', 'id', 'x', 'y', 'z', 'confirmed', 'passed', 'hits')},
               'cam': vision.cam}
        pilot.sightpilot = RecordingSightPilot(pilot, pilot.sight_params, rec)
    for k, v in opts.pilot_set.items():
        if not hasattr(pilot, k):
            raise ValueError(f'TelemetryPilot has no attribute {k!r}')
        setattr(pilot, k, type(getattr(pilot, k))(v) if isinstance(getattr(pilot, k), (int, float)) else v)
    start = np.array(opts.start[:3], dtype=np.float64)
    q_start = quat_from_yaw(np.radians(opts.start[3]))
    idle = np.array([-1.0, 0.0, 0.0, 0.0])

    def fresh_state():
        st = QuadState.hover(1, 'cpu', height=0.0)
        st.pos[0] = torch.as_tensor(start, dtype=torch.float32)
        st.quat[0] = torch.as_tensor(q_start, dtype=torch.float32)
        return vehicle.wrap(st)

    vs = fresh_state()
    delay = deque([idle.copy() for _ in range(delay_steps)])
    n_steps = int(round(opts.seconds / dt))
    rows = []
    events = []
    ts = 0.0
    attempt = 0
    airborne = False
    crashed_at = None
    applied = idle.copy()
    prev_passed_t = None
    prev_ts = 0.0
    use_gpu = brain.device.type == 'cuda'
    wall0 = burst0 = time.time()
    if use_gpu and opts.max_gpu_temp > 0:
        from ..train.thermal import wait_if_hot
        wait_if_hot(opts.max_gpu_temp)
    for k in range(n_steps):
        if use_gpu and opts.burst_s > 0 and time.time() - burst0 > opts.burst_s:
            if opts.verbose:
                print(f'  cool-down pause ({opts.cool_s:.0f} s) after {opts.burst_s:.0f} s of GPU work', flush=True)
            time.sleep(opts.cool_s)
            if opts.max_gpu_temp > 0:
                from ..train.thermal import wait_if_hot
                wait_if_hot(opts.max_gpu_temp)
            burst0 = time.time()
        q = vs.quad
        pos = q.pos[0].numpy().astype(np.float64)
        vel = q.vel[0].numpy().astype(np.float64)
        quat = q.quat[0].numpy().astype(np.float64)
        omega = q.omega[0].numpy().astype(np.float64)
        motor = q.motor[0].numpy().astype(np.float64)
        now = clock()
        vision.update(now, pos, quat)
        fr = frame_from_sim(ts, pos, vel, quat, omega, motor, applied, now)
        if rec is not None:
            rec['row'] = k
            if ts < prev_ts - 0.5:        # the same test TelemetryPilot.step resets on: the track ids start again
                rec['epoch'] += 1
        prev_ts = ts
        pilot.step(fr)
        if rec is not None:
            sp = pilot.sightpilot
            rec['cols'].append(sp.log_values(vision.latest))
            rec['epochs'].append(rec['epoch'])
            for T in sp.tracks:
                rec['snap']['row'].append(k)
                rec['snap']['epoch'].append(rec['epoch'])
                rec['snap']['id'].append(T.id)
                rec['snap']['x'].append(float(T.m[0]))
                rec['snap']['y'].append(float(T.m[1]))
                rec['snap']['z'].append(float(T.m[2]))
                rec['snap']['confirmed'].append(T.confirmed)
                rec['snap']['passed'].append(T.passed)
                rec['snap']['hits'].append(T.hits)
        cmd = pilot.last_cmd.copy()                       # [thr, roll, pitch, yaw] after gains, facing yaw, overrides
        sent = cmd.copy()
        if crashed_at is not None or ts < opts.arm_hold:
            sent = idle.copy()
        elif ts < opts.arm_hold + opts.arm_ramp:
            f = (ts - opts.arm_hold) / opts.arm_ramp
            sent[0] = -1.0 + f * (sent[0] + 1.0)
            sent[1:] *= f
        sent = np.clip(sent, -1.0, 1.0)
        delay.append(sent)
        applied = delay.popleft()
        new = vehicle.step(vs, torch.as_tensor(applied, dtype=torch.float32)[None])
        # ground contact (the simulator flags any contact as a crash): a take-off, a gentle touch-down or a drone
        # already down after a crash rests on the ground; a hard or tilted landing after flight is a crash
        p1 = new.quad.pos[0].numpy().astype(np.float64)
        event = None
        if bool(new.quad.crashed[0]):
            up = float(quat_wxyz_to_mat(quat)[2, 2])
            hard = airborne and (np.linalg.norm(vel) > 2.0 or up < np.cos(np.radians(45)))
            if hard and crashed_at is None:
                event = {'kind': 'crash_ground', 'speed': float(np.linalg.norm(vel)), 'tilt_deg': float(np.degrees(np.arccos(np.clip(up, -1, 1))))}
            qs = new.quad
            yaw = float(np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))
            qs.pos[0, 2] = 0.0
            qs.vel.zero_()
            qs.omega.zero_()
            qs.quat[0] = torch.as_tensor(quat_from_yaw(yaw), dtype=torch.float32)
            qs.crashed.zero_()
            new.ctl.iterm.zero_()
            airborne = False
        elif p1[2] > 0.3:
            airborne = True
        if opts.collide and crashed_at is None and event is None:
            hit = arch_collision(pos, p1, gates)
            if hit is not None:
                event = {'kind': f'crash_{hit[1]}', 'gate': hit[0]}
                new.quad.pos[0, 2] = 0.0
                new.quad.vel.zero_()
                new.quad.omega.zero_()
                airborne = False
        if event is not None:
            crashed_at = ts
            event.update(t=now, ts=ts, attempt=attempt, pos=[round(float(x), 2) for x in p1])
            events.append(event)
            if opts.verbose:
                print(f'  [{ts:6.2f} s] {event["kind"]} at {np.round(p1, 1)}' + (f' (gate {event["gate"]})' if 'gate' in event else ''), flush=True)
        if pilot.vision_passed_t is not None and pilot.vision_passed_t != prev_passed_t:
            passed = pilot._passed[-1][1] if pilot._passed else None
            events.append({'kind': 'pilot_pass', 't': now, 'ts': ts, 'attempt': attempt, 'pos': [round(float(x), 2) for x in pos],
                           'gate_est': None if passed is None else [round(float(x) + float(s), 2) for x, s in zip(passed, start)]})
        prev_passed_t = pilot.vision_passed_t
        det = vision.latest
        est = pilot.vision_gate_w + start if pilot.vision_gate_w is not None else None
        rows.append([now, ts, *pos, *vel, *quat, *pilot.omega, applied[0], applied[3], applied[2], applied[1],
                     float(motor.mean() * MAX_RPM), *pilot.last_brain, *cmd, *sent, *pilot.last_rel_b,
                     *(pilot.last_target + start), ts, int(crashed_at is not None), det.p_visible, det.width_px,
                     now - det.t if det.frames else np.nan, vision.latest_gate, det.u, det.v,
                     *(est if est is not None else (np.nan, np.nan, np.nan)),
                     np.nan if pilot._yaw_override is None else float(pilot._yaw_override),
                     *_sight_values(pilot, det, start),
                     pilot.vision_status.replace(',', ';')])
        vs = new
        ts += dt
        clock.t += dt
        if crashed_at is not None and ts - crashed_at > opts.crash_hold:
            # the fly command's reset: the drone is back at the reset point and the game clock restarts
            vs = fresh_state()
            delay = deque([idle.copy() for _ in range(delay_steps)])
            vision.reset(clock())
            applied = idle.copy()
            crashed_at, airborne, ts = None, False, 0.0
            attempt += 1
        if opts.verbose and k % 500 == 0:
            print(f'  t={k * dt:6.1f} s ts={ts:6.1f} pos={np.round(pos, 1)} speed={np.linalg.norm(vel[:2]):.1f} m/s | '
                  f'{pilot.vision_status}  [{time.time() - wall0:.0f} s wall]', flush=True)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(LOG_COLUMNS)
            for r in rows:
                w.writerow([_fmt(x) for x in r])
    return {'rows': rows, 'events': events, 'detector': dict(vision.stats), 'wall_s': time.time() - wall0,
            'steps': n_steps, 'dt': dt, 'delay_steps': delay_steps, 'rec': rec,
            'clutter': [[round(float(x), 2) for x in o] for o in vision.clutter]}


def _sight_values(pilot, det, start) -> list:
    vals = (pilot.sightpilot.log_values(det, start) if pilot.sight == 'rabbit' and pilot.sightpilot is not None
            else SightPilot.empty_log_values(det))
    return [v for c, v in zip(SIGHT_LOG_COLUMNS, vals) if c not in ('det_u', 'det_v')]


def _fmt(x) -> str:
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return ''
        return f'{x:.4f}' if abs(x) >= 100 else f'{x:.5f}'.rstrip('0').rstrip('.') or '0'
    return str(x)


# ----------------------------------------------------------------------------- score

def _log_dict(rows: list[list]) -> dict[str, np.ndarray]:
    out = {}
    for j, c in enumerate(LOG_COLUMNS):
        col = [r[j] for r in rows]
        out[c] = np.array(col, dtype=object) if c == 'status' else np.array(col, dtype=np.float64)
    return out


def phantom_report(result: dict, gates: list[dict], assoc_m: float = 4.0, ray_deg: float = 6.0) -> dict | None:
    """What the tracker built that was not an arch, by ``liftoff replay-sight``'s own definitions.

    Needs ``RehearsalOptions.track_report``. The classification is ``sightreplay._track_table``'s, unchanged: a
    track belongs to the arch it sits within ``assoc_m`` of for most of its unpassed life, failing that to the arch
    whose bearing it shares to within ``ray_deg`` (an arch at the wrong range is not a phantom), failing that to
    the arch most of its sightings actually saw; a confirmed track left over is a phantom. A rehearsal log cannot
    be handed to ``replay-sight`` (its clock starts at 1000 s, so the detector frames cannot be placed in time),
    which is why the tracks are recorded as the rehearsal runs instead."""
    rec = result.get('rec')
    if not rec:
        return None
    from ..liftoff.sightreplay import _sighting_arches, _track_table
    log = _log_dict(result['rows'])
    res = {'tracks': {k: np.asarray(v) for k, v in rec['snap'].items()}, 'cols': np.asarray(rec['cols']),
           'epoch': np.asarray(rec['epochs']), 'ends': rec['ends'], 'sightings': rec['sightings'],
           'events': rec['events'], 'cam': rec['cam']}
    _sighting_arches(res, gates)
    tab = _track_table(log, res, gates, assoc_m, ray_deg)
    conf = [t for t in tab.values() if t['confirmed']]
    ph = [t for t in conf if t['arch'] is None]
    keys = {(t['epoch'], t['id']) for t in ph}
    passes = [e for e in rec['events'] if e['kind'] != 'unpass' and (e['epoch'], e['id']) in keys]
    sightings = [s for s in res['sightings'] if 'm' in s]
    sim_s = result['steps'] * result['dt']
    out = {'tracks': len(tab), 'confirmed': len(conf), 'phantoms': len(ph),
           'phantoms_per_100s': round(100.0 * len(ph) / max(sim_s, 1e-6), 2),
           'phantom_target_rows': int(sum(t['target_rows'] for t in ph)),
           'phantom_passes': len(passes), 'passes_declared': len([e for e in rec['events'] if e['kind'] != 'unpass']),
           'sightings': len(sightings), 'sightings_at_no_arch': sum(1 for s in sightings if s['arch'] is None),
           'worst': [{k: t[k] for k in ('epoch', 'id', 'hits', 'life_s', 'median_pos', 'nearest_arch',
                                        'nearest_arch_d', 'drone_range_median', 'target_rows', 'end', 'saw')}
                     for t in sorted(ph, key=lambda t: -t['target_rows'])[:6]]}
    if ph:
        out['phantom_hits_median'] = float(np.median([t['hits'] for t in ph]))
        out['phantom_life_s_median'] = round(float(np.median([t['life_s'] for t in ph])), 1)
        out['phantom_arch_d_median'] = round(float(np.median([t['nearest_arch_d'] for t in ph])), 1)
        out['phantom_drone_range_median'] = round(float(np.median([t['drone_range_median'] for t in ph])), 1)
    return out


def summarize(result: dict, gates: list[dict], track: np.ndarray | None = None, up_m: float = CENTRE_UP_M) -> dict:
    """Score a rehearsal: the game logs' score (flightlog.score_attempt) per attempt, plus what only a rehearsal
    knows: time spent slow, jumps of the goal, time in each pilot mode, how far the pilot's gate estimate was from
    the real arch, what the pilot believed it passed, the detector's statistics and the crashes."""
    from ..liftoff.flightlog import attempts, score_attempt
    log = _log_dict(result['rows'])
    dt = result['dt']
    out = {'sim_s': result['steps'] * dt, 'wall_s': round(result['wall_s'], 1), 'delay_steps': result['delay_steps'],
           'detector': result['detector'], 'attempts': []}
    centres = np.array([np.asarray(g['pos'], dtype=np.float64) + [0.0, 0.0, up_m] for g in gates])
    for idx in attempts(log):
        r = score_attempt(log, idx, gates, track)
        P = np.c_[log['px'][idx], log['py'][idx], log['pz'][idx]]
        V = np.c_[log['vx'][idx], log['vy'][idx]]
        air = P[:, 2] > 0.5
        speed = np.linalg.norm(V, axis=1)
        r['slow_s'] = float((air & (speed < 1.0)).sum() * dt)          # airborne below 1 m/s
        r['crawl_s'] = float((air & (speed < 0.3)).sum() * dt)          # airborne and all but stopped
        T = np.c_[log['tx'][idx], log['ty'][idx], log['tz'][idx]]
        jump = np.linalg.norm(np.diff(T, axis=0), axis=1)
        Gb = np.c_[log['gx'][idx], log['gy'][idx], log['gz'][idx]]
        enc = np.tanh(Gb / 2.0)                                           # what the brain sees of the goal
        enc_jump = np.linalg.norm(np.diff(enc, axis=0), axis=1)
        r['goal_jumps'] = {'target_gt_0.5m': int((jump > 0.5).sum()), 'target_gt_2m': int((jump > 2.0).sum()),
                           'target_max_m': float(jump.max()) if len(jump) else 0.0,
                           'encoded_gt_0.25': int((enc_jump > 0.25).sum()),
                           'encoded_p99': float(np.percentile(enc_jump, 99)) if len(enc_jump) else 0.0,
                           'body_goal_median_m': float(np.median(np.linalg.norm(Gb, axis=1)))}
        # a goal far ahead of a pitched-down drone points UP in its body frame (15 m ahead at 10 deg of pitch is
        # 2.6 m up) while the brain's encoding saturates horizontally: the brain climbs although the target is below
        far = air & (np.linalg.norm(Gb, axis=1) > 8.0)
        Pz, Vz = log['pz'][idx], log['vz'][idx]
        r['far_goal'] = {'s': float(far.sum() * dt),
                         'vz_mean': float(Vz[far].mean()) if far.any() else 0.0,
                         'body_gz_mean': float(Gb[far, 2].mean()) if far.any() else 0.0,
                         'target_dz_mean': float((T[far, 2] - Pz[far]).mean()) if far.any() else 0.0}
        status = log['status'][idx]
        modes = {'gate seen': 0, 'gate remembered': 0, 'flying on': 0, 'creeping': 0, 'holding': 0, 'searching': 0}
        for s in status:
            if s.startswith('gate seen'):
                modes['gate seen'] += 1
            elif s.startswith('gate remembered'):
                modes['gate remembered'] += 1
            elif s.startswith('flying on'):
                modes['flying on'] += 1
            elif s.startswith('creeping'):
                modes['creeping'] += 1
            elif 'searching' in s:
                modes['searching'] += 1
            elif s.startswith('no gate'):
                modes['holding'] += 1
        r['mode_s'] = {k: round(v * dt, 1) for k, v in modes.items()}
        E = np.c_[log['est_x'][idx], log['est_y'][idx], log['est_z'][idx]]
        has = np.isfinite(E[:, 0])
        if has.any():
            err = np.linalg.norm(E[has][:, None, :] - centres[None], axis=2)
            near = err.argmin(1)
            e = err.min(1)
            r['gate_estimate'] = {'steps': int(has.sum()), 'err_median_m': float(np.median(e)),
                                  'err_p90_m': float(np.percentile(e, 90)),
                                  'per_gate_median_m': {int(g): round(float(np.median(e[near == g])), 2) for g in np.unique(near)}}
        dg = log['det_gate'][idx]
        r['detections'] = {'arch_reported_s': float((dg >= 0).sum() * dt), 'phantom_s': float((dg == -2).sum() * dt),
                           'clutter_s': float((dg == -3).sum() * dt)}
        # the view: yaw stick reversals (beyond +-0.02) per airborne minute
        cy = log['c_yaw'][idx][air]
        sg = np.sign(cy[np.abs(cy) > 0.02])
        r['yaw_flips_per_min'] = float((np.diff(sg) != 0).sum() / max(air.sum() * dt / 60.0, 1e-6)) if len(sg) > 1 else 0.0
        through = [c for c in r.get('crossings', []) if c['through']]
        r['lateral_max_through'] = float(max(abs(c['lateral_m']) for c in through)) if through else float('nan')
        mode = log['mode'][idx]
        if np.isfinite(mode).any():
            last = {k: float(np.nanmax(log[k][idx])) for k in ('n_passes', 'ghosts', 'unpasses', 'reseeds', 'goal_clips',
                                                              'rej_elev', 'rej_stale', 'rej_offaxis', 'absorbed', 'low',
                                                              'behind', 'orphans', 'sight_errors')}
            pk = log['pass_kind'][idx]
            last['pass_kinds'] = {n: int((pk == v).sum()) for n, v in (('cross', 1), ('travel', 2), ('beside', 3),
                                                                      ('ghost', 4), ('unpass', 5))}
            last['mode_s'] = {n: round(float((mode == v).sum() * dt), 1)
                              for v, n in ((0, 'ground'), (1, 'cruise'), (2, 'target'), (3, 'search'), (4, 'hold'))}
            last['flow_gain_median'] = float(np.nanmedian(log['flow_gain'][idx][air])) if air.any() else float('nan')
            last['rabbit_speed_median'] = float(np.nanmedian(log['rb_v'][idx][air])) if air.any() else float('nan')
            r['sight'] = last
        out['attempts'].append(r)
    ev = result['events']
    out['crashes'] = [e for e in ev if e['kind'].startswith('crash')]
    passes = []
    for e in ev:
        if e['kind'] != 'pilot_pass':
            continue
        g_est = e.get('gate_est')
        if g_est is not None:
            d = np.linalg.norm(centres - np.asarray(g_est), axis=1)
            passes.append({'ts': round(e['ts'], 1), 'attempt': e['attempt'], 'nearest_gate': int(d.argmin()),
                           'est_err_m': round(float(d.min()), 2)})
    out['pilot_passes'] = passes
    out['phantom_tracks'] = phantom_report(result, gates)
    if result.get('clutter'):
        out['clutter'] = result['clutter']
    return out


def describe(summary: dict, name: str = 'rehearsal') -> str:
    from ..liftoff.flightlog import describe as describe_log
    lines = [describe_log(name, summary['attempts'])]
    det = summary['detector']
    lines.append(f'{name}: {summary["sim_s"]:.0f} s simulated in {summary["wall_s"]:.0f} s, latency {summary["delay_steps"]} steps; '
                 f'detector frames {det["frames"]}, arch in view {det["arch_in_view"]}, detected {det["detected"]}, '
                 f'missed {det["missed"]}, flipped {det["flipped"]}, phantoms {det["phantoms"]}'
                 + (f', clutter boxes {det["clutter"]} ({det["clutter_over_arch"]} of them over an arch the detector '
                    f'had found)' if det.get('clutter') else ''))
    if summary.get('phantom_tracks') is not None:
        ph = summary['phantom_tracks']
        lines.append(f'  tracker: {ph["tracks"]} tracks, {ph["confirmed"]} confirmed, {ph["phantoms"]} of them '
                     f'confirmed phantoms ({ph["phantoms_per_100s"]:.1f} per 100 s), {ph["phantom_target_rows"]} '
                     f'steps flown at a phantom, {ph["phantom_passes"]} passes declared at one')
        for p in ph['worst']:
            lines.append(f'  phantom #{p["id"]}: {p["hits"]} hits, {p["life_s"]:.1f} s, at {p["median_pos"]} '
                         f'({p["nearest_arch_d"]:.1f} m from arch {p["nearest_arch"]}), target {p["target_rows"]} '
                         f'rows, ended {p["end"]}')
    for k, r in enumerate(summary['attempts']):
        if 'slow_s' not in r:
            continue
        gj = r['goal_jumps']
        fg = r['far_goal']
        s = (f'  attempt {k + 1}: slow (<1 m/s) {r["slow_s"]:.1f} s, stopped (<0.3 m/s) {r["crawl_s"]:.1f} s, height '
             f'{r["z_range"][0]:.1f}-{r["z_range"][1]:.1f} m | goal jumps: {gj["target_gt_0.5m"]} > 0.5 m, '
             f'{gj["target_gt_2m"]} > 2 m (max {gj["target_max_m"]:.1f} m), encoded goal steps > 0.25: '
             f'{gj["encoded_gt_0.25"]} (p99 {gj["encoded_p99"]:.3f}), goal distance median {gj["body_goal_median_m"]:.1f} m | '
             f'goal beyond 8 m for {fg["s"]:.1f} s (target height {fg["target_dz_mean"]:+.1f} m relative to the drone, body goal z '
             f'{fg["body_gz_mean"]:+.1f} m, climb {fg["vz_mean"]:+.2f} m/s) | modes (s) {r["mode_s"]}')
        s += f' | yaw flips {r["yaw_flips_per_min"]:.1f}/min, largest |lateral| through {r["lateral_max_through"]:.2f} m'
        if 'sight' in r:
            sg = r['sight']
            s += (f' | rabbit: passes {sg["n_passes"]:.0f} {sg["pass_kinds"]}, ghosts {sg["ghosts"]:.0f}, unpasses '
                  f'{sg["unpasses"]:.0f}, reseeds {sg["reseeds"]:.0f}, goal clips {sg["goal_clips"]:.0f}, rejected '
                  f'elev/stale/off-axis {sg["rej_elev"]:.0f}/{sg["rej_stale"]:.0f}/{sg["rej_offaxis"]:.0f}, absorbed '
                  f'{sg["absorbed"]:.0f}, low {sg["low"]:.0f}, orphan passes {sg["orphans"]:.0f}, '
                  f'errors {sg["sight_errors"]:.0f}, modes {sg["mode_s"]}, flow median {sg["flow_gain_median"]:.2f}, '
                  f'rabbit speed median {sg["rabbit_speed_median"]:.2f}')
        if 'gate_estimate' in r:
            ge = r['gate_estimate']
            s += f' | gate estimate error median {ge["err_median_m"]:.1f} m (p90 {ge["err_p90_m"]:.1f}), per gate {ge["per_gate_median_m"]}'
        lines.append(s)
    if summary['pilot_passes']:
        lines.append('  pilot believed it passed: ' + ', '.join(
            f'g{p["nearest_gate"]}@{p["ts"]:.0f}s({p["est_err_m"]:.1f}m)' for p in summary['pilot_passes']))
    for c in summary['crashes']:
        lines.append(f'  crash: {c["kind"]} at ts {c["ts"]:.1f} s (attempt {c["attempt"] + 1}) at {c["pos"]}'
                     + (f', gate {c["gate"]}' if 'gate' in c else ''))
    return '\n'.join(lines)


def rehearse(ckpt: str, camera_yaml: str, gates_json: str, log_path: str | None = None, track_yaml: str | None = None,
             device: str = 'cuda', opts: RehearsalOptions | None = None, detector: DetectorModel | None = None,
             json_out: str | None = None) -> dict:
    import yaml
    from ..train.bptt import load_checkpoint
    from .gates import load_gate_file
    c = yaml.safe_load(Path(camera_yaml).read_text(encoding='utf-8'))
    cam = Camera(int(c['width']), int(c['height']), float(c['f']), float(c['tilt_deg']))
    gates, width_m, up_m = load_gate_file(gates_json)
    track = None
    if track_yaml:
        d = yaml.safe_load(Path(track_yaml).read_text(encoding='utf-8'))
        track = np.array(d['waypoints'] if isinstance(d, dict) else d, dtype=float)[:, :3]
    opts = opts or RehearsalOptions()
    detector = detector or DetectorModel()
    brain, cfg, _ = load_checkpoint(ckpt, device)
    if opts.verbose:
        print(f'rehearsal: brain {ckpt} ({getattr(brain, "N", 0)} neurons on {brain.device}), {len(gates)} arches, camera '
              f'{cam.hfov_deg:.0f} deg HFOV tilt {cam.tilt_deg:.0f} deg, {opts.seconds:.0f} s simulated\n'
              f'  detector: {asdict(detector)}', flush=True)
    res = run_rehearsal(brain, cfg, gates, cam, log_path, opts, detector, width_m, up_m)
    summary = summarize(res, gates, track, up_m)
    summary['options'] = {k: v for k, v in asdict(opts).items() if k != 'verbose'}
    summary['detector_model'] = asdict(detector)
    print(describe(summary, Path(log_path).stem if log_path else 'rehearsal'), flush=True)
    if json_out:
        Path(json_out).write_text(json.dumps(summary, indent=1, default=float), encoding='utf-8')
    return summary
