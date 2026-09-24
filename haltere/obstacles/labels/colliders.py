"""L5: box-collider ray-cast labels (Drawing Board box course only). OFFLINE ONLY.

Inputs: runs/challenge-box-calibration-20260923/offline-geometry.json (Unity frame x right, y up,
z forward; 213 box primitives + ground plane y = 0; runtime_geometry_allowed = false; its 12
checkpoints are fly-through rings, not colliders) and frames of the box course with ABSOLUTE
simulator positions (FrameStore.absolute_pos: launch-relative pos + origin_sim).

Ray casting follows haltere.liftoff.section_geometry.ray_box / collision_depth (same slab test,
same yaw convention about Unity y, same ground plane), vectorised over rays and boxes with one
origin per ray so fan corridors can be cast as ray bundles; tests check it against
section_geometry.collision_depth. Ranges are Euclidean (not optical depth).

Grid: 3 x 3 sub-rays per cell; a hit = EXACT at its range, no hit within RANGE_MAX_M = LOWER
RANGE_MAX_M, reduced with labels.min_over and clipped. The "no hit = free" rule holds only because
this course lists no unknown geometry (``unknown_geometry == 0``); the builder records that
assumption and ``collider_labels`` refuses geometry with unknown meshes.

Fan: the first box point within FAN_CORRIDOR_RADIUS_M of a corridor axis is found as the minimum of
(a) the entry depths of rays parallel to the axis through the corridor cross-section (the axis, 8 at
0.25 m and 64 on the 0.5 m boundary circle, 5.6 deg apart), and (b) the along-axis distance of every
box vertex that lies inside the corridor. For a convex box the front-surface depth over the
cross-section is minimised on the cross-section boundary or at a vertex, so this is exact up to the
boundary sampling (a box thinner than ~5 cm that crosses the boundary between two samples can be
missed). A hit = EXACT, none within FAN_MAX_M = LOWER FAN_MAX_M. Rendered surfaces are not verified against the
colliders (render_alignment_verified = false in the geometry file): K0b measures the agreement.
"""
from __future__ import annotations

import numpy as np

from .. import contract
from ..contract import FAN_CORRIDOR_RADIUS_M, FAN_MAX_M, FAN_SHAPE, RANGE_MAX_M
from . import LabelKind, clip_fan
from .corridors import SUB_SHAPE, grid_from_subrays, subray_dirs_world, unknown_fan

# sim = M @ unity (haltere.liftoff.frames.M); kept local so this module stays a pure-numpy reader.
_M = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
BUNDLE_RINGS = ((0.25, 8), (FAN_CORRIDOR_RADIUS_M, 64))
BUNDLE_SIZE = 1 + sum(n for _, n in BUNDLE_RINGS)
TILE = (9, 12)                     # sub-ray tile cast as one chunk (54 x 96 = 6 x 8 tiles)
ASSUMPTION = ('box course lists no unknown geometry (unknown_geometry == 0): no collider hit within '
              'RANGE_MAX_M / FAN_MAX_M is labelled LOWER free; collider-vs-render alignment unverified')


def sim_to_unity(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) @ _M


class BoxScene:
    """Vectorised box-collider scene (Unity frame) prepared from an offline geometry dict."""

    def __init__(self, geometry: dict):
        if geometry.get('runtime_geometry_allowed') is not False:
            raise ValueError('Expected an explicitly offline geometry contract')
        if geometry.get('unknown_geometry'):
            raise ValueError('Unmodelled meshes prevent complete collider labels for this course')
        prims = geometry['primitives']
        if any(p['kind'] != 'box' for p in prims):
            raise ValueError('Only box collider geometry is supported')
        self.centers = np.array([p['center'] for p in prims], dtype=np.float64).reshape(-1, 3)
        self.halfs = np.array([p['size'] for p in prims], dtype=np.float64).reshape(-1, 3) / 2.0
        yaw = np.deg2rad(np.array([p.get('yaw_deg', 0.0) for p in prims], dtype=np.float64))
        c, s = np.cos(yaw), np.sin(yaw)
        # Same rotation as section_geometry.ray_box: [[c, 0, s], [0, 1, 0], [-s, 0, c]]; local = v @ rot.
        rot = np.zeros((len(prims), 3, 3))
        rot[:, 0, 0], rot[:, 0, 2], rot[:, 1, 1], rot[:, 2, 0], rot[:, 2, 2] = c, s, 1.0, -s, c
        self.rot = rot
        self.instance = np.array([p.get('instance_id', i + 1) for i, p in enumerate(prims)], dtype=np.int64)
        self.radius = np.linalg.norm(self.halfs, axis=1)
        self.ground_y = geometry.get('ground_plane_y')

    def vertices_sim(self) -> np.ndarray:
        """(8 * n_boxes, 3) box corners in the simulator FLU frame (absolute)."""
        if not hasattr(self, '_vertices'):
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
            local = signs[None] * self.halfs[:, None, :]                         # (b, 8, 3)
            world_u = self.centers[:, None, :] + np.einsum('bvk,bjk->bvj', local, self.rot)   # local @ rot.T
            self._vertices = world_u.reshape(-1, 3) @ _M.T                       # sim = M @ unity
        return self._vertices

    def _candidates(self, origins_u, dirs_u, max_range: float):
        """Boxes that some ray of this bundle can reach within max_range (exact culling).

        Range: the bounding sphere must come within max_range + spread of the bundle centre. Cone: a ray
        from an origin within ``spread`` of the centre can only meet a bounding sphere of radius r if the
        parallel ray from the centre meets the sphere grown by ``spread``, i.e. the sphere centre lies within
        asin((r + spread) / dist) of that ray's direction, hence within that of the bundle's axis cone.
        """
        centre = origins_u.mean(axis=0)
        spread = np.linalg.norm(origins_u - centre, axis=1).max()
        v = self.centers - centre
        dist = np.linalg.norm(v, axis=1)
        keep = dist - self.radius <= max_range + spread
        axis = dirs_u.sum(axis=0)
        if np.linalg.norm(axis) > 1e-9:
            axis = axis / np.linalg.norm(axis)
            half = np.arccos(np.clip(dirs_u @ axis, -1.0, 1.0)).max()
            grown = self.radius + spread
            ang = np.arccos(np.clip((v @ axis) / np.maximum(dist, 1e-12), -1.0, 1.0))
            margin = np.arcsin(np.clip(grown / np.maximum(dist, 1e-12), 0.0, 1.0))
            keep &= (dist <= grown) | (ang - margin <= half + 1e-6)
        return self.centers[keep], self.halfs[keep], self.rot[keep]

    def cast(self, origins_u, dirs_u, max_range: float = RANGE_MAX_M, chunk: int = 1024):
        """First hit distance (inf if none within max_range) for rays with per-ray origins (Unity frame)."""
        dirs_u = np.asarray(dirs_u, dtype=np.float64).reshape(-1, 3)
        origins_u = np.broadcast_to(np.asarray(origins_u, dtype=np.float64), dirs_u.shape)
        out = np.full(len(dirs_u), np.inf)
        for a in range(0, len(dirs_u), chunk):
            o, d = origins_u[a:a + chunk], dirs_u[a:a + chunk]
            C, H, R = self._candidates(o, d, max_range)
            best = np.full(len(d), np.inf)
            if len(C):
                lo_ = np.einsum('rj,bjk->rbk', o, R) - np.einsum('bj,bjk->bk', C, R)[None]   # (r, b, 3)
                ld = np.einsum('rj,bjk->rbk', d, R)
                par = np.abs(ld) < 1e-12
                with np.errstate(divide='ignore', invalid='ignore'):
                    t1 = np.where(par, 0.0, (-H[None] - lo_) / np.where(par, 1.0, ld))
                    t2 = np.where(par, 0.0, (H[None] - lo_) / np.where(par, 1.0, ld))
                tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
                outside = np.abs(lo_) > H[None]
                tn = np.where(par, np.where(outside, np.inf, -np.inf), tn)
                tf = np.where(par, np.where(outside, -np.inf, np.inf), tf)
                enter = np.maximum(tn.max(axis=-1), 0.0)
                leave = tf.min(axis=-1)
                hit = np.where(leave >= enter, enter, np.inf)
                best = hit.min(axis=1)
            if self.ground_y is not None:
                down = d[:, 1] < -1e-12
                with np.errstate(divide='ignore', invalid='ignore'):
                    g = np.where(down, (self.ground_y - o[:, 1]) / np.where(down, d[:, 1], -1.0), np.inf)
                g = np.where(g >= 0, g, np.inf)          # as section_geometry.collision_depth
                best = np.minimum(best, g)
            out[a:a + chunk] = np.where(best <= max_range, best, np.inf)
        return out


def _scene(geometry) -> BoxScene:
    return geometry if isinstance(geometry, BoxScene) else BoxScene(geometry)


def subray_ranges(geometry, pos_abs, quat_wb, max_range: float = RANGE_MAX_M) -> np.ndarray:
    """(54, 96) collider range along every sub-ray (inf = no hit within max_range)."""
    scene = _scene(geometry)
    dirs = subray_dirs_world(quat_wb)
    o = sim_to_unity(np.asarray(pos_abs, dtype=np.float64))
    # cast in 9 x 12 sub-ray tiles so each chunk's cone (and its box candidate list) is narrow
    th, tw = TILE
    tiles = dirs.reshape(SUB_SHAPE[0] // th, th, SUB_SHAPE[1] // tw, tw, 3).transpose(0, 2, 1, 3, 4)
    r = scene.cast(o, sim_to_unity(tiles.reshape(-1, 3)), max_range, chunk=th * tw)
    return r.reshape(SUB_SHAPE[0] // th, SUB_SHAPE[1] // tw, th, tw).transpose(0, 2, 1, 3).reshape(SUB_SHAPE)


def ranges_along(geometry, pos_abs, dirs_w, max_range: float = RANGE_MAX_M) -> np.ndarray:
    """Collider range along arbitrary world rays from one absolute position (e.g. through pixels)."""
    scene = _scene(geometry)
    return scene.cast(sim_to_unity(np.asarray(pos_abs, dtype=np.float64)),
                      sim_to_unity(np.asarray(dirs_w, dtype=np.float64).reshape(-1, 3)), max_range)


def _bundle_offsets(d: np.ndarray) -> np.ndarray:
    """(m, 3) unit axes -> (m, n_bundle, 3) perpendicular offsets (axis ray first)."""
    a = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    e1 = np.cross(d, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)
    offs = [np.zeros_like(d)]
    for r, n in BUNDLE_RINGS:
        for k in range(n):
            ang = 2 * np.pi * (k + 0.5 * (r < FAN_CORRIDOR_RADIUS_M)) / n
            offs.append(r * (np.cos(ang) * e1 + np.sin(ang) * e2))
    return np.stack(offs, axis=1)


def fan_hits(geometry, pos_abs, quat_wb, s_max: float = FAN_MAX_M) -> np.ndarray:
    """(4, 9) along-axis distance to the first collider point inside each fan corridor (inf = none)."""
    scene = _scene(geometry)
    d = contract.fan_directions_world(np.asarray(quat_wb, dtype=np.float64)).reshape(-1, 3)
    if not np.isfinite(d).all():
        return np.full(FAN_SHAPE, np.nan)
    o = np.asarray(pos_abs, dtype=np.float64)
    offs = _bundle_offsets(d)                                            # (36, n, 3)
    origins = o[None, None] + offs
    dirs = np.broadcast_to(d[:, None, :], offs.shape)
    t = scene.cast(sim_to_unity(origins.reshape(-1, 3)), sim_to_unity(dirs.reshape(-1, 3)), s_max,
                   chunk=BUNDLE_SIZE)                                  # one corridor per chunk
    best = t.reshape(offs.shape[:2]).min(axis=1)
    # box vertices inside a corridor (small boxes wholly inside the cross-section)
    v = scene.vertices_sim()
    if len(v):
        rel = v - o[None]
        near = np.linalg.norm(rel, axis=1) <= s_max + FAN_CORRIDOR_RADIUS_M
        rel = rel[near]
        if len(rel):
            sv = rel @ d.T                                               # (n_vertices, 36)
            perp2 = np.maximum((rel ** 2).sum(1)[:, None] - sv ** 2, 0.0)
            inside = (sv >= 0) & (sv <= s_max) & (perp2 <= FAN_CORRIDOR_RADIUS_M ** 2)
            vbest = np.where(inside, sv, np.inf).min(axis=0)
            best = np.minimum(best, vbest)
    return best.reshape(FAN_SHAPE)


def collider_labels(geometry, pos_abs, quat_wb, *, return_subrays: bool = False):
    """-> (grid_value (18, 32), grid_kind, fan_value (4, 9), fan_kind) for one frame (LabelSource.COLLIDER)."""
    scene = _scene(geometry)
    r = subray_ranges(scene, pos_abs, quat_wb, RANGE_MAX_M)
    sub_v = np.where(np.isfinite(r), r, RANGE_MAX_M)
    sub_k = np.where(np.isfinite(r), LabelKind.EXACT, LabelKind.LOWER).astype(np.uint8)
    gv, gk = grid_from_subrays(sub_v, sub_k)
    h = fan_hits(scene, pos_abs, quat_wb, FAN_MAX_M)
    if np.isnan(h).all():
        fv, fk = unknown_fan()
    else:
        fv = np.where(np.isfinite(h), h, FAN_MAX_M)
        fk = np.where(np.isfinite(h), LabelKind.EXACT, LabelKind.LOWER).astype(np.uint8)
        fv, fk = clip_fan(fv, fk)
    if return_subrays:
        return gv, gk, fv, fk, sub_v, sub_k
    return gv, gk, fv, fk
