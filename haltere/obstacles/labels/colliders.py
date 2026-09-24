"""L5: box-collider ray-cast labels (Drawing Board box course only). OFFLINE; delivered by the labels agent.

Inputs: runs/challenge-box-calibration-20260923/offline-geometry.json (Unity frame, box primitives
+ ground plane y = 0; runtime_geometry_allowed = false) and store rows of the box course with
ABSOLUTE positions (FrameStore.absolute_pos: launch-relative pos + origin_sim). Use
haltere.liftoff.section_geometry.collision_depth / camera_collision_depth (Euclidean range, not
optical depth) at >= 3 x 3 sub-rays per grid cell of contract.store_camera().

Outputs per frame: grid (18, 32) and fan (4, 9) constraints (value, LabelKind) with
LabelSource.COLLIDER: a hit = EXACT (reduced with labels.min_over); no hit within RANGE_MAX_M =
LOWER RANGE_MAX_M only because this course lists no unknown geometry (record that assumption in
the label manifest). Also provides the E2 test cells and the K0b check L2-vs-collider
(median |err|/r <= 10 % at 2-10 m).
"""
from __future__ import annotations


def collider_labels(geometry: dict, pos_abs, quat_wb):
    """-> (grid_value (18, 32), grid_kind, fan_value (4, 9), fan_kind) for one frame."""
    raise NotImplementedError('L5 colliders: labels build agent')
