"""L1: flown-tube free-space lower bounds. OFFLINE; delivered by the labels agent.

For a frame at t, the drone positions over [t, t + 3 s] (100 Hz telemetry, stopping at the first
contact, terminal impact or reset) swept with a 0.35 m radius sphere are free. Every grid sub-ray
that starts inside the tube gets LOWER s, where s is the distance at which it leaves the tube; a fan
corridor (radius FAN_CORRIDOR_RADIUS_M) counts only as far as it lies inside the tube (record the
exact rule in the label manifest). LabelSource.TUBE. This counters teacher smear inside gate
openings. Unreliable alignments get no tube labels.
"""
from __future__ import annotations

TUBE_HORIZON_S = 3.0
TUBE_RADIUS_M = 0.35


def tube_labels(telemetry_t, telemetry_pos, t_frame: float, pos, quat_wb, stop_time: float | None = None):
    """-> (grid_value, grid_kind, fan_value, fan_kind) LOWER bounds for one frame."""
    raise NotImplementedError('L1 tube: labels build agent')
