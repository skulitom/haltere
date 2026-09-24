"""L3: impact/contact points as exact occupied points. OFFLINE; delivered by the labels agent.

Events: the 15 curated lateral-manifest impacts, plus the blind-labelled remaining Minus Two (6) and
Pine Valley (2) terminal impacts, contacts and near passes within 1 m
(configs/obstacles/events_f12.json), schema labels.EVENT_FIELDS. Blind protocol: every event is
labelled (point_w, normal_w, obstacle, unique_obstacle, free sides) BEFORE any model or baseline
output on it is seen; set blind=true, labeller and labelled_at. Oracle-route flights are marked and
excluded from evaluation sets.

Per frame in [T - 3 s, T - 0.15 s] where point_w projects into the 448 x 252 image: sub-rays that hit
a 0.2 m disc around point_w (normal normal_w) are EXACT at their range (IMPACT); other sub-rays are
untouched. The same points constrain fan corridors (UPPER when the corridor before them is
unobserved). Writes <store>/labels/events.json (superset of <store>/events.json; keeps
store_event_id).
"""
from __future__ import annotations

IMPACT_WINDOW_S = (0.15, 3.0)
IMPACT_DISC_RADIUS_M = 0.2


def impact_labels(event: dict, pos, quat_wb):
    """-> (grid_value, grid_kind, fan_value, fan_kind) for one frame, or None when out of view."""
    raise NotImplementedError('L3 impacts: labels build agent')
