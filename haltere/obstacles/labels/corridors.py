"""Fan corridor queries shared by the label sources. OFFLINE; delivered by the labels agent.

first_blocked(points_w, observed, origin_w, quat_wb) -> (value (4, 9), kind (4, 9)):
directions = contract.fan_directions_world(quat_wb); for each, the smallest along-ray distance
s >= 0 among occupied points whose perpendicular distance to the ray is <= FAN_CORRIDOR_RADIUS_M.
EXACT if the corridor [0, s] is observed free, UPPER s if it has unobserved gaps before s, LOWER
s_obs if it leaves observed space at s_obs first, LOWER FAN_MAX_M if free to FAN_MAX_M. Then
labels.clip_fan.
"""
from __future__ import annotations


def first_blocked(points_w, observed, origin_w, quat_wb):
    raise NotImplementedError('fan corridors: labels build agent')
