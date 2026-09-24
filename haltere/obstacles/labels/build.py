"""Label build orchestration. OFFLINE; delivered by the labels agent.

``python -m haltere.obstacles.labels build --store runs/obstacle-store-v1 --colliders
runs/challenge-box-calibration-20260923/offline-geometry.json --events <lateral manifest.json>
--extra-events configs/obstacles/events_f12.json --flight-lock PATH``

Stages (resumable per run with LabelWriter.is_done/mark_done; ChunkGuard.before_chunk() per run):
events -> L5 colliders -> L1 tube -> L3 impacts -> L2 hindsight maps -> projection and combination
(labels.intersect in COMBINE_ORDER, conflicts counted) -> K0b quality -> LabelWriter.finalize.
"""
from __future__ import annotations


def build(args):
    raise NotImplementedError('label build: labels build agent')


def teacher(args):
    raise NotImplementedError('teacher cache: labels build agent')
