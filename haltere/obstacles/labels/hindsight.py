"""L2: hindsight triangulation fused into per-flight voxel maps. OFFLINE; delivered by the labels agent.

Run haltere.vision.temporal_depth.MultiBaselineDepth forward AND backward over each flight's frames
(store frames or the source at a higher rate), with keyframes at +-0.5, +-1 and +-2 s, masked by
haltere.obstacles.overlays (HUD glyphs, ring stroke and ghost trails excluded; points on ghost-trail
pixels rejected). Keep points with sigma < 0.1 r. Fuse each flight into 0.2 m voxels (one consistent
frame per flight) with visibility carving (free counts along each observation ray up to the point
minus 2 sigma). Project into every frame of that flight: per sub-ray, first occupied voxel = EXACT,
carved free until unobserved = LOWER, else UNKNOWN; reduce with labels.min_over. Unreliable
alignments get no L2. Chunk per flight; ChunkGuard.before_chunk() before each chunk.

K0b checks go into the label manifest 'quality': L2 vs colliders median |err|/r <= 10 % at 2-10 m;
L2 vs L3 within 15 % on >= 80 % of frames 2.0-0.5 s before impact where both exist; >= 30 % of
near-travel fan cells (<= 8 m, +-20 deg) uncensored in the Minus Two and Pine Valley test sets.
"""
from __future__ import annotations

VOXEL_M = 0.2
KEYFRAME_OFFSETS_S = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
MAX_SIGMA_FRACTION = 0.1


def build_flight_map(store, run_id: int, guard):
    """Triangulate and fuse one flight -> voxel map saved under labels/parts/hindsight/rNNNNN.npz."""
    raise NotImplementedError('L2 hindsight: labels build agent')


def hindsight_labels(voxel_map, pos, quat_wb):
    """-> (grid_value, grid_kind, fan_value, fan_kind) for one frame."""
    raise NotImplementedError('L2 hindsight: labels build agent')
