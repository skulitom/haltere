"""Teacher cache: DA-V2-Small relative disparity (Apache-2.0) per frame. OFFLINE; labels agent.

Input: the store frame (unmasked 448 x 252, resized to the model input recorded in
teacher/manifest.json). Output: relative (affine-invariant) disparity averaged over 7 x 7 px blocks,
(N, 36, 64) float16 in <store>/labels/teacher/disparity.npy. GPU in <= 10 min chunks via
ChunkGuard(gpu=True); resumable by row ranges. Used only as a scale/shift-invariant shape
regulariser (weight 0.1) and, in M2, for L4 affine fits to L1-L3 anchors (>= 30 anchors, residual
< 15 %). Never a runtime input.
"""
from __future__ import annotations

TEACHER_MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'
TEACHER_LICENSE = 'Apache-2.0'


def build_teacher_cache(store, labels_root, guard, *, batch: int = 32):
    raise NotImplementedError('teacher cache: labels build agent')
