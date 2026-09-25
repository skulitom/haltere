"""Teacher cache: DA-V2-Small relative disparity (Apache-2.0) per frame. OFFLINE ONLY.

Input: the store frame (unmasked 448 x 252 RGB), resized with bicubic interpolation to the model input
TEACHER_INPUT (602 x 336, multiples of 14; the least biased size in the depth probes) and normalised
with the ImageNet mean/std (the model's preprocessor). Output: the model's relative inverse depth,
resized back to 448 x 252 and averaged over 7 x 7 px blocks: (N, 36, 64) float16 in
<store>/labels/teacher/disparity.npy (affine-invariant: only its shape is meaningful).

Weights are used only when they are already on this machine (TEACHER_MODEL_DIRS, or
$HALTERE_TEACHER_MODEL_DIR): no download. The Hugging Face model card's licence field must say
apache-2.0; weights sha256, revision and licence go to teacher/manifest.json. If the relative model or
the ``transformers`` package is unavailable, the cache is skipped and the reason recorded.

GPU in <= 10 min chunks via ChunkGuard(gpu=True) (flight lock, 75/65 C pause, 80 C hard stop);
resumable by row blocks (teacher/progress.json). Used only as a scale/shift-invariant shape
regulariser (weight 0.1) and, in M2, for L4 affine fits to L1-L3 anchors. Never a runtime input.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import TEACHER, TEACHER_SHAPE

TEACHER_MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'
TEACHER_LICENSE = 'Apache-2.0'
TEACHER_REVISION = '5426e4f0f36572d16453bbda7a8389317b1bef99'   # the revision the depth probes cached
TEACHER_INPUT = (336, 602)                                       # (H, W), multiples of 14
BLOCK_PX = 7
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
# Local copies made by earlier probes (repository-relative); nothing is downloaded.
TEACHER_MODEL_DIRS = ('runs/dense-depth-probe-20260923/model',)
TRANSFORMERS_DIRS = ('runs/dense-depth-probe-20260923/dependencies',)
ROWS_PER_BLOCK = 2048


def block_average(disp: np.ndarray, block: int = BLOCK_PX) -> np.ndarray:
    """(..., 252, 448) -> (..., 36, 64) mean over block x block pixel squares."""
    d = np.asarray(disp, np.float32)
    h, w = d.shape[-2:]
    return d.reshape(d.shape[:-2] + (h // block, block, w // block, block)).mean(axis=(-3, -1))


def _repo_root() -> Path:
    from ..store import REPO_ROOT
    return REPO_ROOT


def find_model_dir() -> Path | None:
    cands = [os.environ.get('HALTERE_TEACHER_MODEL_DIR')]
    roots = [os.environ.get('HALTERE_DATA_ROOT'), _repo_root()]
    for d in TEACHER_MODEL_DIRS:
        cands += [str(Path(r) / d) for r in roots if r]
    for c in cands:
        if c and (Path(c) / 'model.safetensors').exists() and (Path(c) / 'config.json').exists():
            return Path(c)
    return None


def model_licence(model_dir: Path) -> str | None:
    """Licence from the model card front matter (``license: ...``)."""
    card = model_dir / 'README.md'
    if not card.exists():
        return None
    for line in card.read_text(encoding='utf-8', errors='replace').splitlines()[:30]:
        if line.strip().lower().startswith('license:'):
            return line.split(':', 1)[1].strip()
    return None


def _import_transformers():
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        pass
    roots = [os.environ.get('HALTERE_DATA_ROOT'), _repo_root()]
    for d in TRANSFORMERS_DIRS:
        for r in roots:
            if r and (Path(r) / d / 'transformers').exists():
                sys.path.append(str(Path(r) / d))
                try:
                    import transformers  # noqa: F401
                    return True
                except ImportError:
                    sys.path.remove(str(Path(r) / d))
    return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def frame_rows_sha256(index) -> str:
    """sha256 of the store rows' frame identities (run_id, slot). The teacher reads only frame pixels, so a
    timing re-pose that rewrites poses/grades (and hence the index sha256) but keeps these rows keeps the cache."""
    ix = np.asarray(index)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ix['run_id'].astype('<i4')).tobytes())
    h.update(np.ascontiguousarray(ix['slot'].astype('<i4')).tobytes())
    return h.hexdigest()


def teacher_valid_for(manifest: dict, store) -> bool:
    """A complete teacher cache matches this store when its frame-row identity (or index sha256) matches."""
    if manifest.get('status') != 'complete':
        return False
    if manifest.get('frame_rows_sha256'):
        return manifest['frame_rows_sha256'] == frame_rows_sha256(store.index) and manifest.get('frames') == len(store)
    return manifest.get('store_index_sha256') == store.manifest.get('index_sha256')


def preprocess(frames_u8, device):
    """(B, 252, 448, 3) uint8 RGB -> (B, 3, 336, 602) normalised float tensor on ``device``."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(frames_u8)).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=TEACHER_INPUT, mode='bicubic', align_corners=False).clamp(0, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def build_teacher_cache(store, labels_root, guard, *, batch: int = 32, device: str = 'cuda', log=print) -> dict:
    """Fill labels/teacher/disparity.npy for every store row (resumable); returns the teacher manifest."""
    root = Path(labels_root) / 'teacher'
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / 'manifest.json'
    if mpath.exists():
        old = json.loads(mpath.read_text(encoding='utf-8'))
        if teacher_valid_for(old, store):
            if old.get('store_index_sha256') != store.manifest.get('index_sha256') or not old.get('frame_rows_sha256'):
                old.update(store_index_sha256=store.manifest.get('index_sha256'),
                           frame_rows_sha256=frame_rows_sha256(store.index))
                mpath.write_text(json.dumps(old, indent=1) + '\n', encoding='utf-8')
            log('teacher cache complete for these frame rows: nothing to do')
            return old
    model_dir = find_model_dir()
    lic = model_licence(model_dir) if model_dir else None
    reason = None
    if model_dir is None:
        reason = f'relative model {TEACHER_MODEL} is not cached locally ({TEACHER_MODEL_DIRS}); no download'
    elif (lic or '').lower() != TEACHER_LICENSE.lower():
        reason = f'model card licence {lic!r} is not {TEACHER_LICENSE}'
    elif not _import_transformers():
        reason = 'transformers is not importable (venv or the vendored probe dependencies)'
    if reason:
        m = dict(status='skipped', reason=reason, model=TEACHER_MODEL, licence=lic)
        mpath.write_text(json.dumps(m, indent=1) + '\n', encoding='utf-8')
        log(f'teacher cache skipped: {reason}')
        return m
    n = len(store)
    rows_sha = frame_rows_sha256(store.index)
    import torch
    from transformers import AutoModelForDepthEstimation
    path = root / Path(TEACHER.file).name
    ppath = root / 'progress.json'
    prog = json.loads(ppath.read_text()) if ppath.exists() else {}
    if prog and prog.get('frame_rows_sha256') != rows_sha:
        log('teacher progress belongs to other frame rows: starting over')
        prog = {}
        if path.exists():
            path.unlink()
    arr = (np.load(path, mmap_mode='r+') if path.exists() else
           np.lib.format.open_memmap(path, mode='w+', dtype=TEACHER.dtype, shape=(n,) + TEACHER_SHAPE))
    if arr.shape != (n,) + TEACHER_SHAPE:
        raise ValueError(f'{path}: shape {arr.shape} does not match the store ({n} rows)')
    done = set(prog.get('blocks', []))
    weights_sha = _sha256(model_dir / 'model.safetensors')
    model = AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True).eval().to(device).half()
    blocks = [(a, min(a + ROWS_PER_BLOCK, n)) for a in range(0, n, ROWS_PER_BLOCK)]
    t0 = time.monotonic()
    gpu_s = 0.0
    import torch.nn.functional as F
    with torch.inference_mode():
        for a, b in blocks:
            if a in done:
                continue
            guard.before_chunk()
            rows = np.arange(a, b)
            for s in range(0, len(rows), batch):
                if guard.chunk_expired():
                    guard.before_chunk()
                rr = rows[s:s + batch]
                x = preprocess(store.frames(rr), device).half()
                t1 = time.monotonic()
                out = model(pixel_values=x).predicted_depth                      # (B, 336, 602) relative disparity
                out = F.interpolate(out[:, None].float(), size=(252, 448), mode='bilinear', align_corners=False)
                d = F.avg_pool2d(out, BLOCK_PX)[:, 0].cpu().numpy()                 # (B, 36, 64)
                gpu_s += time.monotonic() - t1
                arr[rr] = d.astype(np.float16)
            arr.flush()
            done.add(a)
            ppath.write_text(json.dumps(dict(blocks=sorted(done), frame_rows_sha256=rows_sha)), encoding='utf-8')
            log(f'teacher rows {a}-{b} done ({time.monotonic() - t0:.0f} s)')
    m = dict(status='complete' if len(done) == len(blocks) else 'partial', model=TEACHER_MODEL,
             store_index_sha256=store.manifest.get('index_sha256'), frame_rows_sha256=rows_sha,
             revision=TEACHER_REVISION, model_dir=str(model_dir).replace('\\', '/'), weights_sha256=weights_sha,
             licence=lic, input_hw=list(TEACHER_INPUT), resize='bicubic 448x252 -> 602x336, ImageNet normalisation',
             output='relative inverse depth, bilinear to 448x252, 7x7 block mean -> (36, 64) float16',
             precision='fp16', frames=n, seconds=round(time.monotonic() - t0, 1), gpu_seconds=round(gpu_s, 1),
             guard=guard.summary(), use='shape regulariser / M2 L4 anchor fits only; never a runtime input')
    mpath.write_text(json.dumps(m, indent=1) + '\n', encoding='utf-8')
    return m
