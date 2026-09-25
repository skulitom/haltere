"""Frozen pretrained relative depth for runtime cues: Depth-Anything-V2-Small (Apache-2.0). No training, no download.

Wraps the relative model already on this machine (``runs/dense-depth-probe-20260923/model``, Hugging Face
``depth-anything/Depth-Anything-V2-Small-hf`` revision 5426e4f0; the same weights as the obstacle teacher
cache, weights sha256 3152477c...). Input: 448 x 252 RGB uint8 frames (haltere.obstacles.overlays.
to_model_frame / the store frame), fed UNMASKED as in the teacher cache; ImageNet normalisation; by default
at 448 x 252 itself (32 x 18 patches of 14 px, no resize) in fp16 on CUDA, fp32 on the CPU when CUDA is
unavailable (about 0.3 s per frame with two threads: too slow to keep a 17 Hz cue fresh, so a CPU-only
process sees the gap cue go stale rather than act on old frames). ``input_hw=(336, 602)`` reproduces the
teacher cache preprocessing (bicubic up-resize); outputs are then resized back to 252 x 448 (bilinear).

Output: relative inverse depth (affine-invariant: only ratios within a frame are meaningful), averaged over
``block_px`` x ``block_px`` pixel blocks: (B, 36, 64) float32 for the default 7 px, the grid of the teacher
cache and of haltere.vision.gap_cue. ``provenance`` records the model id, revision, weights sha256, licence,
device, precision and input size; loading refuses weights whose sha256 differs from the expected one unless
``expected_sha256=None``.

Runtime-safe: imports numpy at module level and torch / transformers only when a model is built; never
imports offline label code. ``transformers`` comes from the environment or, like the teacher, from the
vendored copy next to the depth probe (``dependencies/``).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_ID = 'depth-anything/Depth-Anything-V2-Small-hf'
MODEL_REVISION = '5426e4f0f36572d16453bbda7a8389317b1bef99'
MODEL_LICENCE = 'apache-2.0'
EXPECTED_WEIGHTS_SHA256 = '3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70'
FRAME_HW = (252, 448)
RUNTIME_INPUT_HW = (252, 448)        # multiples of 14: no resize
TEACHER_INPUT_HW = (336, 602)        # the obstacle teacher cache preprocessing
BLOCK_PX = 7
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
MODEL_DIR_ENV = 'HALTERE_RELATIVE_DEPTH_MODEL_DIR'
_REL_MODEL_DIR = Path('runs') / 'dense-depth-probe-20260923' / 'model'
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FALLBACK_ROOT = Path('C:/DEV/Haltere')


def _roots():
    roots = [os.environ.get('HALTERE_DATA_ROOT'), str(_REPO_ROOT), str(_FALLBACK_ROOT)]
    return [Path(r) for r in roots if r]


def find_model_dir(explicit=None) -> Path | None:
    """The local DA-V2-Small relative model directory (config.json + model.safetensors), or None."""
    cands = [explicit, os.environ.get(MODEL_DIR_ENV), os.environ.get('HALTERE_TEACHER_MODEL_DIR')]
    cands += [str(r / _REL_MODEL_DIR) for r in _roots()]
    for c in cands:
        if c and (Path(c) / 'model.safetensors').exists() and (Path(c) / 'config.json').exists():
            return Path(c)
    return None


def model_licence(model_dir: Path) -> str | None:
    card = Path(model_dir) / 'README.md'
    if not card.exists():
        return None
    for line in card.read_text(encoding='utf-8', errors='replace').splitlines()[:30]:
        if line.strip().lower().startswith('license:'):
            return line.split(':', 1)[1].strip()
    return None


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _import_transformers(model_dir: Path):
    try:
        import transformers  # noqa: F401
        return
    except ImportError:
        pass
    for deps in (Path(model_dir).parent / 'dependencies',):
        if (deps / 'transformers').exists():
            if str(deps) not in sys.path:
                sys.path.append(str(deps))
            import transformers  # noqa: F401
            return
    raise ImportError('transformers is not importable (environment or the vendored probe dependencies)')


def block_mean(disp: np.ndarray, block: int = BLOCK_PX) -> np.ndarray:
    """(..., H, W) -> (..., H // block, W // block) mean over block x block squares."""
    d = np.asarray(disp, np.float32)
    h, w = d.shape[-2:]
    return d.reshape(d.shape[:-2] + (h // block, block, w // block, block)).mean(axis=(-3, -1))


class RelativeDepth:
    """``RelativeDepth()(frames)`` -> (B, 36, 64) float32 block-mean relative disparity (see module doc)."""

    def __init__(self, model_dir=None, *, device: str | None = None, fp16: bool = True,
                 input_hw: tuple = RUNTIME_INPUT_HW, block_px: int = BLOCK_PX, threads: int = 2,
                 expected_sha256: str | None = EXPECTED_WEIGHTS_SHA256):
        md = find_model_dir(model_dir)
        if md is None:
            raise FileNotFoundError(f'{MODEL_ID} is not on this machine (set {MODEL_DIR_ENV}); nothing is downloaded')
        lic = model_licence(md)
        if (lic or '').lower() != MODEL_LICENCE:
            raise RuntimeError(f'{md}: model card licence {lic!r} is not {MODEL_LICENCE}')
        sha = sha256_file(md / 'model.safetensors')
        if expected_sha256 is not None and sha != expected_sha256:
            raise RuntimeError(f'{md}: weights sha256 {sha[:12]} is not the expected {expected_sha256[:12]}')
        if input_hw[0] % 14 or input_hw[1] % 14:
            raise ValueError('input_hw must be multiples of the 14 px patch')
        if FRAME_HW[0] % block_px or FRAME_HW[1] % block_px:
            raise ValueError(f'block_px must divide {FRAME_HW}')
        _import_transformers(md)
        import torch
        from transformers import DepthAnythingForDepthEstimation
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            torch.set_num_threads(int(threads))
        self.torch = torch
        self.device = torch.device(device)
        self.half = bool(fp16) and self.device.type == 'cuda'
        model = DepthAnythingForDepthEstimation.from_pretrained(str(md), local_files_only=True).eval()
        model = model.to(self.device)
        if self.half:
            model = model.half()
        self.model = model
        self.input_hw = tuple(int(x) for x in input_hw)
        self.block_px = int(block_px)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self.provenance = dict(model=MODEL_ID, revision=MODEL_REVISION, model_dir=str(md).replace('\\', '/'),
                               weights_sha256=sha, licence=lic, device=str(self.device),
                               precision='fp16' if self.half else 'fp32', input_hw=list(self.input_hw),
                               frame_hw=list(FRAME_HW), block_px=self.block_px, masked_input=False,
                               output='relative inverse depth (affine-invariant), block mean')

    @property
    def grid_shape(self) -> tuple:
        return FRAME_HW[0] // self.block_px, FRAME_HW[1] // self.block_px

    def _prep(self, frames):
        torch = self.torch
        import torch.nn.functional as F
        a = np.asarray(frames)
        if a.ndim == 3:
            a = a[None]
        if a.shape[1:] != FRAME_HW + (3,) or a.dtype != np.uint8:
            raise ValueError(f'expected (B, 252, 448, 3) uint8 RGB frames, got {a.shape} {a.dtype}')
        x = torch.from_numpy(np.ascontiguousarray(a)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        if self.input_hw != FRAME_HW:
            x = F.interpolate(x, size=self.input_hw, mode='bicubic', align_corners=False).clamp(0, 1)
        x = (x - self.mean) / self.std
        return x.half() if self.half else x

    def infer_full(self, frames) -> np.ndarray:
        """(B, 252, 448) float32 relative disparity at frame resolution."""
        torch = self.torch
        import torch.nn.functional as F
        with torch.inference_mode():
            out = self.model(pixel_values=self._prep(frames)).predicted_depth[:, None].float()
            if tuple(out.shape[-2:]) != FRAME_HW:
                out = F.interpolate(out, size=FRAME_HW, mode='bilinear', align_corners=False)
            return out[:, 0].cpu().numpy()

    def __call__(self, frames) -> np.ndarray:
        """(B, 252 / block, 448 / block) float32 block-mean relative disparity."""
        torch = self.torch
        import torch.nn.functional as F
        with torch.inference_mode():
            out = self.model(pixel_values=self._prep(frames)).predicted_depth[:, None].float()
            if tuple(out.shape[-2:]) != FRAME_HW:
                out = F.interpolate(out, size=FRAME_HW, mode='bilinear', align_corners=False)
            return F.avg_pool2d(out, self.block_px)[:, 0].cpu().numpy()

    def latency_ms(self, n: int = 20) -> float:
        """Median single-frame latency (ms) on a mid-grey frame, after two warm-up calls."""
        f = np.full(FRAME_HW + (3,), 128, np.uint8)
        for _ in range(2):
            self(f)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            self(f)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts) * 1000)
