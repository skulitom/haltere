"""ClearanceNet: model definition and output contract (runtime-safe). DELIVERED BY THE MODEL AGENT.

Inputs (one frame; nothing else by default):

- ``image``: float (B, 4, 252, 448). Channels 0-2: (rgb / 255 - IMAGENET_MEAN) / IMAGENET_STD of
  ``overlays.masked_input(frame)``; channel 3: ``overlays.validity_channel`` (0 masked, 0.5 propeller
  zone, 1 scene).
- ``gravity``: float (B, 3) unit gravity in the camera frame, ``contract.gravity_camera(quat)``
  (FiLM conditioning).

No speed, position, velocity, ring position, pilot state or command input. The optional temporal
variant (M2) may add cached previous-frame tokens plus the relative pose; it is adopted only if it
wins F12 inner validation AND passes the speed-perturbation leak test (E8).

Outputs (dict of tensors; OUTPUT_SPEC):

- ``grid_log_range`` (B, 2, 18, 32): ln(range m) quantiles [q20, q50] per grid cell (the cell's
  minimum range, contract docstring); q20 <= q50 by construction (q20 = q50 - softplus(.)).
- ``fan_log_free`` (B, 2, 4, 9): ln(first-blocked distance m) quantiles [q20, q50] per fan direction.
- ``fan_logit`` (B, 2, 4, 9): logits of [P(blocked <= 4 m), P(blocked <= 8 m)].

Architectures: 'dav2s' = Depth-Anything-V2-Small relative weights (Apache-2.0) at 252 x 448 (18 x 32
tokens), validity channel through a zero-initialised extra patch-embed weight, gravity FiLM;
'resnet18fpn' = ImageNet ResNet18 + FPN control arm with the same heads.

Files: ``runs/obstacle-train/<FOLD>-v<k>/model.pt`` (state_dict, fp32) and ``model.json`` next to it
with MODEL_CARD_KEYS. ``model_sha256`` = sha256 of model.pt bytes; runtime refuses a model whose
sidecar hash differs. Runtime (M4) runs fp16 + CUDA graph at 252 x 448; offline evaluation eager fp32.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .contract import BLOCKED_WITHIN_M, FAN_SHAPE, GRID_SHAPE, IMAGE_H, IMAGE_W, QUANTILES

ARCHS = ('dav2s', 'resnet18fpn')
INPUT_CHANNELS = 4
INPUT_SHAPE = (INPUT_CHANNELS, IMAGE_H, IMAGE_W)
OUTPUT_SPEC = {
    'grid_log_range': (len(QUANTILES),) + GRID_SHAPE,
    'fan_log_free': (len(QUANTILES),) + FAN_SHAPE,
    'fan_logit': (len(BLOCKED_WITHIN_M),) + FAN_SHAPE,
}
MODEL_CARD_KEYS = {
    'schema': 'haltere.obstacles.model.v1',
    'arch': 'dav2s | resnet18fpn',
    'fold': 'F12 | F3 | F4 | F5 | ALL',
    'train_envs': 'environments whose frames were used for fitting',
    'inner_val_envs': 'early-stopping environments (or "heldback flights" for ALL)',
    'held_out_envs': 'fold test environments (never seen)',
    'seen_environment_note': 'ALL: results on its training environments are seen-environment',
    'store_index_sha256': 'frame store index the data came from',
    'labels_manifest_sha256': 'sha256 of labels/manifest.json',
    'recipe': 'hyper-parameters, losses, augmentation, sampling caps',
    'epochs': 'epochs run and the selected epoch',
    'gpu_minutes': 'GPU wall minutes (chunks)',
    'code_commit': 'git commit',
    'model_sha256': 'sha256 of model.pt',
    'pretrained': 'backbone source, revision and licence',
    'inputs': 'masked RGB + validity + gravity_camera (no speed, route, ring or labels)',
    'created': 'ISO time',
}


def build_model(arch: str = 'dav2s'):
    """-> torch.nn.Module with forward(image (B,4,252,448), gravity (B,3)) -> dict per OUTPUT_SPEC."""
    if arch not in ARCHS:
        raise ValueError(f'arch must be one of {ARCHS}')
    raise NotImplementedError('ClearanceNet: delivered by the model build agent')


def model_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_model_card(model_path: str | Path, *, verify: bool = True) -> dict:
    """Read model.json next to model.pt and check its model_sha256 against the file."""
    model_path = Path(model_path)
    card = json.loads(model_path.with_name('model.json').read_text(encoding='utf-8'))
    if verify and card.get('model_sha256') != model_sha256(model_path):
        raise ValueError(f'{model_path}: sha256 does not match model.json')
    return card


def _np(x):
    if hasattr(x, 'detach'):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def outputs_to_numpy(out: dict) -> dict:
    """Model outputs -> evaluate.PredictionSet fields: metres and probabilities, float32.

    grid_q20/grid_q50 (B, 18, 32) m; fan_q20/fan_q50 (B, 4, 9) m; fan_p4/fan_p8 (B, 4, 9).
    """
    g = _np(out['grid_log_range'])
    f = _np(out['fan_log_free'])
    p = 1.0 / (1.0 + np.exp(-_np(out['fan_logit'])))
    return dict(grid_q20=np.exp(g[:, 0]), grid_q50=np.exp(g[:, 1]),
                fan_q20=np.exp(f[:, 0]), fan_q50=np.exp(f[:, 1]),
                fan_p4=p[:, 0].astype(np.float32), fan_p8=p[:, 1].astype(np.float32))
