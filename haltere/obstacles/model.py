"""ClearanceNet: model definition and output contract (runtime-safe).

Inputs (one frame; nothing else by default):

- ``image``: float (B, 4, 252, 448). Channels 0-2: (rgb / 255 - IMAGENET_MEAN) / IMAGENET_STD of
  ``overlays.masked_input(frame)``; channel 3: ``overlays.validity_channel`` (0 masked, 0.5 propeller
  zone, 1 scene).
- ``gravity``: float (B, 3) unit gravity in the camera frame, ``contract.gravity_camera(quat)``
  (FiLM conditioning, and the fan geometry below).

No speed, position, velocity, ring position, pilot state or command input. The optional temporal
variant (M2) may add cached previous-frame tokens plus the relative pose; it is adopted only if it
wins F12 inner validation AND passes the speed-perturbation leak test (E8).

Outputs (dict of tensors; OUTPUT_SPEC):

- ``grid_log_range`` (B, 2, 18, 32): ln(range m) quantiles [q20, q50] per grid cell (the cell's
  minimum range, contract docstring); q20 <= q50 by construction (q20 = q50 - softplus(.)).
- ``fan_log_free`` (B, 2, 4, 9): ln(first-blocked distance m) quantiles [q20, q50] per fan direction.
- ``fan_logit`` (B, 2, 4, 9): logits of [P(blocked <= 4 m), P(blocked <= 8 m)].

Architectures (v0, M1):

- ``dav2s``: Depth-Anything-V2-Small relative weights (Apache-2.0, depth-anything/Depth-Anything-V2-Small-hf,
  the local copy in runs/dense-depth-probe-20260923/model; nothing is downloaded) at 252 x 448 (18 x 32
  tokens). The validity channel enters through a zero-initialised fourth input channel of the patch embedding.
  Cell features = DPT fusion features at 18 x 32, the 144 x 256 fusion map average- and max-pooled per cell
  (8 x 8 = one 14 x 14 px cell), the last ViT tokens, and the pretrained head's relative disparity
  (cell max and mean, log, per-frame centred, so only its shape is used).
- ``resnet18fpn``: ResNet18 + FPN control arm with the same heads. Random initialisation: no ImageNet
  ResNet18 weights exist on this machine and none were downloaded (licence rule), so the control also
  tests whether the pretrained geometry prior matters at all.

Heads (shared, ``ClearanceHeads``): gravity FiLM on the cell features plus three per-cell geometry channels
derived from gravity alone (elevation of the cell ray, its components along the level heading x_h and y_h);
grid = 1 x 1 conv to (q50, spread). Fan: every fan direction is projected into the image from the gravity
vector alone (``fan_projection``: the heading frame of contract.heading_frame expressed in camera axes, so
no attitude beyond gravity is needed); the cell features and the grid q50 (and its 3 x 3 minimum) are
bilinearly sampled at that point, and an MLP with a learned direction embedding, a global context vector,
the projected position and an in-view flag predicts (q50 residual on the sampled 3 x 3 minimum, spread,
two blocked logits tied to the distance by learnable slopes).

Files: ``runs/obstacle-train/<FOLD>-v<k>/model.pt`` (state_dict, fp32) and ``model.json`` next to it
with MODEL_CARD_KEYS. ``model_sha256`` = sha256 of model.pt bytes; runtime refuses a model whose
sidecar hash differs. Runtime (M4) runs fp16 + CUDA graph at 252 x 448; offline evaluation eager fp32.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from .contract import (BLOCKED_WITHIN_M, FAN_ELEV_DEG, FAN_SHAPE, FAN_YAW_DEG, FOCAL_PX, GRID_H, GRID_SHAPE,
                       GRID_W, IMAGE_H, IMAGE_W, PATCH_PX, QUANTILES, TILT_DEG)

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

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE_DIRS = (REPO_ROOT / 'runs' / 'dense-depth-probe-20260923',
               Path('C:/DEV/Haltere/runs/dense-depth-probe-20260923'))
DAV2S_MODEL_ID = 'depth-anything/Depth-Anything-V2-Small-hf'
DAV2S_LICENCE = 'Apache-2.0'
# config.json of the local DA-V2-Small relative checkpoint (architecture only; weights are loaded separately).
DAV2S_CONFIG = {
    'architectures': ['DepthAnythingForDepthEstimation'],
    'backbone': None,
    'backbone_config': {'architectures': ['Dinov2Model'], 'hidden_size': 384, 'image_size': 518,
                        'model_type': 'dinov2', 'num_attention_heads': 6,
                        'out_features': ['stage3', 'stage6', 'stage9', 'stage12'], 'out_indices': [3, 6, 9, 12],
                        'patch_size': 14, 'reshape_hidden_states': False},
    'fusion_hidden_size': 64, 'head_hidden_size': 32, 'head_in_index': -1, 'initializer_range': 0.02,
    'model_type': 'depth_anything', 'neck_hidden_sizes': [48, 96, 192, 384], 'patch_size': 14,
    'reassemble_factors': [4, 2, 1, 0.5], 'reassemble_hidden_size': 384, 'use_pretrained_backbone': False,
}
LN_8 = math.log(8.0)
FAN_N = FAN_SHAPE[0] * FAN_SHAPE[1]


def dav2s_weights_dir() -> Path | None:
    for d in _PROBE_DIRS:
        if (d / 'model' / 'model.safetensors').exists():
            return d / 'model'
    return None


def _import_transformers():
    """transformers from the environment, else the vendored copy next to the depth probe (offline, no install)."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        for d in _PROBE_DIRS:
            deps = d / 'dependencies'
            if deps.exists():
                if str(deps) not in sys.path:
                    sys.path.insert(0, str(deps))
                break
        import transformers  # noqa: F401
    import transformers
    return transformers


# ----------------------------------------------------------------------------- geometry from gravity

def _geometry_constants():
    """Cell rays (3, 18, 32) and body x (3,) in camera axes; fan directions (36, 3) in the heading frame."""
    v, u = np.mgrid[:GRID_H, :GRID_W].astype(np.float64)
    x = (u * PATCH_PX + PATCH_PX / 2.0 - IMAGE_W / 2.0) / FOCAL_PX
    y = (v * PATCH_PX + PATCH_PX / 2.0 - IMAGE_H / 2.0) / FOCAL_PX
    rays = np.stack([x, y, np.ones_like(x)])
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)
    t = np.deg2rad(TILT_DEG)
    # body x in camera axes (right, down, forward): camera forward is body x tilted up by t about body y
    body_x = np.array([0.0, np.sin(t), np.cos(t)])
    th = np.deg2rad(FAN_ELEV_DEG)[:, None]
    ps = np.deg2rad(FAN_YAW_DEG)[None, :]
    d = np.stack([np.cos(th) * np.cos(ps), np.cos(th) * np.sin(ps), np.sin(th) * np.ones_like(ps)], -1)
    return rays.astype(np.float32), body_x.astype(np.float32), d.reshape(-1, 3).astype(np.float32)


def heading_axes_camera(gravity, body_x):
    """Level heading axes (x_h forward, y_h left, up) in camera axes from unit gravity (B, 3); + valid (B,).

    Same rule as contract.heading_frame: x_h = horizontal part of the optical axis (camera z), or of body x
    when the optical axis is within ~11.5 deg of vertical; invalid when both are degenerate.
    """
    import torch
    g = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    up = -g
    ez = torch.zeros_like(up)
    ez[:, 2] = 1.0
    fwd = ez - up[:, 2:3] * up
    n = fwd.norm(dim=-1, keepdim=True)
    bx = body_x.to(up).expand_as(up)
    alt = bx - (bx * up).sum(-1, keepdim=True) * up
    n_alt = alt.norm(dim=-1, keepdim=True)
    use_alt = n < 0.2
    fwd = torch.where(use_alt, alt, fwd)
    n = torch.where(use_alt, n_alt, n)
    valid = (n >= 0.2).squeeze(-1)
    x_h = fwd / n.clamp_min(1e-6)
    y_h = torch.cross(up, x_h, dim=-1)
    return x_h, y_h, up, valid


def fan_projection(gravity, body_x, fan_dirs_h):
    """Fan directions projected into the store image from gravity alone.

    Returns (uv_norm (B, 36, 2) in grid_sample coordinates clamped to [-1, 1], in_view (B, 36) float,
    d_cam (B, 36, 3) unit directions in camera axes).
    """
    import torch
    x_h, y_h, up, valid = heading_axes_camera(gravity, body_x)
    d = fan_dirs_h.to(gravity)
    d_c = d[None, :, 0:1] * x_h[:, None] + d[None, :, 1:2] * y_h[:, None] + d[None, :, 2:3] * up[:, None]
    z = d_c[..., 2]
    front = z > 0.05
    zs = torch.where(front, z, torch.full_like(z, 0.05))
    u = IMAGE_W / 2.0 + FOCAL_PX * d_c[..., 0] / zs
    v = IMAGE_H / 2.0 + FOCAL_PX * d_c[..., 1] / zs
    inside = front & (u >= 0) & (u < IMAGE_W) & (v >= 0) & (v < IMAGE_H) & valid[:, None]
    uvn = torch.stack([u / IMAGE_W * 2.0 - 1.0, v / IMAGE_H * 2.0 - 1.0], -1).clamp(-1.0, 1.0)
    return uvn, inside.to(gravity.dtype), d_c


def cell_geometry(gravity, rays, body_x):
    """(B, 3, 18, 32): per cell ray, sin(elevation) and its components along x_h and y_h (from gravity alone)."""
    import torch
    x_h, y_h, up, _ = heading_axes_camera(gravity, body_x)
    r = rays.to(gravity)                                         # (3, 18, 32)
    def dot(a):
        return torch.einsum('bc,chw->bhw', a, r)
    return torch.stack([dot(up), dot(x_h), dot(y_h)], 1)


def cell_heading_angle_deg(gravity) -> np.ndarray:
    """numpy (B, 18, 32): angle between each cell's centre ray and the level heading x_h (training weights)."""
    import torch
    rays, body_x, _ = _geometry_constants()
    g = torch.as_tensor(np.asarray(gravity, np.float32))
    x_h, _, _, valid = heading_axes_camera(g, torch.as_tensor(body_x))
    c = torch.einsum('bc,chw->bhw', x_h, torch.as_tensor(rays)).clamp(-1, 1)
    ang = torch.rad2deg(torch.arccos(c)).numpy()
    ang[~valid.numpy()] = 180.0
    return ang


# ----------------------------------------------------------------------------- modules

def _build_torch_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GravityFiLM(nn.Module):
        def __init__(self, ch: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 2 * ch))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, x, g):
            gamma, beta = self.net(g).chunk(2, dim=-1)
            return x * (1.0 + gamma[..., None, None]) + beta[..., None, None]

    class ResBlock(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
            self.n1 = nn.GroupNorm(8, ch)
            self.n2 = nn.GroupNorm(8, ch)

        def forward(self, x):
            h = self.c1(F.gelu(self.n1(x)))
            h = self.c2(F.gelu(self.n2(h)))
            return x + h

    class ClearanceHeads(nn.Module):
        def __init__(self, in_ch: int, ch: int = 128, dir_dim: int = 32, ctx_dim: int = 64, hidden: int = 256):
            super().__init__()
            rays, body_x, fan_dirs = _geometry_constants()
            self.register_buffer('rays', torch.from_numpy(rays), persistent=False)
            self.register_buffer('body_x', torch.from_numpy(body_x), persistent=False)
            self.register_buffer('fan_dirs', torch.from_numpy(fan_dirs), persistent=False)
            self.inp = nn.Conv2d(in_ch + 3, ch, 1)
            self.film = GravityFiLM(ch)
            self.blocks = nn.Sequential(ResBlock(ch), ResBlock(ch))
            self.film2 = GravityFiLM(ch)
            self.grid_out = nn.Conv2d(ch, 2, 1)
            nn.init.normal_(self.grid_out.weight, std=1e-3)
            with torch.no_grad():
                self.grid_out.bias.copy_(torch.tensor([LN_8, 0.0]))
            self.dir_embed = nn.Parameter(torch.randn(FAN_N, dir_dim) * 0.02)
            self.ctx = nn.Linear(ch, ctx_dim)
            fan_in = ch + 2 + dir_dim + ctx_dim + 3 + 3
            self.fan_mlp = nn.Sequential(nn.Linear(fan_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                         nn.Linear(hidden, 4))
            nn.init.normal_(self.fan_mlp[-1].weight, std=1e-3)
            nn.init.zeros_(self.fan_mlp[-1].bias)
            # blocked-within-d logits tied to the distance: a_d * (ln d - q50) + residual
            self.logit_slope = nn.Parameter(torch.full((2,), 2.0))
            self.register_buffer('ln_within', torch.log(torch.tensor(BLOCKED_WITHIN_M, dtype=torch.float32)),
                                 persistent=False)

        def forward(self, feats, g):
            B = feats.shape[0]
            g = g.to(feats.dtype)
            geo = cell_geometry(g.float(), self.rays, self.body_x).to(feats.dtype)
            x = self.inp(torch.cat([feats, geo], 1))
            x = F.gelu(self.film(x, g))
            x = self.blocks(x)
            x = self.film2(x, g)
            go = self.grid_out(F.gelu(x)).float()
            q50 = go[:, 0]
            q20 = q50 - F.softplus(go[:, 1]) - 1e-3
            grid = torch.stack([q20, q50], 1)
            # fan: sample features and ranges where each direction projects
            uvn, inview, _ = fan_projection(g.float(), self.body_x, self.fan_dirs)
            samp_grid = uvn[:, None].to(x.dtype)                                  # (B, 1, 36, 2)
            fs = F.grid_sample(x, samp_grid, mode='bilinear', padding_mode='border', align_corners=False)
            fs = fs[:, :, 0].transpose(1, 2)                                      # (B, 36, ch)
            qmin = -F.max_pool2d(-q50[:, None], 3, stride=1, padding=1)
            qs = F.grid_sample(torch.cat([q50[:, None], qmin], 1), uvn[:, None].float(), mode='bilinear',
                               padding_mode='border', align_corners=False)[:, :, 0].transpose(1, 2)   # (B, 36, 2)
            ctx = self.ctx(x.float().mean((2, 3)).to(x.dtype))
            h = torch.cat([fs.float(), qs, self.dir_embed[None].expand(B, -1, -1).float(),
                           ctx[:, None].expand(-1, FAN_N, -1).float(), uvn.float(), inview[..., None].float(),
                           g.float()[:, None].expand(-1, FAN_N, -1)], -1)
            o = self.fan_mlp(h.to(x.dtype)).float()                               # (B, 36, 4)
            f50 = qs[..., 1] + o[..., 0]
            f20 = f50 - F.softplus(o[..., 1]) - 1e-3
            logit = o[..., 2:4] + self.logit_slope[None, None] * (self.ln_within[None, None] - f50[..., None])
            fan = torch.stack([f20, f50], 1).reshape(B, 2, *FAN_SHAPE)
            logit = logit.permute(0, 2, 1).reshape(B, 2, *FAN_SHAPE)
            return {'grid_log_range': grid, 'fan_log_free': fan, 'fan_logit': logit}

    class DAv2Clearance(nn.Module):
        arch = 'dav2s'

        def __init__(self, pretrained: bool = False, weights_dir=None):
            super().__init__()
            tf = _import_transformers()
            from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
            del tf
            if pretrained:
                wd = Path(weights_dir) if weights_dir else dav2s_weights_dir()
                if wd is None:
                    raise FileNotFoundError('DA-V2-Small relative weights not found locally (no download)')
                self.da = DepthAnythingForDepthEstimation.from_pretrained(str(wd), local_files_only=True)
                self.weights_dir = str(wd)
            else:
                self.da = DepthAnythingForDepthEstimation(DepthAnythingConfig.from_dict(dict(DAV2S_CONFIG)))
                self.weights_dir = None
            self._expand_patch_embed()
            self.tok_proj = nn.Conv2d(384, 64, 1)
            self.pix = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
            self.heads = ClearanceHeads(in_ch=64 + 128 + 64 + 2)

        def _expand_patch_embed(self):
            pe = self.da.backbone.embeddings.patch_embeddings
            old = pe.projection
            new = nn.Conv2d(INPUT_CHANNELS, old.out_channels, kernel_size=old.kernel_size, stride=old.stride)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :3] = old.weight
                new.bias.copy_(old.bias)
            pe.projection = new
            pe.num_channels = INPUT_CHANNELS

        def layer_groups(self):
            """(name, params, depth) for layer-wise learning-rate decay: 0 embeddings .. 12 last block, 13 rest."""
            groups = []
            bb = self.da.backbone
            groups.append(('embed', list(bb.embeddings.parameters()), 0))
            for i, layer in enumerate(bb.encoder.layer):
                groups.append((f'block{i}', list(layer.parameters()), i + 1))
            groups.append(('backbone_norm', list(bb.layernorm.parameters()), len(bb.encoder.layer)))
            groups.append(('dpt', list(self.da.neck.parameters()) + list(self.da.head.parameters()), None))
            new = list(self.tok_proj.parameters()) + list(self.pix.parameters()) + list(self.heads.parameters())
            groups.append(('heads', new, None))
            return groups

        def forward(self, image, gravity, aux: bool = False):
            B, _, H, W = image.shape
            ph, pw = H // PATCH_PX, W // PATCH_PX
            bb = self.da.backbone(image)
            feats = list(bb.feature_maps)
            fused = self.da.neck(feats, ph, pw)
            disp = self.da.head(fused, ph, pw)                                    # (B, 252, 448) >= 0
            tok = feats[-1][:, 1:].reshape(B, ph, pw, -1).permute(0, 3, 1, 2)
            p = self.pix(fused[-1])                                               # (B, 64, 144, 256)
            sh = p.shape[-2] // ph
            pav = F.avg_pool2d(p, sh)
            pmx = F.max_pool2d(p, sh)
            d = disp.float()[:, None]
            dmax = torch.log(F.max_pool2d(d, PATCH_PX) + 1e-3)
            davg = torch.log(F.avg_pool2d(d, PATCH_PX) + 1e-3)
            off = davg.mean((2, 3), keepdim=True)
            dfeat = torch.cat([dmax - off, davg - off], 1).to(p.dtype)
            x = torch.cat([fused[0], pav, pmx, self.tok_proj(tok), dfeat], 1)
            out = self.heads(x, gravity)
            if aux:
                out['disparity'] = disp
            return out

    class ResNetFPNClearance(nn.Module):
        arch = 'resnet18fpn'

        def __init__(self, pretrained: bool = False, fpn_ch: int = 96):
            super().__init__()
            from torchvision.models import resnet18
            r = resnet18(weights=None)          # no ImageNet weights on this machine; none downloaded
            self.conv1 = nn.Conv2d(INPUT_CHANNELS, 64, 7, stride=2, padding=3, bias=False)
            self.bn1, self.relu, self.maxpool = r.bn1, r.relu, r.maxpool
            self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
            self.lat = nn.ModuleList([nn.Conv2d(c, fpn_ch, 1) for c in (64, 128, 256, 512)])
            self.smooth = nn.ModuleList([nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1) for _ in range(4)])
            self.heads = ClearanceHeads(in_ch=4 * fpn_ch)

        def layer_groups(self):
            heads = list(self.heads.parameters())
            ids = {id(p) for p in heads}
            rest = [p for p in self.parameters() if id(p) not in ids]
            return [('backbone', rest, None), ('heads', heads, None)]

        def forward(self, image, gravity, aux: bool = False):
            x = self.maxpool(self.relu(self.bn1(self.conv1(image))))
            c2 = self.layer1(x)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            p5 = self.lat[3](c5)
            p4 = self.lat[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
            p3 = self.lat[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
            p2 = self.lat[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
            p2, p3, p4 = self.smooth[0](p2), self.smooth[1](p3), self.smooth[2](p4)
            p5 = self.smooth[3](p5)
            size = GRID_SHAPE
            x = torch.cat([F.adaptive_avg_pool2d(p2, size), F.adaptive_max_pool2d(p2, size),
                           F.interpolate(p3, size=size, mode='bilinear', align_corners=False),
                           F.interpolate(p4, size=size, mode='bilinear', align_corners=False)], 1)
            return self.heads(x, gravity)

    return dict(DAv2Clearance=DAv2Clearance, ResNetFPNClearance=ResNetFPNClearance, ClearanceHeads=ClearanceHeads)


_MODULES = None


def _modules():
    global _MODULES
    if _MODULES is None:
        _MODULES = _build_torch_modules()
    return _MODULES


def build_model(arch: str = 'dav2s', *, pretrained: bool = False, weights_dir=None):
    """-> torch.nn.Module with forward(image (B,4,252,448), gravity (B,3)) -> dict per OUTPUT_SPEC.

    ``pretrained=True`` (training only) loads the local DA-V2-Small relative weights; evaluation builds the
    architecture and loads model.pt (baselines.load_model).
    """
    if arch not in ARCHS:
        raise ValueError(f'arch must be one of {ARCHS}')
    m = _modules()
    if arch == 'dav2s':
        return m['DAv2Clearance'](pretrained=pretrained, weights_dir=weights_dir)
    return m['ResNetFPNClearance'](pretrained=pretrained)


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
