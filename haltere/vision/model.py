"""GateNet: a small convolutional network that finds the next race gate in an FPV frame.

Input: an RGB frame of the game view (320 x 180). Output: four numbers - is a gate visible (logit),
the gate centre's normalised image position u, v in [-1, 1] (x right, y down), and the log of its
apparent width in pixels (which gives the distance through the camera's focal length and the
gate's known size). It stands in for the fly's visual system: the pilot turns its output into the
goal vector the connectome brain expects.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_W, IN_H = 320, 180
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _block(cin: int, cout: int, k: int = 3, s: int = 2) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride=s, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class GateNet(nn.Module):
    architecture = 'regression'

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(3, w, 5, 2),        # 90 x 160
            _block(w, w, 3, 1),
            _block(w, 2 * w, 3, 2),    # 45 x 80
            _block(2 * w, 2 * w, 3, 1),
            _block(2 * w, 4 * w, 3, 2),    # 23 x 40
            _block(4 * w, 4 * w, 3, 1),
            _block(4 * w, 8 * w, 3, 2),    # 12 x 20
            _block(8 * w, 8 * w, 3, 2),    # 6 x 10
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8 * w * 6 * 10, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                  nn.Linear(256, 4))
        self.register_buffer('mean', MEAN.clone())
        self.register_buffer('std', STD.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 180, 320) in [0, 1] -> (B, 4): [visible logit, u, v, log(width_px / 100)]."""
        x = (x - self.mean) / self.std
        return self.head(self.features(x))

    def predict_and_loss(self, x, target):
        out = self(x)
        loss, parts = self.loss(out, target)
        return out, loss, parts

    @staticmethod
    def loss(out: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """target: (B, 4) [visible, u, v, logw]; position/size terms only count for visible gates."""
        vis = target[:, 0]
        l_vis = F.binary_cross_entropy_with_logits(out[:, 0], vis)
        m = vis > 0.5
        if m.any():
            l_pos = F.smooth_l1_loss(out[m, 1:3], target[m, 1:3], beta=0.05)
            l_size = F.smooth_l1_loss(out[m, 3], target[m, 3], beta=0.1)
        else:
            l_pos = l_size = out.sum() * 0
        return l_vis + 3.0 * l_pos + 0.5 * l_size, {'vis': float(l_vis), 'pos': float(l_pos), 'size': float(l_size)}


class SpatialGateNet(GateNet):
    """Localize an opening with a dense image map instead of an absolute-position MLP.

    The public four-value sensory contract stays unchanged. A spatial training
    target makes off-centre arches distinguishable from the common central
    approach. Background images suppress every candidate location. There is no
    route, gate identity, pose or future trajectory in this model.
    """
    architecture = 'spatial_v1'
    grid_height, grid_width = 23, 40

    def __init__(self, width=32):
        super().__init__(width)
        del self.head
        self.spatial_head = nn.Sequential(
            _block(12 * width, 4 * width, 3, 1),
            _block(4 * width, 2 * width, 3, 1),
            nn.Conv2d(2 * width, 2, 1),
        )

    @staticmethod
    def coordinates(reference):
        y = (torch.arange(SpatialGateNet.grid_height, device=reference.device,
                          dtype=reference.dtype) + .5) * (2 / SpatialGateNet.grid_height) - 1
        x = (torch.arange(SpatialGateNet.grid_width, device=reference.device,
                          dtype=reference.dtype) + .5) * (2 / SpatialGateNet.grid_width) - 1
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack((xx, yy), -1).reshape(-1, 2)

    @classmethod
    def decode_maps(cls, maps):
        scores = maps[:, 0].flatten(1)
        probability = scores.softmax(-1)
        centre = probability @ cls.coordinates(maps)
        log_width = (probability * maps[:, 1].flatten(1)).sum(-1)
        visibility = scores.logsumexp(-1) - math.log(scores.shape[1])
        return torch.cat((visibility[:, None], centre, log_width[:, None]), -1)

    def forward_details(self, x):
        early = self.features[:6]((x - self.mean) / self.std)
        context = self.features[6:](early)
        context = F.interpolate(context, size=early.shape[-2:], mode='bilinear', align_corners=False)
        maps = self.spatial_head(torch.cat((early, context), 1))
        return self.decode_maps(maps), maps

    def forward(self, x):
        return self.forward_details(x)[0]

    def predict_and_loss(self, x, target):
        out, maps = self.forward_details(x)
        loss, parts = self.loss(out, target)
        visible = target[:, 0] > .5
        if visible.any():
            scale = target.new_tensor([self.grid_width / 2, self.grid_height / 2])
            distance = (self.coordinates(target)[None] - target[visible, None, 1:3]) * scale
            distribution = (-distance.square().sum(-1) / (2 * .8 ** 2)).softmax(-1)
            location = -(distribution * maps[visible, 0].flatten(1).log_softmax(-1)).sum(-1).mean()
        else:
            location = maps.sum() * 0
        parts['heatmap'] = float(location.detach())
        return out, loss + .5 * location, parts


def make_gatenet(architecture='regression', width=32):
    if architecture == 'regression':
        return GateNet(width)
    if architecture == 'spatial_v1':
        return SpatialGateNet(width)
    raise ValueError(f'Unknown gate detector architecture: {architecture}')


def decode(out: torch.Tensor) -> torch.Tensor:
    """(B, 4) raw outputs -> (B, 4) [p_visible, u, v, width_px]."""
    p = torch.sigmoid(out[:, 0])
    return torch.stack([p, out[:, 1].clamp(-1.2, 1.2), out[:, 2].clamp(-1.2, 1.2), 100.0 * torch.exp(out[:, 3])], dim=1)
