"""GateNet: a small convolutional network that finds the next race gate in an FPV frame.

Input: an RGB frame of the game view (320 x 180). Output: four numbers - is a gate visible (logit),
the gate centre's normalised image position u, v in [-1, 1] (x right, y down), and the log of its
apparent width in pixels (which gives the distance through the camera's focal length and the
gate's known size). It stands in for the fly's visual system: the pilot turns its output into the
goal vector the connectome brain expects.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_W, IN_H = 320, 180
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _block(cin: int, cout: int, k: int = 3, s: int = 2) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride=s, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class GateNet(nn.Module):
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


def decode(out: torch.Tensor) -> torch.Tensor:
    """(B, 4) raw outputs -> (B, 4) [p_visible, u, v, width_px]."""
    p = torch.sigmoid(out[:, 0])
    return torch.stack([p, out[:, 1].clamp(-1.2, 1.2), out[:, 2].clamp(-1.2, 1.2), 100.0 * torch.exp(out[:, 3])], dim=1)
