"""Small causal visual path predictor for offline human-flight experiments.

This predicts body-frame paths for training supervision and offline diagnostics.
It is not a deployed navigation layer or an intended final product. The exported
fly brain must run without this teacher.
"""
from __future__ import annotations

import torch
from torch import nn


def rotation(q):
    """World-from-body matrix for (..., wxyz) quaternions."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def motion_features(velocity, attitude, time_s):
    """Causal inertial features without global position, heading or past controls."""
    R = rotation(attitude)
    gravity = R[..., 2, :]
    dt = torch.cat((torch.full_like(time_s[:, :1], 1/18), time_s[:, 1:] - time_s[:, :-1]), 1).clamp_min(.001)
    # Relative quaternion: previous orientation conjugate times current.
    a, b = attitude[:, :-1], attitude[:, 1:]
    scalar = (a * b).sum(-1, keepdim=True)
    vector = a[..., :1]*b[..., 1:] - b[..., :1]*a[..., 1:] - torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    vector = vector * torch.where(scalar < 0, -1., 1.)
    size = vector.norm(dim=-1, keepdim=True)
    angle = 2*torch.atan2(size, scalar.abs().clamp_min(1e-8))
    omega = vector * (angle / size.clamp_min(1e-8)) / dt[:, 1:, None]
    omega = torch.cat((torch.zeros_like(velocity[:, :1]), omega), 1)
    return torch.cat((velocity/20, gravity, omega.clamp(-12, 12)/6, dt[..., None]/.06), -1)


def mask_hud(images):
    """Fixed masks at the recording's aspect ratio: sticks, timer, compass, standings."""
    x = images.clone()
    h, w = x.shape[-2:]
    x[..., :round(.23*h), :] = 0
    x[..., round(.84*h):, :] = 0
    x[..., round(.34*h):round(.78*h), round(.82*w):] = 0
    return x


def constant_velocity(velocity, horizons):
    return velocity[..., None, :] * horizons[..., None]


def constant_acceleration(velocity, attitude, time_s, horizons):
    R = rotation(attitude)
    world_velocity = (R @ velocity[..., None]).squeeze(-1)
    dt = (time_s[:, 1:] - time_s[:, :-1]).clamp_min(.001)
    acc = (world_velocity[:, 1:] - world_velocity[:, :-1]) / dt[..., None]
    acc = torch.cat((torch.zeros_like(acc[:, :1]), acc), 1)
    body_acc = (R.transpose(-2, -1) @ acc[..., None]).squeeze(-1)
    return constant_velocity(velocity, horizons) + .5*body_acc[..., None, :]*horizons[..., None].square()


class NavigationNet(nn.Module):
    def __init__(self, vision=True, hidden=64, horizons=(.25, .5, 1.)):
        super().__init__()
        self.vision, self.hidden = vision, hidden
        self.register_buffer('horizons', torch.tensor(horizons, dtype=torch.float32))
        self.features = None
        if vision:
            layers, cin = [], 3
            for cout in (12, 24, 32, 48):
                layers.extend((nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GroupNorm(4, cout), nn.SiLU()))
                cin = cout
            self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((3, 5)), nn.Flatten(), nn.Linear(720, 64), nn.SiLU())
        self.recurrent = nn.GRU(74, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, len(horizons)*3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, images, velocity, attitude, time_s):
        b, t = velocity.shape[:2]
        if self.vision:
            pixels = mask_hud(images)
            features = self.features((pixels.flatten(0, 1)-.5)*2).reshape(b, t, 64)
        else:
            features = velocity.new_zeros(b, t, 64)
        features = torch.cat((features, motion_features(velocity, attitude, time_s)), -1)
        state, _ = self.recurrent(features)
        correction = self.head(state).reshape(b, t, len(self.horizons), 3)
        return constant_velocity(velocity, self.horizons) + correction*self.horizons[:, None].square()


class ResidualNavigationNet(nn.Module):
    """Frozen motion model plus a bounded correction; black images use the base."""

    def __init__(self, base, hidden=64, max_correction_m=.6):
        super().__init__()
        if base.vision or max_correction_m <= 0:
            raise ValueError('Expected a motion-only base and a positive correction bound')
        self.base, self.vision, self.hidden = base, True, hidden
        self.max_correction_m = float(max_correction_m)
        self.register_buffer('horizons', base.horizons.detach().clone())
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.corrector = NavigationNet(vision=True, hidden=hidden, horizons=base.horizons.tolist())

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, images, velocity, attitude, time_s):
        with torch.no_grad():
            base = self.base(images, velocity, attitude, time_s)
        raw = self.corrector(images, velocity, attitude, time_s) - constant_velocity(velocity, self.horizons)
        # Separate bound for each horizon: 0.6 m at 1 s, scaled by horizon squared.
        bound = self.max_correction_m*self.horizons[:, None].square()
        scaled = raw/bound
        delta = bound*scaled/(1+scaled.norm(dim=-1, keepdim=True))
        available = (mask_hud(images).flatten(2).abs().sum(-1) > 0)[..., None, None]
        return base + delta*available


def load_navigation(path, device='cpu'):
    """Load a self-contained inference checkpoint (including the frozen base)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('architecture') == 'residual_navigation':
        base = NavigationNet(vision=False, hidden=checkpoint['base_hidden'], horizons=checkpoint['horizons'])
        model = ResidualNavigationNet(base, hidden=checkpoint['hidden'], max_correction_m=checkpoint['max_correction_m'])
    else:
        model = NavigationNet(vision=checkpoint['vision'], hidden=checkpoint['hidden'], horizons=checkpoint['horizons'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint
