"""Sensory encoders: turn a drone signal into input currents for a population of real fly neurons."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PopulationEncoder(nn.Module):
    """Map a channel vector x [B, d] to input currents for n neurons [B, n].

    Every neuron has a preferred direction u_i (unit vector in channel space) and responds with a
    rectified (softplus) current to the projection of x on it. Preferred directions are assigned
    per *group* of neurons (normally the neuron's cell type and body side, from the connectome
    annotations): neurons of one type share a tuning, so their common downstream targets receive a
    coherent signal instead of a cancelling mixture. Groups get evenly spread directions (+/- along
    each channel dimension first, then random unit vectors) plus a small per-neuron jitter.
    Per-neuron gain and threshold are learnable.
    """

    def __init__(self, d: int, n: int, gain: float = 2.0, groups: np.ndarray | None = None,
                 jitter: float = 0.15, seed: int = 0, learn_tuning: bool = True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        if groups is None:
            groups = np.arange(n)
        _, gid = np.unique(np.asarray(groups), return_inverse=True)
        n_groups = int(gid.max()) + 1
        dirs = torch.zeros(n_groups, d)
        for k in range(n_groups):
            if k < 2 * d:
                dirs[k, k // 2] = 1.0 if k % 2 == 0 else -1.0
            else:
                v = torch.randn(d, generator=g)
                dirs[k] = v / v.norm()
        # shuffle which group gets which direction so the first types are not always the +x detectors
        perm = torch.randperm(n_groups, generator=g)
        dirs = dirs[perm]
        U = dirs[torch.as_tensor(gid, dtype=torch.long)] + jitter * torch.randn(n, d, generator=g)
        U = U / U.norm(dim=1, keepdim=True).clamp(min=1e-6)
        # the preferred direction of every sensory neuron is learnable (a sensor's receptive field is not in
        # the connectome), so gradient descent can route the needed signals through the existing wiring
        self.U = nn.Parameter(U, requires_grad=learn_tuning)
        self.register_buffer('group', torch.as_tensor(gid, dtype=torch.long))
        self.log_gain = nn.Parameter(math.log(gain) + 0.1 * torch.randn(n, generator=g))
        self.threshold = nn.Parameter(0.05 * torch.rand(n, generator=g))
        self.d, self.n, self.n_groups = d, n, n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        U = self.U / self.U.norm(dim=1, keepdim=True).clamp(min=1e-6)
        drive = x @ U.T * torch.exp(self.log_gain) - self.threshold
        return F.softplus(drive, beta=4.0)
