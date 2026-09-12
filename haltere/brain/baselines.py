"""Baseline policies with the same interface as ConnectomeRNN, to check that the training loop and
the task are learnable independently of the connectome (and to have something to compare against)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """Memoryless MLP on the concatenated sensory channels (plus the same output low-pass)."""

    def __init__(self, channels: dict[str, int], n_actions: int = 4, hidden: int = 128, dt: float = 0.01,
                 action_tau: float = 0.03, device='cpu'):
        super().__init__()
        self.channel_dims = dict(channels)
        self.order = list(channels)
        d = sum(channels.values())
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_actions))
        nn.init.normal_(self.net[-1].weight, std=0.05)
        nn.init.zeros_(self.net[-1].bias)
        self.readout = self.net[-1]           # so train() can set the hover throttle bias
        self.n_actions = n_actions
        self.dt = dt
        self.action_tau = action_tau
        self.N = 0
        self.E = 0
        self.log_edge_gain = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.to(device)

    @property
    def device(self):
        return self.readout.weight.device

    def init_state(self, B: int) -> dict:
        return {'act': torch.zeros(B, self.n_actions, device=self.device)}

    @staticmethod
    def where_state(mask, new, old):
        return {'act': torch.where(mask[:, None], new['act'], old['act'])}

    @staticmethod
    def detach_state(state):
        return {k: t.detach() for k, t in state.items()}

    def weight_matrix(self):
        return None

    def forward(self, obs, state, W=None):
        x = torch.cat([obs[k] for k in self.order], dim=1)
        u = torch.tanh(self.net(x))
        alpha = self.dt / max(self.action_tau, self.dt)
        act = state['act'] + alpha * (u - state['act'])
        z = torch.zeros((), device=x.device)
        return act, {'act': act}, {'rate_mean': z, 'rate_motor': z, 'u': u}

    def regularization(self, aux):
        return torch.zeros((), device=self.device)

    def config_dict(self):
        return {'brain': {'model': 'mlp'}, 'channels': self.channel_dims}
