"""The connectome-constrained recurrent network.

Every neuron of the graph is a rate unit. Recurrent weights are fixed in *structure* and *sign* by
the connectome (synapse counts and neurotransmitter predictions) and trainable in *magnitude*
(a per-edge log-gain, initialised so that the effective weight is proportional to the synapse
count). This follows the "connectome-constrained network" recipe of Lappalainen et al. 2024.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn

from ..connectome.graph import BrainGraph
from .encoders import PopulationEncoder
from .sparse import sparse_recurrent, transpose_order


@dataclass
class BrainConfig:
    model: str = 'connectome'        # 'connectome' or 'mlp' (baseline without the connectome)
    dt: float = 0.01                 # s, control/neural time step
    tau_init: float = 0.03           # s, membrane time constant (learnable per neuron)
    tau_min: float = 0.012
    gain_init: float = 3.0           # recurrent input gain per neuron (learnable)
    bias_init: float = -1.0          # resting drive (learnable per neuron); rate = sigmoid(v)
    encoder_gain: float = 6.0        # sensory current per unit input (inputs are roughly in [-1, 1])
    action_tau: float = 0.03         # s, low-pass on the motor output (muscle-like)
    readout_init: float = 0.0        # 0 = start from "hover, do nothing" and let the gradient grow the readout
    rate_max: float = 1.0            # rate = rate_max * sigmoid(v); >1 raises the slope at the operating point
    readout_norm: str = 'batch'      # 'batch': whiten every motor neuron by slowly tracked running statistics
    norm_momentum: float = 1e-3      # EMA rate of the whitening statistics after warm-up
    norm_warmup_steps: int = 200
                                     # (amplifies the state-dependent part of its rate, removes the baseline pattern);
                                     # 'layer': normalise across the motor population per sample; 'none'
    mlp_hidden: int = 128
    learn_edges: bool = True
    learn_unknown_signs: bool = True
    learn_tuning: bool = True        # sensory neurons' preferred directions are learnable
    edge_prior: float = 1e-3         # L2 pull of the per-edge log-gain towards 0 (the connectome prior)
    rate_penalty: float = 1e-3       # pull towards sparse activity
    # channel -> population; every channel of the task must map to a population of the graph
    sensory: dict = field(default_factory=lambda: {
        'haltere': 'haltere', 'wing_cs': 'wing_cs', 'lptc': 'lptc', 'ocelli': 'ocelli', 'jo': 'jo',
        'compass': 'compass', 'goal': 'goal'})
    motor: tuple[str, ...] = ('wing_mn',)
    n_actions: int = 4
    mask_motor_feedback: bool = False  # prevents action imitation from copying action-correlated RPM

    @staticmethod
    def from_dict(d: dict) -> "BrainConfig":
        cfg = BrainConfig()
        for k, v in d.items():
            if not hasattr(cfg, k):
                raise KeyError(f'unknown brain config key: {k}')
            if isinstance(getattr(cfg, k), tuple):
                v = tuple(v)
            setattr(cfg, k, v)
        return cfg


def migrate_state_dict(sd: dict, brain: "ConnectomeRNN") -> dict:
    """Rename keys of checkpoints written before encoders were keyed by channel and population."""
    single = {}
    for key, ch in getattr(brain, 'encoder_channel', {}).items():
        single.setdefault(ch, []).append(key)
    out = {}
    for k, v in sd.items():
        nk = k
        if k.startswith('encoders.') or k.startswith('idx_'):
            parts = k.split('.') if k.startswith('encoders.') else ['idx', k[len('idx_'):]]
            name = parts[1]
            if '__' not in name and name in single and len(single[name]) == 1:
                new_name = single[name][0]
                nk = k.replace(f'encoders.{name}.', f'encoders.{new_name}.', 1) if k.startswith('encoders.') \
                    else f'idx_{new_name}'
        out[nk] = v
    return out


def tuning_groups(graph: BrainGraph, idx: np.ndarray) -> np.ndarray:
    """Group label per neuron for sensory tuning: cell type + body side (neurons of one type share a tuning)."""
    nodes = graph.nodes.iloc[idx]
    t = nodes['type'].fillna('untyped').astype(str) if 'type' in nodes else np.array(['untyped'] * len(idx))
    side = nodes['somaSide'].fillna('?').astype(str) if 'somaSide' in nodes else np.array(['?'] * len(idx))
    return (np.asarray(t, dtype=object) + '_' + np.asarray(side, dtype=object)).astype(str)


class ConnectomeRNN(nn.Module):
    def __init__(self, graph: BrainGraph, channels: dict[str, int], cfg: BrainConfig, device='cpu'):
        super().__init__()
        self.cfg = cfg
        self.n_actions = cfg.n_actions
        self.N = graph.N
        self.E = graph.E
        self.channel_dims = dict(channels)

        # --- edges, sorted by (post, pre) so the COO tensor is coalesced without a sort per step
        order = np.lexsort((graph.pre, graph.post))
        pre = torch.as_tensor(graph.pre[order], dtype=torch.long)
        post = torch.as_tensor(graph.post[order], dtype=torch.long)
        n_syn = torch.as_tensor(graph.n_syn[order], dtype=torch.float32)
        in_total = torch.zeros(self.N).index_add_(0, post, n_syn)
        nnorm = n_syn / in_total[post].clamp(min=1.0)           # each neuron's inputs sum to 1
        self.register_buffer('pre', pre)
        self.register_buffer('post', post)
        self.register_buffer('n_syn', n_syn)
        self.register_buffer('nnorm', nnorm)
        self.register_buffer('edge_index', torch.stack([post, pre]))
        indices_t, perm_t = transpose_order(post, pre)
        self.register_buffer('edge_index_t', indices_t)
        self.register_buffer('perm_t', perm_t)

        sign = torch.as_tensor(graph.sign.astype(np.float32))
        known = sign != 0
        self.register_buffer('sign_fixed', sign)
        self.register_buffer('sign_known', known)
        n_unknown = int((~known).sum())
        unknown_pos = torch.zeros(self.N, dtype=torch.long)
        unknown_pos[~known] = torch.arange(n_unknown)
        self.register_buffer('unknown_pos', unknown_pos)
        self.sign_free = nn.Parameter(torch.full((max(n_unknown, 1),), 1.0),
                                      requires_grad=cfg.learn_unknown_signs)

        # --- neuron parameters
        self.log_edge_gain = nn.Parameter(torch.zeros(self.E), requires_grad=cfg.learn_edges)
        self.log_gain = nn.Parameter(torch.full((self.N,), math.log(cfg.gain_init)))
        self.bias = nn.Parameter(torch.full((self.N,), cfg.bias_init))
        self.log_tau = nn.Parameter(torch.full((self.N,), math.log(cfg.tau_init)))

        # --- sensory encoders (a channel may be written into several populations)
        self.encoders = nn.ModuleDict()
        self.encoder_channel = {}
        for ch, dim in channels.items():
            pops = cfg.sensory.get(ch)
            if pops is None:
                raise KeyError(f'task channel {ch!r} has no population in BrainConfig.sensory')
            for pop in ([pops] if isinstance(pops, str) else list(pops)):
                idx = torch.as_tensor(graph.population(pop), dtype=torch.long)
                if len(idx) == 0:
                    raise ValueError(f'population {pop!r} for channel {ch!r} is empty')
                key = f'{ch}__{pop}'
                self.register_buffer(f'idx_{key}', idx)
                groups = tuning_groups(graph, idx.numpy())
                self.encoders[key] = PopulationEncoder(dim, len(idx), gain=cfg.encoder_gain, groups=groups,
                                                       seed=sum(map(ord, key)), learn_tuning=cfg.learn_tuning)
                self.encoder_channel[key] = ch

        # --- motor readout
        motor_idx = np.concatenate([graph.population(p) for p in cfg.motor])
        self.register_buffer('motor_idx', torch.as_tensor(motor_idx, dtype=torch.long))
        self.readout = nn.Linear(len(motor_idx), cfg.n_actions)
        nn.init.normal_(self.readout.weight, std=cfg.readout_init)
        nn.init.zeros_(self.readout.bias)
        self.motor_norm = nn.BatchNorm1d(len(motor_idx), affine=False, momentum=0.02) if cfg.readout_norm == 'batch' else None
        self.to(device)

    # ------------------------------------------------------------------ parameters
    @property
    def device(self):
        return self.bias.device

    def signs(self) -> torch.Tensor:
        free = torch.tanh(self.sign_free)[self.unknown_pos]
        return torch.where(self.sign_known, self.sign_fixed, free)

    def edge_weights(self) -> torch.Tensor:
        """Effective weight of every edge (pre -> post)."""
        s = self.signs()
        return s[self.pre] * torch.exp(self.log_edge_gain) * self.nnorm * torch.exp(self.log_gain)[self.post]

    def weight_matrix(self) -> torch.Tensor:
        """The effective edge weights, in the (post, pre)-sorted edge order used by ``forward``."""
        return self.edge_weights()

    def dense_sparse_matrix(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(self.edge_index, self.edge_weights(), (self.N, self.N), is_coalesced=True)

    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(min=self.cfg.tau_min)

    def init_state(self, B: int) -> dict[str, torch.Tensor]:
        v = self.bias.detach()[:, None].expand(self.N, B).clone()
        return {'v': v, 'act': torch.zeros(B, self.cfg.n_actions, device=self.device)}

    @staticmethod
    def where_state(mask: torch.Tensor, new: dict, old: dict) -> dict:
        return {'v': torch.where(mask[None, :], new['v'], old['v']),
                'act': torch.where(mask[:, None], new['act'], old['act'])}

    @staticmethod
    def detach_state(state: dict) -> dict:
        return {k: t.detach() for k, t in state.items()}

    # ------------------------------------------------------------------ dynamics
    def forward(self, obs: dict[str, torch.Tensor], state: dict[str, torch.Tensor],
                W: torch.Tensor | None = None) -> tuple[torch.Tensor, dict, dict]:
        v = state['v']                                     # [N,B]
        B = v.shape[1]
        r = self.cfg.rate_max * torch.sigmoid(v)
        if W is None:
            W = self.weight_matrix()
        rec = sparse_recurrent(W, r, self.edge_index, self.edge_index_t, self.perm_t, self.N)  # [N,B]
        I = torch.zeros_like(v)
        for key, enc in self.encoders.items():
            idx = getattr(self, f'idx_{key}')
            ch = self.encoder_channel[key]
            value = obs[ch]
            if self.cfg.mask_motor_feedback and ch == 'wing_cs':
                # Mask inside the exported brain so replay, simulation and live
                # control cannot accidentally use different sensory contracts.
                value = torch.cat((value[..., :2], torch.zeros_like(value[..., 2:3])), -1)
            I = I.index_add(0, idx, enc(value).T)
        k = (self.cfg.dt / self.tau())[:, None]
        v = v + k * (-v + rec + I + self.bias[:, None])
        r_new = self.cfg.rate_max * torch.sigmoid(v)
        m = r_new[self.motor_idx].T                                 # [B, M] motor neuron rates
        if self.motor_norm is not None:
            bn = self.motor_norm
            if self.training and m.shape[0] > 1:
                # track each motor neuron's statistics slowly (EMA over steps and environments); never use the
                # current batch's statistics, which would subtract the very signal the controller needs.
                # Fast warm-up for the first steps, then a long time constant so the readout sees fixed axes.
                with torch.no_grad():
                    md = m.detach()
                    n = bn.num_batches_tracked
                    mom = 0.05 if int(n) < self.cfg.norm_warmup_steps else self.cfg.norm_momentum
                    bn.running_mean.mul_(1 - mom).add_(mom * md.mean(0))
                    bn.running_var.mul_(1 - mom).add_(mom * md.var(0, unbiased=False))
                    n += 1
            m = (m - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps)
        elif self.cfg.readout_norm == 'layer':
            m = torch.nn.functional.layer_norm(m, (m.shape[1],))
        u = torch.tanh(self.readout(m))                             # [B, n_actions]
        alpha = self.cfg.dt / max(self.cfg.action_tau, self.cfg.dt)
        act = state['act'] + alpha * (u - state['act'])
        aux = {'rate_mean': r_new.mean(), 'rate_motor': r_new[self.motor_idx].mean(), 'u': u}
        return act, {'v': v, 'act': act}, aux

    def regularization(self, aux: dict) -> torch.Tensor:
        reg = self.cfg.edge_prior * self.log_edge_gain.pow(2).mean()
        reg = reg + self.cfg.rate_penalty * aux['rate_mean']
        return reg

    def config_dict(self) -> dict:
        return {'brain': asdict(self.cfg), 'channels': self.channel_dims}
