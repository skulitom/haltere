import numpy as np
import pandas as pd
import torch

from haltere.brain.model import BrainConfig, ConnectomeRNN
from haltere.brain.sparse import sparse_recurrent, transpose_order
from haltere.connectome.graph import BrainGraph


def random_graph(N=300, E=3000, seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, N, E)
    post = rng.integers(0, N, E)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    key = np.unique(post * N + pre)
    post, pre = key // N, key % N
    n_syn = rng.integers(1, 30, len(pre)).astype(np.float32)
    sign = rng.choice([1, -1, 0], N, p=[0.6, 0.3, 0.1]).astype(np.int8)
    nodes = pd.DataFrame({'bodyId': np.arange(N), 'type': 'x', 'sign': sign})
    pops = {'sense_a': np.arange(0, 40), 'sense_b': np.arange(40, 60), 'motor': np.arange(N - 20, N)}
    return BrainGraph(nodes, pre, post, n_syn, pops)


def test_sparse_recurrent_matches_reference():
    torch.manual_seed(0)
    N, E, B = 200, 2000, 5
    post = torch.randint(0, N, (E,))
    pre = torch.randint(0, N, (E,))
    key = torch.unique(post * N + pre)
    post, pre = key // N, key % N
    indices = torch.stack([post, pre])
    it, perm = transpose_order(post, pre)
    vals = torch.randn(len(post), dtype=torch.float64, requires_grad=True)
    r = torch.randn(N, B, dtype=torch.float64, requires_grad=True)
    out = sparse_recurrent(vals, r, indices, it, perm, N, chunk=300)
    ref = torch.sparse.mm(torch.sparse_coo_tensor(indices, vals, (N, N)), r)
    assert torch.allclose(out, ref)
    g = torch.randn_like(out)
    gv, gr = torch.autograd.grad((out * g).sum(), (vals, r))
    gv2, gr2 = torch.autograd.grad((ref * g).sum(), (vals, r))
    assert torch.allclose(gv, gv2) and torch.allclose(gr, gr2)


def test_brain_forward_backward_and_signs():
    g = random_graph()
    channels = {'a': 3, 'b': 2}
    cfg = BrainConfig(sensory={'a': 'sense_a', 'b': 'sense_b'}, motor=('motor',), readout_init=0.05)
    brain = ConnectomeRNN(g, channels, cfg, 'cpu')
    B = 4
    state = brain.init_state(B)
    obs = {'a': torch.randn(B, 3), 'b': torch.randn(B, 2)}
    W = brain.weight_matrix()
    loss = 0
    for _ in range(10):
        act, state, aux = brain(obs, state, W)
        assert act.shape == (B, 4)
        loss = loss + act.pow(2).mean()
    loss.backward()
    assert brain.log_edge_gain.grad is not None and brain.log_edge_gain.grad.abs().sum() > 0
    assert brain.readout.weight.grad.abs().sum() > 0
    # fixed signs are respected: an inhibitory presynaptic neuron only has negative outgoing weights
    w = brain.edge_weights().detach()
    s = brain.signs().detach()
    inhib_edges = s[brain.pre] < 0
    assert torch.all(w[inhib_edges] < 0)
    excit_edges = s[brain.pre] > 0
    assert torch.all(w[excit_edges] > 0)
    # each neuron's normalised synapse counts sum to one
    tot = torch.zeros(brain.N).index_add_(0, brain.post, brain.nnorm)
    has_in = tot > 0
    assert torch.allclose(tot[has_in], torch.ones_like(tot[has_in]), atol=1e-5)


def test_state_reset_masking():
    g = random_graph()
    brain = ConnectomeRNN(g, {'a': 3, 'b': 2}, BrainConfig(sensory={'a': 'sense_a', 'b': 'sense_b'}, motor=('motor',), readout_init=0.05), 'cpu')
    s0 = brain.init_state(3)
    s1 = {'v': s0['v'] + 1.0, 'act': s0['act'] + 0.5}
    mask = torch.tensor([True, False, True])
    s = ConnectomeRNN.where_state(mask, s0, s1)
    assert torch.allclose(s['v'][:, 0], s0['v'][:, 0]) and torch.allclose(s['v'][:, 1], s1['v'][:, 1])
    assert torch.allclose(s['act'][2], s0['act'][2]) and torch.allclose(s['act'][1], s1['act'][1])
