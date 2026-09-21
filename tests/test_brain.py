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


def test_navigation_images_preserve_complete_blank_dynamics_and_receive_gradients():
    from haltere.train.bptt import ExperimentConfig
    from haltere.train.navigation_scene import add_navigation_input, KEY
    torch.manual_seed(82)
    graph = random_graph(N=100, E=900)
    graph.populations['lptc'] = graph.population('sense_a')
    graph.populations['goal'] = graph.population('sense_b')
    cfg = ExperimentConfig.from_dict({'brain': dict(sensory={'a':'sense_a','retina':'lptc'},
                                motor=['motor'],readout_init=.1)})
    parent = ConnectomeRNN(graph, {'a':3,'retina':12}, cfg.brain).eval().requires_grad_(False)
    student, _, parameters = add_navigation_input(parent, cfg, graph)
    original = {k:v.clone() for k,v in parent.state_dict().items()}
    optimizer = torch.optim.Adam(parameters, lr=.01)
    # Exercise an actual update first, then prove the full state still matches
    # over a changing nonvisual input history, including cold startup.
    for _ in range(3):
        state = student.init_state(2)
        for _ in range(12):
            action,state,_ = student({'a':torch.randn(2,3),'retina':torch.randn(2,12)},state)
        optimizer.zero_grad();action.square().mean().backward();optimizer.step()
    assert student.encoders[KEY].U.grad.abs().sum() > 0
    assert all(torch.equal(v,student.state_dict()[k]) for k,v in original.items())
    with torch.no_grad():
        sp, ss = parent.init_state(2),student.init_state(2)
        for _ in range(100):
            obs = {'a':torch.randn(2,3),'retina':torch.zeros(2,12)}
            ap,sp,_ = parent(obs,sp)
            actual,ss,_ = student(obs,ss)
            assert torch.equal(ap,actual)
            assert torch.equal(sp['v'],ss['v'])


def test_sensory_override_rejects_unknown_or_conflicting_encoders():
    import pytest
    cfg = BrainConfig(sensory={'a':'sense_a'},motor=('motor',),blank_encoders=('typo',))
    with pytest.raises(ValueError,match='unknown encoders'):
        ConnectomeRNN(random_graph(),{'a':3},cfg)
    cfg.blank_encoders = cfg.opponent_encoders = ('a__sense_a',)
    with pytest.raises(ValueError,match='both blank and opponent'):
        ConnectomeRNN(random_graph(),{'a':3},cfg)
