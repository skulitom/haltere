import torch

from haltere.train.fast_motor_tracking import fast_contract, fit_readout, sink_weights, step_gram


class Readout(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.readout = torch.nn.Linear(width, 4)
        self.other = torch.nn.Parameter(torch.ones(2))


def data(n=400, width=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    slow = torch.randn(n, width//2, generator=g).cumsum(0)/10.
    fast = torch.randn(n, width-width//2, generator=g)       # tick-to-tick noise features
    features = torch.cat((slow, fast), -1)
    labels = torch.stack((slow[:, 0], slow[:, 1]+.3*fast[:, 0], fast[:, 1]), -1)
    return features, labels, torch.diff(features, dim=0)


def test_zero_smoothing_is_the_plain_parent_centred_ridge():
    features, labels, steps = data()
    torch.manual_seed(1)
    a, b = Readout(6), Readout(6)
    b.load_state_dict(a.state_dict())
    fit_readout(a, features, labels, 1e-3)
    changed = fit_readout(b, features, labels, 1e-3, smooth=0., **step_gram(steps))
    assert torch.allclose(a.readout.weight, b.readout.weight) and torch.allclose(a.readout.bias, b.readout.bias)
    assert set(changed) == {'readout.weight', 'readout.bias'}


def test_smoothing_cuts_the_command_change_per_tick_and_only_touches_rows_0_to_2():
    features, labels, steps = data()
    torch.manual_seed(1)
    plain, smooth = Readout(6), Readout(6)
    smooth.load_state_dict(plain.state_dict())
    yaw = plain.readout.weight[3].detach().clone()
    fit_readout(plain, features, labels, 1e-3)
    fit_readout(smooth, features, labels, 1e-3, smooth=10., **step_gram(steps))
    change = lambda m: (steps@m.readout.weight[:3].T).pow(2).mean()
    assert change(smooth) < .5*change(plain)
    assert torch.equal(smooth.readout.weight[3], yaw) and torch.equal(smooth.other, torch.ones(2))


def test_step_gram_has_an_inert_bias_column():
    _, _, steps = data()
    gram = step_gram(steps, chunk=37)
    assert gram['step_count'] == len(steps)
    assert torch.allclose(gram['step_gram'][:-1, :-1], (steps.double().T@steps.double()))
    assert gram['step_gram'][-1].abs().max() == 0


def test_sink_weights_up_weight_steep_requests_with_mean_one():
    request = torch.tensor([[1., 0., -2.], [1., 0., -1.5], [3., 0., -1.], [3., 0., 0.]])
    weights = sink_weights(request, 10.)
    assert torch.allclose(weights.mean(), torch.tensor(1.))
    assert torch.allclose(weights[:2], weights[0].expand(2)) and torch.allclose(weights[0]/weights[2], torch.tensor(10.))


def test_contract_carries_the_vertical_goal_time():
    contract = fast_contract(6., .4, scaled_speed=2.4)
    assert contract['vertical_goal_seconds'] == .4 and abs(contract['goal_seconds']-.4) < 1e-12
