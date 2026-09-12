import math

import torch

from haltere.sim.controller import RateController, RateControllerParams, RateControllerState
from haltere.sim.quad import QuadParams, QuadSim, QuadState, quat_from_euler, quat_to_mat, rotate, rotate_inv
from haltere.sim.rates import betaflight_rate_curve, max_rate_deg_s
from haltere.sim.tasks import HoverTask, HoverTaskConfig
from haltere.sim.vehicle import RatesConfig, Vehicle

DEV = 'cpu'


def test_betaflight_rates():
    assert abs(max_rate_deg_s(1.0, 0.7, 0.0) - 200.0 / 0.3) < 1e-3
    s = torch.linspace(-1, 1, 21)
    r = betaflight_rate_curve(s, 1.0, 0.7, 0.0)
    assert torch.all(r[1:] > r[:-1])            # monotonic
    assert abs(float(r[10])) < 1e-6              # centred
    assert torch.allclose(r, -r.flip(0))         # odd


def test_quaternion_helpers():
    q = quat_from_euler(torch.tensor([0.3]), torch.tensor([-0.2]), torch.tensor([1.0]))
    v = torch.tensor([[1.0, 2.0, 3.0]])
    back = rotate_inv(q, rotate(q, v))
    assert torch.allclose(back, v, atol=1e-5)
    R = quat_to_mat(q)[0]
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)


def test_hover_is_stable_and_differentiable():
    sim = QuadSim(QuadParams(), DEV, dt=0.01, substeps=4)
    ctl = RateController(RateControllerParams(), DEV)
    B = 8
    st = QuadState.hover(B, DEV, height=2.0)
    cs = RateControllerState.zeros(B, DEV)
    thr = torch.full((B,), sim.hover_command(), requires_grad=True)
    for _ in range(100):
        for _ in range(sim.substeps):
            axis, cs = ctl.pid(cs, torch.zeros(B, 3), torch.rad2deg(st.omega), sim.sub_dt)
            st = sim._substep(st, ctl.mixer(thr, axis), sim.sub_dt)
    assert not bool(st.crashed.any())
    assert abs(float(st.pos[:, 2].mean()) - 2.0) < 0.3
    st.pos[:, 2].sum().backward()
    assert float(thr.grad.abs().sum()) > 0


def test_mixer_bounds():
    ctl = RateController(RateControllerParams(idle=0.04), DEV)
    thr = torch.tensor([0.0, 0.5, 1.0, 0.95])
    axis = torch.tensor([[0.3, -0.2, 0.1], [0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [-0.5, 0.2, 0.1]])
    m = ctl.mixer(thr, axis)
    assert torch.all(m >= 0.04 - 1e-6) and torch.all(m <= 1.0 + 1e-6)


def test_vehicle_roll_stick_rolls_right():
    veh = Vehicle(QuadParams(), RateControllerParams(), RatesConfig(), DEV)
    vs = veh.wrap(QuadState.hover(1, DEV, height=5.0))
    hover_stick = 2 * veh.sim.hover_command() - 1
    for _ in range(30):
        vs = veh.step(vs, torch.tensor([[hover_stick, 0.5, 0.0, 0.0]]))
    assert float(vs.quad.omega[0, 0]) > 1.0          # positive roll rate about body x
    vs = veh.wrap(QuadState.hover(1, DEV, height=5.0))
    for _ in range(30):
        vs = veh.step(vs, torch.tensor([[hover_stick, 0.0, 0.5, 0.0]]))
    assert float(vs.quad.omega[0, 1]) > 1.0          # pitch forward -> positive rate about body y
    vs = veh.wrap(QuadState.hover(1, DEV, height=5.0))
    for _ in range(30):
        vs = veh.step(vs, torch.tensor([[hover_stick, 0.0, 0.0, 0.5]]))
    assert float(vs.quad.omega[0, 2]) < -0.5         # yaw right -> negative rate about body z


def test_task_observation_and_cost():
    sim = QuadSim(QuadParams(), DEV)
    task = HoverTask(HoverTaskConfig(), sim, 16, DEV)
    st = task.reset_all(0.5)
    obs = task.observe(st)
    for ch, d in task.channels.items():
        assert obs[ch].shape == (16, d), ch
        assert torch.isfinite(obs[ch]).all()
    c = task.cost(st, torch.zeros(16, 4))
    assert c.shape == (16,) and torch.isfinite(c).all()
    done = task.tick(st)
    assert done.shape == (16,)
