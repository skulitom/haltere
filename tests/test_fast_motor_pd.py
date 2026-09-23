"""FastMotorPD: the declared velocity-command motor baseline beside the teacher PD.

Closed-loop checks run in `IdentifiedSim`, the measured original-drone surrogate,
with the 30 ms command delay that `fast_rehearsal` uses. The measured profile
lives under the gitignored runs/ directory; when it is absent an inline copy with
the same keys (values rounded from the 2026-09-23 low-speed fit) stands in, so a
clean checkout still runs these tests. None of this is flight evidence.
"""
import inspect
import json
from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haltere.brain.motor_baseline import (FastMotorPD, FastPDConfig, MotorPD, MotorPDConfig,
                                          measured_inverse_rate)
from haltere.sim.identified import IdentifiedSim, processed_command
from haltere.sim.quad import quat_from_euler, quat_mul, quat_to_mat
from haltere.sim.vehicle import RatesConfig

ROOT = Path(__file__).resolve().parents[1]
MEASURED_PROFILE = ROOT / 'runs' / 'measured-dynamics-low-speed-20260923' / 'profile.json'

# Same keys as the measured profile; values rounded from it.
INLINE_PROFILE = dict(
    thrust=dict(full_input_twr=3.1378, exponent=1.9729, response_tau_s=.01118),
    axes=dict(roll=dict(coefficient_deg_s=263.40, response_tau_s=.00254, expo=.3, super_rate=.73,
                        super_after_expo=True),
              pitch=dict(coefficient_deg_s=263.76, response_tau_s=.00254, expo=.3, super_rate=.73,
                         super_after_expo=True),
              yaw=dict(coefficient_deg_s=180.17, response_tau_s=.0001, expo=.3, super_rate=.73,
                       super_after_expo=True)),
    translation_drag_s_inv=[.02746, 0., .34904],
    rpm_slope=35776.45, rpm_intercept=0.)

# Deployed brain/pad calibration (brain_to_processed); max_rpm only scales the sim's RPM readout.
CAL = dict(hover_processed=.1357, throttle_scale=.8, hover_stick_sim=-.4302, stick_sign=[-1., 1., 1.],
           max_rpm=40000.)


def load_profile():
    """The measured profile when present, else the inline stand-in."""
    if MEASURED_PROFILE.exists():
        return json.loads(MEASURED_PROFILE.read_text(encoding='utf-8'))
    return INLINE_PROFILE


def _profiles():
    params = [pytest.param(INLINE_PROFILE, id='inline')]
    if MEASURED_PROFILE.exists():
        params.insert(0, pytest.param(load_profile(), id='measured'))
    return params


@pytest.fixture(autouse=True, scope='module')
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fly(profile, request, steps, *, initial_velocity=None, height=20., delay=3, config=None):
    """Hold a constant world velocity request on the measured surrogate; return final state and trace."""
    request = torch.as_tensor(request, dtype=torch.float32).reshape(-1, 3)
    batch = len(request)
    sim = IdentifiedSim(profile, CAL)
    sim.randomize(batch, 0.)
    state = sim.hover(batch, height)
    if initial_velocity is not None:
        state.quad.vel[:] = torch.as_tensor(initial_velocity, dtype=torch.float32)
    pd = FastMotorPD(profile, CAL, config)
    hover = pd.command(sim.sensors(state), torch.zeros(batch, 3))
    pd.reset()
    queue = deque(hover.clone() for _ in range(delay))
    actions, tilt, velocity, height_trace = [], [], [], []
    for _ in range(steps):
        action = pd.command(sim.sensors(state), request)
        actions.append(action)
        queue.append(action)
        state = sim.step(state, queue.popleft())
        rotation = quat_to_mat(state.quad.quat)
        tilt.append(torch.rad2deg(torch.arccos(rotation[:, 2, 2].clamp(-1., 1.))))
        velocity.append(state.quad.vel.clone())
        height_trace.append(state.quad.pos[:, 2].clone())
    return state, dict(actions=torch.stack(actions), tilt=torch.stack(tilt),
                       velocity=torch.stack(velocity), height=torch.stack(height_trace))


@pytest.mark.parametrize('profile', _profiles())
def test_hover_hold_uses_the_measured_hover_drive(profile):
    state, trace = fly(profile, [[0., 0., 0.]], 300)
    assert not state.quad.crashed.any()
    assert float(state.quad.vel.norm()) < .01
    assert float((trace['height'] - 20.).abs().max()) < .02
    assert float(trace['actions'][:, :, 1:].abs().max()) < 1e-3
    # Measured thrust curve: requested processed throttle equals the plant's hover drive.
    processed = processed_command(trace['actions'][-1], CAL)[:, 0]
    torch.testing.assert_close(processed, 2 * state.drive - 1, atol=1e-4, rtol=0)


@pytest.mark.parametrize('profile', _profiles())
def test_velocity_steps_are_tracked_without_cross_coupling(profile):
    requests = torch.tensor([[4., 0., 0.], [0., -4., 0.], [3., 3., 0.], [12., 0., 0.],
                             [0., 0., 2.], [0., 0., -2.]])
    state, trace = fly(profile, requests, 400)
    assert not state.quad.crashed.any()
    final = state.quad.vel
    horizontal = requests[:, 2] == 0
    # Horizontal: the measured horizontal drag is small, so steady error is small.
    assert float((final[horizontal, :2] - requests[horizontal, :2]).norm(dim=-1).max()) < .2
    assert float(final[horizontal, 2].abs().max()) < .1
    assert float((trace['height'][:, horizontal] - 20.).abs().max()) < 1.
    # Vertical: no integral action against the measured .35 1/s vertical drag,
    # so allow the resulting steady error but require the right direction and most of the step.
    vertical = ~horizontal
    ratio = final[vertical, 2] / requests[vertical, 2]
    assert bool(((ratio > .85) & (ratio < 1.05)).all())
    assert float(final[vertical, :2].norm(dim=-1).max()) < .05
    # Early response goes the right way and stays in the declared tilt cone.
    early = trace['velocity'][50]
    assert bool(((early * requests).sum(-1) > .3 * requests.square().sum(-1)).all())
    assert float(trace['tilt'].max()) <= FastPDConfig().max_tilt_deg + 1.5


@pytest.mark.parametrize('profile', _profiles())
def test_brakes_to_rest_from_fast_flight(profile):
    initial = torch.tensor([[8., 3., 0.], [-10., 0., 0.], [0., 6., -2.]])
    state, trace = fly(profile, torch.zeros(3, 3), 300, initial_velocity=initial)
    assert not state.quad.crashed.any()
    assert float(state.quad.vel.norm(dim=-1).max()) < .05
    speed = trace['velocity'].norm(dim=-1)
    assert bool((speed[100] < .5 * initial.norm(dim=-1)).all())
    assert float((trace['height'] - 20.).abs().max()) < 1.
    assert float(trace['tilt'].max()) <= FastPDConfig().max_tilt_deg + 1.5


def test_feedforward_reduces_lag_on_an_acceleration_limited_request():
    """The pilot's slew-limited request carries its own acceleration; using it removes the ramp lag."""
    profile = load_profile()
    ramp = 8.  # m/s^2, inside the declared horizontal acceleration envelope
    errors = []
    for use_feedforward in (False, True):
        sim = IdentifiedSim(profile, CAL)
        state = sim.hover(1, 20.)
        pd = FastMotorPD(profile, CAL)
        queue = deque(pd.command(sim.sensors(state), torch.zeros(1, 3)).clone() for _ in range(3))
        pd.reset()
        for k in range(100):
            request = torch.tensor([[ramp * k * .01, 0., 0.]])
            feedforward = torch.tensor([[ramp, 0., 0.]]) if use_feedforward else None
            queue.append(pd.command(sim.sensors(state), request, feedforward))
            state = sim.step(state, queue.popleft())
        errors.append(float(request[0, 0] - state.quad.vel[0, 0]))
    without, with_feedforward = errors
    assert without > 2.  # ~ramp / velocity_gain behind the request
    assert 0 <= with_feedforward < .4 * without


def test_narrower_declared_tilt_cone_is_respected_in_closed_loop():
    config = FastPDConfig(max_tilt_deg=30.)
    state, trace = fly(load_profile(), [[20., 0., -3.], [-15., 10., 0.]], 250, config=config)
    assert not state.quad.crashed.any()
    assert float(trace['tilt'].max()) <= 30. + 1.5
    assert float(trace['tilt'][-50:].max()) > 20.  # the limit was actually exercised


def _sensors(quat, vel_world, gyro):
    return dict(quat=quat, vel_world=vel_world, gyro=gyro)


def test_command_is_equivariant_under_heading_rotation():
    generator = torch.Generator().manual_seed(3)
    batch = 16
    roll = (torch.rand(batch, generator=generator) - .5) * 1.2
    pitch = (torch.rand(batch, generator=generator) - .5) * 1.2
    yaw = (torch.rand(batch, generator=generator) - .5) * 6.
    quat = quat_from_euler(roll, pitch, yaw)
    velocity = (torch.rand(batch, 3, generator=generator) - .5) * 16
    gyro = (torch.rand(batch, 3, generator=generator) - .5) * 4
    request = (torch.rand(batch, 3, generator=generator) - .5) * 24
    feedforward = (torch.rand(batch, 3, generator=generator) - .5) * 10
    profile = load_profile()
    reference = FastMotorPD(profile, CAL).command(_sensors(quat, velocity, gyro), request, feedforward)
    for heading in (.7, -2., 3.):
        c, s = np.cos(heading), np.sin(heading)
        rz = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=torch.float32)
        qz = torch.tensor([[np.cos(heading / 2), 0., 0., np.sin(heading / 2)]], dtype=torch.float32)
        rotated = quat_mul(qz.expand(batch, 4), quat)
        torch.testing.assert_close(quat_to_mat(rotated), rz @ quat_to_mat(quat), atol=1e-5, rtol=0)
        # Rotate the world only: body rates are unchanged, world vectors rotate.
        action = FastMotorPD(profile, CAL).command(_sensors(rotated, velocity @ rz.T, gyro),
                                                   request @ rz.T, feedforward @ rz.T)
        torch.testing.assert_close(action, reference, atol=2e-4, rtol=0)


def test_outputs_stay_finite_and_inside_brain_action_bounds():
    generator = torch.Generator().manual_seed(11)
    batch = 512
    quat = torch.randn(batch, 4, generator=generator)
    quat = quat / quat.norm(dim=-1, keepdim=True)  # includes inverted and knife-edge attitudes
    sensors = _sensors(quat, (torch.rand(batch, 3, generator=generator) - .5) * 80,
                       (torch.rand(batch, 3, generator=generator) - .5) * 40)
    pd = FastMotorPD(load_profile(), CAL)
    for scale in (1., 60., 1e6):
        request = (torch.rand(batch, 3, generator=generator) - .5) * 2 * scale
        feedforward = (torch.rand(batch, 3, generator=generator) - .5) * scale
        action = pd.command(sensors, request, feedforward)  # second pass uses the stick filter
        assert action.shape == (batch, 4)
        assert torch.isfinite(action).all()
        assert float(action.abs().max()) <= 1.
        assert bool((action[:, 3] == 0).all()), 'yaw belongs to the pilot, not the motor PD'
        assert torch.isfinite(processed_command(action, CAL)).all()
    double = FastMotorPD(load_profile(), CAL).command(
        {k: v.double() for k, v in sensors.items()}, torch.zeros(batch, 3, dtype=torch.float64))
    assert double.dtype == torch.float64 and torch.isfinite(double).all()


def test_level_request_to_climb_or_sink_saturates_throttle_monotonically():
    speeds = (-50., -2., -1., 0., 1., 50.)
    quat = torch.tensor([[1., 0., 0., 0.]]).expand(len(speeds), 4)
    zeros = torch.zeros(len(speeds), 3)
    request = torch.tensor([[0., 0., v] for v in speeds])
    action = FastMotorPD(load_profile(), CAL).command(_sensors(quat, zeros, zeros), request)
    throttle = action[:, 0]
    # gain 3 * -2 m/s already reaches max_sink_acceleration (6 m/s^2): saturated, then strictly increasing.
    assert float(throttle[0]) == pytest.approx(float(throttle[1]), abs=1e-6)
    assert bool((throttle[2:] > throttle[1:-1]).all())
    assert -1. < float(throttle[0]) and float(throttle[-1]) < 1.  # bounded by declared sink/climb accelerations


def test_measured_inverse_rate_round_trips_the_surrogate_rate_curve():
    axis = load_profile()['axes']['roll']
    c, super_rate, expo = axis['coefficient_deg_s'], axis['super_rate'], axis['expo']
    stick = torch.linspace(-.95, .95, 77, dtype=torch.float64)
    shaped = stick * (1 - expo + expo * stick.abs().pow(3))
    rate = c * shaped / (1 - super_rate * shaped.abs())  # IdentifiedSim's post-expo curve
    restored = measured_inverse_rate(rate, c, super_rate, expo)
    torch.testing.assert_close(restored, stick, atol=2e-5, rtol=0)
    huge = measured_inverse_rate(torch.tensor([-1e5, 0., 1e5], dtype=torch.float64), c, super_rate, expo)
    assert float(huge[0]) < -.99 and float(huge[1]) == 0 and float(huge[2]) > .99
    assert float(huge.abs().max()) < 1.


@pytest.mark.parametrize('dt', [.01, .005])
def test_stick_low_pass_filters_roll_and_pitch_only(dt):
    profile = load_profile()
    level = torch.tensor([[1., 0., 0., 0.]])
    sensors = _sensors(level, torch.zeros(1, 3), torch.zeros(1, 3))
    first_request, second_request = torch.tensor([[0., 0., 1.]]), torch.tensor([[6., -4., 0.]])
    pd = FastMotorPD(profile, CAL)
    first = pd.command(sensors, first_request, dt=dt)
    second = pd.command(sensors, second_request, dt=dt)
    raw = FastMotorPD(profile, CAL).command(sensors, second_request, dt=dt)
    alpha = 1 - np.exp(-dt / FastPDConfig().stick_time_constant)
    torch.testing.assert_close(second[:, 1:3], first[:, 1:3] + alpha * (raw[:, 1:3] - first[:, 1:3]),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(second[:, 0], raw[:, 0])  # throttle is not delayed
    assert float((second[:, 1:3] - raw[:, 1:3]).abs().max()) > 1e-3
    pd.reset()
    torch.testing.assert_close(pd.command(sensors, second_request, dt=dt), raw)
    unfiltered = FastMotorPD(profile, CAL, FastPDConfig(stick_time_constant=0.))
    unfiltered.command(sensors, first_request, dt=dt)
    torch.testing.assert_close(unfiltered.command(sensors, second_request, dt=dt), raw)


@pytest.mark.parametrize('overrides', [dict(max_tilt_deg=80.), dict(max_tilt_deg=0.), dict(velocity_gain=0.),
                                       dict(max_rate=0.), dict(attitude_gain=-1.),
                                       dict(max_acceleration=float('nan'))])
def test_invalid_fast_pd_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        FastPDConfig(**overrides)


@pytest.mark.parametrize('field', ['max_acceleration', 'velocity_gain'])
def test_infinite_fast_pd_config_is_rejected_or_stays_finite(field):
    try:
        config = FastPDConfig(**{field: float('inf')})
    except ValueError:
        return
    level = torch.tensor([[1., 0., 0., 0.]])
    action = FastMotorPD(load_profile(), CAL, config).command(
        _sensors(level, torch.zeros(1, 3), torch.zeros(1, 3)), torch.tensor([[3., 0., 1.]]))
    assert torch.isfinite(action).all(), f'{field}=inf was accepted and produced {action.tolist()}'


def test_profile_must_carry_measured_post_expo_curve_and_thrust():
    profile = load_profile()
    pre_expo = json.loads(json.dumps(profile))
    pre_expo['axes']['pitch']['super_after_expo'] = False
    with pytest.raises(ValueError, match='post-expo'):
        FastMotorPD(pre_expo, CAL)
    weak = json.loads(json.dumps(profile))
    weak['thrust']['full_input_twr'] = .9
    with pytest.raises(ValueError, match='thrust-to-weight'):
        FastMotorPD(weak, CAL)


def test_fast_pd_metadata_is_declared_and_json_serializable():
    pd = FastMotorPD(load_profile(), CAL)
    meta = json.loads(json.dumps(pd.metadata()))
    assert meta['kind'] == 'fast_pd'
    assert meta['course_geometry'] is False and meta['brain_controls_motors'] is False
    assert meta['parameters'] == asdict(FastPDConfig())
    assert meta['thrust_twr'] == pytest.approx(load_profile()['thrust']['full_input_twr'])


def test_teacher_motor_pd_contract_is_unchanged():
    """MotorPD is the brain's training-teacher contract; the fast profile must not alter it."""
    assert asdict(MotorPDConfig()) == dict(position_gain=.8, velocity_gain=1.6, vertical_position_gain=.8,
                                           vertical_velocity_gain=2.5, attitude_gain=4., angular_damping=.15,
                                           max_acceleration=5., max_vertical_speed=1.2, max_rate=2.)
    assert list(inspect.signature(MotorPD.__init__).parameters) == ['self', 'quad', 'rates', 'idle', 'config']
    assert inspect.signature(MotorPD.__init__).parameters['idle'].default == .04
    assert list(inspect.signature(MotorPD.command).parameters) == ['self', 'sensors', 'relative_body', 'speed']
    assert inspect.signature(MotorPD.command).parameters['speed'].default == 2.
    assert not issubclass(FastMotorPD, MotorPD) and not issubclass(MotorPD, FastMotorPD)
    quad = SimpleNamespace(twr=2.9, thrust_exp=1.8)
    rates = RatesConfig(rc_rate=(1.55, 1.55, 1.), super_rate=(.73, .73, .73), expo=(.3, .3, .3))
    pd = MotorPD(quad, rates, .04)
    assert pd.metadata()['kind'] == 'pd'
    quat = quat_from_euler(torch.tensor([.1, -.2, 0.]), torch.tensor([.05, .1, -.3]), torch.tensor([0., 1., -2.]))
    sensors = _sensors(quat, torch.tensor([[1., -.5, .2], [3., 1., -.4], [-2., 0., 0.]]),
                       torch.tensor([[.2, -.1, 0.], [0., .3, .1], [-.5, .5, 0.]]))
    target = torch.tensor([[3., 0., 1.], [-1., 2., -.5], [0., 0., 0.]])
    # Golden values from the committed (HEAD) MotorPD; sticks are 14-step bisection outputs.
    golden = {2.: [[0.14964, -0.157532, 0.096008, 0.], [0.048008, -0.141785, -0.358215, 0.],
                   [0.099423, -0.230164, 0.132385, 0.]],
              2.5: [[0.14964, -0.154724, 0.149353, 0.], [0.048008, -0.141785, -0.358215, 0.],
                    [0.099423, -0.230164, 0.132385, 0.]]}
    for speed, expected in golden.items():
        torch.testing.assert_close(pd.command(sensors, target, speed=speed), torch.tensor(expected),
                                   atol=3e-4, rtol=0)
    assert not hasattr(pd, 'previous'), 'the teacher PD stays stateless'
