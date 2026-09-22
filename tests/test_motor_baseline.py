import torch
from dataclasses import replace

from haltere.brain.motor_baseline import MotorPD, inverse_rate
from haltere.sim.controller import RateControllerParams
from haltere.sim.quad import QuadParams, QuadState, quat_from_euler
from haltere.sim.rates import betaflight_rate_curve
from haltere.sim.vehicle import RatesConfig, Vehicle


def test_inverse_rate_round_trip_on_original_drone_rates():
    stick = torch.linspace(-.8, .8, 41)
    rates = betaflight_rate_curve(stick, 1.55, .73, .3)
    restored = inverse_rate(rates, 1.55, .73, .3)
    assert torch.allclose(stick, restored, atol=1e-4)


def test_motor_pd_hover_and_braking_in_dynamics():
    vehicle = Vehicle(QuadParams(gyro_noise=0), RateControllerParams(), RatesConfig(), 'cpu')
    pd = MotorPD(vehicle.sim.p, vehicle.rates)
    q = QuadState.hover(2, 'cpu', 5.)
    q.motor[:] = vehicle.sim.hover_command()
    q.vel[1, 0] = 3.
    state = vehicle.wrap(q)
    for _ in range(300):
        s = vehicle.sim.sensors(state.quad)
        action = pd.command(s, torch.zeros(2, 3))
        state = vehicle.step(state, action)
    assert state.quad.vel[0].norm() < .05
    assert state.quad.vel[1].norm() < .3
    assert not state.quad.crashed.any()
    assert state.quad.pos[:, 2].min() > 4.5


def test_pd_rotates_equivariantly_with_heading():
    vehicle = Vehicle(QuadParams(gyro_noise=0), RateControllerParams(), RatesConfig(), 'cpu')
    pd = MotorPD(vehicle.sim.p, vehicle.rates)
    q = QuadState.hover(2, 'cpu', 5.)
    q.quat = quat_from_euler(torch.zeros(2), torch.zeros(2), torch.tensor([0., 1.57]))
    result = pd.command(vehicle.sim.sensors(q), torch.tensor([[3., 0., 1.], [3., 0., 1.]]))
    assert torch.allclose(result[0], result[1], atol=2e-4)


def test_measured_throttle_fit_includes_vehicle_idle():
    from haltere.liftoff.fit_vertical import equivalent_power_curve, G
    from haltere.brain.retina import brain_to_processed
    calibration = dict(throttle_scale=.8, hover_stick_sim=-.4301975845,
                       hover_processed=.13566077, stick_sign=[-1., 1., 1.])
    fit = dict(hover_processed=.120070746, slope_mps2_per_processed=17.113039)
    curve = equivalent_power_curve(fit, calibration, idle=.04)
    quad = replace(QuadParams(), **curve)
    vehicle = Vehicle(quad, RateControllerParams(idle=.04), RatesConfig(), 'cpu')
    q = QuadState.hover(1, 'cpu', 5.)
    pd = MotorPD(quad, vehicle.rates, idle=.04)
    a = pd.command(vehicle.sim.sensors(q), torch.zeros(1, 3))
    processed = brain_to_processed(a, calibration)[0, 0]
    # The simulator uses 9.81, while the measured fit uses standard gravity.
    assert abs(float(processed)-fit['hover_processed']) < .001
    motor = vehicle.ctl.mixer((a[:, 0]+1)/2, torch.zeros(1, 3))[0, 0]
    assert abs(float(G*quad.twr*motor**quad.thrust_exp)-G) < .001
    dm_dp = (1-.04)/(2*calibration['throttle_scale'])
    slope = G*quad.twr*quad.thrust_exp*float(motor)**(quad.thrust_exp-1)*dm_dp
    assert abs(slope-fit['slope_mps2_per_processed']) < .001
