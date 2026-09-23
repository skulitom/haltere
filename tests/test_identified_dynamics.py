import numpy as np
import pytest
import torch

from haltere.sim.identified import IdentifiedSim,processed_command
from haltere.liftoff.pilot import LiftoffMapping
from haltere.liftoff.stickcal import RadialSticks


CAL=dict(hover_processed=.12,hover_stick_sim=-.3,throttle_scale=.25,stick_sign=[-1,1,1],max_rpm=40000.)
PROFILE=dict(thrust=dict(full_input_twr=3.15,exponent=2.,response_tau_s=.012),
    axes={a:dict(coefficient_deg_s=268 if a!='yaw' else 180,response_tau_s=.004,
                 expo=.3,super_rate=.73,super_after_expo=True) for a in ['roll','pitch','yaw']},
    translation_drag_s_inv=[.1,.1,.35],rpm_slope=35000.,rpm_intercept=0.)


def test_processed_surrogate_matches_actual_pad_mapping_including_saturation():
    mapping=LiftoffMapping(stick_sign=(-1,1,1),hover_stick_sim=-.3,hover_processed_game=.12,
        throttle_scale=.25,stick_model=RadialSticks(.25,dict(roll=-1,pitch=1,yaw=1,throttle=1)))
    actions=torch.tensor([[-.3,0,0,0],[1,.9,.9,1],[-1,-1,.7,-.8]],dtype=torch.float64)
    predicted=processed_command(actions,CAL)
    for expected,action in zip(predicted,actions):
        raw=mapping.to_raw(action.numpy())
        thr,yaw=mapping.stick_model.processed('throttle','yaw',raw[0],raw[3])
        roll,pitch=mapping.stick_model.processed('roll','pitch',raw[1],raw[2])
        assert expected.tolist()==pytest.approx([thr,roll,pitch,yaw])


def test_hover_balance_and_finite_flight_cost_gradients():
    sim=IdentifiedSim(PROFILE,CAL);sim.randomize(2,0.)
    state=sim.hover(2)
    processed=2*state.drive-1
    throttle=CAL['hover_stick_sim']+(processed-CAL['hover_processed'])/CAL['throttle_scale']
    action=torch.stack((throttle,throttle*0,throttle*0,throttle*0),-1).requires_grad_()
    old_height=state.quad.pos[:,2].clone()
    for _ in range(10):state=sim.step(state,action)
    torch.testing.assert_close(state.quad.pos[:,2],old_height,atol=1e-5,rtol=0)
    loss=(state.quad.pos[:,2]-7).square().mean()+(state.quad.pos[:,0]-1).square().mean()
    loss.backward()
    assert torch.isfinite(action.grad).all()
    assert (action.grad[:,0].abs()>1e-5).all() and (action.grad[:,2].abs()>1e-5).all()
    assert not state.quad.crashed.any()


def test_response_signs_and_sensor_delay_match_measured_contract():
    sim=IdentifiedSim(PROFILE,CAL);sim.randomize(3,0.)
    state=sim.hover(3)
    action=torch.zeros(3,4);action[:,0]=-.3
    action[0,1]=.5;action[1,2]=.5;action[2,3]=.5
    result=sim.step(state,action)
    assert result.quad.omega[0,0]>0 and result.quad.omega[1,1]>0 and result.quad.omega[2,2]<0
    assert torch.all(result.sensed_omega.abs()<=result.quad.omega.abs()+1e-6)
    assert torch.isfinite(sim.sensors(result)['gravity_body']).all()
    detached=result.detach()
    assert not detached.quad.pos.requires_grad


def test_uncertainty_is_per_episode_and_repeatable():
    sim=IdentifiedSim(PROFILE,CAL)
    sim.randomize(4,.2,torch.Generator().manual_seed(7));first=sim.twr.clone()
    sim.randomize(4,.2,torch.Generator().manual_seed(7))
    torch.testing.assert_close(first,sim.twr)
    assert first.std()>0
    with pytest.raises(ValueError,match='batch'):
        sim.hover(2)
