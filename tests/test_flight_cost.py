import copy

import torch

from haltere.train.flight_cost import FlightCostRollout,trainable_motor_parameters,evaluation_rank
from tests.test_human_brain import small_brain
from tests.test_identified_dynamics import CAL,PROFILE
from tests.test_visual_assistance import SENSOR


def test_flight_cost_updates_recurrent_magnitudes_without_teacher_or_wiring_changes():
    torch.manual_seed(19)
    brain,cfg,_=small_brain();cfg.brain.mask_motor_feedback=True
    original=copy.deepcopy(brain.state_dict())
    selected=trainable_motor_parameters(brain)
    assert 'log_edge_gain' in selected and 'sign_free' not in selected
    assert all(not n.startswith('encoders.') for n in selected)
    rollout=FlightCostRollout(brain,cfg,dict(calibration=CAL,gate_sensor=SENSOR),PROFILE,
                              batch=2,reference_speed=3.,episode_steps=64)
    opt=torch.optim.Adam(selected.values(),lr=.001)
    for _ in range(2):
        opt.zero_grad();loss,metrics=rollout.loss(16)
        assert torch.isfinite(loss) and loss.requires_grad
        loss.backward()
        assert torch.isfinite(brain.log_edge_gain.grad).all()
        torch.nn.utils.clip_grad_norm_(selected.values(),1.)
        opt.step()
    assert not torch.equal(brain.log_edge_gain,original['log_edge_gain'])
    assert not torch.equal(brain.readout.weight,original['readout.weight'])
    for name,value in brain.state_dict().items():
        if name not in selected:
            assert torch.equal(value,original[name]),name
    assert metrics['delay_steps'] in range(2,7)
    assert metrics['episode']==1 and metrics['step']==32


def test_crash_cannot_skip_evaluation_maneuvers_or_change_following_task():
    brain,cfg,_=small_brain();cfg.brain.mask_motor_feedback=True
    kwargs=dict(batch=2,reference_speed=3.,episode_steps=64,reset_on_crash=False,seed=42)
    failed=FlightCostRollout(brain,cfg,dict(calibration=CAL,gate_sensor=SENSOR),PROFILE,**kwargs)
    other=FlightCostRollout(brain,cfg,dict(calibration=CAL,gate_sensor=SENSOR),PROFILE,**kwargs)
    failed.reset();other.reset();failed.state.quad.crashed[:]=True
    with torch.no_grad():
        for expected_step in (32,64):
            _,row=failed.loss(32)
            assert row['episode']==1 and row['step']==expected_step and row['crashes']==2
        failed.loss(32)
    other.reset()
    assert failed.episode==other.episode==2
    assert failed.speed==other.speed and failed.delay_steps==other.delay_steps
    torch.testing.assert_close(failed.heading0,other.heading0)
    torch.testing.assert_close(failed.sim.rate_coefficient,other.sim.rate_coefficient)


def test_snapshot_selection_prioritizes_fewer_crashes_over_lower_cost():
    clean=dict(crashed_tasks=0,mean_flight_cost=5.)
    fast_crash=dict(crashed_tasks=1,mean_flight_cost=1.)
    improved=dict(crashed_tasks=0,mean_flight_cost=4.)
    assert evaluation_rank(improved)<evaluation_rank(clean)<evaluation_rank(fast_crash)
