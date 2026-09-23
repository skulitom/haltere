from types import SimpleNamespace

import numpy as np
import pytest

from haltere.liftoff.dynamics_calibration import DynamicsCalibration, pulse_plan
from haltere.liftoff.stickcal import RadialSticks
from haltere.liftoff.pilot import LiftoffMapping


CAL = dict(hover_stick_sim=-.3,hover_processed=.12,throttle_scale=1.,stick_sign=[-1,1,1])
RATES = SimpleNamespace(rc_rate=(1.55,1.55,1.),super_rate=(.73,.73,.73),expo=(.3,.3,.3))


def update(c, t, pos=(0,0,15), vel=(0,0,0), q=(1,0,0,0), omega=(0,0,0)):
    return c.update_state(pos,vel,q,omega,t)


def test_hover_completes_only_after_a_continuously_stable_interval():
    c = DynamicsCalibration('hover',CAL)
    update(c,0); update(c,1.5,vel=(1,0,0)); update(c,2); update(c,3.9)
    assert not c.complete
    update(c,4)
    assert c.complete and c.phase=='complete' and not c.events


def test_pulses_require_independent_recovery_and_log_actual_duration():
    c = DynamicsCalibration('throttle',CAL)
    update(c,0);update(c,2)
    assert c.active.processed==.25
    update(c,2.1);assert c.phase=='pulse'
    update(c,2.26)
    assert c.phase=='recover' and c.active is None
    assert c.events[0]['duration_s']==pytest.approx(.26)
    update(c,4,vel=(0,0,1));update(c,5);update(c,6.9)
    assert len(c.events)==1
    update(c,7)
    assert len(c.events)==2 and c.pilot.n_passes==1


def test_angular_pulse_ends_before_requested_duration_at_angle_limit():
    c = DynamicsCalibration('roll',CAL,RATES)
    update(c,0);update(c,2)
    q=[np.cos(np.deg2rad(18)),np.sin(np.deg2rad(18)),0,0]
    update(c,2.03,q=q)
    assert c.phase=='recover' and c.events[-1]['termination']=='angle_limit'
    assert c.events[-1]['duration_s']==pytest.approx(.03)


def test_full_throttle_request_reaches_processed_endpoint_without_yaw_competition():
    c = DynamicsCalibration('throttle',CAL)
    c.active = pulse_plan('throttle')[-1]
    c.pilot.sight_yaw=.2
    mapping=LiftoffMapping(stick_sign=(-1,1,1),hover_stick_sim=-.3,hover_processed_game=.12,
                          stick_model=RadialSticks(.25,dict(roll=-1,pitch=1,yaw=1,throttle=1)))
    raw=mapping.to_raw(c.command([-.3,0,0,.2]))
    assert mapping.stick_model.processed('throttle','yaw',raw[0],raw[3])==pytest.approx((1.,0.))
    c.active=pulse_plan('roll',RATES)[0]
    raw=mapping.to_raw(c.command([-.3,0,0,0]))
    assert mapping.stick_model.processed('roll','pitch',raw[1],raw[2])==pytest.approx((.25,0.))


def test_motion_limits_and_recovery_timeout_are_not_silently_bypassed():
    for kwargs in [dict(pos=(0,0,36)),dict(pos=(16,0,15)),dict(vel=(16,0,0)),
                   dict(q=[np.cos(np.deg2rad(30)),np.sin(np.deg2rad(30)),0,0])]:
        with pytest.raises(RuntimeError,match='motion limit'):
            update(DynamicsCalibration('hover',CAL),0,**kwargs)
    c=DynamicsCalibration('hover',CAL)
    update(c,0)
    with pytest.raises(RuntimeError,match='motion limit'):
        update(c,1,pos=(0,0,2))
    c=DynamicsCalibration('throttle',CAL);update(c,0);update(c,2);update(c,2.3)
    with pytest.raises(RuntimeError,match='stable hover'):
        update(c,38,pos=(0,0,16))
    with pytest.raises(ValueError,match='backwards'):
        update(c,1)
    with pytest.raises(ValueError,match='unit quaternion'):
        update(DynamicsCalibration('hover',CAL),0,q=(2,0,0,0))


def test_plan_and_metadata_distinguish_calibration_from_flight_acceptance():
    assert len(pulse_plan('throttle'))==12
    assert len(pulse_plan('yaw',RATES))==24
    for axis in ('roll','pitch','yaw'):
        plan=pulse_plan(axis,RATES)
        assert sorted(set(p.processed for p in plan))==[-1.,-.75,-.5,-.25,.25,.5,.75,1.]
        assert max(p.duration_s for p in plan)<=.25
    c=DynamicsCalibration('hover',CAL)
    assert not c.metadata()['autonomous_evaluation_eligible']
    c.bind(SimpleNamespace(timestamp=10))
    with pytest.raises(ValueError,match='causal'):
        c.bind(SimpleNamespace(timestamp=9))


def test_fast_pulses_bound_requested_rotation_and_brake_before_lagged_angle_limit():
    assert pulse_plan('roll',RATES)[-1].duration_s < .02
    assert pulse_plan('roll',RATES)[-1].duration_s < pulse_plan('yaw',RATES)[-1].duration_s
    with pytest.raises(ValueError,match='rate profile'):
        pulse_plan('roll')
    c=DynamicsCalibration('roll',CAL,RATES);update(c,0);update(c,2)
    q=[np.cos(np.deg2rad(9)),np.sin(np.deg2rad(9)),0,0]
    update(c,2.02,q=q,omega=(np.deg2rad(600),0,0))
    assert c.phase=='recover' and c.events[-1]['termination']=='predicted_angle_limit'
    assert c.events[-1]['predicted_stop_angle_deg']==pytest.approx(42.)
