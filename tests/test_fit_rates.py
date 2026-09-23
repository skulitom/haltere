import numpy as np
import pytest

from haltere.liftoff.fit_rates import interval_response, rate_curve, fit_response, predict_window


def test_interval_average_preserves_angle_across_unequal_intervals():
    dt=np.array([.01,.02,.01,.03]);target=np.full(4,3.);tau=.025
    averaged=interval_response(dt,target,0.,tau)
    duration=dt.sum()
    assert averaged@dt==pytest.approx(3*(duration-tau*(1-np.exp(-duration/tau))))
    assert averaged[0]<3*(1-np.exp(-dt[0]/tau))  # average differs from endpoint
    with pytest.raises(ValueError):
        interval_response([0],[1],0,tau)


def test_shaped_saturation_is_odd_and_distinct_from_original_curve():
    x=np.array([-.75,-.5,-.25,0,.25,.5,.75])
    after=rate_curve(x,180,.3,.73,True)
    before=rate_curve(x,180,.3,.73,False)
    assert after==pytest.approx(-after[::-1])
    assert np.all(np.diff(after)>0)
    assert after[-1]<before[-1]


def test_fit_keeps_third_repetition_out_of_parameter_estimation():
    truth=dict(coefficient_deg_s=180.,response_tau_s=.025,expo=.3,super_rate=.73,super_after_expo=True)
    windows=[]
    for amplitude in [.25,.5,.75]:
        for repetition in range(3):
            x=np.r_[np.zeros(10),np.full(25,amplitude),np.zeros(40)]
            w=dict(inputs=x,intervals=np.full(len(x),.01),initial_rate=0.,repetition=repetition,
                   event=dict(index=len(windows),processed=amplitude))
            w['rate']=predict_window(w,truth)+(0.1 if repetition==2 else 0.)
            windows.append(w)
    result=fit_response(windows,coefficient_start=220.)
    assert result['profile']['coefficient_deg_s']==pytest.approx(180.,rel=1e-4)
    assert result['profile']['response_tau_s']==pytest.approx(.025,rel=1e-4)
    assert result['fit']['rmse_deg_s']<.001
    assert result['validation']['rmse_deg_s']==pytest.approx(np.rad2deg(.1))
