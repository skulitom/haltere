"""Offline angular-response identification from native telemetry intervals.

Game input is held forward; quaternion differences are interval averages. Never
linearly interpolate a short stick pulse before a nonlinear rate conversion.
The post-expo saturation curve is an empirical hypothesis, not a claim about
Liftoff's source code. Fits do not update vehicle settings or a deployed brain.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def rate_curve(stick, coefficient_deg_s, expo, super_rate, super_after_expo=True):
    stick=np.asarray(stick,dtype=float)
    shaped=stick*(1-expo+expo*abs(stick)**3)
    magnitude=abs(shaped if super_after_expo else stick)
    return np.deg2rad(coefficient_deg_s*shaped/np.maximum(.01,1-super_rate*magnitude))


def interval_response(intervals, targets, initial_rate, response_tau_s):
    """Exact first-order response averaged over each (possibly unequal) interval."""
    intervals=np.asarray(intervals,float);targets=np.asarray(targets,float)
    if (intervals.shape!=targets.shape or not np.isfinite(np.r_[intervals,targets,initial_rate,response_tau_s]).all()
            or np.any(intervals<=0) or response_tau_s<=0):
        raise ValueError('Use finite targets, positive intervals and a positive response time')
    out=[];rate=float(initial_rate)
    for duration,target in zip(intervals,targets):
        decay=np.exp(-duration/response_tau_s)
        out.append(target+(rate-target)*response_tau_s/duration*(1-decay))
        rate=target+(rate-target)*decay
    return np.asarray(out)


def pulse_windows(frame, events, axis):
    """Keep all recorded pulse windows, including gaps and failed repetitions."""
    if axis not in ('roll','pitch','yaw'):
        raise ValueError('Expected an angular axis')
    d=frame.drop_duplicates('ts');t=d.ts.to_numpy();dt=np.diff(t)
    if len(t)<2 or not np.isfinite(t).all() or np.any(dt<=0):
        raise ValueError('Expected increasing finite telemetry timestamps')
    index=('roll','pitch','yaw').index(axis)
    rotation=Rotation.from_quat(d[['qx','qy','qz','qw']].to_numpy())
    rate=(rotation[:-1].inv()*rotation[1:]).as_rotvec()[:,index]/dt
    inputs=d['in_'+axis].to_numpy()[:-1]*(-1,1,-1)[index]
    if not np.isfinite(inputs).all() or np.any(abs(inputs)>1.001):
        raise ValueError('Expected actual game-processed inputs in [-1, 1]')
    windows=[]
    for event in events:
        if 'ended' not in event:
            continue
        ids=np.flatnonzero((t[:-1]>=event['started']-.1)&(t[1:]<=event['ended']+.5))
        if len(ids)<8:
            raise ValueError('Insufficient recorded intervals around pulse')
        windows.append(dict(event=event,intervals=dt[ids],rate=rate[ids],inputs=inputs[ids],
                            time=t[ids]-event['started'],initial_rate=rate[ids[0]],
                            repetition=(event['index']%6)//2))
    if not windows:
        raise ValueError('No complete pulse windows')
    return windows


def predict_window(window, profile):
    targets=rate_curve(window['inputs'],profile['coefficient_deg_s'],profile['expo'],
                       profile['super_rate'],profile['super_after_expo'])
    return interval_response(window['intervals'],targets,window['initial_rate'],profile['response_tau_s'])


def evaluate_response(windows, profile):
    pulses=[];errors=[]
    for window in windows:
        error=predict_window(window,profile)-window['rate'];errors.append(error)
        pulses.append(dict(index=window['event']['index'],processed=window['event']['processed'],
            repetition=window['repetition'],intervals=len(error),
            intervals_over_15ms=int(np.sum(window['intervals']>.015)),
            rmse_deg_s=float(np.rad2deg(np.sqrt(np.mean(error**2))))))
    return dict(rmse_deg_s=float(np.rad2deg(np.sqrt(np.mean(np.concatenate(errors)**2)))),pulses=pulses)


def fit_response(windows, *, coefficient_start, expo=.3, super_rate=.73, super_after_expo=True):
    fit=[w for w in windows if w['repetition'] in (0,1)]
    validation=[w for w in windows if w['repetition']==2]
    if not fit or not validation:
        raise ValueError('Need two fitting repetitions and a separate third repetition')
    profile=dict(coefficient_deg_s=coefficient_start,response_tau_s=.004,expo=expo,
                 super_rate=super_rate,super_after_expo=super_after_expo)
    def residual(parameters):
        candidate={**profile,'coefficient_deg_s':parameters[0],'response_tau_s':parameters[1]}
        return np.concatenate([predict_window(w,candidate)-w['rate'] for w in fit])
    result=least_squares(residual,[coefficient_start,.004],bounds=([20,.0001],[1000,.2]),
                         loss='soft_l1',f_scale=.15)
    profile.update(coefficient_deg_s=float(result.x[0]),response_tau_s=float(result.x[1]))
    return dict(profile=profile,fit=evaluate_response(fit,profile),validation=evaluate_response(validation,profile),
        limitations=['Effective angular response only; not physical torque/inertia identification.',
                     'Sub-frame response times are unresolved by 100 Hz telemetry.',
                     'All sample gaps are retained; the input transition inside a gap is unknown.',
                     'Independent flight validation is required before promotion.'])
