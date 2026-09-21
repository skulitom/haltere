"""Identify the local throttle response from airborne telemetry near hover.

Only use explicitly selected free-flight windows. Terrain support, collisions,
large vertical speeds and unexcited throttle cannot identify rotor thrust.
The resulting power curve is a local equivalent under the existing stick
mapping, not a measurement of full-throttle thrust or the physical motor curve.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation,Slerp

G=9.80665


def fit_vertical(frame,start_s=7.,end_s=110.):
    d=frame.drop_duplicates('ts').copy()
    ts=d.ts.to_numpy()-d.ts.iloc[0]
    if not 0<=start_s<end_s<=ts[-1]:
        raise ValueError('Select a bounded free-flight interval within the recording')
    t=np.arange(0,ts[-1],.01)
    get=lambda columns:np.column_stack([np.interp(t,ts,d[k]) for k in columns])
    velocity=get(['vx','vy','vz'])
    rotation=Slerp(ts,Rotation.from_quat(d[['qx','qy','qz','qw']].values))(t)
    acceleration=gaussian_filter1d(velocity,4,axis=0,order=1)/.01
    specific=rotation.inv().apply(acceleration+np.array([0.,0.,G]))[:,2]
    body_velocity=rotation.inv().apply(velocity)
    good=((t>start_s)&(t<end_s)&(get(['z'])[:,0]>.6)
          &(rotation.as_matrix()[:,2,2]>.95)&(abs(body_velocity[:,2])<.25)
          &(np.linalg.norm(acceleration,axis=1)<5.))
    if good.sum()<200:
        raise ValueError('Insufficient near-hover airborne samples')
    fits=[]
    for lag in (0.,.02,.04,.06,.08):
        # Match input/acceleration bandwidth. Smoothing only the derivative
        # attenuates the identified gain when the controller corrects rapidly.
        throttle=gaussian_filter1d(np.interp(t-lag,ts,d.in_thr),4)[good]
        if np.ptp(throttle)<.025:
            raise ValueError('Insufficient throttle excitation')
        y=specific[good]
        fit=least_squares(lambda p:p[0]+p[1]*throttle-y,[7.5,18.],
                          bounds=([0.,.1],[30.,100.]),loss='soft_l1',f_scale=.2)
        bias,slope=fit.x
        fits.append(dict(input_lag_s=lag,bias_mps2=float(bias),slope_mps2_per_processed=float(slope),
                         hover_processed=float((G-bias)/slope),samples=int(good.sum()),
                         throttle_range=[float(throttle.min()),float(throttle.max())],
                         acceleration_rms=float(np.sqrt(np.mean(fit.fun**2))),
                         robust_error=float(np.mean(np.minimum(fit.fun**2,.5**2)))))
    return min(fits,key=lambda row:row['robust_error'])


def equivalent_power_curve(fit,calibration):
    """Preserve measured hover and local acceleration slope in simulator units."""
    scale=calibration['throttle_scale']
    action=calibration['hover_stick_sim']+(fit['hover_processed']-calibration['hover_processed'])/scale
    hover=(action+1)/2
    if scale<=0 or not 0<hover<1:
        raise ValueError('Invalid calibrated hover/mapping')
    exponent=2*scale*fit['slope_mps2_per_processed']*hover/G
    return dict(twr=float(hover**-exponent),thrust_exp=float(exponent))
