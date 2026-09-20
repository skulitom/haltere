"""Fixed image sampling into the connectome's visual sensory population.

No navigation network, route, future pose or human control is an input here.
The learnable sensory encoder belongs to ConnectomeRNN itself.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

RETINA_SIZE = (12, 20)
RETINA_DIM = 3 * RETINA_SIZE[0] * RETINA_SIZE[1]


def retina_input(images):
    """RGB float images [...,3,H,W] -> fixed, HUD-masked samples [...,720]."""
    if images.shape[-3] != 3:
        raise ValueError('Expected RGB images')
    x = images.clone()
    h, w = x.shape[-2:]
    x[..., :round(.23*h), :] = .5
    x[..., round(.84*h):, :] = .5
    x[..., round(.34*h):round(.78*h), round(.82*w):] = .5
    shape = x.shape[:-3]
    return F.adaptive_avg_pool2d((x.reshape(-1, 3, h, w)-.5)*2, RETINA_SIZE).reshape(*shape, RETINA_DIM)


def visual_observation(sensors, motor_mean, task_config, retina):
    """Shared training/inference senses. External goal and absolute heading are absent."""
    from ..sim.tasks import observe_from_sensors
    zero = torch.zeros_like(sensors['vel_body'])
    obs = observe_from_sensors(sensors, zero, motor_mean, task_config, rel_b=zero)
    # A world heading can identify a particular course; the visual student must
    # decide from what it sees and its body motion, rather than a compass lookup.
    obs['compass'] = torch.zeros_like(obs['compass'])
    obs['retina'] = retina
    return obs


def brain_to_processed(action, calibration):
    """Brain order -> game processed [throttle,roll,pitch,yaw], never raw Xbox."""
    throttle = calibration['hover_processed'] + calibration['throttle_scale'] * (
        action[..., :1] - calibration['hover_stick_sim'])
    signs = action.new_tensor(calibration['stick_sign'])
    return torch.cat((throttle, action[..., 1:] * signs), -1)
