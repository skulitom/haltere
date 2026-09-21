"""Camera gate measurements for the connectome, shared by training and flight.

The detector supplies a point, never motor commands or a predicted route.
"""
import torch

from .retina import visual_observation
from ..sim.tasks import observe_from_sensors


def gate_observation(sensors, motor, task, retina, relative_gate):
    # This stage is trained with the detector as its visual frontend. Keep the
    # older raw-pixel channel inactive in both simulation and live flight.
    obs = visual_observation(sensors, motor, task, torch.zeros_like(retina))
    # Bound horizontal sensory magnitude without attenuating the measured
    # vertical error of distant gates. This is also used in the simulator.
    xy = relative_gate[..., :2]
    xy = xy * (3. / xy.norm(dim=-1, keepdim=True).clamp_min(3.))
    relative = torch.cat((xy, relative_gate[..., 2:].clamp(-3., 3.)), -1)
    obs['goal'] = observe_from_sensors(sensors, torch.zeros_like(relative), motor,
                                       task, rel_b=relative)['goal']
    return obs


def aperture_crossing(previous, current, centre, half_width=1.4, half_height=1.):
    """Forward intersection with an x-normal gate, including its aperture.

    Offline evaluation only. A point behind/beside a gate is not a crossing.
    Returns a boolean per segment and the interpolated intersection point.
    """
    dx = current[..., 0] - previous[..., 0]
    alpha = (centre[..., 0] - previous[..., 0]) / torch.where(dx.abs()>1e-8,dx,torch.ones_like(dx))
    point = previous + alpha[..., None] * (current - previous)
    crossed = ((previous[..., 0] < centre[..., 0]) & (current[..., 0] >= centre[..., 0])
               & (dx > 0) & ((point[..., 1]-centre[..., 1]).abs() < half_width)
               & ((point[..., 2]-centre[..., 2]).abs() < half_height))
    return crossed, point
