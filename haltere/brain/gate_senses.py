"""Camera gate measurements for the connectome, shared by training and flight.

The detector supplies a point, never motor commands or a predicted route.
"""
import torch

from .retina import visual_observation
from ..sim.tasks import observe_from_sensors


def gate_observation(sensors, motor, task, retina, relative_gate, height_invariant=False,
                     gravity_aligned_height=False):
    if height_invariant:
        # Starting elevation is not height above terrain. Use a fixed velocity
        # normalization and constant legacy height channel; neither varies with
        # world coordinates. This contract requires a correspondingly trained brain.
        sensors = {**sensors,'altitude':torch.full_like(sensors['altitude'],1.5)}
    # This stage is trained with the detector as its visual frontend. Keep the
    # older raw-pixel channel inactive in both simulation and live flight.
    obs = visual_observation(sensors, motor, task, torch.zeros_like(retina))
    # Bound horizontal sensory magnitude without attenuating the measured
    # vertical error of distant gates. This is also used in the simulator.
    if gravity_aligned_height:
        # Preserve physical height, not camera/body z: pitch mixes distance into
        # body z, so clipping body x/y alone invents a climb for a distant level
        # gate. Gravity provides a yaw-independent vertical axis.
        up = -sensors['gravity_body']
        up = up/up.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        height = (relative_gate*up).sum(-1,keepdim=True)
        horizontal = relative_gate-up*height
        horizontal = horizontal*(3./horizontal.norm(dim=-1,keepdim=True).clamp_min(3.))
        relative = horizontal+up*height.clamp(-3.,3.)
    else:
        xy = relative_gate[..., :2]
        xy = xy * (3. / xy.norm(dim=-1, keepdim=True).clamp_min(3.))
        relative = torch.cat((xy, relative_gate[..., 2:].clamp(-3., 3.)), -1)
    obs['goal'] = observe_from_sensors(sensors, torch.zeros_like(relative), motor,
                                       task, rel_b=relative)['goal']
    return obs


def aperture_crossing(previous, current, centre, half_width=1.4, half_height=1., normal=None):
    """Forward intersection with an upright gate, including its aperture.

    Offline evaluation only. A point behind/beside a gate is not a crossing.
    Returns a boolean per segment and the interpolated intersection point.
    """
    if normal is None:
        normal = torch.zeros_like(centre)
        normal[...,0] = 1.
    normal = normal / normal.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    side = torch.stack((-normal[...,1],normal[...,0],torch.zeros_like(normal[...,0])),-1)
    before = ((previous-centre)*normal).sum(-1)
    after = ((current-centre)*normal).sum(-1)
    dx = after-before
    alpha = -before / torch.where(dx.abs()>1e-8,dx,torch.ones_like(dx))
    point = previous + alpha[..., None] * (current - previous)
    crossed = ((before < 0) & (after >= 0)
               & (dx > 0) & (((point-centre)*side).sum(-1).abs() < half_width)
               & ((point[..., 2]-centre[..., 2]).abs() < half_height))
    return crossed, point
