"""Local target/velocity coordinates shared by geometry and motor controllers.

This converts guidance, never computes motor commands. The brain contract uses
its training teacher's target gain and the visual assistant's velocity scaling;
actual tracking remains a learned, experimentally measured behavior.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GuidanceVelocityContract:
    horizontal_gain: float
    vertical_gain: float
    horizontal_speed: float
    vertical_speed: float

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Guidance gains and limits must be finite and positive')

    @classmethod
    def brain(cls, speed, reference_speed):
        if not np.isfinite([speed, reference_speed]).all() or min(speed, reference_speed) <= 0:
            raise ValueError('Guidance speeds must be finite and positive')
        scale = max(1., reference_speed / speed)
        return cls(max(.8, reference_speed / 3.) / scale, .8,
                   min(speed, reference_speed), 1.2)

    @classmethod
    def pd(cls, config, speed):
        return cls(config.position_gain, config.vertical_position_gain,
                   speed, config.max_vertical_speed)

    @property
    def gains(self):
        return np.array([self.horizontal_gain, self.horizontal_gain, self.vertical_gain])

    def world_velocity(self, body_target, rotation):
        desired = (rotation @ np.asarray(body_target)) * self.gains
        desired[:2] *= min(1., self.horizontal_speed / max(1e-9, np.linalg.norm(desired[:2])))
        desired[2] = np.clip(desired[2], -self.vertical_speed, self.vertical_speed)
        return desired

    def body_target(self, world_velocity, rotation):
        return rotation.T @ (np.asarray(world_velocity) / self.gains)

    def metadata(self):
        return dict(**asdict(self), frame='world FLU velocity to body-relative local target',
                    supplies_motor_commands=False,
                    measured_tracking_guarantee=False)
