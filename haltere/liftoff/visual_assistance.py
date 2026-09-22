"""Explicit Rabbit pilot assistance around a trained visual connectome.

Reuse the existing visual tracker and guidance, with the checkpoint's detector
geometry and original camera timestamps. There is no route or learned navigation
teacher. The brain supplies throttle/roll/pitch; Rabbit supplies the yaw command.
"""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch

from .sightpilot import MODE_NAMES, SightParams, SightPilot
from ..vision.camera import Camera, quat_wxyz_to_mat
from ..vision.runtime import Detection


class VisualPilotAssistance:
    def __init__(self, sensor, pose_history, speed=2.):
        if not sensor:
            raise ValueError('Rabbit assistance requires a camera gate sensory contract')
        if not np.isfinite(speed) or not 0 < speed <= 5:
            raise ValueError('Assisted speed must be finite and in (0, 5] m/s')
        self.pose_history = pose_history
        self.now = 0.
        self.last_capture_time = None
        self.frames = 0
        self.detection = Detection()
        camera = Camera(320, 180, sensor['focal_320'], sensor['tilt_deg'])
        # These detectors already encode calibrated metric range and may locate
        # the opening itself. The older detector's range correction and 1.5 m
        # vertical offset must not silently be applied to a new measurement.
        params = SightParams(p_min=.8, range_corr=None,
                             centre_offset_m=sensor['centre_offset_m'],
                             v_launch=speed, v_cruise=speed, v_gate=min(speed, 3.2),
                             v_unsure=min(speed, 2.5), v_blind=min(speed, 2.), v_search=min(speed, 2.),
                             v_gate_turn=min(speed, 2.8), goal_max=3.)
        params.validate()
        self.host = SimpleNamespace(clock=lambda: self.now, last_R=np.eye(3),
                                    last_vel=np.zeros(3), omega=np.zeros(3), flow_gain=1.,
                                    pose_at=self.pose_at,
                                    vision=SimpleNamespace(cam=camera, get=lambda: self.detection))
        self.pilot = SightPilot(self.host, params)

    def pose_at(self, stamp):
        position, quaternion = self.pose_history.at(stamp)
        return position, quat_wxyz_to_mat(quaternion)

    def update(self, senses, omega, detection, capture_time, now):
        """Return a bounded body-frame carrot and speed-scaled sensory inputs.

        Real telemetry stays unmodified for flight limits, logs and tracking.
        Repeated/stale camera samples never count as new detector evidence.
        """
        self.now = float(now)
        rotation = quat_wxyz_to_mat(senses['quat'][0].detach().cpu().numpy())
        position = senses['pos'][0].detach().cpu().numpy()
        self.host.last_R = rotation
        self.host.last_vel = senses['vel_world'][0].detach().cpu().numpy()
        self.host.omega = np.asarray(omega)
        if (capture_time is not None and np.isfinite(capture_time)
                and 0 <= now-capture_time <= .12
                and (self.last_capture_time is None or capture_time > self.last_capture_time)):
            self.last_capture_time = float(capture_time)
            self.frames += 1
            item = Detection(t=float(capture_time), frames=self.frames)
            if detection is not None:
                point = np.asarray(detection['point'], dtype=float)
                distance = float(np.linalg.norm(point))
                probability, width = float(detection['p']), float(detection['width'])
                if (point.shape != (3,) or not np.isfinite(point).all()
                        or not np.isfinite([probability, width]).all()):
                    raise ValueError('Invalid camera measurement for Rabbit assistance')
                if distance > 0:
                    pixels, visible = self.host.vision.cam.project_body(point[None])
                    if visible[0]:
                        item = Detection(t=float(capture_time), p_visible=probability,
                                         u=float(pixels[0, 0]), v=float(pixels[0, 1]),
                                         width_px=width, direction_body=point/distance,
                                         dist_m=distance, frames=self.frames)
            self.detection = item
        errors = self.pilot.errors
        relative = self.pilot.goal(position)
        if self.pilot.errors != errors:
            raise RuntimeError('Visual pilot assistance failed; stop this attempt')
        # Match the existing pilot's global horizontal speed-sense scheduling.
        # Preserve the new brain's own altitude and other sensory conventions.
        velocity = senses['vel_world'] * senses['vel_world'].new_tensor(
            [self.host.flow_gain, self.host.flow_gain, 1.])
        body_velocity = velocity @ torch.as_tensor(rotation, dtype=velocity.dtype, device=velocity.device)
        return relative, {**senses, 'vel_world': velocity, 'vel_body': body_velocity}

    def command(self, brain_action):
        command = np.array(brain_action, copy=True)
        command[3] = self.pilot.sight_yaw
        return command

    def metadata(self):
        return dict(mode='rabbit', goal_source='camera Rabbit pilot', yaw_assistance=True,
                    speed_assistance=True, runtime_route_oracle=False,
                    params=asdict(self.pilot.params), estimated_passages=self.pilot.n_passes,
                    tracker_errors=self.pilot.errors,
                    final_mode=MODE_NAMES[self.pilot.mode],
                    action_columns='Unmodified brain output; command columns include pilot yaw',
                    replay_action='Unmodified brain output on assisted sensory inputs')
