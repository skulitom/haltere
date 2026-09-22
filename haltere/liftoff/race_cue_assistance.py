"""Causal visible-checkpoint guidance with the connectome as motor controller."""
from types import SimpleNamespace

import numpy as np
import torch

from ..vision.camera import Camera, quat_wxyz_to_mat


class RaceCueAssistance:
    """Follow a current image bearing; brake and turn when no target is visible.

    A bearing is not a metric gate position. Use a bounded local goal and keep
    height changes proportional to the observed elevation. Never extrapolate
    a complete course or count image switches as completed checkpoints.
    """
    def __init__(self, sensor, pose_history, speed=2., *, reference_speed=2.):
        if not sensor:
            raise ValueError('Race cue assistance requires a calibrated camera')
        if not np.isfinite(speed) or not 0 < speed <= 5:
            raise ValueError('Assisted speed must be finite and in (0, 5] m/s')
        if not np.isfinite(reference_speed) or not 0 < reference_speed <= 10:
            raise ValueError('Invalid trained motor reference speed')
        self.camera = Camera(320, 180, sensor['focal_320'], sensor['tilt_deg'])
        self.pose_history, self.speed = pose_history, speed
        self.reference_speed = reference_speed
        self.host = SimpleNamespace(flow_gain=1.)
        self.pilot = SimpleNamespace(carrot=np.zeros(3), target=None, mode=4,
                                     n_passes=0, sight_yaw=0.)
        self.last_capture = self.last_seen = self.last_time = None
        self.direction = None
        self.edge = False
        self.below = False
        self.above = False
        self.above_hold = None
        self.launching = True
        self.hold = None
        self.side = 1.
        self.cue = None
        self.frames = 0

    def update(self, senses, omega, detection, capture_time, now):
        position = senses['pos'][0].cpu().numpy().astype(float)
        if position[2] >= .6:
            self.launching = False
        velocity = senses['vel_world'][0].cpu().numpy().astype(float)
        rotation = quat_wxyz_to_mat(senses['quat'][0].cpu().numpy())
        dt = .01 if self.last_time is None else float(np.clip(now-self.last_time, 0., .1))
        self.last_time = now
        if self.hold is None:
            self.hold = position+np.array([0., 0., 1.2])
        cue = detection.get('race_cue') if detection else None
        if (capture_time is not None and 0 <= now-capture_time <= .12
                and (self.last_capture is None or capture_time > self.last_capture)):
            self.last_capture = capture_time
            self.cue = cue
            if cue is not None:
                if not np.isfinite([cue['u'], cue['v']]).all() or not (
                        0 <= cue['u'] <= 1 and 0 <= cue['v'] <= 1):
                    raise ValueError('Invalid race cue image position')
                _, q = self.pose_history.at(capture_time)
                aim_u = cue.get('aim_u', cue['u'])
                if not np.isfinite(aim_u) or not 0 <= aim_u <= 1:
                    raise ValueError('Invalid race cue clearance position')
                ray = self.camera.unproject_body(np.array([[aim_u*320, cue['v']*180]]))[0]
                self.direction = quat_wxyz_to_mat(q) @ ray
                self.last_seen, self.edge = capture_time, cue['edge']
                self.below = cue['edge'] and cue['v'] > .96 and .1 < cue['u'] < .9
                self.above = cue['edge'] and cue['v'] < .04 and .1 < cue['u'] < .9
                if self.above and self.above_hold is None:
                    self.above_hold = position.copy()
                elif not self.above:
                    self.above_hold = None
                self.frames += 1
        fresh = self.last_seen is not None and now-self.last_seen < .25
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        world_rate = float((rotation @ np.asarray(omega))[2])
        if fresh:
            d = self.direction
            angle = (np.arctan2(d[1], d[0])-yaw+np.pi) % (2*np.pi)-np.pi
            if abs(angle) > .08:
                self.side = np.sign(angle)
            vertical_edge = self.below or self.above
            moving = (not self.edge or vertical_edge) and abs(angle) < np.deg2rad(55)
            yaw_rate = float(np.clip(1.6*angle-.22*world_rate, -.8, .8))
            if self.edge and not vertical_edge and abs(yaw_rate) < .35:
                yaw_rate = .35*self.side
            if moving:
                direction = d/max(np.linalg.norm(d[:2]), .1)
                # Slow through sharp direction changes and suppress lateral drift.
                lead = 3.*max(.25, np.cos(angle)**2)
                if vertical_edge:
                    lead = 1.  # slow while recovering a vertical off-screen cue
                elif abs(direction[2])*lead > 1.2:
                    # Preserve the observed slope when bounding the goal. Only
                    # clipping height drives too far forward on a steep climb.
                    lead = 1.2/abs(direction[2])
                relative = direction*lead
                relative[:2] -= .8*(velocity[:2]-direction[:2]*(velocity[:2]@direction[:2]))
                relative[2] = np.clip(relative[2], -1.2, 1.2)
                if self.below:
                    relative[2] = -1.2
                elif self.above:
                    # The clipped marker supplies no upper elevation or forward
                    # distance. Recover its bearing while holding horizontal
                    # position instead of advancing toward unseen geometry.
                    relative[:2] = self.above_hold[:2]-position[:2]-1.2*velocity[:2]
                    relative[2] = 1.2
                # Launch clearance is temporary. The start elevation is not
                # terrain height: later checkpoints may be below a rooftop.
                if self.launching and position[2] < .6:
                    relative[2] = max(relative[2], 1.2-position[2])
                self.hold = position.copy()
                self.pilot.mode = 2
            else:
                relative = self.hold-position
                relative[:2] -= 1.2*velocity[:2]
                self.pilot.mode = 4
        else:
            relative = self.hold-position
            relative[:2] -= 1.2*velocity[:2]
            # A missing cue in a current image permits search. A capture outage
            # permits only braking on live odometry, never a blind search turn.
            image_current = capture_time is not None and 0 <= now-capture_time <= .25
            yaw_rate = .45*self.side if image_current else 0.
            self.pilot.mode = 4
        horizontal = np.linalg.norm(relative[:2])
        if horizontal > 3:
            relative[:2] *= 3/horizontal
        relative[2] = np.clip(relative[2], -1.2, 1.2)
        self.pilot.carrot = position+relative
        self.pilot.target = SimpleNamespace(t_last=self.last_seen) if fresh else None
        desired_yaw = -yaw_rate/2.3
        self.pilot.sight_yaw += float(np.clip(desired_yaw-self.pilot.sight_yaw, -2*dt, 2*dt))
        self.host.flow_gain = max(1., self.reference_speed/self.speed)
        scaled_velocity = senses['vel_world']*senses['vel_world'].new_tensor(
            [self.host.flow_gain, self.host.flow_gain, 1.])
        modified = {**senses, 'vel_world': scaled_velocity,
                    'vel_body': scaled_velocity @ torch.as_tensor(rotation, dtype=scaled_velocity.dtype,
                                                                 device=scaled_velocity.device)}
        return rotation.T @ relative, modified

    def command(self, action):
        result = np.array(action, copy=True)
        result[3] = self.pilot.sight_yaw
        return result

    def metadata(self):
        return dict(mode='race-cue', goal_source='visible next-checkpoint ring with local flag clearance',
                    visible_race_cues=True, runtime_route_oracle=False,
                    local_flag_clearance=True, visible_route_arrows_for_clearance_side=True,
                    bottom_edge_recovery='bounded descent with reduced forward goal',
                    top_edge_recovery='bounded climb with horizontal position hold and braking',
                    steep_bearing='reduce horizontal lead to preserve observed vertical slope',
                    launch_clearance='released after first 0.6 m ascent; no persistent start-height floor',
                    capture_outage='brake on live odometry without search yaw after 0.25 s; runner pauses at 0.5 s',
                    yaw_assistance=True, speed_assistance=True, nominal_speed_mps=self.speed,
                    trained_motor_reference_mps=self.reference_speed,
                    effective_speed_setting_mps=min(self.speed, self.reference_speed),
                    cue_frames=self.frames, estimated_passages=None,
                    motor_control='brain throttle/roll/pitch; assisted yaw',
                    limitations='Race guidance only; no freestyle objective or completed-lap inference')
