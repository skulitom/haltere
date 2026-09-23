"""Velocity-level visible-checkpoint guidance for faster, smoother races.

A separate, explicitly declared profile beside `RaceCueAssistance`, whose
behaviour is unchanged. It still reads only the game's visible next-checkpoint
ring through the camera, the drone's own telemetry and elapsed time. It loads
no route, course file, checkpoint list or per-course parameter.

The ring is a fixed-size HUD marker, so it provides a bearing and no range.
Rather than chase a short goal point, this profile requests a world velocity
along the filtered bearing. Speed falls continuously with the turn still
required, commands are acceleration-limited, and a brief cue dropout coasts on
the previous request instead of stopping. Clipped markers set bounded climb or
descent, and a descent that the vehicle cannot achieve is treated as support
by terrain and answered with a short climb. These are generic heuristics, not
obstacle avoidance or a completed-lap estimate.
"""
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
import torch

from ..brain.motor_baseline import measured_inverse_rate
from ..vision.camera import Camera, quat_wxyz_to_mat

# Measured original-drone yaw curve (runs/measured-dynamics-low-speed-20260923):
# coefficient deg/s, super rate applied after expo, expo. A loaded dynamics
# profile replaces this declared default.
DEFAULT_YAW_CURVE = (180.17256995580442, .73, .3)


@dataclass(frozen=True)
class FastCueConfig:
    min_speed_fraction: float = .3
    command_acceleration: float = 10.
    vertical_command_acceleration: float = 5.
    vertical_up: float = 3.5
    vertical_down: float = 3.
    edge_speed: float = 1.2
    below_weak_deg: float = 2.
    below_full_deg: float = 10.
    edge_sweep_after_s: float = 1.5
    edge_sweep_rate: float = 1.
    command_time_constant: float = .25
    surface_sink: float = 1.
    surface_sink_per_m: float = .5
    surface_release_m: float = .5
    side_speed_fraction: float = .3
    side_turn_deg: float = 75.
    coast_s: float = .6
    coast_distance_m: float = 4.
    search_yaw_rate: float = 1.2
    yaw_gain: float = 3.
    yaw_damping: float = .25
    max_yaw_rate: float = 3.
    yaw_slew: float = 8.
    direction_blend: float = .5
    new_target_deg: float = 30.
    feedforward_time_constant: float = .05
    support_after_s: float = .4
    support_climb_s: float = .6
    launch_height: float = .6

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Use finite positive fast cue parameters')
        if not self.min_speed_fraction < 1 or not self.direction_blend <= 1:
            raise ValueError('Fractions must stay below one')
        if not self.below_full_deg > self.below_weak_deg:
            raise ValueError('Bottom-edge evidence needs an increasing depression range')


class FastRaceCue:
    """Follow the visible checkpoint bearing with a continuous velocity request."""

    profile = 'fast-v1'

    def __init__(self, sensor, pose_history, speed=6., *, reference_speed=2., config=None,
                 yaw_curve=DEFAULT_YAW_CURVE, calibration=None):
        if not sensor:
            raise ValueError('Race cue assistance requires a calibrated camera')
        if not np.isfinite(speed) or not 0 < speed <= 20:
            raise ValueError('Fast cue speed must be finite and in (0, 20] m/s')
        if not np.isfinite(reference_speed) or not 0 < reference_speed <= 20:
            raise ValueError('Invalid motor reference speed')
        self.config = config or FastCueConfig()
        self.yaw_curve = tuple(float(v) for v in yaw_curve)
        if len(self.yaw_curve) != 3 or not np.isfinite(self.yaw_curve).all() or min(self.yaw_curve) <= 0:
            raise ValueError('Use a measured (coefficient, super rate, expo) yaw curve')
        # Throttle priority on the shared left stick needs the brain->processed map.
        self.calibration = None if calibration is None else tuple(
            float(calibration[k]) for k in ('hover_processed', 'throttle_scale', 'hover_stick_sim'))
        self.below_weight = 0.
        self.vertical_clip_since = None
        self.camera = Camera(320, 180, sensor['focal_320'], sensor['tilt_deg'])
        self.pose_history, self.speed, self.reference_speed = pose_history, float(speed), float(reference_speed)
        self.host = SimpleNamespace(flow_gain=1.)
        self.pilot = SimpleNamespace(carrot=np.zeros(3), target=None, mode=4, n_passes=0, sight_yaw=0.)
        self.last_capture = self.last_seen = self.last_time = None
        self.direction = None
        self.edge = self.below = self.above = False
        self.side = 1.
        self.cue = None
        self.frames = 0
        self.target_switches = 0
        self.velocity_command = None
        self.feedforward = np.zeros(3)
        self.support_since = None
        self.climb_until = None
        self.launching = True
        self.state = 'launch'
        self.state_time = {}

    def _ingest(self, detection, capture_time, now):
        cue = detection.get('race_cue') if detection else None
        if not (capture_time is not None and 0 <= now-capture_time <= .12
                and (self.last_capture is None or capture_time > self.last_capture)):
            return
        self.last_capture = capture_time
        self.cue = cue
        if cue is None:
            return
        if not np.isfinite([cue['u'], cue['v']]).all() or not (0 <= cue['u'] <= 1 and 0 <= cue['v'] <= 1):
            raise ValueError('Invalid race cue image position')
        aim_u = cue.get('aim_u', cue['u'])
        if not np.isfinite(aim_u) or not 0 <= aim_u <= 1:
            raise ValueError('Invalid race cue clearance position')
        _, q = self.pose_history.at(capture_time)
        ray = self.camera.unproject_body(np.array([[aim_u*320, cue['v']*180]]))[0]
        ray = quat_wxyz_to_mat(q) @ ray
        ray = ray/max(np.linalg.norm(ray), 1e-9)
        if self.direction is None or np.degrees(np.arccos(np.clip(ray @ self.direction, -1, 1))) > self.config.new_target_deg:
            if self.direction is not None:
                self.target_switches += 1
            self.direction = ray
        else:
            blended = self.direction+self.config.direction_blend*(ray-self.direction)
            self.direction = blended/max(np.linalg.norm(blended), 1e-9)
        self.last_seen, self.edge = capture_time, bool(cue['edge'])
        self.below = self.edge and cue['v'] > .96 and .1 < cue['u'] < .9
        self.above = self.edge and cue['v'] < .04 and .1 < cue['u'] < .9
        c = self.config
        if self.below:
            # A bottom clip only bounds the target below the clamped edge ray.
            # Nose-up braking lifts that ray toward the horizon, so weight the
            # descent by how far below the horizon it points; latch within one
            # clip episode so the response does not flicker as attitude changes.
            elevation = np.degrees(np.arcsin(np.clip(ray[2], -1, 1)))
            weight = float(np.clip((-elevation-c.below_weak_deg)/(c.below_full_deg-c.below_weak_deg), 0, 1))
            self.below_weight = max(self.below_weight, weight)
        else:
            self.below_weight = 0.
        if not ((self.below or self.above) and abs(cue['u']-.5) < .1):
            self.vertical_clip_since = None
        elif self.vertical_clip_since is None:
            self.vertical_clip_since = capture_time
        self.frames += 1

    def _desired(self, position, velocity, yaw, now):
        c = self.config
        horizontal_speed = np.linalg.norm(velocity[:2])
        if self.direction is None:
            return np.array([0., 0., 1.5 if self.launching else 0.]), 'wait'
        seen_age = now-self.last_seen
        coast = min(c.coast_s, max(.25, c.coast_distance_m/max(horizontal_speed, 1e-6)))
        if seen_age > coast:
            return np.zeros(3), 'search'
        if seen_age > .25 and self.velocity_command is not None:
            return self.velocity_command.copy(), 'coast'
        d = self.direction
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        dh = d[:2]/max(np.linalg.norm(d[:2]), 1e-9)
        if self.edge and not (self.below or self.above):
            # Beyond the horizontal field of view: turn toward that side.
            angle = np.arctan2(heading[0]*dh[1]-heading[1]*dh[0], heading @ dh)
            self.side = np.sign(angle) if abs(angle) > .02 else self.side
            turn = yaw+self.side*np.radians(c.side_turn_deg)
            return np.r_[np.array([np.cos(turn), np.sin(turn)])*self.speed*c.side_speed_fraction, 0.], 'side'
        reference = velocity[:2]/horizontal_speed if horizontal_speed > 1. else heading
        alignment = max(0., float(reference @ dh))
        fraction = c.min_speed_fraction+(1-c.min_speed_fraction)*alignment**2
        speed = self.speed*fraction
        slope = d[2]/max(np.linalg.norm(d[:2]), 1e-6)
        if self.below:
            # The target lies below the lower image edge: descend and slow down
            # so the depression angle can shrink, in proportion to the evidence.
            w = self.below_weight
            return (np.r_[dh*(speed*(1-w)+min(speed, c.edge_speed)*w), -c.vertical_down*w],
                    'below' if w > 0 else 'below_weak')
        if self.above:
            # The clipped elevation is only a lower bound: preserve that slope.
            speed = min(speed, c.vertical_up/max(slope, .2))
            return np.r_[dh*speed, c.vertical_up], 'above'
        vertical = speed*slope
        limit = c.vertical_up if vertical > 0 else c.vertical_down
        if abs(vertical) > limit:
            speed *= limit/abs(vertical)
            vertical = np.sign(vertical)*limit
        return np.r_[dh*speed, vertical], 'cue'

    def update(self, senses, omega, detection, capture_time, now):
        c = self.config
        position = senses['pos'][0].cpu().numpy().astype(float)
        velocity = senses['vel_world'][0].cpu().numpy().astype(float)
        rotation = quat_wxyz_to_mat(senses['quat'][0].cpu().numpy())
        dt = .01 if self.last_time is None else float(np.clip(now-self.last_time, 0., .1))
        self.last_time = now
        if position[2] >= c.launch_height:
            self.launching = False
        self._ingest(detection, capture_time, now)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        desired, state = self._desired(position, velocity, yaw, now)
        if self.launching:
            desired[:2] *= min(1., 1.5/max(np.linalg.norm(desired[:2]), 1e-9))
            desired[2] = max(desired[2], 1.5)
            state = 'launch'
        elif position[2] > -c.surface_release_m:
            # Height is measured from the launch point, which may be a rooftop
            # or hill. Sink gently near and above that plane; the support rule
            # below still recognises contact because the cap exceeds 0.8 m/s.
            desired[2] = max(desired[2], -min(c.vertical_down, c.surface_sink+c.surface_sink_per_m*max(0., position[2])))
        # Support: a requested descent the vehicle cannot achieve means contact
        # below (terrain or an object), not a controller fault. Climb briefly.
        if self.climb_until is not None and now < self.climb_until:
            desired[2] = max(desired[2], 1.)
            state = 'support_climb'
        elif (self.velocity_command is not None and self.velocity_command[2] < -.8
              and velocity[2] > max(-.25, self.velocity_command[2]+.6)):
            self.support_since = now if self.support_since is None else self.support_since
            if now-self.support_since > c.support_after_s:
                self.climb_until, self.support_since = now+c.support_climb_s, None
        else:
            self.support_since = None
        if self.velocity_command is None:
            self.velocity_command = velocity.copy()
        step = desired-self.velocity_command
        norm = np.linalg.norm(step[:2])
        # Taper the requested acceleration near the goal, so feedforward ends
        # smoothly instead of overshooting into a nose-up brake.
        limit = min(c.command_acceleration*dt, norm*dt/c.command_time_constant)
        if norm > limit:
            step[:2] *= limit/norm
        step[2] = np.clip(step[2], -c.vertical_command_acceleration*dt, c.vertical_command_acceleration*dt)
        previous = self.velocity_command.copy()
        self.velocity_command = self.velocity_command+step
        raw_ff = (self.velocity_command-previous)/max(dt, 1e-3)
        alpha = 1-np.exp(-dt/c.feedforward_time_constant)
        self.feedforward = self.feedforward+alpha*(raw_ff-self.feedforward)
        self.state = state
        self.state_time[state] = self.state_time.get(state, 0.)+dt
        # Yaw faces the observed checkpoint; searching turns toward its last side.
        world_rate = float((rotation @ np.asarray(omega))[2])
        if state == 'search' or self.direction is None:
            yaw_rate = c.search_yaw_rate*self.side if (capture_time is not None and 0 <= now-capture_time <= .25) else 0.
        else:
            angle = (np.arctan2(self.direction[1], self.direction[0])-yaw+np.pi) % (2*np.pi)-np.pi
            if abs(angle) > .08:
                self.side = np.sign(angle)
            yaw_rate = float(np.clip(c.yaw_gain*angle-c.yaw_damping*world_rate, -c.max_yaw_rate, c.max_yaw_rate))
            if (state in ('above', 'below') and self.vertical_clip_since is not None
                    and now-self.vertical_clip_since > c.edge_sweep_after_s):
                # A centred top/bottom clip cannot separate overhead/underneath
                # from behind; a slow yaw moves a target behind off the centre.
                yaw_rate = c.edge_sweep_rate*self.side
        # Invert the measured post-expo yaw curve so max_yaw_rate is honoured.
        desired_yaw = -float(np.clip(measured_inverse_rate(
            torch.tensor([np.degrees(yaw_rate)], dtype=torch.float64), *self.yaw_curve)[0], -1., 1.))
        self.pilot.sight_yaw += float(np.clip(desired_yaw-self.pilot.sight_yaw, -c.yaw_slew*dt, c.yaw_slew*dt))
        self.pilot.mode = {'cue': 2, 'launch': 2, 'below_weak': 2, 'coast': 3, 'side': 5, 'below': 6, 'above': 7,
                           'support_climb': 8}.get(state, 4)
        fresh = self.last_seen is not None and now-self.last_seen < .25
        self.pilot.target = SimpleNamespace(t_last=self.last_seen) if fresh else None
        self.pilot.carrot = position+self.velocity_command
        self.host.flow_gain = max(1., self.reference_speed/self.speed)
        scaled_velocity = senses['vel_world']*senses['vel_world'].new_tensor(
            [self.host.flow_gain, self.host.flow_gain, 1.])
        modified = {**senses, 'vel_world': scaled_velocity,
                    'vel_body': scaled_velocity @ torch.as_tensor(rotation, dtype=scaled_velocity.dtype,
                                                                 device=scaled_velocity.device)}
        # Body-frame one-second lead for goal-point motor contracts and logs.
        return rotation.T @ self.velocity_command, modified

    def command(self, action):
        result = np.array(action, copy=True)
        result[3] = float(np.clip(self.pilot.sight_yaw, -1., 1.))
        if self.calibration is not None:
            # Throttle and yaw share one pad stick clamped to the unit circle:
            # give throttle priority instead of silently losing thrust.
            hover, scale, hover_stick = self.calibration
            throttle = hover+scale*(float(result[0])-hover_stick)
            room = float(np.sqrt(max(.97**2-min(throttle*throttle, .97**2), 0.)))
            result[3] = float(np.clip(result[3], -room, room))
        return result

    def metadata(self):
        return dict(mode='race-cue', profile=self.profile,
                    goal_source='visible next-checkpoint ring with local flag clearance',
                    visible_race_cues=True, runtime_route_oracle=False,
                    local_flag_clearance=True, visible_route_arrows_for_clearance_side=True,
                    guidance='world velocity along the filtered cue bearing, acceleration-limited with feedforward',
                    speed_schedule='min_speed_fraction + (1-min)*cos^2(angle between velocity and bearing)',
                    bottom_edge='descent and slowing weighted by the clamped edge ray depression, latched per clip',
                    top_edge='climb while preserving the clipped slope bound',
                    centred_vertical_clip='slow search yaw after edge_sweep_after_s',
                    launch_surface='sink rate limited near and above the launch plane',
                    yaw_mapping='measured post-expo yaw curve inverse, throttle priority on the shared stick',
                    yaw_curve=list(self.yaw_curve),
                    support='requested descent not achieved for support_after_s -> short climb',
                    cue_dropout='coast on the previous request, then brake and search',
                    parameters=asdict(self.config), yaw_assistance=True, speed_assistance=True,
                    nominal_speed_mps=self.speed, trained_motor_reference_mps=self.reference_speed,
                    cue_frames=self.frames, target_switches_observed=self.target_switches,
                    state_seconds={k: round(v, 3) for k, v in self.state_time.items()},
                    estimated_passages=None,
                    limitations='Race guidance only; no freestyle objective, obstacle model or completed-lap inference')
