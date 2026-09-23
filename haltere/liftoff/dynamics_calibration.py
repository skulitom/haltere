"""Bounded system-identification pulses in an operator-verified empty arena.

This is calibration, not autonomous racing or freestyle evaluation. A PD motor
baseline holds a launch-relative hover target between pulses. The fly brain can
run in shadow; no weights or vehicle settings are changed. Actual processed
inputs, motion and pulse events must be measured before fitting any dynamics.
"""
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np

from ..vision.camera import quat_wxyz_to_mat


@dataclass(frozen=True)
class Pulse:
    axis: str
    processed: float
    duration_s: float


def pulse_plan(mode, rates=None, amplitudes=None):
    if mode not in ('hover', 'throttle', 'roll', 'pitch', 'yaw'):
        raise ValueError('Unknown dynamics calibration mode')
    if mode == 'hover':
        return []
    if mode != 'throttle' and rates is None:
        raise ValueError('Angular pulses require the calibrated vehicle rate profile')
    amplitudes = tuple(amplitudes) if amplitudes is not None else (.25, .5, .75, 1.)
    if (not amplitudes or not np.isfinite(amplitudes).all()
            or any(not 0 < a <= 1 for a in amplitudes)
            or any(a >= b for a, b in zip(amplitudes, amplitudes[1:]))):
        raise ValueError('Use strictly ascending finite pulse amplitudes in (0, 1]')
    result = []
    # Ascending amplitude, independent repetitions, stable recovery between all.
    for amplitude in amplitudes:
        for _ in range(3):
            for sign in ((1.,) if mode == 'throttle' else (1., -1.)):
                if mode == 'throttle':
                    duration = .25 if amplitude <= .5 else .15
                else:
                    import torch
                    from ..sim.rates import betaflight_rate_curve
                    axis = ('roll','pitch','yaw').index(mode)
                    rate = float(betaflight_rate_curve(torch.tensor(amplitude), rates.rc_rate[axis],
                                                       rates.super_rate[axis],rates.expo[axis]))
                    if not np.isfinite(rate) or rate <= 0:
                        raise ValueError('Use a finite positive calibrated rate curve')
                    # A retrospective angle cutoff is too late at high rates:
                    # measured command-to-game input lag was about 30 ms.
                    # Bound requested rotation before sending the pulse, too.
                    duration = min(.25,20./rate)
                result.append(Pulse(mode, sign*amplitude, duration))
    return result


class DynamicsCalibration:
    height = 15.
    stable_s = 2.
    recovery_timeout_s = 35.
    pulse_angle_limit_deg = 35.
    stop_latency_s = .04

    def __init__(self, mode, calibration, rates=None, amplitudes=None):
        self.mode, self.calibration = mode, calibration
        self.plan = pulse_plan(mode,rates,amplitudes)
        self.speed = self.reference_speed = 2.
        self.host = SimpleNamespace(flow_gain=1.)
        self.pilot = SimpleNamespace(carrot=np.array([0., 0., self.height]), target=None,
                                     mode=0, n_passes=0, sight_yaw=0.)
        self.start = self.timestamp = self.last_timestamp = None
        self.phase = 'settle'
        self.stable_since = self.phase_start = None
        self.heading = None
        self.next_pulse = 0
        self.active = None
        self.pulse_quaternion = None
        self.events = []
        self.complete = False
        self.airborne = False
        self.failure = None

    def bind(self, frame):
        timestamp = float(frame.timestamp)
        if not np.isfinite(timestamp) or self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError('Calibration requires causal finite telemetry')
        self.start = timestamp if self.start is None else self.start
        self.timestamp = self.last_timestamp = timestamp

    def update_state(self, position, velocity, quaternion, omega, timestamp):
        """Pure state transition; each pulse needs its own measured recovery."""
        position, velocity, quaternion, omega = [np.asarray(v, float) for v in
                                                (position, velocity, quaternion, omega)]
        if (position.shape != (3,) or velocity.shape != (3,) or quaternion.shape != (4,)
                or omega.shape != (3,) or not np.isfinite(np.r_[position,velocity,quaternion,omega,timestamp]).all()
                or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-4)):
            raise ValueError('Calibration needs finite measured motion and a unit quaternion')
        if self.timestamp is not None and timestamp < self.timestamp:
            raise ValueError('Calibration time went backwards')
        self.timestamp = timestamp
        self.start = timestamp if self.start is None else self.start
        rotation = quat_wxyz_to_mat(quaternion)
        tilt = np.rad2deg(np.arccos(np.clip(rotation[2,2], -1, 1)))
        self.airborne |= position[2] > 3.
        if (position[2] > 35 or self.airborne and position[2] < 3
                or np.linalg.norm(position[:2]) > 15 or np.linalg.norm(velocity) > 15
                or tilt > 55 or np.linalg.norm(omega) > np.deg2rad(1800)):
            self.failure = dict(timestamp=float(timestamp),position=position.tolist(),velocity=velocity.tolist(),
                                quaternion=quaternion.tolist(),omega=omega.tolist(),tilt_deg=float(tilt),
                                phase=self.phase,pulse_index=self.next_pulse-1)
            raise RuntimeError('Dynamics calibration motion limit exceeded')
        yaw = np.arctan2(rotation[1,0],rotation[0,0])
        self.heading = yaw if self.heading is None else self.heading
        heading_error = (self.heading-yaw+np.pi) % (2*np.pi)-np.pi
        self.pilot.sight_yaw = float(np.clip(-.6*heading_error+.025*omega[2],-.2,.2))
        stable = (np.linalg.norm(position-self.pilot.carrot)<.6 and np.linalg.norm(velocity)<.35
                  and np.linalg.norm(omega)<.25 and tilt<10 and abs(heading_error)<.12)
        self.phase_start = timestamp if self.phase_start is None else self.phase_start
        if self.phase == 'pulse':
            angle = np.rad2deg(2*np.arccos(np.clip(abs(quaternion@self.pulse_quaternion),0,1)))
            elapsed = timestamp-self.phase_start
            predicted_angle = angle+np.rad2deg(np.linalg.norm(omega))*self.stop_latency_s
            if elapsed >= self.active.duration_s or predicted_angle >= self.pulse_angle_limit_deg:
                self.events[-1].update(ended=timestamp, duration_s=elapsed,
                    predicted_stop_angle_deg=float(predicted_angle),
                    termination='angle_limit' if angle>=self.pulse_angle_limit_deg else
                                'predicted_angle_limit' if predicted_angle>=self.pulse_angle_limit_deg else 'duration')
                self.active = None
                self.phase, self.phase_start, self.stable_since = 'recover', timestamp, None
                self.pilot.n_passes += 1
        elif self.phase in ('settle','recover'):
            timeout = 70. if self.phase == 'settle' else self.recovery_timeout_s
            if timestamp-self.phase_start > timeout:
                raise RuntimeError('Dynamics calibration did not reach a stable hover')
            self.stable_since = (timestamp if self.stable_since is None else self.stable_since) if stable else None
            if self.stable_since is not None and timestamp-self.stable_since>=self.stable_s:
                if self.next_pulse == len(self.plan):
                    self.phase, self.complete = 'complete', True
                else:
                    self.active = self.plan[self.next_pulse]
                    self.events.append(dict(index=self.next_pulse, **asdict(self.active), started=timestamp))
                    self.next_pulse += 1
                    self.phase, self.phase_start = 'pulse', timestamp
                    self.pulse_quaternion = quaternion.copy()
        self.pilot.mode = dict(settle=0,pulse=1,recover=2,complete=3)[self.phase]
        return rotation.T@(self.pilot.carrot-position)

    def update(self, senses, omega, detection, capture_time, now):
        relative = self.update_state(senses['pos'][0].cpu().numpy(),senses['vel_world'][0].cpu().numpy(),
                                     senses['quat'][0].cpu().numpy(),omega,self.timestamp)
        return relative, senses

    def command(self, motor_action):
        command = np.array(motor_action, dtype=float, copy=True)
        command[3] = self.pilot.sight_yaw
        if self.active is not None:
            pulse, c = self.active, self.calibration
            if pulse.axis == 'throttle':
                command[0] = c['hover_stick_sim']+(pulse.processed-c['hover_processed'])/c['throttle_scale']
                command[3] = 0.  # Full throttle must not share radial travel with yaw.
            else:
                axis = ('roll','pitch','yaw').index(pulse.axis)
                command[axis+1] = pulse.processed/c['stick_sign'][axis]
        return command

    def metadata(self):
        return dict(mode='dynamics-calibration', sequence=self.mode, height_above_launch_m=self.height,
                    plan=[asdict(p) for p in self.plan], events=self.events, complete=self.complete,
                    phase=self.phase, stable_hover_before_each_pulse_s=self.stable_s,
                    angular_pulse_cutoff_deg=self.pulse_angle_limit_deg,
                    requested_rotation_limit_deg=20.,stop_latency_allowance_s=self.stop_latency_s,failure=self.failure,
                    runtime_route_oracle=False, autonomous_evaluation_eligible=False,
                    goal_source='Declared hover and identification pulses in verified empty arena',
                    motor_control='PD hover plus processed-input pulses; brain runs in shadow',
                    limits='Pulse targets are requests, not measured inputs or steady-state rates. No weights changed.')
