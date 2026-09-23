"""Privileged route teacher for explicitly labelled collection/qualification only."""
import hashlib
from types import SimpleNamespace

import numpy as np

from .collection_route import CollectionRoute
from ..vision.camera import quat_wxyz_to_mat


class OracleCollectionAssistance:
    def __init__(self, path, *, reference_speed, motor_controller='pd'):
        if motor_controller not in ('brain','pd'):
            raise ValueError('Unknown oracle motor controller')
        self.motor_controller=motor_controller
        self.route = CollectionRoute(path)
        self.route_sha256 = hashlib.sha256(self.route.path.read_bytes()).hexdigest()
        self.speed = self.route.data['speed_mps']
        self.reference_speed = reference_speed
        self.host = SimpleNamespace(flow_gain=1.)
        self.pilot = SimpleNamespace(carrot=np.zeros(3), target=None, mode=2,
                                     n_passes=0, sight_yaw=0.)
        self.points = None
        self.progress = 0.
        self.last_time = None
        self.launching = True
        self.complete = False
        self.finish_speed=float(self.route.data.get('finish_speed_mps',float('inf')))
        self.finish_hold=float(self.route.data.get('finish_hold_s',0.))
        if (not np.isfinite(self.finish_hold) or self.finish_hold<0 or self.finish_speed<=0
                or np.isnan(self.finish_speed) or self.finish_hold and not np.isfinite(self.finish_speed)):
            raise ValueError('A settling route needs a finite positive finish speed and nonnegative hold duration')
        self.stable_since=None
        self.launch_heading=None
        self.launch_stable_since=None
        self.launch_completed_at=None

    def bind(self, frame):
        if self.points is not None:
            return
        self.points = self.route.relative_waypoints(frame)
        self.lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.distance = np.r_[0., self.lengths.cumsum()]

    def update(self, senses, omega, detection, capture_time, now):
        if self.points is None:
            raise RuntimeError('Oracle collection route has not been aligned to the live reset')
        position = senses['pos'][0].cpu().numpy().astype(float)
        velocity = senses['vel_world'][0].cpu().numpy().astype(float)
        rotation = quat_wxyz_to_mat(senses['quat'][0].cpu().numpy())
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        self.launch_heading=yaw if self.launch_heading is None else self.launch_heading
        dt = .01 if self.last_time is None else float(np.clip(now-self.last_time, 0., .1))
        self.last_time = now
        if self.launching:
            # A measured hover volume, rather than crossing an arbitrary height
            # plane, avoids deadlocking a stable controller with a small offset.
            stable=np.linalg.norm(position-self.points[0])<.6 and np.linalg.norm(velocity)<.6
            self.launch_stable_since=(now if self.launch_stable_since is None else self.launch_stable_since) if stable else None
            if stable and now-self.launch_stable_since>=1.:
                self.launching=False
                self.launch_completed_at=now
        if self.launching:
            target = self.points[0].copy()
        else:
            starts, edges = self.points[:-1], np.diff(self.points, axis=0)
            fractions = np.clip(np.sum((position-starts)*edges, axis=1)/self.lengths**2, 0., 1.)
            along = self.distance[:-1]+fractions*self.lengths
            candidates = (along >= self.progress-.2) & (along <= self.progress+5.)
            errors = np.linalg.norm(starts+fractions[:, None]*edges-position, axis=1)
            if candidates.any():
                index = np.argmin(np.where(candidates, errors, np.inf))
                self.progress = max(self.progress, along[index])
            ahead = min(self.distance[-1], self.progress+self.route.data['lookahead_m'])
            target = np.array([np.interp(ahead, self.distance, self.points[:, k]) for k in range(3)])
            endpoint = bool(self.progress >= self.distance[-1]-.8
                            and np.linalg.norm(position-self.points[-1]) < .8)
            stable=endpoint and (not np.isfinite(self.finish_speed)
                                 or float(senses['vel_world'].norm())<self.finish_speed)
            self.stable_since=(now if self.stable_since is None else self.stable_since) if stable else None
            self.complete=bool(stable and now-self.stable_since>=self.finish_hold)
        relative = target-position
        settling=bool(self.finish_hold and not self.launching and self.distance[-1]-self.progress<3.)
        if self.launching or settling:
            relative[:2]-=1.2*velocity[:2]
        else:
            # Same lateral drift damping as the training and visual race pilot.
            direction=relative[:2]/max(1e-9,np.linalg.norm(relative[:2]))
            relative[:2]-=.8*(velocity[:2]-direction*(velocity[:2]@direction))
        if np.linalg.norm(relative[:2]) > 3.:
            relative[:2] *= 3./np.linalg.norm(relative[:2])
        relative[2] = np.clip(relative[2], -1.2, 1.2)
        horizontal = target-position
        if self.launching:
            horizontal=np.array([np.cos(self.launch_heading),np.sin(self.launch_heading),0.])
        elif settling:
            horizontal=self.points[-1]-self.points[-2]
        angle = (np.arctan2(horizontal[1], horizontal[0])-yaw+np.pi) % (2*np.pi)-np.pi
        world_rate = float((rotation@np.asarray(omega))[2])
        desired_yaw = -np.clip(1.6*angle-.22*world_rate, -.8, .8)/2.3 if np.linalg.norm(horizontal[:2]) > .2 else 0.
        self.pilot.sight_yaw += float(np.clip(desired_yaw-self.pilot.sight_yaw, -2*dt, 2*dt))
        self.pilot.carrot = position+relative
        self.pilot.target = SimpleNamespace(t_last=now)
        modified=senses
        if self.motor_controller=='brain':
            # Match the deployed brain's nominal-speed sensory contract. PD
            # continues to receive actual, unscaled velocity from its caller.
            import torch
            velocity=senses['vel_world']*senses['vel_world'].new_tensor(
                [max(1.,self.reference_speed/self.speed)]*2+[1.])
            modified={**senses,'vel_world':velocity,
                'vel_body':velocity@torch.as_tensor(rotation,dtype=velocity.dtype,device=velocity.device)}
        return rotation.T@relative, modified

    def command(self, action):
        result = np.array(action, copy=True)
        result[3] = self.pilot.sight_yaw
        return result

    def metadata(self):
        return dict(mode='oracle-route', goal_source='PRIVILEGED stored route for collection/qualification',
                    guidance_contract='declared-route-stable-launch-v2',
                    launch_acceptance=dict(radius_m=.6,speed_mps=.6,stable_s=1.),
                    launch_complete=not self.launching,launch_heading_held=True,
                    launch_completed_at=self.launch_completed_at,
                    horizontal_hold_damping_s=1.2,lateral_motion_damping_s=.8,
                    runtime_route_oracle=True, visible_race_cues=False,
                    autonomous_evaluation_eligible=False, yaw_assistance=True,
                    speed_assistance=True, nominal_speed_mps=self.speed,
                    route_path=str(self.route.path),
                    route_sha256=self.route_sha256,
                    source_kind=self.route.data.get('source_kind', 'recorded route'),
                    progress_m=float(self.progress), route_endpoint_reached=self.complete,
                    finish_hold_s=self.finish_hold,finish_speed_mps=self.finish_speed if np.isfinite(self.finish_speed) else None,
                    motor_control=('Brain throttle/roll/pitch; privileged route and assisted yaw'
                                   if self.motor_controller=='brain' else
                                   'PD motors; privileged route guidance; brain in shadow'),
                    limitations='Does not establish causal visual navigation or autonomous race success')
