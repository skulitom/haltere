"""Privileged route teacher for explicitly labelled collection/qualification only."""
import hashlib
from types import SimpleNamespace

import numpy as np

from .collection_route import CollectionRoute
from ..vision.camera import quat_wxyz_to_mat


class OracleCollectionAssistance:
    def __init__(self, path, *, reference_speed):
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
        rotation = quat_wxyz_to_mat(senses['quat'][0].cpu().numpy())
        dt = .01 if self.last_time is None else float(np.clip(now-self.last_time, 0., .1))
        self.last_time = now
        if self.launching and position[2] >= self.points[0, 2]-.15:
            self.launching = False
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
            self.complete = bool(self.progress >= self.distance[-1]-.8
                                 and np.linalg.norm(position-self.points[-1]) < .8)
        relative = target-position
        if np.linalg.norm(relative[:2]) > 3.:
            relative[:2] *= 3./np.linalg.norm(relative[:2])
        relative[2] = np.clip(relative[2], -1.2, 1.2)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        horizontal = target-position
        angle = (np.arctan2(horizontal[1], horizontal[0])-yaw+np.pi) % (2*np.pi)-np.pi
        world_rate = float((rotation@np.asarray(omega))[2])
        desired_yaw = -np.clip(1.6*angle-.22*world_rate, -.8, .8)/2.3 if np.linalg.norm(horizontal[:2]) > .2 else 0.
        self.pilot.sight_yaw += float(np.clip(desired_yaw-self.pilot.sight_yaw, -2*dt, 2*dt))
        self.pilot.carrot = position+relative
        self.pilot.target = SimpleNamespace(t_last=now)
        return rotation.T@relative, senses

    def command(self, action):
        result = np.array(action, copy=True)
        result[3] = self.pilot.sight_yaw
        return result

    def metadata(self):
        return dict(mode='oracle-route', goal_source='PRIVILEGED stored route for collection/qualification',
                    runtime_route_oracle=True, visible_race_cues=False,
                    autonomous_evaluation_eligible=False, yaw_assistance=True,
                    speed_assistance=True, nominal_speed_mps=self.speed,
                    route_path=str(self.route.path),
                    route_sha256=self.route_sha256,
                    source_kind=self.route.data.get('source_kind', 'recorded route'),
                    progress_m=float(self.progress), route_endpoint_reached=self.complete,
                    motor_control='PD motors; privileged route guidance; brain in shadow',
                    limitations='Does not establish causal visual navigation or autonomous race success')
