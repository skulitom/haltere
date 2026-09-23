"""Local velocity proposals from causal surface hypotheses and a task bearing.

This is an offline/shadow planner prototype. The acceleration-limited point-mass
rollout is a surrogate, not a proven model of either deployed motor controller.
Every proposal includes a reaction interval and a braking tail. Positive margins
mean only that sampled observed surfaces were avoided, never certified free space.
No map, route, gate order, future frame or learned motion predictor is consumed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .surface_memory import observed_path_margin
from .camera import quat_wxyz_to_mat


@dataclass(frozen=True)
class TrajectoryConfig:
    horizon_s: float = 2.
    reaction_s: float = .25
    dt: float = .05
    acceleration_mps2: float = 2.
    response_s: float = .6
    horizontal_speed_mps: float = 2.5
    vertical_speed_mps: float = 1.2
    vehicle_radius_m: float = .35
    extra_margin_m: float = .15
    max_surface_age_s: float = .5

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) <= 0 or self.dt > .1:
            raise ValueError('Use finite positive trajectory limits and dt <= 0.1 s')


def bounded_velocity(velocity, config):
    velocity = np.asarray(velocity, float).copy()
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError('Use a finite three-dimensional velocity')
    velocity[:2] *= min(1., config.horizontal_speed_mps / max(1e-9, np.linalg.norm(velocity[:2])))
    velocity[2] = np.clip(velocity[2], -config.vertical_speed_mps, config.vertical_speed_mps)
    return velocity


def rollout(position, velocity, desired_velocity, config=TrajectoryConfig()):
    """Include coasting latency, bounded tracking acceleration, and a full stop.

    The initial velocity is measured and is never clamped to the configured cap.
    Integration samples at <= 0.1 s; this is not continuous collision checking.
    """
    position, velocity = np.asarray(position, float).copy(), np.asarray(velocity, float).copy()
    if (position.shape != (3,) or velocity.shape != (3,)
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()):
        raise ValueError('Use finite three-dimensional position and velocity')
    desired = bounded_velocity(desired_velocity, config)
    points, speeds, times = [position.copy()], [velocity.copy()], [0.]
    for duration, target in [(config.reaction_s, None), (config.horizon_s, desired),
                             (max(np.linalg.norm(velocity), np.linalg.norm(desired)) /
                              config.acceleration_mps2 + 8 * config.response_s, np.zeros(3))]:
        elapsed = 0.
        while elapsed < duration - 1e-9:
            dt = min(config.dt, duration-elapsed)
            acceleration = np.zeros(3) if target is None else (target-velocity)/config.response_s
            acceleration *= min(1., config.acceleration_mps2/max(1e-9, np.linalg.norm(acceleration)))
            position += velocity*dt + .5*acceleration*dt*dt
            velocity += acceleration*dt
            elapsed += dt
            times.append(times[-1]+dt)
            points.append(position.copy())
            speeds.append(velocity.copy())
    return dict(positions=np.asarray(points), velocities=np.asarray(speeds), times=np.asarray(times))


def rollout_batch(position, velocity, desired_velocities, config=TrajectoryConfig()):
    """Evaluate alternatives together with the same integration as ``rollout``.

    Shorter braking tails retain their final sample while other candidates
    finish. Repeated endpoints do not add motion or change collision margins.
    """
    desired = np.asarray([bounded_velocity(v,config) for v in desired_velocities])
    position,velocity = np.asarray(position,float),np.asarray(velocity,float)
    if (not len(desired) or position.shape!=(3,) or velocity.shape!=(3,)
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()):
        raise ValueError('Use nonempty velocities and finite three-dimensional motion')
    positions = np.broadcast_to(position,(len(desired),3)).copy()
    velocities = np.broadcast_to(velocity,positions.shape).copy()
    times = np.zeros(len(desired))
    points,speeds,stamps = [positions.copy()],[velocities.copy()],[times.copy()]
    duration = np.maximum(np.linalg.norm(velocity),np.linalg.norm(desired,axis=1))/config.acceleration_mps2+8*config.response_s
    for duration,target in [(np.full(len(desired),config.reaction_s),None),
                             (np.full(len(desired),config.horizon_s),desired),
                             (duration,np.zeros_like(desired))]:
        elapsed = np.zeros(len(desired))
        while np.any(elapsed < duration-1e-9):
            dt = np.minimum(config.dt,np.maximum(0.,duration-elapsed))
            acceleration = np.zeros_like(desired) if target is None else (target-velocities)/config.response_s
            acceleration *= np.minimum(1.,config.acceleration_mps2 /
                                         np.maximum(1e-9,np.linalg.norm(acceleration,axis=1)))[:,None]
            positions += velocities*dt[:,None]+.5*acceleration*dt[:,None]**2
            velocities += acceleration*dt[:,None]
            elapsed += dt
            times += dt
            points.append(positions.copy());speeds.append(velocities.copy());stamps.append(times.copy())
    return dict(positions=np.stack(points,axis=1),velocities=np.stack(speeds,axis=1),times=np.stack(stamps,axis=1))


class LocalTrajectoryPlanner:
    def __init__(self, config=TrajectoryConfig()):
        self.config = config

    def propose(self, position, velocity, requested_velocity, surfaces, timestamp, *, view=None):
        config = self.config
        requested = bounded_velocity(requested_velocity, config)
        age = timestamp-surfaces['timestamp']
        if not np.isfinite(age) or age < -1e-6:
            raise ValueError('Surface observations must be finite and causal')
        if age > config.max_surface_age_s:
            path = rollout(position, velocity, np.zeros(3), config)
            return dict(velocity=np.zeros(3), path=path, status='stale_geometry_brake',
                        selected_margin_m=None, nominal_margin_m=None, candidates_checked=0,
                        changed=bool(np.linalg.norm(requested) > .01), coverage_certified=False)

        nominal = rollout(position, velocity, requested, config)
        margin = observed_path_margin(nominal['positions'], surfaces,
                                      vehicle_radius=config.vehicle_radius_m)['margin_m']
        if margin >= config.extra_margin_m:
            return dict(velocity=requested, path=nominal, status='nominal_unverified',
                        selected_margin_m=margin, nominal_margin_m=margin, candidates_checked=1,
                        changed=False, coverage_certified=False)

        # Generate alternatives in the task-bearing frame, not map coordinates.
        # Include braking and vertical/sideways motion for occlusions and pillars.
        heading = np.arctan2(requested[1], requested[0])
        horizontal = min(config.horizontal_speed_mps, max(.5, np.linalg.norm(requested[:2])))
        candidates = [np.zeros(3)]
        for fraction in (0., .5, 1.):
            for angle in (0., -np.pi/6, np.pi/6, -np.pi/3, np.pi/3, -np.pi/2, np.pi/2):
                for vertical in (requested[2], -config.vertical_speed_mps, config.vertical_speed_mps):
                    candidate = np.array([fraction*horizontal*np.cos(heading+angle),
                                          fraction*horizontal*np.sin(heading+angle), vertical])
                    if not np.any(np.all(np.isclose(candidate,np.asarray(candidates)),axis=1)):
                        candidates.append(candidate)
        best = None
        brake = None
        batch = rollout_batch(position,velocity,candidates,config)
        for index,candidate in enumerate(candidates):
            path = {name:values[index] for name,values in batch.items()}
            clearance = observed_path_margin(path['positions'], surfaces,
                                             vehicle_radius=config.vehicle_radius_m)['margin_m']
            if brake is None:
                brake = candidate, path, clearance
            if clearance < config.extra_margin_m:
                continue
            if np.linalg.norm(candidate) > .01 and not detour_in_view(path['positions'], view,
                                                                    config.vehicle_radius_m):
                continue
            # Retain task progress while preferring the smallest velocity change.
            # A clearance reward is capped: unknown surfaces cannot earn infinity.
            cost = np.sum((candidate-requested)**2) - .1*min(1., clearance)
            # Keep task-frame enumeration order for numerically equal costs;
            # a translated/rotated scene must not flip a symmetric detour.
            if best is None or cost < best[0]-1e-8:
                best = cost, candidate, path, clearance
        if best is None:
            candidate, path, clearance = brake
            status = 'no_observed_clear_path_brake'
        else:
            _, candidate, path, clearance = best
            status = 'observed_obstacle_detour' if np.linalg.norm(candidate) > .01 else 'observed_obstacle_brake'
        return dict(velocity=candidate, path=path, status=status, selected_margin_m=clearance,
                    nominal_margin_m=margin, candidates_checked=1+len(candidates), changed=True,
                    coverage_certified=False)


def detour_in_view(path, view, radius):
    """Reject new detours through unseen directions; visibility is not clearance.

    The currently occupied near region is omitted from this field-of-view test.
    All points still receive the observed-obstacle query. An absent view permits
    only braking alternatives. This does not certify empty space inside the view.
    """
    if view is None:
        return False
    camera, position, quaternion = view
    body = (path-np.asarray(position)) @ quat_wxyz_to_mat(quaternion)
    distance = np.linalg.norm(body, axis=1)
    body = body[distance > max(1.5, 3*radius)]
    if not len(body):
        return True
    pixels, front = camera.project_body(body)
    optical = (body @ camera.body_to_cam().T)[:,2]
    padding = camera.f*radius/np.maximum(optical, .01)
    return bool(np.all(front & (pixels[:,0] >= padding) & (pixels[:,0] <= camera.width-padding)
                             & (pixels[:,1] >= padding) & (pixels[:,1] <= camera.height-padding)))
