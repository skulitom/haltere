"""Offline closed-loop rehearsal of race-cue pilots on synthetic checkpoint courses.

Development tool only. The plant is the measured original-drone surrogate
(`IdentifiedSim`) with an optional quadratic drag term; checkpoints are
spheres on flat ground; the HUD marker is simulated by projecting the next
checkpoint through the calibrated camera and clamping it to the screen border
when it is out of view. Real terrain, obstacles, gate frames, detector errors
and the game's exact marker rules are absent, so rehearsal results never count
as flight evidence. It exists to find control bugs and compare pilot/motor
variants before spending real flights. No runtime controller imports it.

Optionally (``terrain=CourseTerrain(...)``, off by default) a scoring-only
terrain model adds seeded hills under the descending legs of a course and
scores ground contacts, clearance, flight-path-below-view time and high
checkpoint passes (`DescentScore`). The pilot and the motor never see it; the
drone flies through it. Without it `rehearse` is unchanged.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..brain.motor_baseline import FastMotorPD, MotorPD, MotorPDConfig
from ..sim.identified import IdentifiedSim
from ..sim.quad import quat_to_mat
from ..vision.camera import Camera, quat_wxyz_to_mat
from .camera_pose import CameraPoseHistory

SCREEN = (1280, 720)
EDGE = (30, 27, 1257, 693)


def hud_marker(camera, point, position, quaternion):
    """Normalized marker position and edge flag, clamping off-screen targets."""
    rotation = quat_wxyz_to_mat(quaternion)
    body = (np.asarray(point)-position) @ rotation
    cam = camera.body_to_cam() @ body
    scale = SCREEN[0]/camera.width
    cx, cy = SCREEN[0]/2, SCREEN[1]/2
    if cam[2] > .05:
        u, v = cx+scale*camera.f*cam[0]/cam[2], cy+scale*camera.f*cam[1]/cam[2]
        if EDGE[0] <= u <= EDGE[2] and EDGE[1] <= v <= EDGE[3]:
            return dict(u=u/SCREEN[0], v=v/SCREEN[1], edge=False)
        dx, dy = u-cx, v-cy
    else:
        dx, dy = cam[0], cam[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            dy = 1.
    # Clamp the ray from the screen centre onto the marker border.
    tx = ((EDGE[2]-cx)/dx if dx > 0 else (EDGE[0]-cx)/dx) if abs(dx) > 1e-9 else np.inf
    ty = ((EDGE[3]-cy)/dy if dy > 0 else (EDGE[1]-cy)/dy) if abs(dy) > 1e-9 else np.inf
    t = min(tx, ty)
    u, v = cx+t*dx, cy+t*dy
    return dict(u=float(np.clip(u, 0, SCREEN[0]))/SCREEN[0], v=float(np.clip(v, 0, SCREEN[1]))/SCREEN[1], edge=True)


def synthetic_course(seed, gates=8, laps=1, steep=0.):
    """Seeded loop-free sequence of turns, climbs and drops; not a Liftoff course.

    `steep` is the probability that a leg climbs or descends along a slope of
    15-35 degrees, as racing lines over hills do; the default keeps the
    original gentle height changes.
    """
    rng = np.random.default_rng(seed)
    points, heading, position = [], 0., np.array([0., 0., 0.])
    for index in range(gates):
        heading += rng.uniform(-np.radians(150), np.radians(150)) if index else rng.uniform(-.3, .3)
        leg = rng.uniform(15., 60.)
        if index and steep and rng.random() < steep:
            slope = np.radians(rng.uniform(15., 35.))*rng.choice([-1., 1.])
            height = float(np.clip(position[2]+leg*np.tan(slope), 1.5, 45.))
        else:
            height = float(np.clip(position[2]+rng.normal(0, 4.), 1.5, 25.)) if index else rng.uniform(1.5, 4.)
        position = position+np.array([np.cos(heading)*leg, np.sin(heading)*leg, 0.])
        position[2] = height
        points.append(position.copy())
    return np.asarray(points*laps)


def hill_course(seed, gates=8):
    """Seeded course of climbs to a crest followed by descents down a hill (1-3 legs, 6-30 degrees), as racing
    lines over hills run (e.g. a downhill with several rings); not a Liftoff course. For the terrain model."""
    rng = np.random.default_rng([int(seed), 104729])
    points, heading, position = [], float(rng.uniform(-.3, .3)), np.zeros(3)
    while len(points) < gates:
        for climb in (True, False):
            for _ in range(int(rng.integers(1, 3 if climb else 4))):
                if len(points) >= gates:
                    break
                if points:
                    heading += float(rng.uniform(-1, 1))*np.radians(90. if climb else 40.)
                leg = float(rng.uniform(20., 45.) if climb else rng.uniform(20., 50.))
                slope = np.radians(rng.uniform(5., 25.) if climb else rng.uniform(6., 30.))
                height = position[2]+(1 if climb else -1)*leg*np.tan(slope)
                position = position+np.array([np.cos(heading)*leg, np.sin(heading)*leg, 0.])
                position[2] = float(np.clip(height, 1.5, 45.))
                points.append(position.copy())
    return np.asarray(points)


@dataclass(frozen=True)
class TerrainConfig:
    """Scoring-only hills under descending legs (see `CourseTerrain`).

    A leg whose checkpoint lies at least min_drop_m below the previous one gets a hill: the ground lies a seeded
    clearance (clearance_m range) below each of its two checkpoints and follows a smoothstep between them, with a
    flat crest over a seeded crest fraction of the leg (a convex crest after the upper checkpoint) and a flat toe
    over a seeded toe fraction before the lower one; it depends only on the distance along the leg (a broad
    hillside), stays flat at the crest height behind the upper checkpoint and at the toe height beyond the lower
    one. The hill exists while the drone flies that leg (its target is the lower checkpoint) and for after_pass_s
    after it passes the lower checkpoint. Contacts are separated by separation_s above ground. The pilot and the
    motor never see it; the drone flies through it (scoring only)."""
    min_drop_m: float = 2.
    clearance_min_m: float = 1.2
    clearance_max_m: float = 2.5
    crest_max: float = .3
    toe_max: float = .15
    after_pass_s: float = 1.
    separation_s: float = .3

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Use finite non-negative terrain parameters')
        if not 0 < self.clearance_min_m <= self.clearance_max_m or not self.crest_max+self.toe_max < 1:
            raise ValueError('Use 0 < clearance_min_m <= clearance_max_m and crest_max + toe_max < 1')


class CourseTerrain:
    """Seeded scoring-only hills under the descending legs of one course (`TerrainConfig`).

    Leg k runs from checkpoint k-1 (the start point for k = 0, which never gets a hill) to checkpoint k."""

    def __init__(self, course, seed, config=None, start=(0., 0., 0.)):
        self.config = c = config or TerrainConfig()
        course = np.asarray(course, float)
        rng = np.random.default_rng([int(seed), 7919])
        clearance = rng.uniform(c.clearance_min_m, c.clearance_max_m, len(course))
        self.legs = {}
        for k in range(1, len(course)):
            a, b = course[k-1], course[k]
            length = float(np.linalg.norm((b-a)[:2]))
            crest, toe = float(rng.uniform(0., c.crest_max)), float(rng.uniform(0., c.toe_max))
            top, bottom = float(a[2]-clearance[k-1]), float(b[2]-clearance[k])
            if a[2]-b[2] < c.min_drop_m or length < 1. or top <= bottom:
                continue
            self.legs[k] = dict(a=a[:2].copy(), e=(b-a)[:2]/length, length=length, top=top, bottom=bottom,
                                crest=crest, toe=toe, drop=float(a[2]-b[2]),
                                slope_deg=float(np.degrees(np.arctan2(a[2]-b[2], length))))

    def leg_ground(self, k, xy):
        """(ground height, along-leg slope dh/ds) of leg k's hill at horizontal position xy, or None."""
        leg = self.legs.get(k)
        if leg is None:
            return None
        u = float((np.asarray(xy, float)[:2]-leg['a']) @ leg['e'])/leg['length']
        span = 1.-leg['crest']-leg['toe']
        w = float(np.clip((u-leg['crest'])/span, 0., 1.))
        drop = leg['top']-leg['bottom']
        slope = -drop*6*w*(1-w)/(span*leg['length']) if 0 < w < 1 else 0.
        return leg['top']-drop*(3*w*w-2*w*w*w), slope

    def ground(self, target, last_pass, now, xy):
        """(height, dh/ds along its leg, leg) of the highest active hill (the leg being flown and, for after_pass_s
        after its pass, the previous one), or None."""
        best = None
        for k in (target, target-1):
            if k != target and (last_pass is None or now-last_pass > self.config.after_pass_s):
                continue
            g = self.leg_ground(k, xy)
            if g is not None and (best is None or g[0] > best[0]):
                best = (g[0], g[1], k)
        return best


# Descending flight (for the view metric): vertical speed below this, speed above the next.
VIEW_SINK = .5
VIEW_SPEED = 1.5
HIGH_PASS_M = 1.5
# Hills no steeper than this (the leg's mean slope, degrees) can be descended with the flight path inside the image of
# a level camera tilted 30 degrees up with a 42 degree vertical half field of view (its lower edge lies 12 degrees down).
VIEWABLE_SLOPE_DEG = 12.


def path_below_view(camera, velocity, quaternion):
    """True when the velocity vector points below the camera's lower image edge (or behind the camera, downward)."""
    body = np.asarray(velocity, float) @ quat_wxyz_to_mat(quaternion)
    cam = camera.body_to_cam() @ body
    if cam[2] <= .05*np.linalg.norm(cam):
        return bool(cam[1] > 0)
    return bool(cam[1]/cam[2] > camera.cy/camera.f)


class DescentScore:
    """Per-course descent metrics: terrain contacts (with a `CourseTerrain`), minimum clearance, time with the
    velocity vector below the camera's lower image edge while descending (vz < -VIEW_SINK, speed > VIEW_SPEED),
    and checkpoint passes more than HIGH_PASS_M above the checkpoint centre."""

    def __init__(self, course, camera, terrain=None):
        self.course, self.camera, self.terrain = np.asarray(course, float), camera, terrain
        self.contacts = 0
        self.contact_s = self.hill_s = self.descent_s = self.below_view_s = self.hill_descent_s = 0.
        self.hill_below_view_s = 0.
        self.viewable_descent_s = self.viewable_below_view_s = 0.
        self.min_clearance = np.inf
        self.max_penetration = 0.
        self.max_into_speed = 0.
        self.contact_legs = []
        self.last_contact = self.last_pass = None
        self.pass_heights = []

    def passed(self, target, now, position):
        """Call when the checkpoint `target` has just been passed at `position`."""
        self.pass_heights.append(float(position[2]-self.course[target][2]))
        self.last_pass = now

    def step(self, now, dt, target, position, velocity, quaternion):
        velocity = np.asarray(velocity, float)
        descending = velocity[2] < -VIEW_SINK and float(np.linalg.norm(velocity)) > VIEW_SPEED
        below = descending and path_below_view(self.camera, velocity, quaternion)
        self.descent_s += dt*descending
        self.below_view_s += dt*below
        ground = None if self.terrain is None else self.terrain.ground(target, self.last_pass, now, position)
        if ground is None:
            return
        height, slope, leg = ground
        self.hill_s += dt
        self.hill_descent_s += dt*descending
        self.hill_below_view_s += dt*below
        if self.terrain.legs[leg]['slope_deg'] <= VIEWABLE_SLOPE_DEG:
            self.viewable_descent_s += dt*descending
            self.viewable_below_view_s += dt*below
        clearance = float(position[2]-height)
        self.min_clearance = min(self.min_clearance, clearance)
        if clearance >= 0:
            return
        self.contact_s += dt
        self.max_penetration = max(self.max_penetration, -clearance)
        if self.last_contact is None or now-self.last_contact > self.terrain.config.separation_s+dt/2:
            self.contacts += 1
            self.contact_legs.append(int(leg))
            e = self.terrain.legs[leg]['e']
            normal = np.r_[-slope*e, 1.]/np.sqrt(1+slope*slope)
            self.max_into_speed = max(self.max_into_speed, float(-(velocity @ normal)))
        self.last_contact = now

    def result(self):
        heights = np.asarray(self.pass_heights)
        out = dict(descent_s=round(self.descent_s, 2), below_view_s=round(self.below_view_s, 2),
                   high_passes=int((heights > HIGH_PASS_M).sum()),
                   max_pass_height_m=None if not len(heights) else round(float(heights.max()), 2),
                   pass_heights_m=[round(float(h), 2) for h in heights])
        if self.terrain is not None:
            out.update(hill_legs=len(self.terrain.legs), hill_s=round(self.hill_s, 2),
                       hill_descent_s=round(self.hill_descent_s, 2),
                       hill_below_view_s=round(self.hill_below_view_s, 2),
                       viewable_hill_descent_s=round(self.viewable_descent_s, 2),
                       viewable_hill_below_view_s=round(self.viewable_below_view_s, 2), contacts=self.contacts,
                       contact_s=round(self.contact_s, 2), contact_legs=self.contact_legs,
                       max_penetration_m=round(self.max_penetration, 2),
                       max_into_speed_mps=round(self.max_into_speed, 2),
                       min_clearance_m=None if not np.isfinite(self.min_clearance) else round(self.min_clearance, 2))
        return out


def rehearse(course, pilot_factory, motor, calibration, profile, *, speed, seconds=240., radius=3.,
             camera_period=.055, camera_latency=.06, command_delay_steps=3, dropout=.1,
             quadratic_drag=.0075, seed=0, record=False, terrain=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sim = IdentifiedSim(profile, calibration, 'cpu', .01)
    state = sim.hover(1, 0.)
    state.quad.pos[:] = 0.
    state.quad.vel[:] = 0.
    sensor = dict(focal_320=100., tilt_deg=30.)
    camera = Camera(320, 180, sensor['focal_320'], sensor['tilt_deg'])
    history = CameraPoseHistory()
    pilot = pilot_factory(sensor, history)
    queue = deque(torch.tensor([[-1., 0., 0., 0.]]) for _ in range(command_delay_steps))
    pending, next_capture, target = deque(), 0., 0
    detection, capture_time = None, None
    passes, rows, crashed, previous = [], [], False, None
    chatter, tilt, speeds = [], [], []
    score = None if terrain is None else DescentScore(course, camera, terrain)
    steps = int(seconds/.01)
    for k in range(steps):
        now = k*.01
        position = state.quad.pos[0].numpy().astype(float)
        quaternion = state.quad.quat[0].numpy().astype(float)
        history.append(now, position, quaternion)
        if target < len(course) and np.linalg.norm(course[target]-position) < radius:
            passes.append(now)
            if score is not None:
                score.passed(target, now, position)
            target += 1
            if target == len(course):
                break
        if score is not None:
            score.step(now, .01, target, position, state.quad.vel[0].numpy().astype(float), quaternion)
        if now >= next_capture:
            cue = hud_marker(camera, course[target], position, quaternion) if rng.random() > dropout else None
            if cue is not None:
                cue['aim_u'] = cue['u']
            pending.append((now, now+camera_latency, cue))
            next_capture = now+camera_period
        while pending and pending[0][1] <= now:
            capture_time, _, cue = pending.popleft()
            detection = dict(race_cue=cue) if cue is not None else None
        sensors = sim.sensors(state)
        senses = {k: v for k, v in sensors.items()}
        omega = sensors['gyro'][0].numpy()
        relative, _ = pilot.update(senses, omega, detection, capture_time, now)
        if isinstance(motor, FastMotorPD):
            action = motor.command(sensors, torch.tensor(pilot.velocity_command, dtype=torch.float32)[None],
                                   torch.tensor(pilot.feedforward, dtype=torch.float32)[None])
        else:
            action = motor.command(sensors, torch.tensor(relative, dtype=torch.float32)[None],
                                   speed=min(speed, getattr(pilot, 'reference_speed', speed)))
        command = torch.tensor(pilot.command(action[0].numpy()), dtype=torch.float32)[None]
        if now < 1.:
            command = torch.tensor([[-1., 0., 0., 0.]])
        if previous is not None:
            chatter.append(float((command[0, 1:3]-previous[0, 1:3]).abs().mean()))
        previous = command
        queue.append(command)
        state = sim.step(state, queue.popleft())
        velocity = state.quad.vel[0]
        speed_now = float(velocity.norm())
        drag = quadratic_drag*speed_now*velocity
        state.quad.vel[0] = velocity-drag*.01
        rotation = quat_to_mat(state.quad.quat)[0]
        tilt.append(float(torch.rad2deg(torch.arccos(rotation[2, 2].clamp(-1, 1)))))
        speeds.append(speed_now)
        if now > 1.5 and bool(state.quad.crashed[0]):
            crashed = True
            break
        if now <= 1.5:
            state.quad.pos[0, 2] = state.quad.pos[0, 2].clamp_min(0.)
            state.quad.vel[0, 2] = state.quad.vel[0, 2].clamp_min(0.)
            state.quad.crashed[:] = False
        if record:
            requested = getattr(pilot, 'velocity_command', None)
            rows.append(dict(t=now, pos=position.tolist(), state=getattr(pilot, 'state', ''),
                             cmd=[] if requested is None else np.asarray(requested).tolist()))
    finished = target == len(course)
    result = dict(finished=finished, crashed=crashed, gates=target, of=len(course),
                  time_s=round(passes[-1], 2) if finished else None, elapsed_s=round(now, 2),
                  mean_speed=round(float(np.mean(speeds)), 2), p90_speed=round(float(np.quantile(speeds, .9)), 2),
                  tilt_p95=round(float(np.quantile(tilt, .95)), 1), stick_chatter=round(float(np.mean(chatter)), 4),
                  states={k: round(v, 1) for k, v in getattr(pilot, 'state_time', {}).items()})
    if score is not None:
        result['terrain'] = score.result()
    if record:
        result['rows'] = rows
    return result


def main():
    from .fast_race_cue import FastRaceCue
    from .race_cue_assistance import RaceCueAssistance
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/motor-brain-10-tracking-05/candidate.pt')
    parser.add_argument('--profile', default='runs/measured-dynamics-low-speed-20260923/profile.json')
    parser.add_argument('--speeds', type=float, nargs='+', default=[4., 6., 8., 12.])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--baseline', action='store_true', help='also rehearse the standard race-cue pilot with teacher PD')
    parser.add_argument('--out', default='')
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['visual_brain']
    calibration = checkpoint['calibration']
    profile = json.loads(Path(args.profile).read_text())
    results = []
    begin = time.time()
    for seed in args.seeds:
        course = synthetic_course(seed)
        if args.baseline:
            from types import SimpleNamespace
            from ..liftoff.fit_vertical import equivalent_power_curve
            from ..sim.vehicle import RatesConfig
            fit = checkpoint['gate_training']['dynamics']['profile']['vertical_calibration']['mean']
            curve = equivalent_power_curve(fit, calibration, idle=.04)
            rates = RatesConfig(rc_rate=(1.55, 1.55, 1.0), super_rate=(.73, .73, .73), expo=(.3, .3, .3))
            motor = MotorPD(SimpleNamespace(**curve), rates, .04, MotorPDConfig(position_gain=max(.8, 2.5/3)))
            result = rehearse(course, lambda s, h: RaceCueAssistance(s, h, 2.5, reference_speed=3.), motor,
                              calibration, profile, speed=2.5, seed=seed)
            results.append(dict(seed=seed, variant='standard-2.5', **result))
            print(json.dumps(results[-1]), flush=True)
        for speed in args.speeds:
            motor = FastMotorPD(profile, calibration)
            yaw = profile['axes']['yaw']
            curve = (yaw['coefficient_deg_s'], yaw['super_rate'], yaw['expo'])
            result = rehearse(course, lambda s, h, v=speed: FastRaceCue(s, h, v, reference_speed=v, yaw_curve=curve,
                                                                        calibration=calibration), motor,
                              calibration, profile, speed=speed, seed=seed)
            results.append(dict(seed=seed, variant=f'fast-{speed:g}', **result))
            print(json.dumps(results[-1]), flush=True)
    print('elapsed', round(time.time()-begin, 1), 's', flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
