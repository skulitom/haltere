"""Offline closed-loop rehearsal of race-cue pilots on synthetic checkpoint courses.

Development tool only. The plant is the measured original-drone surrogate
(`IdentifiedSim`) with an optional quadratic drag term; checkpoints are
spheres on flat ground; the HUD marker is simulated by projecting the next
checkpoint through the calibrated camera and clamping it to the screen border
when it is out of view. Real terrain, obstacles, gate frames, detector errors
and the game's exact marker rules are absent, so rehearsal results never count
as flight evidence. It exists to find control bugs and compare pilot/motor
variants before spending real flights. No runtime controller imports it.
"""
from __future__ import annotations

import argparse
from collections import deque
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


def rehearse(course, pilot_factory, motor, calibration, profile, *, speed, seconds=240., radius=3.,
             camera_period=.055, camera_latency=.06, command_delay_steps=3, dropout=.1,
             quadratic_drag=.0075, seed=0, record=False):
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
    steps = int(seconds/.01)
    for k in range(steps):
        now = k*.01
        position = state.quad.pos[0].numpy().astype(float)
        quaternion = state.quad.quat[0].numpy().astype(float)
        history.append(now, position, quaternion)
        if target < len(course) and np.linalg.norm(course[target]-position) < radius:
            passes.append(now)
            target += 1
            if target == len(course):
                break
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
