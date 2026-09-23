"""Distil the fast velocity-command PD into the connectome's motor readout.

Offline measured-drone surrogate only (`IdentifiedSim`, synthetic checkpoint
courses and the synthetic HUD marker from `liftoff.fast_rehearsal`). The fast
race-cue pilot supplies world velocity requests; the brain sees them as a
body-frame goal and its horizontal velocity senses in a declared speed-scaled
contract, so a fast flight looks like its familiar 3 m/s regime. Only the
throttle/roll/pitch readout rows and biases change: connectome wiring,
transmitter signs, encoders and recurrent weights are untouched and verified.
The PD teacher is never saved into or loaded by the exported brain.

Data are gathered DAgger-style: first under the teacher, then under the
brain's own control with teacher labels on the states it visits. Rehearsal
results are development checks, never Liftoff flight evidence.
"""
from __future__ import annotations

import argparse
import copy
from collections import deque
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .bptt import load_checkpoint
from .human_brain import export
from .motor_tracking import load_recorded_retina, motor_features, retina_sequence
from .thermal import wait_if_hot
from ..brain.gate_senses import gate_observation
from ..brain.motor_baseline import FastMotorPD
from ..brain.retina import RETINA_DIM
from ..liftoff.camera_pose import CameraPoseHistory
from ..liftoff.fast_race_cue import FastRaceCue
from ..liftoff.fast_rehearsal import hud_marker, synthetic_course
from ..sim.identified import IdentifiedSim
from ..sim.quad import quat_to_mat
from ..vision.camera import Camera
from ..vision.datasets import sha256


def fast_contract(speed, vertical_goal_seconds=1., scaled_speed=3.):
    """Declared sensory scaling: a nominal-speed request looks like `scaled_speed`.

    The brain's velocity senses saturate (tanh) above about 3 m/s, so a smaller
    scaled speed keeps requests between half and full speed distinguishable.
    """
    if not 0 < speed <= 20 or not 0 < scaled_speed <= 3:
        raise ValueError('Use a nominal speed in (0, 20] m/s and a scaled speed in (0, 3]')
    return dict(nominal_speed_mps=float(speed), goal_seconds=scaled_speed/speed, velocity_scale=scaled_speed/speed,
                scaled_speed_mps=float(scaled_speed),
                vertical_goal_seconds=float(vertical_goal_seconds),
                encoding='horizontal goal = request*goal_seconds, vertical goal = request*vertical_goal_seconds; '
                         'horizontal velocity senses scaled by velocity_scale, vertical unscaled')


def brain_observation(meta, senses, motor, task, retina, request, contract):
    """Exactly the runtime sensory path for brain motors under the fast pilot."""
    rotation = quat_to_mat(senses['quat'])
    scale = contract['velocity_scale']
    velocity = senses['vel_world']*senses['vel_world'].new_tensor([scale, scale, 1.])
    scaled = {**senses, 'vel_world': velocity, 'vel_body': torch.einsum('bji,bj->bi', rotation, velocity)}
    goal = request*request.new_tensor([contract['goal_seconds'], contract['goal_seconds'],
                                       contract['vertical_goal_seconds']])
    relative = torch.einsum('bji,bj->bi', rotation, goal)
    sensor = meta['gate_sensor']
    height = torch.zeros(len(request), 1, device=request.device) if sensor.get('search_height_anchor') else None
    return gate_observation(scaled, motor, task, retina, relative,
                            height_invariant=sensor.get('height_invariant', False),
                            gravity_aligned_height=sensor.get('gravity_aligned_height', False),
                            search_height_error=height, raw_retina_active=sensor.get('raw_retina_active', False))


@torch.no_grad()
def rollout(brain, cfg, meta, profile, contract, courses, *, controller='pd', speed=None, seconds=150.,
            seed=0, randomize=.1, collect=False, retina_stream=None, retina_dropout=.25, dropout=.1,
            camera_period=.055, camera_latency=.06, delay_steps=3, radius=3., quadratic_drag=.0075):
    """Batched closed loop: one synthetic course per drone; controller 'pd' or 'brain'."""
    speed = contract['nominal_speed_mps'] if speed is None else speed
    batch = len(courses)
    device = brain.device
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    calibration = meta['calibration']
    sim = IdentifiedSim(profile, calibration, 'cpu', cfg.brain.dt)
    sim.randomize(batch, randomize, generator)
    state = sim.hover(batch, 0.)
    state.quad.pos[:] = 0.
    state.quad.vel[:] = 0.
    sensor = meta['gate_sensor']
    camera = Camera(320, 180, sensor['focal_320'], sensor['tilt_deg'])
    yaw_axis = profile['axes']['yaw']
    curve = (yaw_axis['coefficient_deg_s'], yaw_axis['super_rate'], yaw_axis['expo'])
    histories = [CameraPoseHistory() for _ in range(batch)]
    pilots = [FastRaceCue(sensor, histories[i], speed, reference_speed=speed, yaw_curve=curve,
                          calibration=calibration) for i in range(batch)]
    teacher = FastMotorPD(profile, calibration)
    idle = torch.tensor([[-1., 0., 0., 0.]]).repeat(batch, 1)
    queue = deque(idle.clone() for _ in range(delay_steps))
    pending = [deque() for _ in range(batch)]
    detections, captures = [None]*batch, [None]*batch
    next_capture = np.zeros(batch)
    targets = np.zeros(batch, int)
    finish = np.full(batch, np.nan)
    crashed = np.zeros(batch, bool)
    brain_state = brain.init_state(batch)
    W = brain.inference_matrix() if device.type == 'cpu' else brain.weight_matrix().detach()
    steps = int(seconds/cfg.brain.dt)
    retinal = (retina_sequence(retina_stream, steps, batch, seed, retina_dropout)
               if retina_stream is not None else None)
    features, labels = [], []
    chatter, speeds, previous = [], [], None
    for k in range(steps):
        now = k*cfg.brain.dt
        positions = state.quad.pos.numpy().astype(float)
        quaternions = state.quad.quat.numpy().astype(float)
        active = ~crashed & np.isnan(finish)
        if not active.any():
            break
        for i in range(batch):
            histories[i].append(now, positions[i], quaternions[i])
            if active[i] and np.linalg.norm(courses[i][targets[i]]-positions[i]) < radius:
                targets[i] += 1
                if targets[i] == len(courses[i]):
                    finish[i] = now
                    continue
            if active[i] and now >= next_capture[i]:
                cue = hud_marker(camera, courses[i][targets[i]], positions[i], quaternions[i]) if rng.random() > dropout else None
                if cue is not None:
                    cue['aim_u'] = cue['u']
                pending[i].append((now, now+camera_latency, cue))
                next_capture[i] = now+camera_period
            while pending[i] and pending[i][0][1] <= now:
                captures[i], _, cue = pending[i].popleft()
                detections[i] = dict(race_cue=cue) if cue is not None else None
        senses = sim.sensors(state)
        requests, feedforward = [], []
        for i in range(batch):
            single = {key: value[i:i+1] for key, value in senses.items()}
            pilots[i].update(single, senses['gyro'][i].numpy(), detections[i], captures[i], now)
            requests.append(pilots[i].velocity_command)
            feedforward.append(pilots[i].feedforward)
        request = torch.tensor(np.asarray(requests), dtype=torch.float32)
        target_action = teacher.command(senses, request, torch.tensor(np.asarray(feedforward), dtype=torch.float32))
        motor = state.quad.motor.mean(-1, keepdim=True)
        retina = retinal[k] if retinal is not None else torch.zeros(batch, RETINA_DIM)
        obs = brain_observation(meta, {key: value.to(device) for key, value in senses.items()}, motor.to(device),
                                cfg.task, retina.to(device), request.to(device), contract)
        if k == 0:
            for _ in range(50):
                _, brain_state, _ = brain(obs, brain_state, W)
        action, brain_state, _ = brain(obs, brain_state, W)
        action = action.cpu()
        command = (target_action if controller == 'pd' else action).clone()
        for i in range(batch):
            command[i] = torch.as_tensor(pilots[i].command(command[i].numpy()), dtype=torch.float32)
        if now < 1.:
            command = idle.clone()
        if collect and k > 50 and k % 5 == 0 and now >= 1. and active.any():
            m = motor_features(brain, brain_state).cpu()
            mask = torch.as_tensor(active)
            features.append(m[mask])
            labels.append(target_action[mask, :3].clamp(-.97, .97).atanh())
        if previous is not None:
            chatter.append(float((command[:, 1:3]-previous[:, 1:3]).abs().mean()))
        previous = command
        queue.append(command)
        state = sim.step(state, queue.popleft())
        velocity = state.quad.vel
        state.quad.vel = velocity-quadratic_drag*velocity.norm(dim=-1, keepdim=True)*velocity*cfg.brain.dt
        speeds.append(float(velocity[torch.as_tensor(active)].norm(dim=-1).mean()) if active.any() else 0.)
        if now <= 1.5:
            state.quad.pos[:, 2] = state.quad.pos[:, 2].clamp_min(0.)
            state.quad.vel[:, 2] = state.quad.vel[:, 2].clamp_min(0.)
            state.quad.crashed[:] = False
        crashed |= state.quad.crashed.numpy() & np.isnan(finish)
        if k % 500 == 0 and device.type == 'cuda':
            wait_if_hot(68.)
    result = dict(controller=controller, speed=speed, seed=seed, drones=batch,
                  finished=int(np.isfinite(finish).sum()), crashed=int(crashed.sum()),
                  finish_s=[None if not np.isfinite(t) else round(float(t), 2) for t in finish],
                  gates=targets.tolist(), mean_speed=round(float(np.mean(speeds)), 2),
                  stick_chatter=round(float(np.mean(chatter)), 5))
    data = dict(features=torch.cat(features), labels=torch.cat(labels)) if collect and features else None
    return result, data


def fit_readout(brain, features, labels, ridge):
    """Parent-centred ridge on throttle/roll/pitch rows; nothing else may change."""
    x = torch.cat((features, torch.ones(len(features), 1)), -1).to(brain.readout.weight.device)
    y = labels.to(x.device)
    parent = torch.cat((brain.readout.weight[:3].detach(), brain.readout.bias[:3, None].detach()), -1)
    gram = x.T@x/len(x)
    delta = torch.linalg.solve(gram+ridge*torch.eye(x.shape[1], device=x.device),
                               x.T@(y-x@parent.T)/len(x)).T
    before = {n: p.detach().cpu().clone() for n, p in brain.named_parameters()}
    with torch.no_grad():
        brain.readout.weight[:3].add_(delta[:, :-1])
        brain.readout.bias[:3].add_(delta[:, -1])
    changed = {n: float((p.detach().cpu()-before[n]).abs().max()) for n, p in brain.named_parameters()
               if not torch.equal(p.detach().cpu(), before[n])}
    if set(changed)-{'readout.weight', 'readout.bias'}:
        raise RuntimeError('Readout distillation changed another brain parameter')
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--out', required=True)
    parser.add_argument('--profile', default='runs/measured-dynamics-low-speed-20260923/profile.json')
    parser.add_argument('--speed', type=float, default=8.)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--ridge', type=float, default=10.)
    parser.add_argument('--rounds', type=int, default=3, help='1 teacher round + brain-controlled DAgger rounds')
    parser.add_argument('--courses', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--retina-data', default='data/vision/observed_scene_v1/train_continuous.npz')
    parser.add_argument('--validation-retina-data', default='data/vision/observed_scene_v1/validation_continuous.npz')
    parser.add_argument('--retina-dropout', type=float, default=.25)
    parser.add_argument('--evaluation-seeds', type=int, nargs='+', default=[900, 901, 902, 903, 904, 905, 906, 907])
    parser.add_argument('--rest', type=float, default=5., help='seconds of rest between rollouts (thermal duty cycle)')
    parser.add_argument('--steep', type=float, default=0., help='probability of a 15-35 degree climbing/descending leg')
    parser.add_argument('--scaled-speed', type=float, default=3., help='apparent speed of a nominal request in the brain senses')
    args = parser.parse_args()
    if args.ridge <= 0 or args.rounds < 1:
        raise ValueError('Use positive ridge and at least one round')
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/'training-source.py')
    brain, cfg, _ = load_checkpoint(args.checkpoint, args.device)
    meta = copy.deepcopy(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['visual_brain'])
    profile = json.loads(Path(args.profile).read_text())
    contract = fast_contract(args.speed, scaled_speed=args.scaled_speed)
    config = dict(**vars(args), parent_sha256=sha256(args.checkpoint), source_sha256=sha256(__file__),
                  profile_sha256=sha256(args.profile), contract=contract,
                  retina_data_sha256=sha256(args.retina_data) if args.retina_data else None,
                  validation_retina_data_sha256=sha256(args.validation_retina_data) if args.validation_retina_data else None,
                  scope='offline measured-drone surrogate on synthetic courses; not Liftoff qualification')
    (out/'config.json').write_text(json.dumps(config, indent=2))
    training_retina = load_recorded_retina(args.retina_data, meta['gate_sensor'])
    evaluation_retina = load_recorded_retina(args.validation_retina_data, meta['gate_sensor'])
    evaluation_courses = [synthetic_course(s, steep=args.steep) for s in args.evaluation_seeds]
    log = open(out/'log.jsonl', 'w')

    def record(entry):
        entry = dict(time=round(time.time(), 1), **entry)
        log.write(json.dumps(entry)+'\n'); log.flush(); print(json.dumps(entry), flush=True)

    for controller in ('pd', 'brain'):
        row, _ = rollout(brain, cfg, meta, profile, contract, evaluation_courses, controller=controller,
                         seconds=args.seconds, seed=17, retina_stream=evaluation_retina, retina_dropout=args.retina_dropout)
        record(dict(stage='baseline', **row))
    rows, targets, history = [], [], []
    for round_index in range(args.rounds):
        controller = 'pd' if round_index == 0 else 'brain'
        seeds = [1000*round_index+s for s in range(args.courses)]
        time.sleep(args.rest)
        row, data = rollout(brain, cfg, meta, profile, contract, [synthetic_course(s, steep=args.steep) for s in seeds],
                            controller=controller, seconds=args.seconds, seed=100+round_index, collect=True,
                            retina_stream=training_retina, retina_dropout=args.retina_dropout)
        record(dict(stage=f'collect-{round_index}', **row, samples=len(data['labels']) if data else 0))
        if data is None:
            break
        rows.append(data['features']); targets.append(data['labels'])
        # Refit from the parent each round on all data gathered so far.
        fresh, _, _ = load_checkpoint(args.checkpoint, args.device)
        brain.load_state_dict(fresh.state_dict())
        changed = fit_readout(brain, torch.cat(rows), torch.cat(targets), args.ridge)
        time.sleep(args.rest)
        row, _ = rollout(brain, cfg, meta, profile, contract, evaluation_courses, controller='brain',
                         seconds=args.seconds, seed=17, retina_stream=evaluation_retina, retina_dropout=args.retina_dropout)
        record(dict(stage=f'evaluate-{round_index}', **row))
        history.append(dict(round=round_index, evaluation=row, changed=changed))
    torch.save(dict(features=torch.cat(rows), labels=torch.cat(targets), config=config), out/'training.pt')
    meta.pop('schema', None)
    meta.update(qualified=False, fast_motor_tracking=dict(
        **contract, teacher='FastMotorPD in the measured surrogate; offline only, never loaded at runtime',
        dynamics_profile_sha256=config['profile_sha256'], parent_sha256=config['parent_sha256'],
        source_sha256=config['source_sha256'], rounds=args.rounds, courses_per_round=args.courses,
        changed_parameters=history[-1]['changed'], runtime_requires_teacher=False,
        recorded_scene_currents=training_retina is not None, evaluation=history[-1]['evaluation']))
    export(out/'candidate.pt', brain, cfg, meta, 1)
    (out/'evaluation.json').write_text(json.dumps(history, indent=2))
    record(dict(stage='exported', candidate=str(out/'candidate.pt'), sha256=sha256(out/'candidate.pt')))


if __name__ == '__main__':
    main()
