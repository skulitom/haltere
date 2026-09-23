"""Motor-only simulation diagnostic and readout distillation.

Ideal local targets are an offline motor test, NOT a visual-navigation result.
The exported model retains the complete parent graph and scene encoder. Only
motor readout weights/bias may change; a PD teacher is never loaded by the brain.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .bptt import load_checkpoint, make_world
from .human_brain import export
from .thermal import wait_if_hot
from ..brain.gate_senses import gate_observation
from ..brain.motor_baseline import MotorPD, MotorPDConfig
from ..brain.retina import RETINA_DIM
from ..liftoff.fit_vertical import equivalent_power_curve
from ..sim.quad import QuadState, quat_from_euler, quat_to_mat
from ..vision.datasets import sha256


def measured_dynamics(cfg, meta):
    cfg = copy.deepcopy(cfg)
    measured = meta['gate_training']['dynamics']['profile']['vertical_calibration']['mean']
    curve = equivalent_power_curve(measured, meta['calibration'], idle=cfg.ctl.idle)
    for name, value in curve.items():
        setattr(cfg.quad, name, value)
    cfg.quad.gyro_noise = 0.
    return cfg


def local_guidance(sensors, heading, altitude, hold, braking):
    """Ideal-target counterpart of race-cue lead/braking; simulation only."""
    R = quat_to_mat(sensors['quat'])
    velocity, position = sensors['vel_world'], sensors['pos']
    direction = torch.stack((heading.cos(), heading.sin()), -1)
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    angle = torch.atan2((heading-yaw).sin(), (heading-yaw).cos())
    moving = (angle.abs() < torch.deg2rad(angle.new_tensor(55.))) & ~braking
    lead = 3.*angle.cos().square().clamp_min(.25)
    relative_xy = direction*lead[:, None]-.8*(velocity[:, :2]-direction*(velocity[:, :2]*direction).sum(-1, keepdim=True))
    stopped = hold[:, :2]-position[:, :2]-1.2*velocity[:, :2]
    relative_xy = torch.where(moving[:, None], relative_xy, stopped)
    relative_xy = relative_xy * (3./relative_xy.norm(dim=-1, keepdim=True).clamp_min(3.))
    relative_z = (altitude-position[:, 2]).clamp(-1.2, 1.2)
    relative_world = torch.cat((relative_xy, relative_z[:, None]), -1)
    relative_body = torch.einsum('bji,bj->bi', R, relative_world)
    world_omega = torch.einsum('bij,bj->bi', R, sensors['gyro'])
    yaw_stick = -(1.6*angle-.22*world_omega[:, 2]).clamp(-.8, .8)/2.3
    return relative_body, yaw_stick, moving


def motor_features(brain, state):
    rates = brain.cfg.rate_max*torch.sigmoid(state['v'][brain.motor_idx]).T
    bn = brain.motor_norm
    if bn is not None:
        return (rates-bn.running_mean)/torch.sqrt(bn.running_var+bn.eps)
    if brain.cfg.readout_norm == 'layer':
        return torch.nn.functional.layer_norm(rates, (rates.shape[1],))
    return rates


def retina_sequence(stream, steps, batch, seed, dropout=0.):
    """Recorded sensory perturbations, independent of the physics random seed.

    Images are not aligned to simulated poses; this tests robustness to scene
    currents and intermittent missing images, never camera navigation.
    """
    if stream.ndim != 2 or stream.shape[1] != RETINA_DIM or not len(stream):
        raise ValueError('Expected a nonempty [time, retina] stream')
    if not torch.isfinite(stream).all() or not 0 <= dropout < 1:
        raise ValueError('Invalid retinal stream or dropout')
    generator = torch.Generator().manual_seed(seed+20000)
    starts = torch.randint(len(stream), (batch,), generator=generator)
    indices = (torch.arange(steps)[:, None]+starts[None]) % len(stream)
    sequence = stream.cpu()[indices].clone()
    # Missing-image intervals last 100 ms, resembling sample-and-hold capture.
    missing = torch.rand((steps+9)//10, batch, generator=generator) < dropout
    sequence[missing.repeat_interleave(10, dim=0)[:steps]] = 0.
    return sequence


def load_recorded_retina(path, sensor):
    if not path:
        return None
    path = Path(path)
    manifest = json.loads((path.parent/'manifest.json').read_text())
    recorded = manifest.get('gate_sensor', {})
    for key in ('retina_mode', 'sha256', 'scene_projection', 'focal_320', 'tilt_deg'):
        if key not in sensor or recorded.get(key) != sensor[key]:
            raise ValueError('Recorded retina differs from the checkpoint sensory contract: '+key)
    with np.load(path) as data:
        return torch.from_numpy(data['retina'].copy()).float()


@torch.no_grad()
def rollout(brain, cfg, meta, *, controller='brain', speed=2., seed=8291,
            seconds=16., batch=12, collect=False, randomize=.1, observation_reference_speed=None,
            retina_stream=None, retina_dropout=0.):
    torch.manual_seed(seed)
    cfg = measured_dynamics(cfg, meta)
    cfg.train.randomize = randomize
    cfg.train.randomize_ctl = randomize
    vehicle, _ = make_world(cfg, batch, brain.device)
    pd = MotorPD(cfg.quad, cfg.rates, cfg.ctl.idle,
                 MotorPDConfig(position_gain=max(.8, speed/3.)))
    q = QuadState.hover(batch, brain.device, 6.)
    heading0 = torch.rand(batch, device=brain.device)*2*torch.pi-torch.pi
    q.quat = quat_from_euler(heading0*0, heading0*0, heading0)
    q.motor[:] = vehicle.sim.hover_command()
    q.vel[:, :2] = torch.stack((heading0.cos(), heading0.sin()), -1)*torch.linspace(0., speed, batch, device=brain.device)[:, None]
    q.pos[:, 2] = torch.linspace(4., 24., batch, device=brain.device)
    altitude0 = q.pos[:, 2].clone()
    vs, state, W = vehicle.wrap(q), brain.init_state(batch), brain.inference_matrix()
    delay = deque(pd.command(vehicle.sim.sensors(q), torch.zeros(batch, 3, device=brain.device), speed)
                  for _ in range(cfg.train.delay_steps))
    turn = torch.linspace(-1.5, 1.5, batch, device=brain.device)
    hold = q.pos.clone()
    old_phase = -1
    features, labels, actions = [], [], []
    errors, speeds, heights, tilts = [], [], [], []
    crashed = torch.zeros(batch, device=brain.device, dtype=torch.bool)
    first_crash = torch.full((batch,), -1., device=brain.device)
    steps = round(seconds/cfg.brain.dt)
    retinal = retina_sequence(retina_stream, steps, batch, seed, retina_dropout).to(brain.device) if retina_stream is not None else None
    for step in range(steps):
        if step % 250 == 0 and brain.device.type == 'cuda':
            wait_if_hot(68.)
        phase = min(3, int(step*cfg.brain.dt/4))
        if phase != old_phase:
            hold = vs.quad.pos.clone()
            old_phase = phase
        heading = heading0+(turn if phase == 1 else -turn if phase == 2 else 0.)
        braking = torch.full((batch,), phase == 3, device=brain.device, dtype=torch.bool)
        altitude = altitude0+(torch.linspace(-2., 2., batch, device=brain.device) if phase in (1, 2) else 0.)
        senses = vehicle.sim.sensors(vs.quad)
        relative, yaw, moving = local_guidance(senses, heading, altitude, hold, braking)
        reference = observation_reference_speed or meta.get('motor_tracking', {}).get('nominal_speed_mps', 2.)
        gain = max(1., reference/speed)
        R_sense = quat_to_mat(senses['quat'])
        velocity = senses['vel_world']*senses['vel_world'].new_tensor([gain, gain, 1.])
        modified = {**senses, 'vel_world': velocity,
                    'vel_body': torch.einsum('bji,bj->bi', R_sense, velocity)}
        obs = gate_observation(modified, vs.quad.motor.mean(-1, keepdim=True), cfg.task,
                               retinal[step] if retinal is not None else torch.zeros(batch, RETINA_DIM, device=brain.device), relative,
                               height_invariant=True, gravity_aligned_height=True,
                               search_height_error=torch.zeros(batch, 1, device=brain.device), raw_retina_active=True)
        if step == 0:
            for _ in range(50):
                _, state, _ = brain(obs, state, W)
        action, state, aux = brain(obs, state, W)
        teacher = pd.command(senses, relative, speed)
        command = (teacher if controller == 'pd' else action).clone()
        command[:, 3] = yaw
        if collect and step > 50 and step % 5 == 0:
            m = motor_features(brain, state)
            if m.shape != (batch, brain.readout.in_features):
                raise ValueError('Motor features do not match per-drone neuron activity')
            valid = ~crashed
            features.append(m[valid].cpu())
            labels.append(teacher[valid, :3].clamp(-.97, .97).atanh().cpu())
            actions.append(action[valid, :3].cpu())
        delay.append(command)
        vs = vehicle.step(vs, delay.popleft())
        first_crash = torch.where(vs.quad.crashed & ~crashed, (step+1)*cfg.brain.dt, first_crash)
        crashed |= vs.quad.crashed
        R = quat_to_mat(vs.quad.quat)
        desired_speed = torch.where(moving, speed*(relative[:, :2].norm(dim=-1)/3).clamp(max=1), 0.)
        desired = torch.stack((heading.cos(), heading.sin()), -1)*desired_speed[:, None]
        errors.append((vs.quad.vel[:, :2]-desired).norm(dim=-1).cpu())
        speeds.append(vs.quad.vel[:, :2].norm(dim=-1).cpu())
        heights.append((vs.quad.pos[:, 2]-altitude).abs().cpu())
        tilts.append(torch.rad2deg(R[:, 2, 2].clamp(-1., 1.).acos()).cpu())
    error, speed_values, height, tilt = map(torch.stack, (errors, speeds, heights, tilts))
    result = dict(controller=controller, nominal_speed_mps=speed, seed=seed, batch=batch,
                  recorded_scene_currents=retinal is not None, retina_dropout=retina_dropout if retinal is not None else 0.,
                  seconds=seconds, crashed=int(crashed.sum()),
                  first_crash_s=first_crash.cpu().tolist(),
                  velocity_error_mean=float(error.mean()), height_error_mean=float(height.mean()),
                  speed_median=float(speed_values.median()), speed_p90=float(speed_values.quantile(.9)),
                  brake_final_speed=float(speed_values[-50:].mean()) if seconds >= 16 else None,
                  tilt_p95_deg=float(tilt.quantile(.95)),
                  phases=[dict(name=name, velocity_error_mean=float(error[i*400:(i+1)*400].mean()),
                               speed_mean=float(speed_values[i*400:(i+1)*400].mean()))
                          for i, name in enumerate(('straight', 'turn', 'reverse_turn', 'brake'))
                          if i*400 < len(error)])
    data = dict(features=torch.cat(features), labels=torch.cat(labels), parent_actions=torch.cat(actions)) if collect else None
    return result, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--out', required=True)
    parser.add_argument('--speed', type=float, default=3.)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--ridge', type=float, default=10.)
    parser.add_argument('--evaluation-seed', type=int, default=8291)
    parser.add_argument('--data-controller', choices=['pd','brain','mixed'], default='pd',
                        help='Brain collects corrections on its visited states; mixed uses PD for seed 1921, brain for 1922/1923')
    parser.add_argument('--reuse-data', default='', help='Retain teacher features with an identical frozen neural representation')
    parser.add_argument('--retina-data', default='', help='Training-only recorded [time,720] retina array in an NPZ; no pose alignment')
    parser.add_argument('--validation-retina-data', default='', help='Separate recorded visual input used only for evaluation')
    parser.add_argument('--retina-dropout', type=float, default=.25, help='Fraction of 100 ms image-missing intervals with recorded input')
    parser.add_argument('--training-speeds', nargs='+', type=float, default=None)
    args = parser.parse_args()
    if not 0 < args.speed <= 10 or args.ridge <= 0:
        raise ValueError('Use 0 < speed <= 10 and positive ridge regularization')
    training_speeds = sorted(set(args.training_speeds or [1.5, 2., args.speed]))
    if any(not 0 < s <= args.speed for s in training_speeds) or not 0 <= args.retina_dropout < 1:
        raise ValueError('Training speeds must be positive and at most the reference; dropout must be in [0,1)')
    torch.set_num_threads(2)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/'training-source.py')
    brain, cfg, _ = load_checkpoint(args.checkpoint, args.device)
    meta = copy.deepcopy(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['visual_brain'])
    config = dict(**vars(args), parent_sha256=sha256(args.checkpoint), source_sha256=sha256(__file__),
                  retina_data_sha256=sha256(args.retina_data) if args.retina_data else None,
                  validation_retina_data_sha256=sha256(args.validation_retina_data) if args.validation_retina_data else None,
                  scope='ideal-target motor simulation, not camera navigation or Liftoff qualification')
    (out/'config.json').write_text(json.dumps(config, indent=2))
    training_retina = load_recorded_retina(args.retina_data, meta['gate_sensor'])
    evaluation_retina = load_recorded_retina(args.validation_retina_data, meta['gate_sensor'])
    results = []
    for motor in ('brain', 'pd'):
        row, _ = rollout(brain, cfg, meta, controller=motor, speed=args.speed, seed=args.evaluation_seed,
                         retina_stream=evaluation_retina, retina_dropout=args.retina_dropout)
        results.append(row); print(json.dumps(row), flush=True)
    (out/'baseline.json').write_text(json.dumps(results, indent=2))
    if not args.train:
        return
    rows, targets = [], []
    if args.reuse_data:
        saved = torch.load(args.reuse_data, map_location='cpu', weights_only=True)
        old = torch.load(saved['config']['checkpoint'], map_location='cpu', weights_only=True)['model']
        current = brain.state_dict()
        # Slim checkpoints omit structural buffers; compare their saved learned
        # state, including all encoders, recurrent weights and normalization.
        for name, value in old.items():
            if not name.startswith('readout.') and not torch.equal(value, current[name].cpu()):
                raise ValueError('Reused features came from a different neural representation: '+name)
        rows.append(saved['features']); targets.append(saved['labels'])
    for speed in training_speeds:
        for seed in (1921, 1922, 1923):
            motor = ('pd' if seed == 1921 else 'brain') if args.data_controller == 'mixed' else args.data_controller
            row, data = rollout(brain, cfg, meta, controller=motor, speed=speed, seed=seed, collect=True,
                                 randomize=.15, observation_reference_speed=args.speed,
                                 retina_stream=training_retina, retina_dropout=args.retina_dropout)
            print(json.dumps(dict(collection=row)), flush=True)
            rows.append(data['features']); targets.append(data['labels'])
    x = torch.cat(rows).to(brain.device)
    y = torch.cat(targets).to(brain.device)
    torch.save(dict(features=x.cpu(), labels=y.cpu(), config=config), out/'training.pt')
    x = torch.cat((x, torch.ones(len(x), 1, device=brain.device)), -1)
    parent = torch.cat((brain.readout.weight[:3].detach(), brain.readout.bias[:3, None].detach()), -1)
    # A parent-centred ridge fit preserves unexcited feature directions.
    wait_if_hot(68.)
    gram = x.T@x/len(x)
    delta = torch.linalg.solve(gram+args.ridge*torch.eye(x.shape[1], device=x.device),
                               x.T@(y-x@parent.T)/len(x)).T
    before = {n: p.detach().cpu().clone() for n, p in brain.named_parameters()}
    with torch.no_grad():
        brain.readout.weight[:3].add_(delta[:, :-1])
        brain.readout.bias[:3].add_(delta[:, -1])
    changed = {n: float((p.detach().cpu()-before[n]).abs().max()) for n, p in brain.named_parameters()
               if not torch.equal(p.detach().cpu(), before[n])}
    if set(changed)-{'readout.weight', 'readout.bias'}:
        raise RuntimeError('Readout distillation changed another brain parameter')
    meta.pop('schema', None)
    meta.update(qualified=False, motor_tracking=dict(**config, training_seeds=[1921, 1922, 1923],
                nominal_speed_mps=args.speed, training_teacher='PD; offline only', changed_parameters=changed,
                runtime_requires_teacher=False, visual_training=False, mixer_idle_corrected=True,
                recorded_scene_currents=training_retina is not None,
                training_speeds_mps=training_speeds))
    cfg = measured_dynamics(cfg, meta)
    export(out/'candidate.pt', brain, cfg, meta, 1)
    evaluation, _ = rollout(brain, cfg, meta, speed=args.speed, seed=args.evaluation_seed,
                            retina_stream=evaluation_retina, retina_dropout=args.retina_dropout)
    (out/'evaluation.json').write_text(json.dumps(evaluation, indent=2))
    print(json.dumps(dict(evaluation=evaluation, changed_parameters=changed)), flush=True)


if __name__ == '__main__':
    main()
