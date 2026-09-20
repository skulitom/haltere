"""Training-only stabilization examples from the previously qualified motor brain.

These counterfactual states vary body motion independently of motor RPM. They
preserve a braking response that ordinary action imitation can unlearn. They
are a training constraint, not a runtime controller or a flight qualification.
"""
from __future__ import annotations

import torch
from collections import deque

from ..brain.retina import RETINA_DIM, brain_to_processed, visual_observation
from ..sim.quad import quat_from_euler, quat_to_mat, yaw_of


@torch.no_grad()
def recovery_examples(teacher, cfg, calibration, count=256, length=64):
    dev = teacher.device
    # Independent errors include rises/falls and attitude/rate perturbations;
    # RPM is fixed at the calibrated hover level, never an action label proxy.
    angles = .18 * torch.randn(count, 3, device=dev)
    angles[:, 2] = 0
    quat = quat_from_euler(*angles.unbind(-1))
    R = quat_to_mat(quat)
    velocity = torch.randn(count, 3, device=dev) * torch.tensor([1., 1., 2.], device=dev)
    gyro = .35 * torch.randn(count, 3, device=dev)
    altitude = .3 + 5.7 * torch.rand(count, 1, device=dev)
    sensors = dict(gyro=gyro, gravity_body=-R[:, 2, :],
                   vel_body=(R.transpose(-1, -2) @ velocity[..., None]).squeeze(-1),
                   vel_world=velocity, pos=torch.zeros(count, 3, device=dev), quat=quat,
                   up=R[:, 2, 2], altitude=altitude, yaw=yaw_of(quat)[:, None])
    motor = torch.full((count, 1), (calibration['hover_stick_sim']+1)/2, device=dev)
    obs = visual_observation(sensors, motor, cfg.task, torch.zeros(count, RETINA_DIM, device=dev))
    teacher_obs = {k: v.clone() for k, v in obs.items() if k in teacher.channel_dims}
    teacher_obs['compass'][:, 0] = 1
    state, W = teacher.init_state(count), teacher.weight_matrix()
    actions = []
    for t in range(length):
        act, state, _ = teacher(teacher_obs, state, W)
        actions.append(brain_to_processed(act, calibration))
    batch = {k: v[:, None].expand(-1, length, -1).clone() for k, v in obs.items()}
    batch['action'] = torch.stack(actions, 1)
    return batch


class RecoveryRollout:
    """DAgger-style recovery: the student causes states, the motor teacher labels them.

    Physics is detached; gradients update the student's recurrent connectome.
    Both neural states persist across windows, and the teacher is training-only.
    """
    def __init__(self, teacher, cfg, calibration, batch_size=8):
        import copy
        from .bptt import make_world
        self.teacher, self.calibration, self.B = teacher, calibration, batch_size
        self.cfg = copy.deepcopy(cfg)
        self.cfg.train.randomize = .2
        self.cfg.train.randomize_ctl = .2
        self.vehicle, self.task = make_world(self.cfg, batch_size, teacher.device)
        self.W = teacher.weight_matrix().detach()
        self.age = 0
        self.student_state = None

    def reset(self, student):
        from ..sim.quad import QuadState
        dev, B = student.device, self.B
        q = QuadState.hover(B, dev, 3.)
        q.motor[:] = self.vehicle.sim.hover_command()
        q.vel = torch.randn(B, 3, device=dev) * .8
        q.quat = quat_from_euler(.15*torch.randn(B, device=dev), .15*torch.randn(B, device=dev),
                                torch.zeros(B, device=dev))
        self.vs = self.vehicle.wrap(q)
        self.student_state, self.teacher_state = student.init_state(B), self.teacher.init_state(B)
        self.age = 0
        self.delay = deque([torch.full((B, 4), 0., device=dev) for _ in range(self.cfg.train.delay_steps)])
        for a in self.delay:
            a[:, 0] = self.calibration['hover_stick_sim']

    def loss(self, student, scale, steps=64):
        if self.student_state is None or self.age >= 512 or bool(self.vs.quad.crashed.any()):
            self.reset(student)
        state, W = student.detach_state(self.student_state), student.weight_matrix()
        losses = []
        retina = torch.zeros(self.B, RETINA_DIM, device=student.device)
        for t in range(steps):
            with torch.no_grad():
                q = self.vs.quad
                obs = visual_observation(self.vehicle.sim.sensors(q),q.motor.mean(-1,keepdim=True),self.cfg.task,retina)
                teacher_obs = {k:v for k,v in obs.items() if k in self.teacher.channel_dims}
                teacher_obs['compass'] = torch.zeros_like(obs['compass'])
                teacher_obs['compass'][:, 0] = 1
                target, self.teacher_state, _ = self.teacher(teacher_obs,self.teacher_state,self.W)
            if t and t%8 == 0:
                state = student.detach_state(state)
            action, state, _ = student(obs,state,W)
            if self.age+t >= 20:
                losses.append(((brain_to_processed(action,self.calibration)
                                -brain_to_processed(target,self.calibration))/scale).square().mean())
            with torch.no_grad():
                self.delay.append(action.detach())
                self.vs = self.vehicle.step(self.vs,self.delay.popleft())
        self.age += steps
        self.student_state = student.detach_state(state)
        return torch.stack(losses).mean()


@torch.no_grad()
def motor_check(brain, cfg, steps=600, seed=881):
    """Closed-loop reflex check with blank imagery, independent of replay score.

    This checks braking/recovery in the training simulator, not navigation or
    real-game performance. No position target is handed to the visual brain.
    Fixed seeds and the same starting states make candidates comparable.
    """
    import copy
    from .bptt import make_world
    from ..sim.quad import QuadState

    with torch.random.fork_rng(devices=[brain.device] if brain.device.type == 'cuda' else []):
        torch.manual_seed(seed)
        cfg = copy.deepcopy(cfg)
        cfg.train.randomize = .1
        cfg.train.randomize_ctl = .1
        B = 24
        vehicle, task = make_world(cfg, B, brain.device)
        quad = QuadState.hover(B, brain.device, 3.)
        quad.motor[:] = vehicle.sim.hover_command()
        quad.vel[:, 2] = torch.tensor([0., 2., -2.], device=brain.device).repeat_interleave(8)
        quad.quat = quat_from_euler(.1*torch.randn(B, device=brain.device),
                                   .1*torch.randn(B, device=brain.device), torch.zeros(B, device=brain.device))
        vs, state, W = vehicle.wrap(quad), brain.init_state(B), brain.weight_matrix()
        retina = torch.zeros(B, RETINA_DIM, device=brain.device)

        def observe(q):
            obs = visual_observation(vehicle.sim.sensors(q), q.motor.mean(-1, keepdim=True), cfg.task, retina)
            if 'retina' not in brain.channel_dims:
                obs.pop('retina')
                obs['compass'][:, 0] = 1
            return obs

        # Warm neural state before releasing it into the simulation, as live
        # arming holds throttle low while the brain runs on fresh observations.
        for _ in range(50):
            action, state, _ = brain(observe(quad), state, W)
        delay = deque([action.clone() for _ in range(cfg.train.delay_steps)])
        heights, speeds, vertical, upright = [], [], [], []
        for _ in range(steps):
            action, state, _ = brain(observe(vs.quad), state, W)
            delay.append(action)
            vs = vehicle.step(vs, delay.popleft())
            heights.append(vs.quad.pos[:, 2])
            speeds.append(vs.quad.vel.norm(dim=-1))
            vertical.append(vs.quad.vel[:, 2])
            upright.append(quat_to_mat(vs.quad.quat)[:, 2, 2])
        height, speed = torch.stack(heights), torch.stack(speeds)
        vertical, upright = torch.stack(vertical), torch.stack(upright)
        result = dict(episodes=B, seconds=steps*cfg.brain.dt, seed=seed,
                      blank_images=True, runtime_requires_teacher=False,
                      crashed_fraction=float(vs.quad.crashed.float().mean()),
                      max_height_m=float(height.max()), min_height_m=float(height.min()),
                      max_speed_mps=float(speed.max()), max_vertical_speed_mps=float(vertical.abs().max()),
                      final_median_abs_vertical_speed_mps=float(vertical[-1].abs().median()),
                      min_up=float(upright.min()), final_median_height_m=float(height[-1].median()))
        result['reflex_pass'] = bool(result['crashed_fraction']==0 and result['max_height_m']<8
                                    and result['max_speed_mps']<5 and result['min_up']>.7
                                    and result['final_median_abs_vertical_speed_mps']<.5)
        return result


def main():
    import argparse
    import json
    from pathlib import Path
    from .bptt import load_checkpoint
    from ..vision.datasets import sha256
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(2)
    brain, cfg, _ = load_checkpoint(args.checkpoint, 'cuda')
    result = dict(checkpoint_sha256=sha256(args.checkpoint), **motor_check(brain, cfg))
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
