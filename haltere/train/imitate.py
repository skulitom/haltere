"""Imitation: train the connectome brain to reproduce a working controller (the MLP baseline) on
that controller's own flights, then measure it in closed loop.

This isolates representation from optimisation: if the brain can imitate the controller's command
stream with high fidelity, the connectome architecture can express the controller, and the
remaining difficulty in flight-cost training is the optimisation through the simulator.
"""
from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .bptt import ExperimentConfig, build_brain, evaluate, load_checkpoint, make_world, save_checkpoint


@dataclass
class ImitateConfig:
    teacher: str = 'runs/mlp300/best.pt'
    run: str = ''
    B: int = 256
    T: int = 32                       # BPTT window for the student
    iters: int = 400
    lr_readout: float = 3e-3
    lr: float = 5e-4
    lr_edges: float = 3e-4
    grad_clip: float = 1.0
    difficulty: float = 0.7           # perturbation level of the rollouts
    student_frac: float = 0.0         # fraction of environments flown by the student itself (DAgger-style)
    student_frac_final: float = 0.5   # ... ramped linearly to this value by the last iteration
    brain_detach_every: int = 8
    eval_every: int = 50
    eval_steps: int = 400
    save_every: int = 100
    seed: int = 0
    device: str = 'cuda'
    freeze_internal: bool = False
    resume: str = ''                  # checkpoint to continue from (model + optimiser state)


def imitate(cfg: ExperimentConfig, ic: ImitateConfig) -> Path:
    device = torch.device(ic.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(ic.seed)
    np.random.seed(ic.seed)
    run_dir = Path(ic.run) if ic.run else Path('runs') / ('imitate-' + time.strftime('%Y%m%d-%H%M%S'))
    run_dir.mkdir(parents=True, exist_ok=True)

    teacher, tcfg, _ = load_checkpoint(ic.teacher, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    cfg.train.substeps = tcfg.train.substeps
    vehicle, task = make_world(cfg, ic.B, device)
    student, graph = build_brain(cfg, task.channels, device)
    hover_stick = max(-0.99, min(0.99, 2 * vehicle.sim.hover_command() - 1))
    with torch.no_grad():
        student.readout.bias[0] = float(np.arctanh(hover_stick))
    if graph is not None:
        print(graph.summary())

    groups = {'edges': [], 'fast': [], 'slow': []}
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        internal = not (n.startswith('readout') or n.startswith('encoders'))
        if ic.freeze_internal and internal:
            p.requires_grad_(False)
            continue
        (groups['edges'] if n == 'log_edge_gain' else groups['slow'] if internal else groups['fast']).append(p)
    opt = torch.optim.Adam([g for g in ({'params': groups['edges'], 'lr': ic.lr_edges},
                                        {'params': groups['fast'], 'lr': ic.lr_readout},
                                        {'params': groups['slow'], 'lr': ic.lr}) if g['params']])
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f'student: {student.N} neurons, {n_params} trainable parameters; teacher {ic.teacher}; '
          f'readout population size {len(student.motor_idx) if hasattr(student, "motor_idx") else "-"}')
    start_iter = 0
    if ic.resume:
        from ..brain.model import migrate_state_dict
        ck = torch.load(ic.resume, map_location=device)
        student.load_state_dict(migrate_state_dict(ck['model'], student))
        if ck.get('opt'):
            try:
                opt.load_state_dict(ck['opt'])
            except ValueError:
                print('optimiser state not compatible with the parameter groups; starting the optimiser fresh')
        start_iter = int(ck.get('iter', 0))
        print(f'resumed from {ic.resume} at iteration {start_iter}')

    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump({'experiment': cfg.to_dict(), 'imitate': ic.__dict__}, f, indent=2)
    log_path = run_dir / 'log.csv'
    new_log = not log_path.exists() or log_path.stat().st_size == 0
    log_f = open(log_path, 'a', newline='', encoding='utf-8')
    log = csv.writer(log_f)
    if new_log:        # not `start_iter == 0`: a run resumed into a fresh directory needs its header too
        log.writerow(['iter', 'mse', 'r2_thr', 'r2_roll', 'r2_pitch', 'r2_yaw', 'student_frac', 'grad_norm', 'time'])

    vs = vehicle.wrap(task.reset_all(ic.difficulty))
    ts = teacher.init_state(ic.B)
    ss = student.init_state(ic.B)
    n_act = student.n_actions
    delay = deque([torch.zeros(ic.B, n_act, device=device) for _ in range(cfg.train.delay_steps)])
    zeros_act = torch.zeros(ic.B, n_act, device=device)
    t_start = time.time()
    best = float('inf')
    for it in range(start_iter, ic.iters):
        frac = ic.student_frac + (ic.student_frac_final - ic.student_frac) * it / max(ic.iters - 1, 1)
        n_student = int(round(frac * ic.B))
        driven_by_student = torch.zeros(ic.B, dtype=torch.bool, device=device)
        driven_by_student[:n_student] = True
        W = student.weight_matrix()
        se = torch.zeros(n_act, device=device)
        var = torch.zeros(n_act, device=device)
        mean_acc = torch.zeros(n_act, device=device)
        loss_acc = torch.zeros((), device=device)
        for t in range(ic.T):
            obs = task.observe(vs.quad)
            with torch.no_grad():
                t_act, ts, t_aux = teacher(obs, ts)
            if ic.brain_detach_every and t % ic.brain_detach_every == 0 and t > 0:
                ss = student.detach_state(ss)
            s_act, ss, s_aux = student(obs, ss, W)
            u_t = t_aux['u']
            u_s = s_aux['u']
            err = (u_s - u_t).pow(2)
            loss_acc = loss_acc + err.mean()
            se += err.detach().sum(0)
            var += (u_t - u_t.mean(0)).pow(2).sum(0)
            a = torch.where(driven_by_student[:, None], s_act.detach(), t_act)
            delay.append(a)
            a_applied = delay.popleft()
            vs = vehicle.step(vs, a_applied)
            done = task.tick(vs.quad)
            if bool(done.any()):
                new_quad = task.reset_where(vs.quad, done, ic.difficulty)
                vs = vs.__class__(new_quad, vs.ctl.where(done, vehicle.wrap(new_quad).ctl))
                ts = teacher.where_state(done, teacher.init_state(ic.B), ts)
                ss = student.where_state(done, student.init_state(ic.B), ss)
                delay = deque(torch.where(done[:, None], zeros_act, d) for d in delay)
        loss = loss_acc / ic.T + student.regularization(s_aux)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(student.parameters(), ic.grad_clip))
        opt.step()
        vs = vs.detach()
        ss = student.detach_state(ss)
        delay = deque(d.detach() for d in delay)
        r2 = (1 - se / var.clamp(min=1e-9)).cpu().numpy()
        row = [it, float(loss_acc / ic.T), *[float(x) for x in r2], frac, gn, time.time() - t_start]
        log.writerow(row)
        log_f.flush()
        if it % 10 == 0:
            print(f'it {it:4d} mse {row[1]:.4f} R2 thr {r2[0]:+.2f} roll {r2[1]:+.2f} pitch {r2[2]:+.2f} yaw {r2[3]:+.2f} '
                  f'student-driven {frac:.2f} |grad| {gn:6.2f} {row[-1]:6.0f}s', flush=True)
        if ic.eval_every and (it + 1) % ic.eval_every == 0:
            student.eval()
            ev = evaluate(student, cfg, ic.eval_steps, min(ic.B, 256), device, difficulty=1.0)
            student.train()
            print(f'  closed loop: dist {ev["dist_mean"]:.2f}m within0.5 {ev["within_0.5m"]:.2f} '
                  f'crashed_ever {ev["crashed_ever"]:.2f} cost {ev["cost"]:.3f}', flush=True)
            with open(run_dir / 'eval.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({'iter': it + 1, **ev}) + '\n')
            if ev['cost'] < best:
                best = ev['cost']
                save_checkpoint(run_dir / 'best.pt', student, opt, cfg, it + 1, cfg.train.graph)
        if ic.save_every and (it + 1) % ic.save_every == 0:
            save_checkpoint(run_dir / 'last.pt', student, opt, cfg, it + 1, cfg.train.graph)
    save_checkpoint(run_dir / 'last.pt', student, opt, cfg, ic.iters, cfg.train.graph)
    log_f.close()
    return run_dir
