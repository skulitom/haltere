"""Train the connectome-constrained brain to fly by back-propagating the flight cost through the
differentiable simulator (truncated BPTT with state carry-over between windows)."""
from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..brain.model import BrainConfig, ConnectomeRNN
from ..config import dataclass_from_dict, dataclass_to_dict
from ..connectome.graph import BrainGraph
from ..sim.controller import RateControllerParams
from ..sim.quad import QuadParams
from ..sim.tasks import HoverTask, HoverTaskConfig
from ..sim.vehicle import RatesConfig, Vehicle, VehicleState
from .thermal import wait_if_hot


@dataclass
class TrainConfig:
    graph: str = 'data/built/flight'
    run: str = ''                     # run directory; empty = runs/<timestamp>
    B: int = 256
    T: int = 64                       # BPTT window (steps of brain.dt)
    iters: int = 2000
    lr: float = 2e-3
    lr_edges: float = 1e-3
    lr_readout: float = 1e-2          # motor readout and sensory encoders
    grad_clip: float = 1.0
    delay_steps: int = 2              # control latency (steps) between brain output and vehicle input
    randomize: float = 0.15           # physical parameter jitter across environments
    randomize_ctl: float = 0.0        # per-axis rate-controller gain jitter across environments
    difficulty_start: float = 0.15
    difficulty_end: float = 1.0
    difficulty_iters: int = 1200
    eval_every: int = 50
    eval_steps: int = 400
    save_every: int = 100
    seed: int = 0
    device: str = 'cuda'
    substeps: int = 2                 # physics/PID sub-steps per control step (200 Hz at dt=0.01)
    brain_detach_every: int = 8       # cut the gradient through the brain's recurrent state every k steps
                                      # (the simulator keeps full-window credit assignment); 0 = never
    freeze_internal: bool = False     # reservoir mode: keep synaptic gains, neuron gains, biases, time constants fixed
    freeze_encoders: bool = False
    max_gpu_temp: float = 0.0         # C; > 0 pauses training while the GPU is hotter than this (see thermal.py)
    temp_check_every: int = 5         # iterations between temperature checks
    iter_sleep: float = 0.0           # s of idle time after every iteration (caps average GPU power)


@dataclass
class ExperimentConfig:
    train: TrainConfig
    task: HoverTaskConfig
    brain: BrainConfig
    quad: QuadParams
    ctl: RateControllerParams
    rates: RatesConfig

    @staticmethod
    def from_dict(d: dict) -> "ExperimentConfig":
        return ExperimentConfig(
            train=dataclass_from_dict(TrainConfig, d.get('train')),
            task=dataclass_from_dict(HoverTaskConfig, d.get('task')),
            brain=dataclass_from_dict(BrainConfig, d.get('brain')),
            quad=dataclass_from_dict(QuadParams, d.get('quad')),
            ctl=dataclass_from_dict(RateControllerParams, d.get('ctl')),
            rates=dataclass_from_dict(RatesConfig, d.get('rates')),
        )

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)


def build_brain(cfg: ExperimentConfig, channels: dict[str, int], device):
    """The connectome brain (default) or a baseline policy, plus the graph (None for baselines)."""
    if cfg.brain.model == 'mlp':
        from ..brain.baselines import MLPPolicy
        return MLPPolicy(channels, cfg.brain.n_actions, cfg.brain.mlp_hidden, cfg.brain.dt, cfg.brain.action_tau,
                         device), None
    graph = BrainGraph.load(Path(cfg.train.graph))
    return ConnectomeRNN(graph, channels, cfg.brain, device), graph


def make_world(cfg: ExperimentConfig, B: int, device) -> tuple[Vehicle, HoverTask]:
    vehicle = Vehicle(cfg.quad, cfg.ctl, cfg.rates, device, dt=cfg.brain.dt, substeps=cfg.train.substeps)
    if cfg.train.randomize > 0:
        vehicle.sim.randomize(B, cfg.train.randomize)
    if cfg.train.randomize_ctl > 0:
        base = torch.tensor(cfg.ctl.axis_gain, device=device)
        vehicle.ctl.axis_gain = base * (1.0 + cfg.train.randomize_ctl * (2 * torch.rand(B, 3, device=device) - 1))
    task = HoverTask(cfg.task, vehicle.sim, B, device)
    return vehicle, task


@torch.no_grad()
def evaluate(brain: ConnectomeRNN, cfg: ExperimentConfig, steps: int, B: int, device,
             difficulty: float = 1.0, record: bool = False) -> dict:
    vehicle, task = make_world(cfg, B, device)
    vs = vehicle.wrap(task.reset_all(difficulty))
    bs = brain.init_state(B)
    W = brain.weight_matrix()
    delay = deque([torch.zeros(B, brain.n_actions, device=device) for _ in range(cfg.train.delay_steps)])
    total = torch.zeros(B, device=device)
    traj = []
    crashed_ever = torch.zeros(B, dtype=torch.bool, device=device)
    for t in range(steps):
        obs = task.observe(vs.quad)
        act, bs, aux = brain(obs, bs, W)
        delay.append(act)
        a = delay.popleft()
        vs = vehicle.step(vs, a)
        total += task.cost(vs.quad, a)
        crashed_ever |= vs.quad.crashed
        task.t += 1
        task.advance_target()
        if record:
            traj.append({'pos': vs.quad.pos[0].cpu().numpy(), 'target': task.target[0].cpu().numpy(),
                         'act': a[0].cpu().numpy(), 'quat': vs.quad.quat[0].cpu().numpy()})
    m = task.metrics(vs.quad)
    m.update({'cost': float(total.mean() / steps), 'crashed_ever': float(crashed_ever.float().mean())})
    if record:
        m['trajectory'] = traj
    return m


def train(cfg: ExperimentConfig, resume: str | None = None) -> Path:
    tc = cfg.train
    device = torch.device(tc.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    run_dir = Path(tc.run) if tc.run else Path('runs') / time.strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)

    vehicle, task = make_world(cfg, tc.B, device)
    brain, graph = build_brain(cfg, task.channels, device)
    # start with the throttle output at the hover stick instead of mid-stick (which is ~2 g of thrust)
    hover_stick = max(-0.99, min(0.99, 2 * vehicle.sim.hover_command() - 1))
    with torch.no_grad():
        brain.readout.bias[0] = float(np.arctanh(hover_stick))
    if graph is not None:
        print(graph.summary())
    n_params = sum(p.numel() for p in brain.parameters() if p.requires_grad)
    print(f'brain ({cfg.brain.model}): {brain.N} neurons, {brain.E} edges, {n_params} trainable parameters on {device}')

    groups = {'edges': [], 'fast': [], 'slow': []}
    for n, p in brain.named_parameters():
        if not p.requires_grad:
            continue
        internal = cfg.brain.model == 'connectome' and not (n.startswith('readout') or n.startswith('encoders'))
        if (tc.freeze_internal and internal) or (tc.freeze_encoders and n.startswith('encoders')):
            p.requires_grad_(False)
            continue
        if n == 'log_edge_gain':
            groups['edges'].append(p)
        elif n.startswith('readout') or n.startswith('encoders') or cfg.brain.model != 'connectome':
            groups['fast'].append(p)
        else:
            groups['slow'].append(p)
    n_params = sum(p.numel() for p in brain.parameters() if p.requires_grad)
    print(f'trainable after freezing: {n_params} parameters '
          f'(internal {"frozen" if tc.freeze_internal else "free"}, encoders {"frozen" if tc.freeze_encoders else "free"})')
    opt = torch.optim.Adam([g for g in ({'params': groups['edges'], 'lr': tc.lr_edges},
                                        {'params': groups['fast'], 'lr': tc.lr_readout},
                                        {'params': groups['slow'], 'lr': tc.lr}) if g['params']])
    start_iter = 0
    if resume:
        from ..brain.model import migrate_state_dict
        ck = torch.load(resume, map_location=device)
        brain.load_state_dict(migrate_state_dict(ck['model'], brain))
        if ck.get('opt'):
            try:
                opt.load_state_dict(ck['opt'])
            except ValueError:
                print('optimiser state not compatible with the parameter groups; starting the optimiser fresh')
        start_iter = ck.get('iter', 0)
        print(f'resumed from {resume} at iteration {start_iter}')

    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, indent=2)
    log_path = run_dir / 'log.csv'
    new_log = not log_path.exists() or log_path.stat().st_size == 0
    log_f = open(log_path, 'a', newline='', encoding='utf-8')
    log = csv.writer(log_f)
    if new_log:        # not `start_iter == 0`: a run resumed into a fresh directory needs its header too
        log.writerow(['iter', 'loss', 'cost', 'rate_mean', 'rate_motor', 'dist_mean', 'crashed', 'difficulty', 'time',
                      'grad_norm'])

    def difficulty_at(it: int) -> float:
        f = min(1.0, it / max(tc.difficulty_iters, 1))
        return tc.difficulty_start + (tc.difficulty_end - tc.difficulty_start) * f

    diff = difficulty_at(start_iter)
    vs = vehicle.wrap(task.reset_all(diff))
    bs = brain.init_state(tc.B)
    delay = deque([torch.zeros(tc.B, brain.n_actions, device=device) for _ in range(tc.delay_steps)])
    zeros_act = torch.zeros(tc.B, brain.n_actions, device=device)
    t_start = time.time()
    best_eval = float('inf')

    for it in range(start_iter, tc.iters):
        if tc.max_gpu_temp > 0 and it % max(tc.temp_check_every, 1) == 0:
            wait_if_hot(tc.max_gpu_temp)
        if tc.iter_sleep > 0:
            time.sleep(tc.iter_sleep)
        diff = difficulty_at(it)
        W = brain.weight_matrix()
        total = torch.zeros((), device=device)
        crashes = 0.0
        for t in range(tc.T):
            obs = task.observe(vs.quad)
            if tc.brain_detach_every and t % tc.brain_detach_every == 0 and t > 0:
                bs = brain.detach_state(bs)
            act, bs, aux = brain(obs, bs, W)
            delay.append(act)
            a = delay.popleft()
            vs = vehicle.step(vs, a)
            total = total + task.cost(vs.quad, a).mean()
            done = task.tick(vs.quad)
            if bool(done.any()):
                crashes += float(vs.quad.crashed.float().sum())
                new_quad = task.reset_where(vs.quad, done, diff)
                vs = VehicleState(new_quad, vs.ctl.where(done, vehicle.wrap(new_quad).ctl))
                bs = brain.where_state(done, brain.init_state(tc.B), bs)
                delay = deque(torch.where(done[:, None], zeros_act, d) for d in delay)
        loss = total / tc.T + brain.regularization(aux)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(brain.parameters(), tc.grad_clip))
        opt.step()
        vs = vs.detach()
        bs = brain.detach_state(bs)
        delay = deque(d.detach() for d in delay)

        m = task.metrics(vs.quad)
        row = [it, float(loss), float(total / tc.T), float(aux['rate_mean']), float(aux['rate_motor']),
               m['dist_mean'], crashes / tc.B, diff, time.time() - t_start, grad_norm]
        log.writerow(row)
        log_f.flush()
        if it % 10 == 0 or it == start_iter:
            print(f'it {it:5d} loss {row[1]:8.4f} cost {row[2]:8.4f} rate {row[3]:.3f} dist {row[5]:6.2f}m '
                  f'crash/env {row[6]:.3f} diff {diff:.2f} |grad| {grad_norm:8.2f} {row[8]:7.1f}s', flush=True)
        if tc.eval_every and (it + 1) % tc.eval_every == 0:
            ev = evaluate(brain, cfg, tc.eval_steps, min(tc.B, 256), device)
            print(f'  eval: dist {ev["dist_mean"]:.2f}m within0.5 {ev["within_0.5m"]:.2f} '
                  f'crashed_ever {ev["crashed_ever"]:.2f} cost {ev["cost"]:.3f}', flush=True)
            with open(run_dir / 'eval.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({'iter': it + 1, **{k: v for k, v in ev.items() if k != 'trajectory'}}) + '\n')
            if ev['cost'] < best_eval:
                best_eval = ev['cost']
                save_checkpoint(run_dir / 'best.pt', brain, opt, cfg, it + 1, tc.graph)
        if tc.save_every and (it + 1) % tc.save_every == 0:
            save_checkpoint(run_dir / 'last.pt', brain, opt, cfg, it + 1, tc.graph)
    save_checkpoint(run_dir / 'last.pt', brain, opt, cfg, tc.iters, tc.graph)
    log_f.close()
    return run_dir


def save_checkpoint(path: Path, brain, opt, cfg: ExperimentConfig, it: int, graph_path: str) -> None:
    torch.save({'model': brain.state_dict(), 'opt': opt.state_dict() if opt is not None else None,
                'config': cfg.to_dict(), 'iter': it, 'graph': str(graph_path),
                'channels': brain.channel_dims}, path)


def load_checkpoint(path: str | Path, device='cuda'):
    """Returns (brain, config, graph); graph is None for baseline policies."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    ck = torch.load(path, map_location=device)
    cfg = ExperimentConfig.from_dict(ck['config'])
    cfg.train.graph = ck['graph']
    if cfg.brain.model != 'mlp' and not Path(cfg.train.graph).with_suffix('.npz').exists():
        # the stored graph path is relative to the repository the brain was trained in: from another working
        # directory (a git worktree, a script elsewhere) look for it next to the checkpoint's ancestors
        for base in Path(path).resolve().parents:
            if (base / cfg.train.graph).with_suffix('.npz').exists():
                cfg.train.graph = str(base / cfg.train.graph)
                break
    brain, graph = build_brain(cfg, ck['channels'], device)
    from ..brain.model import migrate_state_dict
    brain.load_state_dict(migrate_state_dict(ck['model'], brain), strict=not ck.get('slim', False))
    brain.eval()
    return brain, cfg, graph
