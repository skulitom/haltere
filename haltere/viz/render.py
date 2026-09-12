"""Record a flight of the fly brain and render neural activity and the drone side by side.

Neurons are drawn at their real soma positions in the male CNS (from the connectome annotations;
sensory afferents whose cell bodies lie outside the CNS are placed at the centroid of their synaptic
targets), coloured by population and brightened by their firing rate. The drone is drawn in 3D with
its target and a trail, with the four stick outputs underneath. Output: an MP4 (ffmpeg) and a GIF
for the README; ``--live`` shows the same animation in a window instead.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from ..connectome import sources
from ..sim.quad import quat_to_mat

POP_COLORS = {  # base colour per population (RGB in 0..1); order matters (later wins)
    'other':      (0.45, 0.45, 0.55),
    'cx':         (0.95, 0.80, 0.25),
    'descending': (0.35, 0.85, 0.45),
    'premotor':   (0.55, 0.60, 0.95),
    'lptc':       (0.30, 0.85, 0.95),
    'ocelli':     (0.30, 0.85, 0.95),
    'jo':         (0.30, 0.85, 0.95),
    'compass':    (0.95, 0.55, 0.20),
    'goal':       (0.95, 0.55, 0.20),
    'wing_cs':    (0.20, 0.95, 0.85),
    'haltere':    (0.20, 0.95, 0.85),
    'wing_mn':    (0.98, 0.30, 0.35),
}
LEGEND = [('senses (haltere, wing, optic flow, ocelli, antennae)', (0.25, 0.9, 0.9)),
          ('central complex (compass, goal)', (0.95, 0.7, 0.25)),
          ('descending neurons', (0.35, 0.85, 0.45)),
          ('premotor (readout)', (0.55, 0.60, 0.95)),
          ('wing motor neurons (readout)', (0.98, 0.30, 0.35))]


# ----------------------------------------------------------------------------- recording

def record_flight(ckpt: str, seconds: float = 20.0, difficulty: float = 0.8, waypoint_every: float = 4.0,
                  stride: int = 4, device: str = 'cuda', seed: int = 0) -> dict:
    from ..train.bptt import load_checkpoint, make_world
    torch.manual_seed(seed)
    np.random.seed(seed)
    brain, cfg, graph = load_checkpoint(ckpt, device)
    dev = brain.device
    cfg.train.randomize = 0.0
    if waypoint_every > 0:
        cfg.task.switch_target_every = int(round(waypoint_every / cfg.brain.dt))
        cfg.task.episode_steps = 10 ** 9
    cfg.task.target_box = (2.0, 2.0, 0.7)   # keep the flight in a box the camera can frame
    B = 1
    vehicle, task = make_world(cfg, B, dev)
    vs = vehicle.wrap(task.reset_all(difficulty))
    bs = brain.init_state(B)
    W = brain.weight_matrix()
    delay = deque([torch.zeros(B, brain.n_actions, device=dev) for _ in range(cfg.train.delay_steps)])
    steps = int(seconds / cfg.brain.dt)
    rates, pos, quat, target, act, times = [], [], [], [], [], []
    with torch.no_grad():
        for t in range(steps):
            obs = task.observe(vs.quad)
            a_out, bs, aux = brain(obs, bs, W)
            delay.append(a_out)
            a = delay.popleft()
            vs = vehicle.step(vs, a)
            task.tick(vs.quad)
            if t % stride == 0:
                r = brain.cfg.rate_max * torch.sigmoid(bs['v'][:, 0]) if 'v' in bs else None
                rates.append(r.float().cpu().numpy() if r is not None else None)
                pos.append(vs.quad.pos[0].cpu().numpy())
                quat.append(vs.quad.quat[0].cpu().numpy())
                target.append(task.target[0].cpu().numpy())
                act.append(a[0].cpu().numpy())
                times.append(t * cfg.brain.dt)
    return {'rates': np.stack(rates) if rates[0] is not None else None, 'pos': np.stack(pos), 'quat': np.stack(quat),
            'target': np.stack(target), 'act': np.stack(act), 'time': np.asarray(times), 'graph': graph,
            'dt': cfg.brain.dt * stride, 'ckpt': ckpt, 'crashed': bool(vs.quad.crashed.any())}


# ----------------------------------------------------------------------------- layout

def neuron_layout(graph, version: str = sources.DEFAULT_VERSION, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """2-D positions (x left-right, y brain-to-cord) for every node, and a colour per node."""
    path = sources.file_path('annotations', None, version)
    ann = feather.read_feather(path, columns=['bodyId', 'somaLocation']).set_index('bodyId')
    sub = ann.reindex(graph.nodes['bodyId'].to_numpy())
    has = sub['somaLocation'].notna().to_numpy()
    xyz = np.full((graph.N, 3), np.nan)
    xyz[has] = np.stack(sub.loc[has, 'somaLocation'].to_numpy()).astype(float)
    # neurons without a soma in the volume: centroid of their synaptic targets (then of targets' targets)
    rng = np.random.default_rng(seed)
    for _ in range(3):
        missing = np.isnan(xyz[:, 0])
        if not missing.any():
            break
        acc = np.zeros((graph.N, 3))
        cnt = np.zeros(graph.N)
        ok = ~np.isnan(xyz[graph.post, 0]) & missing[graph.pre]
        np.add.at(acc, graph.pre[ok], xyz[graph.post[ok]])
        np.add.at(cnt, graph.pre[ok], 1.0)
        fill = missing & (cnt > 0)
        xyz[fill] = acc[fill] / cnt[fill, None] + rng.normal(0, 1500, (int(fill.sum()), 3))
    still = np.isnan(xyz[:, 0])
    if still.any():
        xyz[still] = np.nanmean(xyz, axis=0) + rng.normal(0, 3000, (int(still.sum()), 3))
    # project: horizontal = x (left-right), vertical = z (anterior brain at the top)
    x = xyz[:, 0]
    y = -xyz[:, 2]
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    colors = np.tile(np.array(POP_COLORS['other']), (graph.N, 1))
    for name, rgb in POP_COLORS.items():
        if name in graph.populations:
            colors[graph.population(name)] = rgb
    return np.stack([x, y], 1), colors


# ----------------------------------------------------------------------------- drawing

def _drone_lines(p: np.ndarray, q: np.ndarray, arm: float = 0.35):
    R = quat_to_mat(torch.as_tensor(q, dtype=torch.float32)[None])[0].numpy()
    lf, rf, lb, rb = (np.array(v) for v in ([arm, arm, 0], [arm, -arm, 0], [-arm, arm, 0], [-arm, -arm, 0]))
    pts = [p + R @ v for v in (lf, rb, rf, lb)]
    return [(pts[0], pts[1]), (pts[2], pts[3])], p + R @ np.array([arm * 1.3, 0, 0])


class FlightFigure:
    def __init__(self, rec: dict, layout: np.ndarray, colors: np.ndarray, width: int = 1280, height: int = 720):
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt
        self.rec = rec
        self.layout = layout
        self.base = colors
        r = rec['rates']
        self.mu = r.mean(0)
        self.sd = r.std(0) + 1e-3
        dpi = 100
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor='black')
        gs = self.fig.add_gridspec(4, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1, 1, 0.55], left=0.02, right=0.98,
                                   top=0.88, bottom=0.05, wspace=0.05, hspace=0.25)
        self.ax_brain = self.fig.add_subplot(gs[:, 0])
        self.ax_drone = self.fig.add_subplot(gs[0:3, 1], projection='3d')
        self.ax_sticks = self.fig.add_subplot(gs[3, 1])
        for ax in (self.ax_brain, self.ax_sticks):
            ax.set_facecolor('black')
        self.ax_brain.set_xlim(-0.03, 1.03)
        self.ax_brain.set_ylim(-0.03, 1.03)
        self.ax_brain.set_aspect('equal')
        self.ax_brain.axis('off')
        n = rec['graph'].N
        rgba = np.concatenate([self.base * 0.35, np.full((n, 1), 0.8)], 1)
        self.scat = self.ax_brain.scatter(layout[:, 0], layout[:, 1], s=2.0, c=rgba, linewidths=0, rasterized=True)
        self.ax_brain.set_title(f'male CNS v1.0 connectome: {n:,} neurons, {rec["graph"].E:,} synaptic connections\n'
                                f'colour = population, brightness = firing rate above rest',
                                color='white', fontsize=10, loc='left')
        for i, (label, rgb) in enumerate(LEGEND):
            self.ax_brain.scatter([], [], s=18, c=[rgb], label=label)
        self.ax_brain.legend(loc='lower left', fontsize=7.5, frameon=False, labelcolor='white', markerscale=1.0)
        # drone axes
        ax = self.ax_drone
        ax.set_facecolor('black')
        ax.xaxis.set_pane_color((0, 0, 0, 1)); ax.yaxis.set_pane_color((0, 0, 0, 1)); ax.zaxis.set_pane_color((0, 0, 0, 1))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color('white'); axis._axinfo['grid']['color'] = (0.25, 0.25, 0.3, 1)
        ax.tick_params(colors='white', labelsize=7)
        lim = float(np.abs(np.concatenate([rec['pos'][:, :2], rec['target'][:, :2]])).max()) + 0.8
        lim = max(2.0, lim)
        zmax = float(max(rec['pos'][:, 2].max(), rec['target'][:, 2].max())) + 0.8
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, zmax)
        ax.set_box_aspect((2 * lim, 2 * lim, zmax))
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
        ax.view_init(elev=24, azim=-50)
        ax.set_title('drone (simulator, full difficulty): white = drone, yellow x = target', color='white', fontsize=10,
                     loc='left')
        self.trail, = ax.plot([], [], [], color=(0.6, 0.6, 1.0), lw=1.2, alpha=0.9)
        self.arm1, = ax.plot([], [], [], color='white', lw=3.0)
        self.arm2, = ax.plot([], [], [], color='white', lw=3.0)
        self.nose, = ax.plot([], [], [], color=(1.0, 0.4, 0.4), lw=3.0)
        self.tgt, = ax.plot([], [], [], marker='x', color=(1.0, 0.85, 0.2), markersize=12, mew=2.5, ls='')
        self.tgt_line, = ax.plot([], [], [], color=(1.0, 0.85, 0.2), lw=0.8, alpha=0.5, ls='--')
        self.shadow, = ax.plot([], [], [], color=(0.5, 0.5, 0.5), lw=1.0, alpha=0.7, ls=':')
        self.arm_len = 0.12 * lim
        # sticks
        axs = self.ax_sticks
        axs.set_xlim(-1.05, 1.05); axs.set_ylim(-0.6, 3.6)
        axs.set_yticks(range(4)); axs.set_yticklabels(['throttle', 'roll', 'pitch', 'yaw'], color='white', fontsize=8)
        axs.set_xticks([-1, 0, 1]); axs.tick_params(colors='white', labelsize=7)
        for s in axs.spines.values():
            s.set_color((0.3, 0.3, 0.3))
        axs.axvline(0, color=(0.35, 0.35, 0.35), lw=1)
        self.bars = axs.barh(range(4), [0, 0, 0, 0], color=[(0.98, 0.3, 0.35)] + [(0.55, 0.6, 0.95)] * 3, height=0.6)
        self.text = self.fig.text(0.98, 0.955, '', color='white', fontsize=12, family='monospace', ha='right')
        self.fig.text(0.02, 0.955, 'Haltere: a connectome-constrained fly brain flying a drone', color=(0.85, 0.85, 0.85),
                      fontsize=13)

    def draw(self, k: int) -> None:
        rec = self.rec
        act = np.clip((rec['rates'][k] - self.mu) / self.sd, -1.0, 4.0)
        bright = np.clip(0.12 + 0.32 * np.clip(act, 0, 4), 0.0, 1.0)
        rgb = np.clip(self.base * (0.25 + 1.1 * bright[:, None]), 0, 1)
        rgb = np.clip(rgb + 0.35 * np.clip(act - 2.0, 0, 2)[:, None] / 2.0, 0, 1)   # very active neurons whiten
        rgba = np.concatenate([rgb, np.full((len(act), 1), 0.95)], 1)
        self.scat.set_facecolors(rgba)
        self.scat.set_sizes(1.2 + 14.0 * np.clip(act, 0, 4) / 4.0)
        p, q, tgt = rec['pos'][k], rec['quat'][k], rec['target'][k]
        lines, nose = _drone_lines(p, q, arm=self.arm_len)
        for line, (a, b) in zip((self.arm1, self.arm2), lines):
            line.set_data_3d([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
        self.nose.set_data_3d([p[0], nose[0]], [p[1], nose[1]], [p[2], nose[2]])
        k0 = max(0, k - int(2.0 / rec['dt']))
        tr = rec['pos'][k0:k + 1]
        self.trail.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])
        self.tgt.set_data_3d([tgt[0]], [tgt[1]], [tgt[2]])
        self.tgt_line.set_data_3d([tgt[0], tgt[0]], [tgt[1], tgt[1]], [0.0, tgt[2]])
        self.shadow.set_data_3d([p[0], p[0]], [p[1], p[1]], [0.0, p[2]])
        for bar, v in zip(self.bars, rec['act'][k]):
            bar.set_width(float(v))
        d = float(np.linalg.norm(p - tgt))
        self.text.set_text(f't = {rec["time"][k]:5.1f} s   distance to target {d:4.2f} m')

    def frame(self, k: int) -> np.ndarray:
        self.draw(k)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()


def render_video(rec: dict, out_dir: Path, fps: int | None = None, gif_seconds: float = 12.0, gif_width: int = 720,
                 width: int = 1280, height: int = 720) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout, colors = neuron_layout(rec['graph'])
    fig = FlightFigure(rec, layout, colors, width, height)
    fps = fps or int(round(1.0 / rec['dt']))
    ffmpeg = shutil.which('ffmpeg')
    mp4 = out_dir / 'flight.mp4'
    n = len(rec['time'])
    t0 = time.time()
    if ffmpeg:
        cmd = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
               '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'medium', str(mp4)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for k in range(n):
            proc.stdin.write(fig.frame(k).tobytes())
            if k % 50 == 0:
                print(f'  frame {k}/{n} ({time.time() - t0:.0f}s)', file=sys.stderr, flush=True)
        proc.stdin.close()
        proc.wait()
        gif = out_dir / 'flight.gif'
        gif_frames = int(min(gif_seconds, n / fps) * fps)
        subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-t', str(gif_frames / fps), '-i', str(mp4),
                        '-vf', f'fps=15,scale={gif_width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4',
                        str(gif)], check=True)
    else:
        print('ffmpeg not found on PATH; writing PNG frames instead', file=sys.stderr)
        for k in range(n):
            fig.fig.savefig(out_dir / f'frame_{k:05d}.png', dpi=100, facecolor='black')
        gif = None
    still = out_dir / 'flight_still.png'
    fig.draw(n // 2)
    fig.fig.savefig(still, dpi=100, facecolor='black')
    meta = {'ckpt': rec['ckpt'], 'frames': n, 'fps': fps, 'seconds': float(rec['time'][-1]), 'crashed': rec['crashed'],
            'mean_distance_m': float(np.linalg.norm(rec['pos'] - rec['target'], axis=1).mean()),
            'mp4': str(mp4) if ffmpeg else None, 'gif': str(gif) if gif else None, 'still': str(still)}
    with open(out_dir / 'flight.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return meta


def live_view(rec: dict, speed: float = 1.0) -> None:
    import matplotlib
    matplotlib.use('TkAgg')
    layout, colors = neuron_layout(rec['graph'])
    fig = FlightFigure(rec, layout, colors)
    plt = fig.plt
    plt.ion()
    fig.fig.show()
    n = len(rec['time'])
    t0 = time.time()
    k = 0
    while k < n and plt.fignum_exists(fig.fig.number):
        fig.draw(k)
        fig.fig.canvas.draw_idle()
        fig.fig.canvas.flush_events()
        k = min(n - 1, int((time.time() - t0) * speed / rec['dt']) + 1) if speed > 0 else k + 1
        plt.pause(0.001)
    plt.ioff()
    plt.show()
