"""Batched surrogate rehearsal of the fast race-cue pilot over scoring-only hills (development tool, no flight).

The closed loop is `haltere.train.fast_motor_tracking.rollout` (the fast-brain development gate: measured-drone
surrogate `IdentifiedSim` with 10% per-drone randomisation, synthetic HUD marker with latency and dropout, three-tick
command delay, quadratic drag, one-second arming hold, sim seed 17) for either motor: the fast PD or a fast-contract
brain checkpoint (its own `fast_motor_tracking` contract). Two things are added:
- pilot keyword arguments for the pilot under test (e.g. ``descent_view=DescentViewConfig``);
- per-course scoring with `fast_rehearsal.DescentScore`: ground contacts on the scoring-only hills of
  `fast_rehearsal.CourseTerrain` (the pilot and motor never see them; the drone flies through), minimum clearance,
  time with the velocity vector below the camera's lower image edge while descending, and checkpoint passes more than
  1.5 m above the checkpoint centre.

Course sets (seeded, no Liftoff geometry): ``flat`` = synthetic_course(seed) without terrain, ``steep`` =
synthetic_course(seed, steep=0.4) with hills under its descending legs, ``hill`` = hill_course(seed) with hills.
Nothing here is flight evidence.

usage:
  python -m haltere.liftoff.descent_rehearsal run --controller pd|brain --checkpoint CKPT --set hill:6000-6007 \
      [--descent-view configs/pilot/descent_view.json] --out OUT.json
  python -m haltere.liftoff.descent_rehearsal score --gates configs/pilot/descent_view_gates.json --dir DIR --json OUT
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..brain.motor_baseline import FastMotorPD
from ..brain.retina import RETINA_DIM
from ..sim.identified import IdentifiedSim
from ..vision.camera import Camera
from .camera_pose import CameraPoseHistory
from .fast_race_cue import FastRaceCue
from .fast_rehearsal import CourseTerrain, DescentScore, hill_course, hud_marker, synthetic_course

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = 'runs/measured-dynamics-low-speed-20260923/profile.json'
DEFAULT_CHECKPOINT = 'runs/fast-brain-08-vgs04-s10r03m30/candidate.pt'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def parse_set(spec):
    """'kind:a-b' or 'kind:a,b,c' -> (kind, [seeds])."""
    kind, _, seeds = spec.partition(':')
    if kind not in ('flat', 'steep', 'hill') or not seeds:
        raise ValueError(f'Use flat|steep|hill:SEEDS, not {spec!r}')
    out = []
    for part in seeds.split(','):
        lo, _, hi = part.partition('-')
        out += list(range(int(lo), int(hi or lo)+1))
    return kind, out


def course_set(kind, seeds):
    """Courses and their terrains (None for flat) of a course set."""
    courses, terrains = [], []
    for seed in seeds:
        course = synthetic_course(seed) if kind == 'flat' else (
            synthetic_course(seed, steep=.4) if kind == 'steep' else hill_course(seed))
        courses.append(course)
        terrains.append(None if kind == 'flat' else CourseTerrain(course, seed))
    return courses, terrains


def load_controller(kind, checkpoint):
    """dict(kind, brain, cfg, meta, contract); the PD uses the checkpoint's calibration and camera only."""
    from ..train.bptt import load_checkpoint
    meta = torch.load(checkpoint, map_location='cpu', weights_only=True)['visual_brain']
    out = dict(kind=kind, meta=meta, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
    brain, cfg, _ = load_checkpoint(checkpoint, 'cpu')
    out.update(brain=brain if kind == 'brain' else None, cfg=cfg,
               contract=meta.get('fast_motor_tracking') if kind == 'brain' else None)
    if kind == 'brain' and not out['contract']:
        raise ValueError('A brain controller needs a fast-contract checkpoint')
    return out


@torch.no_grad()
def run_batch(controller, profile, courses, terrains, *, pilot_kwargs=None, speed=None, seconds=110., seed=17,
              randomize=.1, dropout=.1, camera_period=.055, camera_latency=.06, delay_steps=3, radius=3.,
              quadratic_drag=.0075, record=False, gate_top=None):
    """fast_motor_tracking.rollout's closed loop (one course per drone) with pilot kwargs and descent scoring.
    ``gate_top``: an optional gate_top.GateTopConfig: each course's checkpoints also get scoring-only arch bars (each
    row gains ``gate_top``: the plane crossings, top-bar hits and passes over an arch); with its ``pass_on_plane`` a
    checkpoint is passed where the drone crosses its arch plane within ``radius`` of the centre (sideways and vertically,
    as a gate is flown through) instead of within ``radius`` of the centre point (up to 3 m before the plane, after which
    a last checkpoint's crossing is flown without guidance). None (the default) leaves the rollout and its rows exactly
    as before."""
    from ..train.fast_motor_tracking import brain_observation
    meta, cfg, brain = controller['meta'], controller['cfg'], controller['brain']
    contract = controller['contract']
    speed = (contract['nominal_speed_mps'] if contract else 6.) if speed is None else speed
    batch = len(courses)
    dt = cfg.brain.dt
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    calibration = meta['calibration']
    sim = IdentifiedSim(profile, calibration, 'cpu', dt)
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
                          calibration=calibration, **(pilot_kwargs or {})) for i in range(batch)]
    scores = [DescentScore(courses[i], camera, terrains[i]) for i in range(batch)]
    tops = None
    if gate_top is not None:
        from .gate_top import GateTop
        tops = [GateTop(courses[i], config=gate_top) for i in range(batch)]
    plane = gate_top is not None and gate_top.pass_on_plane
    before = [None]*batch
    teacher = FastMotorPD(profile, calibration)
    idle = torch.tensor([[-1., 0., 0., 0.]]).repeat(batch, 1)
    queue = deque(idle.clone() for _ in range(delay_steps))
    pending = [deque() for _ in range(batch)]
    detections, captures = [None]*batch, [None]*batch
    next_capture = np.zeros(batch)
    targets = np.zeros(batch, int)
    finish = np.full(batch, np.nan)
    crashed = np.zeros(batch, bool)
    use_brain = controller['kind'] == 'brain'
    if use_brain:
        brain_state = brain.init_state(batch)
        W = brain.inference_matrix()
    steps = int(seconds/dt)
    chatter, chatter_n = np.zeros(batch), np.zeros(batch)
    speed_sum, speed_n = np.zeros(batch), np.zeros(batch)
    previous = None
    trace = [] if record else None
    last_state, support_climbs = [None]*batch, np.zeros(batch, int)
    for k in range(steps):
        now = k*dt
        positions = state.quad.pos.numpy().astype(float)
        quaternions = state.quad.quat.numpy().astype(float)
        velocities = state.quad.vel.numpy().astype(float)
        active = ~crashed & np.isnan(finish)
        if not active.any():
            break
        for i in range(batch):
            histories[i].append(now, positions[i], quaternions[i])
            if plane:
                passed = active[i] and before[i] is not None and tops[i].through(targets[i], before[i], positions[i],
                                                                               radius)
                before[i] = positions[i].copy()
            else:
                passed = active[i] and np.linalg.norm(courses[i][targets[i]]-positions[i]) < radius
            if passed:
                scores[i].passed(targets[i], now, positions[i])
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
        live = ~crashed & np.isnan(finish)
        for i in np.flatnonzero(live):
            scores[i].step(now, dt, targets[i], positions[i], velocities[i], quaternions[i])
        if tops is not None:
            for i in np.flatnonzero(~crashed):
                tops[i].step(positions[i])
        senses = sim.sensors(state)
        requests, feedforward = [], []
        for i in range(batch):
            single = {key: value[i:i+1] for key, value in senses.items()}
            pilots[i].update(single, senses['gyro'][i].numpy(), detections[i], captures[i], now)
            requests.append(pilots[i].velocity_command)
            feedforward.append(pilots[i].feedforward)
            if live[i] and pilots[i].state == 'support_climb' and last_state[i] != 'support_climb':
                support_climbs[i] += 1
            last_state[i] = pilots[i].state
        request = torch.tensor(np.asarray(requests), dtype=torch.float32)
        if use_brain:
            motor = state.quad.motor.mean(-1, keepdim=True)
            obs = brain_observation(meta, senses, motor, cfg.task, torch.zeros(batch, RETINA_DIM), request, contract)
            if k == 0:
                for _ in range(50):
                    _, brain_state, _ = brain(obs, brain_state, W)
            action, brain_state, _ = brain(obs, brain_state, W)
            command = action.cpu().clone()
        else:
            command = teacher.command(senses, request, torch.tensor(np.asarray(feedforward), dtype=torch.float32)).clone()
        for i in range(batch):
            command[i] = torch.as_tensor(pilots[i].command(command[i].numpy()), dtype=torch.float32)
        if now < 1.:
            command = idle.clone()
        if previous is not None:
            chatter += live*(command[:, 1:3]-previous[:, 1:3]).abs().mean(-1).numpy()
            chatter_n += live
        previous = command
        if record:
            trace.append(dict(pos=positions.astype(np.float32), vel=velocities.astype(np.float32),
                              quat=quaternions.astype(np.float32), req=np.asarray(requests, np.float32),
                              cmd=command.numpy().astype(np.float32), target=targets.copy(), live=live.copy(),
                              state=[p.state for p in pilots]))
        queue.append(command)
        state = sim.step(state, queue.popleft())
        velocity = state.quad.vel
        state.quad.vel = velocity-quadratic_drag*velocity.norm(dim=-1, keepdim=True)*velocity*dt
        speed_sum += live*velocity.norm(dim=-1).numpy()
        speed_n += live
        if now <= 1.5:
            state.quad.pos[:, 2] = state.quad.pos[:, 2].clamp_min(0.)
            state.quad.vel[:, 2] = state.quad.vel[:, 2].clamp_min(0.)
            state.quad.crashed[:] = False
        crashed |= state.quad.crashed.numpy() & np.isnan(finish)
    rows = []
    for i in range(batch):
        p = pilots[i]
        rows.append(dict(finished=bool(np.isfinite(finish[i])), crashed=bool(crashed[i]),
                         finish_s=None if not np.isfinite(finish[i]) else round(float(finish[i]), 2),
                         gates=int(targets[i]), of=len(courses[i]),
                         mean_speed=round(float(speed_sum[i]/max(speed_n[i], 1)), 3),
                         stick_chatter=round(float(chatter[i]/max(chatter_n[i], 1)), 5),
                         states={s: round(v, 2) for s, v in p.state_time.items()},
                         support_climbs=int(support_climbs[i]),
                         descent_view=p.descent_view_summary() if hasattr(p, 'descent_view_summary') else None,
                         # the contact-support rule's own onsets (descent view version 2), only when it is declared
                         **({} if getattr(p, 'contact_support', None) is None
                            else dict(contact_support=p.contact_summary())),
                         # the sighted descent's seconds and withheld sink, only when it is declared
                         **({} if getattr(p, 'sighted_descent', None) is None
                            else dict(sighted_descent=p.sighted_summary())),
                         **scores[i].result()))
        if tops is not None:
            rows[-1]['gate_top'] = tops[i].result()
    return rows, trace


def _summary(rows):
    finished = [r for r in rows if r['finished']]
    total = lambda key: sum((r.get(key) or 0) for r in rows)
    return dict(courses=len(rows), finished=len(finished), crashed=sum(r['crashed'] for r in rows),
                contacts=total('contacts'), contact_s=round(total('contact_s'), 2),
                descent_s=round(total('descent_s'), 2), below_view_s=round(total('below_view_s'), 2),
                below_view_fraction=round(total('below_view_s')/max(total('descent_s'), 1e-9), 4),
                high_passes=total('high_passes'),
                mean_finish_s=None if not finished else round(float(np.mean([r['finish_s'] for r in finished])), 2))


def main_run(args):
    torch.set_num_threads(2)
    kind, seeds = parse_set(args.set)
    courses, terrains = course_set(kind, seeds)
    profile_path = Path(args.profile)
    profile = json.loads(profile_path.read_text())
    controller = load_controller(args.controller, args.checkpoint)
    pilot_kwargs, declaration = {}, None
    if args.descent_view:
        from .fast_race_cue import descent_view_config
        text = Path(args.descent_view).read_text(encoding='utf-8')
        declaration = json.loads(text)
        pilot_kwargs['descent_view'] = descent_view_config(declaration)
    begin = time.time()
    rows, _ = run_batch(controller, profile, courses, terrains, pilot_kwargs=pilot_kwargs, seconds=args.seconds,
                        seed=args.sim_seed)
    for seed, row in zip(seeds, rows):
        row['seed'] = seed
    out = dict(controller=args.controller, checkpoint=args.checkpoint,
               checkpoint_sha256=controller['checkpoint_sha256'], profile=str(profile_path),
               profile_sha256=sha256(profile_path), set=args.set, kind=kind, seeds=seeds, seconds=args.seconds,
               sim_seed=args.sim_seed, variant='descent_view' if args.descent_view else 'baseline',
               descent_view=None if declaration is None else dict(
                   path=args.descent_view, version=declaration.get('version'), sha256=declaration.get('sha256'),
                   file_sha256=sha256(args.descent_view)),
               elapsed_s=round(time.time()-begin, 1),
               scope='measured surrogate (IdentifiedSim) on seeded synthetic courses with scoring-only hills; '
                     'development evidence, not flight evidence',
               summary=_summary(rows), courses=rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(out=args.out, elapsed_s=out['elapsed_s'], **out['summary'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--controller', choices=('pd', 'brain'), required=True)
    run.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    run.add_argument('--profile', default=DEFAULT_PROFILE)
    run.add_argument('--set', required=True, help='flat|steep|hill:SEEDS, e.g. hill:6000-6007')
    run.add_argument('--descent-view', default='', help='a descent-view declaration (JSON); omitted = baseline')
    run.add_argument('--seconds', type=float, default=110.)
    run.add_argument('--sim-seed', type=int, default=17)
    run.add_argument('--out', required=True)
    score = sub.add_parser('score')
    score.add_argument('--gates', required=True)
    score.add_argument('--dir', required=True)
    score.add_argument('--json', required=True)
    args = parser.parse_args()
    if args.command == 'run':
        main_run(args)
    else:
        from .descent_gates import main_score
        main_score(args)


if __name__ == '__main__':
    main()
