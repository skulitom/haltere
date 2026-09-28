"""Surrogate scenarios and live-window replays for the motor assist (development tool, no flight).

The closed loop is `descent_rehearsal.run_batch`'s (the fast-brain development gate's): the measured-drone surrogate
`IdentifiedSim` with 10% per-drone randomisation, synthetic HUD marker (camera period 0.055 s, latency 0.06 s, 10%
dropout), three-tick command delay and quadratic drag, flown by a fast-contract brain checkpoint or the fast PD under the
fast race-cue pilot with any pilot keyword arguments (the round-4 stack, the descent view, the motor assist). Added here:

- ``hairpin``: a scripted wall scenario like the Minus Two hairpin. From the ground (the gate harness's launch: one-second
  arming hold, the pilot's launch climb) the drone flies a straight leg north to ring R1, turns right by `turn_deg` to
  the arch R2 `arch_m` further, behind which a wall stands `wall_m` ahead across the R2 leg (a vertical plane segment,
  `wall_half_m` to each side); the next ring R3 lies `side_m` to the left, `back_m` before the wall, then R4 further left
  (a hairpin). Rings at `height_m`. The marker disappears for `dropout_s` after the R2 pass (the live Minus hairpin: 0.45 s).
  Synthetic looming at 18 Hz, received 0.085 s after capture: the time to contact of the travel ray (horizontal
  velocity direction) with the wall segment, when the ray meets it within 20 m and within 55 deg of the heading
  (below_fraction 0.5, ttc_lower = ttc: a wall, as the frozen motor-assist gates v1 score it; with
  ``live_wall_samples=True``, report only, below_fraction None and no lower-surface TTC, as the live Minus Two wall
  samples: below_fraction was missing in 78-100% of the short-TTC samples of the round-3/4 logs and the lower window
  explained 0-7% of the brains' ones); otherwise no evidence. Scoring only: the wall is a plane the drone
  contacts when its centre comes within 0.3 m (arm radius) inside the segment, a floor at 0 and a ceiling at
  `ceiling_m` (the Minus Two garage: ~2.2 m).
- ``accelerate``: stop-then-accelerate. From the ground (launch as above) ring R1 lies `near_m` ahead at `height_m`: the
  drone reaches it slowly, low; R2 lies `ring_m` beyond it, `bearing_deg` off the R1 leg, at the same height, then R3 20 m
  further: the height lost while the brain turns and accelerates from low speed to cruise near the floor, under a
  scoring-only ceiling at `ceiling_m` (the Minus Two garage: ~2.2 m).
- ``live_window``: a logged live window replayed in the surrogate from the logged state (brain warmed on the recorded
  inputs; haltere.train.brake_gates.Flight): the pilot's own logged request and, with a motor assist, the assisted
  request computed from the surrogate's measured velocity and attitude with the binding caps the pilot had at each
  logged state (an open-loop replay of the logged pilot through the same stack: haltere.obstacles.vertical_replay).

Nothing here is flight evidence: no Liftoff terrain, walls, contact physics or HUD rules.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch

from ..brain.motor_baseline import FastMotorPD
from ..brain.retina import RETINA_DIM
from ..sim.identified import IdentifiedSim
from ..vision.camera import Camera
from .camera_pose import CameraPoseHistory
from .fast_race_cue import FastRaceCue
from .fast_rehearsal import hud_marker

LOOMING_PERIOD = 1/18.
LOOMING_DELAY = .085
LOOMING_RANGE = 20.
LOOMING_FOV_DEG = 55.
ARM_M = .3


def _yaw_quat(yaw):
    return torch.tensor([np.cos(yaw/2), 0., 0., np.sin(yaw/2)], dtype=torch.float32)


def hairpin_scenario(turn_deg=90., wall_m=2.1, side_m=6., back_m=1., leg_m=20., arch_m=7., height_m=.8,
                     ceiling_m=2.2, wall_half_m=6., dropout_s=.45):
    """A Minus-Two-like hairpin (see the module docstring): start pose, rings, one wall segment, ceiling."""
    start = np.zeros(3)
    north = np.array([0., 1., 0.])
    r1 = np.array([0., leg_m, height_m])
    a = np.radians(turn_deg)
    d2 = np.array([np.sin(a), np.cos(a), 0.])            # right turn by turn_deg from north
    r2 = r1+d2*arch_m
    wall_c = r2+d2*wall_m
    left = np.array([-d2[1], d2[0], 0.])
    wall = (wall_c[:2]-left[:2]*wall_half_m, wall_c[:2]+left[:2]*wall_half_m)
    r3 = wall_c-d2*back_m+left*side_m
    r4 = r3+left*15.
    return dict(kind='hairpin', start=start, yaw=np.pi/2, rings=[r1, r2, r3, r4], walls=[wall], ceiling=ceiling_m,
                dropout_after=1, dropout_s=dropout_s, wall_ring=1, height=height_m,
                params=dict(turn_deg=turn_deg, wall_m=wall_m, side_m=side_m, back_m=back_m, leg_m=leg_m, arch_m=arch_m,
                            height_m=height_m, ceiling_m=ceiling_m, wall_half_m=wall_half_m, dropout_s=dropout_s))


def accelerate_scenario(height_m=.8, bearing_deg=0., ring_m=25., near_m=3., ceiling_m=2.2):
    """Stop-then-accelerate (see the module docstring); a scoring-only ceiling at `ceiling_m` (None: none)."""
    start = np.zeros(3)
    r1 = np.array([near_m, 0., height_m])
    b = np.radians(bearing_deg)
    d = np.array([np.cos(b), np.sin(b), 0.])
    r2 = r1+d*ring_m
    return dict(kind='accelerate', start=start, yaw=0., rings=[r1, r2, r2+d*20.], walls=[], ceiling=ceiling_m,
                dropout_after=None, dropout_s=0., wall_ring=None, height=height_m, accelerate_ring=0,
                params=dict(height_m=height_m, bearing_deg=bearing_deg, ring_m=ring_m, near_m=near_m,
                            ceiling_m=ceiling_m))


def _wall_hit(position, velocity, heading, walls):
    """(distance along the horizontal travel ray to the nearest wall segment it meets in view, closing speed) or None."""
    speed = float(np.hypot(velocity[0], velocity[1]))
    if speed <= .5:
        return None
    ray = velocity[:2]/speed
    if np.degrees(np.arccos(np.clip(ray @ heading, -1, 1))) > LOOMING_FOV_DEG:
        return None
    best = None
    for a, b in walls:
        e = b-a
        m = np.array([[ray[0], -e[0]], [ray[1], -e[1]]])
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        s, u = np.linalg.solve(m, a-position[:2])
        if s > 0 and 0 <= u <= 1 and s <= LOOMING_RANGE and (best is None or s < best):
            best = float(s)
    return None if best is None else (best, speed)


def _wall_gap(position, walls):
    """Signed distance of the drone centre to each wall plane within its segment (min), or inf."""
    best = np.inf
    for a, b in walls:
        e = b-a
        length = float(np.linalg.norm(e))
        u = float((position[:2]-a) @ e)/length**2
        if not 0 <= u <= 1:
            continue
        normal = np.array([-e[1], e[0]])/length
        best = min(best, abs(float((position[:2]-a) @ normal)))
    return best


@contextmanager
def horizontal_governor_ray():
    """REPORT-ONLY diagnostic, not part of any declared rule: while active, the looming governor's cap acts along the
    horizontal part of its ray (the pilot's cap, turn-first and the motor assist see that ray). A stand-in for the
    round-4b contact fix (a clearance brake along an up-tilted ray pushed the vertical request down, minus-fast6-r4-02);
    that fix is another branch's and may differ. Used only to show how the motor assist composes with such a fix."""
    from . import fast_race_cue as frc
    original = frc.TtcClearanceGovernor.limits

    def limits(self, *args, **kwargs):
        cap, ray, climb = original(self, *args, **kwargs)
        if ray is not None:
            flat = np.array([ray[0], ray[1], 0.], dtype=float)
            size = float(np.linalg.norm(flat))
            ray = flat/size if size > 1e-6 else ray
        return cap, ray, climb
    frc.TtcClearanceGovernor.limits = limits
    try:
        yield
    finally:
        frc.TtcClearanceGovernor.limits = original


def run_scenarios(controller, profile, scenarios, *, flat_governor_ray=False, **kwargs):
    """One drone per scenario, closed loop (descent_rehearsal.run_batch's loop), synthetic wall looming and scoring.
    Returns rows (per scenario) and an optional trace. ``flat_governor_ray``: the report-only diagnostic
    `horizontal_governor_ray` (never a gate)."""
    if not flat_governor_ray:
        return _run_scenarios(controller, profile, scenarios, **kwargs)
    with horizontal_governor_ray():
        return _run_scenarios(controller, profile, scenarios, **kwargs)


@torch.no_grad()
def _run_scenarios(controller, profile, scenarios, *, pilot_kwargs=None, speed=None, seconds=14., seed=17,
                   randomize=.1, dropout=.1, camera_period=.055, camera_latency=.06, delay_steps=3, radius=1.5,
                   quadratic_drag=.0075, record=False, live_wall_samples=False):
    from ..train.fast_motor_tracking import brain_observation
    meta, cfg, brain = controller['meta'], controller['cfg'], controller['brain']
    contract = controller['contract']
    speed = (contract['nominal_speed_mps'] if contract else 6.) if speed is None else speed
    batch = len(scenarios)
    dt = cfg.brain.dt
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    calibration = meta['calibration']
    sim = IdentifiedSim(profile, calibration, 'cpu', dt)
    sim.randomize(batch, randomize, generator)
    state = sim.hover(batch, 0.)
    for i, sc in enumerate(scenarios):
        state.quad.pos[i] = torch.tensor(sc['start'], dtype=torch.float32)
        state.quad.quat[i] = _yaw_quat(sc['yaw'])
    state.quad.vel[:] = 0.
    sensor = meta['gate_sensor']
    camera = Camera(320, 180, sensor['focal_320'], sensor['tilt_deg'])
    yaw_axis = profile['axes']['yaw']
    curve = (yaw_axis['coefficient_deg_s'], yaw_axis['super_rate'], yaw_axis['expo'])
    histories = [CameraPoseHistory() for _ in range(batch)]
    pilots = [FastRaceCue(sensor, histories[i], speed, reference_speed=speed, yaw_curve=curve,
                          calibration=calibration, **(pilot_kwargs or {})) for i in range(batch)]
    teacher = FastMotorPD(profile, calibration)
    idle = torch.tensor([[-1., 0., 0., 0.]]).repeat(batch, 1)
    queue = deque(idle.clone() for _ in range(delay_steps))
    pending = [deque() for _ in range(batch)]
    looming = [deque() for _ in range(batch)]
    detections, captures = [None]*batch, [None]*batch
    next_capture, next_loom = np.zeros(batch), np.zeros(batch)
    targets = np.zeros(batch, int)
    passes = [[] for _ in range(batch)]
    finish = np.full(batch, np.nan)
    crashed = np.zeros(batch, bool)
    use_brain = controller['kind'] == 'brain'
    if use_brain:
        brain_state = brain.init_state(batch)
        W = brain.inference_matrix()
    z0 = np.array([sc['height'] for sc in scenarios])
    low, high = np.full(batch, np.inf), np.zeros(batch)
    airborne = np.zeros(batch, bool)
    wall_gap = np.full(batch, np.inf)
    wall_contact = np.zeros(batch, bool)
    wall_speed = np.zeros(batch)
    contact_t = np.full(batch, np.nan)
    ceiling_contact, floor_contact = np.zeros(batch, bool), np.zeros(batch, bool)
    chatter, chatter_n = np.zeros(batch), np.zeros(batch)
    previous = None
    min_z_after = [[] for _ in range(batch)]
    trace = [] if record else None
    steps = int(seconds/dt)
    for k in range(steps):
        now = k*dt
        positions = state.quad.pos.numpy().astype(float)
        quaternions = state.quad.quat.numpy().astype(float)
        velocities = state.quad.vel.numpy().astype(float)
        active = ~crashed & np.isnan(finish) & ~wall_contact
        if not active.any():
            break
        for i in range(batch):
            histories[i].append(now, positions[i], quaternions[i])
            sc = scenarios[i]
            if not active[i]:
                continue
            if np.linalg.norm(sc['rings'][targets[i]]-positions[i]) < radius:
                passes[i].append(now)
                targets[i] += 1
                if targets[i] == len(sc['rings']):
                    finish[i] = now
                    continue
            if now >= next_capture[i]:
                hidden = (sc['dropout_after'] is not None and len(passes[i]) > sc['dropout_after']
                          and now-passes[i][sc['dropout_after']] < sc['dropout_s'])
                cue = None if hidden or rng.random() <= dropout else hud_marker(camera, sc['rings'][targets[i]],
                                                                                 positions[i], quaternions[i])
                if cue is not None:
                    cue['aim_u'] = cue['u']
                pending[i].append((now, now+camera_latency, cue))
                next_capture[i] = now+camera_period
            while pending[i] and pending[i][0][1] <= now:
                captures[i], _, cue = pending[i].popleft()
                detections[i] = dict(race_cue=cue) if cue is not None else None
            if sc['walls'] and now >= next_loom[i]:
                w, x, y, z = quaternions[i]
                heading = np.array([1-2*(y*y+z*z), 2*(x*y+w*z)])
                heading /= max(np.linalg.norm(heading), 1e-9)
                hit = _wall_hit(positions[i], velocities[i], heading, sc['walls'])
                sample = (dict(time=now, ttc=None, distance=None, below_fraction=None, ttc_lower=None) if hit is None
                          else dict(time=now, ttc=hit[0]/hit[1], distance=hit[0],
                                    below_fraction=None if live_wall_samples else .5,
                                    ttc_lower=None if live_wall_samples else hit[0]/hit[1]))
                looming[i].append((now+LOOMING_DELAY, sample))
                next_loom[i] = now+LOOMING_PERIOD
        senses = sim.sensors(state)
        requests = []
        feedforward = []
        for i in range(batch):
            single = {key: value[i:i+1] for key, value in senses.items()}
            clearance = None
            while looming[i] and looming[i][0][0] <= now:
                _, clearance = looming[i].popleft()
            pilots[i].update(single, senses['gyro'][i].numpy(), detections[i], captures[i], now,
                             **(dict(clearance=clearance) if clearance is not None else {}))
            requests.append(pilots[i].velocity_command)
            feedforward.append(pilots[i].feedforward)
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
        live = ~crashed & np.isnan(finish) & ~wall_contact
        if previous is not None:
            chatter += live*(command[:, 1:3]-previous[:, 1:3]).abs().mean(-1).numpy()
            chatter_n += live
        previous = command
        if record:
            trace.append(dict(pos=positions.astype(np.float32), vel=velocities.astype(np.float32),
                              req=np.asarray(requests, np.float32), target=targets.copy(),
                              state=[p.state for p in pilots],
                              direction=np.asarray([p.direction if p.direction is not None else np.full(3, np.nan)
                                                    for p in pilots], np.float32),
                              cue=[None if d is None else d.get('race_cue') for d in detections],
                              quat=quaternions.astype(np.float32),
                              own=np.asarray([p.pilot_command if getattr(p, 'pilot_command', None) is not None
                                              else p.velocity_command for p in pilots], np.float32)))
        queue.append(command)
        state = sim.step(state, queue.popleft())
        velocity = state.quad.vel
        state.quad.vel = velocity-quadratic_drag*velocity.norm(dim=-1, keepdim=True)*velocity*dt
        positions = state.quad.pos.numpy().astype(float)
        velocities = state.quad.vel.numpy().astype(float)
        if now <= 1.5:
            state.quad.pos[:, 2] = state.quad.pos[:, 2].clamp_min(0.)
            state.quad.vel[:, 2] = state.quad.vel[:, 2].clamp_min(0.)
            state.quad.crashed[:] = False
            positions = state.quad.pos.numpy().astype(float)
        for i in np.flatnonzero(live):
            sc = scenarios[i]
            # heights are scored once the drone first passes the first ring (after its launch)
            airborne[i] |= len(passes[i]) > 0
            if airborne[i]:
                low[i], high[i] = min(low[i], positions[i, 2]), max(high[i], positions[i, 2])
            if sc['ceiling'] is not None and positions[i, 2] > sc['ceiling']-.1:
                ceiling_contact[i] = True
            if airborne[i] and positions[i, 2] < .05:
                floor_contact[i] = True
            if sc['walls']:
                gap = _wall_gap(positions[i], sc['walls'])
                wall_gap[i] = min(wall_gap[i], gap)
                if gap < ARM_M:
                    wall_contact[i] = True
                    contact_t[i] = now
                    wall_speed[i] = float(np.hypot(velocities[i, 0], velocities[i, 1]))
        crashed |= state.quad.crashed.numpy() & np.isnan(finish)
    rows = []
    for i, sc in enumerate(scenarios):
        p = pilots[i]
        wr = sc['wall_ring']
        rows.append(dict(kind=sc['kind'], params=sc['params'], finished=bool(np.isfinite(finish[i])),
                         crashed=bool(crashed[i]), rings=int(targets[i]), of=len(sc['rings']),
                         finish_s=None if not np.isfinite(finish[i]) else round(float(finish[i]), 2),
                         pass_s=[round(t, 2) for t in passes[i]],
                         height_loss_m=None if not np.isfinite(low[i]) else round(float(z0[i]-low[i]), 3),
                         min_height_m=None if not np.isfinite(low[i]) else round(float(low[i]), 3),
                         max_height_m=round(float(high[i]), 3),
                         floor_contact=bool(floor_contact[i]), ceiling_contact=bool(ceiling_contact[i]),
                         wall_contact=bool(wall_contact[i]),
                         wall_contact_speed=round(float(wall_speed[i]), 3) if wall_contact[i] else None,
                         wall_contact_s=None if not np.isfinite(contact_t[i]) else round(float(contact_t[i]), 2),
                         min_wall_gap_m=None if not np.isfinite(wall_gap[i]) else round(float(wall_gap[i]), 3),
                         past_wall_ring=None if wr is None else bool(targets[i] > wr+1),
                         stick_chatter=round(float(chatter[i]/max(chatter_n[i], 1)), 5),
                         states={s: round(v, 2) for s, v in p.state_time.items()},
                         turn_first=None if p.turn_first is None else dict(p.turn_first_counts),
                         motor_assist=p.motor_assist_summary() if hasattr(p, 'motor_assist_summary') else None))
    return rows, trace


def hairpin_set(spec):
    """Hairpin scenarios of a declared set: the product of the listed parameter values."""
    import itertools
    keys = sorted(spec)
    return [hairpin_scenario(**dict(zip(keys, values))) for values in itertools.product(*(spec[k] for k in keys))]


def accelerate_set(spec):
    import itertools
    keys = sorted(spec)
    return [accelerate_scenario(**dict(zip(keys, values))) for values in itertools.product(*(spec[k] for k in keys))]


# ---------------------------------------------------------------------------------------------
# Live windows (semi-closed loop): logged requests, assisted from the surrogate's measured state
# ---------------------------------------------------------------------------------------------
def window_sources(flight, tree, declaration, runs):
    """Per logged tick: (the pilot's own request, binding caps, pilot state) from an open-loop replay of
    the logged pilot through the flown stack with the motor assist (vertical_replay.replay)."""
    from ..obstacles.vertical_replay import replay
    side = json.loads((Path(runs)/f'{flight}.json').read_text(encoding='utf-8'))
    flown_on = (side.get('obstacle_stack') or {}).get('mode') == 'on'
    descent = side['pilot_assistance'].get('descent_view')
    dv = None
    if descent is not None:
        from .fast_race_cue import DescentViewConfig
        dv = DescentViewConfig(**descent['parameters'])
    sources = []
    arrays, pilot, info = replay(flight, tree, runs, stack='on' if flown_on else 'none', vertical=None,
                                 near_on_path=True, descent_view=dv, motor_assist=declaration, sources=sources)
    return arrays, sources, info


@torch.no_grad()
def live_window(ctl, profile, flight, t0, horizon_s, *, assist=None, arrays=None, sources=None, warm_s=3.):
    """Replay a logged window in the surrogate (see the module docstring). ``assist``: a MotorAssistConfig applied to
    the logged pilot's own request with the surrogate's measured velocity/attitude and the replayed binding caps
    (``arrays``/``sources`` from window_sources, aligned with the flight's logged ticks). Returns per-tick position,
    velocity and the request flown."""
    from ..train.brake_gates import DT
    k0 = flight.index(t0)
    H = min(int(round(horizon_s/DT)), flight.impact_after(k0)-k0-1)
    sim = IdentifiedSim(profile, ctl.calibration, 'cpu', DT)
    sim.randomize(1, 0.)
    s = flight.state_fn(k0)(sim, 1)
    state = flight.warm(ctl, k0, warm_s) if ctl.kind == 'brain' else None
    pd = FastMotorPD(profile, ctl.calibration)
    pd.reset()
    if ctl.kind == 'pd':
        flight.pd_warm(profile, ctl, k0)(pd)
    q = deque(torch.tensor(flight.cmds[k0-3+i], dtype=torch.float32)[None] for i in range(3))
    cal = ctl.calibration
    helper = None
    if assist is not None:
        helper = FastRaceCue(dict(focal_320=100., tilt_deg=30.), CameraPoseHistory(), 6., motor_assist=assist)
        helper.launching = False
        # the replay's tick index for each logged row (the replay skips rows without a capture/image age)
        valid = np.flatnonzero(np.isfinite(flight.d.capture_time.to_numpy(float))
                               & np.isfinite(flight.d.image_age.to_numpy(float)))
        index = {int(k): j for j, k in enumerate(valid)}
    pos, vel, req = np.zeros((H, 3)), np.zeros((H, 3)), np.zeros((H, 3))
    for j in range(H):
        k = k0+j
        senses = sim.sensors(s)
        r = np.array(flight.req[k], float)
        if helper is not None and k in index:
            jj = index[k]
            own = np.array([arrays['assist_pilot_vx'][jj], arrays['assist_pilot_vy'][jj], arrays['assist_pilot_vz'][jj]])
            if np.isfinite(own).all():
                r = own
            srcs, pstate = sources[jj]
            velocity = s.quad.vel[0].numpy().astype(float)
            r = helper._motor_assist(r, velocity, DT, pstate, srcs)
        rt = torch.tensor(r[None], dtype=torch.float32)
        if ctl.kind == 'pd':
            a = pd.command(senses, rt, torch.tensor(flight.ff[k:k+1], dtype=torch.float32), dt=DT)
        else:
            a, state = ctl.act(senses, s.quad.motor.mean(-1, keepdim=True), rt, state)
        command = a.clone()
        thr = cal['hover_processed']+cal['throttle_scale']*(float(command[0, 0])-cal['hover_stick_sim'])
        room = float(np.sqrt(max(.97**2-min(thr*thr, .97**2), 0.)))
        command[0, 3] = float(np.clip(flight.cmds[k, 3], -room, room))
        q.append(command)
        s = sim.step(s, q.popleft())
        v = s.quad.vel
        s.quad.vel = v-.0075*v.norm(dim=-1, keepdim=True)*v*DT
        s.quad.crashed[:] = False
        pos[j], vel[j], req[j] = s.quad.pos[0].numpy(), s.quad.vel[0].numpy(), r
    return dict(k0=k0, H=H, pos=pos, vel=vel, req=req)
