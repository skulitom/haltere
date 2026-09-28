"""Contact support of the fast pilot (descent-view declaration version 2): scenarios, the detector on logged flights,
and the frozen gates (configs/pilot/contact_support_gates.json). Offline development evidence only, never flight
evidence.

Three kinds of evidence:
- ``resting_scenario``: the round-4 review's synthetic case (review/scenarios.py): the drone rests on a surface (the
  measured velocity is held at (speed, 0, 0), level) with the next ring clipped at the bottom edge. The review's
  version had no motor, so no throttle; here the motor that would fly (the fast PD, or a fast-contract brain) issues
  its throttle for the pilot's request at that state, after a prologue of level free flight (hover thrust, ring
  ahead) that arms the rule. Measured: the support climb's delay after the rest begins.
- ``floor_scenario``: the measured surrogate (IdentifiedSim, nominal drone: no randomisation) with a flat floor: the
  drone flies level, then follows a ring below it onto the floor and slides on it (the floor holds z and removes any
  downward velocity). Measured: the support climb's delay after the touchdown.
- ``detector_on_log``: the rule's own code (FastRaceCue._contact_step) fed a logged flight's recorded motion, attitude,
  issued throttle (command_thr) and logged velocity command, tick by tick as the runner fed the pilot: what the rule
  would have concluded about the motion that was flown. A support climb it starts changes nothing in the log.

The surrogate runs use `haltere.liftoff.descent_rehearsal.run_batch`, and the open-loop replays of whole pilots
`haltere.obstacles.vertical_replay` (--descent-view with a version-2 declaration, --throttle-column command_thr).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .camera_pose import CameraPoseHistory
from .fast_race_cue import ContactSupportConfig, FastRaceCue
from ..vision.camera import Camera, quat_wxyz_to_mat

REPO = Path(__file__).resolve().parents[2]
RUNS = Path('C:/DEV/Haltere/runs/fast-stack-20260923')
PROFILE = 'runs/measured-dynamics-low-speed-20260923/profile.json'
# The runner's pad calibration of the original drone (identical in every fast-stack sidecar and in the fast
# checkpoints' metadata).
CALIBRATION = dict(hover_processed=0.13566076755523682, throttle_scale=0.8, hover_stick_sim=-0.43019758448357537,
                   stick_sign=[-1., 1., 1.], max_rpm=40278.14442235099)
SENSOR = dict(focal_320=100., tilt_deg=30., centre_offset_m=0., missing_gate='zero_goal_neural_search',
              height_invariant=True, gravity_aligned_height=True, raw_retina_active=True, retina_mode='scene_v2')
BELOW = dict(u=.5, v=.99, edge=True)


def _tensor_senses(position, velocity, quaternion, omega=(0., 0., 0.)):
    rotation = quat_wxyz_to_mat(np.asarray(quaternion, float))
    t = lambda x: torch.tensor([list(map(float, x))], dtype=torch.float32)  # noqa: E731
    velocity = np.asarray(velocity, float)
    return dict(pos=t(position), quat=t(quaternion), vel_world=t(velocity), vel_body=t(rotation.T @ velocity),
                gyro=t(omega), gravity_body=t(-rotation[2, :]), altitude=t([position[2]]),
                yaw=t([np.arctan2(rotation[1, 0], rotation[0, 0])]), up=torch.tensor([float(rotation[2, 2])]))


def _load_profile(profile=None):
    path = Path(profile or PROFILE)
    path = path if path.is_absolute() else Path('C:/DEV/Haltere')/path
    return json.loads(path.read_text(encoding='utf-8'))


class _Motor:
    """The fast PD, or a fast-contract brain checkpoint (its own contract), issuing brain-order actions."""

    def __init__(self, kind, checkpoint=None, profile=None):
        from ..brain.motor_baseline import FastMotorPD
        self.kind = kind
        self.profile = profile or _load_profile()
        self.pd = FastMotorPD(self.profile, CALIBRATION)
        self.brain = None
        if kind == 'brain':
            from .descent_rehearsal import load_controller
            c = load_controller('brain', checkpoint)
            self.brain, self.meta, self.cfg, self.contract = c['brain'], c['meta'], c['cfg'], c['contract']
            self.state = self.brain.init_state(1)
            self.W = self.brain.inference_matrix()
            self.warm = False

    @torch.no_grad()
    def act(self, senses, request, feedforward, motor_fraction):
        if self.brain is None:
            return self.pd.command(senses, torch.tensor(request, dtype=torch.float32)[None],
                                   torch.tensor(feedforward, dtype=torch.float32)[None])[0].numpy()
        from ..brain.retina import RETINA_DIM
        from ..train.fast_motor_tracking import brain_observation
        obs = brain_observation(self.meta, senses, torch.tensor([[motor_fraction]], dtype=torch.float32),
                                self.cfg.task, torch.zeros(1, RETINA_DIM),
                                torch.tensor(request, dtype=torch.float32)[None], self.contract)
        if not self.warm:
            for _ in range(50):
                _, self.state, _ = self.brain(obs, self.state, self.W)
            self.warm = True
        action, self.state, _ = self.brain(obs, self.state, self.W)
        return action[0].numpy()


def _pilot(contact_support, descent_view, speed, calibration=CALIBRATION):
    return FastRaceCue(SENSOR, CameraPoseHistory(), speed, reference_speed=speed, calibration=calibration,
                       descent_view=descent_view, contact_support=contact_support)


def _level_ahead_cue(camera):
    pixels, ok = camera.project_body(np.array([[20., 0., 0.]]))
    return dict(u=float(pixels[0, 0]/320), v=float(pixels[0, 1]/180), edge=False)


def resting_scenario(contact_support, speed, *, motor=None, descent_view=None, prologue_s=3.5, rest_s=3., height=5.,
                     dt=.01, latency=.05):
    """The review's resting case with a motor: level free flight at (speed, 0, 0) with the ring ahead for prologue_s
    (hover thrust: the motor's own), then resting on a surface (measured velocity held at (speed, 0, 0), vz 0) with
    the ring clipped at the bottom edge for rest_s. Returns the support-climb onsets after the rest begins (seconds
    after it), the contact rule's onsets, and the command and throttle at the first onset."""
    motor = motor or _Motor('pd')
    pilot = _pilot(contact_support, descent_view, 6.)
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    ahead = _level_ahead_cue(camera)
    q = np.array([1., 0., 0., 0.])
    hover_fraction = float((1/motor.pd.twr)**(1/motor.pd.exponent))
    rows = []
    start = 10.
    for k in range(int(round((prologue_s+rest_s)/dt))):
        now = start+k*dt
        resting = now-start >= prologue_s
        position = np.array([speed*(now-start), 0., height])
        velocity = np.array([speed, 0., 0.])
        senses = _tensor_senses(position, velocity, q)
        pilot.pose_history.append(now, position, q)
        cue = BELOW if resting else ahead
        pilot.update(senses, [0., 0., 0.], dict(race_cue=dict(cue)), now-latency, now)
        action = motor.act(senses, pilot.velocity_command, pilot.feedforward, hover_fraction)
        pilot.command(action)
        rows.append((now-start-prologue_s, pilot.state, float(pilot.velocity_command[2]), float(action[0]),
                     float(pilot.contact_fired) if pilot.contact_support is not None else 0.,
                     float(pilot.contact_unexplained) if pilot.contact_support is not None else float('nan')))
    t = np.array([r[0] for r in rows])
    state = np.array([r[1] for r in rows])
    onsets = np.flatnonzero((state == 'support_climb') & np.r_[True, state[:-1] != 'support_climb'])
    fires = np.flatnonzero(np.array([r[4] for r in rows]) > 0)
    first = next((i for i in onsets if t[i] >= 0), None)
    return dict(speed=speed, motor=motor.kind, onsets_after_rest=[round(float(t[i]), 3) for i in onsets if t[i] >= 0],
                onsets_before_rest=int(sum(t[i] < 0 for i in onsets)),
                contact_onsets=[round(float(t[i]), 3) for i in fires],
                first_delay_s=None if first is None else round(float(t[first]), 3),
                command_vz_at_first=None if first is None else round(rows[first-1][2], 3),
                throttle_at_first=None if first is None else round(rows[first-1][3], 3),
                max_unexplained=round(float(np.nanmax([r[5] for r in rows])), 3)
                if np.isfinite([r[5] for r in rows]).any() else None)


def floor_scenario(contact_support, speed, *, motor=None, descent_view=None, level_s=5., seconds=16., height=2.,
                   dt=.01, latency=.06, period=.055):
    """The measured surrogate (nominal drone) with a flat floor at z = 0: level flight toward a ring ahead at the
    start height for level_s, then a ring far ahead below the floor, so the pilot descends onto the floor and slides
    on it at the pilot's speed. The floor holds z >= 0 and removes downward velocity (a frictionless inelastic
    surface). Returns the first touchdown after level_s (the descent), the support-climb onsets after it and the
    contact rule's; touchdowns during the level phase (a motor settling) are counted apart."""
    from ..sim.identified import IdentifiedSim
    from .fast_rehearsal import hud_marker
    motor = motor or _Motor('pd')
    profile = motor.profile
    sim = IdentifiedSim(profile, CALIBRATION)
    state = sim.hover(1, height)
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    pilot = _pilot(contact_support, descent_view, speed)
    pilot.launching = False            # airborne from the start (the launch rule is not under test)
    from collections import deque
    queue = deque()
    hover = None
    pending, next_capture, detection, capture_time = deque(), 0., None, None
    touchdown, rows, early = None, [], 0
    for k in range(int(round(seconds/dt))):
        now = k*dt
        position = state.quad.pos[0].numpy().astype(float)
        quaternion = state.quad.quat[0].numpy().astype(float)
        pilot.pose_history.append(now, position, quaternion)
        target = (np.array([position[0]+40., 0., height]) if now < level_s
                  else np.array([position[0]+60., 0., -30.]))
        if now >= next_capture:
            cue = hud_marker(camera, target, position, quaternion)
            cue['aim_u'] = cue['u']
            pending.append((now, now+latency, cue))
            next_capture = now+period
        while pending and pending[0][1] <= now:
            capture_time, _, cue = pending.popleft()
            detection = dict(race_cue=cue)
        senses = sim.sensors(state)
        pilot.update(senses, senses['gyro'][0].numpy(), detection, capture_time, now)
        action = motor.act(senses, pilot.velocity_command, pilot.feedforward,
                           float(state.quad.motor.mean()))
        command = torch.as_tensor(pilot.command(action), dtype=torch.float32)[None]
        if hover is None:
            hover = command.clone()
            queue.extend(hover.clone() for _ in range(3))
        queue.append(command)
        state = sim.step(state, queue.popleft())
        if float(state.quad.pos[0, 2]) < 0.:
            state.quad.pos[0, 2] = 0.
            state.quad.vel[0, 2] = state.quad.vel[0, 2].clamp_min(0.)
            state.quad.crashed[:] = False
            if now >= level_s:
                touchdown = now if touchdown is None else touchdown
            else:
                early += 1
        rows.append((now, pilot.state, float(pilot.contact_fired) if pilot.contact_support is not None else 0.,
                     float(state.quad.pos[0, 2]), float(np.hypot(*state.quad.vel[0, :2].numpy()))))
    t = np.array([r[0] for r in rows])
    st = np.array([r[1] for r in rows])
    onsets = np.flatnonzero((st == 'support_climb') & np.r_[True, st[:-1] != 'support_climb'])
    fires = np.flatnonzero(np.array([r[2] for r in rows]) > 0)
    after = [float(t[i]-touchdown) for i in onsets if touchdown is not None and t[i] >= touchdown]
    speed_at = None if touchdown is None else float(rows[int(round(touchdown/dt))][4])
    return dict(speed=speed, motor=motor.kind, touchdown_s=None if touchdown is None else round(touchdown, 2),
                speed_at_touchdown=None if speed_at is None else round(speed_at, 2),
                onsets_after_touchdown=[round(x, 3) for x in after],
                onsets_before_touchdown=int(sum(touchdown is None or t[i] < touchdown for i in onsets)),
                level_phase_floor_ticks=early,
                contact_onsets=[round(float(t[i]), 3) for i in fires],
                first_delay_s=round(after[0], 3) if after else None)


def detector_on_log(flight, contact_support, runs=RUNS):
    """The contact rule's code on a logged flight: per tick, FastRaceCue._contact_step with the recorded velocity and
    attitude, the logged velocity command of the previous tick (cmd_vx..z), the issued throttle of the previous tick
    (command_thr), launching until the recorded height first reaches the pilot's launch_height, on the controller
    clock (capture_time + image_age) as in the runner. Returns per-tick arrays (t = phase)."""
    import pandas as pd
    d = pd.read_csv(Path(runs)/f'{flight}.csv', low_memory=False)
    pilot = _pilot(contact_support, None, 6.)
    keys = ('t', 'now', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'command', 'unexplained', 'gain', 'fire', 'suspect',
            'launching')
    out = {k: [] for k in keys}
    previous_command, previous_throttle, last = None, None, None
    for r in d.itertuples(index=False):
        if np.isfinite(r.capture_time) and np.isfinite(r.image_age):
            now = float(r.capture_time)+float(r.image_age)
        else:
            continue
        if last is not None and now <= last:
            now = last+1e-4
        issued_at, last = last, now
        dt = .01 if issued_at is None else float(np.clip(now-issued_at, 0., .1))
        velocity = np.array([r.vx, r.vy, r.vz], float)
        rotation = quat_wxyz_to_mat(np.array([r.qw, r.qx, r.qy, r.qz], float))
        if r.z >= pilot.config.launch_height:
            pilot.launching = False
        pilot.velocity_command = previous_command
        pilot.issued_throttle = previous_throttle
        pilot._contact_step(now, issued_at, velocity, rotation, dt)
        if pilot.climb_until is not None and now >= pilot.climb_until:
            pilot.climb_until = None
        values = dict(t=float(r.phase), now=now, x=r.x, y=r.y, z=r.z, vx=r.vx, vy=r.vy, vz=r.vz,
                      command=np.nan if previous_command is None else previous_command[2],
                      unexplained=pilot.contact_unexplained, gain=pilot.contact_gain, fire=float(pilot.contact_fired),
                      suspect=float(pilot.contact_since is not None), launching=float(pilot.launching))
        for k in keys:
            out[k].append(values[k])
        cmd = np.array([r.cmd_vx, r.cmd_vy, r.cmd_vz], float)
        previous_command = cmd if np.isfinite(cmd).all() else None
        previous_throttle = float(r.command_thr) if np.isfinite(r.command_thr) else None
    return {k: np.asarray(v) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------
# Frozen gates (configs/pilot/contact_support_gates.json), run and scored after the freeze
# ---------------------------------------------------------------------------------------------
GATES_PATH = REPO/'configs'/'pilot'/'contact_support_gates.json'


def file_sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gates(path=GATES_PATH):
    """The frozen gates and their content hash; refuses an unfrozen or edited file, and gates written for another
    descent-view declaration than this tree's (configs/pilot/descent_view.json)."""
    from .gap_stack import config_sha256
    gates = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = config_sha256(gates)
    if gates.get('frozen') is not True or gates.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    declared = json.loads((REPO/gates['descent_view']['file']).read_text(encoding='utf-8'))
    if (declared.get('version') != gates['descent_view']['version']
            or declared.get('sha256') != gates['descent_view']['sha256']):
        raise ValueError(f'{path} scores descent-view version {gates["descent_view"]["version"]}, not the declaration '
                         'in this tree')
    return gates, digest


def _check_controller(spec):
    if file_sha256(spec['checkpoint']) != spec['sha256']:
        raise ValueError(f"{spec['checkpoint']} is not the checkpoint the gates name")


def run_rest(gates, out):
    """CS_Rest: the resting scenario for every declared motor at every declared speed (JSON per motor; resumable)."""
    from .visual_brain import load_descent_view
    from .fast_race_cue import contact_support_config, descent_view_config
    g = gates['gates']['CS_Rest']
    declaration, _ = load_descent_view(REPO/gates['descent_view']['file'])
    dv, cs = descent_view_config(declaration), contact_support_config(declaration)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in g['motors']:
        path = out/f'rest_{name}.json'
        if path.exists():
            continue
        spec = gates['controllers'][name]
        _check_controller(spec)
        rows = []
        for speed in g['speeds']:
            motor = _Motor(spec['kind'], spec['checkpoint'] if spec['kind'] == 'brain' else None)
            rows.append(resting_scenario(cs, float(speed), motor=motor, descent_view=dv, prologue_s=g['prologue_s'],
                                         rest_s=g['rest_s']))
            print(name, json.dumps(rows[-1]), flush=True)
        path.write_text(json.dumps(dict(motor=name, rows=rows), indent=1), encoding='utf-8')


def surrogate_kwargs(contract, variant='full'):
    """The FastRaceCue kwargs of the surrogate variant, as the runner builds them from this tree's declarations:
    'full' = --obstacle-stack on for the motor contract (lag turn, gap aim, wall pilot with the contract's stopping
    model, vertical guard) + --descent-view on (with its contact support); 'dv' = --descent-view on alone."""
    from . import visual_brain as vb
    from . import fast_race_cue as frc
    from .gap_aim import GapAimConfig
    from .gap_stack import load_gap_pilot
    declaration, _ = vb.load_descent_view()
    kw = dict(descent_view=frc.descent_view_config(declaration))
    contact = frc.contact_support_config(declaration)
    if contact is not None:
        kw['contact_support'] = contact
    if variant == 'full':
        lag, _ = vb.load_lag_turn_declaration(vb.LAG_TURN_DECLARATION)
        wall, _ = vb.load_wall_pilot()
        vertical, _ = vb.load_vertical_guard()
        gap, _ = load_gap_pilot()
        kw.update(lag_turn=frc.lag_turn_for_contract(lag, contract), gap_aim=GapAimConfig.from_dict(gap['pilot']),
                  vertical_guard=frc.vertical_guard_config(vertical), **frc.wall_pilot_configs(wall, contract))
    return kw


def run_surrogate(gates, out, controllers=None):
    """CS_Surrogate: the declared variant on the gate course sets for every declared motor (resumable; one JSON per
    motor and set)."""
    from . import descent_rehearsal as dr
    g = gates['gates']['CS_Surrogate']
    profile_path = Path('C:/DEV/Haltere')/g['profile']
    if file_sha256(profile_path) != g['profile_sha256']:
        raise ValueError('the surrogate profile changed')
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in controllers or g['motors']:
        spec = gates['controllers'][name]
        _check_controller(spec)
        controller = None
        kw = surrogate_kwargs(spec['contract'], g['variant'])
        for set_spec in g['sets']:
            path = out/f"{name}_{set_spec.replace(':', '_')}_{g['variant']}.json"
            if path.exists():
                continue
            if controller is None:
                controller = dr.load_controller(spec['kind'], spec['checkpoint'])
            kind, seeds = dr.parse_set(set_spec)
            courses, terrains = dr.course_set(kind, seeds)
            rows, _ = dr.run_batch(controller, profile, courses, terrains, pilot_kwargs=kw, seconds=g['seconds'],
                                   seed=g['sim_seed'])
            for seed, row in zip(seeds, rows):
                row['seed'] = seed
            path.write_text(json.dumps(dict(controller=name, set=set_spec, variant=g['variant'], courses=rows,
                                            summary=dr._summary(rows)), indent=1), encoding='utf-8')
            print(name, set_spec, json.dumps(dr._summary(rows)),
                  [(r.get('contact_support') or {}).get('onsets') for r in rows], flush=True)


def run_clean(gates, out, runs=RUNS):
    """CS_Clean: the rule's code on every declared log (detector_on_log with this tree's declaration): its onsets
    (support climbs it would start) per log. Logs without the needed columns are listed as skipped."""
    from .visual_brain import load_descent_view
    from .fast_race_cue import contact_support_config
    from .contact_audit import logs_of
    g = gates['gates']['CS_Clean']
    declaration, _ = load_descent_view(REPO/gates['descent_view']['file'])
    cs = contact_support_config(declaration)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out/'clean_detector.json'
    result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for log in logs_of(runs, g['min_frames']):
        if log in result:
            continue
        try:
            a = detector_on_log(log, cs, runs)
        except (AttributeError, KeyError, ValueError) as exc:
            result[log] = dict(skipped=repr(exc)[:200])
            continue
        fires = np.flatnonzero(a['fire'] > 0)
        minutes = round(float(a['t'][-1]-a['t'][0])/60, 3) if len(a['t']) else 0.
        result[log] = dict(ticks=int(len(a['t'])), minutes=minutes,
                           onsets=[dict(t=round(float(a['t'][i]), 3), pos=[round(float(a[k][i]), 2) for k in 'xyz'])
                                   for i in fires],
                           gain=[round(float(np.nanmin(a['gain'])), 4), round(float(np.nanmax(a['gain'])), 4)]
                           if len(a['t']) else None)
        path.write_text(json.dumps(result, indent=1), encoding='utf-8')
        print(log, len(fires), flush=True)
    return result


def _load_npz(prefix, tag, flight):
    return dict(np.load(f'{prefix}_{tag}_{flight}.npz', allow_pickle=False))


def replay_tag(stack, stream, *, dv=None, cthr=False, wp=None, near=True):
    """The vertical_replay file tag of a variant (stack on/shadow/none with the tree's vertical guard, the gap
    samples with near_on_path for the stack, then -cthr, -dv<version>, -wp<version> as vertical_replay.main adds
    them)."""
    from ..obstacles.vertical_replay import variant_tag
    vertical = {'on': 'on', 'shadow': 'shadow', 'none': 'off'}[stack]
    tag = variant_tag(stack, 'off', vertical, stream)
    if near and stack != 'none':
        tag += '-nop'
    if cthr:
        tag += '-cthr'
    if dv is not None:
        tag += f'-dv{dv}'
    if wp is not None:
        tag += f'-wp{wp}'
    return tag


def identity_pairs(flights, stream_flights):
    """(flight, this tree's tag, the baseline tree's tag) of the identity gates: the default pilot, the shadow stack,
    the stack with wall-pilot version 4, and the stack with wall-pilot version 4 and descent-view version 1 (this
    tree's kept declarations against the baseline tree's own)."""
    pairs = []
    for f in flights:
        s = f in stream_flights
        pairs += [(f, replay_tag('none', s), replay_tag('none', s)),
                  (f, replay_tag('shadow', s), replay_tag('shadow', s)),
                  (f, replay_tag('on', s, wp=4), replay_tag('on', s)),
                  (f, replay_tag('on', s, dv=1, wp=4), replay_tag('on', s, dv=1))]
    return pairs


def score_identity(prefix, baseline, flights, stream_flights):
    """Bitwise equality of the command arrays (vertical_replay.COMMAND_KEYS) of every identity pair."""
    from ..obstacles.vertical_replay import identical
    rows = []
    for flight, mine, base in identity_pairs(flights, stream_flights):
        try:
            same, keys = identical(_load_npz(prefix, mine, flight), _load_npz(baseline, base, flight))
        except FileNotFoundError as exc:
            same, keys = None, str(exc)
        rows.append(dict(flight=flight, variant=mine, baseline_variant=base, identical=same,
                         differing=[k for k, v in keys.items() if not v] if isinstance(keys, dict) else keys))
    return dict(pairs=rows, identical=sum(bool(r['identical']) for r in rows), of=len(rows),
                passed=bool(rows) and all(r['identical'] for r in rows))


def score(gates, out, prefix, baseline, audit_results, runs=RUNS):
    """Every CS gate from the run outputs (`out`), the replay files of this tree (`prefix`) and of the baseline tree
    (`baseline`), and the frozen contact audit's results over every log (`audit_results`: its audit_results.json)."""
    from .contact_audit import _log_end
    g = gates['gates']
    result = {}
    gi = g['CS_Identity']
    result['CS_Identity'] = score_identity(prefix, baseline, gi['flights'], set(gi['stream_flights']))
    gr = g['CS_Rest']
    rest = {}
    for name in gr['motors']:
        rows = json.loads((Path(out)/f'rest_{name}.json').read_text(encoding='utf-8'))['rows']
        rest[name] = [dict(speed=r['speed'], first_delay_s=r['first_delay_s'], before_rest=r['onsets_before_rest'],
                           onsets_after_rest=r['onsets_after_rest'], contact_onsets=r['contact_onsets'])
                      for r in rows]
    delays = [r['first_delay_s'] if r['first_delay_s'] is not None else np.inf
              for rows in rest.values() for r in rows]
    worst = max(delays) if delays else np.inf
    result['CS_Rest'] = dict(motors=rest, worst_first_delay_s=None if not np.isfinite(worst) else worst,
                             passed=bool(np.isfinite(worst) and worst <= gr['max_delay_s']
                                         and all(r['before_rest'] == 0 for rows in rest.values() for r in rows)))
    g4 = g['CS_R402']
    a = _load_npz(prefix, g4['tag'], g4['flight'])
    fires = a['t'][a['contact_fire'] > 0]
    inside = [round(float(t), 3) for t in fires if g4['window'][0] <= t <= g4['window'][1]]
    result['CS_R402'] = dict(onsets=[round(float(t), 3) for t in fires], in_window=inside, passed=bool(inside))
    gs = g['CS_Surrogate']
    rows, same, compared = [], 0, 0
    for name in gs['motors']:
        for set_spec in gs['sets']:
            stem = set_spec.replace(':', '_')
            mine = json.loads((Path(out)/f"{name}_{stem}_{gs['variant']}.json").read_text(encoding='utf-8'))
            ref_path = Path(gs['reference_dir'])/f"{name}_{stem}_{gs['reference_variant']}.json"
            ref = json.loads(ref_path.read_text(encoding='utf-8'))
            onsets = [(c.get('contact_support') or {}).get('onsets', 0) for c in mine['courses']]
            differing = []
            for m, r in zip(mine['courses'], ref['courses']):
                keys = [k for k in gs['compare_keys'] if m.get(k) != r.get(k)]
                compared += 1
                same += not keys
                if keys:
                    differing.append(dict(seed=m['seed'], keys=keys))
            rows.append(dict(motor=name, set=set_spec, contact_onsets=int(sum(onsets)), summary=mine['summary'],
                             reference_summary=ref['summary'], reference_sha256=file_sha256(ref_path),
                             differing_courses=differing,
                             gains=[(c.get('contact_support') or {}).get('gain') for c in mine['courses']]))
    total = sum(r['contact_onsets'] for r in rows)
    result['CS_Surrogate'] = dict(rows=rows, contact_onsets=total, courses_identical=same, courses=compared,
                                  passed=total == 0 and same == compared)
    gc = g['CS_Clean']
    clean = json.loads((Path(out)/'clean_detector.json').read_text(encoding='utf-8'))
    audit = {r['log']: r for r in json.loads(Path(audit_results).read_text(encoding='utf-8'))['results']}
    per_log, outside_total, inside_total, minutes = {}, 0, 0, 0.
    for log, entry in clean.items():
        if 'skipped' in entry:
            per_log[log] = entry
            continue
        windows = [(c['t_start']-gc['margin_s'], c['t_end']+gc['margin_s'])
                   for c in (audit.get(log) or {}).get('contacts', [])]
        end, impact = _log_end(log, runs)
        if impact:
            windows.append((end-gc['terminal_s'], end+1.))
        outside = [o for o in entry['onsets'] if not any(lo <= o['t'] <= hi for lo, hi in windows)]
        per_log[log] = dict(onsets=len(entry['onsets']), outside=outside, minutes=entry['minutes'], gain=entry['gain'])
        outside_total += len(outside)
        inside_total += len(entry['onsets'])-len(outside)
        minutes += entry['minutes']
    result['CS_Clean'] = dict(logs=per_log, onsets_outside_contacts=outside_total, onsets_inside_contacts=inside_total,
                              minutes=round(minutes, 2), skipped=sorted(k for k, v in per_log.items() if 'skipped' in v),
                              passed=outside_total <= gc['max_outside'])
    result['passed'] = {k: result[k]['passed'] for k in ('CS_Identity', 'CS_Rest', 'CS_R402', 'CS_Surrogate',
                                                         'CS_Clean')}
    return result


def main(argv=None):
    import argparse
    import os
    import sys
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description='Run or score the frozen contact-support gates')
    parser.add_argument('command', choices=['rest', 'surrogate', 'clean', 'score'])
    parser.add_argument('--gates', default=str(GATES_PATH))
    parser.add_argument('--out', required=True)
    parser.add_argument('--controllers', nargs='*', default=None)
    parser.add_argument('--prefix', help="this tree's replay prefix (score)")
    parser.add_argument('--baseline', help="the baseline tree's replay prefix (score)")
    parser.add_argument('--audit', help="the frozen contact audit's audit_results.json (score)")
    parser.add_argument('--json', help='where to write the scores (score)')
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    gates, digest = load_gates(args.gates)
    if args.command == 'rest':
        run_rest(gates, args.out)
    elif args.command == 'surrogate':
        run_surrogate(gates, args.out, args.controllers)
    elif args.command == 'clean':
        run_clean(gates, args.out)
    else:
        result = score(gates, args.out, args.prefix, args.baseline, args.audit)
        result.update(gates_sha256=digest, gates_version=gates['version'])
        Path(args.json).write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
        print(json.dumps(result['passed']))


if __name__ == '__main__':
    main()
