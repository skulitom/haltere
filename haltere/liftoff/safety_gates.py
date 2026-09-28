"""Round-5 safety gates (configs/pilot/safety_gates.json): the ceiling guard's any_climb (wall-pilot declaration
version 6) and contact support version 3 (descent-view declaration version 3). Offline development and held-out
evidence, never flight evidence.

Kinds of evidence (the gates file says which are development and which held out):
- synthetic scenarios (``scenarios``): the round-4b review's ceiling case on the TTC governor, a pilot climbing by
  each climb source under overhead samples, and turn-first during a support climb;
- the resting scenario with each motor (``rest``; contact_support_eval.resting_scenario);
- the motor-assist harness's hairpin and accelerate sets on sim seeds the design never ran (``arming``;
  motor_assist_eval.run_scenarios: 10% per-drone randomisation of the thrust curve, no ground reaction after the launch,
  so every contact-rule onset there is false);
- the descent surrogate on course seeds the design never ran (``surrogate``; descent_rehearsal.run_batch);
- the contact rule's own code on every logged flight, versions 2 and 3 (``detector``;
  contact_support_eval.detector_on_log), scored against the frozen contact audit (development logs:
  docs/experiments/contact_audit_v1_contacts.json; the held-out round-4b logs: their own .contact-audit.json);
- open-loop replays (haltere/obstacles/vertical_replay.py, this tree and a git archive of m4b; the variants the gates
  file lists).

usage:
  python -m haltere.liftoff.safety_gates scenarios|rest|arming|surrogate|detector --out DIR [--controllers ...]
  python haltere/obstacles/vertical_replay.py ...   (each variant of the gates file, --out PREFIX, this tree / m4b)
  python -m haltere.liftoff.safety_gates score --out DIR --prefix THIS --baseline M4B --json SCORES
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
GATES_PATH = REPO/'configs'/'pilot'/'safety_gates.json'
RUNS = Path('C:/DEV/Haltere/runs/fast-stack-20260923')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gates(path=GATES_PATH):
    """The frozen gates and their content hash; refuses an unfrozen or edited file, gates written for other
    declarations than this tree's, and kept declarations that changed."""
    from .gap_stack import config_sha256
    gates = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = config_sha256(gates)
    if gates.get('frozen') is not True or gates.get('sha256') != digest:
        raise ValueError(f'{path} is not frozen or changed after the freeze: gates are scored only when frozen')
    for name, spec in {**gates['declarations'], **gates['kept_declarations']}.items():
        declared = json.loads((REPO/spec['file']).read_text(encoding='utf-8'))
        if declared.get('version') != spec['version'] or declared.get('sha256') != spec['sha256']:
            raise ValueError(f'{path} scores {name} version {spec["version"]} ({spec["sha256"][:12]}), not the '
                             f'declaration {spec["file"]} in this tree')
    return gates, digest


def _declaration(gates, name):
    spec = {**gates['declarations'], **gates['kept_declarations']}[name]
    return json.loads((REPO/spec['file']).read_text(encoding='utf-8'))


def stack_kwargs(gates, contract, variant):
    """FastRaceCue kwargs of a declared stack variant: 'm4b' (wall pilot 5, descent view 2 with contact support 2),
    'm5' (wall pilot 6, descent view 3 with contact support 3 on), 'm5_csoff' (contact support off),
    'm5_csshadow' (contact support in shadow). Every variant has the obstacle stack of the runner (lag turn, gap aim,
    vertical guard, wall pilot with the contract's stopping model) and the view rule."""
    from . import fast_race_cue as frc
    from . import visual_brain as vb
    from .gap_aim import GapAimConfig
    from .gap_stack import GAP_PILOT_PATH, load_gap_pilot
    lag, _ = vb.load_lag_turn_declaration(vb.LAG_TURN_DECLARATION)
    gap, _ = load_gap_pilot(GAP_PILOT_PATH)
    vertical, _ = vb.load_vertical_guard()
    old = variant == 'm4b'
    wall = _declaration(gates, 'wall_pilot_v5' if old else 'wall_pilot')
    view = _declaration(gates, 'descent_view_v2' if old else 'descent_view')
    kw = dict(lag_turn=frc.lag_turn_for_contract(lag, contract), gap_aim=GapAimConfig.from_dict(gap['pilot']),
              vertical_guard=frc.vertical_guard_config(vertical), **frc.wall_pilot_configs(wall, contract),
              descent_view=frc.descent_view_config(view))
    if variant != 'm5_csoff':
        kw['contact_support'] = frc.contact_support_config(view)
    if variant == 'm5_csshadow':
        kw['contact_apply'] = False
    return kw


# ---------------------------------------------------------------------------------------------
# Synthetic scenarios
# ---------------------------------------------------------------------------------------------
def _senses(position, velocity, yaw=0., omega=(0., 0., 0.)):
    from ..vision.camera import quat_wxyz_to_mat
    q = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
    rotation = quat_wxyz_to_mat(q)
    t = lambda x: torch.tensor([list(map(float, x))], dtype=torch.float32)  # noqa: E731
    velocity = np.asarray(velocity, float)
    return dict(pos=t(position), quat=t(q), vel_world=t(velocity), vel_body=t(rotation.T @ velocity), gyro=t(omega),
                gravity_body=t([0., 0., -1.]), altitude=t([position[2]]), yaw=t([yaw]), up=torch.ones(1))


def ceiling_governor_case(ceiling, *, terrain_first, below=.1, lower=None):
    """The round-4b review's case (scratchpad m4b/review/ceiling_gate.py): the TTC governor (the flown TTC policy, the
    vertical guard of this tree and `ceiling`) receives overhead samples at 20 Hz (TTC 0.9 -> 0.3 s, `below` /
    `lower`) while the drone rises at 1 m/s at 3 m/s ahead; with terrain_first a terrain climb is already running
    (case B), without it the drone climbs by another source (case A). Returns the overhead engagements and the first
    time (s after the first sample) at which the whole vertical request is bounded."""
    from . import fast_race_cue as frc
    from . import visual_brain as vb
    vertical, _ = vb.load_vertical_guard()
    g = frc.TtcClearanceGovernor(frc.TtcClearanceConfig(), ceiling=ceiling, vertical=frc.vertical_guard_config(vertical))
    pos, vel = np.array([0., 0., 1.4]), np.array([3., 0., 1.])
    ray = vel/np.linalg.norm(vel)
    now, first = 10., None
    if terrain_first:
        g.climb, g.climb_hold_until, g.terrain_at, g.climb_base = 1., now+1., now, 1.3
    for k in range(40):
        now += .01
        pos = pos+vel*.01
        if k % 5 == 0:
            ttc = max(.3, .9-.02*k)
            g.ingest(now-.06, ttc, ttc*float(np.linalg.norm(vel)), below, pos-vel*.06, ray, float(np.linalg.norm(vel)),
                     received=now, ttc_lower=lower)
        g.limits(pos, vel, now, .01, 3.5)
        if first is None and g.vertical_cap is not None:
            first = round(now-10.01, 3)
    return dict(engagements=int(g.counts.get('overhead_engagements', 0)), first_bound_s=first)


def _cue_toward(body_point):
    """The HUD cue of a body-frame point, projected through the calibrated camera (level attitude, yaw 0)."""
    from ..vision.camera import Camera
    from .contact_support_eval import SENSOR
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    pixels, _ = camera.project_body(np.array([body_point], dtype=float))
    u, v = float(pixels[0, 0]/320), float(pixels[0, 1]/180)
    return dict(u=u, v=v, edge=False, aim_u=u)


RING_AHEAD = _cue_toward([20., 0., 0.])     # a level ring ahead
RING_ABOVE = _cue_toward([20., 0., 4.])     # a ring ahead and 11 deg above (the pilot's own climb, about 1.2 m/s)


# When the ceiling samples start in ceiling_pilot_case, per climb source: once that climb is established.
CEILING_SAMPLES_AFTER = dict(support=.45, contact=.45, search=1.2, pilot=.6)


def ceiling_pilot_case(wall, source, *, seconds=2.5, tau=.15):
    """A pilot of the obstacle stack (the runner's lag turn is not needed here: the wall-pilot rules and the vertical
    guard of this tree, `wall` a wall-pilot declaration) at 3 m/s and 1.2 m, whose measured vertical speed follows its
    vertical request with time constant `tau`. At 0.2 s a climb starts by `source`: 'support' (a support climb, 1 m/s
    for 0.6 s, injected as the older rules start it), 'contact' (the same as contact support starts it), 'search' (the
    ring lost: the search climb, 0.5 m/s), 'pilot' (a ring above: the pilot's own climb toward it, state cue). From
    CEILING_SAMPLES_AFTER[source] the looming samples read a ceiling ahead of the rising path (20 Hz, TTC 0.9 -> 0.3 s,
    below_fraction None, lower-surface TTC 1.23 s: the live Minus garage pattern). Returns the largest vertical request
    before the ceiling samples, the time from the first ceiling sample until the request is at most 0.05 m/s (None:
    never), and the largest request after that."""
    overhead_after = CEILING_SAMPLES_AFTER[source]
    from . import fast_race_cue as frc
    from .camera_pose import CameraPoseHistory
    from .contact_support_eval import SENSOR
    configs = frc.wall_pilot_configs(wall, 'fast_velocity_pd_v1')
    from . import visual_brain as vb
    vertical, _ = vb.load_vertical_guard()
    history = CameraPoseHistory()
    pilot = frc.FastRaceCue(SENSOR, history, 6., reference_speed=6., vertical_guard=frc.vertical_guard_config(vertical),
                            **configs)
    z, vz, start = 1.2, 0., 10.
    first_sample = level_at = None
    before, after = -np.inf, -np.inf
    for k in range(int(round(seconds/.01))):
        now = start+k*.01
        s = _senses((3.*k*.01, 0., z), (3., 0., vz))
        history.append(now, [3.*k*.01, 0., z], s['quat'][0].numpy())
        cue = RING_AHEAD
        if source == 'search' and k*.01 >= .2:
            cue = None
        elif source == 'pilot' and k*.01 >= .2:
            cue = RING_ABOVE
        if source in ('support', 'contact') and abs(k*.01-.2) < .005:
            pilot.climb_until, pilot.climb_source = now+pilot.config.support_climb_s, source
        clearance = None
        if k*.01 >= overhead_after and k % 5 == 0:
            ttc = max(.3, .9-.03*(k-int(overhead_after*100))/5)
            clearance = dict(time=now-.06, ttc=ttc, distance=ttc*3., below_fraction=None, ttc_lower=1.23)
            first_sample = now if first_sample is None else first_sample
        elif k % 5 == 0:
            clearance = dict(time=now-.06, ttc=None, distance=None, below_fraction=None, ttc_lower=None)
        pilot.update(s, [0., 0., 0.], None if cue is None else dict(race_cue=dict(cue)), now-.05, now,
                     clearance=clearance)
        request = float(pilot.velocity_command[2])
        if first_sample is None:
            before = max(before, request)
        else:
            if level_at is None and request <= .05:
                level_at = round(now-first_sample, 3)
            if level_at is not None:
                after = max(after, request)
        vz += (request-vz)*(1-np.exp(-.01/tau))
        z += vz*.01
    return dict(source=source, max_request_before=round(before, 3), level_after_s=level_at,
                max_request_after_level=None if level_at is None else round(after, 3),
                engagements=int(pilot.clearance.counts.get('overhead_engagements', 0)))


def turn_first_case(wall, contact, source):
    """Turn-first during a support climb: a pilot braked to a stand-off at a wall ahead (+x) with the ring clamped at
    the left edge engages turn-first; 0.1 s later a support climb of `source` ('contact': the contact rule's, 'support':
    the older rules') is injected as each rule starts it. Returns whether the episode survived the climb, how it ended,
    and the largest request speed toward the wall during the climb."""
    from . import fast_race_cue as frc
    from .camera_pose import CameraPoseHistory
    from .contact_support_eval import CALIBRATION, SENSOR
    configs = frc.wall_pilot_configs(wall, 'fast_velocity_pd_v1')
    history = CameraPoseHistory()
    pilot = frc.FastRaceCue(SENSOR, history, 6., reference_speed=6., calibration=CALIBRATION,
                            contact_support=contact, **configs)
    ahead = RING_AHEAD
    left = dict(u=.01, v=.5, edge=True, aim_u=.01)
    now, speed = 10., 6.
    for k in range(200):                     # cruise
        now = 10.+k*.01
        s = _senses((0., 0., 2.), (6., 0., 0.))
        history.append(now, [0., 0., 2.], s['quat'][0].numpy())
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(ahead)), now-.05, now)
    for k in range(120):                     # brake to a stand-off at a wall
        now += .01
        clearance = None
        if k % 5 == 0 and speed > 1.5:
            clearance = dict(time=now-.08, ttc=.45, distance=.45*speed, below_fraction=.5, ttc_lower=.45)
        s = _senses((0., 0., 2.), (speed, 0., 0.))
        history.append(now, [0., 0., 2.], s['quat'][0].numpy())
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(ahead)), now-.05, now, clearance=clearance)
        speed = max(.3, speed-8.*.01)
    ray = np.asarray(pilot.clearance.cap_ray, float)[:2]
    ray = ray/max(np.linalg.norm(ray), 1e-9)
    engaged, toward, alive = False, -np.inf, []
    for k in range(100):                     # side clamp: turn first; a support climb at 0.1 s
        now += .01
        if k == 10:
            pilot.climb_until, pilot.climb_source = now+pilot.config.support_climb_s, source
        s = _senses((0., 0., 2.), (.3, 0., 0.))
        history.append(now, [0., 0., 2.], s['quat'][0].numpy())
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(left)), now-.05, now)
        engaged |= pilot.turn_first_active and k < 10
        if 10 < k < 10+int(pilot.config.support_climb_s*100):
            alive.append(pilot.turn_first_active)
            toward = max(toward, float(pilot.velocity_command[:2] @ ray))
    return dict(source=source, engaged_before=bool(engaged), active_through_climb=bool(alive) and all(alive),
                ends=dict(pilot.turn_first_counts), max_toward_wall_mps=round(toward, 3))


def run_scenarios(gates, out):
    from . import fast_race_cue as frc
    wall6, wall5 = _declaration(gates, 'wall_pilot'), _declaration(gates, 'wall_pilot_v5')
    c6 = frc.wall_pilot_configs(wall6, 'fast_velocity_pd_v1')['ceiling_guard']
    c5 = frc.wall_pilot_configs(wall5, 'fast_velocity_pd_v1')['ceiling_guard']
    result = dict(governor={}, pilot={}, turn_first={})
    for name, ceiling in (('v5', c5), ('v6', c6)):
        for case, terrain in (('A', False), ('B', True)):
            for kind, kw in (('above', dict(below=.1)), ('live', dict(below=None, lower=1.23)),
                             ('explained', dict(below=None, lower=.2))):
                result['governor'][f'{name}_{case}_{kind}'] = ceiling_governor_case(ceiling, terrain_first=terrain, **kw)
        for source in gates['gates']['SG_Ceiling_Pilot']['sources']:
            result['pilot'][f'{name}_{source}'] = ceiling_pilot_case(wall6 if name == 'v6' else wall5, source)
    v3 = frc.contact_support_config(_declaration(gates, 'descent_view'))
    v2 = frc.contact_support_config(_declaration(gates, 'descent_view_v2'))
    for name, contact in (('v2', v2), ('v3', v3)):
        for source in ('contact', 'support'):
            result['turn_first'][f'{name}_{source}'] = turn_first_case(wall6, contact, source)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out)/'scenarios.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))
    return result


# ---------------------------------------------------------------------------------------------
# Motors: rest, arming (motor-assist harness), surrogate
# ---------------------------------------------------------------------------------------------
def _controller_spec(gates, name):
    spec = gates['controllers'][name]
    if file_sha256(spec['checkpoint']) != spec['sha256']:
        raise ValueError(f"{spec['checkpoint']} is not the checkpoint the gates name")
    return spec


def run_rest(gates, out, controllers=None):
    """SG_Contact_Rest: contact_support_eval.resting_scenario with descent view 3 for every declared motor and speed."""
    from . import fast_race_cue as frc
    from .contact_support_eval import _Motor, resting_scenario
    g = gates['gates']['SG_Contact_Rest']
    view = _declaration(gates, 'descent_view')
    dv, cs = frc.descent_view_config(view), frc.contact_support_config(view)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in controllers or g['motors']:
        path = out/f'rest_{name}.json'
        if path.exists():
            continue
        spec = _controller_spec(gates, name)
        rows = []
        for speed in g['speeds']:
            motor = _Motor(spec['kind'], spec['checkpoint'] if spec['kind'] == 'brain' else None)
            rows.append(resting_scenario(cs, float(speed), motor=motor, descent_view=dv, prologue_s=g['prologue_s'],
                                         rest_s=g['rest_s']))
            print(name, json.dumps(rows[-1]), flush=True)
        path.write_text(json.dumps(dict(motor=name, rows=rows), indent=1), encoding='utf-8')


def run_arming(gates, out, controllers=None):
    """SG_Contact_Arming: the motor-assist harness's hairpin and accelerate sets on the declared fresh sim seeds, per
    motor and stack variant (resumable: one JSON per motor, seed, part and variant)."""
    from . import motor_assist_eval as mae
    from .descent_rehearsal import load_controller
    g = gates['gates']['SG_Contact_Arming']
    spec_sets = json.loads((REPO/g['sets_from']).read_text(encoding='utf-8'))
    profile_path = Path(g['profile'])
    if file_sha256(profile_path) != g['profile_sha256']:
        raise ValueError('the surrogate profile changed')
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in controllers or g['motors']:
        spec = _controller_spec(gates, name)
        controller = None
        for seed in g['sim_seeds']:
            for part in g['parts']:
                for variant in g['variants']:
                    path = out/f'arming_{name}_{seed}_{part}_{variant}.json'
                    if path.exists():
                        continue
                    if controller is None:
                        controller = load_controller(spec['kind'], spec['checkpoint'])
                    scenarios = (mae.accelerate_set if part == 'accelerate' else mae.hairpin_set)(spec_sets[part]['set'])
                    rows, _ = mae.run_scenarios(controller, profile, scenarios,
                                                pilot_kwargs=stack_kwargs(gates, spec['contract'], variant),
                                                seconds=spec_sets[part]['seconds'], seed=seed)
                    path.write_text(json.dumps(dict(motor=name, seed=seed, part=part, variant=variant, rows=rows),
                                               indent=1, default=float), encoding='utf-8')
                    print(name, seed, part, variant, [(r.get('contact_support') or {}).get('onsets') for r in rows],
                          'ceiling', sum(r['ceiling_contact'] for r in rows), flush=True)


def run_surrogate(gates, out, controllers=None):
    """SG_Contact_Surrogate: the descent surrogate on the declared fresh course seeds, per motor and variant."""
    from . import descent_rehearsal as dr
    g = gates['gates']['SG_Contact_Surrogate']
    profile_path = Path(g['profile'])
    if file_sha256(profile_path) != g['profile_sha256']:
        raise ValueError('the surrogate profile changed')
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in controllers or g['motors']:
        spec = _controller_spec(gates, name)
        controller = None
        for set_spec in g['sets']:
            for variant in g['variants']:
                path = out/f"surrogate_{name}_{set_spec.replace(':', '_')}_{variant}.json"
                if path.exists():
                    continue
                if controller is None:
                    controller = dr.load_controller(spec['kind'], spec['checkpoint'])
                kind, seeds = dr.parse_set(set_spec)
                courses, terrains = dr.course_set(kind, seeds)
                rows, _ = dr.run_batch(controller, profile, courses, terrains,
                                       pilot_kwargs=stack_kwargs(gates, spec['contract'], variant),
                                       seconds=g['seconds'], seed=g['sim_seed'])
                for seed, row in zip(seeds, rows):
                    row['seed'] = seed
                path.write_text(json.dumps(dict(motor=name, set=set_spec, variant=variant, courses=rows), indent=1),
                                encoding='utf-8')
                print(name, set_spec, variant, [(r.get('contact_support') or {}).get('onsets') for r in rows],
                      flush=True)


# ---------------------------------------------------------------------------------------------
# The contact rule on the logged flights
# ---------------------------------------------------------------------------------------------
def _runs_ge(mask, now, hold):
    out, start = [], None
    for i in range(len(mask)+1):
        if i < len(mask) and mask[i]:
            start = i if start is None else start
        elif start is not None:
            if now[i-1]-now[start] >= hold:
                out.append((start, i))
            start = None
    return out


def run_detector(gates, out, runs=RUNS):
    """Versions 2 and 3 of the contact rule on every logged flight (contact_support_eval.detector_on_log): per log its
    onsets (logged command) and the command-agnostic potential runs (unexplained >= unexplained_on for >= hold_s while
    armed, not excluded: a support climb had the pilot asked to sink), with their times (log phase)."""
    from . import fast_race_cue as frc
    from .contact_audit import logs_of
    from .contact_support_eval import detector_on_log
    g = gates['gates']['SG_Contact_Clean']
    rules = dict(v2=frc.contact_support_config(_declaration(gates, 'descent_view_v2')),
                 v3=frc.contact_support_config(_declaration(gates, 'descent_view')))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out/'detector.json'
    result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for log in logs_of(runs, g['min_frames']):
        if log in result:
            continue
        entry = {}
        for name, cs in rules.items():
            try:
                a = detector_on_log(log, cs, runs)
            except (AttributeError, KeyError, ValueError) as exc:
                entry = dict(skipped=repr(exc)[:200])
                break
            t, now = a['t'], a['now']
            high = np.isfinite(a['unexplained']) & (a['unexplained'] >= cs.unexplained_on) & (a['armed'] > 0)
            entry[name] = dict(
                fires=[round(float(t[i]), 3) for i in np.flatnonzero(a['fire'] > 0)],
                potential=[[round(float(t[s]), 3), round(float(t[e-1]), 3)] for s, e in _runs_ge(high, now, cs.hold_s)],
                armed_at=None if not (a['armed'] > 0).any() else round(float(now[np.argmax(a['armed'] > 0)]-now[0]), 3),
                gain=[round(float(np.nanmin(a['gain'])), 4), round(float(np.nanmax(a['gain'])), 4)],
                excluded_s=round(float(np.sum(a['excluded'] > 0))*.01, 2),
                minutes=round(float(t[-1]-t[0])/60, 3))
        result[log] = entry
        path.write_text(json.dumps(result, indent=1), encoding='utf-8')
        print(log, {k: (v['fires'][:6], len(v['potential'])) for k, v in entry.items() if isinstance(v, dict)},
              flush=True)
    return result


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------
def _load(prefix, tag, flight):
    return dict(np.load(f'{prefix}_{tag}_{flight}.npz', allow_pickle=False))


def replay_tag(gates, variant, flight):
    """The vertical_replay file tag of a declared variant for a flight: its tag with {stream} replaced by -stream for
    the flights replayed with the offline looming stream."""
    stream = '-stream' if flight in gates['replays']['stream_flights'] else ''
    return gates['replays']['variants'][variant]['tag'].replace('{stream}', stream)


def _audit_contacts(gates, runs=RUNS):
    dev = json.loads((REPO/gates['audit']['development']['file']).read_text(encoding='utf-8'))
    contacts = {r['log']: r for r in dev['results']}
    for log, spec in gates['audit']['held_out'].items():
        d = json.loads((Path(runs)/spec['file']).read_text(encoding='utf-8'))
        for r in d['results']:
            contacts[r['log']] = r
    return contacts


def _windows(entry, log, margin, terminal, runs=RUNS):
    """The audit's contacts of a log widened by margin, and the last `terminal` seconds of a run that ended on an
    impact (as contact_support_gates CS_Clean)."""
    from .contact_audit import _log_end
    w = [(c['t_start']-margin, c['t_end']+margin) for c in (entry or {}).get('contacts', [])]
    end, impact = _log_end(log, runs)
    if impact:
        w.append((end-terminal, end+1.))
    return w


def score(gates, out, prefix, baseline, runs=RUNS):
    from ..obstacles.vertical_replay import identical
    g = gates['gates']
    out = Path(out)
    result = {}
    held_out = set(gates['held_out']['logs'])

    # identity: default behaviour with the new rules off (and in shadow) is m4b's, bit for bit
    gi = g['SG_Identity']
    pairs = []
    for flight in gi['flights']:
        for mine, base, where in gi['pairs']:
            other = baseline if where == 'baseline' else prefix
            try:
                same, keys = identical(_load(prefix, replay_tag(gates, mine, flight), flight),
                                       _load(other, replay_tag(gates, base, flight), flight))
            except FileNotFoundError as exc:
                same, keys = None, str(exc)
            pairs.append(dict(flight=flight, variant=mine, against=base, tree=where, identical=same,
                              differing=[k for k, v in keys.items() if not v] if isinstance(keys, dict) else keys))
    result['SG_Identity'] = dict(pairs=pairs, identical=sum(bool(p['identical']) for p in pairs), of=len(pairs),
                                 passed=bool(pairs) and all(p['identical'] for p in pairs))

    scen = json.loads((out/'scenarios.json').read_text(encoding='utf-8'))
    gg = scen['governor']
    gc = g['SG_Ceiling_Governor']
    ok = (all(gg[f'v6_A_{k}']['engagements'] >= 1 and gg[f'v6_A_{k}']['first_bound_s'] is not None
              and gg[f'v6_A_{k}']['first_bound_s'] <= gc['max_bound_s'] for k in ('above', 'live'))
          and all(gg[f'v6_B_{k}'] == gg[f'v5_B_{k}'] for k in ('above', 'live', 'explained'))
          and gg['v6_A_explained']['engagements'] == 0)
    result['SG_Ceiling_Governor'] = dict(cases=gg, passed=bool(ok))

    gp = g['SG_Ceiling_Pilot']
    pc = scen['pilot']
    ok = all(pc[f'v6_{s}']['engagements'] >= 1 and pc[f'v6_{s}']['level_after_s'] is not None
             and pc[f'v6_{s}']['level_after_s'] <= gp['max_level_s']
             and pc[f'v6_{s}']['max_request_after_level'] <= gp['max_after'] for s in gp['sources'])
    result['SG_Ceiling_Pilot'] = dict(cases=pc, passed=bool(ok))

    gt = g['SG_TurnFirst']
    tf = scen['turn_first']
    ok = (tf['v3_contact']['engaged_before'] and tf['v3_contact']['active_through_climb']
          and tf['v3_contact']['max_toward_wall_mps'] <= gt['max_toward_wall_mps']
          and tf['v3_support']['ends'].get('handoff', 0) >= 1 and tf['v2_contact']['ends'].get('handoff', 0) >= 1)
    result['SG_TurnFirst'] = dict(cases=tf, passed=bool(ok))

    # the ceiling guard on the replays: the Pine mound climb and quiet away from ceilings
    gm = g['SG_Ceiling_Pine_Mound']
    a6 = _load(prefix, replay_tag(gates, gm['variant'], gm['flight']), gm['flight'])
    a5 = _load(prefix, replay_tag(gates, gm['against'], gm['flight']), gm['flight'])
    window = (a6['t'] >= gm['window'][0]) & (a6['t'] <= gm['window'][1])
    same = bool(np.array_equal(a6['cvz'][window], a5['cvz'][window]))
    result['SG_Ceiling_Pine_Mound'] = dict(identical=same, max_request=round(float(np.max(a6['cvz'][window])), 3),
                                           passed=same)
    gq = g['SG_Ceiling_Quiet']
    rows, worst = {}, 0.
    for flight in gq['flights']:
        a6 = _load(prefix, replay_tag(gates, gq['variant'], flight), flight)
        a5 = _load(prefix, replay_tag(gates, gq['against'], flight), flight)
        t = a6['t']
        dt = np.diff(t, prepend=t[0])
        lowered = a6['cvz'] < a5['cvz']-gq['report_mps']
        minutes = max(float(t[-1]-t[0])/60, 1e-9)
        per_min = float(dt[lowered].sum())/minutes
        rows[flight] = dict(lowered_s=round(float(dt[lowered].sum()), 2), per_min=round(per_min, 3),
                            max_lowered=round(float(np.max(a5['cvz']-a6['cvz'], initial=0.)), 3),
                            held_out=flight in held_out)
        worst = max(worst, per_min)
    result['SG_Ceiling_Quiet'] = dict(flights=rows, worst_per_min=round(worst, 3),
                                      passed=worst <= gq['max_lowered_s_per_min'])
    gmr = g['SG_Ceiling_Minus_Report']
    report = {}
    for flight in gmr['flights']:
        a6 = _load(prefix, replay_tag(gates, gmr['variant'], flight), flight)
        a5 = _load(prefix, replay_tag(gates, gmr['against'], flight), flight)
        dt = np.diff(a6['t'], prepend=a6['t'][0])
        lowered = a6['cvz'] < a5['cvz']-gq['report_mps']
        changed = ((np.abs(a6['cvz']-a5['cvz']) > 1e-9) | (np.abs(a6['cvx']-a5['cvx']) > 1e-9)
                   | (np.abs(a6['cvy']-a5['cvy']) > 1e-9))
        report[flight] = dict(lowered_s=round(float(dt[lowered].sum()), 2), changed_s=round(float(dt[changed].sum()), 2),
                              max_lowered=round(float(np.max(a5['cvz']-a6['cvz'], initial=0.)), 3),
                              first_lowered_t=None if not lowered.any() else round(float(a6['t'][lowered][0]), 2),
                              held_out=flight in held_out)
    result['SG_Ceiling_Minus_Report'] = dict(flights=report)

    # contact support on the scenarios and motors
    gr = g['SG_Contact_Rest']
    rest, delays, arm_ok, before = {}, [], True, 0
    for name in gr['motors']:
        rows_ = json.loads((out/f'rest_{name}.json').read_text(encoding='utf-8'))['rows']
        rest[name] = [dict(speed=r['speed'], first_delay_s=r['first_delay_s'], armed_at_s=r.get('armed_at_s'),
                           before_rest=r['onsets_before_rest']) for r in rows_]
        for r in rows_:
            delays.append(np.inf if r['first_delay_s'] is None else r['first_delay_s'])
            before += r['onsets_before_rest']
            arm_ok &= r.get('armed_at_s') is not None and r['armed_at_s'] <= gr['max_armed_at_s']
    worst = max(delays) if delays else np.inf
    result['SG_Contact_Rest'] = dict(motors=rest, worst_first_delay_s=None if not np.isfinite(worst) else worst,
                                     passed=bool(np.isfinite(worst) and worst <= gr['max_delay_s'] and before == 0
                                                 and arm_ok))
    g4 = g['SG_Contact_R402']
    a = _load(prefix, replay_tag(gates, g4['variant'], g4['flight']), g4['flight'])
    fires = [round(float(x), 3) for x in a['t'][a['contact_fire'] > 0]]
    inside = [x for x in fires if g4['window'][0] <= x <= g4['window'][1]]
    a_sh = _load(prefix, replay_tag(gates, g4['shadow_variant'], g4['flight']), g4['flight'])
    shadow = [round(float(x), 3) for x in a_sh['t'][a_sh['contact_fire'] > 0]]
    result['SG_Contact_R402'] = dict(onsets=fires, in_window=inside, shadow_marks=shadow,
                                     passed=bool(inside) and any(g4['window'][0] <= x <= g4['window'][1]
                                                                for x in shadow))

    det = json.loads((out/'detector.json').read_text(encoding='utf-8'))
    audit = _audit_contacts(gates, runs)
    gd = g['SG_Contact_Detection']
    per, missing, late = [], [], []
    for log, entry in det.items():
        if 'skipped' in entry:
            continue
        for c in (audit.get(log) or {}).get('contacts', []):
            if c.get('kind') == 'impact':
                continue
            lo, hi = c['t_start'], c['t_end']+gd['after_s']
            f2 = [x for x in entry['v2']['fires'] if lo <= x <= hi]
            f3 = [x for x in entry['v3']['fires'] if lo <= x <= hi]
            if not f2:
                continue
            row = dict(log=log, t=c['t_start'], v2=round(f2[0]-lo, 3), v3=None if not f3 else round(f3[0]-lo, 3),
                       held_out=log in held_out)
            per.append(row)
            if not f3:
                missing.append(row)
            elif f3[0]-f2[0] > gd['max_later_s']:
                late.append(row)
    result['SG_Contact_Detection'] = dict(contacts=per, v2_detected=len(per), v3_missing=missing, v3_late=late,
                                          passed=not missing and not late)
    gcl = g['SG_Contact_Clean']
    outside, minutes, logs = [], 0., 0
    for log, entry in det.items():
        if 'skipped' in entry:
            continue
        end = float(entry['v3']['minutes'])*60
        w = _windows(audit.get(log), log, gcl['margin_s'], gcl['terminal_s'], runs)
        for x in entry['v3']['fires']:
            if not any(lo <= x <= hi for lo, hi in w):
                outside.append(dict(log=log, t=x, held_out=log in held_out))
        minutes += end/60
        logs += 1
    result['SG_Contact_Clean'] = dict(outside=outside, logs=logs, minutes=round(minutes, 2),
                                      passed=len(outside) <= gcl['max_outside'])
    gf = g['SG_Contact_Flare']
    flare = {}
    for log in gf['logs']:
        entry = det[log]
        w = _windows(audit.get(log), log, gcl['margin_s'], gcl['terminal_s'], runs)
        pick = lambda runs_: [r for r in runs_ if not any(lo <= r[0] <= hi for lo, hi in w)]  # noqa: E731
        flare[log] = dict(v2=pick(entry['v2']['potential']), v3=pick(entry['v3']['potential']))
    result['SG_Contact_Flare'] = dict(logs=flare, v3_runs=sum(len(v['v3']) for v in flare.values()),
                                      v2_runs=sum(len(v['v2']) for v in flare.values()),
                                      passed=sum(len(v['v3']) for v in flare.values()) <= gf['max_v3_runs'])

    ga = g['SG_Contact_Arming']
    arming, onsets, ceiling = {}, 0, {}
    for name in ga['motors']:
        for seed in ga['sim_seeds']:
            for part in ga['parts']:
                for variant in ga['variants']:
                    rows_ = json.loads((out/f'arming_{name}_{seed}_{part}_{variant}.json').read_text(
                        encoding='utf-8'))['rows']
                    n = sum((r.get('contact_support') or {}).get('onsets', 0) for r in rows_)
                    c = sum(r['ceiling_contact'] for r in rows_)
                    arming[f'{name}_{seed}_{part}_{variant}'] = dict(
                        onsets=n, ceiling=c, wall=sum(r['wall_contact'] for r in rows_),
                        floor=sum(r['floor_contact'] for r in rows_), finished=sum(r['finished'] for r in rows_),
                        armed_at=[(r.get('contact_support') or {}).get('armed_at_s') for r in rows_])
                    if variant == 'm5':
                        onsets += n
                    ceiling[(name, variant)] = ceiling.get((name, variant), 0)+c
    ceiling_ok = all(ceiling[(name, 'm5')] <= ceiling[(name, 'm5_csoff')] for name in ga['motors'])
    result['SG_Contact_Arming'] = dict(runs=arming, m5_onsets=onsets,
                                       ceiling={f'{k[0]}_{k[1]}': v for k, v in ceiling.items()},
                                       passed=onsets == 0 and ceiling_ok)
    gs = g['SG_Contact_Surrogate']
    surrogate, onsets, same, compared = {}, 0, 0, 0
    for name in gs['motors']:
        for set_spec in gs['sets']:
            stem = set_spec.replace(':', '_')
            on = json.loads((out/f'surrogate_{name}_{stem}_m5.json').read_text(encoding='utf-8'))['courses']
            off = json.loads((out/f'surrogate_{name}_{stem}_m5_csoff.json').read_text(encoding='utf-8'))['courses']
            n = sum((c.get('contact_support') or {}).get('onsets', 0) for c in on)
            differing = [c['seed'] for c, d in zip(on, off)
                         if any(c.get(k) != d.get(k) for k in gs['compare_keys'])]
            onsets += n
            compared += len(on)
            same += len(on)-len(differing)
            surrogate[f'{name}_{stem}'] = dict(
                onsets=n, differing=differing, finished=sum(c['finished'] for c in on),
                crashed=sum(c['crashed'] for c in on), gains=[(c.get('contact_support') or {}).get('gain') for c in on],
                armed_at=[(c.get('contact_support') or {}).get('armed_at_s') for c in on])
    result['SG_Contact_Surrogate'] = dict(sets=surrogate, onsets=onsets, identical=same, of=compared,
                                          passed=onsets == 0 and same == compared)
    result['passed'] = {k: v['passed'] for k, v in result.items() if isinstance(v, dict) and 'passed' in v}
    return result


def main(argv=None):
    import argparse
    import os
    import sys
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description='Run or score the frozen round-5 safety gates')
    parser.add_argument('command', choices=['scenarios', 'rest', 'arming', 'surrogate', 'detector', 'score'])
    parser.add_argument('--gates', default=str(GATES_PATH))
    parser.add_argument('--out', required=True)
    parser.add_argument('--controllers', nargs='*', default=None)
    parser.add_argument('--prefix', help="this tree's replay prefix (score)")
    parser.add_argument('--baseline', help="the m4b tree's replay prefix (score)")
    parser.add_argument('--json', help='where to write the scores (score)')
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    gates, digest = load_gates(args.gates)
    if args.command == 'scenarios':
        run_scenarios(gates, args.out)
    elif args.command == 'rest':
        run_rest(gates, args.out, args.controllers)
    elif args.command == 'arming':
        run_arming(gates, args.out, args.controllers)
    elif args.command == 'surrogate':
        run_surrogate(gates, args.out, args.controllers)
    elif args.command == 'detector':
        run_detector(gates, args.out)
    else:
        result = score(gates, args.out, args.prefix, args.baseline)
        result.update(gates_sha256=digest, gates_version=gates['version'])
        Path(args.json).write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
        print(json.dumps(result['passed']))


if __name__ == '__main__':
    main()
