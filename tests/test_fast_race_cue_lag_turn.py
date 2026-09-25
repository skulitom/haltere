"""FastRaceCue lag-aware turns (`LagTurnConfig`, runner flag --lag-turn): off by default.

The motors follow a velocity request late (brain-08 by 0.3-0.4 s, the fast PD by
0.1-0.15 s), so after a checkpoint switch the flown path swings outside the line to
the new ring. With the option, for the first second after an in-view ring bearing jumps
(against the filtered bearing or the in-view frames of the last 0.25 s), the horizontal
goal leads the bearing by a clipped multiple of the flown course's error and the request
heading turns with a shorter time constant, while the ring stays in view. These tests pin:
the default pilot is unchanged, the trigger (including a marker that moves over two frames
and clamped markers that never trigger), the window and the lead geometry, no effect in
the side/coast/search and edge-clamped states, the declared per-motor-contract file, and
(kinematic plants and the measured surrogate) a path that converges onto the line faster
without extra braking. None of it is flight evidence.
"""
from collections import deque
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from haltere.brain.motor_baseline import FastMotorPD
from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (LAG_TURN_STATES, FastCueConfig, FastRaceCue, LagTurnConfig,
                                           lag_turn_for_contract)
from haltere.liftoff.fast_rehearsal import hud_marker
from haltere.sim.identified import IdentifiedSim
from haltere.vision.camera import Camera
from tests.test_fast_motor_pd import CAL, load_profile
from tests.test_fast_race_cue import cue_toward, drive, states
from tests.test_visual_assistance import SENSOR, senses

CONFIG = FastCueConfig()
LAG = LagTurnConfig()
DT = .01
# Kinematic motor response models (delay s, first-order time constant s, max acceleration m/s^2),
# as declared in configs/obstacles/lag_turn.json.
BRAIN = (.15, .3, 6.5)
PD = (.09, .05, 12.)


@pytest.fixture(autouse=True, scope='module')
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def heading_deg(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def azimuth_cue(degrees, distance=20.):
    """In-view cue for a level ring at a world azimuth (the drive() helper keeps yaw 0)."""
    a = np.radians(degrees)
    return cue_toward([distance*np.cos(a), distance*np.sin(a), 0.])


def fly(lag_turn, plant, turn_deg=30., speed=6., seconds=7.5, approach=30., beyond=25., height=5.):
    """Kinematic closed loop. The velocity follows the pilot's request after a delay, with a
    first-order lag and an acceleration bound; yaw faces the pilot's filtered bearing. Ring A
    lies `approach` m ahead, ring B `beyond` m past it at turn_deg to the left; the target
    switches to B within 3 m of A. HUD cues come from the synthetic marker at 18 Hz, 60 ms old.
    Returns per-tick rows (time, position, velocity, request, state, lag weight), the switch
    (tick, position) and ring B."""
    delay, tau, amax = plant
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed, reference_speed=speed, lag_turn=lag_turn)
    turn = np.radians(turn_deg)
    rings = [np.array([approach, 0., height]),
             np.array([approach+beyond*np.cos(turn), beyond*np.sin(turn), height])]
    p, v, yaw = np.array([0., 0., height]), np.array([speed, 0., 0.]), 0.
    target, switch = 0, None
    pending, next_capture, detection, capture = deque(), 0., None, None
    issued = deque()
    rows = []
    for k in range(int(round(seconds/DT))):
        now = 10.+k*DT
        q = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
        history.append(now, p.copy(), q)
        if target == 0 and np.linalg.norm(rings[0]-p) < 3.:
            target, switch = 1, (k, p.copy())
        if now >= next_capture:
            cue = hud_marker(camera, rings[target], p, q)
            cue['aim_u'] = cue['u']
            pending.append((now, now+.06, cue))
            next_capture = now+.055
        while pending and pending[0][1] <= now:
            capture, _, cue = pending.popleft()
            detection = dict(race_cue=cue)
        pilot.update(senses(position=tuple(p), velocity=tuple(v), yaw=yaw), np.zeros(3), detection, capture, now)
        issued.append((now, pilot.velocity_command.copy()))
        while len(issued) > 1 and issued[1][0] <= now-delay+1e-9:
            issued.popleft()
        command = issued[0][1] if issued[0][0] <= now-delay+1e-9 else np.array([speed, 0., 0.])
        acceleration = (command-v)/tau
        norm = float(np.linalg.norm(acceleration))
        if norm > amax:
            acceleration *= amax/norm
        v = v+acceleration*DT
        p = p+v*DT
        if pilot.direction is not None:
            yaw = float(np.arctan2(pilot.direction[1], pilot.direction[0]))
        rows.append((now, p.copy(), v.copy(), pilot.velocity_command.copy(), pilot.state, pilot.lag_turn_weight))
    return rows, switch, rings[1], pilot


def lateral(rows, switch, ring, after):
    """Distance from the line (switch position -> ring), positive outside the (left) turn."""
    k, p0 = switch
    u = (ring-p0)[:2]/np.linalg.norm((ring-p0)[:2])
    n = np.array([-u[1], u[0]])
    return np.array([-(rows[j][1][:2]-p0[:2]) @ n for j in range(k, min(len(rows), k+int(round(after/DT))+1))])


# ---------------------------------------------------------------------------------------------
# Default behaviour and declaration
# ---------------------------------------------------------------------------------------------
def test_off_by_default_and_declared_when_on():
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6.)
    assert pilot.lag_turn is None and pilot.metadata()['lag_turn'] is None
    history = CameraPoseHistory()
    on = FastRaceCue(SENSOR, history, 6., lag_turn=LAG)
    drive(on, history, lambda now: azimuth_cue(0.) if now < 10.3 else azimuth_cue(25.), 80, velocity=(6., 0., 0.))
    meta = json.loads(json.dumps(on.metadata()))
    assert meta['lag_turn']['parameters'] == asdict(LAG)
    assert meta['lag_turn']['triggers'] >= 1 and meta['lag_turn']['active_seconds'] > 0
    assert 'course_lead' in meta['lag_turn']['rule'] and meta['parameters'] == asdict(CONFIG)
    with pytest.raises(ValueError):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., lag_turn=asdict(LAG))


@pytest.mark.parametrize('overrides', [dict(course_lead=0.), dict(course_lead=-.5), dict(heading_time_constant=0.),
                                       dict(course_lead_max_deg=90.), dict(trigger_deg=95.), dict(fade_s=1.5),
                                       dict(window_s=float('inf')), dict(max_lead_angle_deg=200.)])
def test_lag_turn_config_validates(overrides):
    with pytest.raises(ValueError):
        LagTurnConfig(**overrides)


def test_declaration_assigns_parameters_per_motor_contract():
    declaration = dict(contracts=dict(fast_velocity_pd_v1=asdict(LAG), fast_velocity_brain_v1=None))
    assert lag_turn_for_contract(declaration, 'fast_velocity_pd_v1') == LAG
    assert lag_turn_for_contract(declaration, 'fast_velocity_brain_v1') is None
    assert lag_turn_for_contract(declaration, 'motor_tracking_teacher_v1') is None
    with pytest.raises(ValueError):
        lag_turn_for_contract(dict(course_lead=.6), 'fast_velocity_pd_v1')
    with pytest.raises(TypeError):
        lag_turn_for_contract(dict(contracts=dict(fast_velocity_pd_v1=dict(course_lead=.6, route_gate=3))),
                              'fast_velocity_pd_v1')


def test_repository_declaration_is_frozen_and_edits_are_refused(tmp_path):
    from haltere.liftoff.visual_brain import (LAG_TURN_DECLARATION, lag_turn_declaration_sha256,
                                              load_lag_turn_declaration)
    declaration, digest = load_lag_turn_declaration(LAG_TURN_DECLARATION)
    assert declaration['frozen'] is True and declaration['sha256'] == digest
    assert set(declaration['contracts']) == {'fast_velocity_brain_v1', 'fast_velocity_pd_v1'}
    for contract in declaration['contracts']:
        config = lag_turn_for_contract(declaration, contract)
        assert config is None or isinstance(config, LagTurnConfig)
    assert lag_turn_for_contract(declaration, 'fast_velocity_pd_v1') is not None
    edited = json.loads(json.dumps(declaration))
    edited['contracts']['fast_velocity_pd_v1']['course_lead'] = 1.2
    path = tmp_path/'edited.json'
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError):
        load_lag_turn_declaration(path)
    unfrozen = {k: v for k, v in edited.items() if k not in ('frozen', 'frozen_at', 'sha256')}
    unfrozen['sha256'] = lag_turn_declaration_sha256(unfrozen)
    path.write_text(json.dumps(unfrozen))
    with pytest.raises(ValueError):
        load_lag_turn_declaration(path)


def test_runner_accepts_lag_turn_only_with_the_fast_pilot():
    from haltere.liftoff.visual_brain import VisualController
    with pytest.raises(ValueError, match='fast pilot'):
        VisualController('missing.pt', 'missing.json', 'cpu', pilot_assistance='race-cue', lag_turn='lag.json')


def test_default_heading_taper_is_unchanged():
    rng = np.random.default_rng(5)
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6.)
    for _ in range(2000):
        current = rng.uniform(-8., 8., 2)
        goal = rng.choice([0., 1.])*rng.uniform(-8., 8., 2)
        dt = float(rng.choice([0., .004, .01, .02, .1]))
        default = pilot._horizontal_step(current, goal, dt)
        np.testing.assert_array_equal(default, pilot._horizontal_step(current, goal, dt, None))
        np.testing.assert_array_equal(default, pilot._horizontal_step(current, goal, dt, CONFIG.command_time_constant))


def test_shorter_heading_constant_turns_faster_within_the_turn_bound_and_without_overshoot():
    rng = np.random.default_rng(7)
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6.)
    for _ in range(3000):
        speed, target = rng.uniform(1., 8.), rng.uniform(1., 8.)
        a0, a1 = rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)
        current, goal = speed*np.array([np.cos(a0), np.sin(a0)]), target*np.array([np.cos(a1), np.sin(a1)])
        dt = float(rng.choice([.004, .01, .02, .1]))
        fast = current+pilot._horizontal_step(current, goal, dt, LAG.heading_time_constant)
        slow = current+pilot._horizontal_step(current, goal, dt)
        turn = lambda after: abs(np.arctan2(current[0]*after[1]-current[1]*after[0], current @ after))
        remaining = abs(np.arctan2(current[0]*goal[1]-current[1]*goal[0], current @ goal))
        assert turn(fast) >= turn(slow)-1e-9
        assert turn(fast) <= min(remaining, CONFIG.turn_acceleration/max(speed, 1.)*dt)+1e-9
        assert np.linalg.norm(fast-current) <= CONFIG.command_acceleration*dt+1e-12


# ---------------------------------------------------------------------------------------------
# Window and lead geometry (tracking plant or a fixed measured velocity)
# ---------------------------------------------------------------------------------------------
def test_a_bearing_jump_opens_a_window_that_fades_out():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., lag_turn=LAG)
    weights = []
    for k in range(300):
        now = 10.+k*DT
        cue = azimuth_cue(0.) if now < 10.5 else azimuth_cue(25.)
        drive(pilot, history, cue, 1, velocity=(6., 0., 0.), start=now, plant='static')
        weights.append((now, pilot.lag_turn_since, pilot.lag_turn_weight))
    # The first fresh frame of the new bearing (captured 50 ms earlier) opens the window;
    # the blended bearing may re-open it on the next frame.
    assert 1 <= pilot.lag_turn_triggers <= 2
    assert 10.45-1e-9 <= pilot.lag_turn_since <= 10.5
    assert all(weight == 0. for now, since, weight in weights if now < 10.5-1e-9)
    for now, since, weight in weights:
        if since is None:
            assert weight == 0.
            continue
        age = now-since
        if age >= LAG.window_s:
            assert weight == 0.
        elif age <= LAG.window_s-LAG.fade_s:
            assert weight == 1.
        else:
            assert weight == pytest.approx((LAG.window_s-age)/LAG.fade_s)
    assert max(weight for *_, weight in weights) == 1. and weights[-1][2] == 0.


def frames(pilot, history, cues, period=.06, start=10., velocity=(6., 0., 0.)):
    """One cue per camera frame (period s apart, 50 ms old), pilot ticks every 10 ms in between."""
    ticks = int(round(period/DT))
    for i, cue in enumerate(cues):
        for j in range(ticks):
            now = start+(i*ticks+j)*DT
            drive(pilot, history, cue, 1, start=now, velocity=velocity, plant='static', camera=j == 0)
    return pilot


def test_a_marker_that_moves_over_two_frames_still_opens_the_window():
    """Pillar A (minus-brain08-01): the HUD marker went 8.4, 9.5, 13.9, 19.6, 20.2 deg in consecutive frames.
    Against the blended bearing no single frame jumps 10 deg; against the frames of the last 0.25 s it does."""
    azimuths = [8.4]*8+[9.5, 13.9, 19.6, 20.2, 20.6, 20.9, 21.]
    history = CameraPoseHistory()
    pilot = frames(FastRaceCue(SENSOR, history, 6., lag_turn=LAG), history, [azimuth_cue(a) for a in azimuths])
    assert pilot.lag_turn_triggers == 1
    assert pilot.lag_turn_since == pytest.approx(10.+10*.06-.05)      # the 19.6 deg frame
    history = CameraPoseHistory()
    slow = frames(FastRaceCue(SENSOR, history, 6., lag_turn=replace(LAG, trigger_span_s=.01)), history,
                  [azimuth_cue(a) for a in azimuths])
    assert slow.lag_turn_triggers == 0


def test_edge_clamped_markers_never_trigger_and_get_no_lead():
    """A bottom-clamped marker's azimuth is unreliable (it jumps as the clamp point moves): no trigger,
    and a window opened in view does not lead the bottom-edge descent."""
    clamps = [dict(u=u, v=.99, edge=True) for u in (.55, .56, .58, .3, .2, .45, .6, .35, .55, .5)]
    history = CameraPoseHistory()
    pilot = frames(FastRaceCue(SENSOR, history, 6., lag_turn=LAG), history, clamps)
    assert pilot.lag_turn_triggers == 0 and pilot.lag_turn_time == 0.
    cues = [azimuth_cue(0.)]*5+[azimuth_cue(25.)]*2+[dict(u=.3, v=.99, edge=True)]*8
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., lag_turn=LAG)
    seen = []
    for i, cue in enumerate(cues):
        for j in range(6):
            drive(pilot, history, cue, 1, start=10.+(i*6+j)*DT, velocity=(6., 0., 0.), plant='static', camera=j == 0)
            seen.append((pilot.state, pilot.lag_turn_weight, pilot.lag_turn_lead_deg))
    assert pilot.lag_turn_triggers >= 1
    assert any(state == 'cue' and lead > 5. for state, _, lead in seen), 'led while the ring was in view'
    clamped = [(weight, lead) for state, weight, lead in seen if state in ('below', 'below_weak')]
    assert clamped and all(weight > 0 for weight, _ in clamped), 'the window is still open'
    assert all(lead == 0. for _, lead in clamped), 'but a clamped marker is not led'


def test_small_bearing_changes_do_not_open_a_window():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., lag_turn=LAG)
    rows = drive(pilot, history, lambda now: azimuth_cue(0. if now < 10.5 else 6.), 200, velocity=(6., 0., 0.))
    assert pilot.lag_turn_triggers == 0 and pilot.lag_turn_time == 0.
    default_history = CameraPoseHistory()
    default = FastRaceCue(SENSOR, default_history, 6.)
    reference = drive(default, default_history, lambda now: azimuth_cue(0. if now < 10.5 else 6.), 200,
                      velocity=(6., 0., 0.))
    for (_, _, command, *_), (_, _, expected, *_) in zip(rows, reference):
        np.testing.assert_array_equal(command, expected)


@pytest.mark.parametrize('bearing, course, lead', [(20., 0., 12.), (30., 0., 15.), (-40., 0., -15.),
                                                   (20., 30., -6.), (5., 0., 3.)])
def test_goal_leads_the_bearing_by_the_clipped_course_error(bearing, course, lead):
    """The measured course is held fixed (a motor that has not responded yet): the request turns
    to bearing + clip(course_lead*(bearing - course), +-course_lead_max_deg) while the window lasts,
    at the unchanged scheduled speed, and back to the bearing afterwards."""
    velocity = tuple(6.*np.array([np.cos(np.radians(course)), np.sin(np.radians(course)), 0.]))
    config = replace(LAG, trigger_deg=3.)
    runs = {}
    for name, lag_turn in (('on', config), ('off', None)):
        history = CameraPoseHistory()
        pilot = FastRaceCue(SENSOR, history, 6., lag_turn=lag_turn)
        rows = drive(pilot, history, lambda now: azimuth_cue(course if now < 10.3 else bearing), 250,
                     plant='static', velocity=velocity)
        runs[name] = (pilot, rows)
    pilot, rows = runs['on']
    assert pilot.lag_turn_triggers >= 1 and 10.2 < pilot.lag_turn_since < 10.35
    check = pilot.lag_turn_since+LAG.window_s-LAG.fade_s-.02    # full weight, request settled
    j = int(np.argmin([abs(now-check) for now, *_ in rows]))
    state, command = rows[j][1], rows[j][2]
    assert state in LAG_TURN_STATES and pilot.lag_turn_since+.6 < rows[j][0] <= check+.005
    # (a 55 deg request turn at the turn_acceleration bound is still settling: 1.5 deg)
    assert heading_deg(command[:2]) == pytest.approx(bearing+lead, abs=.5 if abs(bearing+lead) < 40 else 1.5)
    reference = runs['off'][1][j][2]     # the default pilot, still turning toward the bearing itself
    assert abs(heading_deg(command[:2])-course) > abs(heading_deg(reference[:2])-course)+abs(lead)/2
    assert np.linalg.norm(command[:2]) == pytest.approx(np.linalg.norm(reference[:2]), abs=.05)
    assert heading_deg(rows[-1][2][:2]) == pytest.approx(bearing, abs=.5)


def test_side_coast_and_search_states_are_unchanged():
    def cue(now):
        if now < 10.5:
            return azimuth_cue(0.)
        if now < 11.5:
            return dict(u=.005, v=.5, edge=True)   # ring beyond the left edge after the switch
        if now < 12.:
            return azimuth_cue(40.)
        return None                                 # dropout: coast, then search
    runs = []
    for lag_turn in (LAG, None):
        history = CameraPoseHistory()
        pilot = FastRaceCue(SENSOR, history, 6., lag_turn=lag_turn)
        runs.append(drive(pilot, history, cue, 350, velocity=(6., 0., 0.)))
    on, off = runs
    assert {'side', 'coast', 'search'} <= set(states(on)) and states(on) == states(off)
    changed = [now for (now, state, command, *_), (_, _, expected, *_) in zip(on, off)
               if not np.array_equal(command, expected)]
    # Only the in-view bearing states after the edge (11.5-12 s) may differ, and the carried-over
    # request until the side, coast and search slews have absorbed it.
    assert changed and min(changed) >= 11.5
    for now, state, command, *_ in on:
        if 10.5 <= now < 11.5:
            assert state == 'side'


# ---------------------------------------------------------------------------------------------
# Closed loop: lagging motors converge onto the line to the new ring faster
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('plant', [BRAIN, PD], ids=['brain-like', 'pd-like'])
@pytest.mark.parametrize('turn_deg', [20., 30., 40.])
def test_lagging_motor_converges_onto_the_line_faster_without_extra_braking(plant, turn_deg):
    results = {}
    for name, lag_turn in (('off', None), ('on', LAG)):
        rows, switch, ring, pilot = fly(lag_turn, plant, turn_deg)
        assert switch is not None
        lat = lateral(rows, switch, ring, 2.)
        k = switch[0]
        results[name] = dict(
            at1=lat[int(round(1./DT))], inside=float(lat.min()),
            request_min=min(float(np.linalg.norm(r[3][:2])) for r in rows[k:k+200]),
            speed_min=min(float(np.linalg.norm(r[2][:2])) for r in rows[k:k+200]),
            before=[r[3] for r in rows[:k]], pilot=pilot)
    on, off = results['on'], results['off']
    # Identical approach: nothing changes before the switch.
    for a, b in zip(on['before'], off['before']):
        np.testing.assert_array_equal(a, b)
    assert on['pilot'].lag_turn_triggers >= 1
    assert on['at1'] < off['at1']-.1, (on['at1'], off['at1'])
    assert on['inside'] > -.35, 'the lead may overshoot the line only slightly'
    # No brake-and-reaccelerate: the request never slows more than without the option (the
    # acceleration-bounded motor may lose a little more speed while it turns faster).
    assert on['request_min'] >= off['request_min']-.05
    assert on['speed_min'] >= off['speed_min']-.2


def surrogate_turn(lag_turn, turn_deg=35., seconds=12., speed=6.):
    """Pilot + FastMotorPD in the measured surrogate: a gate 25 m ahead, then one 25 m beyond at turn_deg."""
    profile = load_profile()
    sim = IdentifiedSim(profile, CAL)
    state = sim.hover(1, 5.)
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed, lag_turn=lag_turn)
    motor = FastMotorPD(profile, CAL)
    hover = motor.command(sim.sensors(state), torch.zeros(1, 3))
    motor.reset()
    queue = deque(hover.clone() for _ in range(3))
    t = np.radians(turn_deg)
    gates = [np.array([25., 0., 5.]), np.array([25.+25.*np.cos(t), 25.*np.sin(t), 5.])]
    target, passes, track = 0, [], []
    pending, next_capture, detection, capture_time = deque(), 0., None, None
    for k in range(int(seconds/DT)):
        now = k*DT
        position = state.quad.pos[0].numpy().astype(float)
        quaternion = state.quad.quat[0].numpy().astype(float)
        history.append(now, position, quaternion)
        if target < len(gates) and np.linalg.norm(gates[target]-position) < 3.:
            target += 1
            passes.append((k, position.copy()))
            if target == len(gates):
                break
        if now >= next_capture:
            cue = hud_marker(camera, gates[target], position, quaternion)
            cue['aim_u'] = cue['u']
            pending.append((now, now+.06, cue))
            next_capture = now+.055
        while pending and pending[0][1] <= now:
            capture_time, _, cue = pending.popleft()
            detection = dict(race_cue=cue)
        sensors = sim.sensors(state)
        pilot.update(sensors, sensors['gyro'][0].numpy(), detection, capture_time, now)
        action = motor.command(sensors, torch.tensor(pilot.velocity_command, dtype=torch.float32)[None],
                               torch.tensor(pilot.feedforward, dtype=torch.float32)[None])
        queue.append(torch.tensor(pilot.command(action[0].numpy()), dtype=torch.float32)[None])
        state = sim.step(state, queue.popleft())
        track.append((state.quad.pos[0].numpy().astype(float).copy(),
                      float(torch.linalg.vector_norm(state.quad.vel[0, :2]))))
    return passes, track, gates[1]


def test_surrogate_gate_switch_with_lag_turn_keeps_momentum_and_cuts_the_swing():
    results = {}
    for name, lag_turn in (('off', None), ('on', LAG)):
        passes, track, ring = surrogate_turn(lag_turn)
        assert len(passes) == 2, 'both gates are reached'
        k, p0 = passes[0]
        rows = [(None, position) for position, _ in track]
        lat = lateral(rows, (k, p0), ring, 1.)
        speeds = np.array([s for _, s in track])
        results[name] = dict(at1=lat[-1], before=speeds[k-50:k].mean(), low=speeds[k:k+200].min())
    on, off = results['on'], results['off']
    assert on['at1'] < off['at1'], (on['at1'], off['at1'])
    assert on['low'] > .55*on['before'] and on['low'] >= off['low']-.2
