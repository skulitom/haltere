"""FastRaceCue coordinated turn: a new bearing is flown as an arc, not a chord.

The previous slew moved the horizontal request along the straight line to the
new request, so a large bearing change passed through low speeds, and a ring
clamped at a side edge asked for 30% speed. These tests pin the replacement: a
heading rotation limited to turn_acceleration/max(|v|, 1) rad/s plus a speed
change within the rest of command_acceleration (room to slow while turning), a
moderate, level side-edge request just beyond the clamped edge ray, and no change
for small corrections. The
closed-loop case uses the measured surrogate (or its inline stand-in) and the
synthetic HUD clamp; it is a development check, not flight evidence.
"""
from collections import deque
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from haltere.brain.motor_baseline import FastMotorPD
from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import FastCueConfig, FastRaceCue
from haltere.sim.identified import IdentifiedSim
from haltere.vision.camera import Camera
from tests.test_fast_motor_pd import CAL, load_profile
from tests.test_fast_race_cue import cue_toward, drive, states
from tests.test_visual_assistance import SENSOR

CONFIG = FastCueConfig()
DT = .01


@pytest.fixture(autouse=True, scope='module')
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def polar(speed, degrees):
    return speed*np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])


def chord_step(current, goal, dt, config=CONFIG):
    """The previous straight-line slew (tapered, command_acceleration bounded), for comparison."""
    step = goal-current
    norm = float(np.linalg.norm(step))
    limit = min(config.command_acceleration*dt, norm*dt/config.command_time_constant)
    return step*(limit/norm) if norm > limit else step


def slew(goal, start, seconds, rule=None, config=None):
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., config=config)
    rule = rule or pilot._horizontal_step
    trail = [np.asarray(start, float)]
    for _ in range(int(round(seconds/DT))):
        trail.append(trail[-1]+rule(trail[-1], np.asarray(goal, float), DT))
    return np.array(trail)


def heading_deg(v):
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def test_ninety_degree_change_is_an_arc_at_constant_speed():
    arc = slew(polar(6., 90.), polar(6., 0.), 3.)
    chord = slew(polar(6., 90.), polar(6., 0.), 3., rule=chord_step)
    speed = np.linalg.norm(arc, axis=1)
    assert speed.min() >= 6.-1e-9, 'the arc keeps the speed while it turns'
    assert np.linalg.norm(chord, axis=1).min() < 4.3, 'the old chord passed through 6 cos 45 deg'
    # Turn rate turn_acceleration/|v|: 90 deg at 6 m/s in 6*(pi/2)/turn_acceleration s plus the taper.
    reached = np.flatnonzero(heading_deg(arc) > 88.)[0]*DT
    assert 6.*np.pi/2/CONFIG.turn_acceleration <= reached < 6.*np.pi/2/CONFIG.turn_acceleration+.5
    np.testing.assert_allclose(arc[-1], polar(6., 90.), atol=2e-2)
    # Monotone heading: no overshoot through the new bearing.
    assert (np.diff(heading_deg(arc)) >= -1e-9).all() and heading_deg(arc).max() <= 90.+1e-6


@pytest.mark.parametrize('turn, start, goal', [(120., 6., 1.8), (75., 6., 3.9), (60., 3., 6.), (170., 6., 6.)])
def test_speed_moves_monotonically_between_the_two_request_speeds(turn, start, goal):
    """Never below both endpoint speeds (the chord's dip) and never beyond the goal speed."""
    arc = slew(polar(goal, turn), polar(start, 0.), 4.)
    speed = np.linalg.norm(arc, axis=1)
    assert speed.min() >= min(start, goal)-1e-9 and speed.max() <= max(start, goal)+1e-9
    assert (np.sign(goal-start)*np.diff(speed) >= -1e-9).all()
    np.testing.assert_allclose(arc[-1], polar(goal, turn), atol=2e-2)


def test_request_change_stays_within_the_declared_accelerations():
    rng = np.random.default_rng(3)
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6.)
    for _ in range(4000):
        current = polar(rng.uniform(0., 8.), rng.uniform(-180., 180.))
        goal = polar(rng.choice([0., rng.uniform(0., 8.)]), rng.uniform(-180., 180.))
        dt = float(rng.choice([0., .004, .01, .02, .1]))
        step = pilot._horizontal_step(current, goal, dt)
        assert np.linalg.norm(step) <= CONFIG.command_acceleration*dt+1e-12
        after = current+step
        if min(np.linalg.norm(current), np.linalg.norm(after)) > CONFIG.turn_min_speed and dt > 0:
            rotation = abs(np.arctan2(current[0]*after[1]-current[1]*after[0], current @ after))
            assert rotation <= CONFIG.turn_acceleration/max(np.linalg.norm(current), 1.)*dt+1e-9


def test_a_turn_that_never_converges_can_still_slow_down():
    """A goal that keeps running ahead of the request (circling a close checkpoint) must not hold
    the speed: turn_acceleration < command_acceleration leaves room to change speed while turning."""
    assert CONFIG.turn_acceleration < CONFIG.command_acceleration
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6.)
    current, speeds = polar(6., 0.), []
    for _ in range(150):
        goal = polar(1.8, heading_deg(current)+90.)
        current = current+pilot._horizontal_step(current, goal, DT)
        speeds.append(np.linalg.norm(current))
    room = np.sqrt(CONFIG.command_acceleration**2-CONFIG.turn_acceleration**2)
    assert speeds[29] <= 6.-.3*room+.05
    assert speeds[-1] == pytest.approx(1.8, abs=.05)


@pytest.mark.parametrize('turn, tolerance', [(0., 1e-12), (3., .02), (8., .04)])
def test_small_corrections_follow_the_previous_straight_line_slew(turn, tolerance):
    """Same trajectory as the straight-line slew to within a few percent of the request change."""
    for start, goal in ((6., 6.), (5., 6.), (6., 4.)):
        arc = slew(polar(goal, turn), polar(start, 0.), 1.5)
        chord = slew(polar(goal, turn), polar(start, 0.), 1.5, rule=chord_step)
        change = np.linalg.norm(polar(goal, turn)-polar(start, 0.))
        assert np.abs(arc-chord).max() <= tolerance*change
        if turn == 0.:
            assert np.abs(arc-chord).max() == 0.


def test_without_a_heading_the_straight_line_slew_applies():
    """Braking to a stop (search) is unchanged, and so is starting from (near) rest."""
    np.testing.assert_array_equal(slew((0., 0.), (5., 1.), 1.), slew((0., 0.), (5., 1.), 1., rule=chord_step))
    for start, goal in (((0., 0.), (-3., 2.)), ((.2, 0.), (0., 6.)), ((5., 0.), (.3, .1))):
        chord = slew(goal, start, 1., rule=chord_step)
        arc = slew(goal, start, 1.)
        slow = (np.linalg.norm(chord, axis=1) < CONFIG.turn_min_speed) | (np.linalg.norm(goal) < CONFIG.turn_min_speed)
        first = int(np.argmin(slow)) if not slow.all() else len(slow)
        assert first > 0
        np.testing.assert_allclose(arc[:first+1], chord[:first+1], rtol=0, atol=1e-12)


def test_side_edge_switch_keeps_a_moderate_speed_toward_the_edge():
    """Perfect-tracking plant at 6 m/s; the ring jumps to the left screen edge."""
    speed = 6.
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed)
    ahead = cue_toward([10., 0., 0.])
    rows = drive(pilot, history, lambda now: ahead if now < 10.5 else dict(u=.005, v=.5, edge=True), 300,
                 velocity=(speed, 0., 0.))
    after = [(state, command) for now, state, command, *_ in rows if now > 10.6]
    assert {state for state, _ in after} == {'side'}
    magnitude = np.array([np.linalg.norm(command[:2]) for _, command in after])
    assert magnitude.min() >= speed*CONFIG.side_speed_fraction-1e-6 >= .6*speed-1e-6
    edge = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg']).unproject_body(np.array([[1.6, 90.]]))[0]
    final = after[-1][1]
    assert heading_deg(final[:2]) == pytest.approx(heading_deg(edge[:2])+CONFIG.side_margin_deg, abs=1.)
    assert magnitude[-1] == pytest.approx(speed*CONFIG.side_speed_fraction, abs=2e-2)


@pytest.mark.parametrize('v', [.5, .8, .99])
def test_side_clamp_height_is_not_vertical_evidence(v):
    """Liftoff places side-clamped markers near the lower corners even for rings above:
    wherever the marker sits on the edge, the side request keeps its speed and stays level."""
    speed = 6.
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed)
    rows = drive(pilot, history, dict(u=.005, v=v, edge=True), 300, height=20.)
    assert set(states(rows)) == {'side'}
    command = rows[-1][2]
    assert np.linalg.norm(command[:2]) == pytest.approx(speed*CONFIG.side_speed_fraction, abs=2e-2)
    assert abs(command[2]) < 1e-9


def test_config_validates_and_metadata_declares_the_turn():
    for overrides in (dict(turn_acceleration=CONFIG.command_acceleration+1.), dict(turn_acceleration=0.),
                      dict(side_speed_fraction=1.2), dict(side_margin_deg=90.), dict(turn_min_speed=-1.)):
        with pytest.raises(ValueError):
            FastCueConfig(**overrides)
    meta = json.loads(json.dumps(FastRaceCue(SENSOR, CameraPoseHistory(), 6.).metadata()))
    assert meta['parameters'] == asdict(CONFIG)
    assert {'turn_acceleration', 'turn_min_speed', 'side_speed_fraction', 'side_margin_deg'} <= set(meta['parameters'])
    assert 'side_turn_deg' not in meta['parameters']
    assert 'turn_acceleration' in meta['turn'] and 'side_margin_deg' in meta['side_edge']


def two_gate_turn(turn_deg, seconds=12., speed=6.):
    """Pilot + FastMotorPD in the surrogate: a gate 25 m ahead, then one 25 m beyond at turn_deg."""
    from haltere.liftoff.fast_rehearsal import hud_marker
    profile = load_profile()
    sim = IdentifiedSim(profile, CAL)
    state = sim.hover(1, 5.)
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed)
    motor = FastMotorPD(profile, CAL)
    hover = motor.command(sim.sensors(state), torch.zeros(1, 3))
    motor.reset()
    queue = deque(hover.clone() for _ in range(3))
    t = np.radians(turn_deg)
    gates = [np.array([25., 0., 5.]), np.array([25.+25.*np.cos(t), 25.*np.sin(t), 5.])]
    target, passes, speeds = 0, [], []
    pending, next_capture, detection, capture_time = deque(), 0., None, None
    for k in range(int(seconds/.01)):
        now = k*.01
        position = state.quad.pos[0].numpy().astype(float)
        quaternion = state.quad.quat[0].numpy().astype(float)
        history.append(now, position, quaternion)
        if target < len(gates) and np.linalg.norm(gates[target]-position) < 3.:
            target += 1
            passes.append(k)
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
        speeds.append(float(torch.linalg.vector_norm(state.quad.vel[0, :2])))
    return passes, np.array(speeds)


def test_gate_switch_in_the_surrogate_keeps_momentum():
    passes, speeds = two_gate_turn(90.)
    assert len(passes) == 2, 'both gates are reached'
    before = speeds[passes[0]-50:passes[0]].mean()
    assert before > 5.
    # The previous chord and 30% side request dipped to 1.8 m/s (0.3 of the speed) here; this rule to 3.9.
    assert speeds[passes[0]:passes[0]+200].min() > .55*before
