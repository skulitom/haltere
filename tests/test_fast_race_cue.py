"""FastRaceCue: velocity-level guidance from the visible next-checkpoint HUD ring.

The ring is a disclosed generic race cue. These tests pin the declared
behaviour (bearing -> world velocity, edge handling, dropout, support, limits)
and the input contract: camera cue, own telemetry and time, nothing read from a
course, route or file. Closed-loop cases use the measured surrogate and the
synthetic HUD clamp from the offline rehearsal tool; none of it is flight evidence.
"""
import ast
import json
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from haltere.brain.motor_baseline import FastMotorPD, measured_inverse_rate
from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import DEFAULT_YAW_CURVE, FastCueConfig, FastRaceCue
from haltere.sim.identified import IdentifiedSim, processed_command
from haltere.vision.camera import Camera
from tests.test_fast_motor_pd import CAL, load_profile
from tests.test_visual_assistance import SENSOR, senses

SOURCE = Path(__file__).resolve().parents[1] / 'haltere' / 'liftoff' / 'fast_race_cue.py'
CONFIG = FastCueConfig()
BELOW = dict(u=.5, v=.99, edge=True)
ABOVE = dict(u=.5, v=.01, edge=True)


@pytest.fixture(autouse=True, scope='module')
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def cue_toward(body_point):
    """HUD cue for a body-frame point, projected through the calibrated camera."""
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    pixels, ok = camera.project_body(np.array([body_point], dtype=float))
    assert ok[0]
    return dict(u=float(pixels[0, 0] / 320), v=float(pixels[0, 1] / 180), edge=False)


def drive(pilot, history, cue, steps, *, plant='perfect', yaw=0., height=5., start=10., velocity=(0., 0., 0.),
          latency=.05, camera=True, dt=.01):
    """Step the pilot with a cue (dict, None or callable of time); return (time, state, command, yaw) rows.

    plant='perfect' reports the previous velocity request as the measured motion (a plant
    that tracks); plant='static' keeps the measured velocity fixed (e.g. resting on a surface).
    """
    measured = np.array(velocity, dtype=float)
    rows = []
    for k in range(steps):
        now = start + k * dt
        s = senses(position=(0., 0., height), velocity=tuple(measured.tolist()), yaw=yaw)
        history.append(now, [0., 0., height], s['quat'][0].numpy())
        current = cue(now) if callable(cue) else cue
        detection = dict(race_cue=dict(current)) if current is not None else None
        relative, _ = pilot.update(s, [0., 0., 0.], detection, now - latency if camera else None, now)
        rows.append((now, pilot.state, pilot.velocity_command.copy(), pilot.pilot.sight_yaw, relative))
        if plant == 'perfect':
            measured = pilot.velocity_command.copy()
    return rows


def states(rows):
    return [state for _, state, *_ in rows]


def hover_action():
    profile = load_profile()
    sim = IdentifiedSim(profile, CAL)
    state = sim.hover(1, 5.)
    return FastMotorPD(profile, CAL).command(sim.sensors(state), torch.zeros(1, 3))[0].numpy()


def yaw_rate_in_surrogate(command, steps=5):
    """World yaw rate the measured surrogate produces for a full brain-order command."""
    sim = IdentifiedSim(load_profile(), CAL)
    state = sim.hover(1, 5.)
    for _ in range(steps):
        state = sim.step(state, torch.tensor(command, dtype=torch.float32)[None])
    return float(state.quad.omega[0, 2])


@pytest.mark.parametrize('speed', [.5, 6., 20.])
def test_accepts_speeds_up_to_twenty(speed):
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), speed)
    assert pilot.speed == speed and pilot.metadata()['nominal_speed_mps'] == speed


@pytest.mark.parametrize('speed', [0., -1., 20.01, 40., float('nan'), float('inf')])
def test_invalid_speed_is_rejected(speed):
    with pytest.raises(ValueError, match='speed'):
        FastRaceCue(SENSOR, CameraPoseHistory(), speed)


@pytest.mark.parametrize('reference', [0., 20.5, float('nan')])
def test_invalid_reference_speed_is_rejected(reference):
    with pytest.raises(ValueError, match='reference speed'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., reference_speed=reference)


def test_requires_calibrated_camera_and_valid_config():
    with pytest.raises(ValueError, match='calibrated camera'):
        FastRaceCue(None, CameraPoseHistory())
    for overrides in (dict(min_speed_fraction=1.), dict(direction_blend=1.5), dict(coast_s=0.),
                      dict(command_acceleration=-1.), dict(yaw_gain=float('nan')), dict(support_after_s=float('inf'))):
        with pytest.raises(ValueError):
            FastCueConfig(**overrides)


@pytest.mark.parametrize('yaw', [0., 1.2, -2.5])
def test_cue_bearing_sets_world_velocity_direction(yaw):
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6.)
    rows = drive(pilot, history, cue_toward([10., 5., 0.]), 200, plant='static', yaw=yaw)
    _, state, command, sight_yaw, relative = rows[-1]
    assert state == 'cue' and pilot.pilot.mode == 2
    bearing = np.arctan2(5., 10.)
    expected_direction = np.array([np.cos(yaw + bearing), np.sin(yaw + bearing)])
    # At rest the alignment reference is the heading: speed = v * (min + (1-min) cos^2(bearing)).
    expected_speed = 6. * (CONFIG.min_speed_fraction + (1 - CONFIG.min_speed_fraction) * np.cos(bearing) ** 2)
    # The command ramp tapers exponentially onto its goal (command_time_constant).
    np.testing.assert_allclose(command[:2], expected_direction * expected_speed, atol=2e-2)
    assert abs(command[2]) < 1e-3
    if yaw == 0.:
        assert command[1] > 1., 'a target ahead-left must give a positive (left) world y velocity'
    # The body-frame lead and the yaw assistance do not depend on the absolute heading.
    np.testing.assert_allclose(relative, np.r_[np.array([np.cos(bearing), np.sin(bearing)]) * expected_speed, 0.],
                               atol=2e-2)
    assert sight_yaw < -.1, 'facing a left target: negative brain yaw (turns left in the measured plant)'


def test_yaw_assistance_turns_toward_the_cue_in_the_measured_plant():
    for body_point, sign in (([10., 5., 0.], 1.), ([10., -5., 0.], -1.)):
        history = CameraPoseHistory()
        pilot = FastRaceCue(SENSOR, history, 6.)
        drive(pilot, history, cue_toward(body_point), 60, plant='static')
        command = pilot.command(hover_action())
        assert sign * yaw_rate_in_surrogate(command) > .2


def test_bottom_edge_cue_descends_with_reduced_speed():
    speed = 10.
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed)
    rows = drive(pilot, history, BELOW, 150, height=20.)
    _, state, command, *_ = rows[-1]
    assert state == 'below' and pilot.pilot.mode == 6
    assert set(states(rows)) == {'below'}, 'a tracked descent is not support'
    # A level drone's bottom edge points about 10 degrees below the horizon:
    # nearly full descent evidence, latched for the clip episode.
    weight = pilot.below_weight
    assert .9 < weight <= 1.
    horizontal = np.linalg.norm(command[:2])
    expected = speed * (1 - weight) + max(CONFIG.edge_speed, CONFIG.below_speed_fraction * speed) * weight
    assert horizontal == pytest.approx(expected, rel=2e-2)
    # Descend along a slope just steeper than the clamped edge ray, not a dive.
    slope = np.degrees(np.arctan2(-command[2], horizontal))
    assert slope == pytest.approx((pilot.edge_depression + CONFIG.below_slope_margin_deg), abs=1.)
    assert -command[2] <= CONFIG.vertical_down + 1e-9
    assert horizontal < .6 * speed and command[0] > .5 and abs(command[1]) < 1e-6
    # The same bearing inside the image flies at the full aligned speed instead.
    free = FastRaceCue(SENSOR, CameraPoseHistory(), speed)
    free_rows = drive(free, free.pose_history, cue_toward([10., 0., 0.]), 250, height=20.)
    assert np.linalg.norm(free_rows[-1][2][:2]) > 1.5 * horizontal


def test_top_edge_cue_climbs_and_bounds_forward_speed():
    speed = 10.
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed)
    rows = drive(pilot, history, ABOVE, 150)
    _, state, command, *_ = rows[-1]
    assert state == 'above' and pilot.pilot.mode == 7
    assert command[2] == pytest.approx(CONFIG.vertical_up, abs=1e-6)
    horizontal = np.linalg.norm(command[:2])
    d = pilot.direction
    slope = d[2] / np.linalg.norm(d[:2])
    assert slope > 1.
    assert horizontal <= CONFIG.vertical_up / max(slope, .2) + 1e-6
    assert command[0] > 0 and horizontal < speed * .5


@pytest.mark.parametrize('u, side', [(.005, 1.), (.995, -1.)])
def test_side_edge_cue_turns_toward_that_side(u, side):
    speed = 10.
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed)
    rows = drive(pilot, history, dict(u=u, v=.5, edge=True), 200, plant='static')
    _, state, command, sight_yaw, _ = rows[-1]
    assert state == 'side' and pilot.pilot.mode == 5 and pilot.side == side
    assert side * command[1] > 0 and abs(command[2]) < 1e-9
    # A moderate speed toward side_margin_deg beyond the clamped edge ray (about 61 deg for this camera).
    assert np.linalg.norm(command[:2]) == pytest.approx(speed * CONFIG.side_speed_fraction, abs=2e-2)
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    edge = camera.unproject_body(np.array([[u * 320, 90.]]))[0]
    expected = np.degrees(np.arctan2(edge[1], edge[0])) + side * CONFIG.side_margin_deg
    assert 55. < abs(expected) - CONFIG.side_margin_deg < 65.
    assert np.degrees(np.arctan2(command[1], command[0])) == pytest.approx(expected, abs=.5)
    assert side * sight_yaw < 0
    assert side * yaw_rate_in_surrogate(pilot.command(hover_action())) > .5


@pytest.mark.parametrize('u', [.005, .995])
def test_assisted_yaw_command_stays_inside_the_stick_range(u):
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 10.)
    hover = hover_action()
    worst = 0.
    for k in range(100):
        drive(pilot, history, dict(u=u, v=.5, edge=True), 1, plant='static', start=10. + k * .01)
        command = pilot.command(hover)
        worst = max(worst, abs(float(command[3])))
    thrust = processed_command(torch.tensor(pilot.command(hover), dtype=torch.float32)[None], CAL)[0, 0]
    assert worst <= 1., (f'|yaw| reached {worst:.3f}; processed throttle {float(thrust):.4f} instead of '
                         f'{CAL["hover_processed"] + CAL["throttle_scale"] * (hover[0] - CAL["hover_stick_sim"]):.4f}')


@pytest.mark.parametrize('speed, coasts', [(6., True), (18., False)])
def test_dropout_coasts_then_brakes_and_searches(speed, coasts):
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed)
    ahead = cue_toward([10., 0., 0.])
    seen_steps = 250  # the ring is visible for 2.5 s, then frames keep arriving without it
    rows = drive(pilot, history, lambda now: ahead if now < 10. + (seen_steps - .5) * .01 else None, 520)
    last_seen = pilot.last_seen
    assert last_seen == pytest.approx(rows[seen_steps - 1][0] - .05)
    at_loss = rows[seen_steps][2]
    assert np.linalg.norm(at_loss[:2]) == pytest.approx(speed, rel=1e-2)
    # 4 m of coasting, capped at 0.6 s and never shorter than the 0.25 s frame-gap allowance.
    coast_window = min(CONFIG.coast_s, max(.25, CONFIG.coast_distance_m / speed))
    for k, (now, state, command, *_) in enumerate(rows):
        age = now - last_seen
        if k < seen_steps or age <= .25:
            assert state == 'cue'  # a few missed frames keep the last bearing
        elif age <= coast_window:
            assert state == 'coast'
            np.testing.assert_allclose(command, at_loss, atol=1e-3)  # coast on the previous request
        else:
            assert state == 'search'
    assert ('coast' in states(rows)) == coasts
    final = rows[-1][2]
    np.testing.assert_allclose(final, np.zeros(3), atol=.1)  # braking toward a zero-velocity request
    assert pilot.pilot.target is None and pilot.pilot.mode == 4
    expected = -float(measured_inverse_rate(torch.tensor([np.degrees(CONFIG.search_yaw_rate * pilot.side)],
                                                         dtype=torch.float64), *DEFAULT_YAW_CURVE)[0])
    assert rows[-1][3] == pytest.approx(expected, abs=1e-6)
    # A capture outage (no current image) permits braking only, never a blind search turn.
    drive(pilot, history, None, 60, start=rows[-1][0] + .01, camera=False)
    assert pilot.state == 'search' and pilot.pilot.sight_yaw == pytest.approx(0., abs=1e-9)


def test_support_detection_climbs_when_commanded_descent_is_not_achieved():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 10.)
    rows = drive(pilot, history, BELOW, 200, plant='static')  # measured vz stays 0: something is underneath
    trail = states(rows)
    assert 'support_climb' in trail and pilot.pilot.mode in (6, 8)
    descending = next(now for now, _, command, *_ in rows if command[2] < -.8)
    climbing = next(i for i, state in enumerate(trail) if state == 'support_climb')
    assert rows[climbing][0] - descending == pytest.approx(CONFIG.support_after_s, abs=.03)
    # During the climb the vertical request rises monotonically and turns upward.
    end = climbing + next(i for i, state in enumerate(trail[climbing:]) if state != 'support_climb')
    vz = np.array([command[2] for _, _, command, *_ in rows[climbing:end]])
    assert (np.diff(vz) >= -1e-12).all() and vz.max() > 0
    assert rows[end][0] - rows[climbing][0] == pytest.approx(CONFIG.support_climb_s, abs=.03)
    assert trail[end] == 'below'


def test_support_detection_ignores_a_tracked_descent():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 10.)
    rows = drive(pilot, history, BELOW, 300, height=30.)
    assert 'support_climb' not in states(rows)
    assert rows[-1][2][2] < -1.


def test_descent_path_governor_slows_horizontally_when_the_sink_is_not_achieved():
    # A vehicle that sinks at a third of the request (e.g. a controller with little
    # downward authority) gets a slower horizontal request; a tracking one does not.
    def run(fraction):
        history = CameraPoseHistory()
        pilot = FastRaceCue(SENSOR, history, 6.)
        measured = np.zeros(3)
        for k in range(150):
            now = 10.+k*.01
            s = senses(position=(0., 0., 30.), velocity=tuple(measured.tolist()))
            history.append(now, [0., 0., 30.], s['quat'][0].numpy())
            pilot.update(s, [0., 0., 0.], dict(race_cue=dict(BELOW)), now-.05, now)
            measured = pilot.velocity_command*np.array([1., 1., fraction])
        return pilot
    tracked, lagging = run(1.), run(1/3)
    assert tracked.descent_scale == 1. and lagging.descent_scale < .7
    assert np.linalg.norm(lagging.velocity_command[:2]) < .8*np.linalg.norm(tracked.velocity_command[:2])
    assert lagging.velocity_command[2] < -.5  # the descent itself is still requested


def closed_loop(target, steps, height, *, floor=None, speed=8.):
    """Pilot + FastMotorPD on the measured surrogate with a synthetic, delayed HUD cue."""
    from haltere.liftoff.fast_rehearsal import hud_marker
    profile = load_profile()
    sim = IdentifiedSim(profile, CAL)
    state = sim.hover(1, height)
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed)
    motor = FastMotorPD(profile, CAL)
    hover = motor.command(sim.sensors(state), torch.zeros(1, 3))
    motor.reset()
    queue = deque(hover.clone() for _ in range(3))
    pending, next_capture, detection, capture_time = deque(), 0., None, None
    trail, vertical = [], []
    for k in range(steps):
        now = k * .01
        position = state.quad.pos[0].numpy().astype(float)
        quaternion = state.quad.quat[0].numpy().astype(float)
        history.append(now, position, quaternion)
        if now >= next_capture:
            cue = hud_marker(camera, target, position, quaternion)
            cue['aim_u'] = cue['u']
            pending.append((now, now + .06, cue))
            next_capture = now + .055
        while pending and pending[0][1] <= now:
            capture_time, _, cue = pending.popleft()
            detection = dict(race_cue=cue)
        sensors = sim.sensors(state)
        pilot.update(sensors, sensors['gyro'][0].numpy(), detection, capture_time, now)
        action = motor.command(sensors, torch.tensor(pilot.velocity_command, dtype=torch.float32)[None],
                               torch.tensor(pilot.feedforward, dtype=torch.float32)[None])
        queue.append(torch.tensor(pilot.command(action[0].numpy()), dtype=torch.float32)[None])
        state = sim.step(state, queue.popleft())
        if floor is not None and float(state.quad.pos[0, 2]) < floor:
            state.quad.pos[0, 2] = floor  # a surface under the drone
            state.quad.vel[0, 2] = state.quad.vel[0, 2].clamp_min(0.)
        trail.append(pilot.state)
        vertical.append(float(state.quad.vel[0, 2]))
    return trail, np.array(vertical), state


def test_free_air_descent_in_the_measured_plant_is_not_mistaken_for_support():
    trail, vertical, state = closed_loop(np.array([25., 0., 1.]), 250, 20.)
    assert 'below' in trail
    assert 'support_climb' not in trail
    assert vertical.min() < -2.5 and not bool(state.quad.crashed[0])


def test_surface_contact_in_the_measured_plant_triggers_support_climb():
    trail, vertical, _ = closed_loop(np.array([25., 0., -30.]), 150, 5., floor=4.8)
    assert 'support_climb' in trail
    assert vertical.max() > .1  # the short climb lifts off the surface


def test_velocity_command_is_acceleration_limited():
    rng = np.random.default_rng(5)
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 20.)
    measured = np.array([5., 0., 0.])
    now, previous, previous_time = 10., None, None
    for k in range(600):
        now += float(rng.choice([0., .004, .01, .02, .15], p=[.05, .3, .45, .15, .05]))
        height = .3 if k < 20 else 5.  # includes the launch phase
        s = senses(position=(0., 0., height), velocity=tuple(measured.tolist()), yaw=float(rng.uniform(-3, 3)))
        history.append(now, [0., 0., height], s['quat'][0].numpy())
        cue = None
        if rng.random() > .2:
            cue = dict(u=float(rng.uniform(0, 1)), v=float(rng.uniform(0, 1)), edge=bool(rng.random() < .3))
        pilot.update(s, rng.normal(0, 1, 3), dict(race_cue=cue) if cue else None, now - .05, now)
        command = pilot.velocity_command
        if previous is None:
            # The first request starts from the measured motion, not from rest.
            assert np.linalg.norm(command - measured) <= np.hypot(CONFIG.command_acceleration,
                                                                  CONFIG.vertical_command_acceleration) * .01 + 1e-9
        else:
            dt = min(now - previous_time, .1)
            step = command - previous
            assert np.linalg.norm(step[:2]) <= CONFIG.command_acceleration * dt + 1e-9
            assert abs(step[2]) <= CONFIG.vertical_command_acceleration * dt + 1e-9
        assert np.isfinite(pilot.feedforward).all()
        assert np.linalg.norm(pilot.feedforward[:2]) <= CONFIG.command_acceleration + 1e-6
        assert abs(pilot.feedforward[2]) <= CONFIG.vertical_command_acceleration + 1e-6
        previous, previous_time = command.copy(), now
        measured = command + rng.normal(0, .3, 3)


def test_launch_climbs_before_following_the_cue():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 10.)
    rows = drive(pilot, history, cue_toward([10., 0., 0.]), 100, height=.2)
    assert set(states(rows)) == {'launch'}
    command = rows[-1][2]
    assert command[2] == pytest.approx(1.5) and np.linalg.norm(command[:2]) <= 1.5 + 1e-9
    drive(pilot, history, cue_toward([10., 0., 0.]), 1, height=1., start=11.)
    assert not pilot.launching
    rows = drive(pilot, history, cue_toward([10., 0., 0.]), 5, height=.2, start=11.01)
    assert 'launch' not in states(rows), 'launch clearance is released after the first ascent'


def test_only_fresh_causal_frames_are_ingested():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6.)
    s = senses()
    history.append(10., [0., 0., 1.5], s['quat'][0].numpy())
    cue = dict(race_cue=cue_toward([10., 0., 0.]))
    pilot.update(s, [0., 0., 0.], cue, 9.8, 10.)  # older than 120 ms
    pilot.update(s, [0., 0., 0.], cue, 10.05, 10.01)  # from the future
    assert pilot.frames == 0 and pilot.direction is None and pilot.state == 'wait'
    pilot.update(s, [0., 0., 0.], cue, 10., 10.02)
    pilot.update(s, [0., 0., 0.], cue, 10., 10.03)  # the same frame again
    assert pilot.frames == 1
    with pytest.raises(ValueError, match='image position'):
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(u=1.2, v=.5, edge=False)), 10.04, 10.05)
    with pytest.raises(ValueError, match='clearance'):
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(u=.5, v=.5, aim_u=float('nan'), edge=False)), 10.06, 10.07)


def test_senses_are_not_mutated_and_command_only_replaces_yaw():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6.)
    s = senses(velocity=(2., 1., 0.), yaw=.4)
    original = {k: v.clone() for k, v in s.items()}
    history.append(10., [0., 0., 1.5], s['quat'][0].numpy())
    _, modified = pilot.update(s, [0., 0., 0.], dict(race_cue=cue_toward([10., 3., 0.])), 9.95, 10.)
    assert all(torch.equal(s[k], original[k]) for k in s)
    assert modified['pos'] is s['pos']
    torch.testing.assert_close(modified['vel_world'], s['vel_world'])  # reference <= speed: no flow scaling
    action = np.array([.1, .2, .3, .4])
    np.testing.assert_array_equal(pilot.command(action), [.1, .2, .3, pilot.pilot.sight_yaw])
    np.testing.assert_array_equal(action, [.1, .2, .3, .4])


def test_metadata_is_json_serializable_and_declares_the_cue():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 8., reference_speed=8.)
    drive(pilot, history, lambda now: BELOW if now < 11. else None, 250, plant='static')
    assert {'below', 'support_climb', 'search'} <= set(pilot.state_time)
    meta = json.loads(json.dumps(pilot.metadata()))
    assert meta['profile'] == 'fast-v1' and meta['mode'] == 'race-cue'
    assert meta['visible_race_cues'] is True and meta['runtime_route_oracle'] is False
    assert meta['estimated_passages'] is None
    assert meta['parameters'] == asdict(CONFIG)
    assert meta['nominal_speed_mps'] == 8. and meta['cue_frames'] == pilot.frames > 0
    assert set(meta['state_seconds']) == set(pilot.state_time)


COURSE_NAMES = ['gates_strawbale', 'track_strawbale', 'obstacles_strawbale', 'gates_pinevalley',
                'load_gate_file', 'gate_passes', 'collection_route', 'oracle_assistance', 'challenge_courses',
                'course_pool', 'qualification_route', 'fast_rehearsal', 'synthetic_course']


def test_fast_cue_reads_no_files_and_imports_no_course_or_route_module():
    src = SOURCE.read_text(encoding='utf-8')
    for token in ('open(', 'read_text', 'read_bytes', 'Path(', 'json', 'yaml', 'np.load', 'torch.load'):
        assert token not in src, f'fast_race_cue.py contains {token!r}: the flight pilot must not read files'
    found = [name for name in COURSE_NAMES if name in src]
    assert not found, f'fast_race_cue.py reaches for {found}; it must fly from the visible cue alone'
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add('.' * node.level + (node.module or ''))
    assert imported == {'dataclasses', 'types', 'numpy', 'torch', '..vision.camera', '..brain.motor_baseline'}, imported
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            called.add(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else '')
    forbidden = called & {'open', 'load', 'safe_load', 'loads', 'read_text', 'read_bytes', 'Path', '__import__',
                          'import_module'}
    assert not forbidden, f'FastRaceCue calls {forbidden}'


def test_fast_brain_contract_scales_goal_and_velocity_senses():
    from haltere.train.fast_motor_tracking import fast_contract
    contract = fast_contract(8.)
    assert contract['goal_seconds'] * 8. == pytest.approx(3.)
    assert contract['velocity_scale'] * 8. == pytest.approx(3.)
    with pytest.raises(ValueError):
        fast_contract(0.)
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 8., reference_speed=8., velocity_scale=contract['velocity_scale'])
    drive(pilot, history, cue_toward([10., 0., 0.]), 5, velocity=(4., 0., 0.))
    assert pilot.host.flow_gain == pytest.approx(.375)
    s = senses(position=(0., 0., 5.), velocity=(4., 0., 0.))
    _, modified = pilot.update(s, np.zeros(3), None, None, 20.)
    assert float(modified['vel_world'][0, 0]) == pytest.approx(1.5)
    assert float(modified['vel_world'][0, 2]) == pytest.approx(float(s['vel_world'][0, 2]))
    with pytest.raises(ValueError):
        FastRaceCue(SENSOR, CameraPoseHistory(), 8., velocity_scale=0.)
