"""FastRaceCue clearance response: an optional forward time-to-contact / clearance input.

These tests pin the declared stopping-distance policy (`ClearanceConfig`; the fast
pilot's default is the TTC-graded policy, tests/test_fast_race_cue_ttc.py): stopping-speed cap along the looming
ray, confirmation, hold and release, dead reckoning of sample age, terrain climb,
no-evidence handling) and check in the measured surrogate that a perfect but delayed
clearance stops the drone before a wall. None of this is flight evidence.
"""
import json
from collections import deque

import numpy as np
import pytest
import torch

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (ClearanceConfig, ClearanceGovernor, FastCueConfig, FastRaceCue,
                                           stopping_speed)
from tests.test_fast_race_cue import cue_toward, drive, single_thread  # noqa: F401 (fixture)
from tests.test_visual_assistance import SENSOR, senses

CC = ClearanceConfig()
AHEAD = cue_toward([10., 0., 0.])


def step(pilot, history, now, *, clearance=None, velocity=(6., 0., 0.), position=(0., 0., 5.), cue=AHEAD):
    s = senses(position=position, velocity=velocity)
    history.append(now, list(position), s['quat'][0].numpy())
    pilot.update(s, [0., 0., 0.], dict(race_cue=dict(cue)), now-.05, now, clearance=clearance)
    return pilot.velocity_command.copy()


def cruising(speed=6., steps=300):
    """A pilot that already flies straight ahead at `speed` (perfect plant)."""
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed, reference_speed=speed, clearance_config=ClearanceConfig())
    drive(pilot, history, AHEAD, steps, velocity=(speed, 0., 0.), height=5.)
    return pilot, history, 10.+steps*.01


def wall(now, distance, speed=6., below=.5):
    return dict(time=now, ttc=distance/speed, distance=distance, below_fraction=below)


@pytest.mark.parametrize('distance', [.2, .5, 1., 3., 8., 30.])
def test_stopping_speed_is_the_kinematic_bound(distance):
    a, L, m = 10., .15, .5
    v = stopping_speed(distance, a, L, m)
    assert v >= 0
    if distance <= m:
        assert v == 0.
    else:
        assert v*L+v*v/(2*a)+m == pytest.approx(distance)
    assert stopping_speed(distance+.1, a, L, m) >= v


def test_config_validation():
    for bad in (dict(deceleration=0.), dict(margin=float('nan')), dict(confirm=1.5), dict(confirm=4),
                dict(terrain_fraction=1.2),
                dict(terrain_on_s=.3, terrain_full_s=.4), dict(blind_after_s=0.)):
        with pytest.raises(ValueError):
            ClearanceConfig(**bad)
    assert ClearanceConfig(blind_after_s=.5).blind_after_s == .5


def test_without_clearance_input_nothing_changes():
    runs = []
    for with_none in (False, True):
        history = CameraPoseHistory()
        pilot = FastRaceCue(SENSOR, history, 6.)
        rows = []
        for k in range(200):
            now = 10.+k*.01
            s = senses(position=(0., 0., 5.), velocity=(min(6., k*.05), 0., 0.))
            history.append(now, [0., 0., 5.], s['quat'][0].numpy())
            kwargs = dict(clearance=None) if with_none else {}
            pilot.update(s, [0., 0., 0.], dict(race_cue=dict(AHEAD)), now-.05, now, **kwargs)
            rows.append(np.r_[pilot.velocity_command, pilot.feedforward, pilot.pilot.sight_yaw])
        runs.append(np.array(rows))
        assert pilot.clearance is None and pilot.metadata()['clearance_response'] is None
    np.testing.assert_array_equal(runs[0], runs[1])


def test_one_distant_sample_waits_for_confirmation_but_an_urgent_one_brakes():
    pilot, history, now = cruising()
    before = step(pilot, history, now, clearance=wall(now-.05, 2.4))   # ttc 0.4 s, not urgent
    assert pilot.clearance.cap is None and not pilot.clearance_braking
    command = step(pilot, history, now+.06, clearance=wall(now+.01, 2.4-.36))
    assert pilot.clearance_braking and command[0] < before[0]
    urgent, h2, t2 = cruising()
    step(urgent, h2, t2, clearance=wall(t2-.05, 1.2))                   # ttc 0.2 s < urgent_ttc_s
    assert urgent.clearance_braking


def test_cap_bounds_the_speed_along_the_ray_and_keeps_the_lateral_request():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., clearance_config=ClearanceConfig())
    side = dict(u=.005, v=.5, edge=True)                                 # side state: 75 deg left of the heading
    drive(pilot, history, side, 200, plant='static', velocity=(6., 0., 0.))
    lateral_before = pilot.velocity_command[1]
    assert pilot.velocity_command[0] > .4
    now = 12.
    for k in range(60):                                                  # a wall 0.55 m ahead (urgent)
        t = now+k*.01
        step(pilot, history, t, clearance=wall(t-.05, .55) if k % 5 == 0 else None, cue=side)
    cap = pilot.clearance.cap
    assert pilot.state == 'side' and pilot.clearance_braking
    assert cap == pytest.approx(stopping_speed(.55, CC.deceleration, CC.latency, CC.margin))
    assert pilot.velocity_command[0] <= cap+1e-6
    assert pilot.velocity_command[1] == pytest.approx(lateral_before, abs=.05)


def test_sample_age_is_dead_reckoned_with_odometry():
    gov = ClearanceGovernor()
    # two samples captured at x = 0 place a wall 3.6 m ahead along +x
    gov.ingest(11.0, .6, 3.6, .5, [0., 0., 5.], [1., 0., 0.], 6.)
    gov.ingest(11.05, .6, 3.6, .5, [0., 0., 5.], [1., 0., 0.], 6.)
    lateral = ClearanceGovernor()
    for t in (11.0, 11.05):
        lateral.ingest(t, .6, 3.6, .5, [0., 0., 5.], [1., 0., 0.], 6.)
    # 0.2 s later the drone has moved 1.2 m along the ray: the wall is 2.4 m away now
    cap, ray, _ = gov.limits([1.2, 0., 5.], [6., 0., 0.], 11.25, .01, 3.5)
    assert cap == pytest.approx(stopping_speed(3.6-1.2, CC.deceleration, CC.latency, CC.margin))
    np.testing.assert_allclose(ray, [1., 0., 0.])
    # motion across the ray does not shorten the distance along it
    cap_lateral, *_ = lateral.limits([0., 1.2, 5.], [0., 6., 0.], 11.25, .01, 3.5)
    assert cap_lateral == pytest.approx(stopping_speed(3.6, CC.deceleration, CC.latency, CC.margin))


def test_hold_then_release_at_the_declared_rate():
    gov = ClearanceGovernor()
    for t in (0., .055):
        gov.ingest(t, .4, 2.4, .5, [0., 0., 0.], [1., 0., 0.], 6.)
    cap0, *_ = gov.limits([0., 0., 0.], [6., 0., 0.], .06, .01, 3.5)
    for t in (.11, .165):                                                # clear evidence arrives
        gov.ingest(t, 10., 60., .5, [0., 0., 0.], [1., 0., 0.], 6.)
    caps = [gov.limits([0., 0., 0.], [6., 0., 0.], .17+k*.01, .01, 3.5)[0] for k in range(40)]
    held = [c for k, c in enumerate(caps) if .17+k*.01 < .06+CC.hold_s-1e-9]
    assert held and all(c == pytest.approx(cap0) for c in held)
    rising = np.diff([c for c in caps if c is not None])
    assert (rising <= CC.release*.01+1e-9).all() and rising.max() > 0


def test_stale_future_and_repeated_samples_are_ignored_and_bad_values_raise():
    pilot, history, now = cruising()
    step(pilot, history, now, clearance=wall(now-.4, 1.))               # older than max_age_s
    step(pilot, history, now+.01, clearance=wall(now+.2, 1.))           # from the future
    assert pilot.clearance.counts['samples'] == 0
    step(pilot, history, now+.02, clearance=wall(now, 3.))
    step(pilot, history, now+.03, clearance=wall(now, 3.))              # the same capture again
    assert pilot.clearance.counts['samples'] == 1
    for bad in (dict(time=now+.04, ttc=-1., distance=None), dict(time=now+.05, ttc=None, distance=float('nan')),
                dict(time=now+.06, ttc=.5, distance=3., below_fraction=1.5), dict(ttc=.5)):
        with pytest.raises(ValueError):
            step(pilot, history, now+.07, clearance=bad)


def test_no_evidence_is_neither_free_space_nor_an_obstacle():
    pilot, history, now = cruising()
    for k in range(50):                                                  # low texture only: nothing changes
        step(pilot, history, now+k*.02, clearance=dict(time=now+k*.02-.05, ttc=None, distance=None))
    assert pilot.clearance.cap is None and pilot.velocity_command[0] == pytest.approx(6., abs=1e-6)
    assert pilot.clearance.counts['no_evidence'] == 50
    # after a confirmed wall, loss of evidence keeps the dead-reckoned memory while the drone is near it
    now += 1.
    for j in range(3):   # seen in `window` samples: a sustained wall
        step(pilot, history, now+j*.055, clearance=wall(now-.05+j*.055, .8), velocity=(1., 0., 0.))
    now += .11
    held = stopping_speed(.8, CC.deceleration, CC.latency, CC.margin)
    for k in range(150):   # too slow for looming: no evidence, the samples are kept up to memory_max_s
        step(pilot, history, now+.06+k*.01, velocity=(.2, 0., 0.),
             clearance=dict(time=now+.01+k*.01, ttc=None, distance=None) if k % 5 == 0 else None)
    assert pilot.clearance.cap == pytest.approx(held) and pilot.velocity_command[0] <= held+1e-6
    for k in range(150, 300):   # beyond memory_max_s the samples expire and the cap is released gradually
        step(pilot, history, now+.06+k*.01, velocity=(.2, 0., 0.),
             clearance=dict(time=now+.01+k*.01, ttc=None, distance=None) if k % 5 == 0 else None)
    assert pilot.clearance.cap > held


def test_terrain_below_the_path_climbs_instead_of_braking():
    pilot, history, now = cruising()
    ttc = CC.terrain_full_s+.25                                          # inside the proportional band
    for k in range(40):
        t = now+k*.01
        clearance = dict(time=t-.05, ttc=ttc, distance=6.*ttc, below_fraction=.9) if k % 5 == 0 else None
        command = step(pilot, history, t, clearance=clearance)
    # terrain samples age in time (the median sample is 0.05-0.2 s old here); the floor keeps its peak
    up = FastCueConfig().vertical_up
    floor = lambda x: up*np.clip((CC.terrain_on_s-x)/(CC.terrain_on_s-CC.terrain_full_s), 0, 1)
    assert floor(ttc)-1e-9 <= pilot.clearance.climb <= floor(ttc-.2)+1e-9
    assert command[2] > 1. and pilot.clearance.cap is None
    assert command[0] == pytest.approx(6., abs=1e-3), 'terrain alone does not brake'
    # a very short terrain TTC is braked for as well
    urgent, h2, t2 = cruising()
    for k in range(20):
        t = t2+k*.01
        step(urgent, h2, t, clearance=dict(time=t-.05, ttc=.2, distance=1.2, below_fraction=.9) if k % 5 == 0 else None)
    assert urgent.clearance_braking


def test_command_rates_with_clearance():
    rng = np.random.default_rng(3)
    pilot, history, now = cruising()
    previous = pilot.velocity_command.copy()
    top_h = max(FastCueConfig().command_acceleration, CC.brake_slew)
    top_v = max(FastCueConfig().vertical_command_acceleration, CC.terrain_climb_acceleration)
    for k in range(400):
        t = now+k*.01
        clearance = None
        if k % 5 == 0:
            ttc = [None, .2, .4, .8, 10.][int(rng.integers(5))]
            clearance = dict(time=t-.05, ttc=ttc, distance=None if ttc is None else ttc*6.,
                             below_fraction=float(rng.uniform(0, 1)))
        command = step(pilot, history, t, clearance=clearance, velocity=tuple(previous+rng.normal(0, .2, 3)))
        assert np.linalg.norm(command[:2]-previous[:2]) <= top_h*.01+1e-6
        assert abs(command[2]-previous[2]) <= top_v*.01+1e-6
        previous = command


def test_launch_ignores_clearance_and_blind_cap_is_opt_in():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6.)
    rows = drive(pilot, history, AHEAD, 5, height=.2)
    s = senses(position=(0., 0., .2), velocity=(0., 0., 1.))
    history.append(10.1, [0., 0., .2], s['quat'][0].numpy())
    pilot.update(s, [0., 0., 0.], dict(race_cue=dict(AHEAD)), 10.05, 10.1, clearance=wall(10.08, .3))
    assert pilot.state == 'launch' and pilot.clearance is None
    blind, h2, t2 = cruising()
    blind.clearance_config = ClearanceConfig(blind_after_s=.5, blind_speed=3.)
    for k in range(120):
        t = t2+k*.01
        step(blind, h2, t, clearance=dict(time=t-.05, ttc=None, distance=None) if k % 5 == 0 else None)
    assert blind.clearance.blind and blind.velocity_command[0] < 4.
    assert blind.metadata()['clearance_response']['counts']['blind_engagements'] == 1


def test_metadata_declares_the_clearance_response():
    pilot, history, now = cruising()
    step(pilot, history, now, clearance=wall(now-.05, 2.))
    step(pilot, history, now+.06, clearance=wall(now+.01, 1.7))
    meta = json.loads(json.dumps(pilot.metadata()))
    response = meta['clearance_response']
    assert response['parameters']['blind_after_s'] is None                # inf is serialized as None (off)
    assert response['counts']['brake_engagements'] == 1 and 'brake' in response['status_seconds']


def surrogate_wall(distance, *, delay, clearance=True, speed=6., start_speed=3.25, seconds=4., side_at=None,
                   clearance_config=None):
    """Measured surrogate + FastMotorPD; a wall `distance` ahead; perfect TTC at 18 Hz, `delay` late.
    `clearance_config` defaults to the stopping-distance policy (`ClearanceConfig`)."""
    from haltere.brain.motor_baseline import FastMotorPD
    from haltere.liftoff.fast_rehearsal import hud_marker
    from haltere.sim.identified import IdentifiedSim
    from haltere.vision.camera import Camera
    from tests.test_fast_motor_pd import CAL, load_profile
    profile = load_profile()
    sim = IdentifiedSim(profile, CAL)
    state = sim.hover(1, 3.)
    state.quad.vel[0] = torch.tensor([start_speed, 0., 0.])
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed, reference_speed=speed,
                        clearance_config=clearance_config or ClearanceConfig())
    pilot.launching = False
    motor = FastMotorPD(profile, CAL)
    queue = deque(motor.command(sim.sensors(state), torch.zeros(1, 3)).clone() for _ in range(3))
    motor.reset()
    pending, cues, next_frame, detection, capture, sample = deque(), deque(), 0., None, None, None
    x_wall, min_gap, side = float(state.quad.pos[0, 0])+distance, np.inf, False
    for k in range(int(seconds/.01)):
        now = k*.01
        position = state.quad.pos[0].numpy().astype(float)
        velocity = state.quad.vel[0].numpy().astype(float)
        quaternion = state.quad.quat[0].numpy().astype(float)
        history.append(now, position, quaternion)
        side = side or (side_at is not None and x_wall-position[0] <= side_at)
        if now >= next_frame:
            cue = dict(u=.015, v=.5, edge=True) if side else hud_marker(
                camera, position+np.array([40., 0., 0.]), position, quaternion)
            cue['aim_u'] = cue['u']
            gap = x_wall-position[0]
            forward = velocity[0]
            ttc = gap/forward if forward > 1.5 else None                # the estimator's 'slow' gives no evidence
            cues.append((now+.06, now, cue))                                  # ring detection latency
            pending.append((now+delay, dict(time=now, ttc=ttc, distance=None if ttc is None
                                            else ttc*np.linalg.norm(velocity), below_fraction=.5)))
            next_frame = now+1/18.
        while cues and cues[0][0] <= now:
            _, capture, cue = cues.popleft()
            detection = dict(race_cue=cue)
        while pending and pending[0][0] <= now:
            _, sample = pending.popleft()
        sensors = sim.sensors(state)
        pilot.update(sensors, sensors['gyro'][0].numpy(), detection, capture, now,
                     clearance=sample if clearance else None)
        sample = None
        action = motor.command(sensors, torch.tensor(pilot.velocity_command, dtype=torch.float32)[None],
                               torch.tensor(pilot.feedforward, dtype=torch.float32)[None])
        queue.append(torch.tensor(pilot.command(action[0].numpy()), dtype=torch.float32)[None])
        state = sim.step(state, queue.popleft())
        min_gap = min(min_gap, x_wall-float(state.quad.pos[0, 0]))
        if min_gap < 0:
            return dict(contact=True, speed=float(state.quad.vel[0].norm()), min_gap=min_gap, pilot=pilot)
    return dict(contact=False, speed=0., min_gap=min_gap, pilot=pilot)


@pytest.mark.parametrize('delay', [.1, .2])
def test_measured_surrogate_stops_before_a_wall_with_delayed_perfect_ttc(delay):
    free = surrogate_wall(10., delay=delay, clearance=False)
    assert free['contact'] and free['speed'] > 5.
    result = surrogate_wall(10., delay=delay)
    assert not result['contact'] and result['min_gap'] > .1
    assert result['pilot'].clearance.counts['brake_engagements'] >= 1


def test_measured_surrogate_hairpin_side_turn_needs_the_clearance_cap():
    # The Minus Two crash shape: the ring leaves the image edge 1.56 m before a wall at ~6 m/s.
    assert surrogate_wall(10., delay=.2, clearance=False, side_at=1.56)['contact']
    assert not surrogate_wall(10., delay=.2, side_at=1.56)['contact']
