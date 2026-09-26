"""Braking data for the fast brain distillation (synthetic governor caps, slow legs, brake weights, metrics).

Training-only machinery in haltere.train.fast_motor_tracking; nothing here runs at flight time.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import FastRaceCue
from haltere.train.fast_motor_tracking import (SyntheticCaps, SyntheticCapsConfig, brake_mask, brake_metrics,
                                               brake_weights, slow_leg_speeds)
from tests.test_fast_race_cue import cue_toward, drive, single_thread  # noqa: F401 (fixture)
from tests.test_visual_assistance import SENSOR

# one event on the first eligible tick: a 2 m/s cap held 1 s, ray exactly along the velocity
EVERY_TICK = SyntheticCapsConfig(rate_per_min=6000., absolute_share=1., absolute=(2., 2.), hold_s=(1., 1.),
                                 ray_jitter_deg=0.)


def pilot_stub(state='cue', launching=False, speed=6.):
    return SimpleNamespace(state=state, launching=launching, speed=speed)


def step(caps, velocity, seconds, start=0., dt=.01):
    rows = []
    for k in range(int(round(seconds/dt))):
        now = start+k*dt
        rows.append((now, *caps.limits([0., 0., 5.], velocity, now, dt, 3.5)))
    return rows


def test_config_validation():
    for bad in (dict(rate_per_min=0.), dict(absolute=(4., 1.)), dict(relative=(.2, 1.5)), dict(hold_s=(-1., 2.)),
                dict(absolute_share=1.5), dict(ray_jitter_deg=95.), dict(brake_rate=float('nan')), dict(states=())):
        with pytest.raises(ValueError):
            SyntheticCapsConfig(**bad)


def test_no_cap_while_launching_or_outside_the_declared_states_or_below_min_speed():
    for pilot, velocity in ((pilot_stub(launching=True), [6., 0., 0.]), (pilot_stub(state='search'), [6., 0., 0.]),
                            (pilot_stub(state='above'), [6., 0., 0.]), (pilot_stub(), [2.9, 0., 0.])):
        caps = SyntheticCaps(EVERY_TICK, np.random.default_rng(0), pilot)
        rows = step(caps, velocity, 1.)
        assert all(cap is None and ray is None and climb == 0. for _, cap, ray, climb in rows)
        assert caps.events == [] and caps.status == 'none'


def test_cap_falls_at_the_governor_brake_rate_holds_then_releases_and_ends():
    pilot = pilot_stub()
    caps = SyntheticCaps(EVERY_TICK, np.random.default_rng(0), pilot)
    rows = []
    for k in range(600):
        now = k*.01
        rows.append((now, *caps.limits([0., 0., 5.], [6., 0., 0.], now, .01, 3.5)))
        if caps.events and caps.events[0]['end'] is not None:
            pilot.state = 'search'          # no further event
    assert len(caps.events) == 1
    e = caps.events[0]
    assert e['onset'] == 0. and e['target'] == 2. and e['kind'] == 'absolute' and abs(e['closing']-6.) < 1e-9
    np.testing.assert_allclose(e['ray'], [1., 0., 0.])
    caps_seen = np.array([np.nan if cap is None else cap for _, cap, _, _ in rows])
    # 6 -> 2 m/s at 8 m/s^2: 0.5 s
    assert abs(e['reached']-.5) < .015
    assert np.all(np.diff(caps_seen[:50]) <= 1e-9) and abs(caps_seen[10]-(6.-8.*.11)) < 1e-6
    assert abs(e['release']-(e['reached']+1.)) < .015
    held = caps_seen[int(e['reached']/.01)+1:int(e['release']/.01)]
    assert np.allclose(held, 2.)
    # release at 3 m/s^2 until above the pilot speed + 1 m/s (7 m/s): 5/3 s
    assert abs(e['end']-(e['release']+5./3.)) < .03
    assert np.isnan(caps_seen[int(e['end']/.01)+1]) and caps.cap is None and caps.event is None
    assert caps.counts['synthetic_events'] == 1


def test_rays_stay_within_the_jitter_of_the_travel_direction_and_targets_follow_the_declared_ranges():
    config = SyntheticCapsConfig(rate_per_min=6000., hold_s=(0., 0.), release=1000.)
    caps = SyntheticCaps(config, np.random.default_rng(3), pilot_stub())
    velocity = np.array([3., 4., 0.])   # 5 m/s
    step(caps, velocity, 20.)
    assert len(caps.events) > 20
    heading = velocity/5.
    for e in caps.events:
        angle = np.degrees(np.arccos(np.clip(e['ray'] @ heading, -1., 1.)))
        assert angle <= 15.+1e-6 and e['ray'][2] == 0.
        if e['kind'] == 'absolute':
            assert 1. <= e['target'] <= 4.
        else:
            assert .2*5. <= e['target'] <= .7*5.+1e-9
    kinds = {e['kind'] for e in caps.events}
    assert kinds == {'absolute', 'relative'}


def test_event_rate_is_about_the_declared_rate():
    config = SyntheticCapsConfig(rate_per_min=8., hold_s=(0., 0.), release=1000., absolute=(1., 1.))
    caps = SyntheticCaps(config, np.random.default_rng(5), pilot_stub())
    step(caps, [6., 0., 0.], 600.)
    per_min = len(caps.events)/10.
    assert 5. < per_min < 11.


def test_the_fast_pilot_brakes_its_request_along_a_synthetic_cap(single_thread):
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6.)
    ahead = cue_toward([10., 0., 0.])
    drive(pilot, history, ahead, 150, velocity=(6., 0., 0.), height=5.)
    assert pilot.state == 'cue' and pilot.velocity_command[0] > 5.5 and pilot.clearance is None
    caps = SyntheticCaps(EVERY_TICK, np.random.default_rng(0), pilot)
    pilot.clearance = caps
    rows = drive(pilot, history, ahead, 150, velocity=(6., 0., 0.), height=5., start=11.5)
    requests = np.array([r[2][0] for r in rows])
    e = caps.events[0]
    assert abs(e['onset']-11.5) < 1e-9 and abs(e['closing']-requests[0]) < .5
    # the request follows the cap down at up to brake_slew (15 m/s^2), no taper, and stays at the 2 m/s target
    assert np.all(np.diff(requests[:40]) >= -15.*.01-1e-6)
    assert np.all(requests[60:int((e['release']-11.5)/.01)] <= 2.+1e-6)
    assert pilot.clearance_time.get('brake', 0.) > .5 and caps.counts['brake_engagements'] >= 1


def test_brake_mask_selects_aligned_level_at_speed_over_speed_samples():
    request = torch.tensor([[4., 0., 0.],    # aligned, level, 6 m/s flown: over-speed 2 -> yes
                            [4., 0., 1.],    # climbing request -> no
                            [4., 2., 0.],    # 26.6 deg off the track -> no
                            [4., 0., 0.],    # flown 4.5: over-speed 0.5 -> no
                            [2., 0., 0.],    # flown 2.9 (< 3 m/s) -> no
                            [0., 0., 0.]])   # no horizontal request -> no
    velocity = torch.tensor([[6., 0., 0.], [6., 0., 0.], [6., 0., 0.], [4.5, 0., 0.], [2.9, 0., 0.], [6., 0., 0.]])
    assert brake_mask(request, velocity).tolist() == [True, False, False, False, False, False]
    weights = brake_weights(request, velocity, 5.)
    assert torch.allclose(weights.mean(), torch.tensor(1.)) and torch.allclose(weights[0]/weights[1], torch.tensor(5.))


def test_slow_leg_speeds_are_a_seeded_share_within_the_range():
    assert slow_leg_speeds(10, 0., 2.5, 4.5, 6., 100) is None
    speeds = np.array(slow_leg_speeds(10, .3, 2.5, 4.5, 6., 100))
    slow = speeds < 6.
    assert slow.sum() == 3 and np.all((speeds[slow] >= 2.5) & (speeds[slow] <= 4.5))
    assert slow_leg_speeds(10, .3, 2.5, 4.5, 6., 100) == speeds.tolist()
    assert slow_leg_speeds(10, .3, 2.5, 4.5, 6., 101) != speeds.tolist()


def test_brake_metrics_classes_come_from_the_request_side():
    dt, T = .01, 800
    t = np.arange(T)*dt
    request = np.zeros((T, 2, 3)); velocity = np.zeros((T, 2, 3))
    request[:, :, 0] = 6.; velocity[:, :, 0] = 6.
    # drone 0: a cap to 2 m/s from 4 s; the request drops at once, the drone decelerates at 2 m/s^2
    capped = t >= 4.
    request[capped, 0, 0] = 2.
    velocity[capped, 0, 0] = np.maximum(2., 6.-2.*(t[capped]-4.))
    # drone 1: sustained 3.5 m/s request, flown at 4.0 (over-speed 0.5 along the track)
    request[:, 1, 0] = 3.5; velocity[:, 1, 0] = 4.
    active = np.ones((T, 2), bool)
    events = [[dict(onset=4., target=2., ray=np.array([1., 0., 0.]), closing=6., reached=4.5, release=7., end=None)], []]
    m = brake_metrics(t, request, velocity, active, events, [6., 3.5], 6.)
    # cruise: drone 0 before the cap (excess 0), drone 1 all along (0.5), both from 3 s
    assert m['cruise_excess_nominal_mps'] == 0. and m['cruise_excess_slow_mps'] == .5
    # cap window 4.5-7 s: excess (v - 2) with v = max(2, 6 - 2 (t - 4)) -> within 0.5 m/s at 4 + 1.75 s
    window = (t >= 4.5) & (t < 7.)
    expected = float((velocity[window, 0, 0]-2.).mean())
    assert abs(m['cap_excess_mps']-round(expected, 3)) < 1e-9 and m['cap_events'] == 1
    assert m['cap_within_events'] == 1 and abs(m['cap_within_median_s']-1.75) < .015 and m['cap_within_1s'] == 0.
    # slow ticks (< 4.2 m/s requested): drone 0 in its cap, drone 1 aligned; no turn ticks
    assert m['slow_cap_s'] > 0 and m['slow_aligned_s'] > 0 and m['slow_turn_s'] == 0. and m['slow_excess_aligned_mps'] == .5
