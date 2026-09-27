"""brain-11 training options (all off by default): climbing synthetic caps, capped-turn brake weights, hill courses and the
deployed pilot in DAgger rollouts."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haltere.liftoff.fast_rehearsal import hill_course, synthetic_course
from haltere.train.fast_motor_tracking import (SyntheticCaps, SyntheticCapsConfig, brake_mask, caps_record,
                                               collection_courses)

OLD_EVENT_KEYS = {'onset', 'target', 'kind', 'hold', 'ray', 'closing', 'reached', 'release', 'end'}


def pilot(state='cue', speed=6.):
    return SimpleNamespace(launching=False, state=state, speed=speed)


def run(caps, seconds=6., velocity=(6., 0., 0.)):
    out = []
    for k in range(int(seconds/.01)):
        cap, ray, climb = caps.limits(np.zeros(3), np.asarray(velocity), k*.01, .01, 3.5)
        out.append((cap, climb))
    return out


def test_default_caps_record_and_event_stream_are_the_brain10_ones():
    config = SyntheticCapsConfig()
    record = caps_record(config)
    assert set(record) == set(asdict(config))-{'climb_share', 'climb', 'climb_s'}
    assert caps_record(None) is None
    assert set(caps_record(SyntheticCapsConfig(climb_share=.5))) == set(asdict(config))
    every = SyntheticCapsConfig(rate_per_min=6000., absolute_share=1., absolute=(2., 2.), hold_s=(1., 1.))
    caps = SyntheticCaps(every, np.random.default_rng(3), pilot())
    ticks = run(caps)
    assert caps.events and all(set(e) == OLD_EVENT_KEYS for e in caps.events)
    assert all(climb == 0. for _, climb in ticks)


def test_climbing_caps_request_their_climb_from_onset_for_their_duration_until_the_release():
    config = SyntheticCapsConfig(rate_per_min=6000., absolute_share=1., absolute=(2., 2.), hold_s=(3., 3.),
                                 climb_share=1., climb=(1., 1.), climb_s=(.8, .8))
    caps = SyntheticCaps(config, np.random.default_rng(3), pilot())
    ticks = run(caps, 3.)
    event = caps.events[0]
    assert event['climb'] == 1. and abs(event['climb_until']-event['onset']-.8) < 1e-9
    climbing = [k*.01 for k, (_, climb) in enumerate(ticks) if climb > 0]
    assert abs(climbing[0]-event['onset']) < 1e-9 and climbing[-1] < event['climb_until'] <= climbing[-1]+.011
    assert caps.climb == 0.
    short = SyntheticCapsConfig(rate_per_min=6000., absolute_share=1., absolute=(2., 2.), hold_s=(.1, .1),
                                climb_share=1., climb=(1., 1.), climb_s=(5., 5.))
    caps = SyntheticCaps(short, np.random.default_rng(3), pilot())
    ticks = run(caps, 3.)
    first = caps.events[0]
    releasing = [climb for k, (_, climb) in enumerate(ticks) if first['release'] <= k*.01 < first['end']]
    assert releasing and all(climb == 0 for climb in releasing)
    assert any(climb > 0 for k, (_, climb) in enumerate(ticks) if k*.01 < first['release'])
    none = SyntheticCapsConfig(rate_per_min=6000., climb_share=1e-9)
    caps = SyntheticCaps(none, np.random.default_rng(3), pilot())
    assert all(climb == 0 for _, climb in run(caps, 2.))


def test_climbing_cap_config_validation():
    for bad in (dict(climb_share=1.5), dict(climb=(1., .5)), dict(climb=(0., 4.)), dict(climb_s=(-1., 1.)),
                dict(climb_share=float('nan'))):
        with pytest.raises(ValueError):
            SyntheticCapsConfig(**bad)


def test_turn_brakes_are_off_by_default_and_count_only_speed_over_the_request_magnitude():
    velocity = torch.tensor([[5.5, 0., 0.]]*4)
    a = np.radians(60.)
    request = torch.tensor([[4., 0., 0.],                               # aligned cap: a brake either way
                            [5.5*np.cos(a), 5.5*np.sin(a), 0.],         # uncapped 60 deg turn: geometry only
                            [3.5*np.cos(a), 3.5*np.sin(a), 0.],         # capped 60 deg turn
                            [3.5*np.cos(a), 3.5*np.sin(a), 1.2]], dtype=torch.float32)   # ... while climbing
    assert brake_mask(request, velocity).tolist() == [True, False, False, False]
    assert brake_mask(request, velocity, turn_deg=None).tolist() == [True, False, False, False]
    assert brake_mask(request, velocity, turn_deg=90.).tolist() == [True, False, True, False]
    assert brake_mask(request, velocity, turn_deg=90., level=1.6).tolist() == [True, False, True, True]
    assert brake_mask(request, velocity, turn_deg=45.).tolist() == [True, False, False, False]


def test_collection_courses_default_are_the_synthetic_ones_and_hills_are_a_seeded_share():
    courses, hills = collection_courses(2, 10, .4, 0.)
    assert hills == []
    for i, course in enumerate(courses):
        assert np.array_equal(course, synthetic_course(2000+i, steep=.4))
    courses, hills = collection_courses(2, 10, .4, .3)
    assert len(hills) == 3 and hills == collection_courses(2, 10, .4, .3)[1]
    for i, course in enumerate(courses):
        expected = hill_course(2000+i) if i in hills else synthetic_course(2000+i, steep=.4)
        assert np.array_equal(course, expected)


def test_deployed_pilot_is_the_runners_stack_and_descent_view_for_the_motor_contract():
    from haltere.liftoff import fast_race_cue as frc
    from haltere.liftoff.visual_brain import (load_descent_view, load_lag_turn_declaration, load_vertical_guard,
                                              load_wall_pilot, LAG_TURN_DECLARATION)
    from haltere.train.deployed_pilot import deployed_pilot_kwargs
    kwargs, record = deployed_pilot_kwargs('fast_velocity_brain_v1')
    assert set(kwargs) == {'lag_turn', 'gap_aim', 'vertical_guard', 'turn_first', 'ceiling_guard', 'descent_view'}
    lag = load_lag_turn_declaration(LAG_TURN_DECLARATION)
    assert kwargs['lag_turn'] == frc.lag_turn_for_contract(lag[0], 'fast_velocity_brain_v1')
    assert record['lag_turn']['sha256'] == lag[1]
    wall = load_wall_pilot()
    assert kwargs['turn_first'] == frc.wall_pilot_configs(wall[0], 'fast_velocity_brain_v1')['turn_first']
    assert record['wall_pilot']['sha256'] == wall[1] and record['vertical_guard']['sha256'] == load_vertical_guard()[1]
    assert kwargs['descent_view'] == frc.descent_view_config(load_descent_view()[0])
    assert record['motor_contract'] == 'fast_velocity_brain_v1'
    view_only, record = deployed_pilot_kwargs('fast_velocity_brain_v1', stack=False)
    assert set(view_only) == {'descent_view'} and set(record) == {'descent_view'}
    with pytest.raises(ValueError):
        deployed_pilot_kwargs('unknown')
    json.dumps(record)


def test_a_climbing_synthetic_cap_raises_the_deployed_pilots_vertical_request_while_it_brakes(single_thread):
    """A climbing synthetic cap in the place of the looming governor of the deployed pilot: the request brakes along
    the ray and its vertical part rises to the climb (the governor's terrain-climb path), then returns."""
    from haltere.liftoff.camera_pose import CameraPoseHistory
    from haltere.liftoff.fast_race_cue import FastRaceCue
    from haltere.train.deployed_pilot import deployed_pilot_kwargs
    kwargs, _ = deployed_pilot_kwargs('fast_velocity_brain_v1')
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kwargs)
    ahead = cue_toward([10., 0., 0.])
    drive(pilot, history, ahead, 150, velocity=(6., 0., 0.), height=5.)
    assert pilot.state == 'cue' and pilot.velocity_command[0] > 5.5
    config = SyntheticCapsConfig(rate_per_min=6000., absolute_share=1., absolute=(2., 2.), hold_s=(1., 1.),
                                 ray_jitter_deg=0., climb_share=1., climb=(1., 1.), climb_s=(.6, .6))
    caps = SyntheticCaps(config, np.random.default_rng(0), pilot)
    pilot.clearance = caps
    rows = drive(pilot, history, ahead, 150, velocity=(6., 0., 0.), height=5., start=11.5)
    requests = np.array([r[2] for r in rows])
    assert np.all(requests[60:100, 0] <= 2.+1e-6)                      # braked to the cap along the ray
    assert requests[:55, 2].max() > .95                                  # climbing during the climb
    assert requests[100:, 2].max() < .5                                  # the climb ended after 0.6 s
    assert pilot.descent_view is not None and pilot.turn_first is not None


from tests.test_fast_race_cue import cue_toward, drive, single_thread  # noqa: E402,F401 (fixture)
from tests.test_visual_assistance import SENSOR  # noqa: E402
