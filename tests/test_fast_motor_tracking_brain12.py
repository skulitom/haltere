"""brain-12 training options (all off by default): mirrored collection courses, the motor assist in the deployed pilot,
descent weights and cap rate ranges."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haltere.train.fast_motor_tracking import (COLLECTION_DEFAULTS, SyntheticCaps, SyntheticCapsConfig, caps_record,
                                               collection_courses, descent_mask, descent_weights, mirror_course,
                                               side_balance_weights)

REPO = Path(__file__).resolve().parents[1]


def pilot(state='cue', speed=6.):
    return SimpleNamespace(launching=False, state=state, speed=speed)


def run(caps, seconds=6., velocity=(6., 0., 0.)):
    out = []
    for k in range(int(seconds/.01)):
        cap, ray, climb = caps.limits(np.zeros(3), np.asarray(velocity), k*.01, .01, 3.5)
        out.append(cap)
    return out


def test_mirrored_courses_follow_the_originals_and_turn_the_other_way():
    plain, hills = collection_courses(2, 4, .4, .5)
    both, both_hills = collection_courses(2, 4, .4, .5, mirror=True)
    assert len(both) == 8 and all(np.array_equal(a, b) for a, b in zip(plain, both[:4]))
    assert both_hills == hills+[4+i for i in hills]
    for a, m in zip(both[:4], both[4:]):
        assert np.array_equal(m[:, 0], a[:, 0]) and np.array_equal(m[:, 1], -a[:, 1]) and np.array_equal(m[:, 2], a[:, 2])
    def turn(c):
        d = np.diff(c[:, :2], axis=0)
        return d[:-1, 0]*d[1:, 1]-d[:-1, 1]*d[1:, 0]
    assert np.allclose(turn(both[0]), -turn(both[4]))
    assert np.array_equal(mirror_course(mirror_course(plain[1])), plain[1])


def test_default_collection_courses_are_unchanged():
    a, ha = collection_courses(1, 5, .4, .2)
    b, hb = collection_courses(1, 5, .4, .2, mirror=False)
    assert ha == hb and all(np.array_equal(x, y) for x, y in zip(a, b))
    assert COLLECTION_DEFAULTS['mirror_courses'] is False and COLLECTION_DEFAULTS['motor_assist'] is False


def test_rate_ranges_are_off_by_default_and_do_not_change_the_default_event_stream():
    config = SyntheticCapsConfig()
    assert set(caps_record(config)) == set(asdict(config))-{'climb_share', 'climb', 'climb_s', 'brake_rate_max',
                                                               'release_max'}
    every = dict(rate_per_min=6000., absolute_share=1., absolute=(2., 2.), hold_s=(.5, .5))
    base = SyntheticCaps(SyntheticCapsConfig(**every), np.random.default_rng(5), pilot())
    ticks = run(base)
    assert all('brake_rate' not in e and 'release_rate' not in e for e in base.events)
    # the cap falls at 8 m/s^2 and rises at 3 m/s^2 (the governor's defaults)
    drops = np.diff([c for c in ticks[:40] if c is not None])
    assert np.isclose(drops[drops < 0].min(), -.08)


def test_rate_ranges_draw_per_event_rates_and_use_them():
    config = SyntheticCapsConfig(rate_per_min=6000., absolute_share=1., absolute=(1., 1.), hold_s=(.3, .3),
                                 brake_rate_max=15., release_max=10.)
    record = caps_record(config)
    assert record['brake_rate_max'] == 15. and record['release_max'] == 10.
    caps = SyntheticCaps(config, np.random.default_rng(7), pilot(speed=6.))
    ticks = run(caps, 4.)
    first = caps.events[0]
    assert 8. <= first['brake_rate'] <= 15. and 3. <= first['release_rate'] <= 10.
    k0 = int(round(first['onset']/.01))
    falling = np.diff([ticks[k] for k in range(k0, k0+20) if ticks[k] is not None])
    assert np.isclose(falling[falling < 0].min(), -first['brake_rate']*.01)
    kr = int(round(first['release']/.01))
    rising = np.diff([ticks[k] for k in range(kr, kr+10) if ticks[k] is not None])
    assert np.isclose(rising.max(), first['release_rate']*.01)
    for bad in (dict(brake_rate_max=5.), dict(release_max=2.), dict(brake_rate_max=-1.)):
        with pytest.raises(ValueError):
            SyntheticCapsConfig(**bad)


def test_brain12_caps_file_declares_the_ranges():
    caps = json.loads((REPO/'configs/brain12_caps.json').read_text(encoding='utf-8'))
    config = SyntheticCapsConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in caps.items()
                                    if not k.startswith('_')})
    assert config.brake_rate == 8. and config.brake_rate_max == 15. and config.release == 3. and config.release_max == 10.
    eleven = json.loads((REPO/'configs/brain11_caps.json').read_text(encoding='utf-8'))
    assert {k: v for k, v in caps.items() if k in eleven and not k.startswith('_')} == \
        {k: v for k, v in eleven.items() if not k.startswith('_')}


def test_descent_mask_selects_aligned_fast_descents_at_speed():
    velocity = torch.tensor([[5.5, 0., -1.5]]*5)
    request = torch.tensor([[6., 0., -1.5],      # aligned fast descent at speed
                            [6., 0., -.5],       # gentle descent
                            [3., 0., -1.5],      # slow request
                            [0., 6., -1.5],      # 90 deg off the velocity
                            [6., 0., 1.]])       # climb
    assert descent_mask(request, velocity, 6.).tolist() == [True, False, False, False, False]
    slow = torch.tensor([[2., 0., -1.5]])
    assert not descent_mask(torch.tensor([[6., 0., -1.5]]), slow, 6.).any()
    w = descent_weights(request, velocity, 6., 3.)
    assert torch.isclose(w.mean(), torch.tensor(1.)) and w[0] == 3*w[1]


def test_side_balance_gives_both_sides_equal_weight_per_bin():
    a = np.radians(40.)
    velocity = torch.tensor([[5., 0., 0.]]*8)
    right = [5*np.cos(a), -5*np.sin(a), 0.]
    left = [5*np.cos(a), 5*np.sin(a), 0.]
    request = torch.tensor([right]*6+[left]*2, dtype=torch.float32)
    w = side_balance_weights(request, velocity, max_gain=4.)
    assert torch.isclose(w.mean(), torch.tensor(1.))
    assert torch.isclose(w[:6].sum(), w[6:].sum()) and torch.isclose(w[6]/w[0], torch.tensor(3.))
    capped = side_balance_weights(request, velocity, max_gain=1.5)
    assert torch.isclose(capped[6]/capped[0], torch.tensor(1.5/(8/12)))
    # aligned, slow or one-sided samples keep weight one
    one_sided = side_balance_weights(torch.tensor([right, [5., 0., 0.]], dtype=torch.float32),
                                     torch.tensor([[5., 0., 0.]]*2))
    assert torch.allclose(one_sided, torch.ones(2))


def test_motor_assist_in_the_deployed_pilot_is_the_round5_declaration():
    from haltere.train.deployed_pilot import deployed_pilot_kwargs
    kwargs, record = deployed_pilot_kwargs('fast_velocity_brain_v1', motor_assist=True)
    assert 'motor_assist' in kwargs and record['motor_assist']['version'] == 3
    plain, plain_record = deployed_pilot_kwargs('fast_velocity_brain_v1')
    assert 'motor_assist' not in plain and 'motor_assist' not in plain_record
    pd, _ = deployed_pilot_kwargs('fast_velocity_pd_v1', motor_assist=True)
    assert 'motor_assist' not in pd


def test_synthetic_turns_turn_both_ways_hold_and_return():
    from haltere.train.fast_motor_tracking import SyntheticTurnConfig, SyntheticTurns
    config = SyntheticTurnConfig(rate_per_min=600., angle_deg=(60., 60.), rate=(1., 1.), hold_s=(.5, .5))
    turns = SyntheticTurns(config, 64, 3)
    seen_left = seen_right = False
    for k in range(400):
        angle, omega, active = turns.step(k*.01, .01, np.zeros(64, bool), np.full(64, 5.))
        assert np.all(np.abs(angle) <= np.radians(60.)+1e-9) and np.all(np.abs(omega) <= 1.+1e-9)
        assert np.all(angle[~active] == 0.)
        seen_left |= bool((angle > np.radians(59.9)).any())
        seen_right |= bool((angle < -np.radians(59.9)).any())
    assert seen_left and seen_right and 0 < turns.left < turns.count
    idle = SyntheticTurns(config, 4, 3)
    for k in range(200):
        angle, _, active = idle.step(k*.01, .01, np.ones(4, bool), np.full(4, 5.))   # launching: never
        assert not active.any()
    slow = SyntheticTurns(config, 4, 3)
    assert not slow.step(0., .01, np.zeros(4, bool), np.zeros(4))[2].any()
    for bad in (dict(angle_deg=(0., 30.)), dict(rate=(2., 1.)), dict(angle_deg=(30., 170.)), dict(rate_per_min=0.)):
        with pytest.raises(ValueError):
            SyntheticTurnConfig(**bad)


def test_rotate_request_turns_the_request_and_adds_the_rotation_to_the_feedforward():
    from haltere.train.fast_motor_tracking import rotate_request
    r, f = rotate_request(np.array([4., 0., -1.]), np.array([1., 0., .5]), np.pi/2, .5)
    assert np.allclose(r, [0., 4., -1.]) and np.allclose(f, [-.5*4., 1., .5])
    same, ff = rotate_request(np.array([4., 0., 0.]), np.zeros(3), 0., 0.)
    assert np.array_equal(same, [4., 0., 0.]) and np.array_equal(ff, np.zeros(3))


def test_lateral_mask_selects_turns_on_either_side_at_speed():
    from haltere.train.fast_motor_tracking import lateral_mask, lateral_weights
    v = torch.tensor([[5., 0., 0.]]*4)
    a = np.radians(30.)
    request = torch.tensor([[5., 0., 0.], [5*np.cos(a), 5*np.sin(a), 0.], [5*np.cos(a), -5*np.sin(a), 0.],
                            [.5, .1, 0.]], dtype=torch.float32)
    assert lateral_mask(request, v).tolist() == [False, True, True, False]
    w = lateral_weights(request, v, 3.)
    assert torch.isclose(w.mean(), torch.tensor(1.)) and torch.isclose(w[1]/w[0], torch.tensor(3.))


def test_descent_sink_relief_reduces_only_the_sink_beyond_the_free_part():
    from haltere.train.fast_motor_tracking import LabelTeacher, descent_sink_relief
    request = torch.tensor([[6., 0., -2.], [6., 0., -.4], [6., 0., 1.], [0., 0., -3.]])
    shaped = descent_sink_relief(request, .1)
    assert torch.allclose(shaped[:, 2], torch.tensor([-1.85, -.4, 1., -2.75]))
    assert torch.equal(shaped[:, :2], request[:, :2]) and torch.equal(descent_sink_relief(request, 0.), request)
    assert LabelTeacher.__init__.__defaults__[-1] == 0.
