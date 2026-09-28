"""brain-11 parts of the surrogate gate runner (haltere.train.brake_gates): schedules, logged-window inputs, the frozen
brain-11 gate file."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from haltere.train.brake_gates import (NEW_PARTS, PARTS, _capped_turn, _slew_vec, _turn_cases, _window_inputs,
                                       load_gates, verdict)


def test_capped_turn_falls_at_the_brake_slew_and_turns_at_the_declared_rate():
    plan = _capped_turn(6., 0., 3., 90., 300, brake_slew=15., rate=1.)
    speed = np.linalg.norm(plan, axis=1)
    angle = np.degrees(np.arctan2(plan[:, 1], plan[:, 0]))
    assert abs(speed[0]-5.85) < 1e-9 and np.allclose(speed[20:], 3.)
    assert abs(angle[0]-np.degrees(.01)) < 1e-6 and abs(angle[-1]-90.) < 1e-6
    assert np.all(np.diff(angle) <= np.degrees(.01)+1e-6)
    right = _capped_turn(4., np.pi/2, 4., -45., 100)
    assert np.allclose(np.linalg.norm(right, axis=1), 4.)
    assert abs(np.degrees(np.arctan2(right[-1, 1], right[-1, 0]))-45.) < 1e-6


def test_vector_slew_keeps_the_pilots_acceleration_bound_and_taper():
    plan = _slew_vec([.5, 0.], [0., 5.], 300)
    steps = np.linalg.norm(np.diff(np.vstack(([.5, 0.], plan)), axis=0), axis=1)
    assert steps.max() <= .1+1e-9 and np.allclose(plan[-1], [0., 5.], atol=1e-3)
    chord = plan-np.array([.5, 0.])
    assert np.allclose(chord[:, 0]*5.+chord[:, 1]*.5, 0., atol=1e-9)    # along the straight chord (-0.5, 5)


def test_turn_cases_leave_the_straight_level_cap_to_g1():
    cases = _turn_cases(dict(cases=dict(targets=[3., 4.], turns_deg=[0., 45., 90.], climbs=[0., 1.])))
    assert len(cases) == 10 and (3., 0., 0.) not in cases and (3., 0., 1.) in cases and (4., 90., 1.) in cases


def test_window_inputs_hold_the_last_logged_request_past_the_log():
    req = np.array([[1., 0., 0.], [2., 0., .5], [3., 1., .5]])
    flight = SimpleNamespace(d=[0, 1, 2], req=req, cmds=np.array([[0, 0, 0, .1], [0, 0, 0, .2], [0, 0, 0, .3]]),
                             ff=np.array([[0., 0, 0], [100., 0, 50.], [100., 100, 0.]]))
    request, yaw, ff = _window_inputs(flight, 1, 4)
    assert np.array_equal(request, [[2., 0., .5], [3., 1., .5], [3., 1., .5], [3., 1., .5]])
    assert np.array_equal(yaw, [.2, .3, .3, .3])
    decay = np.exp(-.01/.05)
    assert np.array_equal(ff[:2], flight.ff[1:]) and np.allclose(ff[2], ff[1]*decay) and np.allclose(ff[3], ff[2]*decay)


def test_new_parts_are_separate_from_the_brain09_parts():
    assert not set(PARTS) & set(NEW_PARTS)


def test_verdict_reads_flat_window_rows():
    report = dict(r4_windows=[dict(name='a', controller_speed_excess_mean=.3, logged_minus_live_min_z_span=.01),
                              dict(name='b', controller_speed_excess_mean=.9, logged_minus_live_min_z_span=.02)])
    gates = dict(gates=dict(g=dict(checks=[dict(rows='r4_windows', select=dict(name=['a']),
                                                field='controller_speed_excess_mean', op='le', value=.5),
                                           dict(rows='r4_windows', field='logged_minus_live_min_z_span', op='abs_le',
                                                value=.1)])))
    assert verdict(report, gates)['gates']['g']['passed']
    report['r4_windows'][0]['controller_speed_excess_mean'] = .6
    assert not verdict(report, gates)['gates']['g']['passed']


def test_brain11_gates_are_frozen_and_keep_the_brain10_tests_and_gates():
    b11, sha11 = load_gates('configs/brain11_gates.json', require_frozen=True)
    b10, sha10 = load_gates('configs/brain10_gates.json', require_frozen=True)
    assert b11['version'] == 1 and b11['previous_versions'][0]['sha256'] == sha10
    assert b11['reference'] == b10['reference']
    for name, test in b10['tests'].items():
        assert b11['tests'][name] == test
    for name, gate in b10['gates'].items():
        kept = dict(b11['gates'][name])
        if name == 'G3_live_swaps':
            assert kept.pop('primary') is False     # reported; G3v2 is the primary G3 for brain-11 candidates
        assert kept == gate
    assert set(b11['tests'])-set(b10['tests']) == set(NEW_PARTS)
    assert b11['gates']['G3v2_live_swaps_latency'].get('primary', True)
    primary = [n for n, g in b11['gates'].items() if g.get('primary', True)]
    assert len(primary) == 12 and 'G3_live_swaps' not in primary
    assert b11['selection']['candidates_dir_prefix'] == 'C:/DEV/Haltere/runs/fast-brain-11-'
