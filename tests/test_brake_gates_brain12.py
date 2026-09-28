"""brain-12 parts of the surrogate gate runner (haltere.train.brake_gates): declared regression16 courses, the round-5
pilot in full_pilot, the round-4b windows, the hairpin harness and the frozen brain-12 gate file."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from haltere.train import brake_gates as bg

REPO = Path(__file__).resolve().parents[1]


def test_brain12_parts_are_separate_and_run_only_when_declared():
    assert not set(bg.PARTS12) & set(bg.PARTS+bg.NEW_PARTS)
    eleven = json.loads((REPO/'configs/brain11_gates.json').read_text(encoding='utf-8'))
    assert not set(bg.PARTS12) & set(eleven['tests'])


def _cmd(monkeypatch, tmp_path, spec):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen['cmd'] = cmd
        (tmp_path/'h'/'regression16.json').write_text(json.dumps(dict(controllers=dict(brain=dict(
            finished=1, crashed=0, mean_speed=1., stick_chatter=.1, mean_finish_s_own=1., switch_bins={},
            lateral_switch_bins={})))))
        raise StopIteration
    monkeypatch.setattr(subprocess, 'run', fake_run)
    monkeypatch.setattr(bg, 'sha256', lambda path: 'pinned')
    ctl = SimpleNamespace(kind='brain', path='ckpt.pt')
    with pytest.raises(StopIteration):
        bg.regression16(ctl, dict(harness='harness.py', harness_sha256='pinned', **spec), tmp_path/'h', REPO)
    return seen['cmd']


def test_regression16_command_is_unchanged_without_declared_courses(monkeypatch, tmp_path):
    base = _cmd(monkeypatch, tmp_path, {})
    assert '--seeds' not in base and '--groups' not in base and '--sim-seed' not in base
    fresh = _cmd(monkeypatch, tmp_path, dict(seeds=[7300, 7301], groups=[['flat', 0.0], ['steep', 0.4]], sim_seed=37))
    assert fresh[:len(base)] == base
    assert fresh[len(base):] == ['--seeds', '7300', '7301', '--groups', 'flat:0.0,steep:0.4', '--sim-seed', '37']


def test_nearest_along_finds_the_closest_live_point():
    path = np.array([[0., 0.], [0., -1.], [0., -2.], [0., -3.]])
    assert bg._nearest_along(path, np.array([[.1, -2.2], [5., -.4], [0., -9.]])).tolist() == [2, 0, 3]


def test_hairpin_variants_are_declared():
    ctl = SimpleNamespace(kind='pd', meta={}, cfg=None, brain=None, contract=None)
    with pytest.raises(ValueError):
        bg.hairpin_tests(ctl, {}, dict(contract_brain='fast_velocity_brain_v1', contract_pd='fast_velocity_pd_v1',
                                       set=dict(turn_deg=[40.]), variants=['sometimes'], seconds=1., sim_seed=1))


def test_brain12_gate_file_is_frozen_with_fresh_sets():
    gates, sha = bg.load_gates(REPO/'configs/brain12_gates.json', require_frozen=True)
    assert gates['sha256'] == sha and gates['version'] == 1
    t = gates['tests']
    fresh = set(range(7300, 7308))
    training = {1000*r+i for r in range(10) for i in range(40)}     # DAgger collection seeds of every round
    old_gates = set(range(3000, 3008)) | set(range(6000, 6012)) | set(range(900, 908)) | set(range(5000, 5008))
    for part in ('rollout_speed', 'in_course_caps', 'evaluation8', 'in_course_caps2', 'regression16'):
        assert set(t[part]['seeds']) == fresh and t[part]['sim_seed'] == 37
    assert t['full_pilot']['sets'] == ['flat:7300-7307', 'steep:7300-7307', 'hill:7400-7411']
    assert t['full_pilot']['motor_assist'] is True
    assert not (fresh | set(range(7400, 7412))) & (training | old_gates | set(range(7100, 7108)) | set(range(7200, 7216)))
    assert {t['in_course_caps']['cap_seed'], t['in_course_caps2']['cap_seed']} == {9101, 9102}
    # brain-11 gate definitions kept
    eleven, _ = bg.load_gates(REPO/'configs/brain11_gates.json', require_frozen=True)
    for part in ('hover', 'live', 'swaps', 'r4_windows', 'swaps_delayed', 'capped_turns', 'accelerate'):
        assert t[part] == eleven['tests'][part]
    for name in ('G1_cap_step', 'G2_sustained', 'G3v2_live_swaps_latency', 'G5_in_course_caps', 'G8_regressions',
                 'G9_r4_live_windows', 'G10_capped_turns', 'G11_accelerate', 'G12_in_course_caps2'):
        assert gates['gates'][name]['checks'] == eleven['gates'][name]['checks']
    assert [c['path'] for c in gates['gates']['G13_full_pilot']['checks']].count('full_pilot.overall.contact_s') == 1
    assert {'G14_r4b_live_windows', 'G15_right_turns_accelerate', 'G16_hairpins'} <= set(gates['gates'])
    assert t['capped_turns_right']['cases']['turns_deg'] == [0.0, -45.0, -90.0]
    assert t['accelerate_right']['bearings_deg'] == [-30.0, -60.0]
