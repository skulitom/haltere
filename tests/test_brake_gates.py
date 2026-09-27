"""Pure parts of the brain-09 surrogate gate runner (haltere.train.brake_gates): schedules, freezing, verdicts."""
import json

import numpy as np
import pytest

from haltere.train.brake_gates import _drop, _feedforward, _slew, derived, freeze, gates_sha256, load_gates, verdict


def test_cap_schedule_falls_at_the_slew_without_taper_and_holds():
    speed = _drop(6., 2., 15., 50)
    assert abs(speed[0]-5.85) < 1e-9 and np.allclose(np.diff(speed[:26]), -.15) and np.all(speed[27:] == 2.)
    assert np.all(_drop(2., 3., 15., 5) == 3.)


def test_pilot_slew_tapers_and_reaches_the_goal():
    speed = _slew(0., 6., 400)
    assert abs(speed[0]-.1) < 1e-9 and np.all(np.diff(speed) >= 0) and abs(speed[-1]-6.) < 1e-3
    assert np.max(np.diff(speed)) <= .1+1e-12


def test_feedforward_is_the_pilots_low_passed_request_derivative():
    request = np.zeros((300, 1, 3))
    request[:, 0, 0] = np.minimum(np.arange(300)*.01*10., 6.)   # 10 m/s^2 ramp to 6 m/s
    ff = _feedforward(request)
    assert abs(ff[50, 0, 0]-10.) < .1 and abs(ff[-1, 0, 0]) < 1e-3


def test_freeze_pins_the_content_and_refuses_later_changes(tmp_path):
    path = tmp_path/'gates.json'
    path.write_text(json.dumps(dict(version=1, tests=dict(a=1))))
    obj = freeze(path)
    assert obj['frozen'] and obj['sha256'] == gates_sha256(dict(version=1, tests=dict(a=1)))
    assert load_gates(path, require_frozen=True)[1] == obj['sha256']
    changed = json.loads(path.read_text()); changed['tests']['a'] = 2
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError):
        load_gates(path)


def test_verdict_reads_values_rows_selections_and_where_clauses():
    report = dict(a=dict(b=.2), rows=[dict(window='w1', target=1., v=.1, valid=.1),
                                      dict(window='w1', target=2., v=.5, valid=.1),
                                      dict(window='w2', target=1., v=.9, valid=.9)])
    gates = dict(gates=dict(
        single=dict(checks=[dict(path='a.b', op='le', value=.3)]),
        selected=dict(checks=[dict(rows='rows', select=dict(target=[1.]), field='v', op='le', value=.3,
                                   where=dict(field='valid', op='abs_le', value=.3))]),
        failing=dict(checks=[dict(rows='rows', field='v', op='le', value=.3)]),
        diagnostic=dict(primary=False, checks=[dict(path='a.missing', op='le', value=1.)])))
    v = verdict(report, gates)
    assert v['gates']['single']['passed'] and v['gates']['selected']['passed']
    assert v['gates']['selected']['checks'][0]['excluded_rows'] == 1
    assert not v['gates']['failing']['passed'] and not v['gates']['diagnostic']['passed']
    assert v['passed_primary'] == 2 and v['primary_total'] == 3 and not v['all_primary_passed']


def test_history_gap_compares_live_and_hover_steady_speeds():
    report = dict(hover=dict(sustained=[dict(target=3.5, settled=3.6)]),
                  live=[dict(window='w', target=3.5, steady=4.1), dict(window='w', target=1., steady=1.2)])
    assert derived(report)['history_gap'] == [dict(window='w', target=3.5, gap=.5)]


def test_brain10_gates_are_frozen_and_keep_the_brain09_tests_and_gates():
    b10, sha10 = load_gates('configs/brain10_gates.json', require_frozen=True)
    b09, sha09 = load_gates('configs/brain09_gates.json', require_frozen=True)
    assert b10['version'] == 1 and b10['previous_versions'][0]['sha256'] == sha09
    assert b10['tests'] == b09['tests'] and b10['reference'] == b09['reference']
    for name, gate in b09['gates'].items():
        assert b10['gates'][name] == gate
    assert set(b10['gates'])-set(b09['gates']) == {'G7_smoothness', 'G8_regressions'}
    chatter = {c['path']: c['value'] for c in b10['gates']['G7_smoothness']['checks']}
    assert chatter['regression16.stick_chatter'] <= 1.15*b10['baselines']['brain08']['regression16']['stick_chatter']
