"""The standard flight scorer must accept actual visual-runner telemetry."""
import csv

import numpy as np
import pytest

from haltere.liftoff.flightlog import describe, load_log, score_log


@pytest.mark.parametrize('shadow', [True, False])
@pytest.mark.parametrize('explicit_phase', [True, False])
def test_visual_logs_keep_observed_inputs_and_assistance(tmp_path, shadow, explicit_phase):
    path = tmp_path / 'visual.csv'
    cols = ['ts', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz',
            'omega_x', 'omega_y', 'omega_z', 'in_thr', 'in_roll', 'in_pitch', 'in_yaw',
            'raw_thr', 'raw_yaw', 'yaw', 'command_yaw', 'shadow', 'pilot_assisted',
            'neural_search', 'search_height_reference', 'pilot_mode']
    if explicit_phase:
        cols.append('phase')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for _ in range(2):  # both attempts have their own three-second arming ramp
            for i in range(1000):
                t = i * .01
                row = dict.fromkeys(cols, 0.)
                row.update(ts=50+t, x=2*t, z=2., vx=2., qw=1., shadow=shadow,
                           pilot_assisted=True, neural_search=False,
                           search_height_reference=float('nan'), pilot_mode=2,
                           in_yaw=.125, raw_yaw=.7, yaw=.1, command_yaw=.5)
                if explicit_phase:
                    row['phase'] = t
                writer.writerow(row)
        f.write('60,1,2\n')  # capture interrupted mid-row
    log = load_log(str(path))
    assert len(log['ts']) == 2000
    assert np.allclose(log['phase'][[0, 999, 1000, 1999]], [0, 9.99, 0, 9.99])
    assert np.all(log['in_yaw'] == .125)  # score game response, never substituted commands
    results = score_log(str(path))
    assert len(results) == 2
    for r in results:
        assert r['speed_median'] == pytest.approx(2.)
        assert r['airborne_s'] == pytest.approx(6.99, abs=.02)
        assert r['pilot_assistance'] == 'rabbit'
        assert r['control_mode'] == ('shadow: no control output' if shadow else 'live control')
        assert 'gates_through' not in r  # motion metrics cannot invent race completion
    assert results[0]['control_mode'] in describe('visual', results)


def test_empty_visual_log_is_not_a_flight(tmp_path):
    path = tmp_path / 'empty.csv'
    path.write_text('ts,x,y,z,shadow\n')
    assert score_log(str(path)) == []


def test_unknown_log_schema_has_actionable_error(tmp_path):
    path = tmp_path / 'wrong.csv'
    path.write_text('ts,x,y,z\n1,0,0,2\n')
    with pytest.raises(ValueError, match='missing telemetry columns'):
        score_log(str(path))
