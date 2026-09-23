"""The standard flight scorer must accept actual visual-runner telemetry."""
import csv
import json

import numpy as np
import pytest

from haltere.liftoff.flightlog import describe, load_log, score_log


@pytest.mark.parametrize('shadow', [True, False])
@pytest.mark.parametrize('explicit_phase', [True, False])
@pytest.mark.parametrize('motor', [None, 'brain', 'pd'])
def test_visual_logs_keep_observed_inputs_and_assistance(tmp_path, shadow, explicit_phase, motor):
    path = tmp_path / 'visual.csv'
    cols = ['ts', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz',
            'omega_x', 'omega_y', 'omega_z', 'in_thr', 'in_roll', 'in_pitch', 'in_yaw',
            'raw_thr', 'raw_yaw', 'yaw', 'command_yaw', 'shadow', 'pilot_assisted',
            'neural_search', 'search_height_reference', 'pilot_mode', 'geometry_control_status']
    if explicit_phase:
        cols.append('phase')
    if motor:
        cols.append('motor_controller')
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
                row['geometry_control_status']='disabled'
                if explicit_phase:
                    row['phase'] = t
                if motor:
                    row['motor_controller'] = motor
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
        assert r['control_mode'] == ('shadow: no control output' if shadow else
                                    'PD motor baseline; brain in shadow' if motor == 'pd' else 'live control')
        assert 'gates_through' not in r  # motion metrics cannot invent race completion
        assert not r['image_geometry_control']
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


def test_elapsed_flight_time_uses_timestamps_when_control_ticks_are_missed():
    from haltere.liftoff.flightlog import score_attempt
    # Most steps remain 10 ms, but occasional missed ticks add real elapsed time.
    t = np.cumsum(np.where(np.arange(1000) % 5 == 0, .03, .01))
    log = {k: np.zeros(len(t)) for k in ('px','py','pz','vx','vy','vz','qw','qx','qy','qz',
           'wx','wy','wz','in_thr','in_roll','in_pitch','in_yaw')}
    log.update(ts=t, phase=t+5, pz=np.full(len(t),2.), qw=np.ones(len(t)), vx=np.ones(len(t)))
    scored = score_attempt(log, np.arange(len(t)))
    assert scored['airborne_s'] == pytest.approx(t[-1]-t[0])
    assert scored['airborne_s'] > len(t)*np.median(np.diff(t))+3


def test_oracle_attribution_and_flight_below_launch_height_are_preserved():
    from haltere.liftoff.flightlog import score_attempt
    t=np.arange(1000)*.01
    log={k:np.zeros(len(t)) for k in ('px','py','pz','vx','vy','vz','qw','qx','qy','qz',
           'wx','wy','wz','in_thr','in_roll','in_pitch','in_yaw')}
    log.update(ts=t,phase=t+5,pz=np.linspace(2.,-5.,len(t)),qw=np.ones(len(t)),
               shadow=np.zeros(len(t)),pilot_assisted=np.ones(len(t)),pilot_kind=np.full(len(t),3.),
               motor_controller=np.full(len(t),'pd',dtype=object))
    result=score_attempt(log,np.arange(len(t)))
    assert result['airborne_s']==pytest.approx(t[-1])
    assert result['pilot_assistance']=='oracle-route'
    assert result['runtime_route_oracle'] and not result['autonomous_evaluation_eligible']
    assert 'PRIVILEGED' in result['control_mode']


@pytest.mark.parametrize('sidecar_case', ['matching', 'mismatch', 'after_short_reset'])
def test_terminal_impact_is_not_lost_when_guard_stops_before_logging(tmp_path, sidecar_case):
    path = tmp_path / 'stopped.csv'
    cols = ['ts', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz',
            'omega_x', 'omega_y', 'omega_z', 'in_thr', 'in_roll', 'in_pitch', 'in_yaw',
            'shadow', 'pilot_assisted', 'phase']
    with path.open('w', newline='') as source:
        writer = csv.DictWriter(source, fieldnames=cols)
        writer.writeheader()
        for i in range(250):
            # A short final reset is omitted from scored attempts. Its impact
            # must not be attributed to the preceding, longer attempt.
            j = i - 240 if sidecar_case == 'after_short_reset' and i >= 240 else i
            row = dict.fromkeys(cols, 0.)
            row.update(ts=j*.01, x=j*.02, z=2., vx=2., qw=1., shadow=False,
                       pilot_assisted=True, phase=5+j*.01)
            writer.writerow(row)
    impact = dict(timestamp=.1 if sidecar_case == 'after_short_reset' else 2.5,
                  acceleration_mps2=50., unexplained_mps2=40.)
    path.with_suffix('.json').write_text(json.dumps(dict(
        ticks=900 if sidecar_case == 'mismatch' else 250, impact=impact,
        stop_reason='Impact detected from flight motion')))
    (result,) = score_log(str(path))
    assert result['collisions'] == []  # the CSV ends just before the terminal sample
    if sidecar_case == 'matching':
        assert result['terminal_impact'] == impact
        assert 'terminal impact recorded' in describe('stopped', [result])
        assert 'stop: Impact detected' in describe('stopped', [result])
    else:
        assert 'terminal_impact' not in result
        assert 'stop_reason' not in result
