import pytest

from haltere.liftoff.flight_monitor import CsvTail, ProgressMonitor


def row(t, x=0.):
    return dict(wall=1000+t, ts=t, x=x, y=0., z=0., vx=0., vy=0., vz=0.)


def test_stall_requires_complete_continuous_window():
    monitor = ProgressMonitor(window_s=2.)
    for t in (0., .5, 1., 1.5):
        monitor.update(row(t))
    assert not monitor.snapshot(now=1001.5)['stall_detected']
    monitor.update(row(2.))
    assert monitor.snapshot(now=1002.)['state'] == 'stalled'
    assert monitor.snapshot(now=1002.)['finish_confirmed'] is None
    monitor.update(row(2.5, 1.1))
    assert not monitor.snapshot(now=1002.5)['stall_detected']


def test_reset_gap_and_stale_log_are_distinct_from_stall():
    monitor = ProgressMonitor(window_s=1.)
    for t in (0., .5, 1.):
        monitor.update(row(t))
    assert monitor.snapshot(now=1010.)['state'] == 'telemetry_stale'
    monitor.update(row(3.))
    assert not monitor.snapshot(now=1003.)['stall_detected']
    assert monitor.gaps == 1
    monitor.update(row(0.))
    assert monitor.resets == 1
    assert monitor.snapshot(now=1000.)['elapsed_s'] == 0


def test_invalid_values_are_not_movement_evidence():
    monitor = ProgressMonitor()
    monitor.update(row(0., float('nan')))
    monitor.update({})
    assert monitor.snapshot()['state'] == 'waiting_for_telemetry'
    assert monitor.invalid_rows == 2
    with pytest.raises(ValueError):
        ProgressMonitor(window_s=float('nan'))


def test_tail_waits_for_partial_rows_and_rejects_reused_log(tmp_path):
    path = tmp_path/'flight.csv'
    tail = CsvTail(path)
    assert tail.read() == []
    path.write_bytes(b'wall,ts,x\r\n1,2,')
    assert tail.read() == []
    with path.open('ab') as output:
        output.write(b'3\r\n4,5,6\n')
    assert tail.read() == [dict(wall='1', ts='2', x='3'), dict(wall='4', ts='5', x='6')]
    assert tail.read() == []
    path.write_bytes(b'wall,ts,x\n')
    with pytest.raises(RuntimeError, match='truncated'):
        tail.read()
