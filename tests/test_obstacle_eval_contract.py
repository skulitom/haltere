import json
import shutil

import numpy as np
import pytest

from haltere.obstacles import evaluate as ev
from haltere.obstacles import thermal


def test_thresholds_draft_is_not_frozen_and_freeze_is_hash_locked(tmp_path):
    obj, sha = ev.load_thresholds(require_frozen=False)
    assert obj['schema'] == 'haltere.obstacles.thresholds.v1'
    assert obj['timing']['latency_s'] == ev.DEPLOYED_LATENCY_S
    assert obj['E1']['bins_m'] == [[0.5, 2.0], [2.0, 4.0], [4.0, 7.0], [7.0, 16.0]]
    if not obj['frozen']:
        with pytest.raises(RuntimeError):
            ev.load_thresholds()
    path = tmp_path / 'thresholds.json'
    shutil.copy(ev.THRESHOLDS_PATH, path)
    draft = json.loads(path.read_text())
    draft.update(frozen=False, frozen_at=None, sha256=None)
    path.write_text(json.dumps(draft))
    frozen = ev.freeze_thresholds(path)
    assert frozen['frozen'] and frozen['sha256'] == ev.thresholds_sha256(draft)
    again, sha2 = ev.load_thresholds(path)
    assert sha2 == frozen['sha256']
    assert ev.freeze_thresholds(path)['frozen_at'] == frozen['frozen_at']      # idempotent
    changed = json.loads(path.read_text())
    changed['E6']['min_episode_s'] = 0.1
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError):
        ev.load_thresholds(path)
    with pytest.raises(ValueError):
        ev.freeze_thresholds(path)


def test_ledger_scores_each_frozen_version_once(tmp_path):
    led = ev.Ledger(tmp_path / 'ledger.jsonl')
    assert not led.scored('abc', 'F12', 't1')
    led.append('abc', 'F12', 't1', [dict(metric='E1')])
    assert led.scored('abc', 'F12', 't1')
    with pytest.raises(RuntimeError):
        led.append('abc', 'F12', 't1', [])
    led.append('abc', 'F12', 't2', [])           # a new frozen thresholds version is a new score
    led.append('abc', 'F4', 't1', [])
    assert len(led.entries()) == 3
    with pytest.raises(ValueError):
        led.append('', 'F12', 't1', [])


def test_latest_available_is_causal():
    t_wall = np.array([0.0, 0.1, 0.2])
    avail = ev.available_times(t_wall)
    idx = ev.latest_available(avail, [0.0, 0.064, 0.065, 0.17, 1.0])
    assert idx.tolist() == [-1, -1, 0, 1, 2]
    with pytest.raises(ValueError):
        ev.latest_available([0.2, 0.1], [0.3])


def test_prediction_set_contract(tmp_path):
    rows = np.array([3, 7, 9])
    arrays = dict(grid_q50=np.full((3, 18, 32), 5.0, np.float32), fan_p8=np.zeros((3, 4, 9), np.float32),
                  side=np.array([0, 1, 2], np.int8))
    p = ev.PredictionSet('F12-v0', 'model', True, rows, fold='F12', sha256='abc', arrays=arrays).validate(10)
    p.save(tmp_path / 'p.npz')
    q = ev.PredictionSet.load(tmp_path / 'p.npz')
    assert q.name == 'F12-v0' and q.latency_s == ev.DEPLOYED_LATENCY_S and np.array_equal(q.rows, rows)
    assert np.allclose(q.arrays['grid_q50'], 5.0)
    with pytest.raises(ValueError):
        ev.PredictionSet('x', 'model', True, rows, arrays=dict(grid_q50=np.zeros((3, 4, 9)))).validate()
    with pytest.raises(ValueError):
        ev.PredictionSet('x', 'model', True, np.array([2, 1])).validate()
    with pytest.raises(ValueError):
        ev.PredictionSet('B4', 'baseline', True, rows, baseline_id='B4').validate()
    ev.PredictionSet('B4', 'baseline', False, rows, baseline_id='B4').validate()
    assert ev.BASELINES['B4']['causal'] is False


def test_records_carry_provenance_and_environment_flags():
    p = ev.PredictionSet('F12-v0', 'model', True, np.array([0]), fold='F12', sha256='abc')
    r = ev.make_record('E1', pred=p, fold='F12', env='Minus Two', value={'2-4': 1.1}, n=40, n_events=3,
                       thresholds_sha256='t', store_index_sha256='s', labels_manifest_sha256='l', variant='bin 2-4 m')
    assert r['held_out'] is True and r['seen_environment'] is False and r['sealed'] is False
    assert r['predictor']['sha256'] == 'abc' and r['latency_s'] == 0.065 and r['thresholds_sha256'] == 't'
    assert ev.environment_flags('F12', 'Straw Bale') == dict(held_out=False, seen_environment=True, sealed=False)
    assert ev.environment_flags('ALL', 'Minus Two')['seen_environment'] is True
    assert ev.environment_flags('F12', 'Hall 26')['sealed'] is True
    with pytest.raises(ValueError):
        ev.make_record('E9', pred=p, fold='F12', env='Minus Two', value=0, n=0, thresholds_sha256='t',
                       store_index_sha256='s', labels_manifest_sha256=None)


def test_wilson_interval():
    lo, hi = ev.wilson_interval(5, 8)
    assert abs(lo - 0.306) < 0.002 and abs(hi - 0.863) < 0.002
    assert ev.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = ev.wilson_interval(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35


def test_flight_lock_pauses_without_touching_the_file(tmp_path):
    lock = tmp_path / 'FLIGHT_LOCK'
    assert thermal.wait_for_flight_lock(lock, sleep=lambda s: None) == 0.0
    lock.write_text('flying')
    calls = []

    def fake_sleep(s):
        calls.append(s)
        if len(calls) == 3:
            lock.unlink()          # the operator's session ends the flight, not the job

    waited = thermal.wait_for_flight_lock(lock, sleep=fake_sleep, log=lambda m: None)
    assert calls == [20.0, 20.0, 20.0] and waited == 60.0


def test_chunk_guard_limits(tmp_path, monkeypatch):
    monkeypatch.delenv(thermal.FLIGHT_LOCK_ENV, raising=False)
    with pytest.raises(SystemExit):
        thermal.require_flight_lock_path(None)
    monkeypatch.setenv(thermal.FLIGHT_LOCK_ENV, str(tmp_path / 'lock'))
    assert thermal.require_flight_lock_path(None) == tmp_path / 'lock'
    with pytest.raises(ValueError):
        thermal.ChunkGuard(tmp_path / 'lock', gpu=True, chunk_max_s=900)
    import haltere.train.thermal as tt
    monkeypatch.setattr(tt, 'wait_if_hot', lambda *a, **k: 0.0)
    g = thermal.ChunkGuard(tmp_path / 'lock', gpu=True, temperature=lambda: 81.0, log=lambda m: None)
    with pytest.raises(thermal.GpuTooHot):
        g.before_chunk()
    g = thermal.ChunkGuard(tmp_path / 'lock', gpu=True, temperature=lambda: 60.0, log=lambda m: None)
    g.before_chunk()
    assert g.chunks == 1 and not g.chunk_expired() and g.chunk_max_s == 600.0
