"""The round-5 gate tooling (haltere.obstacles.stale_evidence_gates) on synthetic inputs: the frozen gates file, the
window, path and speed helpers, and the logged-detection verdicts. No log, video or surrogate is read here."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from haltere.obstacles import stale_evidence_gates as seg

ROOT = Path(__file__).resolve().parents[1]


def test_the_gates_file_is_frozen_and_names_the_frozen_declarations(tmp_path):
    gates, digest = seg.load_gates()
    assert digest == gates['sha256'] and gates['version'] == 1
    assert set(gates['declarations']) == {'stale_evidence', 'ring_marker'}
    assert set(gates['development']) == {'minus-fast6-r4b-01', 'straw-brain11cw13-r4b-noassist-02'}
    assert not set(gates['development']) & set(gates['reader']['flights'])
    assert set(gates['development']) <= set(gates['replay_flights'])
    edited = dict(gates, note=gates['note']+' edited')
    path = tmp_path/'g.json'
    path.write_text(json.dumps(edited), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen'):
        seg.load_gates(path)


def arrays(t, cvx, cvy, vx=6., vy=0.):
    n = len(t)
    return dict(t=np.asarray(t, float), cvx=np.asarray(cvx, float), cvy=np.asarray(cvy, float), cvz=np.zeros(n),
                vx=np.full(n, vx), vy=np.full(n, vy))


def test_changed_windows_merge_and_pad():
    t = np.arange(0., 10., .01)
    a = arrays(t, np.full(len(t), 6.), np.zeros(len(t)))
    b = arrays(t, a['cvx'].copy(), a['cvy'].copy())
    b['cvx'][(t > 2.) & (t < 2.2)] = 5.
    b['cvx'][(t > 2.5) & (t < 2.6)] = 5.
    b['cvy'][(t > 6.) & (t < 6.1)] = .5
    w = seg.changed_windows(a, b, merge_s=.5, pre_s=.5, post_s=1.)
    assert len(w) == 2
    assert w[0][0] == pytest.approx(2.01-.5, abs=.011) and w[0][1] == pytest.approx(2.59+1., abs=.011)
    assert seg.changed_windows(a, a) == []


def test_travel_speedup_measures_the_request_along_the_velocity():
    t = np.arange(0., 1., .01)
    a = arrays(t, np.full(len(t), 5.), np.zeros(len(t)))
    b = arrays(t, np.full(len(t), 5.4), np.full(len(t), 3.))
    su = seg.travel_speedup(a, b)
    assert np.allclose(su, .4)
    slow = arrays(t, a['cvx'], a['cvy'], vx=.5)
    assert np.isnan(seg.travel_speedup(slow, b)).all()


def test_path_compare_lateral_and_progress():
    n = 100
    pos = np.zeros((n, 3))
    pos[:, 0] = np.linspace(0., 6., n)
    vel = np.zeros((n, 3))
    vel[:, 0] = 6.
    shifted = pos.copy()
    shifted[:, 1] += np.linspace(0., .3, n)
    shifted[:, 0] -= np.linspace(0., .6, n)
    lat, lost = seg.path_compare(pos, vel, shifted)
    assert lat == pytest.approx(.3) and lost == pytest.approx(.1)


def test_side_of_the_contact_point():
    pos = np.zeros((10, 3))
    pos[:, 0] = np.arange(10.)
    vel = np.tile([1., 0., 0.], (10, 1))
    assert 'path left' in seg.side_of(pos, vel, np.array([5., -.5]))
    assert 'path right' in seg.side_of(pos, vel, np.array([5., .5]))


def test_logged_verdicts_match_candidates_and_the_overlay(tmp_path, monkeypatch):
    """Two logged detections: one on a candidate the rule rejects (not confirmed by the overlay), one on a kept candidate
    the overlay confirms; a third far from any frame is unmatched."""
    wall0 = 1000.
    rows = []
    for k, (u, v) in enumerate([(.6, .78), (.49, .84), (.5, .5)]):
        rows.append(dict(wall=wall0+k, frame_time=10.+k, capture_time=10.+k, cue_u=u, cue_v=v, cue_edge=False,
                         phase=float(k)))
    pd.DataFrame(rows).to_csv(tmp_path/'f.csv', index=False)
    monkeypatch.setattr('haltere.obstacles.store_build.EPOCH_PERCENTILE', 50.)
    frames = dict(t_wall=np.array([wall0, wall0+1., wall0+5.]),
                  cand_frame=np.array([0, 0, 1]), cand_u=np.array([.6+2/1280, .2, .49]),
                  cand_v=np.array([.78, .2, .84]), cand_ok=np.array([False, True, True]),
                  overlay=np.array([[np.nan, np.nan], [.49+3/1280, .84], [np.nan, np.nan]]))
    v = seg.logged_verdicts('f', tmp_path, frames, dict(tolerance_px=12, match_dt_s=.06))
    assert list(v['verdict']) == ['rejected', 'kept', 'unmatched']
    assert list(v['confirmed']) == [False, True, None]
