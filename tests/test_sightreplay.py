"""The offline replay of the rabbit tracker (haltere.liftoff.sightreplay) on a tiny synthetic `fly --log`.

A kinematic drone flies straight through two arches while the real pilot tracks a synthetic detector; every telemetry
frame is written in the fly --log format. Replaying that log must rebuild the same tracker: the same target ids and
estimates, the same passes, and a score that puts the tracks on the arches that made them.
"""
import csv
import math

import numpy as np
import pytest
import torch

from haltere.liftoff.commands import _fly_log_header, _fly_log_row
from haltere.liftoff.flightlog import load_log
from haltere.liftoff.pilot import LiftoffMapping, TelemetryPilot
from haltere.liftoff.sightpilot import LOG_COLUMNS, SightParams
from haltere.liftoff.sightreplay import (COL, describe, detections_from_log, params_for, replay, score, validate,
                                         write_replayed_log)
from haltere.sim.tasks import HoverTaskConfig
from haltere.vision.camera import Camera
from haltere.vision.model import IN_H, IN_W
from haltere.vision.rehearse import SimClock, frame_from_sim, quat_from_yaw
from haltere.vision.runtime import Detection, detection_geometry

CAM = Camera(640, 360, 200.0, 30.0)
GATES = [{'pos': [20.0, 0.0, 1.2], 'heading': 0.0}, {'pos': [45.0, 0.0, 1.2], 'heading': 0.0}]
DT = 0.01
EPOCH = 1789000000.0        # a wall clock of the size time.time() gives: the log writes it to 0.1 ms


def SIGHT_PARAMS(**kw):
    """The pilot this fixture flies. ``range_corr`` is off: that table is GateNet's range bias, measured on the
    game, and ArchVision below reports the label's width exactly - correcting a bias that is not there would only
    put one in, which the pilot's own width estimate would then have to take back out."""
    return SightParams(v_cruise=2.5, range_corr=None, **kw)


class DummyBrain:
    def weight_matrix(self):
        return torch.zeros(1)

    def init_state(self, B):
        return {}


class ArchVision:
    """The nearest arch ahead, seen at 15 Hz and handed over 60 ms later, stamped with the grab time."""

    def __init__(self, gates, rate=15.0, latency=0.06):
        self.cam = CAM.scaled(IN_W, IN_H)
        self.centres = [np.asarray(g['pos'], dtype=float) + [0.0, 0.0, 1.5] for g in gates]
        self.latest = Detection()
        self.rate, self.latency = rate, latency
        self.queue, self.next_t, self.n = [], None, 0

    def get(self):
        return self.latest

    def update(self, now, pos, R):
        if self.next_t is None:
            self.next_t = now
        if now >= self.next_t - 1e-9:
            self.next_t += 1.0 / self.rate
            det = None
            for c in sorted(self.centres, key=lambda c: np.linalg.norm(c - pos)):
                rel = R.T @ (c - pos)
                d = float(np.linalg.norm(rel))
                px, ok = self.cam.project_body(rel[None])
                u, v = px[0]
                if ok[0] and 4 < u < self.cam.width - 4 and 4 < v < self.cam.height - 4 and 1.5 < d < 45.0:
                    f = self.cam.f
                    du, dv = u - self.cam.width / 2, v - self.cam.height / 2
                    width = f * 4.0 * np.sqrt(f * f + du * du) * np.sqrt(f * f + du * du + dv * dv) / (f * f) / d
                    direction, dist = detection_geometry(self.cam, u, v, width)
                    det = Detection(now, 0.97, u, v, width, direction, dist)
                    break
            if det is None:
                direction, dist = detection_geometry(self.cam, self.cam.width / 2, self.cam.height / 2, 30.0)
                det = Detection(now, 0.02, self.cam.width / 2, self.cam.height / 2, 30.0, direction, dist)
            self.n += 1
            det.frames = self.n
            self.queue.append((now + self.latency, det))
        while self.queue and self.queue[0][0] <= now + 1e-9:
            self.latest = self.queue.pop(0)[1]


def fly_synthetic_log(path, seconds=24.0, speed=2.5, params=None):
    """Write a fly --log of the real pilot tracking the two arches while the drone flies straight through them."""
    pilot = TelemetryPilot(DummyBrain(), HoverTaskConfig(), LiftoffMapping(), 'cpu')
    clock = SimClock(EPOCH)
    pilot.clock = clock
    pilot.vision = ArchVision(GATES)
    pilot.sight = 'rabbit'
    pilot.sight_params = params or SIGHT_PARAMS()
    rows = []
    head = None
    for k in range(int(seconds / DT)):
        t = k * DT
        pos = np.array([speed * max(t - 1.0, 0.0), 0.0, 0.05 + 1.45 * min(t, 1.0)])
        vel = np.array([speed if t > 1.0 else 0.0, 0.0, 1.45 if t < 1.0 else 0.0])
        q = quat_from_yaw(0.0)
        now = clock()
        pilot.vision.update(now, pos, np.eye(3))
        fr = frame_from_sim(t, pos, vel, q, np.zeros(3), np.full(4, 0.5), np.zeros(4), now)
        s = pilot.sensors(fr)
        p = s['pos'][0].numpy().astype(np.float64)
        pilot.last_pos = p
        rel_b = pilot.vision_goal(p)
        pilot.last_brain = pilot.last_cmd = np.zeros(4)
        pilot.last_rel_b = rel_b
        pilot.last_target = p + rel_b
        pilot.last_quat = q
        if head is None:
            head = _fly_log_header(pilot)
        rows.append(_fly_log_row(pilot, fr, now, np.zeros(4), t + 5.0, False))
        clock.t += DT
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    return pilot


@pytest.fixture(scope='module')
def flight(tmp_path_factory):
    path = tmp_path_factory.mktemp('replay') / 'synthetic.csv'
    live = fly_synthetic_log(path)
    log = load_log(str(path))
    res = replay(log, SIGHT_PARAMS(), CAM.scaled(IN_W, IN_H))
    return {'path': str(path), 'log': log, 'res': res, 'live': live}


def test_detections_are_rebuilt_from_the_log_with_the_live_geometry(flight):
    log, cam = flight['log'], CAM.scaled(IN_W, IN_H)
    dets = detections_from_log(log, cam)
    assert 300 < len(dets) < 400                                  # 15 Hz over 24 s, each frame once
    rows = [r for r, _ in dets]
    assert rows == sorted(rows) and len(set(rows)) == len(rows)
    for r, d in dets[:20]:
        direction, dist = detection_geometry(cam, d.u, d.v, d.width_px)
        assert np.allclose(d.direction_body, direction) and abs(d.dist_m - dist) < 1e-9
        assert d.t <= log['wall'][r] and log['det_t'][r] == pytest.approx(d.t)


def test_the_replay_reproduces_the_logged_tracker(flight):
    log, res = flight['log'], flight['res']
    val = validate(log, res)
    assert val['tgt_id_match'] > 0.98 and val['n_passes_match'] > 0.99
    assert val['passes_logged'] == val['passes_replayed'] == val['passes_matched'] == 2
    assert val['tgt_pos_median_any_id'] < 0.05 and val['tgt_within_1m_any_id'] > 0.99
    assert val['rabbit_xy_err_median'] < 0.1
    # the last row: the same tracks, the same passes, the same counters
    C = res['cols']
    for c in ('n_passes', 'n_conf', 'ghosts', 'absorbed', 'rej_elev'):
        assert C[-1, COL[c]] == pytest.approx(log[c][-1])


def test_the_score_puts_the_tracks_on_the_arches_that_made_them(flight):
    sc = score(flight['log'], flight['res'], GATES)
    (att,) = sc['attempts']
    assert att['gates_through'] == [0, 1]
    assert not att['phantoms'] and not att['false_passes'] and not att['missed_passes']
    assert att['target_switches'] <= 2
    for ap in att['approaches']:
        assert ap['fragments'] >= 1 and ap['crossing']['through']
        assert len(ap['passes']) == 1 and abs(ap['passes'][0]['dt']) < 1.0
        near = ap['err_by_range'][0]                                     # the target estimate inside 5 m
        assert near['n'] > 50 and near['across_abs_median'] < 1.0 and abs(near['along_median']) < 2.0
        assert ap['axis_err_3_12'] is not None and abs(ap['axis_err_3_12']['median']) < 15.0
    for t in sc['tracks']:
        if t['confirmed']:
            assert t['arch'] in (0, 1) and t['assoc'] in ('near', 'seen', 'ray')
    assert describe('synthetic', validate(flight['log'], flight['res']), sc).count('\n') > 3


def test_the_sync_of_a_detection_to_the_step_that_used_it(flight):
    log, cam = flight['log'], CAM.scaled(IN_W, IN_H)
    by_log = detections_from_log(log, cam, 'log')
    first = detections_from_log(log, cam, 'first')
    later = detections_from_log(log, cam, 'next')
    assert [d.t for _, d in by_log] == [d.t for _, d in first] == [d.t for _, d in later]
    assert all(r0 >= r1 for (r0, _), (r1, _) in zip(by_log, first))      # never before the row that shows it
    assert all(r2 == r1 + 1 for (r2, _), (r1, _) in zip(later, first))
    # the log's own mark (det_gap 0 in the step that consumed a fresh frame) is what 'log' follows
    assert all(log['det_gap'][r] == 0.0 for r, _ in by_log)


def test_a_replayed_log_is_a_log(flight, tmp_path):
    out = tmp_path / 'replayed.csv'
    write_replayed_log(flight['path'], str(out), flight['res'])
    back = load_log(str(out))
    src = flight['log']
    assert set(back) == set(src) and len(back['wall']) == len(src['wall'])
    assert np.allclose(back['px'], src['px'])
    assert np.allclose(np.nan_to_num(back['tgt_id'], nan=-1.0),
                       np.nan_to_num(flight['res']['cols'][:, COL['tgt_id']], nan=-1.0))


def test_the_flights_flags_are_parsed_and_the_parameters_validated():
    import argparse

    from haltere.liftoff.sightreplay import FLIGHT_FLAGS, add_cli_args, replay_epilog
    q = argparse.ArgumentParser()
    add_cli_args(q)
    P = params_for(q.parse_args(['log.csv', '--preset', 'w17']))
    assert (P.z_aim, P.up_bias, P.next_min_hits, P.bisector_cap, P.v_cruise) == (0.0, 0.0, 10, 35.0, 2.5)
    P = params_for(q.parse_args(['log.csv', '--preset', 'w18', '--set', 'lead=2.5']))
    assert (P.v_cruise, P.v_gate, P.v_gate_turn, P.flow_min, P.lead) == (3.5, 3.2, 2.8, 0.6, 2.5)
    assert params_for(q.parse_args(['log.csv', '--preset', 'w19'])).flow_alt == 'ground'
    assert params_for(q.parse_args(['log.csv', '--preset', 'w16'])).z_aim == 0.3
    with pytest.raises(SystemExit):                                    # the fly command's own parameter check
        params_for(q.parse_args(['log.csv', '--set', 'flow_ref=0']))
    for k in FLIGHT_FLAGS:
        assert k in replay_epilog()
    assert all(c in LOG_COLUMNS for c in COL)


def test_changed_parameters_change_what_the_replay_tracks(flight):
    """The point of the replay: the same flight flown by a differently tuned tracker."""
    res = replay(flight['log'], SIGHT_PARAMS(confirm_span=1e6), CAM.scaled(IN_W, IN_H))
    C = res['cols']
    ids = np.nan_to_num(C[:, COL['tgt_id']], nan=-1.0)
    assert np.nanmax(C[:, COL['n_conf']]) == 0 and ids.max() == -1      # nothing confirms, so nothing is flown at
    assert np.nanmax(C[:, COL['n_passes']]) == 0 and not res['events']
    assert np.nanmax(C[:, COL['n_tent']]) >= 1                          # the sightings still make tentative tracks
    assert math.isfinite(validate(flight['log'], res)['tgt_id_match'])

