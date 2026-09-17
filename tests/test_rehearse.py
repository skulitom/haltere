"""By-sight rehearsal: the synthetic detector's projection and timing, and the closed loop on a tiny brain."""
import math

import numpy as np
import torch

from haltere.vision.camera import Camera, quat_wxyz_to_mat
from haltere.vision.gates import gate_label
from haltere.vision.model import IN_H, IN_W
from haltere.vision.oddcourse import HOME_GATES
from haltere.vision.rehearse import (ClutterModel, DetectorModel, RehearsalOptions, SyntheticGateVision,
                                     arch_collision, arch_views, clutter_objects, phantom_report, quat_from_yaw,
                                     run_rehearsal, summarize)

CAM = Camera(640, 360, 200.0, 30.0)
GATES = [{'pos': [20.0, 0.0, 1.2], 'heading': 0.0}, {'pos': [45.0, 6.0, 1.2], 'heading': 0.3}]
# Straw Bale itself, which is the course the detector's false positives were measured on
COURSE = [{'pos': [x, y, z], 'heading': h, 'gate': i} for i, (x, y, z, h) in enumerate(HOME_GATES)]


def test_arch_views_match_the_dataset_labels():
    rng = np.random.default_rng(0)
    n_vis = 0
    for _ in range(200):
        pos = np.array([rng.uniform(-5, 15), rng.uniform(-6, 6), rng.uniform(0.5, 4)])
        q = quat_from_yaw(rng.uniform(-0.8, 0.8))
        lab = gate_label(pos, q, GATES, CAM)
        if 'gate' not in lab:
            continue
        view = arch_views(pos, q, GATES, CAM)[lab['gate']]
        assert view['visible'] == lab['visible']
        if 'u' in lab:
            assert np.allclose([view['u'], view['v'], view['width_px'], view['dist_m']],
                               [lab['u'], lab['v'], lab['width_px'], lab['dist_m']])
        n_vis += lab['visible']
    assert n_vis > 50


def test_arches_are_seen_from_behind_and_at_an_angle():
    # past the first gate, looking back at it: an arch in view is an arch, whether or not it has been flown
    # through, and the labeller says so too since it started labelling the nearest arch in view (b5cee9c)
    pos, q = np.array([30.0, 0.0, 2.7]), quat_from_yaw(np.pi)
    lab = gate_label(pos, q, GATES[:1], CAM)
    assert lab['visible'] == 1
    v = arch_views(pos, q, GATES[:1], CAM)[0]
    assert v['visible'] == 1 and abs(v['view_deg']) < 1e-6 and abs(v['u'] - 320) < 1e-6
    assert abs(v['u'] - lab['u']) < 1e-6 and abs(v['v'] - lab['v']) < 1e-6
    # 10 m in front of it and 10 m to the side, facing it: 45 deg off its axis, right of the image centre when it
    # is to the right of the drone
    pos = np.array([10.0, 10.0, 2.7])
    yaw = np.arctan2(0.0 - 10.0, 20.0 - 10.0)
    v = arch_views(pos, quat_from_yaw(yaw), GATES[:1], CAM)[0]
    assert v['visible'] == 1 and abs(v['view_deg'] - 45.0) < 1e-6 and abs(v['u'] - 320) < 1e-6
    v2 = arch_views(pos, quat_from_yaw(yaw + 0.2), GATES[:1], CAM)[0]
    assert v2['u'] > v['u']                                    # nose turned left -> the arch moves right
    # the width model: oblique shrink and the range bias
    m = DetectorModel(oblique=0.6, width_bias=())
    assert abs(m.reported_width(40.0, 60.0, 10.0) - 40.0 * (1 - 0.6 * 0.5)) < 1e-9
    assert abs(DetectorModel().reported_width(40.0, 0.0, 12.0) - 40.0 * 0.98) < 1e-9
    assert DetectorModel().find_probability(60.0, 10.0) < DetectorModel().find_probability(0.0, 10.0)


def test_synthetic_detector_timing_and_geometry():
    model = DetectorModel.clean()
    vis = SyntheticGateVision(GATES, CAM, model, np.random.default_rng(1))
    pos, q = np.array([0.0, 0.0, 2.0]), quat_from_yaw(0.05)
    deliveries, now = [], 1000.0
    for _ in range(100):                                       # 1 s at the control rate
        vis.update(now, pos, q)
        det = vis.get()
        if det.frames and (not deliveries or deliveries[-1][1] != det.frames):
            deliveries.append((now, det.frames))
        now += 0.01
    assert len(deliveries) in (14, 15)
    det = vis.get()
    assert abs(det.t - (1000.0 + (det.frames - 1) / 15.0)) < 0.011                 # stamped with the capture time ...
    assert abs(deliveries[-1][0] - det.t - 0.08) < 0.011                             # ... and delivered 80 ms later
    assert vis.latest_gate == 0 and det.p_visible > 0.5                              # the nearer, wider arch
    centre = np.asarray(GATES[0]['pos']) + [0.0, 0.0, 1.5]
    to_gate = (centre - pos) / np.linalg.norm(centre - pos)
    assert np.degrees(np.arccos(quat_wxyz_to_mat(q) @ det.direction_body @ to_gate)) < 0.5
    assert abs(det.dist_m - np.linalg.norm(centre - pos)) < 0.1 * np.linalg.norm(centre - pos)
    assert 0 <= det.u <= IN_W and 0 <= det.v <= IN_H
    # the arch just flown past is still reported (from behind) when it is the widest in view
    vis.reset(now)
    for _ in range(20):
        vis.update(now, np.array([23.0, 0.0, 2.7]), quat_from_yaw(np.pi))
        now += 0.01
    assert vis.latest_gate == 0
    # a noisy detector with total dropout reports nothing plausible
    blind = SyntheticGateVision(GATES, CAM, DetectorModel(miss=1.0, false_pos=0.0), np.random.default_rng(2))
    for _ in range(50):
        blind.update(now, pos, q)
        now += 0.01
    assert blind.get().p_visible < 0.5 and blind.latest_gate == -1


def test_arch_collision():
    g = GATES[:1]
    assert arch_collision(np.array([19.9, 0.5, 1.5]), np.array([20.1, 0.5, 1.5]), g) is None      # through
    assert arch_collision(np.array([19.9, 2.0, 1.5]), np.array([20.1, 2.0, 1.5]), g) == (0, 'post')
    assert arch_collision(np.array([19.9, 0.0, 4.7]), np.array([20.1, 0.0, 4.7]), g) == (0, 'top')
    assert arch_collision(np.array([19.9, 5.0, 1.5]), np.array([20.1, 5.0, 1.5]), g) is None      # beside it


def test_rehearsal_loop_runs_the_real_pilot(tmp_path):
    from haltere.brain.baselines import MLPPolicy
    from haltere.liftoff.flightlog import load_log
    from haltere.sim.tasks import HoverTask
    from haltere.train.bptt import ExperimentConfig

    class TinyBrain(MLPPolicy):
        def weight_matrix(self):
            return torch.zeros(1)

    torch.manual_seed(0)
    cfg = ExperimentConfig.from_dict({'brain': {'model': 'mlp'}, 'train': {'delay_steps': 6}})
    brain = TinyBrain(HoverTask.channels, 4, 16, cfg.brain.dt, cfg.brain.action_tau)
    hover = 2 * (1.0 / cfg.quad.twr) ** (1.0 / cfg.quad.thrust_exp) - 1
    with torch.no_grad():
        brain.readout.bias[0] = float(np.arctanh(hover + 0.1))
    log = tmp_path / 'rehearsal.csv'
    res = run_rehearsal(brain, cfg, GATES, CAM, log, RehearsalOptions(seconds=4.0, verbose=False),
                        DetectorModel.clean())
    assert res['delay_steps'] == 6 and len(res['rows']) == 400
    d = load_log(str(log))
    assert len(d['ts']) == 400 and np.nanmax(d['pz']) > 0.3                 # took off from the ground
    assert any(s.startswith('gate seen') for s in d['status'])              # the pilot followed the synthetic arch
    ages = d['det_age'][np.isfinite(d['det_age'])]
    assert ages.min() >= 0.075 and ages.max() < 0.16                       # sim clock: grabbed 80-150 ms ago
    assert np.allclose(d['det_t'][np.isfinite(d['det_t'])], (d['wall'] - d['det_age'])[np.isfinite(d['det_t'])], atol=2e-4)
    assert np.isnan(d['rb_x']).all()                                        # the legacy pilot leaves the rabbit columns empty
    s = summarize(res, GATES)
    assert s['detector']['detected'] > 40 and 'goal_jumps' in s['attempts'][0]


def test_rehearsal_loop_runs_the_rabbit_pilot(tmp_path):
    from haltere.brain.baselines import MLPPolicy
    from haltere.liftoff.commands import _sight_log_cells
    from haltere.liftoff.flightlog import load_log
    from haltere.liftoff.sightpilot import LOG_COLUMNS, SightParams
    from haltere.sim.tasks import HoverTask
    from haltere.train.bptt import ExperimentConfig

    class TinyBrain(MLPPolicy):
        def weight_matrix(self):
            return torch.zeros(1)

    torch.manual_seed(0)
    cfg = ExperimentConfig.from_dict({'brain': {'model': 'mlp'}, 'train': {'delay_steps': 6}})
    brain = TinyBrain(HoverTask.channels, 4, 16, cfg.brain.dt, cfg.brain.action_tau)
    hover = 2 * (1.0 / cfg.quad.twr) ** (1.0 / cfg.quad.thrust_exp) - 1
    with torch.no_grad():
        brain.readout.bias[0] = float(np.arctanh(hover + 0.1))
    log = tmp_path / 'rabbit.csv'
    res = run_rehearsal(brain, cfg, GATES, CAM, log,
                        RehearsalOptions(seconds=5.0, verbose=False, sight='rabbit', sight_params=SightParams(yaw_rate=3.8),
                                         flow_gain=0.9), DetectorModel.clean())
    d = load_log(str(log))                                                  # every rabbit column is numeric
    assert len(d['ts']) == 500 and all(c in d for c in LOG_COLUMNS)
    assert np.nanmax(d['tgt_hits']) >= 3 and np.nanmax(d['mode']) == 2       # it tracked the arch and targeted it
    assert np.allclose(d['c_yaw'], d['sight_yaw'], atol=1e-4)                # the yaw stick is the rabbit's
    assert np.nanmax(d['flow_gain']) <= 0.9 + 1e-6                          # --flow-gain caps the speed sense
    s = summarize(res, GATES)
    assert 'sight' in s['attempts'][0] and s['attempts'][0]['sight']['sight_errors'] == 0
    # the fly command writes the same values as CSV cells that load_log reads back
    pilot = type('P', (), {})()
    from haltere.liftoff.sightpilot import SightPilot
    host = type('H', (), {})()
    pilot.sightpilot = SightPilot(host, SightParams())
    cells = _sight_log_cells(pilot, None)
    assert len(cells) == len(LOG_COLUMNS) and all(c == '' or float(c) == float(c) for c in cells)


# ------------------------------------------------------------------- the detector's false positives

def test_clutter_is_off_by_default():
    """Every bench number taken before this existed stays comparable: nothing is placed, nothing fires."""
    assert DetectorModel().clutter is None and DetectorModel.clean().clutter is None
    vis = SyntheticGateVision(COURSE, CAM, DetectorModel(), np.random.default_rng(0))
    assert len(vis.clutter) == 0
    now, seen = 1000.0, []
    for k in range(300):
        vis.update(now, np.array([1.0 + 0.1 * k, 0.0, 2.0]), quat_from_yaw(0.0))
        seen.append(vis.latest_gate)
        now += 0.01
    assert vis.stats['clutter'] == 0 and vis.stats['clutter_in_view'] == 0 and -3 not in seen


def test_clutter_stands_beside_the_course_and_clear_of_the_arches():
    m = ClutterModel(n_per_100m=10.0)
    obj = clutter_objects(COURSE, m, seed=3, start=(0.0, 0.0, 0.0))
    assert np.allclose(obj, clutter_objects(COURSE, m, seed=3, start=(0.0, 0.0, 0.0)))     # a seed is a seed
    assert not np.allclose(obj[:, :2].sum(), clutter_objects(COURSE, m, seed=4, start=(0.0, 0.0, 0.0))[:, :2].sum())
    pts = [np.array([0.0, 0.0])] + [np.array(g['pos'][:2], dtype=float) for g in COURSE]
    length = sum(float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:]))
    assert abs(len(obj) - round(10.0 * length / 100.0)) <= 1
    cen = np.array([g['pos'][:2] for g in COURSE], dtype=float)

    def line_d(p, a, b):
        d = b - a
        return abs(float(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0]))) / float(np.linalg.norm(d))

    for x, y, z, w in obj:
        d_arch = float(np.hypot(cen[:, 0] - x, cen[:, 1] - y).min())
        d_line = min(line_d(np.array([x, y]), a, b) for a, b in zip(pts[:-1], pts[1:]))
        assert d_arch >= m.min_arch_m                       # nearer than this the tracker folds it into the arch
        assert d_line <= m.lateral_m[1] + 1e-6 and z >= 0.3
        assert m.size_m[0] <= w <= m.size_m[1]


def test_a_clutter_object_is_reported_where_it_stands():
    """Its box carries the object's own bearing and apparent size, so the pilot's range conversion places every
    sighting of it at one world point (scaled by 4 m / size_m), whatever the camera does. That is the coherence."""
    m = DetectorModel(centre_px=0.0, width_frac=0.0, miss=1.0, false_pos=0.0,
                      clutter=ClutterModel(fire=1.0, n_per_100m=0.0))
    vis = SyntheticGateVision(COURSE, CAM, m, np.random.default_rng(0))
    vis.clutter = np.array([[40.0, 12.0, 3.0, 4.0]])                   # a 4 m object 12 m off the line
    placed = []
    for k in range(12):
        pos = np.array([10.0 + 2.0 * k, 0.0, 2.0])
        q = quat_from_yaw(math.atan2(12.0, 40.0 - pos[0]) + math.radians(6.0 * math.sin(k)))
        det, gate = vis.detect(1000.0 + k, pos, q)
        assert gate == -3 and det.p_visible >= 0.5
        placed.append(pos + det.dist_m * (quat_wxyz_to_mat(q) @ det.direction_body))
    assert np.linalg.norm(np.array(placed) - [40.0, 12.0, 3.0], axis=1).max() < 0.5
    # half the true size reads twice the range -- the same factor from every viewpoint, so still one point
    vis.clutter = np.array([[40.0, 12.0, 3.0, 2.0]])
    for x0 in (28.0, 31.0):
        det, gate = vis.detect(1100.0 + x0, np.array([x0, 0.0, 2.0]), quat_from_yaw(math.atan2(12.0, 40.0 - x0)))
        assert gate == -3
        assert abs(det.dist_m / float(np.linalg.norm([40.0 - x0, 12.0, 1.0])) - 2.0) < 0.03


def _scripted_flight(detector, seed, v=3.2, dt=0.02, wobble_deg=8.0, gates=None):
    """Fly a scripted path along ``gates`` and step the real tracker on what the detector reports: the drone's
    motion is fixed, so this measures the detector alone. The tracks come back as ``liftoff replay-sight`` reads
    them (its own ``_track_table``), so 'phantom' means here exactly what it means on a game flight."""
    from haltere.liftoff.sightpilot import LOG_COLUMNS, SightParams
    from haltere.liftoff.sightreplay import RecordingSightPilot, ReplayHost, _sighting_arches, _track_table
    gates = gates or COURSE
    pts = [np.array([0.0, 0.0, 1.2])] + [np.array(g['pos'], dtype=float) for g in gates]
    phase = np.random.default_rng(seed).uniform(0, 2 * np.pi)
    poses, t = [], 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(2, int(float(np.linalg.norm(b - a)) / (v * dt)))
        psi = math.atan2(b[1] - a[1], b[0] - a[0])
        for i in range(n):
            poses.append((t, a + (b - a) * (i / n),
                          quat_from_yaw(psi + math.radians(wobble_deg) * math.sin(6.0 * t + phase))))
            t += dt
    vision = SyntheticGateVision(gates, CAM, detector, np.random.default_rng(seed), clutter_seed=seed,
                                 start=(0.0, 0.0, 1.2))
    host = ReplayHost(vision, 1.0)
    rec = {'row': -1, 'epoch': 0, 'events': [], 'ends': [], 'sightings': []}
    sp = RecordingSightPilot(host, SightParams(v_cruise=3.5, v_gate=3.2, v_gate_turn=2.8, flow_min=0.6), rec)
    host.sightpilot = sp
    snap = {k: [] for k in ('row', 'epoch', 'id', 'x', 'y', 'z', 'confirmed', 'passed', 'hits')}
    cols = np.full((len(poses), len(LOG_COLUMNS)), np.nan)
    log = {k: np.zeros(len(poses)) for k in ('wall', 'px', 'py', 'pz')}
    for r, (t, p, q) in enumerate(poses):
        now = 1000.0 + t
        rec['row'] = r
        vision.update(now, p, q)
        R = quat_wxyz_to_mat(q)
        host.now, host.last_R, host.last_vel = now, R, np.array([v, 0.0, 0.0])
        host._pose_hist.append((now, p.copy(), R.copy()))
        while host._pose_hist and now - host._pose_hist[0][0] > 1.0:
            host._pose_hist.pop(0)
        sp.goal(p.copy())
        cols[r] = sp.log_values(vision.get())
        log['wall'][r] = now
        log['px'][r], log['py'][r], log['pz'][r] = p
        for T in sp.tracks:
            for key, val in (('row', r), ('epoch', 0), ('id', T.id), ('x', float(T.m[0])), ('y', float(T.m[1])),
                             ('z', float(T.m[2])), ('confirmed', T.confirmed), ('passed', T.passed), ('hits', T.hits)):
                snap[key].append(val)
    res = {'tracks': {k: np.asarray(v2) for k, v2 in snap.items()}, 'cols': cols,
           'epoch': np.zeros(len(poses), dtype=int), 'ends': rec['ends'], 'sightings': rec['sightings'],
           'cam': vision.cam}
    _sighting_arches(res, gates)
    tab = _track_table(log, res, gates, 4.0, 6.0)
    st = vision.stats
    boxes = st['detected'] + st['phantoms'] + st['clutter']
    return {'phantoms': [t for t in tab.values() if t['confirmed'] and t['arch'] is None], 'stats': dict(st),
            'seconds': poses[-1][0], 'false_box_frac': (st['phantoms'] + st['clutter']) / max(boxes, 1),
            'false_per_frame': (st['phantoms'] + st['clutter']) / max(st['frames'], 1),
            'no_arch_points': [np.asarray(s['m'], dtype=float)
                               for s in res['sightings'] if 'm' in s and s['arch'] is None]}


def _grouped_fraction(runs, radius=4.0):
    """The statistic measured on the game flights: what share of the boxes that sat on no arch were placed,
    from different viewpoints, within ``radius`` of two or more others -- i.e. on the same thing."""
    got = tot = 0
    for r in runs:
        groups: list[list] = []
        for q in r['no_arch_points']:
            for g in groups:
                if math.hypot(q[0] - g[0][0], q[1] - g[0][1]) < radius:
                    g[1].append(q)
                    g[0] = np.mean(g[1], axis=0)
                    break
            else:
                groups.append([q.copy(), [q]])
        got += sum(len(g[1]) for g in groups if len(g[1]) >= 3)
        tot += len(r['no_arch_points'])
    return got / max(tot, 1)


_FLIGHTS: dict = {}
_DETECTORS = {'shipped': lambda: DetectorModel(),
              'loose10x': lambda: DetectorModel(false_pos=0.30),
              'clutter': lambda: DetectorModel(false_pos=0.0, clutter=ClutterModel())}


def _flights(name, seeds):
    """The scripted flight is the slow part of these tests; fly each (detector, seed) once."""
    for s in seeds:
        if (name, s) not in _FLIGHTS:
            _FLIGHTS[(name, s)] = _scripted_flight(_DETECTORS[name](), s)
    return [_FLIGHTS[(name, s)] for s in seeds]


def test_a_loose_phantom_cannot_confirm_but_a_clutter_object_does():
    """Why the bench was blind to the 2026-09-17 regression, in one comparison.

    ``false_pos`` puts an INDEPENDENT box at a random place in the image, so no two of them meet in the tracker's
    association gate: at ten times its shipped rate it still builds few, and what it does build has the bare three
    sightings ``confirm_hits`` asks for and then dies. False boxes that come from objects standing in the world
    confirm several times as often per box, accumulate sightings and live long enough to be chosen. The game logs
    say the real detector is the second kind: 62-92 % of its false boxes fall into groups that triangulate
    together."""
    loose, solid = _flights('loose10x', range(3)), _flights('clutter', range(3))

    def per_false_box(runs):
        boxes = sum(r['stats']['phantoms'] + r['stats']['clutter'] for r in runs)
        return sum(len(r['phantoms']) for r in runs) / max(boxes, 1)

    assert np.mean([r['false_per_frame'] for r in loose]) > np.mean([r['false_per_frame'] for r in solid])
    assert per_false_box(solid) >= 2 * per_false_box(loose)      # more phantoms, out of FEWER false boxes
    # (measured 2.7x on these three seeds, 3.9x over six seeds at the pilot's own 100 Hz step)
    # the reason, in the statistic the game flights were measured by: are the false boxes on the same thing?
    assert 0.62 <= _grouped_fraction(solid) <= 0.92              # the six game flights: 62-92 %
    assert _grouped_fraction(loose) < 0.15                       # a random box per frame is on nothing twice
    assert max((t['hits'] for r in loose for t in r['phantoms']), default=0) <= 6
    assert max(t['hits'] for r in solid for t in r['phantoms']) >= 12
    # and at the shipped rate the loose kind builds none at all, which is the state the bench was in
    assert sum(len(r['phantoms']) for r in _flights('shipped', range(3))) == 0


def test_the_calibrated_clutter_lies_at_the_rate_the_game_logs_measured():
    """The spreads are over w22, w23, w27, w28, w29 and w30 (see ClutterModel's docstring)."""
    runs = _flights('clutter', range(4))
    assert 0.042 <= np.mean([r['false_per_frame'] for r in runs]) <= 0.139
    assert 0.067 <= np.mean([r['false_box_frac'] for r in runs]) <= 0.269
    assert 3.3 <= np.mean([100.0 * len(r['phantoms']) / r['seconds'] for r in runs]) <= 7.1
    hits = sorted((t['hits'] for r in runs for t in r['phantoms']), reverse=True)
    assert 4 <= np.median(hits) <= 12 and hits[0] >= 12
    over = sum(r['stats']['clutter_over_arch'] for r in runs) / max(sum(r['stats']['detected'] for r in runs), 1)
    assert 0.018 <= over <= 0.159                # the box an arch would have had, taken by something else


def test_the_rehearsal_reports_its_phantoms(tmp_path):
    from haltere.brain.baselines import MLPPolicy
    from haltere.liftoff.sightpilot import SightParams
    from haltere.sim.tasks import HoverTask
    from haltere.train.bptt import ExperimentConfig

    class TinyBrain(MLPPolicy):
        def weight_matrix(self):
            return torch.zeros(1)

    torch.manual_seed(0)
    cfg = ExperimentConfig.from_dict({'brain': {'model': 'mlp'}, 'train': {'delay_steps': 6}})
    brain = TinyBrain(HoverTask.channels, 4, 16, cfg.brain.dt, cfg.brain.action_tau)
    hover = 2 * (1.0 / cfg.quad.twr) ** (1.0 / cfg.quad.thrust_exp) - 1
    with torch.no_grad():
        brain.readout.bias[0] = float(np.arctanh(hover + 0.1))
    det = DetectorModel(false_pos=0.0, clutter=ClutterModel(n_per_100m=30.0, fire=0.6))
    opts = RehearsalOptions(seconds=8.0, verbose=False, sight='rabbit', sight_params=SightParams(yaw_rate=3.8),
                            track_report=True)
    res = run_rehearsal(brain, cfg, COURSE, CAM, tmp_path / 'clutter.csv', opts, det)
    assert res['detector']['clutter'] > 0 and len(res['clutter']) > 0
    rep = phantom_report(res, COURSE)
    assert rep['tracks'] >= rep['confirmed'] >= rep['phantoms'] >= 1
    assert rep['sightings_at_no_arch'] > 0 and rep['phantom_hits_median'] >= 3
    s = summarize(res, COURSE)
    assert s['phantom_tracks']['phantoms'] == rep['phantoms']
    assert s['attempts'][0]['detections']['clutter_s'] > 0
    # without the recording there is nothing to report, and the rehearsal is what it was
    res2 = run_rehearsal(brain, cfg, COURSE, CAM, None, RehearsalOptions(seconds=2.0, verbose=False, sight='rabbit'),
                         DetectorModel())
    assert phantom_report(res2, COURSE) is None and summarize(res2, COURSE)['phantom_tracks'] is None
