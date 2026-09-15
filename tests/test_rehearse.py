"""By-sight rehearsal: the synthetic detector's projection and timing, and the closed loop on a tiny brain."""
import numpy as np
import torch

from haltere.vision.camera import Camera, quat_wxyz_to_mat
from haltere.vision.gates import gate_label
from haltere.vision.model import IN_H, IN_W
from haltere.vision.rehearse import (DetectorModel, RehearsalOptions, SyntheticGateVision, arch_collision, arch_views,
                                     quat_from_yaw, run_rehearsal, summarize)

CAM = Camera(640, 360, 200.0, 30.0)
GATES = [{'pos': [20.0, 0.0, 1.2], 'heading': 0.0}, {'pos': [45.0, 6.0, 1.2], 'heading': 0.3}]


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
    # past the first gate, looking back at it: the labels' next-gate rule hides it, a detector does not
    pos, q = np.array([30.0, 0.0, 2.7]), quat_from_yaw(np.pi)
    assert gate_label(pos, q, GATES[:1], CAM)['visible'] == 0
    v = arch_views(pos, q, GATES[:1], CAM)[0]
    assert v['visible'] == 1 and abs(v['view_deg']) < 1e-6 and abs(v['u'] - 320) < 1e-6
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
    assert abs(det.t - (1000.0 + 0.08 + (det.frames - 1) / 15.0)) < 0.011          # delivered 80 ms after capture
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
    assert ages.min() >= 0.0 and ages.max() < 0.15                         # sim clock: detections are 0-70 ms old
    s = summarize(res, GATES)
    assert s['detector']['detected'] > 40 and 'goal_jumps' in s['attempts'][0]
