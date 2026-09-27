"""brain-10 label teacher: gain overrides, capped-turn relief and per-row smoothing (training only, off by default)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from haltere.brain.motor_baseline import FastMotorPD, FastPDConfig
from haltere.train.fast_motor_tracking import LabelTeacher, capped_turn_relief, fit_readout, parse_gains

PROFILE = next((p for p in (Path('runs/measured-dynamics-low-speed-20260923/profile.json'),
                            Path('C:/DEV/Haltere/runs/measured-dynamics-low-speed-20260923/profile.json')) if p.exists()),
               Path('runs/measured-dynamics-low-speed-20260923/profile.json'))
CALIBRATION = dict(hover_processed=-.2, hover_stick_sim=-.3, throttle_scale=1.1)


def rotated(speed, degrees):
    a = np.radians(degrees)
    return [speed*np.cos(a), speed*np.sin(a), 0.]


def test_relief_is_off_at_zero_and_leaves_aligned_fast_and_hairpin_requests_alone():
    velocity = torch.tensor([[5.5, 0., 0.]]*4)
    request = torch.tensor([[4., 0., .3],                  # aligned cap: the magnitude brake is kept
                            rotated(6.5, 45.),             # turn at or above the flown speed: not capped
                            rotated(4., 100.),             # hairpin beyond zero_deg: full Cartesian brake
                            rotated(4., 45.)], dtype=torch.float32)
    request[1:, 2] = .2
    assert torch.equal(capped_turn_relief(request, velocity, 0.), request)
    shaped = capped_turn_relief(request, velocity, 1.)
    assert torch.allclose(shaped[:3], request[:3])
    assert torch.equal(shaped[:, 2], request[:, 2])


def test_relief_removes_only_the_turn_geometry_brake_in_a_capped_turn():
    velocity = torch.tensor([[5.5, 0., 0.], [5.5, 0., 0.]])
    request = torch.tensor([rotated(4., 30.), rotated(4., 67.5)], dtype=torch.float32)
    shaped = capped_turn_relief(request, velocity, 1.)
    # 30 deg (<= 45): along-track request raised to the magnitude, lateral part unchanged
    assert abs(float(shaped[0, 0])-4.) < 1e-5 and abs(float(shaped[0, 1])-float(request[0, 1])) < 1e-6
    # 67.5 deg: half weight
    along = 4.*np.cos(np.radians(67.5))
    assert abs(float(shaped[1, 0])-(along+.5*(4.-along))) < 1e-5
    half = capped_turn_relief(request, velocity, .5)
    assert abs(float(half[0, 0])-(4.*np.cos(np.radians(30.))+.5*(4.-4.*np.cos(np.radians(30.))))) < 1e-5


def test_relief_ramps_in_with_the_request_side_over_speed():
    velocity = torch.tensor([[5.5, 0., 0.]]*3)
    request = torch.tensor([rotated(5.3, 30.), rotated(4.85, 30.), rotated(4.5, 30.)], dtype=torch.float32)
    shaped = capped_turn_relief(request, velocity, 1.)
    gap = request[:, :2].norm(dim=-1)-request[:, 0]
    raised = shaped[:, 0]-request[:, 0]
    assert float(raised[0]) == 0.                                   # 0.2 m/s under the speed: not a cap
    assert abs(float(raised[1])-.5*float(gap[1])) < 1e-5              # 0.65 m/s: halfway up the ramp
    assert abs(float(raised[2])-float(gap[2])) < 1e-5                 # 1.0 m/s: full


def test_label_teacher_without_overrides_is_fast_motor_pd():
    if not PROFILE.exists():
        pytest.skip('measured profile not present')
    profile = json.loads(PROFILE.read_text())
    torch.manual_seed(0)
    senses = dict(quat=torch.nn.functional.normalize(torch.randn(3, 4), dim=-1), vel_world=torch.randn(3, 3)*3,
                  gyro=torch.randn(3, 3))
    request, ff = torch.randn(3, 3)*4, torch.randn(3, 3)
    a, b = FastMotorPD(profile, CALIBRATION), LabelTeacher(profile, CALIBRATION)
    for _ in range(3):
        assert torch.equal(a.command(senses, request, ff), b.command(senses, request, ff))
    slow = LabelTeacher(profile, CALIBRATION, {'attitude_gain': 3.})
    assert slow.pd.config == FastPDConfig(attitude_gain=3.)
    assert not torch.equal(slow.command(senses, request, ff), FastMotorPD(profile, CALIBRATION).command(senses, request, ff))


def test_default_rollout_is_unchanged_and_a_label_teacher_without_overrides_labels_identically():
    parent = Path('C:/DEV/Haltere/runs/motor-brain-10-tracking-05/candidate.pt')
    if not PROFILE.exists() or not parent.exists():
        pytest.skip('parent checkpoint or measured profile not present')
    from haltere.liftoff.fast_rehearsal import synthetic_course
    from haltere.train.bptt import load_checkpoint
    from haltere.train.fast_motor_tracking import SyntheticCapsConfig, fast_contract, rollout
    torch.set_num_threads(1)
    profile = json.loads(PROFILE.read_text())
    brain, cfg, _ = load_checkpoint(str(parent), 'cpu')
    meta = torch.load(parent, map_location='cpu', weights_only=True)['visual_brain']
    contract = fast_contract(6., .4, scaled_speed=2.4)
    courses = [synthetic_course(1000, steep=.4), synthetic_course(1001, steep=.4)]
    runs = []
    for factory in (None, LabelTeacher):
        row, data = rollout(brain, cfg, meta, profile, contract, courses, controller='pd', seconds=2.5, seed=100,
                            collect=True, caps=SyntheticCapsConfig(), cap_fraction=.5, teacher_factory=factory)
        runs.append((row, data))
    (a, da), (b, db) = runs
    assert a == b
    for key in ('features', 'labels', 'request', 'velocity', 'step_gram'):
        assert torch.equal(da[key], db[key])


def test_label_lead_pairs_each_sample_with_the_teachers_label_later():
    parent = Path('C:/DEV/Haltere/runs/motor-brain-10-tracking-05/candidate.pt')
    if not PROFILE.exists() or not parent.exists():
        pytest.skip('parent checkpoint or measured profile not present')
    from haltere.liftoff.fast_rehearsal import synthetic_course
    from haltere.train.bptt import load_checkpoint
    from haltere.train.fast_motor_tracking import fast_contract, rollout
    torch.set_num_threads(1)
    profile = json.loads(PROFILE.read_text())
    brain, cfg, _ = load_checkpoint(str(parent), 'cpu')
    meta = torch.load(parent, map_location='cpu', weights_only=True)['visual_brain']
    contract = fast_contract(6., .4, scaled_speed=2.4)
    courses = [synthetic_course(1000, steep=.4), synthetic_course(1001, steep=.4)]
    outputs = {}

    def recording(lead):
        outputs[lead] = []

        class Recording(FastMotorPD):
            def command(self, *args, **kwargs):
                out = super().command(*args, **kwargs)
                outputs[lead].append(out.clone())
                return out
        return Recording

    data = {}
    for lead in (0, 3):
        _, data[lead] = rollout(brain, cfg, meta, profile, contract, courses, controller='pd', seconds=2.5, seed=100,
                                collect=True, teacher_factory=recording(lead), label_lead=lead)
    assert torch.equal(torch.stack(outputs[0]), torch.stack(outputs[3]))   # same flight either way
    ticks = [k for k in range(250) if k > 50 and k % 5 == 0 and k*.01 >= 1. and k+3 < 250]
    assert len(data[3]['labels']) == 2*len(ticks) and torch.equal(data[3]['features'], data[0]['features'][:2*len(ticks)])
    expected = torch.cat([outputs[3][k+3][:, :3].clamp(-.97, .97).atanh() for k in ticks])
    assert torch.equal(data[3]['labels'], expected)
    assert torch.equal(data[0]['labels'], torch.cat([outputs[0][k][:, :3].clamp(-.97, .97).atanh()
                                                     for k in range(250) if k > 50 and k % 5 == 0 and k*.01 >= 1.]))
    assert torch.equal(data[3]['request'], data[0]['request'][:2*len(ticks)])


def test_sag_mask_selects_samples_losing_height_they_were_asked_to_keep():
    from haltere.train.fast_motor_tracking import sag_mask, sag_weights
    request = torch.tensor([[5., 0., .2],     # asked +0.2, sinking at 1.0: sag 1.2 -> yes
                            [5., 0., .2],     # sinking at 0.5: sag 0.7 -> no
                            [5., 0., -1.],    # asked a fast descent, sinking at 2.5: -> no
                            [5., 0., -.4],    # asked a gentle descent, sinking at 1.5: sag 1.1 -> yes
                            [5., 0., 1.]])    # climbing request, flown level: sag 1.0 -> yes
    velocity = torch.tensor([[5., 0., -1.], [5., 0., -.5], [5., 0., -2.5], [5., 0., -1.5], [5., 0., 0.]])
    assert sag_mask(request, velocity).tolist() == [True, False, False, True, True]
    weights = sag_weights(request, velocity, 4.)
    assert torch.allclose(weights.mean(), torch.tensor(1.)) and torch.allclose(weights[0]/weights[1], torch.tensor(4.))


def test_parse_gains_validates_names_and_values():
    assert parse_gains(['attitude_gain=4', 'velocity_gain=2']) == dict(attitude_gain=4., velocity_gain=2.)
    assert parse_gains([]) == {}
    for bad in (['attitude=4'], ['attitude_gain'], ['max_tilt_deg=85']):
        with pytest.raises((ValueError, TypeError)):
            parse_gains(bad)


class Readout(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.readout = torch.nn.Linear(width, 4)
        self.other = torch.nn.Parameter(torch.zeros(2))


def test_smooth_rows_of_ones_is_the_single_solve_and_other_multipliers_act_per_row():
    torch.manual_seed(1)
    n, width = 400, 6
    features, labels = torch.randn(n, width), torch.randn(n, 3)
    steps = torch.randn(200, width)
    gram = torch.zeros(width+1, width+1, dtype=torch.float64)
    gram[:-1, :-1] = steps.double().T@steps.double()
    base = Readout(width)
    fits = {}
    for key, rows in (('none', None), ('ones', [1., 1., 1.]), ('pitch', [1., 1., 5.])):
        brain = Readout(width)
        brain.load_state_dict(base.state_dict())
        fit_readout(brain, features, labels, .1, None, 2., gram, 200, smooth_rows=rows)
        fits[key] = (brain.readout.weight.detach().clone(), brain.readout.bias.detach().clone())
    assert torch.equal(fits['none'][0], fits['ones'][0]) and torch.equal(fits['none'][1], fits['ones'][1])
    w, p = fits['none'][0], fits['pitch'][0]
    assert torch.allclose(w[:2], p[:2], atol=1e-5) and not torch.allclose(w[2], p[2], atol=1e-4)
    assert torch.equal(w[3], base.readout.weight[3].detach())
    # more smoothing on the pitch row: smaller mean squared command step on that row
    def step_energy(weight):
        return float(((steps@weight[2]).pow(2)).mean())
    assert step_energy(p) < step_energy(w)
    with pytest.raises(ValueError):
        fit_readout(Readout(width), features, labels, .1, None, 0., gram, 200, smooth_rows=[1., 1., 2.])
