import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from haltere.brain.model import BrainConfig, ConnectomeRNN
from haltere.brain.retina import RETINA_DIM, brain_to_processed, retina_input, visual_observation
from haltere.connectome.graph import BrainGraph
from haltere.sim.tasks import HoverTask, HoverTaskConfig
from haltere.train.bptt import ExperimentConfig, load_checkpoint
from haltere.train.human_brain import causal_indices, export, rollout


def small_brain():
    rng = np.random.default_rng(2)
    n = 90
    key = np.unique(rng.integers(0,n*n,1800))
    pre, post = key%n, key//n
    pops = {name:np.arange(i*10,i*10+10) for i,name in enumerate(HoverTask.channels)}
    pops['wing_mn'] = np.arange(70,90)
    nodes = pd.DataFrame(dict(bodyId=np.arange(n),type='unit',sign=np.ones(n,dtype=np.int8)))
    graph = BrainGraph(nodes,pre,post,np.ones(len(pre)),pops)
    cfg = ExperimentConfig.from_dict({'brain':{'motor':['wing_mn'],'readout_norm':'none','readout_init':.03,
                                              'sensory':{**{k:k for k in HoverTask.channels},'retina':'lptc'}}})
    brain = ConnectomeRNN(graph,{**HoverTask.channels,'retina':RETINA_DIM},cfg.brain).eval()
    return brain,cfg,graph


def test_images_mask_controls_and_causal_selection():
    a = torch.rand(2,3,90,160)
    b = a.clone()
    b[...,:round(.23*90),:] = 1
    b[...,round(.84*90):,:] = 0
    b[...,round(.34*90):round(.78*90),round(.82*160):] = 1
    assert torch.equal(retina_input(a),retina_input(b))
    assert causal_indices([1.,2.,3.],[.9,1.,1.9,2.,3.1]).tolist()==[-1,0,0,1,2]


def test_processed_target_units_and_no_goal_input():
    calibration = dict(hover_processed=.14,hover_stick_sim=-.5,throttle_scale=.8,stick_sign=[-1,1,1])
    a = torch.tensor([[-.5,.2,-.3,.4]])
    assert torch.allclose(brain_to_processed(a,calibration),torch.tensor([[.14,-.2,-.3,.4]]))
    s = dict(gyro=torch.zeros(1,3),gravity_body=torch.tensor([[0.,0.,-1.]]),vel_body=torch.zeros(1,3),
             vel_world=torch.zeros(1,3),pos=torch.tensor([[999.,999.,999.]]),quat=torch.tensor([[1.,0.,0.,0.]]),
             up=torch.ones(1),altitude=torch.ones(1,1),yaw=torch.ones(1,1))
    obs = visual_observation(s,torch.zeros(1,1),HoverTaskConfig(),torch.zeros(1,RETINA_DIM))
    assert torch.count_nonzero(obs['goal'])==0
    assert torch.count_nonzero(obs['compass'])==0
    higher = visual_observation({**s, 'altitude':torch.full((1,1),3.)},torch.zeros(1,1),
                                 HoverTaskConfig(),torch.zeros(1,RETINA_DIM))
    assert torch.equal(obs['lptc'],higher['lptc'])  # motion cannot reveal height at rest
    assert not torch.equal(obs['altitude'],higher['altitude'])


def test_processed_commands_match_radial_gamepad_path():
    from haltere.liftoff.pilot import LiftoffMapping
    from haltere.liftoff.stickcal import RadialSticks
    c = dict(hover_processed=.14,hover_stick_sim=-.5,throttle_scale=.8,stick_sign=[-1,1,1])
    radial = RadialSticks(.25,{'roll':-1})
    mapping = LiftoffMapping(stick_sign=tuple(c['stick_sign']),hover_stick_sim=c['hover_stick_sim'],
                             hover_processed_game=c['hover_processed'],throttle_scale=c['throttle_scale'],
                             stick_model=radial)
    action = torch.tensor([[-.5,.2,-.3,.4]])
    raw = mapping.to_raw(action[0].numpy())
    thr,yaw = radial.processed('throttle','yaw',raw[0],raw[3])
    roll,pitch = radial.processed('roll','pitch',raw[1],raw[2])
    assert np.allclose([thr,roll,pitch,yaw],brain_to_processed(action,c)[0].numpy())


def test_teacher_loss_updates_recurrent_brain_without_teacher_inputs():
    torch.manual_seed(7)
    brain,cfg,graph = small_brain()
    batch = {k:torch.randn(2,12,d) for k,d in brain.channel_dims.items()}
    batch['action'] = torch.zeros(2,12,4)
    batch['teacher'] = torch.randn(2,12,3,3)
    probe = nn.Linear(len(graph.population('goal')),9)
    probe.register_buffer('neuron_idx',torch.tensor(graph.population('goal')))
    output,path,_ = rollout(brain,batch,probe=probe)
    # Auxiliary supervision by itself must reach the brain, including visual inputs.
    (path-batch['teacher'].flatten(-2)).square().mean().backward()
    assert brain.log_edge_gain.grad.abs().sum()>0
    assert brain.log_gain.grad.abs().sum()>0
    assert brain.encoders['retina__lptc'].U.grad.abs().sum()>0
    altered = copy.deepcopy(batch)
    altered['action'].fill_(100)
    altered['teacher'].fill_(100)
    with torch.no_grad():
        again,_,_ = rollout(brain,altered)
    assert torch.equal(output.detach(),again)


def test_export_runs_without_teacher_or_training_head(tmp_path,monkeypatch):
    brain,cfg,graph = small_brain()
    path = tmp_path/'brain.pt'
    export(path,brain,cfg,{'runtime_requires_teacher':False,'teacher_training_only':True},1)
    ck = torch.load(path,weights_only=True)
    assert ck['visual_brain']['runtime_requires_teacher'] is False
    assert not any('teacher' in k or 'probe' in k for k in ck['model'])
    monkeypatch.setattr(BrainGraph,'load',lambda path:graph)
    loaded,_,_ = load_checkpoint(path,'cpu')
    batch = {k:torch.zeros(1,4,d) for k,d in brain.channel_dims.items()}
    batch['action'] = torch.zeros(1,4,4)
    with torch.no_grad():
        expected,_,_ = rollout(brain,batch)
        actual,_,_ = rollout(loaded,batch)
    assert torch.equal(expected,actual)


def test_live_attempt_limits_detect_climb_and_departure():
    from haltere.liftoff.visual_brain import flight_limit_reason
    assert flight_limit_reason(np.array([0,0,2]),np.zeros(3),8,10,20) is None
    assert 'height' in flight_limit_reason(np.array([0,0,9]),np.zeros(3),8,10,20)
    assert 'speed' in flight_limit_reason(np.zeros(3),np.array([0,0,11]),8,10,20)
    assert 'distance' in flight_limit_reason(np.array([21,0,2]),np.zeros(3),8,10,20)


def test_motor_feedback_mask_is_exported_and_does_not_mutate_senses(tmp_path, monkeypatch):
    brain, cfg, graph = small_brain()
    cfg.brain.mask_motor_feedback = True
    batch = {k: torch.rand(2, 12, d) for k, d in brain.channel_dims.items()}
    batch['action'] = torch.zeros(2, 12, 4)
    before = batch['wing_cs'].clone()
    changed = copy.deepcopy(batch)
    changed['wing_cs'][..., 2] = -100
    with torch.no_grad():
        a, _, _ = rollout(brain, batch)
        b, _, _ = rollout(brain, changed)
    assert torch.equal(a, b)
    assert torch.equal(batch['wing_cs'], before)
    path = tmp_path/'masked.pt'
    export(path, brain, cfg, {'runtime_requires_teacher': False}, 1)
    monkeypatch.setattr(BrainGraph, 'load', lambda path: graph)
    loaded, _, _ = load_checkpoint(path, 'cpu')
    assert loaded.cfg.mask_motor_feedback
    with torch.no_grad():
        c, _, _ = rollout(loaded, changed)
    assert torch.equal(a, c)


def test_recovery_rollout_updates_student_across_windows_without_teacher_gradients():
    from haltere.train.recovery import RecoveryRollout
    student, cfg, _ = small_brain()
    teacher = copy.deepcopy(student).requires_grad_(False)
    student.cfg.mask_motor_feedback = True
    calibration = dict(hover_processed=.14, hover_stick_sim=-.5, throttle_scale=.8, stick_sign=[-1, 1, 1])
    recovery = RecoveryRollout(teacher, cfg, calibration, batch_size=2)
    optimizer = torch.optim.Adam(student.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss = recovery.loss(student, torch.ones(4), steps=24)
        assert torch.isfinite(loss)
        loss.backward()
        assert student.log_edge_gain.grad.abs().sum()>0
        optimizer.step()
    assert recovery.age == 48
    assert not recovery.student_state['v'].requires_grad
    assert all(p.grad is None for p in teacher.parameters())


def test_takeoff_teacher_goal_does_not_enter_student_observation():
    from haltere.train.recovery import RecoveryRollout
    student, cfg, _ = small_brain()
    teacher = copy.deepcopy(student).requires_grad_(False)
    goals = {'student': [], 'teacher': []}
    handles = [b.register_forward_pre_hook(lambda module,args,name=name:goals[name].append(args[0]['goal'].clone()))
               for name,b in [('student',student),('teacher',teacher)]]
    calibration = dict(hover_processed=.14,hover_stick_sim=-.5,throttle_scale=.8,stick_sign=[-1,1,1])
    recovery = RecoveryRollout(teacher,cfg,calibration,batch_size=2,takeoff=True)
    recovery.loss(student,torch.ones(4),steps=24).backward()
    for h in handles:
        h.remove()
    assert all(torch.count_nonzero(v)==0 for v in goals['student'])
    assert any(torch.count_nonzero(v)>0 for v in goals['teacher'])


def test_selection_never_silently_falls_back_to_initial_model(tmp_path):
    from haltere.train.human_brain import selected_checkpoint
    for name in ('initial.pt','last.pt','best.pt'):
        (tmp_path/name).touch()
    selected, reason = selected_checkpoint(tmp_path,True)
    assert selected.name == 'last.pt' and 'no candidate passed' in reason
    (tmp_path/'motor-qualified.pt').touch()
    assert selected_checkpoint(tmp_path,True)[0].name == 'motor-qualified.pt'
    assert selected_checkpoint(tmp_path,False)[0].name == 'best.pt'
    (tmp_path/'best.pt').unlink()
    assert selected_checkpoint(tmp_path,False)[0].name == 'last.pt'


def test_differentiable_recovery_cost_survives_ground_contact_and_window_boundary():
    from haltere.train.recovery import RecoveryRollout
    student,cfg,_ = small_brain()
    teacher = copy.deepcopy(student).requires_grad_(False)
    calibration = dict(hover_processed=.14,hover_stick_sim=-.5,throttle_scale=.8,stick_sign=[-1,1,1])
    recovery = RecoveryRollout(teacher,cfg,calibration,batch_size=2,takeoff=True,physics_weight=1.)
    optimizer = torch.optim.Adam(student.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss = recovery.loss(student,torch.ones(4),steps=24)
        loss.backward()
        assert torch.isfinite(student.readout.weight.grad).all()
        assert recovery.last_metrics['physics_cost']>0
        assert not recovery.vs.quad.vel.requires_grad
        optimizer.step()
