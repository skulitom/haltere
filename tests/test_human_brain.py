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


def test_scene_sampling_keeps_lower_obstacles_but_masks_observed_sticks():
    base=torch.full((1,3,90,160),.5)
    hud=base.clone();hud[...,round(.84*90):,round(.40*160):round(.60*160)]=1.
    assert torch.equal(retina_input(base,'scene_v2'),retina_input(hud,'scene_v2'))
    obstacle=base.clone();obstacle[...,76:90,110:114]=1.
    assert torch.equal(retina_input(base),retina_input(obstacle))
    assert not torch.equal(retina_input(base,'scene_v2'),retina_input(obstacle,'scene_v2'))


def test_scene_adapter_training_cannot_change_blank_retina_parent_behaviour():
    from haltere.train.scene_brain import visual_parameters_only
    brain,_,_=small_brain();parent=copy.deepcopy(brain).requires_grad_(False)
    parameters=visual_parameters_only(brain);opt=torch.optim.Adam(parameters,lr=.03)
    obs={k:torch.randn(3,d) for k,d in brain.channel_dims.items()}
    W=brain.weight_matrix();state=brain.init_state(3)
    for _ in range(12):action,state,_=brain(obs,state,W)
    action.square().sum().backward();opt.step()
    assert not torch.equal(brain.encoders['retina__lptc'].U,parent.encoders['retina__lptc'].U)
    for name,p in brain.named_parameters():
        if name not in ('encoders.retina__lptc.U','encoders.retina__lptc.log_gain'):
            assert torch.equal(p,parent.state_dict()[name])
    obs['retina'].zero_();s=brain.init_state(3);p=parent.init_state(3)
    with torch.no_grad():
        for _ in range(30):
            a,s,_=brain(obs,s);b,p,_=parent(obs,p)
            assert torch.equal(a,b) and torch.equal(s['v'],p['v'])


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
    recovery = RecoveryRollout(teacher,cfg,calibration,batch_size=2,takeoff=True,physics_weight=1.,
                               horizontal_weight=2.,angular_weight=.5,hold_heading=True)
    optimizer = torch.optim.Adam(student.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss = recovery.loss(student,torch.ones(4),steps=24)
        loss.backward()
        assert torch.isfinite(student.readout.weight.grad).all()
        assert recovery.last_metrics['physics_cost']>0
        assert not recovery.vs.quad.vel.requires_grad
        optimizer.step()


def test_new_curriculum_rejects_previous_training_in_holdout(tmp_path):
    import json
    import pytest
    from haltere.train.human_brain import validate_data_continuation
    from haltere.vision.datasets import sha256
    old_data, new_data, old_replay, new_replay = [tmp_path/n for n in ('old_data','new_data','old_replay','new_replay')]
    for folder in (old_data,new_data,old_replay,new_replay):
        folder.mkdir()
    old_take = dict(split='train',source_hashes={'telemetry.csv':'old_log'},frame_hashes={'frame':'old_image'})
    held = dict(split='validation',source_hashes={'telemetry.csv':'held_log'},frame_hashes={'frame':'held_image'})
    (old_data/'manifest.json').write_text(json.dumps({'takes':[old_take]}))
    (new_data/'manifest.json').write_text(json.dumps({'takes':[held]}))
    contract = dict(teacher_sha256='teacher',initial_brain_sha256='brain',graph_sha256='graph',calibration={},dt=.01)
    old = {**contract,'dataset_sha256':sha256(old_data/'manifest.json')}
    (old_replay/'manifest.json').write_text(json.dumps(old))
    warm = dict(prepared_sha256=sha256(old_replay/'manifest.json'))
    manifest = {**contract,'dataset_path':'../new_data','dataset_sha256':sha256(new_data/'manifest.json')}
    config = dict(warm_start_prepared=str(old_replay),warm_start_dataset=str(old_data))
    lineage = validate_data_continuation(warm,manifest,new_replay,config)
    assert lineage == {'telemetry':['old_log'],'frames':['old_image']}
    # Renaming a take cannot conceal use of its original telemetry or image.
    held['source_hashes']['telemetry.csv'] = 'old_log'
    (new_data/'manifest.json').write_text(json.dumps({'takes':[held]}))
    manifest['dataset_sha256'] = sha256(new_data/'manifest.json')
    with pytest.raises(ValueError,match='overlaps'):
        validate_data_continuation(warm,manifest,new_replay,config)
    held['source_hashes']['telemetry.csv'] = 'held_log'
    (new_data/'manifest.json').write_text(json.dumps({'takes':[held]}))
    manifest['dataset_sha256'] = sha256(new_data/'manifest.json')
    warm['training_lineage'] = {'frames':['held_image']}
    with pytest.raises(ValueError,match='overlaps'):
        validate_data_continuation(warm,manifest,new_replay,config)


def test_navigation_teacher_paths_supervise_motor_output_without_entering_student_senses():
    from haltere.train.human_brain import navigation_motor_targets
    torch.manual_seed(28)
    student,cfg,_ = small_brain()
    teacher = copy.deepcopy(student).requires_grad_(False)
    batch = {k:torch.zeros(2,24,d) for k,d in student.channel_dims.items()}
    batch['action'] = torch.zeros(2,24,4)
    batch['teacher'] = torch.zeros(2,24,3,3)
    batch['teacher'][0,...,0] = 2.
    batch['teacher'][1,...,1] = 2.
    targets = navigation_motor_targets(teacher,cfg.task,batch)
    assert not targets.requires_grad
    assert (targets[0,:,3]-targets[1,:,3]).abs().min()>.1
    assert torch.count_nonzero(batch['goal'])==0
    action,_,_ = rollout(student,batch)
    assert torch.allclose(action[0],action[1])  # same observations, different training labels
    (action-targets).square().mean().backward()
    assert student.log_edge_gain.grad.abs().sum()>0
    assert student.readout.weight.grad.abs().sum()>0
