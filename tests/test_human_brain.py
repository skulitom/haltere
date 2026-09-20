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
