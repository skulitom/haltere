"""Motor tracking through an experimental identified differentiable plant.

Local targets are synthetic training tasks, not visual navigation. No motor
teacher supplies action labels. Flight costs train the connectome through its
own consequences. Wiring, neurotransmitter signs and sensory encoders remain
fixed; recurrent magnitudes, neuron dynamics and the motor readout may learn.
"""
from collections import deque

import torch

from .motor_tracking import local_guidance,retina_sequence
from ..brain.gate_senses import gate_observation
from ..brain.retina import RETINA_DIM
from ..sim.identified import IdentifiedSim
from ..sim.quad import quat_from_euler,quat_to_mat


def trainable_motor_parameters(brain):
    allowed={'log_edge_gain','log_gain','log_tau','bias','readout.weight','readout.bias'}
    for name,parameter in brain.named_parameters():
        parameter.requires_grad_(name in allowed)
    # Preserve normalization statistics as part of the deployed sensory/motor
    # contract. eval() does not disable autograd.
    brain.eval()
    return {name:p for name,p in brain.named_parameters() if p.requires_grad}


class FlightCostRollout:
    def __init__(self,brain,cfg,meta,profile,*,batch=4,reference_speed=3.,seed=20001,
                 retina_stream=None,randomize=.2,episode_steps=800,controller='brain',reset_on_crash=True,
                 curriculum='turns'):
        if not cfg.brain.mask_motor_feedback:
            raise ValueError('Effective plant training requires masked RPM feedback; its rotor model is not qualified')
        if batch<1 or not 0<reference_speed<=10 or episode_steps<64:
            raise ValueError('Use positive batch/speed and an episode of at least 64 steps')
        if controller not in ('brain','pd'):
            raise ValueError('Expected brain or matched PD diagnostic')
        if curriculum not in ('turns','launch-turn-stop'):
            raise ValueError('Unknown motor curriculum')
        self.curriculum=curriculum
        self.controller=controller
        self.reset_on_crash=reset_on_crash
        self.brain,self.cfg,self.meta=brain,cfg,meta
        self.batch,self.reference_speed,self.retina_stream=batch,reference_speed,retina_stream
        self.sim=IdentifiedSim(profile,meta['calibration'],brain.device,cfg.brain.dt)
        self.randomize,self.episode_steps=randomize,episode_steps
        self.generator=torch.Generator(device=brain.device).manual_seed(seed)
        self.seed,self.episode,self.age=seed,0,episode_steps
        self.state=None

    def reset(self):
        B,dev=self.batch,self.brain.device
        rand=lambda *shape:torch.rand(*shape,device=dev,generator=self.generator)
        self.sim.randomize(B,self.randomize,self.generator)
        self.state=self.sim.hover(B,6.)
        self.heading0=rand(B)*2*torch.pi-torch.pi
        self.state.quad.quat=quat_from_euler((rand(B)-.5)*.2,(rand(B)-.5)*.2,self.heading0)
        self.speed=self.reference_speed*float(.5+.5*rand(()))
        if self.controller=='pd':
            from .motor_tracking import measured_dynamics
            from ..brain.motor_baseline import MotorPD,MotorPDConfig
            measured=measured_dynamics(self.cfg,self.meta)
            self.pd=MotorPD(measured.quad,measured.rates,measured.ctl.idle,
                            MotorPDConfig(position_gain=max(.8,self.speed/3)))
        self.state.quad.vel[:,:2]=torch.stack((self.heading0.cos(),self.heading0.sin()),-1)*self.speed*rand(B,1)
        self.state.quad.vel[:,2]=(rand(B)-.5)*.4
        self.altitude0=self.state.quad.pos[:,2].clone()
        if self.curriculum=='launch-turn-stop':
            # An airborne climb avoids inventing a ground/contact model. Its
            # target is 2--4 m above the start, unlike the old near-hover starts.
            self.state.quad.pos[:,2]=2+2*rand(B)
            self.state.quad.vel[:,:2]=0.
        self.turn=(rand(B)*2-1)*1.2
        self.climb=(rand(B)*2-1)*1.2
        self.hold=self.state.quad.pos.clone();self.phase=-1
        self.neural=self.brain.init_state(B)
        self.previous=torch.zeros(B,4,device=dev)
        c=self.meta['calibration'];processed=2*self.state.drive-1
        self.previous[:,0]=c['hover_stick_sim']+(processed-c['hover_processed'])/c['throttle_scale']
        # Independent per-episode command delay, 20--60 ms at a 100 Hz loop.
        self.delay_steps=int(torch.randint(2,7,(),device=dev,generator=self.generator))
        self.delay=deque(self.previous.clone() for _ in range(self.delay_steps))
        self.retina=(retina_sequence(self.retina_stream,self.episode_steps,B,self.seed+self.episode,.25).to(dev)
                     if self.retina_stream is not None else None)
        self.age=0;self.episode+=1

    def loss(self,steps=32):
        if not 8<=steps<=self.episode_steps:
            raise ValueError('Use a bounded flight-cost window of at least eight steps')
        if (self.state is None or self.age+steps>self.episode_steps
                or (self.reset_on_crash and bool(self.state.quad.crashed.any()))):
            self.reset()
        brain=self.brain;neural=brain.detach_state(self.neural);W=brain.weight_matrix()
        state=self.state.detach();costs=[];errors=[];height_errors=[]
        for t in range(steps):
            age=self.age+t
            if self.curriculum=='launch-turn-stop':
                phase=0 if age<self.episode_steps/3 else min(4,1+int(6*(age-self.episode_steps/3)/self.episode_steps))
                moving_phase=phase-1
            else:
                phase=min(3,int(4*age/self.episode_steps));moving_phase=phase
            if phase!=self.phase:
                self.hold=state.quad.pos.detach().clone();self.phase=phase
            heading=self.heading0+(self.turn if moving_phase==1 else -self.turn if moving_phase==2 else 0)
            altitude=self.altitude0+(self.climb if moving_phase==1 else -self.climb if moving_phase==2 else 0)
            braking=torch.full((self.batch,),moving_phase in (-1,3),dtype=torch.bool,device=brain.device)
            senses=self.sim.sensors(state)
            relative,yaw,moving=local_guidance(senses,heading,altitude,self.hold,braking)
            gain=max(1.,self.reference_speed/self.speed)
            velocity=senses['vel_world']*relative.new_tensor([gain,gain,1.])
            modified={**senses,'vel_world':velocity,
                'vel_body':torch.einsum('bji,bj->bi',quat_to_mat(senses['quat']),velocity)}
            retina=self.retina[age] if self.retina is not None else relative.new_zeros(self.batch,RETINA_DIM)
            sensor=self.meta['gate_sensor']
            obs=gate_observation(modified,state.quad.motor.mean(-1,keepdim=True),self.cfg.task,retina,relative,
                height_invariant=sensor['height_invariant'],gravity_aligned_height=sensor['gravity_aligned_height'],
                search_height_error=relative.new_zeros(self.batch,1),raw_retina_active=sensor['raw_retina_active'])
            if self.controller=='pd':
                action=self.pd.command(senses,relative,self.speed)
            else:
                if age==0:
                    with torch.no_grad():
                        for _ in range(50):_,neural,_=brain(obs,neural,W.detach())
                if t and t%8==0:neural=brain.detach_state(neural)
                action,neural,_=brain(obs,neural,W)
            command=torch.cat((action[:,:3],yaw[:,None]),-1)
            self.delay.append(command);state=self.sim.step(state,self.delay.popleft())
            desired_speed=torch.where(moving,self.speed*(relative[:,:2].norm(dim=-1)/3).clamp(max=1),0.)
            # The requested motion is a reference, not a differentiable escape
            # from the objective by changing the attitude-dependent guidance.
            desired_velocity=(torch.stack((heading.cos(),heading.sin()),-1)*desired_speed[:,None]).detach()
            velocity_error=(state.quad.vel[:,:2]-desired_velocity).square().sum(-1)
            height_error=(state.quad.pos[:,2]-altitude).square()
            hold_error=(state.quad.pos[:,:2]-self.hold[:,:2]).square().sum(-1)*braking
            up=quat_to_mat(state.quad.quat)[:,2,2]
            cost=(velocity_error+2*height_error+.5*hold_error+state.quad.vel[:,2].square()
                  +.5*(1-up)+.02*state.quad.omega.square().sum(-1)
                  +.05*(command-self.previous).square().sum(-1)
                  +20*torch.relu(1-state.quad.pos[:,2]).square()+100*state.quad.crashed.float())
            costs.append(cost.mean());errors.append(velocity_error.detach().mean())
            height_errors.append(height_error.detach().mean())
            self.previous=command.detach()
        self.age+=steps;self.neural=brain.detach_state(neural);self.state=state.detach()
        self.delay=deque(a.detach() for a in self.delay)
        metrics=dict(flight_cost=float(torch.stack(costs).mean().detach()),
            velocity_rmse=float(torch.stack(errors).mean().sqrt()),crashes=int(state.quad.crashed.sum()),
            height_rmse=float(torch.stack(height_errors).mean().sqrt()),
            episode=self.episode,step=self.age,speed_mps=self.speed,delay_steps=self.delay_steps)
        return torch.stack(costs).mean(),metrics


@torch.no_grad()
def evaluate(brain,cfg,meta,profile,*,retina_stream,reference_speed,seed=8293,episodes=2,batch=4,
             controllers=('brain','pd'),episode_steps=800,randomize=.2,curriculum='turns'):
    """Matched complete tasks: a crash cannot advance the random task schedule.

    Every controller sees the same starts, dynamics and four maneuver phases.
    Crashes remain failures for the rest of that fixed episode. Development
    selection and final evaluation must use distinct seeds.
    """
    from .thermal import wait_if_hot
    if episodes<1 or episode_steps<64 or episode_steps%32:
        raise ValueError('Evaluate positive complete episodes in 32-step windows')
    reports={}
    for controller in controllers:
        rollout=FlightCostRollout(brain,cfg,meta,profile,batch=batch,reference_speed=reference_speed,
            seed=seed,retina_stream=retina_stream,controller=controller,
            episode_steps=episode_steps,reset_on_crash=False,randomize=randomize,curriculum=curriculum)
        rows=[];crashed_tasks=0
        for episode in range(episodes):
            rollout.reset()
            for window in range(episode_steps//32):
                if window%5==0 and brain.device.type=='cuda':wait_if_hot(68.)
                _,metrics=rollout.loss(32);rows.append(metrics)
            crashed_tasks+=int(rollout.state.quad.crashed.sum())
        reports[controller]=dict(mean_flight_cost=sum(r['flight_cost'] for r in rows)/len(rows),
            velocity_rmse=(sum(r['velocity_rmse']**2 for r in rows)/len(rows))**.5,
            height_rmse=(sum(r['height_rmse']**2 for r in rows)/len(rows))**.5,
            crash_windows=sum(r['crashes']>0 for r in rows),crashed_tasks=crashed_tasks,
            tasks=episodes*batch,episodes=episodes,seed=seed,windows=len(rows),rows=rows)
    return reports


def evaluation_rank(report):
    """Any additional crash is worse than a reduction in average tracking cost."""
    return report['crashed_tasks'],report['mean_flight_cost']


def acceptable_tracking(report,parent):
    """Do not conceal a material velocity/height regression in the total cost."""
    return all(report[key]<=1.05*parent[key] for key in ('velocity_rmse','height_rmse'))


def main():
    import argparse,copy,hashlib,json,subprocess,time
    from pathlib import Path
    from .bptt import load_checkpoint
    from .human_brain import export
    from .motor_tracking import load_recorded_retina
    from .thermal import wait_if_hot
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('--profile',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--updates',type=int,default=200)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--window',type=int,default=32)
    p.add_argument('--speed',type=float,default=3.);p.add_argument('--seed',type=int,default=20001)
    p.add_argument('--retina-data',required=True);p.add_argument('--validation-retina-data',required=True)
    p.add_argument('--evaluation-episodes',type=int,default=2)
    p.add_argument('--test-episodes',type=int,default=4)
    p.add_argument('--validation-every',type=int,default=100)
    p.add_argument('--edge-lr',type=float,default=1e-5)
    p.add_argument('--neuron-lr',type=float,default=1e-5)
    p.add_argument('--readout-lr',type=float,default=1e-6)
    p.add_argument('--randomize',type=float,default=.05)
    p.add_argument('--episode-steps',type=int,default=1920)
    p.add_argument('--curriculum',choices=['turns','launch-turn-stop'],default='launch-turn-stop')
    p.add_argument('--development-seed',type=int,default=8293)
    p.add_argument('--test-seed',type=int,default=17491)
    a=p.parse_args()
    if min(a.updates,a.evaluation_episodes,a.test_episodes,a.validation_every)<1:
        raise ValueError('Use positive training and evaluation budgets')
    if not all(0<lr<.01 for lr in (a.edge_lr,a.neuron_lr,a.readout_lr)):
        raise ValueError('Use finite positive learning rates below 0.01')
    if (not 0<=a.randomize<1 or a.episode_steps<64 or a.episode_steps%32
            or not 8<=a.window<=a.episode_steps or a.episode_steps%a.window):
        raise ValueError('Use bounded uncertainty and complete fixed training/evaluation windows')
    if len({a.seed,a.development_seed,a.test_seed})!=3:
        raise ValueError('Training, development and final test seeds must differ')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    sha=lambda name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
    profile=json.loads(Path(a.profile).read_text())
    if not all(profile['angular_independent_validation'][axis]['passed'] for axis in ('roll','pitch','yaw')):
        raise ValueError('The measured angular profile did not pass independent validation')
    brain,cfg,graph=load_checkpoint(a.checkpoint,a.device)
    if graph is None:
        raise ValueError('This trainer requires the connectome')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);meta=ck['visual_brain']
    train_retina=load_recorded_retina(a.retina_data,meta['gate_sensor'])
    validation_retina=load_recorded_retina(a.validation_retina_data,meta['gate_sensor'])
    if sha(a.retina_data)==sha(a.validation_retina_data):
        raise ValueError('Use different recorded scene takes for training and validation')
    config=vars(a)|dict(protocol=3,parent_sha256=sha(a.checkpoint),profile_sha256=sha(a.profile),
        retina_data_sha256=sha(a.retina_data),validation_retina_data_sha256=sha(a.validation_retina_data),
        retina_manifest_sha256=sha(Path(a.retina_data).parent/'manifest.json'),
        validation_retina_manifest_sha256=sha(Path(a.validation_retina_data).parent/'manifest.json'),
        source_sha256=sha(__file__),loss='Differentiable flight cost; no action imitation',
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={name:sha(name) for name in ['haltere/train/flight_cost.py','haltere/sim/identified.py',
            'haltere/train/motor_tracking.py','haltere/brain/model.py','haltere/brain/sparse.py',
            'haltere/brain/gate_senses.py','haltere/brain/retina.py','haltere/sim/quad.py']},
        observation_scope='Synthetic local targets with independent recorded retinal currents, not visual navigation',
        frozen='Connectome wiring, known/unknown signs, sensory encoders and normalization statistics')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    initial={k:v.detach().cpu().clone() for k,v in brain.state_dict().items()}
    selected=trainable_motor_parameters(brain)
    priors={k:p.detach().clone() for k,p in selected.items()}
    opt=torch.optim.Adam([dict(params=[selected['log_edge_gain']],lr=a.edge_lr),
        dict(params=[v for k,v in selected.items() if k.startswith('readout.')],lr=a.readout_lr),
        dict(params=[v for k,v in selected.items() if k not in ('log_edge_gain','readout.weight','readout.bias')],
             lr=a.neuron_lr)])
    wait_if_hot(68.)
    evaluation_options=dict(episode_steps=a.episode_steps,randomize=a.randomize,curriculum=a.curriculum)
    baseline=evaluate(brain,cfg,meta,profile,retina_stream=validation_retina,reference_speed=a.speed,
                      episodes=a.evaluation_episodes,batch=a.batch,seed=a.development_seed,**evaluation_options)
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2))
    print('Baseline',json.dumps({k:{n:v for n,v in r.items() if n!='rows'} for k,r in baseline.items()}),flush=True)
    best_state=initial;best_report=baseline['brain'];best_iteration=0

    def save_candidate(path,iteration,report):
        audit={}
        for name,value in brain.state_dict().items():
            old=initial[name];current=value.detach().cpu()
            if not torch.equal(old,current):
                if name not in selected:raise RuntimeError('Unexpected changed brain tensor: '+name)
                audit[name]=dict(changed=int((old!=current).sum()),total=old.numel(),
                                 max_abs_change=float((old-current).abs().max()))
        provenance=copy.deepcopy(meta)
        provenance.pop('schema',None)
        provenance['motor_tracking']={**provenance.get('motor_tracking',{}),'nominal_speed_mps':a.speed}
        provenance['flight_cost_training']=dict(**config,selected_iteration=iteration,changed_tensors=audit,
            simulation_evaluation={k:{n:v for n,v in r.items() if n!='rows'} for k,r in report.items()},
            live_qualified=False,limits='Experimental motor surrogate; no new autonomous race or freestyle evidence')
        provenance['qualified']=False
        export(path,brain,cfg,provenance,iteration)
        return dict(changes=audit,wiring_and_signs_unchanged=True,candidate_sha256=sha(path))

    rollout=FlightCostRollout(brain,cfg,meta,profile,batch=a.batch,reference_speed=a.speed,
                              seed=a.seed,retina_stream=train_retina,**evaluation_options)
    start=time.monotonic()
    with (out/'training.jsonl').open('x') as log:
        for iteration in range(1,a.updates+1):
            wait_if_hot(68.)
            opt.zero_grad();loss,metrics=rollout.loss(a.window)
            prior=sum((p-priors[k]).square().mean() for k,p in selected.items())
            total=loss+.01*prior
            if not torch.isfinite(total):raise RuntimeError('Nonfinite training loss')
            total.backward()
            norm=torch.nn.utils.clip_grad_norm_(selected.values(),1.,error_if_nonfinite=True)
            opt.step()
            row=dict(iteration=iteration,elapsed_s=time.monotonic()-start,gradient_norm=float(norm),**metrics)
            log.write(json.dumps(row)+'\n');log.flush()
            if iteration==1 or iteration%10==0:print(json.dumps(row),flush=True)
            if iteration%a.validation_every==0 or iteration==a.updates:
                result=evaluate(brain,cfg,meta,profile,retina_stream=validation_retina,reference_speed=a.speed,
                    episodes=a.evaluation_episodes,batch=a.batch,controllers=('brain',),
                    seed=a.development_seed,**evaluation_options)
                (out/f'development-{iteration:05d}.json').write_text(json.dumps(result,indent=2))
                save_candidate(out/f'step-{iteration:05d}.pt',iteration,result)
                if (evaluation_rank(result['brain'])<evaluation_rank(best_report)
                        and acceptable_tracking(result['brain'],baseline['brain'])):
                    best_state={k:v.detach().cpu().clone() for k,v in brain.state_dict().items()}
                    best_report=result['brain'];best_iteration=iteration
                print('Development',iteration,json.dumps({k:v for k,v in result['brain'].items() if k!='rows'}),
                      'selected',best_iteration,flush=True)
    # This seed is never used to select a snapshot. Evaluate both the unchanged
    # parent and selected candidate after selection, on the same complete tasks.
    brain.load_state_dict(initial)
    test_parent=evaluate(brain,cfg,meta,profile,retina_stream=validation_retina,reference_speed=a.speed,
        seed=config['test_seed'],episodes=a.test_episodes,batch=a.batch,**evaluation_options)
    brain.load_state_dict(best_state)
    test_candidate=evaluate(brain,cfg,meta,profile,retina_stream=validation_retina,reference_speed=a.speed,
        seed=config['test_seed'],episodes=a.test_episodes,batch=a.batch,controllers=('brain',),**evaluation_options)
    result=dict(selected_iteration=best_iteration,parent=test_parent,candidate=test_candidate,
        improved_on_test=(best_iteration>0 and evaluation_rank(test_candidate['brain'])<evaluation_rank(test_parent['brain'])
                          and acceptable_tracking(test_candidate['brain'],test_parent['brain'])))
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    audit=save_candidate(out/'candidate.pt',best_iteration,test_candidate)
    (out/'weights-audit.json').write_text(json.dumps(audit,indent=2))
    print('Candidate',str(out/'candidate.pt'),'iteration',best_iteration,'improved on test',result['improved_on_test'],flush=True)


if __name__=='__main__':
    main()
