"""Fine-tune the latest fly brain for gate approach and aperture crossings.

Synthetic gate positions are available only in this training simulator. Live
flight obtains the same body-relative measurement from the camera detector.
No motor teacher or navigation predictor is needed by the exported model.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path

import torch

from .bptt import ExperimentConfig, load_checkpoint, make_world
from .human_brain import export
from .thermal import wait_if_hot
from ..brain.gate_senses import gate_observation, aperture_crossing
from ..brain.retina import RETINA_DIM
from ..sim.quad import QuadState, quat_from_euler, quat_to_mat
from ..vision.datasets import sha256


def calibrated_dynamics(cfg, profile):
    result = cfg.to_dict()
    overrides = profile['overrides']
    if not overrides or set(overrides)-{'quad','rates','ctl'}:
        raise ValueError('Dynamics calibration may only override quad, rates and ctl')
    for section,values in overrides.items():
        if set(values)-set(result[section]):
            raise ValueError(f'Unknown dynamics field in {section}')
        result[section].update(values)
    return ExperimentConfig.from_dict(result)


def turn_objective(position, velocity, rotation, gate, normal, crossed):
    """Rotation-invariant training targets; none of these targets run in flight."""
    delta = gate[...,:2]-position[...,:2]
    direction = delta/delta.norm(dim=-1,keepdim=True).clamp_min(.1)
    direction = torch.where(crossed[...,None].clone(),normal[...,:2],direction)
    facing = (rotation[...,:2,0]*direction).sum(-1)
    speed = 2.*(position[...,2].detach()/1.).clamp(0,1)*facing.detach().clamp(0,1).square()
    desired = speed[...,None]*direction
    cost = (3.*(velocity[...,:2]-desired).square().sum(-1)
            +4.*(position[...,2]-gate[...,2]).square()+3.*velocity[...,2].square()
            +2.*(1-rotation[...,2,2])+4.*(1-facing))
    return cost


def gate_in_view(relative, rotation, camera_rotation, centre_offset=1.5):
    """Synthetic detector visibility, including the calibrated visual offset."""
    point = (relative.detach()+centre_offset*rotation[:,2,:].detach()) @ camera_rotation.T
    depth = point[:,2].clamp_min(.01)
    return ((point[:,2]>.1) & (100*point[:,0].abs()/depth<160)
            & (100*point[:,1].abs()/depth<90))


def search_objective(velocity, rotation, omega, height_error=None):
    # No hidden gate direction or altitude enters the search target. Learn to
    # brake and keep vertical velocity near zero while looking to the left.
    cost=(4.*velocity.square().sum(-1)+2.*(1-rotation[:,2,2])
          +.15*omega[:,:2].square().sum(-1)+(omega[:,2]-.65).square())
    return cost if height_error is None else cost+6.*height_error.square()


def teacher_hover_command(sim, vertical):
    """Training-only feedforward using each world's actual thrust law.

    A fixed nominal hover value conflicts with the randomized physics, and
    square-root tilt compensation is correct only for a quadratic thrust law.
    No simulation parameters are exposed to the deployed brain.
    """
    exponent=sim.thrust_exp.reshape(-1)
    return (1./(sim.twr.reshape(-1)*vertical.clamp_min(.6))).pow(1./exponent)


class GateRollout:
    def __init__(self, brain, cfg, batch=12, evaluation=False, teacher=None, turns=False, motor_anchor=.25,
                 search=False,elevation=False,height_invariant=False,centre_offset=1.5,gravity_aligned_height=False,
                 throttle_anchor=1.,height_gain=.6,search_height_anchor=False,vertical_recovery_speed=0.):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.train.randomize = .1
        self.cfg.train.randomize_ctl = .1
        self.cfg.quad.gyro_noise = 0.
        self.brain, self.B, self.evaluation = brain, batch, evaluation
        self.turns = turns
        self.search = search
        self.elevation,self.height_invariant,self.centre_offset = elevation,height_invariant,centre_offset
        self.gravity_aligned_height = gravity_aligned_height
        self.search_height_anchor=search_height_anchor
        self.motor_anchor = motor_anchor
        self.throttle_anchor,self.height_gain=throttle_anchor,height_gain
        self.vertical_recovery_speed=vertical_recovery_speed
        self.teacher = teacher
        self.teacher_W = teacher.weight_matrix().detach() if teacher is not None else None
        if evaluation or search:
            from ..vision.camera import body_to_cam_matrix
            self.camera_rotation = torch.tensor(body_to_cam_matrix(30.),device=brain.device,dtype=torch.float32)
        self.vehicle, _ = make_world(self.cfg, batch, brain.device)
        self.reset()

    def reset(self):
        B, dev = self.B, self.brain.device
        q = QuadState.hover(B, dev, 1.5)
        q.motor[:] = self.vehicle.sim.hover_command()
        q.pos[:, 2] = .8 + 1.2*torch.rand(B, device=dev)
        q.pos[:B//2, 2] = .03
        q.motor[:B//2] = .04
        yaw = .4*(2*torch.rand(B, device=dev)-1)
        q.quat = quat_from_euler(yaw*0, yaw*0, yaw)
        self.gate = torch.zeros(B, 3, device=dev)
        self.gate[:, 0] = (23. if self.evaluation else 6.) + (0. if self.evaluation else 12.)*torch.rand(B, device=dev)
        self.gate[:, 1] = 2*(2*torch.rand(B, device=dev)-1)
        self.gate[:, 2] = 1.3 + .4*torch.rand(B, device=dev)
        self.normal = torch.zeros_like(self.gate)
        self.normal[:,0] = 1.
        if self.turns:
            heading = torch.rand(B,device=dev)*2*torch.pi-torch.pi
            angle = 1.2*(2*torch.rand(B,device=dev)-1)
            angle[:B//3] *= .15  # retain takeoff/straight approaches
            if self.evaluation:
                angle = torch.linspace(-1.05,1.05,B,device=dev)
            if self.search:
                angle[B//3:] = torch.pi*(2*torch.rand(B-B//3,device=dev)-1)
                if self.evaluation:
                    angle[B//3:] = torch.linspace(-2.8,2.8,B-B//3,device=dev)
            q.quat = quat_from_euler(heading*0,heading*0,heading)
            q.pos[B//3:,2] = .8+1.2*torch.rand(B-B//3,device=dev)
            q.motor[B//3:] = self.vehicle.sim.hover_command()
            speed = .8+torch.rand(B,device=dev)
            speed[:B//3] = 0.
            q.vel[:,:2] = torch.stack((heading.cos(),heading.sin()),-1)*speed[:,None]
            direction = heading+angle
            self.normal[:,:2] = torch.stack((direction.cos(),direction.sin()),-1)
            distance = 12.+8.*torch.rand(B,device=dev)
            self.gate[:,:2] = self.normal[:,:2]*distance[:,None]
        if self.elevation:
            # Retain takeoff/level examples, and translate the rest across the
            # whole hillside height band with both uphill and downhill goals.
            count=B-B//3
            q.pos[B//3:,2]=2.+26.*torch.rand(count,device=dev)
            dz=5.*(2*torch.rand(count,device=dev)-1)
            if self.evaluation:
                q.pos[B//3:,2]=torch.linspace(2.,28.,count,device=dev)
                dz=torch.tensor([5.,-1.,3.,-4.,5.,-5.,2.,-3.],device=dev).repeat((count+7)//8)[:count]
            self.gate[B//3:,2]=(q.pos[B//3:,2]+dz).clamp_min(1.2)
            if not self.evaluation and self.vertical_recovery_speed:
                # Expose the student to climbs/descents it does not yet produce
                # on its own. Otherwise an over-damped parent never visits the
                # vertical speeds needed to learn a sustained hillside climb.
                q.vel[B//3:,2]=self.vertical_recovery_speed*(2*torch.rand(count,device=dev)-1)
        self.bias = torch.randn(B,3,device=dev)*torch.tensor([.2,.08,.08],device=dev)
        self.vs, self.state = self.vehicle.wrap(q), self.brain.init_state(B)
        self.teacher_state = self.teacher.init_state(B) if self.teacher is not None else None
        self.age = 0
        self.airborne = q.pos[:,2] > .3
        self.crossed = torch.zeros(B,dtype=torch.bool,device=dev)
        self.ever_crashed = self.crossed.clone()
        self.out_of_view_steps = torch.zeros(B,device=dev)
        self.lost_gate = self.crossed.clone()
        self.measurement_age = torch.full((B,),float('inf'),device=dev)
        self.searching = self.crossed.clone()
        self.search_height=q.pos[:,2].detach().clone()
        self.ever_searched = self.crossed.clone()
        self.reacquired = self.crossed.clone()
        self.search_streak = torch.zeros(B,device=dev)
        self.max_search = torch.zeros(B,device=dev)
        self.first_crossing_s = torch.full((B,),float('nan'),device=dev)
        self.crashed_before_crossing = self.crossed.clone()
        a = torch.zeros(B,4,device=dev)
        a[:,0] = 2*self.vehicle.sim.hover_command()-1
        self.delay = deque(a.clone() for _ in range(self.cfg.train.delay_steps))
        # Live arming runs the neural dynamics at neutral throttle before motor
        # release. Avoid training a launch impulse caused by an uninitialized RNN.
        R = quat_to_mat(q.quat)
        relative = (R.transpose(-1,-2)@(self.gate+self.bias-q.pos)[...,None]).squeeze(-1)
        if self.search:
            relative = self.camera_measurement(relative,R)
        obs = gate_observation(self.vehicle.sim.sensors(q),q.motor.mean(-1,keepdim=True),self.cfg.task,
                               torch.zeros(B,RETINA_DIM,device=dev),relative,self.height_invariant,self.gravity_aligned_height,
                               self.search_height_cue())
        with torch.no_grad():
            W = self.brain.weight_matrix()
            for _ in range(50):
                _,self.state,_ = self.brain(obs,self.state,W)

    def camera_measurement(self, relative, rotation):
        visible = gate_in_view(relative,rotation,self.camera_rotation,self.centre_offset)
        self.measurement_age = torch.where(visible,0.,self.measurement_age+self.cfg.brain.dt)
        nearby = (relative.detach().norm(dim=-1)<8.) & (relative[:,0].detach()>-1.)
        valid = self.measurement_age <= torch.where(nearby,2.,.5)
        self.search_height=torch.where(~valid & ~self.searching,self.vs.quad.pos[:,2].detach(),self.search_height)
        self.reacquired |= self.searching & visible & ~self.crossed
        self.searching = ~valid
        self.ever_searched |= self.searching & ~self.crossed
        self.search_streak = torch.where(self.searching,self.search_streak+self.cfg.brain.dt,0.)
        self.max_search = torch.maximum(self.max_search,torch.where(self.crossed,0.,self.search_streak))
        # Hidden gate coordinates are never delivered to the brain. The same
        # zero goal marks an expired camera measurement in the live runtime.
        return torch.where(valid[:,None],relative,torch.zeros_like(relative))

    def search_height_cue(self):
        if not self.search_height_anchor:
            return None
        return torch.where(self.searching,self.vs.quad.pos[:,2]-self.search_height,0.)[:,None]

    def window(self, steps=64):
        brain = self.brain
        W = brain.weight_matrix()
        self.state = brain.detach_state(self.state)
        costs, imitation, motor_errors = [], [], []
        retina = torch.zeros(self.B,RETINA_DIM,device=brain.device)
        for t in range(steps):
            q = self.vs.quad
            R = quat_to_mat(q.quat)
            relative = (R.transpose(-1,-2) @ (self.gate+self.bias-q.pos)[...,None]).squeeze(-1)
            if self.evaluation:
                # Visibility is checked before crossing, except the final 2 m
                # when an arch can fill the image with its centre outside it.
                visible = gate_in_view(relative,R,self.camera_rotation,self.centre_offset)
                approaching = (~self.crossed) & (((self.gate-q.pos.detach())*self.normal).sum(-1)>2.)
                self.out_of_view_steps = torch.where(approaching & ~visible,self.out_of_view_steps+1,0.)
                self.lost_gate |= self.out_of_view_steps*self.cfg.brain.dt>.5
            if self.search:
                relative = self.camera_measurement(relative,R)
            obs = gate_observation(self.vehicle.sim.sensors(q),q.motor.mean(-1,keepdim=True),self.cfg.task,retina,relative,
                                   self.height_invariant,self.gravity_aligned_height,self.search_height_cue())
            if t and t%8 == 0:
                self.state = brain.detach_state(self.state)
            action,self.state,_ = brain(obs,self.state,W)
            if self.teacher is not None:
                with torch.no_grad():
                    teacher_obs = {k:v.detach() for k,v in obs.items() if k in self.teacher.channel_dims}
                    if 'retina' not in self.teacher.channel_dims:
                        teacher_obs['compass'] = torch.zeros_like(obs['compass'])
                        teacher_obs['compass'][:,0] = 1
                    target,self.teacher_state,_ = self.teacher(teacher_obs,self.teacher_state,self.teacher_W)
                    target = target.clone()
                    # Training-only camera-facing labels. The exported brain
                    # learns the yaw output; flight applies no yaw correction.
                    bearing = torch.atan2(relative[:,1],relative[:,0].clamp_min(.5))
                    target[:,3] = (-(.4 if self.turns else .18)*bearing
                                    +(.12 if self.turns else .08)*q.omega[:,2]).clamp(-.4,.4)
                    if self.search:
                        target[:,3] = torch.where(self.searching,-.24,target[:,3])
                    if self.elevation:
                        # Training-only vertical velocity labels replace the
                        # older hover brain's preference for launch altitude.
                        # Use only the visible/remembered measurement (zero in
                        # search), so a hidden gate cannot leak into supervision.
                        measured_world=(R@relative.detach()[...,None]).squeeze(-1)
                        desired_vz=(self.height_gain*measured_world[:,2]).clamp(-1.2,1.2)
                        if self.search_height_anchor:
                            desired_vz=torch.where(self.searching,
                                (-.6*(q.pos[:,2]-self.search_height)).clamp(-1.2,1.2),desired_vz)
                        hover=teacher_hover_command(self.vehicle.sim,R[:,2,2])
                        target[:,0]=(2*hover-1+.1*(desired_vz-q.vel[:,2])).clamp(-.8,.2)
                scale = action.new_tensor([.23,.19,.084,.184])
                motor_errors.append((action.detach()-target).square().mean(0))
                errors = ((action-target)/scale).square()
                if self.turns:
                    # Preserve learned motor stabilization while allowing the
                    # physical loss to teach braking and sideways correction.
                    errors = errors*action.new_tensor([self.throttle_anchor,self.motor_anchor,self.motor_anchor,1.])
                imitation.append(errors.mean())
            self.delay.append(action)
            self.vs = self.vehicle.step(self.vs,self.delay.popleft())
            after = self.vs.quad
            self.airborne |= after.pos[:,2].detach()>.3
            after.crashed = after.crashed & self.airborne
            self.ever_crashed |= after.crashed.detach()
            self.crashed_before_crossing |= after.crashed.detach() & ~self.crossed
            crossed,_ = aperture_crossing(q.pos.detach(),after.pos.detach(),self.gate,normal=self.normal)
            self.first_crossing_s = torch.where(crossed & ~self.crossed & ~self.ever_crashed,
                                                (self.age+t+1)*self.cfg.brain.dt,self.first_crossing_s)
            self.crossed |= crossed & ~self.ever_crashed
            # Move through the opening, rather than stopping at its centre.
            # Desired velocities exist only in this differentiable training loss.
            desired_x = 2.*(after.pos[:,2].detach()/1.).clamp(0,1)
            desired_y = (.7*(self.gate[:,1]-after.pos[:,1])).clamp(-1.,1.)
            R2 = quat_to_mat(after.quat)
            cost = (2.*(after.vel[:,0]-desired_x).square()
                    +3.*(after.vel[:,1]-desired_y).square()
                    +4.*(after.pos[:,2]-self.gate[:,2]).square()+3.*after.vel[:,2].square()
                    +2.*(1-R2[:,2,2])+.15*after.omega.square().sum(-1)
                    +2.*(1-R2[:,0,0])+.04*action[:,1:].square().sum(-1))
            if self.turns:
                cost = (turn_objective(after.pos,after.vel,R2,self.gate,self.normal,self.crossed)
                        +.15*after.omega.square().sum(-1)+.04*action[:,1:].square().sum(-1))
            if self.search:
                height_error=after.pos[:,2]-self.search_height if self.search_height_anchor else None
                cost = torch.where(self.searching.clone(),search_objective(after.vel,R2,after.omega,height_error),cost)
            costs.append(cost.mean())
        self.age += steps
        physical = torch.stack(costs).mean()
        self.last_diagnostics=dict(physical_loss=float(physical.detach()))
        if motor_errors:
            self.last_diagnostics['motor_teacher_rmse']=torch.stack(motor_errors).mean(0).sqrt().tolist()
        return (.8 if self.turns else .3)*physical+3.*torch.stack(imitation).mean() if imitation else physical

    def detach(self):
        self.vs = self.vs.detach()
        self.state = self.brain.detach_state(self.state)
        self.delay = deque(a.detach() for a in self.delay)


@torch.no_grad()
def evaluate(brain,cfg,seconds=25,seed=481,turns=False,search=False,elevation=False,
             height_invariant=False,centre_offset=1.5,gravity_aligned_height=False,search_height_anchor=False):
    with torch.random.fork_rng(devices=[brain.device] if brain.device.type=='cuda' else []):
        torch.manual_seed(seed)
        r = GateRollout(brain,cfg,batch=12,evaluation=True,turns=turns,search=search,elevation=elevation,
                        height_invariant=height_invariant,centre_offset=centre_offset,
                        gravity_aligned_height=gravity_aligned_height,search_height_anchor=search_height_anchor)
        max_speed, max_height = 0., 0.
        for _ in range(round(seconds/cfg.brain.dt/50)):
            r.window(50)
            max_speed = max(max_speed,float(r.vs.quad.vel.norm(dim=-1).max()))
            max_height = max(max_height,float(r.vs.quad.pos[:,2].max()))
        return dict(seed=seed,episodes=r.B,seconds=seconds,turns=turns,crossings=int(r.crossed.sum()),
                    elevation=elevation,height_invariant=height_invariant,centre_offset_m=centre_offset,
                    gravity_aligned_height=gravity_aligned_height,
                    search_height_anchor=search_height_anchor,
                    search=search,searched=int(r.ever_searched.sum()),reacquired=int(r.reacquired.sum()),
                    recovered_gate_crossings=int((r.crossed & r.reacquired).sum()),
                    camera_viable_crossings=int((r.crossed & ~(r.max_search>15.) & ~r.crashed_before_crossing).sum())
                        if search else int((r.crossed & ~r.lost_gate).sum()),
                    lost_gate=int(r.lost_gate.sum()),search_timeouts=int((r.max_search>15.).sum()),
                    max_search_seconds=r.max_search.tolist(),
                    first_crossing_seconds=[float(t) if torch.isfinite(t) else None for t in r.first_crossing_s],
                    crashed=int(r.ever_crashed.sum()),max_speed=max_speed,max_height=max_height,
                    final_positions=r.vs.quad.pos.tolist(),gates=r.gate.tolist(),
                    measured_gate_input=True,synthetic_perception=True,runtime_requires_teacher=False)


def train(args):
    if args.search and not args.turns:
        raise ValueError('Search training requires --turns')
    if args.elevation and not (args.turns and args.search and args.motor_teacher):
        raise ValueError('Elevation training requires --turns, --search and a training-only motor teacher')
    if args.search_height_anchor and not args.elevation:
        raise ValueError('Search height anchoring requires height-invariant elevation training')
    if not 0<=args.vertical_recovery_speed<=2 or (args.vertical_recovery_speed and not args.elevation):
        raise ValueError('Vertical recovery requires elevation training and a speed in [0, 2] m/s')
    if not 0<args.throttle_anchor<=100 or not 0<args.height_gain<=3:
        raise ValueError('Use 0 < throttle anchor <= 100 and 0 < height gain <= 3')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    brain,cfg,_ = load_checkpoint(args.checkpoint,args.device)
    if args.dynamics:
        cfg = calibrated_dynamics(cfg,json.loads(Path(args.dynamics).read_text()))
    parent = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    meta = copy.deepcopy(parent['visual_brain'])
    meta.pop('perception_rebind',None)  # this run updates the brain itself
    args.detector = args.detector or meta.get('gate_sensor',{}).get('checkpoint','artifacts/gatenet_best.pt')
    if args.centre_offset_m is None:
        args.centre_offset_m = meta.get('gate_sensor',{}).get('centre_offset_m',1.5)
    if not 0<=args.centre_offset_m<=3:
        raise ValueError('Invalid detector centre offset')
    evaluation_args=dict(turns=args.turns,search=args.search,elevation=args.elevation,
                         height_invariant=args.elevation,centre_offset=args.centre_offset_m,
                         gravity_aligned_height=args.gravity_aligned_height,search_height_anchor=args.search_height_anchor)
    meta.pop('schema',None)
    meta.update(runtime_requires_teacher=False,gate_sensor=dict(
        checkpoint=args.detector,sha256=sha256(args.detector),focal_320=100.,tilt_deg=30.,
        centre_offset_m=args.centre_offset_m,raw_retina_active=False,height_invariant=args.elevation,
        gravity_aligned_height=args.gravity_aligned_height,
        search_height_anchor=args.search_height_anchor,
        goal_encoding='horizontal magnitude bounded at 3m, vertical error bounded at 3m'),
        gate_training=dict(parent_sha256=sha256(args.checkpoint),objective='2 m/s through varied gate apertures',
                           seed=args.seed,teacher_used=False,learning_rate=args.lr,
                           iterations=args.iters,neural_warmup_steps=50),qualified=False)
    if args.turns:
        meta['gate_training'].update(objective='camera-facing moving turns with approach braking',
                                     turns=True,global_heading_randomized=True,motor_anchor=args.motor_anchor)
    if args.search:
        meta['gate_sensor']['missing_gate'] = 'zero_goal_neural_search'
        meta['gate_training'].update(search=True,search_yaw_label=-.24,
                                     camera_visibility_masked=True,close_memory_s=2.,far_memory_s=.5)
    if args.elevation:
        meta['gate_training'].update(elevation=True,altitude_range_m=[2.,28.],gate_height_delta_m=[-5.,5.],
                                    vertical_teacher='measured relative-height velocity target; training only',
                                    height_invariant=True,flow_velocity_reference_m=1.5,
                                    throttle_anchor=args.throttle_anchor,height_gain=args.height_gain)
        meta['gate_training']['hover_teacher']='per-environment randomized thrust law and tilt; training only'
        meta['gate_training']['initial_vertical_speed_range_mps']=[-args.vertical_recovery_speed,args.vertical_recovery_speed]
    if args.gravity_aligned_height:
        meta['gate_sensor']['goal_encoding']='gravity-aligned horizontal distance and height bounded at 3m; expressed in body frame'
    if args.search_height_anchor:
        meta['gate_sensor']['search_height_reference']='odometry height at start of each missing-gate search'
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    if args.dynamics:
        meta['gate_training']['dynamics'] = dict(path=args.dynamics,sha256=sha256(args.dynamics),
                                                 profile=json.loads(Path(args.dynamics).read_text()))
    teacher = None
    if args.motor_teacher:
        teacher,_,_ = load_checkpoint(args.motor_teacher,args.device)
        teacher.requires_grad_(False)
        meta['gate_training'].update(teacher_used=True,motor_teacher_sha256=sha256(args.motor_teacher),
                                    teacher_training_only=True)
    before = {n:p.detach().cpu().clone() for n,p in brain.named_parameters()}
    opt = torch.optim.Adam([
        dict(params=[p for n,p in brain.named_parameters() if n=='log_edge_gain'],lr=args.lr*.3),
        dict(params=[p for n,p in brain.named_parameters() if n!='log_edge_gain'],lr=args.lr)])
    baseline = evaluate(brain,cfg,seconds=45 if args.search else 25,**evaluation_args)
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2))
    print(json.dumps({'baseline':baseline}),flush=True)
    rollout = GateRollout(brain,cfg,teacher=teacher,motor_anchor=args.motor_anchor,
                          throttle_anchor=args.throttle_anchor,height_gain=args.height_gain,
                          vertical_recovery_speed=args.vertical_recovery_speed,**evaluation_args)
    started = time.time()
    with (out/'training.jsonl').open('w') as log:
        for it in range(args.iters):
            if it%10==0:
                wait_if_hot(78.)
            beyond = ((rollout.vs.quad.pos-rollout.gate)*rollout.normal).sum(-1)>3
            if rollout.age>=(3000 if args.search else 1500) or bool(rollout.ever_crashed.any()) or bool(beyond.any()):
                rollout.reset()
            loss = rollout.window()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(brain.parameters(),1.)
            opt.step()
            rollout.detach()
            if it%10==0 or it+1==args.iters:
                row = dict(iter=it+1,loss=float(loss.detach()),grad=float(norm),seconds=time.time()-started,
                           mean_pos=rollout.vs.quad.pos.mean(0).tolist(),crossings=int(rollout.crossed.sum()),
                           **rollout.last_diagnostics)
                print(json.dumps(row),flush=True)
                log.write(json.dumps(row)+'\n'); log.flush()
            if (it+1)%50==0 or it+1==args.iters:
                changes = {n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in brain.named_parameters()}
                meta['gate_training']['weight_max_changes'] = changes
                export(out/'last.pt',brain,cfg,meta,it+1)
    ev = evaluate(brain,cfg,seconds=45 if args.search else 25,**evaluation_args)
    (out/'evaluation.json').write_text(json.dumps(ev,indent=2))
    print(json.dumps(ev),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint'); p.add_argument('--out',required=True)
    p.add_argument('--detector',default='',help='Visual frontend; defaults to the parent checkpoint contract')
    p.add_argument('--centre-offset-m',type=float,default=None,help='Detector target offset; defaults to parent contract')
    p.add_argument('--device',default='cuda'); p.add_argument('--iters',type=int,default=150)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,default=917)
    p.add_argument('--motor-teacher',default='',help='Optional frozen motor brain for training-only stabilization labels')
    p.add_argument('--turns',action='store_true',help='Train moving approaches with varied gate bearing and world heading')
    p.add_argument('--search',action='store_true',help='Train neural search with missing-gate inputs and camera visibility')
    p.add_argument('--elevation',action='store_true',help='Train climbs/descents without launch-altitude sensory dependence')
    p.add_argument('--gravity-aligned-height',action='store_true',help='Preserve true vertical error while bounding distant gates')
    p.add_argument('--search-height-anchor',action='store_true',help='Train neural search to retain its starting height from local odometry')
    p.add_argument('--vertical-recovery-speed',type=float,default=0.,
                   help='Initial vertical speed variation during elevation training only (0 to 2 m/s)')
    p.add_argument('--dynamics',default='',help='JSON with measured quad/rates/ctl overrides and source provenance')
    p.add_argument('--motor-anchor',type=float,default=.25,help='Roll/pitch motor-teacher loss weight in turn training')
    p.add_argument('--throttle-anchor',type=float,default=1.,help='Throttle imitation weight in turn/elevation training')
    p.add_argument('--height-gain',type=float,default=.6,help='Training-only vertical velocity target per metre of height error')
    train(p.parse_args())


if __name__=='__main__':
    main()
