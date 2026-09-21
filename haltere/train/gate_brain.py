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

from .bptt import load_checkpoint, make_world
from .human_brain import export
from .thermal import wait_if_hot
from ..brain.gate_senses import gate_observation, aperture_crossing
from ..brain.retina import RETINA_DIM
from ..sim.quad import QuadState, quat_from_euler, quat_to_mat
from ..vision.datasets import sha256


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


class GateRollout:
    def __init__(self, brain, cfg, batch=12, evaluation=False, teacher=None, turns=False):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.train.randomize = .1
        self.cfg.train.randomize_ctl = .1
        self.cfg.quad.gyro_noise = 0.
        self.brain, self.B, self.evaluation = brain, batch, evaluation
        self.turns = turns
        self.teacher = teacher
        self.teacher_W = teacher.weight_matrix().detach() if teacher is not None else None
        if evaluation:
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
        self.bias = torch.randn(B,3,device=dev)*torch.tensor([.2,.08,.08],device=dev)
        self.vs, self.state = self.vehicle.wrap(q), self.brain.init_state(B)
        self.teacher_state = self.teacher.init_state(B) if self.teacher is not None else None
        self.age = 0
        self.airborne = q.pos[:,2] > .3
        self.crossed = torch.zeros(B,dtype=torch.bool,device=dev)
        self.ever_crashed = self.crossed.clone()
        self.out_of_view_steps = torch.zeros(B,device=dev)
        self.lost_gate = self.crossed.clone()
        a = torch.zeros(B,4,device=dev)
        a[:,0] = 2*self.vehicle.sim.hover_command()-1
        self.delay = deque(a.clone() for _ in range(self.cfg.train.delay_steps))
        # Live arming runs the neural dynamics at neutral throttle before motor
        # release. Avoid training a launch impulse caused by an uninitialized RNN.
        R = quat_to_mat(q.quat)
        relative = (R.transpose(-1,-2)@(self.gate+self.bias-q.pos)[...,None]).squeeze(-1)
        obs = gate_observation(self.vehicle.sim.sensors(q),q.motor.mean(-1,keepdim=True),self.cfg.task,
                               torch.zeros(B,RETINA_DIM,device=dev),relative)
        with torch.no_grad():
            W = self.brain.weight_matrix()
            for _ in range(50):
                _,self.state,_ = self.brain(obs,self.state,W)

    def window(self, steps=64):
        brain = self.brain
        W = brain.weight_matrix()
        self.state = brain.detach_state(self.state)
        costs, imitation = [], []
        retina = torch.zeros(self.B,RETINA_DIM,device=brain.device)
        for t in range(steps):
            q = self.vs.quad
            R = quat_to_mat(q.quat)
            relative = (R.transpose(-1,-2) @ (self.gate+self.bias-q.pos)[...,None]).squeeze(-1)
            if self.evaluation:
                # Visibility is checked before crossing, except the final 2 m
                # when an arch can fill the image with its centre outside it.
                visual_point = relative.detach()+1.5*R[:,2,:].detach()
                camera_point = visual_point @ self.camera_rotation.T
                depth = camera_point[:,2].clamp_min(.01)
                visible = ((camera_point[:,2]>.1) & (100*camera_point[:,0].abs()/depth<192)
                           & (100*camera_point[:,1].abs()/depth<108))
                approaching = (~self.crossed) & (((self.gate-q.pos.detach())*self.normal).sum(-1)>2.)
                self.out_of_view_steps = torch.where(approaching & ~visible,self.out_of_view_steps+1,0.)
                self.lost_gate |= self.out_of_view_steps*self.cfg.brain.dt>.5
            obs = gate_observation(self.vehicle.sim.sensors(q),q.motor.mean(-1,keepdim=True),self.cfg.task,retina,relative)
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
                scale = action.new_tensor([.23,.19,.084,.184])
                errors = ((action-target)/scale).square()
                if self.turns:
                    # Preserve learned motor stabilization while allowing the
                    # physical loss to teach braking and sideways correction.
                    errors = errors*action.new_tensor([1.,.25,.25,1.])
                imitation.append(errors.mean())
            self.delay.append(action)
            self.vs = self.vehicle.step(self.vs,self.delay.popleft())
            after = self.vs.quad
            self.airborne |= after.pos[:,2].detach()>.3
            after.crashed = after.crashed & self.airborne
            self.ever_crashed |= after.crashed.detach()
            crossed,_ = aperture_crossing(q.pos.detach(),after.pos.detach(),self.gate,normal=self.normal)
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
            costs.append(cost.mean())
        self.age += steps
        physical = torch.stack(costs).mean()
        return (.8 if self.turns else .3)*physical+3.*torch.stack(imitation).mean() if imitation else physical

    def detach(self):
        self.vs = self.vs.detach()
        self.state = self.brain.detach_state(self.state)
        self.delay = deque(a.detach() for a in self.delay)


@torch.no_grad()
def evaluate(brain,cfg,seconds=25,seed=481,turns=False):
    with torch.random.fork_rng(devices=[brain.device] if brain.device.type=='cuda' else []):
        torch.manual_seed(seed)
        r = GateRollout(brain,cfg,batch=12,evaluation=True,turns=turns)
        max_speed, max_height = 0., 0.
        for _ in range(round(seconds/cfg.brain.dt/50)):
            r.window(50)
            max_speed = max(max_speed,float(r.vs.quad.vel.norm(dim=-1).max()))
            max_height = max(max_height,float(r.vs.quad.pos[:,2].max()))
        return dict(seed=seed,episodes=r.B,seconds=seconds,turns=turns,crossings=int(r.crossed.sum()),
                    camera_viable_crossings=int((r.crossed & ~r.lost_gate).sum()),lost_gate=int(r.lost_gate.sum()),
                    crashed=int(r.ever_crashed.sum()),max_speed=max_speed,max_height=max_height,
                    final_positions=r.vs.quad.pos.tolist(),gates=r.gate.tolist(),
                    measured_gate_input=True,synthetic_perception=True,runtime_requires_teacher=False)


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    brain,cfg,_ = load_checkpoint(args.checkpoint,args.device)
    parent = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    meta = copy.deepcopy(parent['visual_brain'])
    meta.pop('schema',None)
    meta.update(runtime_requires_teacher=False,gate_sensor=dict(
        checkpoint=args.detector,sha256=sha256(args.detector),focal_320=100.,tilt_deg=30.,
        centre_offset_m=1.5,raw_retina_active=False,
        goal_encoding='horizontal magnitude bounded at 3m, vertical error bounded at 3m'),
        gate_training=dict(parent_sha256=sha256(args.checkpoint),objective='2 m/s through varied gate apertures',
                           seed=args.seed,teacher_used=False,learning_rate=args.lr,
                           iterations=args.iters,neural_warmup_steps=50),qualified=False)
    if args.turns:
        meta['gate_training'].update(objective='camera-facing moving turns with approach braking',
                                     turns=True,global_heading_randomized=True)
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
    baseline = evaluate(brain,cfg,turns=args.turns)
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2))
    print(json.dumps({'baseline':baseline}),flush=True)
    rollout = GateRollout(brain,cfg,teacher=teacher,turns=args.turns)
    started = time.time()
    with (out/'training.jsonl').open('w') as log:
        for it in range(args.iters):
            if it%10==0:
                wait_if_hot(78.)
            beyond = ((rollout.vs.quad.pos-rollout.gate)*rollout.normal).sum(-1)>3
            if rollout.age>=1500 or bool(rollout.ever_crashed.any()) or bool(beyond.any()):
                rollout.reset()
            loss = rollout.window()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(brain.parameters(),1.)
            opt.step()
            rollout.detach()
            if it%10==0 or it+1==args.iters:
                row = dict(iter=it+1,loss=float(loss.detach()),grad=float(norm),seconds=time.time()-started,
                           mean_pos=rollout.vs.quad.pos.mean(0).tolist(),crossings=int(rollout.crossed.sum()))
                print(json.dumps(row),flush=True)
                log.write(json.dumps(row)+'\n'); log.flush()
            if (it+1)%50==0 or it+1==args.iters:
                changes = {n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in brain.named_parameters()}
                meta['gate_training']['weight_max_changes'] = changes
                export(out/'last.pt',brain,cfg,meta,it+1)
    ev = evaluate(brain,cfg,turns=args.turns)
    (out/'evaluation.json').write_text(json.dumps(ev,indent=2))
    print(json.dumps(ev),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint'); p.add_argument('--out',required=True)
    p.add_argument('--detector',default='artifacts/gatenet_best.pt')
    p.add_argument('--device',default='cuda'); p.add_argument('--iters',type=int,default=150)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,default=917)
    p.add_argument('--motor-teacher',default='',help='Optional frozen motor brain for training-only stabilization labels')
    p.add_argument('--turns',action='store_true',help='Train moving approaches with varied gate bearing and world heading')
    train(p.parse_args())


if __name__=='__main__':
    main()
