"""Experimental differentiable surrogate of measured game-input response.

This bypasses the physical PID/mixer model: it identifies effective thrust and
angular response, not motor torques. Calibration uses game-processed inputs;
brain commands must first pass through the deployed mapping and radial limits.
No runtime controller imports this module. Pulse validation alone does not
qualify its translation model, high-speed drag, collisions or acrobatics.
"""
from dataclasses import dataclass

import torch

from ..brain.retina import brain_to_processed
from .quad import QuadState, quat_from_axis_angle, quat_mul, quat_to_mat, yaw_of


def processed_command(action, calibration):
    """Match the inverse-pad mapping followed by the game's unit-circle limits."""
    desired=brain_to_processed(action,calibration)
    left=desired[:,[0,3]];right=desired[:,[1,2]]
    left=left/left.norm(dim=-1,keepdim=True).clamp_min(1.)
    right=right/right.norm(dim=-1,keepdim=True).clamp_min(1.)
    return torch.stack((left[:,0],right[:,0],right[:,1],left[:,1]),-1)


@dataclass
class IdentifiedState:
    quad: QuadState
    drive: torch.Tensor
    sensed_omega: torch.Tensor

    def detach(self):
        return IdentifiedState(self.quad.detach(),self.drive.detach(),self.sensed_omega.detach())


class IdentifiedSim:
    """Explicit measured profile, with per-episode uncertainty for training.

    Profile keys: thrust, axes, translation_drag_s_inv, rpm_slope, rpm_intercept.
    Horizontal drag and sub-frame response constants need declared uncertainty;
    they must not be described as identified by short angular pulses alone.
    Command/observation delays and image gaps belong to the rollout, outside this
    plant. The gyro observation reproduces the live half-life filter.
    """
    def __init__(self,profile,calibration,device='cpu',dt=.01):
        if dt<=0:
            raise ValueError('Use a positive simulation timestep')
        self.profile,self.calibration,self.device,self.dt=profile,calibration,device,dt
        values=[profile['thrust'][k] for k in ('full_input_twr','exponent','response_tau_s')]
        values += [profile['axes'][a][k] for a in ('roll','pitch','yaw')
                   for k in ('coefficient_deg_s','response_tau_s')]
        if not all(float(v)>0 and torch.isfinite(torch.tensor(v)) for v in values):
            raise ValueError('Use finite positive measured response parameters')
        if any(not profile['axes'][a]['super_after_expo'] for a in ('roll','pitch','yaw')):
            raise ValueError('This surrogate requires explicitly fitted post-expo saturation')
        drag=torch.as_tensor(profile['translation_drag_s_inv'])
        if drag.shape!=(3,) or not torch.isfinite(drag).all() or (drag<0).any() or calibration['max_rpm']<=0:
            raise ValueError('Use three nonnegative finite drag coefficients and positive RPM normalization')
        self.randomize(1,0.)

    def randomize(self,batch,scale=.2,generator=None):
        if not 0<=scale<1:
            raise ValueError('Use a bounded uncertainty scale in [0,1)')
        def jitter(value):
            base=torch.as_tensor(value,dtype=torch.float32,device=self.device)
            shape=(batch,)+tuple(base.shape)
            if scale==0:
                return base.expand(shape).clone()
            return base*(1+scale*(2*torch.rand(shape,device=self.device,generator=generator)-1))
        thrust=self.profile['thrust'];axes=self.profile['axes']
        self.twr=jitter(thrust['full_input_twr'])
        self.exponent=jitter(thrust['exponent'])
        self.thrust_tau=jitter(thrust['response_tau_s'])
        self.rate_coefficient=jitter([axes[a]['coefficient_deg_s'] for a in ('roll','pitch','yaw')])
        self.rate_tau=jitter([axes[a]['response_tau_s'] for a in ('roll','pitch','yaw')])
        if scale and 'rate_response_uncertainty_s' in self.profile:
            low,high=self.profile['rate_response_uncertainty_s']
            if not 0<low<high:
                raise ValueError('Use positive ordered response uncertainty bounds')
            self.rate_tau=low+(high-low)*torch.rand((batch,3),device=self.device,generator=generator)
        self.expo=torch.tensor([axes[a]['expo'] for a in ('roll','pitch','yaw')],device=self.device)
        self.super_rate=torch.tensor([axes[a]['super_rate'] for a in ('roll','pitch','yaw')],device=self.device)
        self.drag=jitter(self.profile['translation_drag_s_inv'])
        if scale and 'translation_drag_uncertainty_s_inv' in self.profile:
            bounds=torch.as_tensor(self.profile['translation_drag_uncertainty_s_inv'],device=self.device)
            if (bounds.shape!=(3,2) or not torch.isfinite(bounds).all()
                    or (bounds[:,0]<0).any() or (bounds[:,0]>bounds[:,1]).any()):
                raise ValueError('Use three finite nonnegative ordered drag uncertainty bounds')
            self.drag=bounds[:,0]+(bounds[:,1]-bounds[:,0])*torch.rand((batch,3),device=self.device,generator=generator)

    def hover(self,batch,height=6.):
        if self.twr.shape[0]!=batch:
            raise ValueError('Randomize for the requested batch before creating states')
        q=QuadState.hover(batch,self.device,height)
        drive=(1/self.twr).pow(1/self.exponent)
        q.motor=self.motor_fraction(drive)
        return IdentifiedState(q,drive,q.omega.clone())

    def motor_fraction(self,drive):
        rpm=self.profile['rpm_intercept']+self.profile['rpm_slope']*drive
        return (rpm/self.calibration['max_rpm']).clamp(0,1)[:,None].expand(-1,4)

    def step(self,state,action):
        q=state.quad;dt=self.dt
        command=processed_command(action,self.calibration)
        target_drive=(command[:,0]+1)*.5
        decay=torch.exp(-dt/self.thrust_tau)
        drive=target_drive+(state.drive-target_drive)*decay
        mid_drive=target_drive+(state.drive-target_drive)*torch.sqrt(decay)
        mean_power=(state.drive.clamp_min(0).pow(self.exponent)
                    +4*mid_drive.clamp_min(0).pow(self.exponent)+drive.clamp_min(0).pow(self.exponent))/6
        sticks=command[:,1:]*command.new_tensor([-1.,1.,-1.])
        shaped=sticks*(1-self.expo+self.expo*sticks.abs().pow(3))
        target_rate=torch.deg2rad(self.rate_coefficient*shaped/(1-self.super_rate*shaped.abs()).clamp_min(.01))
        rate_decay=torch.exp(-dt/self.rate_tau)
        omega=target_rate+(q.omega-target_rate)*rate_decay
        average_rate=target_rate+(q.omega-target_rate)*(self.rate_tau/dt)*(1-rate_decay)
        quaternion=quat_mul(q.quat,quat_from_axis_angle(average_rate*dt))
        quaternion=quaternion/quaternion.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        mid_quaternion=quat_mul(q.quat,quat_from_axis_angle(average_rate*dt*.5))
        rotation=quat_to_mat(mid_quaternion)
        body_velocity=torch.einsum('bji,bj->bi',rotation,q.vel)
        drag=torch.einsum('bij,bj->bi',rotation,self.drag*body_velocity)
        acceleration=rotation[:,:,2]*(9.80665*self.twr*mean_power)[:,None]-drag
        acceleration=acceleration+action.new_tensor([0.,0.,-9.80665])
        position=q.pos+q.vel*dt+.5*acceleration*dt*dt
        velocity=q.vel+acceleration*dt
        crashed=q.crashed|(position[:,2]<0)
        quad=QuadState(position,velocity,quaternion,omega,self.motor_fraction(drive),crashed)
        sensed_omega=.5*state.sensed_omega+.5*average_rate
        return IdentifiedState(quad,drive,sensed_omega)

    def sensors(self,state):
        q=state.quad;rotation=quat_to_mat(q.quat)
        return dict(gyro=state.sensed_omega,gravity_body=-rotation[:,2,:],
            vel_body=torch.einsum('bji,bj->bi',rotation,q.vel),vel_world=q.vel,pos=q.pos,quat=q.quat,
            up=rotation[:,2,2],altitude=q.pos[:,2:3],yaw=yaw_of(q.quat)[:,None])
