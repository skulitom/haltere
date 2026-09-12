"""Batched, differentiable quadrotor rigid-body simulator (PyTorch).

Frames: world is right-handed with z up; body is FLU (x forward, y left, z up).
Quaternions are [w, x, y, z] and rotate body vectors into the world frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

G = 9.81


@dataclass
class QuadParams:
    """Physical parameters (nominal values; see QuadSim.randomize for per-environment jitter)."""
    mass: float = 0.62                  # kg (5-inch class, e.g. Liftoff's Borrum 180 X)
    inertia: tuple[float, float, float] = (3.5e-3, 4.0e-3, 7.0e-3)  # kg m^2 (diagonal)
    arm_x: float = 0.09                 # m, motor longitudinal offset
    arm_y: float = 0.09                 # m, motor lateral offset
    twr: float = 6.0                    # thrust-to-weight ratio at full throttle
    thrust_exp: float = 1.5             # thrust = Tmax * cmd**thrust_exp
    motor_tau: float = 0.03             # s, motor first-order lag
    yaw_coeff: float = 0.012            # m, reaction torque per Newton of thrust
    spin: tuple[float, float, float, float] = (1.0, -1.0, -1.0, 1.0)  # (LF, RF, LB, RB), +1 = CCW from above
    drag_lin: tuple[float, float, float] = (0.05, 0.05, 0.10)   # N s/m, body frame
    drag_quad: tuple[float, float, float] = (0.25, 0.25, 0.45)  # N s^2/m^2, body frame
    ang_damp: float = 0.005             # N m s
    gyro_noise: float = 0.0             # rad/s


@dataclass
class QuadState:
    pos: torch.Tensor      # [B,3] world
    vel: torch.Tensor      # [B,3] world
    quat: torch.Tensor     # [B,4] body->world, [w,x,y,z]
    omega: torch.Tensor    # [B,3] body rad/s
    motor: torch.Tensor    # [B,4] motor speed after lag, [0,1]
    crashed: torch.Tensor  # [B] bool

    @property
    def B(self) -> int:
        return self.pos.shape[0]

    def detach(self) -> "QuadState":
        return QuadState(self.pos.detach(), self.vel.detach(), self.quat.detach(), self.omega.detach(),
                         self.motor.detach(), self.crashed)

    def clone(self) -> "QuadState":
        return QuadState(self.pos.clone(), self.vel.clone(), self.quat.clone(), self.omega.clone(),
                         self.motor.clone(), self.crashed.clone())

    @staticmethod
    def hover(B: int, device, height: float = 1.5) -> "QuadState":
        pos = torch.zeros(B, 3, device=device)
        pos[:, 2] = height
        quat = torch.zeros(B, 4, device=device)
        quat[:, 0] = 1.0
        return QuadState(pos, torch.zeros(B, 3, device=device), quat, torch.zeros(B, 3, device=device),
                         torch.zeros(B, 4, device=device), torch.zeros(B, dtype=torch.bool, device=device))

    def where(self, mask: torch.Tensor, other: "QuadState") -> "QuadState":
        """Rows where mask is True are taken from ``other``."""
        m1 = mask[:, None]
        return QuadState(torch.where(m1, other.pos, self.pos), torch.where(m1, other.vel, self.vel),
                         torch.where(m1, other.quat, self.quat), torch.where(m1, other.omega, self.omega),
                         torch.where(m1, other.motor, self.motor), torch.where(mask, other.crashed, self.crashed))


# ----------------------------------------------------------------------------- quaternion helpers

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    return torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], -1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], -1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], -1)], dim=-2)


def quat_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    small = angle < 1e-8
    k = torch.where(small, 0.5 - angle * angle / 48.0, torch.sin(half) / angle.clamp(min=1e-12))
    return torch.cat([torch.cos(half), axis_angle * k], dim=-1)


def quat_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """ZYX intrinsic (yaw about z, then pitch about y, then roll about x)."""
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], dim=-1)


def yaw_of(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate body vectors v [B,3] into the world frame."""
    return torch.einsum('bij,bj->bi', quat_to_mat(q), v)


def rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors v [B,3] into the body frame."""
    return torch.einsum('bji,bj->bi', quat_to_mat(q), v)


# ----------------------------------------------------------------------------- simulator

class QuadSim:
    def __init__(self, params: QuadParams, device, dt: float = 0.01, substeps: int = 4):
        self.p = params
        self.device = device
        self.dt = dt
        self.substeps = substeps
        self.sub_dt = dt / substeps
        t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)  # noqa: E731
        self.mass = t(params.mass)
        self.twr = t(params.twr)
        self.thrust_exp = t(params.thrust_exp)
        self.yaw_coeff = t(params.yaw_coeff)
        self.motor_tau = t(params.motor_tau)
        self.inertia = t(params.inertia)
        self.spin = t(params.spin)
        self.drag_lin = t(params.drag_lin)
        self.drag_quad = t(params.drag_quad)
        ax, ay = params.arm_x, params.arm_y
        self.motor_x = t([ax, ax, -ax, -ax])
        self.motor_y = t([ay, -ay, ay, -ay])
        self.g_vec = t([0.0, 0.0, -G])

    def randomize(self, B: int, scale: float = 0.15, generator=None) -> None:
        """Per-environment domain randomization of mass, thrust, drag and motor lag (+- scale)."""
        def jitter(base, shape):
            base = torch.as_tensor(base, dtype=torch.float32, device=self.device)
            r = 1.0 + scale * (2 * torch.rand(shape, device=self.device, generator=generator) - 1)
            return base * r
        self.mass = jitter(self.p.mass, (B,))
        self.twr = jitter(self.p.twr, (B,))
        self.drag_lin = jitter(self.p.drag_lin, (B, 3))
        self.drag_quad = jitter(self.p.drag_quad, (B, 3))
        self.motor_tau = jitter(self.p.motor_tau, (B, 1))
        self.thrust_exp = jitter(self.p.thrust_exp, (B, 1))
        self.yaw_coeff = jitter(self.p.yaw_coeff, (B, 1))

    def tmax_per_motor(self) -> torch.Tensor:
        return self.twr * self.mass * G / 4.0

    def hover_command(self) -> float:
        """Nominal per-motor command that balances gravity."""
        return float((1.0 / self.p.twr) ** (1.0 / self.p.thrust_exp))

    def step(self, st: QuadState, motor_cmd: torch.Tensor) -> QuadState:
        """Advance one control step (self.dt) with constant motor command; integrates `substeps` times."""
        for _ in range(self.substeps):
            st = self._substep(st, motor_cmd, self.sub_dt)
        return st

    def _substep(self, st: QuadState, motor_cmd: torch.Tensor, dt: float) -> QuadState:
        tau = self.motor_tau if self.motor_tau.dim() else self.motor_tau.expand(1)
        motor = st.motor + (motor_cmd - st.motor) * (1.0 - torch.exp(-dt / tau))
        tmax = self.tmax_per_motor()
        tmax = tmax[:, None] if tmax.dim() else tmax
        F = tmax * motor.clamp(min=1e-6).pow(self.thrust_exp)  # [B,4]
        thrust = F.sum(dim=1)
        taux = (F * self.motor_y).sum(dim=1)
        tauy = -(F * self.motor_x).sum(dim=1)
        tauz = -(F * self.spin * self.yaw_coeff).sum(dim=1)
        torque = torch.stack([taux, tauy, tauz], dim=1)

        R = quat_to_mat(st.quat)
        v_b = torch.einsum('bji,bj->bi', R, st.vel)
        drag_b = -self.drag_lin * v_b - self.drag_quad * v_b.abs() * v_b
        f_b = torch.stack([torch.zeros_like(thrust), torch.zeros_like(thrust), thrust], dim=1) + drag_b
        mass = self.mass[:, None] if self.mass.dim() else self.mass
        acc = self.g_vec + torch.einsum('bij,bj->bi', R, f_b) / mass

        Iw = self.inertia * st.omega
        gyro_torque = torch.cross(st.omega, Iw, dim=1)
        omega_dot = (torque - gyro_torque - self.p.ang_damp * st.omega) / self.inertia

        omega = st.omega + omega_dot * dt
        vel = st.vel + acc * dt
        pos = st.pos + vel * dt
        quat = quat_mul(st.quat, quat_from_axis_angle(omega * dt))
        quat = quat / quat.norm(dim=1, keepdim=True)

        crashed = st.crashed | (pos[:, 2] < 0.0)
        # hold crashed vehicles on the ground (a non-differentiable event, like a real crash)
        c = crashed[:, None]
        pos = torch.where(c, torch.cat([pos[:, :2], pos[:, 2:].clamp(min=0.0)], dim=1), pos)
        vel = torch.where(c, torch.zeros_like(vel), vel)
        omega = torch.where(c, torch.zeros_like(omega), omega)
        return QuadState(pos, vel, quat, omega, motor, crashed)

    # ------------------------------------------------------------------ sensors
    def sensors(self, st: QuadState, noise: bool = True) -> dict[str, torch.Tensor]:
        R = quat_to_mat(st.quat)
        gravity_body = torch.einsum('bji,bj->bi', R, self.g_vec.expand(st.B, 3)) / G  # unit vector "down"
        vel_body = torch.einsum('bji,bj->bi', R, st.vel)
        gyro = st.omega
        if noise and self.p.gyro_noise > 0:
            gyro = gyro + self.p.gyro_noise * torch.randn_like(gyro)
        return {
            'gyro': gyro,                    # rad/s, body
            'gravity_body': gravity_body,    # unit vector of gravity in body frame
            'vel_body': vel_body,            # m/s
            'vel_world': st.vel,
            'pos': st.pos,
            'quat': st.quat,
            'up': R[:, 2, 2],                # cos(tilt)
            'altitude': st.pos[:, 2:3],
            'yaw': yaw_of(st.quat)[:, None],
        }
