"""Coordinate conversions between Liftoff (Unity: left-handed, x right, y up, z forward) and the
simulator (right-handed, x forward, y left, z up). Quaternions from Liftoff are [x, y, z, w]; the
simulator uses [w, x, y, z]."""
from __future__ import annotations

import numpy as np

# sim = M @ unity  (an improper rotation: it also flips handedness)
M = np.array([[0.0, 0.0, 1.0],
              [-1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0]])


def unity_vec_to_sim(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) @ M.T


def sim_vec_to_unity(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) @ M


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Shepperd's method (robust for all rotations)."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def unity_quat_to_sim(q_xyzw: np.ndarray) -> np.ndarray:
    """Unity body->world quaternion [x,y,z,w] -> simulator body->world quaternion [w,x,y,z].

    Done through rotation matrices so the handedness flip is handled exactly: R_sim = M R_u M^T."""
    R_u = quat_xyzw_to_mat(np.asarray(q_xyzw, dtype=np.float64))
    R_s = M @ R_u @ M.T
    return mat_to_quat_wxyz(R_s)


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return quat_xyzw_to_mat(np.array([x, y, z, w]))


def sim_quat_to_unity(q_wxyz: np.ndarray) -> np.ndarray:
    """Inverse of unity_quat_to_sim: simulator [w,x,y,z] -> Unity [x,y,z,w]."""
    R_s = quat_wxyz_to_mat(np.asarray(q_wxyz, dtype=np.float64))
    R_u = M.T @ R_s @ M
    w, x, y, z = mat_to_quat_wxyz(R_u)
    return np.array([x, y, z, w])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def omega_from_quats(q_prev: np.ndarray, q_next: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity (rad/s) that takes q_prev to q_next in dt seconds (both [w,x,y,z])."""
    dq = quat_mul(quat_conj(q_prev), q_next)
    if dq[0] < 0:
        dq = -dq
    v = dq[1:]
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(n, dq[0])
    return v / n * angle / dt


def yaw_of(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
