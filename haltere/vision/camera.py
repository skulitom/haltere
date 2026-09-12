"""Pinhole model of Liftoff's FPV camera and its relation to the drone's body frame.

Frames: the simulator's body frame is FLU (x forward, y left, z up). The camera frame is the usual
computer-vision one (x right, y down, z forward), tilted up by ``tilt_deg`` about the body's y axis
(FPV camera uptilt, 30 degrees for the drone used here, from its .drone configuration). The image
principal point is the image centre; the focal length is calibrated from flight data
(``haltere vision calibrate``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def body_to_cam_matrix(tilt_deg: float) -> np.ndarray:
    """Rotation taking body (FLU) vectors to camera (right, down, forward) vectors."""
    t = np.deg2rad(tilt_deg)
    # camera axes expressed in the body frame: forward tilted up, right = -left, down = -up (tilted)
    fwd = np.array([np.cos(t), 0.0, np.sin(t)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(fwd, right)          # right-handed: y_c = z_c x x_c
    return np.stack([right, down, fwd])  # rows = camera axes in body coordinates -> R_cb @ v_b = v_c


@dataclass
class Camera:
    width: int = 640
    height: int = 360
    f: float = 300.0              # focal length in pixels at this resolution
    tilt_deg: float = 30.0

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / (2 * self.f))))

    def body_to_cam(self) -> np.ndarray:
        return body_to_cam_matrix(self.tilt_deg)

    def project_body(self, v_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Body-frame points (N, 3) -> pixel coordinates (N, 2) and a mask of points in front of the camera."""
        v_c = np.asarray(v_b, dtype=np.float64) @ self.body_to_cam().T
        z = v_c[:, 2]
        ok = z > 0.05
        zs = np.where(ok, z, 1.0)
        px = np.stack([self.cx + self.f * v_c[:, 0] / zs, self.cy + self.f * v_c[:, 1] / zs], axis=1)
        return px, ok

    def unproject_body(self, px: np.ndarray) -> np.ndarray:
        """Pixel coordinates (N, 2) -> unit direction vectors in the body frame (N, 3)."""
        px = np.asarray(px, dtype=np.float64)
        d_c = np.stack([(px[:, 0] - self.cx) / self.f, (px[:, 1] - self.cy) / self.f, np.ones(len(px))], axis=1)
        d_c /= np.linalg.norm(d_c, axis=1, keepdims=True)
        return d_c @ self.body_to_cam()

    def scaled(self, width: int, height: int) -> "Camera":
        return Camera(width, height, self.f * width / self.width, self.tilt_deg)


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def world_to_body(points_w: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """World points (N, 3) -> body frame of a drone at pos with attitude quat (world-from-body)."""
    R = quat_wxyz_to_mat(np.asarray(quat_wxyz, dtype=np.float64))
    return (np.asarray(points_w, dtype=np.float64) - np.asarray(pos, dtype=np.float64)) @ R
