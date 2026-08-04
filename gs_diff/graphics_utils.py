"""Minimal graphics helpers (adapted from graphdeco-inria/gaussian-splatting)."""

from __future__ import annotations

import math

import numpy as np
import torch


def getWorld2View(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.zeros((4, 4), dtype=np.float64)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def getWorld2View2(
    R: np.ndarray,
    t: np.ndarray,
    translate: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    if translate is None:
        translate = np.zeros(3, dtype=np.float64)
    Rt = np.zeros((4, 4), dtype=np.float64)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return Rt


def getProjectionMatrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
    tan_half_fov_y = math.tan(float(fovY) * 0.5)
    tan_half_fov_x = math.tan(float(fovX) * 0.5)

    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def focal2fov(focal: float, pixels: float) -> float:
    return 2.0 * math.atan(pixels / (2.0 * focal))


def blender_c2w_to_3dgs_rt(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Blender/OpenGL camera-to-world → 3DGS (COLMAP-style) R, T for getWorld2View2.

    Blender: +X right, +Y up, -Z forward.
    COLMAP / 3DGS: +X right, +Y down, +Z forward.
    """
    c2w = np.array(c2w, dtype=np.float64).reshape(4, 4).copy()
    c2w[:3, 1:3] *= -1.0
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]
    return R, T
