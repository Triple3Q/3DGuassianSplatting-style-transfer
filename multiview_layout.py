"""Shared multi-view camera layouts (Blender Z-up, azimuth from +X toward +Y)."""

from __future__ import annotations

import math
from typing import Any

_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


def location_from_spherical(radius: float, elevation_deg: float, azimuth_deg: float) -> tuple[float, float, float]:
    """仰角为与水平面 XY 的夹角，方位角从 +X 逆时针朝 +Y。"""
    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)
    r_xy = radius * math.cos(elev)
    z = radius * math.sin(elev)
    x = r_xy * math.cos(azim)
    y = r_xy * math.sin(azim)
    return float(x), float(y), float(z)


def generate_fibonacci_views(
    num_views: int,
    min_elevation_deg: float,
    max_elevation_deg: float,
) -> list[dict[str, float]]:
    if num_views < 4:
        raise ValueError("num_views 至少为 4")

    views: list[dict[str, float]] = []
    elev_span = max_elevation_deg - min_elevation_deg
    for i in range(num_views):
        t = (i + 0.5) / num_views
        elevation_deg = min_elevation_deg + elev_span * t
        azimuth_deg = (360.0 * i / _GOLDEN_RATIO) % 360.0
        views.append(
            {
                "elevation_deg": float(elevation_deg),
                "azimuth_deg": float(azimuth_deg),
            }
        )
    return views


def generate_multi_ring_views(
    ring_elevations_deg: list[float],
    azimuths_per_ring: int,
    *,
    add_top_view: bool = True,
    top_elevation_deg: float = 82.0,
) -> list[dict[str, float]]:
    views: list[dict[str, float]] = []
    for ring_idx, elevation_deg in enumerate(ring_elevations_deg):
        azimuth_offset = (180.0 / azimuths_per_ring) if ring_idx % 2 else 0.0
        for j in range(azimuths_per_ring):
            azimuth_deg = (360.0 * j / azimuths_per_ring + azimuth_offset) % 360.0
            views.append(
                {
                    "elevation_deg": float(elevation_deg),
                    "azimuth_deg": float(azimuth_deg),
                    "ring_index": float(ring_idx),
                }
            )
    if add_top_view:
        views.append(
            {
                "elevation_deg": float(top_elevation_deg),
                "azimuth_deg": 0.0,
                "ring_index": float(len(ring_elevations_deg)),
                "top_view": 1.0,
            }
        )
    return views


def look_at_blender_c2w(
    eye: tuple[float, float, float] | list[float],
    target: tuple[float, float, float] | list[float],
    up_hint: tuple[float, float, float] | list[float] = (0.0, 0.0, 1.0),
) -> list[list[float]]:
    """
    Build 4x4 camera-to-world in Blender convention (+X right, +Y up, -Z forward).
    """
    import numpy as np

    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_v = np.asarray(up_hint, dtype=np.float64)

    forward = target_v - eye_v
    n = float(np.linalg.norm(forward))
    if n < 1e-8:
        raise ValueError("相机与注视点重合")
    forward /= n

    z_axis = -forward
    x_axis = np.cross(up_v, z_axis)
    if float(np.linalg.norm(x_axis)) < 1e-8:
        up_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(up_v, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = eye_v
    return c2w.tolist()


def intrinsics_from_vertical_fov(width: int, height: int, fovy_deg: float) -> dict[str, Any]:
    fovy = math.radians(fovy_deg)
    fy = height / (2.0 * math.tan(fovy / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": float(width),
        "height": float(height),
        "K": [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        "angle_y_rad": float(fovy),
        "angle_x_rad": float(2.0 * math.atan(width / (2.0 * fx))),
    }
