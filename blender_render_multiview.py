"""
在 Blender 中运行（非普通 Python 环境）：
  blender --background --python blender_render_multiview.py

或在 Blender 文本编辑器中打开后运行。

功能：
  - 在球面上较均匀地布置多视角相机，环绕目标渲染 PNG 到 output_dir
  - 写出 cameras.json：每帧文件名、内参 fx/fy/cx/cy、4x4 的 c2w / w2c（与 Blender 坐标系一致）

相机布局（CAMERA_LAYOUT）：
  - fibonacci（默认）：仰角分层 + 黄金角方位，覆盖水平/斜上/俯视角，避免旧版 vertical 全挤在同一经线
  - multi_ring：多条等高环，每环均匀方位角
  - legacy：保留旧版 horizontal + vertical 两圈（兼容旧数据命名）

后续 Python 读取示例：
  import json
  data = json.load(open(".../cameras.json", encoding="utf-8"))
  for fr in data["frames"]:
      img = fr["file"]
      fx, fy = fr["fx"], fr["fy"]
      c2w = fr["transform_matrix_c2w"]  # 4x4 行优先

说明：若下游使用 COLMAP / OpenCV 相机约定，可能需要对 c2w 乘固定坐标变换矩阵，请查阅目标管线文档。
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import bpy
from mathutils import Matrix, Vector

# ====================== 配置 ======================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "rendered_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAMERAS_JSON = os.path.join(OUTPUT_DIR, "cameras.json")

bpy.context.scene.render.resolution_x = 800
bpy.context.scene.render.resolution_y = 800
bpy.context.scene.render.image_settings.file_format = "PNG"

# 相机布局：fibonacci | multi_ring | legacy
CAMERA_LAYOUT = "fibonacci"

# 环绕半径与注视点（模型大致中心）
RADIUS = 3.0
LOOK_AT = Vector((0.0, 0.0, 0.0))

# fibonacci：总视角数（建议 24~48；越多覆盖越全，但 style_transfer 越慢）
NUM_VIEWS = 16  # 方案 A：与 render_multiview_obj 一致
# 仰角范围（度，相对水平面 XY；Z 为上）
# 5°≈平视略抬，75°≈俯看；可略设负值拍到略低于水平的位置
MIN_ELEVATION_DEG = 5.0
MAX_ELEVATION_DEG = 75.0

# multi_ring：多条等高环（仅 CAMERA_LAYOUT="multi_ring" 时使用）
RING_ELEVATIONS_DEG = [8, 22, 36, 50, 64]
AZIMUTHS_PER_RING = 8
# 是否在最高环之上再加一张近顶视
ADD_TOP_VIEW = True
TOP_ELEVATION_DEG = 82.0

# legacy 旧参数（CAMERA_LAYOUT="legacy"）
LEGACY_RADIUS = 3.0
LEGACY_HEIGHT = 2.0
LEGACY_NUM_HORIZONTAL = 12
LEGACY_NUM_VERTICAL = 6

_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


def _render_scale() -> float:
    return bpy.context.scene.render.resolution_percentage / 100.0


def get_intrinsics_from_camera(camera_obj: bpy.types.Object) -> dict[str, Any]:
    """
    从 Blender 透视相机得到像素坐标系下的 fx, fy, cx, cy 与 3x3 矩阵 K。
    使用 Camera.angle_x / angle_y（与 Blender 内部投影一致），并考虑 shift。
    """
    cam = camera_obj.data
    scene = bpy.context.scene
    scale = _render_scale()
    w = scene.render.resolution_x * scale
    h = scene.render.resolution_y * scale

    if cam.type != "PERSP":
        raise RuntimeError("仅支持透视相机（PERSP）")

    angle_x = cam.angle_x
    angle_y = cam.angle_y
    fx = w / (2.0 * math.tan(angle_x * 0.5))
    fy = h / (2.0 * math.tan(angle_y * 0.5))

    cx = w * (0.5 + cam.shift_x)
    cy = h * (0.5 + cam.shift_y)

    K = [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ]
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": float(w),
        "height": float(h),
        "K": K,
        "angle_x_rad": float(angle_x),
        "angle_y_rad": float(angle_y),
        "focal_length_mm": float(cam.lens),
        "sensor_width_mm": float(cam.sensor_width),
        "sensor_height_mm": float(cam.sensor_height),
        "sensor_fit": str(cam.sensor_fit),
    }


def mat4_to_nested(m: Matrix) -> list[list[float]]:
    return [list(map(float, row)) for row in m]


def get_extrinsics_from_camera(camera_obj: bpy.types.Object) -> dict[str, Any]:
    mw = camera_obj.matrix_world.copy()
    inv = mw.inverted()
    return {
        "transform_matrix_c2w": mat4_to_nested(mw),
        "transform_matrix_w2c": mat4_to_nested(inv),
    }


def build_frame_record(
    rel_name: str,
    camera_obj: bpy.types.Object,
    intrinsics: dict[str, Any],
    *,
    layout_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ext = get_extrinsics_from_camera(camera_obj)
    rec = {
        "name": rel_name.replace(".png", ""),
        "file": rel_name,
        "relative_file_path": rel_name,
        **intrinsics,
        **ext,
    }
    if layout_meta:
        rec.update(layout_meta)
    return rec


def _location_from_spherical(radius: float, elevation_deg: float, azimuth_deg: float) -> Vector:
    """Blender Z-up：仰角为与水平面 XY 的夹角，方位角从 +X 逆时针朝 +Y。"""
    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)
    r_xy = radius * math.cos(elev)
    z = radius * math.sin(elev)
    x = r_xy * math.cos(azim)
    y = r_xy * math.sin(azim)
    return Vector((x, y, z))


def place_camera_look_at(cam: bpy.types.Object, location: Vector, target: Vector) -> None:
    cam.location = location
    direction = target - location
    if direction.length < 1e-8:
        raise ValueError("相机与注视点重合，无法朝向")
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def generate_fibonacci_views() -> list[dict[str, float]]:
    """
    在 [MIN_ELEVATION, MAX_ELEVATION] 内均匀分配仰角，
    方位角用黄金角错开，避免相邻视角挤在同一经线。
    """
    if NUM_VIEWS < 4:
        raise ValueError("NUM_VIEWS 至少为 4")

    views: list[dict[str, float]] = []
    elev_span = MAX_ELEVATION_DEG - MIN_ELEVATION_DEG
    for i in range(NUM_VIEWS):
        t = (i + 0.5) / NUM_VIEWS
        elevation_deg = MIN_ELEVATION_DEG + elev_span * t
        azimuth_deg = (360.0 * i / _GOLDEN_RATIO) % 360.0
        views.append(
            {
                "elevation_deg": float(elevation_deg),
                "azimuth_deg": float(azimuth_deg),
            }
        )
    return views


def generate_multi_ring_views() -> list[dict[str, float]]:
    views: list[dict[str, float]] = []
    for ring_idx, elevation_deg in enumerate(RING_ELEVATIONS_DEG):
        # 相邻环错开半格方位，减少视角共线
        azimuth_offset = (180.0 / AZIMUTHS_PER_RING) if ring_idx % 2 else 0.0
        for j in range(AZIMUTHS_PER_RING):
            azimuth_deg = (360.0 * j / AZIMUTHS_PER_RING + azimuth_offset) % 360.0
            views.append(
                {
                    "elevation_deg": float(elevation_deg),
                    "azimuth_deg": float(azimuth_deg),
                    "ring_index": float(ring_idx),
                }
            )
    if ADD_TOP_VIEW:
        views.append(
            {
                "elevation_deg": float(TOP_ELEVATION_DEG),
                "azimuth_deg": 0.0,
                "ring_index": float(len(RING_ELEVATIONS_DEG)),
                "top_view": 1.0,
            }
        )
    return views


def generate_legacy_views() -> list[tuple[str, Vector]]:
    """旧版：水平一圈 + 垂直半圈（vertical 均在 x 轴经线上，覆盖不均衡）。"""
    specs: list[tuple[str, Vector]] = []
    for i in range(LEGACY_NUM_HORIZONTAL):
        angle = 2 * math.pi * i / LEGACY_NUM_HORIZONTAL
        loc = Vector(
            (
                LEGACY_RADIUS * math.cos(angle),
                LEGACY_RADIUS * math.sin(angle),
                LEGACY_HEIGHT,
            )
        )
        specs.append((f"horizontal_{i:03d}.png", loc))

    for i in range(LEGACY_NUM_VERTICAL):
        t = i / (LEGACY_NUM_VERTICAL - 1) if LEGACY_NUM_VERTICAL > 1 else 0.0
        vert_angle = 2 * math.pi * t
        loc = Vector(
            (
                LEGACY_RADIUS * math.sin(vert_angle),
                0.0,
                LEGACY_RADIUS * math.cos(vert_angle),
            )
        )
        specs.append((f"vertical_{i:03d}.png", loc))
    return specs


def render_view(
    scene: bpy.types.Scene,
    cam: bpy.types.Object,
    rel_name: str,
    frames: list[dict[str, Any]],
    *,
    layout_meta: dict[str, Any] | None = None,
) -> None:
    filepath = os.path.join(OUTPUT_DIR, rel_name)
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    intr = get_intrinsics_from_camera(cam)
    frames.append(build_frame_record(rel_name, cam, intr, layout_meta=layout_meta))


def main() -> None:
    scene = bpy.context.scene
    scene.render.resolution_percentage = 100

    cam = scene.camera
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        scene.camera = cam

    bpy.context.view_layer.update()
    frames: list[dict[str, Any]] = []

    layout = CAMERA_LAYOUT.strip().lower()

    if layout == "fibonacci":
        view_specs = generate_fibonacci_views()
        print(f"layout=fibonacci, views={len(view_specs)}, elev=[{MIN_ELEVATION_DEG}, {MAX_ELEVATION_DEG}] deg")
        for i, spec in enumerate(view_specs):
            loc = _location_from_spherical(RADIUS, spec["elevation_deg"], spec["azimuth_deg"])
            place_camera_look_at(cam, loc, LOOK_AT)
            bpy.context.view_layer.update()
            rel_name = f"view_{i:03d}.png"
            meta = {
                "layout": "fibonacci",
                "view_index": float(i),
                **spec,
            }
            render_view(scene, cam, rel_name, frames, layout_meta=meta)

    elif layout == "multi_ring":
        view_specs = generate_multi_ring_views()
        print(
            f"layout=multi_ring, views={len(view_specs)}, "
            f"rings={RING_ELEVATIONS_DEG}, azimuths_per_ring={AZIMUTHS_PER_RING}"
        )
        for i, spec in enumerate(view_specs):
            loc = _location_from_spherical(RADIUS, spec["elevation_deg"], spec["azimuth_deg"])
            place_camera_look_at(cam, loc, LOOK_AT)
            bpy.context.view_layer.update()
            rel_name = f"view_{i:03d}.png"
            meta = {
                "layout": "multi_ring",
                "view_index": float(i),
                **spec,
            }
            render_view(scene, cam, rel_name, frames, layout_meta=meta)

    elif layout == "legacy":
        legacy_specs = generate_legacy_views()
        print(f"layout=legacy, views={len(legacy_specs)}")
        for rel_name, loc in legacy_specs:
            place_camera_look_at(cam, loc, LOOK_AT)
            bpy.context.view_layer.update()
            render_view(
                scene,
                cam,
                rel_name,
                frames,
                layout_meta={"layout": "legacy"},
            )
    else:
        raise ValueError(f"未知 CAMERA_LAYOUT: {CAMERA_LAYOUT}")

    out_data: dict[str, Any] = {
        "format_version": 2,
        "coordinate_system": "blender_camera_matrix_world_as_c2w",
        "output_dir": OUTPUT_DIR.replace("\\", "/"),
        "camera_layout": layout,
        "camera_layout_config": {
            "radius": RADIUS,
            "look_at": [float(LOOK_AT.x), float(LOOK_AT.y), float(LOOK_AT.z)],
            "num_views": NUM_VIEWS if layout == "fibonacci" else None,
            "min_elevation_deg": MIN_ELEVATION_DEG if layout == "fibonacci" else None,
            "max_elevation_deg": MAX_ELEVATION_DEG if layout == "fibonacci" else None,
            "ring_elevations_deg": RING_ELEVATIONS_DEG if layout == "multi_ring" else None,
            "azimuths_per_ring": AZIMUTHS_PER_RING if layout == "multi_ring" else None,
        },
        "global_render": {
            "resolution_x": scene.render.resolution_x,
            "resolution_y": scene.render.resolution_y,
            "file_format": scene.render.image_settings.file_format,
        },
        "shared_intrinsics_note": "若分辨率与镜头不变，每帧 fx/fy/cx/cy 应一致；已按帧写入 frames 便于单帧使用。",
        "frames": frames,
    }

    with open(CAMERAS_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"cameras.json: {CAMERAS_JSON} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
