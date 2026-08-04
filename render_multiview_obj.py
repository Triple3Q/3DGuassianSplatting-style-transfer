"""
render_multiview_obj.py - 无需 Blender，从 OBJ 网格直接渲染多视角 PNG + cameras.json

用法：
  python render_multiview_obj.py head/head.obj
  python render_multiview_obj.py head/head.obj --texture head/texture.png --texture-bleed 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrender
import trimesh
from PIL import Image

from multiview_layout import (
    generate_fibonacci_views,
    generate_multi_ring_views,
    intrinsics_from_vertical_fov,
    location_from_spherical,
    look_at_blender_c2w,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "rendered_images"

DEFAULT_LAYOUT = "fibonacci"
DEFAULT_NUM_VIEWS = 16  # 方案 A：16 视角，风格迁移约减半耗时
DEFAULT_MIN_ELEVATION_DEG = 5.0
DEFAULT_MAX_ELEVATION_DEG = 75.0
DEFAULT_RADIUS = 3.0
DEFAULT_RESOLUTION = 800
DEFAULT_FOVY_DEG = 50.0
DEFAULT_TEXTURE_BLEED = 8
DEFAULT_RING_ELEVATIONS = [8, 22, 36, 50, 64]
DEFAULT_AZIMUTHS_PER_RING = 8

_TEXTURE_CANDIDATES = (
    "texture.png",
    "Texture.png",
    "diffuse.png",
    "albedo.png",
    "material.png",
)


def _parse_mtl_texture(mtl_path: Path) -> Path | None:
    if not mtl_path.is_file():
        return None
    for line in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.lower().startswith("map_kd"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                tex = parts[1].strip().strip('"').strip("'")
                candidate = (mtl_path.parent / tex).resolve()
                if candidate.is_file():
                    return candidate
    return None


def _parse_obj_mtllib(obj_path: Path) -> Path | None:
    for line in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.lower().startswith("mtllib"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                candidate = (obj_path.parent / parts[1].strip()).resolve()
                if candidate.is_file():
                    return candidate
    return None


def resolve_texture_path(obj_path: Path, texture_arg: Path | None) -> Path | None:
    if texture_arg is not None:
        tex = texture_arg.resolve()
        if not tex.is_file():
            raise FileNotFoundError(f"找不到纹理: {tex}")
        return tex

    obj_dir = obj_path.parent
    for name in _TEXTURE_CANDIDATES:
        candidate = obj_dir / name
        if candidate.is_file():
            return candidate.resolve()

    mtl_path = _parse_obj_mtllib(obj_path)
    if mtl_path is not None:
        tex = _parse_mtl_texture(mtl_path)
        if tex is not None:
            return tex

    for mtl in obj_dir.glob("*.mtl"):
        tex = _parse_mtl_texture(mtl)
        if tex is not None:
            return tex
    return None


def _load_trimesh(obj_path: Path) -> trimesh.Trimesh:
    """
    加载 OBJ 并在 UV 接缝处展开顶点。

    带贴图的 OBJ 在接缝处「一个 3D 点对应多个 UV」；若 process=False，
    trimesh 只保留 25 万顶点各 1 套 UV，而实际有 25.4 万套，渲染会出现大面积白色裂缝。
    """
    # 勿用 maintain_order=True：会阻止 UV 接缝处顶点展开，导致大面积白色裂缝
    loaded = trimesh.load(str(obj_path), force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"OBJ 场景中没有三角网格: {obj_path}")
        loaded = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    if not isinstance(loaded, trimesh.Trimesh) or loaded.vertices.shape[0] == 0:
        raise ValueError(f"无法读取或网格为空: {obj_path}")
    return loaded


def prepare_mesh_for_render(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray, float]:
    """居中归一化 + 平滑法线；返回 (mesh, center, scale) 供 cameras.json 对齐 PLY。"""
    m = mesh.copy()
    bounds = m.bounds
    center = (bounds[0] + bounds[1]) * 0.5
    scale = float(np.max(bounds[1] - bounds[0]))
    m.vertices -= center
    if scale > 1e-8:
        m.vertices /= scale

    m.fix_normals()
    _ = m.vertex_normals
    return m, np.asarray(center, dtype=np.float64), scale


def _get_mesh_uv(mesh: trimesh.Trimesh) -> np.ndarray | None:
    visual = mesh.visual
    if isinstance(visual, trimesh.visual.TextureVisuals) and visual.uv is not None:
        return np.asarray(visual.uv, dtype=np.float64)
    return None


def _trimesh_has_uv(mesh: trimesh.Trimesh) -> bool:
    uv = _get_mesh_uv(mesh)
    return uv is not None and len(uv) > 0


def rasterize_uv_mask(mesh: trimesh.Trimesh, tex_w: int, tex_h: int, *, flip_v: bool) -> np.ndarray:
    """在贴图像素空间标记 UV 覆盖区域。"""
    mask = np.zeros((tex_h, tex_w), dtype=np.uint8)
    uvs = _get_mesh_uv(mesh)
    if uvs is None:
        return mask

    for tri in mesh.faces:
        pts = uvs[tri].copy()
        xs = np.clip((pts[:, 0] * (tex_w - 1)).round(), 0, tex_w - 1)
        if flip_v:
            ys = np.clip((1.0 - pts[:, 1]) * (tex_h - 1), 0, tex_h - 1)
        else:
            ys = np.clip((pts[:, 1] * (tex_h - 1)), 0, tex_h - 1)
        poly = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.fillConvexPoly(mask, poly, 255)
    return mask


def bleed_texture_for_uv_seams(
    tex_rgb: np.ndarray,
    uv_mask: np.ndarray,
    *,
    bleed_px: int,
) -> np.ndarray:
    """
    将 UV 岛边缘颜色向外扩展，避免采样到贴图空白区产生「裂纹」黑线。
    Blender 导出/预览时通常自带 margin，pyrender 双线性过滤更敏感。
    """
    if bleed_px <= 0 or uv_mask.max() == 0:
        return tex_rgb

    out = tex_rgb.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current_mask = (uv_mask > 0).astype(np.uint8)

    for _ in range(bleed_px):
        dilated = cv2.dilate(current_mask, kernel)
        expand = (dilated > 0) & (current_mask == 0)
        if not np.any(expand):
            break
        for c in range(3):
            src = out[:, :, c]
            dilated_color = cv2.dilate(src, kernel)
            out[:, :, c] = np.where(expand, dilated_color, src)
        current_mask = dilated

    return out


def load_texture_image(
    texture_path: Path,
    mesh: trimesh.Trimesh,
    *,
    bleed_px: int,
    flip_v: bool,
) -> np.ndarray:
    tex = np.array(Image.open(texture_path).convert("RGB"))
    if not _trimesh_has_uv(mesh):
        return tex

    h, w = tex.shape[:2]
    uv_mask = rasterize_uv_mask(mesh, w, h, flip_v=flip_v)
    covered = float((uv_mask > 0).mean()) * 100.0
    print(f"  uv coverage: {covered:.1f}% of texture, bleed={bleed_px}px")
    return bleed_texture_for_uv_seams(tex, uv_mask, bleed_px=bleed_px)


def build_pyrender_mesh(
    mesh: trimesh.Trimesh,
    tex_rgb: np.ndarray | None,
) -> pyrender.Mesh:
    if tex_rgb is not None:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=tex_rgb,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            doubleSided=True,
        )
    else:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.78, 0.78, 0.78, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
            doubleSided=True,
        )
    # smooth=True：使用顶点法线，避免硬边造成的假「裂纹」
    return pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)


class PyrenderViewRenderer:
    def __init__(self, pr_mesh: pyrender.Mesh, *, width: int, height: int, fovy_deg: float) -> None:
        self.width = width
        self.height = height
        self.fovy = math.radians(fovy_deg)
        # 纯环境光，减少明暗造成的接缝感
        self.scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[1.0, 1.0, 1.0])
        self.scene.add(pr_mesh)
        self._renderer = pyrender.OffscreenRenderer(width, height)

    def render(self, c2w: list[list[float]]) -> np.ndarray:
        pose = np.asarray(c2w, dtype=np.float64)
        camera = pyrender.PerspectiveCamera(
            yfov=self.fovy,
            aspectRatio=self.width / self.height,
            znear=0.01,
            zfar=100.0,
        )
        cam_node = self.scene.add(camera, pose=pose)
        try:
            color, _ = self._renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            self.scene.remove_node(cam_node)
        return color[:, :, :3]

    def close(self) -> None:
        self._renderer.delete()


def build_frame_record(
    rel_name: str,
    intrinsics: dict[str, Any],
    c2w: list[list[float]],
    *,
    layout_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64)).tolist()
    rec: dict[str, Any] = {
        "name": rel_name.replace(".png", ""),
        "file": rel_name,
        "relative_file_path": rel_name,
        **intrinsics,
        "transform_matrix_c2w": c2w,
        "transform_matrix_w2c": w2c,
    }
    if layout_meta:
        rec.update(layout_meta)
    return rec


def collect_view_specs(args: argparse.Namespace) -> list[dict[str, float]]:
    layout = args.layout.strip().lower()
    if layout == "fibonacci":
        return generate_fibonacci_views(args.num_views, args.min_elevation, args.max_elevation)
    if layout == "multi_ring":
        elevations = [float(x) for x in args.ring_elevations.split(",") if x.strip()]
        return generate_multi_ring_views(
            elevations,
            args.azimuths_per_ring,
            add_top_view=not args.no_top_view,
            top_elevation_deg=args.top_elevation,
        )
    raise ValueError(f"未知 layout: {args.layout}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 OBJ 渲染多视角 PNG + cameras.json（pyrender，支持 UV 贴图）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("obj", type=Path)
    ap.add_argument("--texture", type=Path, default=None)
    ap.add_argument("--no-texture", action="store_true")
    ap.add_argument("--texture-bleed", type=int, default=DEFAULT_TEXTURE_BLEED,
                    help="UV 接缝贴图扩展像素，消除裂纹黑线（0=关闭）")
    ap.add_argument("--flip-v", action="store_true", help="翻转 V 坐标（若贴图整体上下颠倒时用）")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT, choices=["fibonacci", "multi_ring"])
    ap.add_argument("--num-views", type=int, default=DEFAULT_NUM_VIEWS)
    ap.add_argument("--min-elevation", type=float, default=DEFAULT_MIN_ELEVATION_DEG)
    ap.add_argument("--max-elevation", type=float, default=DEFAULT_MAX_ELEVATION_DEG)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    ap.add_argument("--fovy-deg", type=float, default=DEFAULT_FOVY_DEG)
    ap.add_argument("--ring-elevations", type=str, default=",".join(map(str, DEFAULT_RING_ELEVATIONS)))
    ap.add_argument("--azimuths-per-ring", type=int, default=DEFAULT_AZIMUTHS_PER_RING)
    ap.add_argument("--no-top-view", action="store_true")
    ap.add_argument("--top-elevation", type=float, default=82.0)
    args = ap.parse_args()

    obj_path = args.obj.resolve()
    if not obj_path.is_file():
        print(f"错误: 找不到 OBJ 文件 {obj_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"load: {obj_path}")
    tm_raw = _load_trimesh(obj_path)
    tm, norm_center, norm_scale = prepare_mesh_for_render(tm_raw)
    print(f"  mesh: {len(tm.vertices)} verts, {len(tm.faces)} faces, smooth normals, UV seams expanded")
    print(f"  normalization: center={norm_center.round(4).tolist()}, scale={norm_scale:.6f}")

    texture_path: Path | None = None
    if not args.no_texture:
        try:
            texture_path = resolve_texture_path(obj_path, args.texture)
        except FileNotFoundError as e:
            print(f"错误: {e}", file=sys.stderr)
            return 1

    has_uv = _trimesh_has_uv(tm)
    tex_rgb: np.ndarray | None = None
    if texture_path is not None:
        print(f"  texture: {texture_path}")
        if not has_uv:
            print("  警告: OBJ 缺少 UV", file=sys.stderr)
        else:
            tex_rgb = load_texture_image(
                texture_path, tm, bleed_px=args.texture_bleed, flip_v=args.flip_v
            )
    else:
        print("  texture: (none)")

    pr_mesh = build_pyrender_mesh(tm, tex_rgb)

    look_at = (0.0, 0.0, 0.0)
    view_specs = collect_view_specs(args)
    width = height = int(args.resolution)
    intr_template = intrinsics_from_vertical_fov(width, height, args.fovy_deg)

    print(
        f"render: pyrender, views={len(view_specs)}, res={width}, "
        f"textured={tex_rgb is not None}"
    )

    frames: list[dict[str, Any]] = []
    renderer = PyrenderViewRenderer(pr_mesh, width=width, height=height, fovy_deg=args.fovy_deg)
    try:
        for i, spec in enumerate(view_specs):
            eye = location_from_spherical(args.radius, spec["elevation_deg"], spec["azimuth_deg"])
            c2w = look_at_blender_c2w(eye, look_at)
            rel_name = f"view_{i:03d}.png"
            rgb = renderer.render(c2w)
            Image.fromarray(rgb.astype(np.uint8)).save(output_dir / rel_name)
            frames.append(
                build_frame_record(
                    rel_name,
                    intr_template,
                    c2w,
                    layout_meta={"layout": args.layout, "view_index": float(i), "renderer": "pyrender", **spec},
                )
            )
            print(f"  [{i + 1}/{len(view_specs)}] {rel_name}")
    finally:
        renderer.close()

    cameras_json = output_dir / "cameras.json"
    with open(cameras_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "format_version": 2,
                "coordinate_system": "blender_camera_matrix_world_as_c2w",
                "renderer": "render_multiview_obj.py (pyrender)",
                "source_mesh": str(obj_path).replace("\\", "/"),
                "source_texture": str(texture_path).replace("\\", "/") if texture_path else None,
                "texture_bleed_px": args.texture_bleed,
                "mesh_normalization": {
                    "center": norm_center.tolist(),
                    "scale": norm_scale,
                    "formula": "xyz_cam = (xyz_ply - center) / scale",
                },
                "background_color": [1.0, 1.0, 1.0],
                "output_dir": str(output_dir).replace("\\", "/"),
                "frames": frames,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"cameras.json: {cameras_json} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
