"""Build 3DGS-style cameras from blender_render_multiview.py cameras.json frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gs_diff.graphics_utils import blender_c2w_to_3dgs_rt, focal2fov, getProjectionMatrix, getWorld2View2


@dataclass
class DiffCamera:
    image_name: str
    image_width: int
    image_height: int
    FoVx: float
    FoVy: float
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
    gt_image: torch.Tensor  # (3, H, W), [0, 1]


def _scale_intrinsics(
    fx: float, fy: float, cx: float, cy: float, frame_w: float, frame_h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    if frame_w and frame_h and (abs(frame_w - img_w) > 1 or abs(frame_h - img_h) > 1):
        sx = img_w / float(frame_w)
        sy = img_h / float(frame_h)
        return fx * sx, fy * sy, cx * sx, cy * sy
    return fx, fy, cx, cy


def load_camera_from_frame(
    frame: dict,
    styled_path: Path,
    *,
    device: torch.device,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> DiffCamera:
    if not styled_path.is_file():
        raise FileNotFoundError(f"缺少风格图: {styled_path}")

    pil = Image.open(styled_path).convert("RGB")
    img_w, img_h = pil.size
    img = torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0).permute(2, 0, 1).to(device)

    fx = float(frame["fx"])
    fy = float(frame["fy"])
    cx = float(frame["cx"])
    cy = float(frame["cy"])
    fw = float(frame.get("width", img_w))
    fh = float(frame.get("height", img_h))
    fx, fy, cx, cy = _scale_intrinsics(fx, fy, cx, cy, fw, fh, img_w, img_h)

    FoVx = focal2fov(fx, img_w)
    FoVy = focal2fov(fy, img_h)

    c2w = np.array(frame["transform_matrix_c2w"], dtype=np.float64)
    R, T = blender_c2w_to_3dgs_rt(c2w)

    world_view = torch.tensor(getWorld2View2(R, T), dtype=torch.float32, device=device).transpose(0, 1)
    proj = getProjectionMatrix(znear, zfar, FoVx, FoVy).transpose(0, 1).to(device)
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    camera_center = world_view.inverse()[3, :3]

    stem = styled_path.stem.replace("_styled", "")
    return DiffCamera(
        image_name=stem,
        image_width=img_w,
        image_height=img_h,
        FoVx=FoVx,
        FoVy=FoVy,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
        camera_center=camera_center,
        gt_image=img,
    )


def load_cameras_from_json(
    cameras_json: Path,
    styled_dir: Path,
    *,
    device: torch.device,
) -> list[DiffCamera]:
    import json

    with open(cameras_json, encoding="utf-8") as f:
        data = json.load(f)

    cameras: list[DiffCamera] = []
    for frame in data.get("frames", []):
        file_name = frame.get("file") or frame.get("name")
        if not file_name:
            continue
        stem = Path(file_name).stem
        styled_path = styled_dir / f"{stem}_styled.png"
        if not styled_path.is_file():
            print(f"skip camera: {styled_path.name} missing")
            continue
        cameras.append(load_camera_from_frame(frame, styled_path, device=device))
        print(f"  camera {stem}: {styled_path.name} ({cameras[-1].image_width}x{cameras[-1].image_height})")

    if not cameras:
        raise RuntimeError("没有可用的相机/风格图对，请检查 cameras.json 与 output/*_styled.png")
    return cameras
