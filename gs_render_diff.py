"""
gs_render_diff.py - 使用原版 3DGS 可微光栅化，多视角联合优化颜色

流程：
  1. 调用 diff_gaussian_rasterization 将 3D 高斯渲染成 2D 图像
  2. 与每张 output/*_styled.png 计算图像级 L1 loss
  3. 仅反向更新 f_dc（颜色），几何（位置/尺度/旋转/透明度）固定

用法：
  python gs_render_diff.py INPUT.ply OUTPUT.ply
  python gs_render_diff.py -i head.ply -o head_diff.ply --iterations 3000 --lr 0.0025
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import numpy as np

from gs_diff.cameras import load_cameras_from_json
from gs_diff.gaussian_data import GaussianData
from gs_diff.renderer import RenderPipe, render

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CAMERAS_JSON = PROJECT_ROOT / "rendered_images" / "cameras.json"
DEFAULT_STYLED_DIR = PROJECT_ROOT / "output"
DEFAULT_PLY_INPUT = PROJECT_ROOT / "3DGS_output" / "head.ply"


def _normalization_from_obj(obj_path: Path) -> dict | None:
    try:
        import trimesh
    except ImportError:
        return None
    if not obj_path.is_file():
        return None
    m = trimesh.load(str(obj_path), force="mesh", process=True)
    if isinstance(m, trimesh.Scene):
        geoms = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        m = trimesh.util.concatenate(geoms) if geoms else None
    if m is None or m.vertices.shape[0] == 0:
        return None
    bounds = m.bounds
    center = ((bounds[0] + bounds[1]) * 0.5).astype(np.float64)
    scale = float(np.max(bounds[1] - bounds[0]))
    return {"center": center.tolist(), "scale": scale}


def _normalization_from_ply_xyz(xyz: np.ndarray) -> dict:
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    center = ((mn + mx) * 0.5).astype(np.float64)
    scale = float((mx - mn).max())
    return {"center": center.tolist(), "scale": scale}


def _load_cameras_json_meta(cameras_json: Path) -> dict:
    with open(cameras_json, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "mesh_normalization": data.get("mesh_normalization"),
        "background_color": data.get("background_color"),
        "source_mesh": data.get("source_mesh"),
    }


def _resolve_normalization(
    gaussians: GaussianData,
    meta: dict,
    *,
    mode: str,
    obj_path: Path | None,
) -> None:
    if mode == "none":
        print("  mesh normalization: disabled")
        return

    if mode == "obj":
        norm = meta.get("mesh_normalization")
        if (not norm or not norm.get("scale")) and obj_path and obj_path.is_file():
            norm = _normalization_from_obj(obj_path)
        if (not norm or not norm.get("scale")) and meta.get("source_mesh"):
            for candidate in (
                Path(meta["source_mesh"]),
                PROJECT_ROOT / meta["source_mesh"],
                PROJECT_ROOT / "head" / "head.obj",
            ):
                if candidate.is_file():
                    norm = _normalization_from_obj(candidate)
                    if norm:
                        break
        if not norm or not norm.get("scale"):
            raise RuntimeError("normalize-from=obj 需要 cameras.json 的 mesh_normalization 或 --obj-path")
        center = np.asarray(norm["center"], dtype=np.float32)
        scale = float(norm["scale"])
        src_label = "obj / cameras.json"
    else:
        # 默认 ply：用 PLY 自身包围盒，与「各自居中归一化」的 OBJ 渲染对齐
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        norm = _normalization_from_ply_xyz(xyz)
        center = np.asarray(norm["center"], dtype=np.float32)
        scale = float(norm["scale"])
        src_label = "ply bbox"

    gaussians.apply_mesh_normalization(center, scale)
    print(f"  mesh normalization ({src_label}): center={center.round(4).tolist()}, scale={scale:.6f}")


def _masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, *, bg_threshold: float) -> torch.Tensor:
    """仅在非背景像素上计算 L1（背景≈全白时避免 loss 虚高）。"""
    if bg_threshold <= 0:
        return F.l1_loss(pred, target)
    # target: (3,H,W) -> 任一带高于阈值视为背景
    fg = (target.max(dim=0).values < bg_threshold).float()
    if fg.sum() < 1:
        return F.l1_loss(pred, target)
    diff = (pred - target).abs().mean(dim=0)
    return (diff * fg).sum() / fg.sum()


def run_diff_optimize(
    ply_path: Path,
    cameras_json: Path,
    styled_dir: Path,
    output_ply: Path | None = None,
    *,
    iterations: int = 3000,
    lr: float = 0.0025,
    reg_weight: float = 0.01,
    lambda_dssim: float = 0.0,
    bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    bg_mask_threshold: float = 0.95,
    normalize_from: str = "ply",
    obj_path: Path | None = None,
    seed: int = 42,
    log_every: int = 100,
    device: str | None = None,
) -> Path:
    if output_ply is None:
        output_ply = ply_path.parent / "gs_styled_diff.ply"

    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if dev.type != "cuda":
        raise RuntimeError(
            "diff_gaussian_rasterization 需要 CUDA GPU。"
            "请在有 NVIDIA GPU 的环境中运行 gs_render_diff.py。"
        )

    random.seed(seed)
    torch.manual_seed(seed)

    print(f"device: {dev}")
    meta = _load_cameras_json_meta(cameras_json)
    print("load cameras...")
    cameras = load_cameras_from_json(cameras_json, styled_dir, device=dev)

    print(f"load ply: {ply_path}")
    gaussians = GaussianData()
    gaussians.load_ply(ply_path, dev)
    _resolve_normalization(gaussians, meta, mode=normalize_from, obj_path=obj_path)

    gaussians.freeze_geometry()
    f_dc_orig = gaussians.get_features_dc.detach().clone()
    print(f"  points: {gaussians.get_xyz.shape[0]}, sh_degree: {gaussians.active_sh_degree}")

    optimizer = torch.optim.Adam([gaussians.get_features_dc], lr=lr)
    pipe = RenderPipe()
    bg_rgb = meta.get("background_color") or list(bg_color)
    bg = torch.tensor(bg_rgb, dtype=torch.float32, device=dev)
    print(f"  bg={bg_rgb}, fg_mask_threshold={bg_mask_threshold}")

    print(f"optimize: {iterations} iters, lr={lr}, views={len(cameras)}")
    for it in range(1, iterations + 1):
        cam = cameras[random.randint(0, len(cameras) - 1)]
        optimizer.zero_grad(set_to_none=True)

        pkg = render(cam, gaussians, pipe, bg)
        image = pkg["render"]

        loss_l1 = _masked_l1_loss(image, cam.gt_image, bg_threshold=bg_mask_threshold)
        if lambda_dssim > 0:
            # 简易 SSIM 替代：用 MSE 混合，避免额外依赖
            loss = (1.0 - lambda_dssim) * loss_l1 + lambda_dssim * F.mse_loss(image, cam.gt_image)
        else:
            loss = loss_l1

        if reg_weight > 0:
            loss = loss + reg_weight * F.mse_loss(gaussians.get_features_dc, f_dc_orig)

        loss.backward()
        optimizer.step()

        if it == 1 or it % log_every == 0 or it == iterations:
            print(
                f"iter {it:5d}/{iterations}  loss={float(loss.item()):.6f}  "
                f"l1={float(loss_l1.item()):.6f}  view={cam.image_name}"
            )

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(output_ply)
    print(f"wrote {output_ply.resolve()}")
    print(f'  viz: python visualized_gs.py "{output_ply}"')
    return output_ply


def main() -> None:
    p = argparse.ArgumentParser(
        description="可微 3DGS 光栅化 + 多视角风格图联合优化（仅 f_dc）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="示例:  python gs_render_diff.py head.ply head_diff.ply --iterations 5000",
    )
    p.add_argument("input_ply", nargs="?", type=Path, default=None)
    p.add_argument("output_ply", nargs="?", type=Path, default=None)
    p.add_argument("--ply", "-i", type=Path, default=DEFAULT_PLY_INPUT)
    p.add_argument("--output", "-o", type=Path, default=None)
    p.add_argument("--cameras-json", type=Path, default=DEFAULT_CAMERAS_JSON)
    p.add_argument("--styled-dir", type=Path, default=DEFAULT_STYLED_DIR)
    p.add_argument("--iterations", type=int, default=3000, help="优化迭代次数")
    p.add_argument("--lr", type=float, default=0.0025, help="f_dc 学习率（与 3DGS feature_lr 同量级）")
    p.add_argument("--reg-weight", type=float, default=0.01, help="贴近原始 f_dc 的正则")
    p.add_argument("--bg-threshold", type=float, default=0.95, help="GT 像素 max>此值视为背景，不参与 loss（0=整图）")
    p.add_argument(
        "--normalize-from",
        choices=("ply", "obj", "none"),
        default="ply",
        help="与渲染图对齐：ply=用 PLY 自身包围盒（默认，配 render_multiview_obj）；obj=用 OBJ 归一化；none=不变换",
    )
    p.add_argument("--obj-path", type=Path, default=None, help="normalize-from=obj 时的 OBJ 路径")
    p.add_argument("--lambda-dssim", type=float, default=0.0, help="混合 MSE 权重（0=纯 L1）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    ply_in = args.input_ply if args.input_ply is not None else args.ply
    ply_out = args.output_ply if args.output_ply is not None else args.output

    try:
        run_diff_optimize(
            ply_path=ply_in.resolve(),
            cameras_json=args.cameras_json.resolve(),
            styled_dir=args.styled_dir.resolve(),
            output_ply=ply_out.resolve() if ply_out is not None else None,
            iterations=args.iterations,
            lr=args.lr,
            reg_weight=args.reg_weight,
            lambda_dssim=args.lambda_dssim,
            bg_mask_threshold=args.bg_threshold,
            normalize_from=args.normalize_from,
            obj_path=args.obj_path.resolve() if args.obj_path else None,
            seed=args.seed,
            log_every=args.log_every,
            device=args.device,
        )
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
