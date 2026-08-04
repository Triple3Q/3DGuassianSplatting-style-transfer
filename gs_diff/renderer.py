""" Differentiable render using graphdeco-inria/diff-gaussian-rasterization. """

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gs_diff.cameras import DiffCamera
from gs_diff.gaussian_data import GaussianData


@dataclass
class RenderPipe:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    antialiasing: bool = False


def _import_rasterizer():
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

        return GaussianRasterizationSettings, GaussianRasterizer, "standard"
    except ImportError as e:
        raise ImportError(
            "未安装 diff_gaussian_rasterization（可微光栅化 CUDA 扩展）。\n"
            "请在项目根目录执行：\n"
            "  cd third_party/gaussian-splatting/submodules/diff-gaussian-rasterization\n"
            "  pip install . --no-build-isolation\n"
            "安装成功后，在项目根目录验证（不要在源码目录里 import）：\n"
            "  cd F:\\learning\\Project\n"
            "  python -c \"from diff_gaussian_rasterization import GaussianRasterizer; print('OK')\"\n"
        ) from e


def render(
    viewpoint: DiffCamera,
    gaussians: GaussianData,
    pipe: RenderPipe,
    bg_color: torch.Tensor,
    *,
    scaling_modifier: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    与原版 gaussian-splatting/gaussian_renderer/render 接口一致（精简版）。
    返回 dict: render (3,H,W), viewspace_points, visibility_filter, radii
    """
    GaussianRasterizationSettings, GaussianRasterizer, _ = _import_rasterizer()

    screenspace_points = torch.zeros_like(gaussians.get_xyz, requires_grad=True, device=gaussians.get_xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint.FoVx * 0.5)
    tanfovy = math.tan(viewpoint.FoVy * 0.5)

    settings_kwargs = dict(
        image_height=int(viewpoint.image_height),
        image_width=int(viewpoint.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint.world_view_transform,
        projmatrix=viewpoint.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    # 新版 rasterizer 支持 antialiasing
    try:
        settings = GaussianRasterizationSettings(antialiasing=pipe.antialiasing, **settings_kwargs)
    except TypeError:
        settings = GaussianRasterizationSettings(**settings_kwargs)

    rasterizer = GaussianRasterizer(raster_settings=settings)

    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        raise NotImplementedError("compute_cov3D_python 暂未实现，请保持 False")
    else:
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

    shs = gaussians.get_features
    colors_precomp = None

    out = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    if isinstance(out, tuple) and len(out) == 3:
        rendered_image, radii, _depth = out
    else:
        rendered_image, radii = out

    rendered_image = rendered_image.clamp(0.0, 1.0)
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(as_tuple=False),
        "radii": radii,
    }
