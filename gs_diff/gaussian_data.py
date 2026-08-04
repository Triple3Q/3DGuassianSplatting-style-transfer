"""Load 3DGS PLY into tensors for diff rasterization (geometry fixed, colors optimizable)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData


def _inverse_sigmoid(x: float) -> float:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return float(np.log(x / (1.0 - x)))


class GaussianData(nn.Module):
    """
    Lightweight 3DGS parameter container compatible with diff_gaussian_rasterization.

    Only `_features_dc` is intended to be optimized in gs_render_diff.py.
    """

    def __init__(self, max_sh_degree: int = 0) -> None:
        super().__init__()
        self.max_sh_degree = max_sh_degree
        self.active_sh_degree = max_sh_degree

        self._xyz = nn.Parameter(torch.empty(0))
        self._features_dc = nn.Parameter(torch.empty(0))
        self.register_buffer("_features_rest", torch.empty(0))
        self.register_buffer("_opacity", torch.empty(0))
        self.register_buffer("_scaling", torch.empty(0))
        self.register_buffer("_rotation", torch.empty(0))

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_features_dc(self) -> torch.Tensor:
        return self._features_dc

    @property
    def get_features_rest(self) -> torch.Tensor:
        return self._features_rest

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    def load_ply(self, path: Path, device: torch.device) -> None:
        ply = PlyData.read(str(path))
        v = ply["vertex"].data
        names = v.dtype.names or ()

        xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)

        n = xyz.shape[0]
        features_dc = np.zeros((n, 3, 1), dtype=np.float32)
        for i, ch in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
            if ch in names:
                features_dc[:, i, 0] = np.asarray(v[ch])
            else:
                raise ValueError(f"PLY 缺少 {ch}，无法优化颜色")

        rest_names = sorted(
            [p.name for p in ply["vertex"].properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        n_rest = len(rest_names)
        if n_rest > 0:
            sh_degree = int(round(((n_rest + 3) / 3) ** 0.5 - 1.0))
            self.max_sh_degree = sh_degree
            self.active_sh_degree = sh_degree
        else:
            sh_degree = 0
            self.max_sh_degree = 0
            self.active_sh_degree = 0

        expected_rest = 3 * (sh_degree + 1) ** 2 - 3
        features_rest = np.zeros((n, expected_rest), dtype=np.float32)
        for idx, attr in enumerate(rest_names[:expected_rest]):
            features_rest[:, idx] = np.asarray(v[attr])
        features_rest = features_rest.reshape(n, 3, (sh_degree + 1) ** 2 - 1) if expected_rest else np.zeros((n, 3, 0), dtype=np.float32)

        if "opacity" in names:
            opacities = np.asarray(v["opacity"], dtype=np.float32)[:, None]
        else:
            opacities = np.full((n, 1), _inverse_sigmoid(0.95), dtype=np.float32)

        scale_names = sorted(
            [p.name for p in ply["vertex"].properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if scale_names:
            scales = np.stack([np.asarray(v[s]) for s in scale_names], axis=1).astype(np.float32)
        else:
            scales = np.full((n, 3), -6.0, dtype=np.float32)  # exp(-6) ≈ 小高斯

        rot_names = sorted(
            [p.name for p in ply["vertex"].properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if rot_names:
            rots = np.stack([np.asarray(v[s]) for s in rot_names], axis=1).astype(np.float32)
        else:
            rots = np.zeros((n, 4), dtype=np.float32)
            rots[:, 0] = 1.0

        self._xyz = nn.Parameter(torch.tensor(xyz, device=device), requires_grad=False)
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, device=device).transpose(1, 2).contiguous(),
            requires_grad=True,
        )
        self._features_rest = torch.tensor(features_rest, device=device).transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacities, device=device)
        self._scaling = torch.tensor(scales, device=device)
        self._rotation = torch.tensor(rots, device=device)

    def save_ply(self, path: Path) -> None:
        from plyfile import PlyElement

        if self.has_mesh_normalization():
            self.denormalize_for_save()

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rotations = self._rotation.detach().cpu().numpy()

        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(f_dc.shape[1]):
            attrs.append(f"f_dc_{i}")
        for i in range(f_rest.shape[1]):
            attrs.append(f"f_rest_{i}")
        attrs.append("opacity")
        for i in range(scales.shape[1]):
            attrs.append(f"scale_{i}")
        for i in range(rotations.shape[1]):
            attrs.append(f"rot_{i}")

        dtype = [(a, "f4") for a in attrs]
        arr = np.empty(xyz.shape[0], dtype=dtype)
        block = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rotations), axis=1)
        arr[:] = list(map(tuple, block))
        PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))

    def freeze_geometry(self) -> None:
        self._xyz.requires_grad_(False)
        self._opacity.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._features_rest.requires_grad_(False)
        self._features_dc.requires_grad_(True)

    def apply_mesh_normalization(self, center: np.ndarray, scale: float) -> None:
        """
        与 render_multiview_obj.py 一致：xyz' = (xyz - center) / scale，并同步调整高斯尺度。
        """
        if scale <= 0:
            raise ValueError("scale 必须 > 0")
        self._norm_center = np.asarray(center, dtype=np.float32)
        self._norm_scale = float(scale)
        c = torch.tensor(self._norm_center, device=self._xyz.device, dtype=self._xyz.dtype)
        self._xyz.data = (self._xyz.data - c) / scale
        log_s = math.log(scale)
        self._scaling.data = self._scaling.data - log_s

    def has_mesh_normalization(self) -> bool:
        return hasattr(self, "_norm_center") and hasattr(self, "_norm_scale")

    def denormalize_for_save(self) -> None:
        if not self.has_mesh_normalization():
            return
        scale = self._norm_scale
        c = torch.tensor(self._norm_center, device=self._xyz.device, dtype=self._xyz.dtype)
        self._xyz.data = self._xyz.data * scale + c
        self._scaling.data = self._scaling.data + math.log(scale)
