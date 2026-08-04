"""
用 Open3D 查看 3DGS 导出的 PLY。

说明：Open3D 默认 read_point_cloud 不认 f_dc_*，只会当普通顶点读入，颜色往往不对或全灰。
      本脚本用 plyfile 读 f_dc_0/1/2，按 DC 项近似 rgb = clip(0.5 + SH_C0 * f_dc, 0, 1) 上色。
      对比两个 PLY 的 f_dc 差异时可用 --diff-with（放大显示逐点 |Δrgb|）。

用法示例（在项目根目录下）：
  python visualized_gs.py my.ply
  python visualized_gs.py before.ply after.ply
  python visualized_gs.py --ply my.ply --diff-with other.ply

位置参数：第一个为主 PLY；若给第二个，则等价于 --diff-with（逐点差分显示）。

若要照片级渲染效果，请用 Gaussian Splatting 专用查看器。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

SH_C0 = 0.28209479177387814

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PLY = PROJECT_ROOT / "3DGS_output" / "gs_styled_50k.ply"


def _dc_to_rgb(v) -> np.ndarray:
    """vertex structured array -> (N,3) RGB in [0,1]"""
    dc = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float64)
    return np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)


def load_3dgs_ply_with_dc_colors(ply_path: Path) -> o3d.geometry.PointCloud:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    names = v.dtype.names or ()
    if not ("x" in names and "y" in names and "z" in names):
        raise ValueError("PLY 顶点缺少 x/y/z")

    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if "f_dc_0" in names and "f_dc_1" in names and "f_dc_2" in names:
        rgb = _dc_to_rgb(v)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if pcd.is_empty():
            raise ValueError("无法读取点云或缺少 f_dc 且无标准颜色")

    return pcd


def load_diff_pcd(
    base_ply: Path,
    other_ply: Path,
    *,
    diff_scale: float = 8.0,
) -> tuple[o3d.geometry.PointCloud, str]:
    """
    用两文件 f_dc 近似 RGB 的逐点差 |Δrgb|，乘 diff_scale 后 clip 到 [0,1]，灰度显示「改了多少」。
    顶点数须一致且顺序相同（同一导出管线）。
    """
    vb = PlyData.read(str(base_ply))["vertex"].data
    vo = PlyData.read(str(other_ply))["vertex"].data
    if len(vb) != len(vo):
        raise ValueError(f"顶点数不一致: {len(vb)} vs {len(vo)}，无法逐点差分")
    xyz = np.column_stack([vb["x"], vb["y"], vb["z"]]).astype(np.float64)
    ra, rb = _dc_to_rgb(vb), _dc_to_rgb(vo)
    diff = np.abs(rb - ra)
    mag = np.linalg.norm(diff, axis=1)
    gray = np.clip(mag * float(diff_scale), 0.0, 1.0)
    rgb_vis = np.column_stack([gray, gray, gray])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb_vis)
    stats = (
        f"|Δrgb| L2: mean={mag.mean():.5f} max={mag.max():.5f} "
        f"(乘 {diff_scale} 后作为灰度；越白=变化越大)"
    )
    return pcd, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="查看 3DGS PLY（f_dc 上色）；支持位置参数指定文件名",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="示例:  python visualized_gs.py 3DGS_output/head.ply\n"
        "        python visualized_gs.py base.ply updated.ply",
    )
    ap.add_argument(
        "ply_pos",
        nargs="?",
        type=Path,
        default=None,
        metavar="PLY",
        help="主 PLY 路径（可直接写在此，不必再加 --ply）",
    )
    ap.add_argument(
        "diff_pos",
        nargs="?",
        type=Path,
        default=None,
        metavar="OTHER.ply",
        help="差分对比第二路 PLY（可选；可直接写位置参数，不必再加 --diff-with；须与主 PLY 顶点数一致）",
    )
    ap.add_argument(
        "--ply",
        type=Path,
        default=DEFAULT_PLY,
        help=f"主 PLY（默认: {DEFAULT_PLY}；若写了位置参数则位置参数优先）",
    )
    ap.add_argument(
        "--diff-with",
        type=Path,
        default=None,
        help="与主 PLY 逐点对比 f_dc→RGB 差异，灰度放大显示（须同顶点数/顺序）",
    )
    ap.add_argument(
        "--diff-scale",
        type=float,
        default=8.0,
        help="差分灰度放大倍数，默认 8（越大越容易看见）",
    )
    args = ap.parse_args()

    main_ply = args.ply_pos if args.ply_pos is not None else args.ply
    diff_ply = args.diff_pos if args.diff_pos is not None else args.diff_with

    ply_path = main_ply.resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"未找到 PLY：{ply_path}")

    if diff_ply is not None:
        other = diff_ply.resolve()
        if not other.is_file():
            raise FileNotFoundError(f"未找到 PLY：{other}")
        pcd, stats = load_diff_pcd(ply_path, other, diff_scale=args.diff_scale)
        print(f"diff: {stats}")
        o3d.visualization.draw_geometries([pcd], window_name=f"{ply_path.name} vs {other.name}", width=800, height=600)
        return

    pcd = load_3dgs_ply_with_dc_colors(ply_path)
    if pcd.is_empty():
        print("empty point cloud")
        return

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"3DGS {ply_path.name}",
        width=800,
        height=600,
    )


if __name__ == "__main__":
    main()
