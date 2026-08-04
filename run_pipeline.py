"""
run_pipeline.py - 按顺序检查并运行风格迁移流水线

用法（在项目根目录）：
  python run_pipeline.py              # 运行全部可执行步骤
  python run_pipeline.py --step 0     # 只运行某一步
  python run_pipeline.py --check      # 只检查前置条件

步骤：
  0  prepare_style_resize.py
  1  blender_render_multiview.py     （需 Blender + .blend 场景）
     或 render_multiview_obj.py      （仅需 OBJ，无需 Blender）
  2  style_transfer.py
  3  gs_render_diff.py               （需 diff_gaussian_rasterization）
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PY = sys.executable


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def run_cmd(cmd: list[str], *, cwd: Path = PROJECT_ROOT) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(cwd))


def find_blender() -> Path | None:
    for name in ("blender", "blender.exe"):
        p = shutil.which(name)
        if p:
            return Path(p)
    for pattern in (
        r"C:\Program Files\Blender Foundation\Blender *\blender.exe",
        r"D:\Blender\blender.exe",
    ):
        import glob

        hits = glob.glob(pattern)
        if hits:
            return Path(sorted(hits)[-1])
    return None


def check_env() -> bool:
    good = True
    try:
        import torch

        ok(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            warn("CUDA 不可用，style_transfer / gs_render_diff 可能无法运行")
    except ImportError:
        fail("未安装 torch，请先: conda activate GSstyletransfer")
        good = False

    try:
        from diff_gaussian_rasterization import GaussianRasterizer  # noqa: F401

        ok("diff_gaussian_rasterization 已安装")
    except ImportError:
        warn("diff_gaussian_rasterization 未安装，步骤 3 将跳过")
        warn("安装: cd .../diff-gaussian-rasterization && pip install . --no-build-isolation")

    if not (PROJECT_ROOT / "CSGO" / "csgo_4_32_v2.bin").is_file():
        fail("缺少 CSGO/csgo_4_32_v2.bin")
        good = False
    else:
        ok("CSGO 权重存在")

    if not (PROJECT_ROOT / "Attention Distillation" / "pipeline_sd.py").is_file():
        fail("缺少 Attention Distillation/pipeline_sd.py")
        good = False
    else:
        ok("Attention Distillation 存在")

    return good


def check_data(step: int, *, obj_mesh: Path | None = None) -> bool:
    style = PROJECT_ROOT / "style.png"
    style_r = PROJECT_ROOT / "style_resize.png"
    rendered = PROJECT_ROOT / "rendered_images"
    output = PROJECT_ROOT / "output"
    ply_dir = PROJECT_ROOT / "3DGS_output"

    if step <= 0:
        if not style.is_file():
            fail(f"缺少 {style}（步骤 0 需要原始风格图）")
            return False
        ok(f"style.png 存在")

    if step <= 1 and step >= 1:
        if obj_mesh is not None:
            if not obj_mesh.is_file():
                fail(f"缺少 OBJ 文件: {obj_mesh}")
                return False
            ok(f"OBJ: {obj_mesh}")
        else:
            blender = find_blender()
            if blender is None:
                fail("未找到 Blender，请安装并加入 PATH，或使用 --obj 走 render_multiview_obj.py")
                return False
            ok(f"Blender: {blender}")

    if step <= 2 and step >= 2:
        if not style_r.is_file():
            fail(f"缺少 {style_r}，请先运行步骤 0")
            return False
        imgs = list(rendered.glob("*.png")) if rendered.is_dir() else []
        if not imgs:
            fail(f"{rendered}/ 下无渲染图，请先运行步骤 1")
            return False
        ok(f"rendered_images: {len(imgs)} 张")

    if step <= 3 and step >= 3:
        styled = list(output.glob("*_styled.png")) if output.is_dir() else []
        if not styled:
            fail(f"{output}/ 下无 *_styled.png，请先运行步骤 2")
            return False
        plies = list(ply_dir.glob("*.ply")) if ply_dir.is_dir() else []
        if not plies:
            fail(f"{ply_dir}/ 下无 .ply 文件")
            return False
        ok(f"output: {len(styled)} 张风格图; 3DGS_output: {len(plies)} 个 PLY")

    return True


def step0() -> int:
    if not check_data(0):
        return 1
    return run_cmd([PY, "prepare_style_resize.py"])


def step1(blend: Path | None, obj_mesh: Path | None) -> int:
    if not check_data(1, obj_mesh=obj_mesh):
        return 1
    if obj_mesh is not None:
        return run_cmd([PY, "render_multiview_obj.py", str(obj_mesh.resolve())])
    blender = find_blender()
    if blender is None:
        return 1
    script = PROJECT_ROOT / "blender_render_multiview.py"
    if blend and blend.is_file():
        return run_cmd([str(blender), "--background", str(blend), "--python", str(script)])
    warn("未指定 --blend，尝试无场景文件运行（需脚本内已有默认场景或会失败）")
    return run_cmd([str(blender), "--background", "--python", str(script)])


def step2(limit: int | None) -> int:
    if not check_data(2):
        return 1
    cmd = [PY, "style_transfer.py"]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    return run_cmd(cmd)


def step3(ply: Path | None, out: Path | None, iterations: int) -> int:
    try:
        from diff_gaussian_rasterization import GaussianRasterizer  # noqa: F401
    except ImportError:
        fail("diff_gaussian_rasterization 未安装，无法运行步骤 3")
        return 1
    if not check_data(3):
        return 1
    ply_dir = PROJECT_ROOT / "3DGS_output"
    ply_in = ply or next(iter(sorted(ply_dir.glob("*.ply"))), None)
    if ply_in is None:
        fail("找不到输入 PLY")
        return 1
    ply_out = out or (ply_dir / f"{ply_in.stem}_diff.ply")
    return run_cmd(
        [PY, "gs_render_diff.py", str(ply_in), str(ply_out), "--iterations", str(iterations)]
    )


def main() -> int:
    os.chdir(PROJECT_ROOT)
    ap = argparse.ArgumentParser(description="一键运行 Project 流水线")
    ap.add_argument("--check", action="store_true", help="只检查环境/数据")
    ap.add_argument("--step", type=int, default=-1, help="只运行指定步骤 0-3；-1 表示全部")
    ap.add_argument("--blend", type=Path, default=None, help="Blender .blend 场景（步骤 1，与 --obj 二选一）")
    ap.add_argument("--obj", type=Path, default=None, help="OBJ 网格路径（步骤 1，无需 Blender）")
    ap.add_argument("--ply", type=Path, default=None, help="输入 PLY（步骤 3）")
    ap.add_argument("--output-ply", type=Path, default=None, help="输出 PLY（步骤 3）")
    ap.add_argument("--limit", type=int, default=None, help="style_transfer 只处理前 N 张")
    ap.add_argument("--iterations", type=int, default=3000, help="gs_render_diff 迭代次数")
    args = ap.parse_args()

    print("=== 环境检查 ===")
    check_env()

    if args.check:
        print("\n=== 数据检查 ===")
        for s in range(4):
            print(f"\n-- 步骤 {s} --")
            check_data(s)
        return 0

    steps = [0, 1, 2, 3] if args.step < 0 else [args.step]
    for s in steps:
        print(f"\n========== 步骤 {s} ==========")
        if s == 0:
            rc = step0()
        elif s == 1:
            rc = step1(args.blend, args.obj)
        elif s == 2:
            rc = step2(args.limit)
        elif s == 3:
            rc = step3(args.ply, args.output_ply, args.iterations)
        else:
            fail(f"未知步骤 {s}")
            rc = 1
        if rc != 0:
            fail(f"步骤 {s} 失败 (exit {rc})")
            return rc
        ok(f"步骤 {s} 完成")

    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
