"""
从 style.png 生成 style_resize.png，供 style_transfer.py 默认使用。

阶段一 Attention Distillation 对风格图调用 load_image(path, size=(512, 512))，
即 BICUBIC 缩放到固定 512×512（见 style_transfer.run_stage1_ad_with_pipe）。
本脚本用相同规则预处理，避免手工尺寸不一致。

用法（在项目根目录下）：
  python prepare_style_resize.py
  python prepare_style_resize.py -i my_style.jpg -o style_resize.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent

# 与 style_transfer.run_stage1_ad_with_pipe 中 load_image(..., size=(512, 512)) 一致
DEFAULT_STYLE_W = 512
DEFAULT_STYLE_H = 512


def main() -> None:
    p = argparse.ArgumentParser(
        description="将风格图缩放为 AD 阶段所需的尺寸（默认 512×512，BICUBIC）"
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=PROJECT_ROOT / "style.png",
        help="原始风格图（默认 style.png）",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=PROJECT_ROOT / "style_resize.png",
        help="输出路径（默认 style_resize.png）",
    )
    p.add_argument(
        "--width",
        type=int,
        default=DEFAULT_STYLE_W,
        help="输出宽，须与 style_transfer 中 load_image 的 size 一致（默认 512）",
    )
    p.add_argument(
        "--height",
        type=int,
        default=DEFAULT_STYLE_H,
        help="输出高（默认 512）",
    )
    args = p.parse_args()

    src = args.input.expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"找不到输入: {src}")

    w, h = args.width, args.height
    if w < 1 or h < 1:
        raise SystemExit("width/height 须为正整数")

    im = Image.open(src).convert("RGB")
    out = im.resize((w, h), Image.BICUBIC)

    dst = args.output.expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    suf = dst.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        out.save(dst, quality=95)
    else:
        out.save(dst)

    print(f"wrote {dst} ({w}x{h}) <- {src}")


if __name__ == "__main__":
    main()
