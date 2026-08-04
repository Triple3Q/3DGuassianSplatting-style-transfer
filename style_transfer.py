"""
2D 多视角批量风格迁移（独立项目，不修改 DDTS）。

- 流程：AD (SD1.5) → CSGO (SDXL)；两阶段均**不使用**掩码（整图风格迁移）。
- 阶段二结束后：可选 OpenCV grabCut 得到前景掩码，仅用于**最终合成**——前景用风格图、背景用原图替换；掩码不落盘。
- 依赖与 DDTS 相同：在「2dstyletransfer 的上一级」使用 **Attention Distillation/**、**CSGO/** 两个文件夹；
  阶段一从 AD 仓库调 `pipeline_sd` / `utils`，阶段二从 **CSGO 仓库** 调 `ip_adapter` 与 **CSGO/base_models/** 下 SDXL 资源、
  根目录的 **csgo_4_32_v2.bin**。这些都不需要复制到 `2dstyletransfer/` 与脚本同目录。
- 可选：临时覆盖时可使用 `2dstyletransfer/attention_distillation/` 或 `2dstyletransfer/CSGO/`（与 DDTS 一致）。
- 仅最终图写入 output/。

目标约 8～10 分钟/张（16 视角 + 方案 A 默认步数；因 GPU 差异大，需自测微调）时，还可继续动：
  阶段一：num_inference_steps ↓、iters ↓、start_layer/end_layer 收窄、mixed_precision 用 fp16、
         ad_width/ad_height ↓（如 448 或 384）。
  阶段二：num_inference_steps ↓（如 12～15，当前默认 15）。
  若明显快于目标可略增步数换画质；若超时则继续减步数或分辨率。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

# -----------------------------------------------------------------------------
# 路径：项目根目录下放置 Attention Distillation/ 与 CSGO/；大模型在 CSGO/base_models/ 等。
# 顺序：先 CSGO 再 AD 加入 sys.path，避免命名冲突。
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_ad_root() -> Path:
    sibling = PROJECT_ROOT / "Attention Distillation"
    if (sibling / "pipeline_sd.py").is_file():
        return sibling
    local = PROJECT_ROOT / "attention_distillation"
    if (local / "pipeline_sd.py").is_file():
        return local
    raise FileNotFoundError(
        "找不到 Attention Distillation：请在项目根目录放置 "
        "'Attention Distillation/'（内含 pipeline_sd.py），"
        "或仅在需要时使用 attention_distillation/ 覆盖。"
    )


def _resolve_csgo_root() -> Path:
    sibling = PROJECT_ROOT / "CSGO"
    if (sibling / "ip_adapter" / "__init__.py").is_file():
        return sibling
    raise FileNotFoundError(
        "找不到 CSGO：请在项目根目录放置 'CSGO/'（内含 ip_adapter/）。"
    )


def _ensure_syspath(include: Path) -> None:
    s = str(include.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


AD_ROOT = _resolve_ad_root()
CSGO_ROOT = _resolve_csgo_root()
_ensure_syspath(CSGO_ROOT)
_ensure_syspath(AD_ROOT)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# HuggingFace（SD1.5、BLIP 等 from_pretrained）走镜像，减轻 WinError 10060 / httpx 超时。
# 若已在环境变量中设置 HF_ENDPOINT，则不会覆盖；需官方源时可 export HF_ENDPOINT=https://huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionXLControlNetPipeline,
)
from pipeline_sd import ADPipeline
from transformers import BlipForConditionalGeneration, BlipProcessor
from utils import Controller, load_image

from ip_adapter import CSGO
from ip_adapter.utils import BLOCKS, controlnet_BLOCKS, resize_content

MODEL_SD15 = "runwayml/stable-diffusion-v1-5"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 内容图缩放边长；略小于 512 可明显加速，画质略降。与 run_stage1_ad_with_pipe 默认一致。
DEFAULT_AD_WIDTH = 448
DEFAULT_AD_HEIGHT = 448

# 第 i 张图使用 seed = SEED_BASE + i；若改 run_stage2_csgo_with_stack 的 seed 语义，可只改循环里传参
SEED_BASE = 42

# grabCut：中心矩形 ROI（四边各留 margin）；物体偏置时可在 grabcut_foreground_mask_tensor 里改 rect 策略
GRABCUT_RECT_MARGIN = 0.08  # 0.08 表示中心约 84% 区域作为「可能含物体」的初始框
GRABCUT_ITER_COUNT = 5  # grabCut 迭代次数；略增可能更稳但更慢

# 阶段二后与原图合成时，对掩码做轻微高斯羽化（像素，奇数核由内部换算）；0 表示不羽化
COMPOSITE_MASK_FEATHER_PX = 5


# ----- 前景掩码（仅最终合成用；AD/CSGO 内部不传掩码） -----


def grabcut_foreground_mask_tensor(
    image_path: Path,
    width: int,
    height: int,
    *,
    rect_margin: float = GRABCUT_RECT_MARGIN,
    iter_count: int = GRABCUT_ITER_COUNT,
) -> torch.Tensor:
    """
    用 cv2.grabCut 区分前景/背景：中心矩形 ROI 初始化（GC_INIT_WITH_RECT）。
    返回 CPU 上 [1,1,H,W] 浮点掩码，前景 1、背景 0；仅用于 composite，不参与阶段一/二。
    若近似全黑则退化为全 1（合成后等价整图均为风格结果）。
    """
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    h, w = bgr.shape[:2]

    mx = max(1, int(round(w * rect_margin)))
    my = max(1, int(round(h * rect_margin)))
    rw = max(1, w - 2 * mx)
    rh = max(1, h - 2 * my)
    rect = (mx, my, rw, rh)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, iter_count, cv2.GC_INIT_WITH_RECT)

    # 确定前景：GC_FGD 与 GC_PR_FGD
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.float32)
    t = torch.from_numpy(fg).float().unsqueeze(0).unsqueeze(0)
    if float(t.max()) < 1e-3:
        print("grabcut: empty fg mask, using full image")
        t.fill_(1.0)
    return t


def _mask_tensor_to_pil_l(
    mask_t: torch.Tensor,
    size_wh: tuple[int, int],
    *,
    feather_px: int = COMPOSITE_MASK_FEATHER_PX,
) -> Image.Image:
    """[1,1,h,w] 浮点前景掩码 → 与 CSGO 输出同尺寸的 L 模式图（前景白）。"""
    m = mask_t.detach().cpu().numpy()[0, 0]
    m = np.clip(m, 0.0, 1.0)
    m = (m * 255.0).astype(np.uint8)
    im = Image.fromarray(m, mode="L")
    im = im.resize(size_wh, Image.BILINEAR)
    if feather_px > 0:
        k = 2 * int(feather_px) + 1
        arr = np.array(im)
        arr = cv2.GaussianBlur(arr, (k, k), 0)
        im = Image.fromarray(arr, mode="L")
    return im


def composite_styled_foreground_preserve_background(
    styled_path: Path,
    original_img_path: Path,
    mask_t: torch.Tensor,
) -> Image.Image:
    """
    阶段二整图风格化后：前景用风格结果，背景用原图（与 CSGO 相同的 resize_content 几何）。
    """
    orig_base = Image.open(original_img_path).convert("RGB")
    orig_base = orig_base.resize((DEFAULT_AD_WIDTH, DEFAULT_AD_HEIGHT), Image.BICUBIC)
    _w, _h, orig_for_csgo = resize_content(orig_base)

    styled = Image.open(styled_path).convert("RGB")
    if styled.size != orig_for_csgo.size:
        styled = styled.resize(orig_for_csgo.size, Image.BICUBIC)

    mask_pil = _mask_tensor_to_pil_l(mask_t, orig_for_csgo.size)
    return Image.composite(styled, orig_for_csgo, mask_pil)


# ----- 阶段一：可复用 pipe -----


def load_ad_pipeline(
    *,
    model_name: str = MODEL_SD15,  # SD1.5 权重名
    vae_override: str = "",  # 非空则替换 VAE
    device: str | None = None,  # None 则 cuda:0 或 cpu
):
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    pipe = ADPipeline.from_pretrained(
        model_name,
        scheduler=scheduler,
        safety_checker=None,
    )
    if vae_override:
        pipe.vae = AutoencoderKL.from_pretrained(vae_override)
    pipe.classifier = pipe.unet
    pipe = pipe.to(dev)
    return pipe, dev


def run_stage1_ad_with_pipe(
    pipe,
    style_resize_path: Path,
    initial_texture_path: Path,
    out_dir: Path,
    *,
    model_name: str = MODEL_SD15,  # 仅用于 save_latent 时 meta
    ad_width: int = DEFAULT_AD_WIDTH,  # 内容图宽
    ad_height: int = DEFAULT_AD_HEIGHT,  # 内容图高
    lr: float = 0.05,  # 潜变量优化学习率
    iters: int = 5,  # 方案 A：每 DDIM 步内优化次数（原 8）
    weight: float = 0.25,  # 内容损失 q_loss 权重
    batch_size: int = 1,
    num_inference_steps: int = 30,  # 方案 A：DDIM 步数（原 40）
    start_layer: int = 11,  # 参与 AD 的自注意力层起始（含）；区间越窄越快
    end_layer: int = 15,  # 参与 AD 的自注意力层结束（含）
    mixed_precision: str = "fp16",  # fp16 通常快于 no；若 NaN/不稳可改 no 或 bf16
    save_latent: bool = True,  # 是否写 ad_latent_sd15.pt；批量时可 False 省时间与磁盘
) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = next(pipe.unet.parameters()).device

    style_image_paths = [str(style_resize_path)]
    style_image = torch.cat(
        [load_image(p, size=(512, 512)) for p in style_image_paths]
    )
    content_image = load_image(
        str(initial_texture_path), size=(ad_width, ad_height)
    ).to(dev)

    controller = Controller(self_layers=(start_layer, end_layer))

    result = pipe.optimize(
        lr=lr,
        batch_size=batch_size,
        iters=iters,
        width=ad_width,
        height=ad_height,
        weight=weight,
        controller=controller,
        style_image=style_image,
        content_image=content_image,
        mixed_precision=mixed_precision,
        num_inference_steps=num_inference_steps,
        content_mask=None,
    )

    ad_png = out_dir / "ad_result.png"
    save_image(result, ad_png)

    latent_path: Path | None = None
    if save_latent:
        with torch.no_grad():
            latent = pipe.image2latent(result)
        latent_path = out_dir / "ad_latent_sd15.pt"
        torch.save(
            {"latent": latent.cpu(), "meta": {"model": model_name, "stage": "SD1.5-VAE"}},
            latent_path,
        )

    print(f"stage1: {ad_png.name}" + (f", latent ok" if latent_path else ""))
    return ad_png, latent_path


# ----- 阶段二：可复用 CSGO 栈 -----


def load_csgo_stack(
    csgo_root: Path | None = None,  # None 则使用模块级 CSGO_ROOT
    *,
    hf_endpoint: str | None = None,  # 如 https://hf-mirror.com；None 用环境变量或默认
    device: str | None = None,  # None 则 cuda:0 或 cpu
):
    root = Path(csgo_root) if csgo_root is not None else CSGO_ROOT
    if not (root / "ip_adapter" / "__init__.py").is_file():
        raise FileNotFoundError(f"无效的 CSGO 根目录: {root}")

    if hf_endpoint is not None:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    base_models_dir = root / "base_models"
    base_model_path = base_models_dir / "stable-diffusion-xl-base-1.0"
    image_encoder_path = base_models_dir / "IP-Adapter" / "sdxl_models" / "image_encoder"
    csgo_ckpt = (root / "csgo_4_32_v2.bin").resolve()
    if not csgo_ckpt.is_file():
        raise FileNotFoundError(
            f"缺少 CSGO 权重（应放在 CSGO 仓库根目录、与 ip_adapter 同级，无需放到 2dstyletransfer/）: {csgo_ckpt}"
        )
    pretrained_vae_name_or_path = base_models_dir / "sdxl-vae-fp16-fix"
    controlnet_path = base_models_dir / "TTPLanet_SDXL_Controlnet_Tile_Realistic"

    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(dev)

    vae = AutoencoderKL.from_pretrained(
        str(pretrained_vae_name_or_path), torch_dtype=torch.float16
    )
    controlnet = ControlNetModel.from_pretrained(
        str(controlnet_path), torch_dtype=torch.float16, use_safetensors=True
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        str(base_model_path),
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False,
        vae=vae,
    )
    pipe.enable_vae_tiling()

    target_content_blocks = BLOCKS["content"]
    target_style_blocks = BLOCKS["style"]
    controlnet_target_content_blocks = controlnet_BLOCKS["content"]
    controlnet_target_style_blocks = controlnet_BLOCKS["style"]

    csgo = CSGO(
        pipe,
        str(image_encoder_path),
        str(csgo_ckpt),
        dev,
        num_content_tokens=4,  # 内容图像 token 数（CSGO 默认）
        num_style_tokens=32,  # 风格图像 token 数
        target_content_blocks=target_content_blocks,
        target_style_blocks=target_style_blocks,
        controlnet_adapter=True,
        controlnet_target_content_blocks=controlnet_target_content_blocks,
        controlnet_target_style_blocks=controlnet_target_style_blocks,
        content_model_resampler=True,
        style_model_resampler=True,
    )

    return {
        "csgo": csgo,
        "blip_processor": blip_processor,
        "blip_model": blip_model,
        "device": dev,
        "resize_content": resize_content,
    }


def run_stage2_csgo_with_stack(
    stack: dict,
    ad_result_path: Path,
    style_resize_path: Path,
    out_path: Path,
    *,
    seed: int = 42,  # 随机种子；批量中常传 SEED_BASE + 图像序号
    num_inference_steps: int = 15,  # 方案 A：SDXL 去噪步数（原 20）
    content_scale: float = 0.5,  # 内容分支强度
    style_scale: float = 1.0,  # 风格分支强度
    guidance_scale: float = 10.0,  # 文本 CFG
    controlnet_conditioning_scale: float = 0.6,  # ControlNet 强度
    num_images: int = 1,  # 每轮生成张数
) -> Path:
    csgo = stack["csgo"]
    blip_processor = stack["blip_processor"]
    blip_model = stack["blip_model"]
    device = stack["device"]
    resize_content = stack["resize_content"]

    style_image = Image.open(style_resize_path).convert("RGB")
    content_image = Image.open(ad_result_path).convert("RGB")

    with torch.no_grad():
        inputs = blip_processor(content_image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

    width, height, content_image = resize_content(content_image)

    images = csgo.generate(
        pil_content_image=content_image,
        pil_style_image=style_image,
        prompt=caption,
        negative_prompt=(
            "text, watermark, lowres, low quality, worst quality, deformed, glitch, "
            "low contrast, noisy, saturation, blurry"
        ),
        height=height,
        width=width,
        content_scale=content_scale,
        style_scale=style_scale,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        num_samples=1,
        num_inference_steps=num_inference_steps,
        seed=seed,
        image=content_image.convert("RGB"),
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_file)
    print(f"stage2: {out_file.name}")
    return out_file


def _natural_sort_key(p: Path):
    name = p.stem
    parts = re.split(r"(\d+)", name)
    key = []
    for x in parts:
        if x.isdigit():
            key.append(int(x))
        else:
            key.append(x.lower())
    return key


def _list_rendered_images(rendered_dir: Path) -> list[Path]:
    if not rendered_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {rendered_dir}")
    files = [
        p
        for p in rendered_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]
    files.sort(key=_natural_sort_key)
    return files


def run_batch(
    *,
    rendered_dir: Path | None = None,  # None 则 TWO_D_ROOT / "rendered_images"
    style_path: Path | None = None,  # None 则 TWO_D_ROOT / "style_resize.png"
    output_dir: Path | None = None,  # None 则 TWO_D_ROOT / "output"
    limit: int | None = None,  # 仅处理前 N 张；None 表示全部
    hf_endpoint: str | None = None,  # HuggingFace 镜像等
    use_object_mask: bool = True,  # True：阶段二后 grabCut 合成，背景换回原图；False：直接输出阶段二
) -> list[Path]:
    """对 rendered_dir 下每张图跑 AD→CSGO；可选在最后用 grabCut 掩码做背景替换；掩码不落盘。"""
    rendered_dir = rendered_dir or (PROJECT_ROOT / "rendered_images")
    style_path = style_path or (PROJECT_ROOT / "style_resize.png")
    output_dir = output_dir or (PROJECT_ROOT / "output")

    if not style_path.is_file():
        raise FileNotFoundError(f"风格图不存在: {style_path}")

    images = _list_rendered_images(rendered_dir)
    if limit is not None:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"{rendered_dir} 下未找到支持的图片（{sorted(_IMAGE_EXTS)}）")

    output_dir.mkdir(parents=True, exist_ok=True)

    pipe, _dev = load_ad_pipeline()
    stack = load_csgo_stack(hf_endpoint=hf_endpoint)

    results: list[Path] = []
    try:
        for i, img_path in enumerate(images, start=1):
            final_name = f"{img_path.stem}_styled.png"
            final_path = output_dir / final_name
            print(f"[{i}/{len(images)}] {img_path.name} -> {final_path.name}")

            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                style_in = Path(shutil.copy2(style_path, td / "style_resize.png"))

                ad_result, _ = run_stage1_ad_with_pipe(
                    pipe,
                    style_in,
                    img_path,
                    td,
                    save_latent=False,
                )

                run_stage2_csgo_with_stack(
                    stack,
                    ad_result,
                    style_in,
                    td / "_stage2.png",
                    seed=SEED_BASE + i,
                )
                if use_object_mask:
                    mask_cpu = grabcut_foreground_mask_tensor(
                        img_path,
                        DEFAULT_AD_WIDTH,
                        DEFAULT_AD_HEIGHT,
                    )
                    out_pil = composite_styled_foreground_preserve_background(
                        td / "_stage2.png",
                        img_path,
                        mask_cpu,
                    )
                    out_pil.save(final_path)
                else:
                    shutil.copy2(td / "_stage2.png", final_path)
                results.append(final_path)
    finally:
        del pipe, stack
        torch.cuda.empty_cache()

    return results


def main():
    p = argparse.ArgumentParser(
        description="多视角批量风格迁移；路径可通过命令行覆盖，其余超参请改本文件各函数默认形参。",
    )
    p.add_argument(
        "--rendered-dir",
        type=Path,
        default=None,
        help="渲染图目录（默认 rendered_images）",
    )
    p.add_argument(
        "--style",
        type=Path,
        default=None,
        help="风格图（默认 style_resize.png）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出目录（默认 output）",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 张",
    )
    p.add_argument(
        "--hf-endpoint",
        type=str,
        default=None,
        help="HF 镜像，如 https://hf-mirror.com",
    )
    p.add_argument(
        "--no-object-mask",
        action="store_true",
        help="不做 grabCut 与最终背景替换，直接输出阶段二整图",
    )
    args = p.parse_args()

    run_batch(
        rendered_dir=args.rendered_dir.resolve() if args.rendered_dir else None,
        style_path=args.style.resolve() if args.style else None,
        output_dir=args.output.resolve() if args.output else None,
        limit=args.limit,
        hf_endpoint=args.hf_endpoint,
        use_object_mask=not args.no_object_mask,
    )
    print("done.")


if __name__ == "__main__":
    main()
