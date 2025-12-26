#!/usr/bin/env python3
"""
Triton + Torch GPU contrast metrics (Michelson + Weber) with CLI image loading.

- Torch-native decode via torchvision.io.read_image -> uint8 Tensor [3,H,W]
- GPU compute via a single Triton kernel pass:
    RGB -> (optional) sRGB->linear -> Rec.709 luminance
    global min/max (Michelson) + ROI sums/counts (Weber)
  reduced with Triton atomics into per-image scalars.

Examples:
  # Michelson only (global):
  python triton_contrast_cli.py image.jpg

  # Michelson + Weber with ROI boxes:
  python triton_contrast_cli.py image.jpg \
    --target 100,100,200,200 --background 0,0,80,80

  # Enable sRGB->linear luminance (more physically meaningful, more compute):
  python triton_contrast_cli.py image.jpg --linearize-srgb

  # Multiple images (batched if they share H,W):
  python triton_contrast_cli.py a.jpg b.png c.jpg --target 10,10,120,120 --background 0,0,50,50

Notes:
- Weber requires BOTH target and background boxes.
- Atomic float sums are not bitwise deterministic (order differs across blocks).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

try:
    from torchvision.io import ImageReadMode, read_image
except Exception as e:  # pragma: no cover
    read_image = None
    ImageReadMode = None


# -----------------------------
# Data types
# -----------------------------


@dataclass(frozen=True)
class Box:
    """Pixel box with x1,y1 exclusive (Python slicing semantics)."""

    x0: int
    y0: int
    x1: int
    y1: int


def _parse_box(s: str) -> Box:
    parts = s.split(",")
    if len(parts) != 4:
        raise ValueError(f"Expected 'x0,y0,x1,y1', got: {s}")
    x0, y0, x1, y1 = map(int, parts)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid box (need x1>x0 and y1>y0): {s}")
    return Box(x0, y0, x1, y1)


def _clip_box_to_image(box: Box, h: int, w: int) -> Box:
    x0 = max(0, min(w, box.x0))
    y0 = max(0, min(h, box.y0))
    x1 = max(0, min(w, box.x1))
    y1 = max(0, min(h, box.y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Box becomes empty after clipping to image size (H={h}, W={w}): {box}")
    return Box(x0, y0, x1, y1)


# -----------------------------
# Triton kernel (single-pass + atomic reductions)
# -----------------------------

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_W": 256, "BLOCK_H": 4}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_W": 256, "BLOCK_H": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_W": 512, "BLOCK_H": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_W": 128, "BLOCK_H": 8}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["H", "W"])
@triton.jit
def _contrast_atomic_kernel(
    img_ptr,  # * [B, 3, H, W] contiguous
    out_min_ptr,  # * [B] float32 (+inf init)
    out_max_ptr,  # * [B] float32 (-inf init)
    out_sum_t_ptr,  # * [B] float32 (0 init)
    out_sum_b_ptr,  # * [B] float32 (0 init)
    out_cnt_t_ptr,  # * [B] int32 (0 init)
    out_cnt_b_ptr,  # * [B] int32 (0 init)
    H,
    W,  # runtime ints
    t_x0,
    t_y0,
    t_x1,
    t_y1,
    b_x0,
    b_y0,
    b_x1,
    b_y1,
    INPUT_IS_UINT8: tl.constexpr,
    LINEARIZE_SRGB: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    # tile_x0 is always a multiple of BLOCK_W; our configs are multiples of 16.
    tile_x0 = pid_w * BLOCK_W
    tile_x0 = tl.multiple_of(tile_x0, 16)

    offs_w = tile_x0 + tl.arange(0, BLOCK_W)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    offs_w = tl.max_contiguous(offs_w, BLOCK_W)
    offs_h = tl.max_contiguous(offs_h, BLOCK_H)

    x = offs_w[None, :]  # [1, BW]
    y = offs_h[:, None]  # [BH, 1]
    in_bounds = (x < W) & (y < H)

    hw = H * W
    pix = y * W + x
    base = pid_b * (3 * hw) + pix

    # Load RGB (uint8 or float), then lift to float32
    r = tl.load(img_ptr + base + 0 * hw, mask=in_bounds, other=0).to(tl.float32)
    g = tl.load(img_ptr + base + 1 * hw, mask=in_bounds, other=0).to(tl.float32)
    b = tl.load(img_ptr + base + 2 * hw, mask=in_bounds, other=0).to(tl.float32)

    if INPUT_IS_UINT8:
        inv255 = 1.0 / 255.0
        r *= inv255
        g *= inv255
        b *= inv255

    # Optional sRGB -> linear (IEC 61966-2-1 inverse EOTF)
    if LINEARIZE_SRGB:
        th = 0.04045
        a = 0.055
        inv_1055 = 1.0 / 1.055
        inv_1292 = 1.0 / 12.92
        gamma = 2.4
        eps = 1e-8

        def srgb_to_linear(x_):
            lo = x_ * inv_1292
            hi = (x_ + a) * inv_1055
            hi = tl.maximum(hi, eps)
            hi = tl.exp2(tl.log2(hi) * gamma)
            return tl.where(x_ <= th, lo, hi)

        r = srgb_to_linear(r)
        g = srgb_to_linear(g)
        b = srgb_to_linear(b)

    # Rec.709 luminance
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # Ensure no negative (and avoid any weird signed-zero corner cases)
    luma = tl.maximum(luma, 0.0)

    # Local tile min/max (mask to identities)
    pos_inf = float("inf")
    neg_inf = -float("inf")
    luma_min_in = tl.where(in_bounds, luma, pos_inf)
    luma_max_in = tl.where(in_bounds, luma, neg_inf)

    tile_min = tl.min(tl.min(luma_min_in, axis=0), axis=0)  # scalar
    tile_max = tl.max(tl.max(luma_max_in, axis=0), axis=0)  # scalar

    # ROI membership
    in_t = in_bounds & (x >= t_x0) & (x < t_x1) & (y >= t_y0) & (y < t_y1)
    in_b = in_bounds & (x >= b_x0) & (x < b_x1) & (y >= b_y0) & (y < b_y1)

    sum_t = tl.sum(tl.sum(tl.where(in_t, luma, 0.0), axis=0), axis=0)
    sum_b = tl.sum(tl.sum(tl.where(in_b, luma, 0.0), axis=0), axis=0)

    cnt_t = tl.sum(tl.sum(in_t.to(tl.int32), axis=0), axis=0)
    cnt_b = tl.sum(tl.sum(in_b.to(tl.int32), axis=0), axis=0)

    # Atomic reductions into per-image scalars
    out_i = pid_b
    tl.atomic_min(out_min_ptr + out_i, tile_min)
    tl.atomic_max(out_max_ptr + out_i, tile_max)
    tl.atomic_add(out_sum_t_ptr + out_i, sum_t)
    tl.atomic_add(out_sum_b_ptr + out_i, sum_b)
    tl.atomic_add(out_cnt_t_ptr + out_i, cnt_t)
    tl.atomic_add(out_cnt_b_ptr + out_i, cnt_b)


def michelson_weber_triton(
    imgs: torch.Tensor,
    target: Box | None = None,
    background: Box | None = None,
    linearize_srgb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute Michelson (global) and optionally Weber (ROI) contrast for a batch of images.

    Args:
        imgs: [B,3,H,W] torch.uint8 or torch.float32 on GPU
        target: ROI for target (Weber numerator). If None, Weber is skipped.
        background: ROI for background (Weber denominator). If None, Weber is skipped.
        linearize_srgb: If True, apply sRGB->linear before computing luminance.

    Returns:
        michelson: [B] float32 on GPU
        weber: [B] float32 on GPU (or None if target/background not provided)
    """
    if imgs.dim() != 4 or imgs.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W], got {imgs.shape}")
    if imgs.device.type != "cuda":
        raise ValueError("Images must be on CUDA device")

    B, _, H, W = imgs.shape
    input_is_uint8 = imgs.dtype == torch.uint8

    # Convert to contiguous float32 if needed
    if not input_is_uint8:
        imgs = imgs.float().contiguous()
    else:
        imgs = imgs.contiguous()

    # Clip boxes to image bounds
    if target is not None:
        target = _clip_box_to_image(target, H, W)
    if background is not None:
        background = _clip_box_to_image(background, H, W)

    # Prepare output buffers
    out_min = torch.full((B,), float("inf"), dtype=torch.float32, device=imgs.device)
    out_max = torch.full((B,), float("-inf"), dtype=torch.float32, device=imgs.device)
    out_sum_t = torch.zeros((B,), dtype=torch.float32, device=imgs.device)
    out_sum_b = torch.zeros((B,), dtype=torch.float32, device=imgs.device)
    out_cnt_t = torch.zeros((B,), dtype=torch.int32, device=imgs.device)
    out_cnt_b = torch.zeros((B,), dtype=torch.int32, device=imgs.device)

    # Default boxes if not provided (empty ROI)
    t_x0, t_y0, t_x1, t_y1 = (
        (0, 0, 0, 0) if target is None else (target.x0, target.y0, target.x1, target.y1)
    )
    b_x0, b_y0, b_x1, b_y1 = (
        (0, 0, 0, 0)
        if background is None
        else (background.x0, background.y0, background.x1, background.y1)
    )

    # Launch kernel
    def grid(meta):
        return (
            triton.cdiv(W, meta["BLOCK_W"]),
            triton.cdiv(H, meta["BLOCK_H"]),
            B,
        )

    _contrast_atomic_kernel[grid](  # type: ignore[index]
        imgs,
        out_min,
        out_max,
        out_sum_t,
        out_sum_b,
        out_cnt_t,
        out_cnt_b,
        H,
        W,
        t_x0,
        t_y0,
        t_x1,
        t_y1,
        b_x0,
        b_y0,
        b_x1,
        b_y1,
        INPUT_IS_UINT8=input_is_uint8,
        LINEARIZE_SRGB=linearize_srgb,
    )

    # Compute Michelson contrast
    michelson = (out_max - out_min) / (out_max + out_min + 1e-8)

    # Compute Weber contrast if ROIs provided
    weber = None
    if target is not None and background is not None:
        mean_t = out_sum_t / (out_cnt_t.float() + 1e-8)
        mean_b = out_sum_b / (out_cnt_b.float() + 1e-8)
        weber = (mean_t - mean_b) / (mean_b + 1e-8)

    return michelson, weber


# -----------------------------
# CLI
# -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Triton GPU contrast metrics (Michelson + Weber)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("images", nargs="+", help="Image paths")
    parser.add_argument("--target", type=str, help="Target ROI: x0,y0,x1,y1")
    parser.add_argument("--background", type=str, help="Background ROI: x0,y0,x1,y1")
    parser.add_argument(
        "--linearize-srgb", action="store_true", help="sRGB->linear before luminance"
    )
    parser.add_argument("--bench", action="store_true", help="Benchmark mode")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    args = parser.parse_args()

    if read_image is None:
        print("ERROR: torchvision.io.read_image not available", file=sys.stderr)
        print("Install GPU deps: uv sync --group gpu", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1

    # Parse ROI boxes
    target_box = _parse_box(args.target) if args.target else None
    background_box = _parse_box(args.background) if args.background else None

    if (target_box is None) != (background_box is None):
        print("ERROR: Both --target and --background required for Weber contrast", file=sys.stderr)
        return 1

    # Load images
    print(f"Loading {len(args.images)} image(s)...")
    cpu_images = []
    for path in args.images:
        if not os.path.exists(path):
            print(f"ERROR: Image not found: {path}", file=sys.stderr)
            return 1
        img = read_image(path, mode=ImageReadMode.RGB)  # type: ignore[union-attr]
        cpu_images.append(img)

    # Group by shape for batching
    from collections import defaultdict

    shape_groups = defaultdict(list)
    for i, img in enumerate(cpu_images):
        shape_groups[img.shape].append((i, img))

    # Process each shape group
    all_michelson = [None] * len(args.images)
    all_weber = [None] * len(args.images)

    for shape, group in shape_groups.items():
        indices, imgs = zip(*group)
        batch = torch.stack(imgs, dim=0).cuda()

        if args.bench:
            # Warmup
            for _ in range(3):
                michelson_weber_triton(batch, target_box, background_box, args.linearize_srgb)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(args.iters):
                michelson_weber_triton(batch, target_box, background_box, args.linearize_srgb)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(
                f"Batch shape {shape}: {elapsed / args.iters * 1000:.3f} ms/iter (avg over {args.iters})"
            )

        # Compute final results
        michelson, weber = michelson_weber_triton(
            batch, target_box, background_box, args.linearize_srgb
        )

        for i, idx in enumerate(indices):
            all_michelson[idx] = michelson[i].item()
            if weber is not None:
                all_weber[idx] = weber[i].item()

    # Print results
    print("\nResults:")
    for i, path in enumerate(args.images):
        print(f"{path}:")
        print(f"  Michelson: {all_michelson[i]:.6f}")
        if all_weber[i] is not None:
            print(f"  Weber: {all_weber[i]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
