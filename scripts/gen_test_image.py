#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from torchvision.io import write_png
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "torchvision is required for this helper.\n"
        "Install GPU deps: uv sync --group gpu\n"
        f"Import error: {e}"
    ) from e


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a simple PNG for smoke tests.")
    ap.add_argument("--out", type=Path, default=Path("frame.png"))
    ap.add_argument("--size", type=int, default=720)
    args = ap.parse_args()

    h = w = int(args.size)
    # Simple gradient + noise, uint8 [3,H,W]
    yy = torch.linspace(0, 255, h, dtype=torch.uint8).unsqueeze(1).expand(h, w)
    xx = torch.linspace(0, 255, w, dtype=torch.uint8).unsqueeze(0).expand(h, w)
    r = xx
    g = yy
    b = ((xx.to(torch.int16) + yy.to(torch.int16)) // 2).to(torch.uint8)
    img = torch.stack([r, g, b], dim=0).contiguous()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_png(img, str(args.out))
    print(f"Wrote {args.out} ({h}x{w})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
