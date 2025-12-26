#!/usr/bin/env bash
set -euo pipefail

# Example:
#   ./scripts/bench.sh ./frame.png 50

IMG="${1:-frame.png}"
ITERS="${2:-50}"

uv sync --group gpu

python triton_contrast_cli.py "${IMG}" --bench --iters "${ITERS}"
