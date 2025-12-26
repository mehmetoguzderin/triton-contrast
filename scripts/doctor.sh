#!/usr/bin/env bash
set -euo pipefail

echo "== python =="
python --version || true

echo
echo "== uv =="
uv --version || true

echo
echo "== nvidia-smi =="
nvidia-smi || echo "nvidia-smi not available (no GPU exposed?)"

echo
echo "== torch cuda =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
