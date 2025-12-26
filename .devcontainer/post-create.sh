#!/usr/bin/env bash
set -euo pipefail

echo "== uv =="
uv --version

echo
echo "== syncing (all groups, includes GPU) =="
# Compile bytecode for faster startup in container-like workflows
uv sync --all-groups --compile-bytecode

echo
echo "== environment doctor =="
bash scripts/doctor.sh || true

echo
echo "Devcontainer ready."
