#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
export UV_CACHE_DIR="$task_root/.cache/uv"
export HF_HOME="$task_root/.cache/huggingface"
export PIP_CACHE_DIR="$task_root/.cache/pip"
export XDG_CACHE_HOME="$task_root/.cache"
"$uv_bin" venv --python 3.11.17 "$task_root/.venv"
"$uv_bin" pip install --python "$task_root/.venv/bin/python" --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match -r "$task_root/requirements.lock.txt"
"$uv_bin" venv --python 3.10.22 "$task_root/.venv-louvain"
"$uv_bin" pip install --python "$task_root/.venv-louvain/bin/python" -r "$task_root/requirements-louvain.lock.txt"
echo "Environments ready under $task_root"
