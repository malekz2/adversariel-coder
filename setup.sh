#!/usr/bin/env bash
# One-time setup for Linux or macOS (no sudo needed). Usage:
#   bash setup.sh [--models-dir DIR] [--no-models] [--skip-selftest]
# Everything goes into ./tools and ./.venv next to this script.
set -euo pipefail

UV_VER=0.13.1; NODE_VER=v24.22.0; LLAMA=b11700; PI_VER=1.3.0; TORCH=2.14.1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$ROOT/tools"; DL="$TOOLS/_downloads"
MODELS_DIR=""; NO_MODELS=0; SKIP_SELFTEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --no-models) NO_MODELS=1; shift ;;
    --skip-selftest) SKIP_SELFTEST=1; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
mkdir -p "$TOOLS" "$DL" "$ROOT/state"
FREE_GB=$(df -Pk "$ROOT" | awk 'NR==2 {printf "%.1f", $4/1048576}')
echo "free space for $ROOT: $FREE_GB GB"
if awk "BEGIN {exit !($FREE_GB < 5)}"; then
  echo "Only $FREE_GB GB free. Setup needs ~3-4 GB for tools plus the models (~30 GB for K2-Horizon-7B, 9-13 GB for the smaller CPU profiles). Free space or clone elsewhere."; exit 1
fi
say() { printf '\n\033[36m==> %s\033[0m\n' "$*"; }
fetch() { [ -f "$DL/$2" ] || { echo "    downloading $1"; curl -fL --retry 3 -o "$DL/$2.part" "$1"; mv "$DL/$2.part" "$DL/$2"; }; }

OS=$(uname -s); ARCH=$(uname -m)
case "$OS-$ARCH" in
  Linux-x86_64)  UV_T=x86_64-unknown-linux-gnu;  NODE_T=linux-x64;    LL_T=ubuntu-x64 ;;
  Linux-aarch64) UV_T=aarch64-unknown-linux-gnu; NODE_T=linux-arm64;  LL_T=ubuntu-arm64 ;;
  Darwin-arm64)  UV_T=aarch64-apple-darwin;      NODE_T=darwin-arm64; LL_T=macos-arm64 ;;
  Darwin-x86_64) UV_T=x86_64-apple-darwin;       NODE_T=darwin-x64;   LL_T=macos-x64 ;;
  *) echo "unsupported platform $OS-$ARCH"; exit 1 ;;
esac
HAS_NVIDIA=0; command -v nvidia-smi >/dev/null 2>&1 && HAS_NVIDIA=1
[ "$HAS_NVIDIA" = 1 ] && [ "$OS" = Linux ] && [ "$ARCH" = x86_64 ] && LL_T=ubuntu-cuda-12.8-x64

export UV_PYTHON_INSTALL_DIR="$TOOLS/python" UV_CACHE_DIR="$TOOLS/uv-cache"   # uv falls back to copying on exFAT by itself
export HF_HOME="$TOOLS/hf-home" HF_HUB_DISABLE_SYMLINKS_WARNING=1

say "1/7 uv and Python 3.12"
if [ ! -x "$TOOLS/uv/uv" ]; then
  fetch "https://github.com/astral-sh/uv/releases/download/$UV_VER/uv-$UV_T.tar.gz" uv.tgz
  mkdir -p "$TOOLS/uv" && tar -xzf "$DL/uv.tgz" -C "$TOOLS/uv" --strip-components=1
fi
UV="$TOOLS/uv/uv"; PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || "$UV" venv "$ROOT/.venv" --python 3.12 --managed-python

say "2/7 Python packages"
# explicit backend: "auto" also detects Intel/AMD GPUs and would pick a multi-GB XPU/ROCm build
TORCH_BACKEND=cpu; [ "$HAS_NVIDIA" = 1 ] && TORCH_BACKEND=auto
echo "    torch backend: $TORCH_BACKEND"
"$UV" pip install --python "$PY" "torch==$TORCH" --torch-backend "$TORCH_BACKEND"
"$UV" pip install --python "$PY" -r "$ROOT/requirements.txt"
if [ "$OS-$ARCH" = Darwin-arm64 ]; then
  echo "    Apple Silicon: installing MLX (mac_mlx profile: MiniCPM5 Maker + K2-Horizon Solver)"
  "$UV" pip install --python "$PY" mlx==0.32.4 mlx-lm==0.32.2
fi
if [ "$HAS_NVIDIA" = 1 ]; then
  "$UV" pip install --python "$PY" bitsandbytes==0.50.2
  "$UV" pip install --python "$PY" flash-linear-attention || echo "    (optional fast kernels skipped)"
fi
export PYTHONPATH="$ROOT/src"
"$PY" -m adv.hardware --write "$ROOT/state/hardware.json"
PROFILE=$("$PY" -c "import json;print(json.load(open('$ROOT/state/hardware.json'))['profile'])")
echo "    hardware profile: $PROFILE"

say "3/7 Node.js $NODE_VER and Pi $PI_VER"
if [ ! -x "$TOOLS/node/bin/node" ]; then
  fetch "https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-$NODE_T.tar.gz" node.tgz
  mkdir -p "$TOOLS/node" && tar -xzf "$DL/node.tgz" -C "$TOOLS/node" --strip-components=1
fi
export PATH="$TOOLS/node/bin:$PATH" npm_config_cache="$TOOLS/npm-cache"
npm install --prefix "$TOOLS/pi" --no-fund --no-audit "@earendil-works/pi-coding-agent@$PI_VER"

say "4/7 bash: using the system shell (PortableGit is only needed on Windows)"

say "5/7 llama.cpp $LLAMA ($LL_T) + LoRA converter"
if [ -z "$(find "$TOOLS/llama" -name llama-server -type f 2>/dev/null | head -1)" ]; then
  fetch "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA/llama-$LLAMA-bin-$LL_T.tar.gz" llama.tgz
  mkdir -p "$TOOLS/llama" && tar -xzf "$DL/llama.tgz" -C "$TOOLS/llama"
fi
if [ -z "$(find "$TOOLS/llama.cpp-src" -name convert_lora_to_gguf.py 2>/dev/null | head -1)" ]; then
  fetch "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$LLAMA.zip" llama-src.zip
  mkdir -p "$TOOLS/llama.cpp-src"
  P="llama.cpp-$LLAMA"
  unzip -q -o "$DL/llama-src.zip" "$P/convert_lora_to_gguf.py" "$P/convert_hf_to_gguf.py" "$P/gguf-py/*" "$P/conversion/*" -d "$TOOLS/llama.cpp-src"
fi

if [ "$NO_MODELS" = 0 ]; then
  say "6/7 models for profile '$PROFILE'"
  if [ -z "$MODELS_DIR" ]; then
    read -r -p "    Folder for models (Enter = $ROOT/models): " MODELS_DIR || true
    MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
  fi
  "$PY" -m adv.download --profile "$PROFILE" --models-dir "$MODELS_DIR" --yes
else
  say "6/7 models skipped (--no-models)"
fi

if [ "$NO_MODELS" = 0 ] && [ "$SKIP_SELFTEST" = 0 ]; then
  say "7/7 self-test (llama.cpp LoRA + d3-flash)"
  "$PY" -m adv.selftest --profile "$PROFILE"
else
  say "7/7 self-test skipped"
fi
rm -rf "$DL"
"$UV" cache clean >/dev/null 2>&1 || true   # installed packages are copies; the cache only wastes space
say "Setup finished. Next:  bash run.sh --dry-run --steps 15   then   bash run.sh"
