#!/usr/bin/env bash
# Start (or continue) training. Extra arguments go to python -m adv.train, e.g.
#   bash run.sh --resume      bash run.sh --dry-run --steps 15      bash run.sh --max-hours 6
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run setup.sh first."; exit 1; }
export PATH="$ROOT/tools/node/bin:$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT/src" HF_HOME="$ROOT/tools/hf-home" PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
echo "Training. Watch log.txt and train.png in $ROOT (they refresh every step). Ctrl+C stops safely."
exec "$PY" -m adv.train "$@"
