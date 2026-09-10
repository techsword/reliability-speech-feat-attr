#!/usr/bin/env bash
# Launch a fine-tuning run with uv.
#
# All arguments are forwarded to run_finetune.py, for example:
#   scripts/run_finetune.sh --seed 42 --model_type wav2vec2 \
#       --taskname iemocap --num_epochs 5 --overwrite_output_dir
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

uv run src/finetuning/run_finetune.py "$@"
