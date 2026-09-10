#!/usr/bin/env bash
# Submit the full attribution sweep to Slurm with submitit.
#
# run_attribution.py's __main__ calls submitit_main(), which builds a hardcoded
# sweep over tasks, seeds, attribution methods, and input types. It ignores
# command-line arguments, so this wrapper takes none.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

uv run src/attributing/run_attribution.py
