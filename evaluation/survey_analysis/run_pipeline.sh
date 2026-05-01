#!/usr/bin/env bash
# End-to-end runner for the PHOENIX evaluation pipeline.
#
# Usage:
#   bash evaluation/survey_analysis/run_pipeline.sh [--mode pseudo|real]
#                                                   [--n-runs N]
#                                                   [--cases C01 C02 ...]
#                                                   [--parts part1 ...]
#
# Defaults: --mode pseudo --n-runs 5 (all cases, all parts).
#
# Real mode requires OPENROUTER_API_KEY in the environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo "PHOENIX evaluation pipeline"
echo "Working dir: ${SCRIPT_DIR}"
echo "Python:      $(${PYTHON} --version 2>&1)"
echo "============================================================"

# Forward all args to the orchestrator unchanged.
${PYTHON} -u pipeline.py "$@"
