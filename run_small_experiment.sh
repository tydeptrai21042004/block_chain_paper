#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-real}"
if [[ "$MODE" == "demo" ]]; then
  python3 experiments/run_experiment.py --demo --out experiments/results/demo
  python3 experiments/plot_results.py --results experiments/results/demo --out figures/demo
  echo "Demo completed. IMPORTANT: demo numbers are synthetic and must not be used in the manuscript."
elif [[ "$MODE" == "real" ]]; then
  python3 experiments/run_experiment.py --out experiments/results/current
  python3 experiments/plot_results.py --results experiments/results/current --out figures
  echo "Measured experiment completed. Review raw CKB logs before copying values into the manuscript."
else
  echo "Usage: ./run_small_experiment.sh [real|demo]" >&2
  exit 2
fi
