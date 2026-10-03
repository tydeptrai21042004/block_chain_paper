#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 -m compileall -q experiments ckb_bench
python3 ckb_bench/generate_merkle_vectors.py --check
python3 ckb_bench/generate_lenet_vectors.py --check
python3 -m unittest discover -s experiments/tests -v

TMP_OUT="${TMPDIR:-/tmp}/cellvg_hndt_demo_smoke_$$"
trap 'rm -rf "$TMP_OUT"' EXIT
python3 experiments/run_experiment.py --demo --out "$TMP_OUT" >/dev/null

echo "All unit tests and the synthetic integration smoke test passed."
echo "The smoke test uses synthetic data only and does not modify manuscript result tables."
