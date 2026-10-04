#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 -m compileall -q experiments ckb_bench
python3 ckb_bench/generate_merkle_vectors.py --check
python3 ckb_bench/generate_lenet_vectors.py --check

# Regression guard for the real no_std CKB contract build.  The test is
# executed whenever Cargo and the RISC-V target are available (as they are in
# the paper/Kaggle reproduction pipeline).  Developer environments without
# the target can still run the Python/unit-test suite.
if command -v cargo >/dev/null 2>&1 && command -v rustup >/dev/null 2>&1 && \
   rustup target list --installed | grep -qx 'riscv64imac-unknown-none-elf'; then
  echo "[TEST] CKB RISC-V no_std contract build"
  export RUSTFLAGS="${RUSTFLAGS:-} -C passes=lower-atomic"
  BENCH_KIND="relu" BENCH_SIZE="16" cargo build \
    --manifest-path ckb_bench/Cargo.toml \
    --release \
    --target riscv64imac-unknown-none-elf
  test -x ckb_bench/target/riscv64imac-unknown-none-elf/release/cellvg-ckb-bench
  echo "[PASS] CKB RISC-V no_std contract build"
else
  echo "[SKIP] CKB RISC-V build regression test (Cargo/rustup/target unavailable)."
fi
python3 -m unittest discover -s experiments/tests -v

TMP_OUT="${TMPDIR:-/tmp}/cellvg_hndt_demo_smoke_$$"
trap 'rm -rf "$TMP_OUT"' EXIT
python3 experiments/run_experiment.py --demo --out "$TMP_OUT" >/dev/null

echo "All unit tests and the synthetic integration smoke test passed."
echo "The smoke test uses synthetic data only and does not modify manuscript result tables."
