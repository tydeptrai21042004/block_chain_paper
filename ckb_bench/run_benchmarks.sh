#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log

if ! command -v cargo >/dev/null 2>&1; then
  echo "ERROR: cargo is required." >&2
  exit 1
fi
if ! command -v ckb-debugger >/dev/null 2>&1; then
  echo "ERROR: ckb-debugger is required. See ckb_bench/README.md." >&2
  exit 1
fi

# The modern ckb-std build path uses Rust atomic lowering for CKB-VM compatibility.
export RUSTFLAGS="${RUSTFLAGS:-} -C passes=lower-atomic"
TARGET="riscv64imac-unknown-none-elf"

run_one() {
  local kind="$1"
  local size="$2"
  local tag="${kind}_${size}"
  echo "==> $tag"
  BENCH_KIND="$kind" BENCH_SIZE="$size" cargo build \
    --manifest-path "$ROOT/Cargo.toml" --release --target "$TARGET" >/dev/null
  local bin="$ROOT/target/$TARGET/release/cellvg-ckb-bench"
  ckb-debugger --bin "$bin" 2>&1 | tee "$LOG_DIR/$tag.log"
}

for n in 16 64 256 1024; do run_one relu "$n"; done
for n in 16 32 64 128 256; do run_one dot "$n"; done
for n in 8 16 24 32; do run_one gemm "$n"; done
for n in 8 16 24 32; do run_one conv "$n"; done
for n in 3 4 5 6 7 8; do run_one merkle "$n"; done

python3 "$ROOT/parse_logs.py" --logs "$LOG_DIR" --out "$ROOT/ckb_primitive_measurements.csv"
echo "Measurements: $ROOT/ckb_primitive_measurements.csv"
