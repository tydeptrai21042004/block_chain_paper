#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs_lenet"
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

python3 "$ROOT/generate_lenet_vectors.py" --check

export RUSTFLAGS="${RUSTFLAGS:-} -C passes=lower-atomic"
TARGET="riscv64imac-unknown-none-elf"

run_interval() {
  local i="$1"
  local j="$2"
  local tag="lenet_${i}_${j}"
  echo "==> $tag"
  BENCH_KIND="lenet" BENCH_START="$i" BENCH_END="$j" cargo build \
    --manifest-path "$ROOT/Cargo.toml" --release --target "$TARGET" >/dev/null
  local bin="$ROOT/target/$TARGET/release/cellvg-ckb-bench"
  ckb-debugger --bin "$bin" 2>&1 | tee "$LOG_DIR/$tag.log"
}

# State-chained publication cost set: every consecutive interval of
# spans 1--4. Each binary starts from the exact generated S_i and verifies S_j. This ensures fixed g=2 and g=4 baselines can settle every
# interval at or below their advertised terminal granularity.
for span in 1 2 3 4; do
  for ((i=0; i+span<=12; i++)); do
    run_interval "$i" "$((i+span))"
  done
done

python3 "$ROOT/parse_lenet_logs.py" --logs "$LOG_DIR" \
  --out "$ROOT/lenet_interval_measurements.csv"
echo "Measurements: $ROOT/lenet_interval_measurements.csv"
