#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE="${1:-lenet}"
MAX_SPAN="${MAX_SPAN:-4}"
N="${TRACE_N:-12}"

case "$TRACE" in
  lenet|conv_heavy|gemm_heavy) ;;
  *) echo "ERROR: trace must be one of: lenet, conv_heavy, gemm_heavy" >&2; exit 2 ;;
esac

LOG_DIR="$ROOT/logs_${TRACE}"
OUT_CSV="$ROOT/${TRACE}_interval_measurements.csv"
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

# Deterministically regenerate/check all vector files before compiling.
python3 "$ROOT/generate_lenet_vectors.py" --check
python3 "$ROOT/generate_trace_vectors.py"

export RUSTFLAGS="${RUSTFLAGS:-} -C passes=lower-atomic"
TARGET="riscv64imac-unknown-none-elf"

run_interval() {
  local i="$1"
  local j="$2"
  local tag="${TRACE}_${i}_${j}"
  echo "==> $tag"
  if [[ "$TRACE" == "lenet" ]]; then
    BENCH_KIND="lenet" BENCH_START="$i" BENCH_END="$j" cargo build \
      --manifest-path "$ROOT/Cargo.toml" --release --target "$TARGET" >/dev/null
  else
    BENCH_KIND="trace" BENCH_TRACE="$TRACE" BENCH_START="$i" BENCH_END="$j" cargo build \
      --manifest-path "$ROOT/Cargo.toml" --release --target "$TARGET" >/dev/null
  fi
  local bin="$ROOT/target/$TARGET/release/cellvg-ckb-bench"
  ckb-debugger --bin "$bin" 2>&1 | tee "$LOG_DIR/$tag.log"
}

for ((span=1; span<=MAX_SPAN; span++)); do
  for ((i=0; i+span<=N; i++)); do
    run_interval "$i" "$((i+span))"
  done
done

python3 "$ROOT/parse_trace_logs.py" --logs "$LOG_DIR" --trace "$TRACE" \
  --n "$N" --max-span "$MAX_SPAN" --out "$OUT_CSV"
echo "Measurements: $OUT_CSV"
