# V5 code/result revision — change manifest

This revision is based directly on the uploaded `block_chain_paper-main(6).zip` and keeps the Pareto-HNDT Bellman objective and minimax-safe policy rule unchanged.

## Regression baseline

- Original repository: `75 passed + 17 subtests`.
- V5 repository: `84 passed + 17 subtests`.
- Existing LeNet/default experiment semantics remain the reference case: uniform fault weights and query scale 1.0.
- No real CKB/zkVM cost has been synthesized, interpolated, or substituted for missing measurements.

## Core code changes

### Multi-trace real-measurement support

Added two deterministic 12-transition ordered traces in addition to the original LeNet trace:

- `conv_heavy`
- `gemm_heavy`

All three use 13 checkpoints and support genuine span-1 through span-4 interval execution (42 intervals per trace). New tooling:

- `ckb_bench/generate_trace_vectors.py`
- `ckb_bench/run_trace_intervals.sh`
- `ckb_bench/parse_trace_logs.py`
- `ckb_bench/src/extra_trace_vectors.rs`
- `experiments/data/traces/*.csv`

The Rust build uses compile-time trace selection so the original LeNet binary does not silently include the extra trace tables. `build.rs` now tracks `BENCH_TRACE` to prevent stale trace binaries.

### Pareto instrumentation without policy modification

`experiments/hndt/pareto.py` now optionally records:

- generated labels;
- retained labels;
- duplicate and dominance pruning;
- root/peak frontier size;
- peak candidate count per interval;
- pruning ratio.

Instrumentation is optional and regression-tested not to change the selected policy or objective values.

### Sensitivity and fairness support

Added reusable support for:

- query-cost scaling;
- uniform/front/back/cost-derived fault-weight profiles;
- deterministic bounded cost perturbation;
- weighted mean-cost and weighted-round reporting.

The same fault weights are passed consistently to Pareto evaluation and weight-sensitive baselines such as the Hu-Tucker objective adaptation.

### New experiment drivers

- `experiments/run_campaign.py` — multi-trace campaign.
- `experiments/run_sensitivity.py` — fixed query-cost/fault-weight grid.
- `experiments/run_robustness.py` — bounded measurement-noise study.
- `experiments/run_scalability.py` — solver runtime/frontier/pruning scaling.
- `experiments/run_v5_suite.py` — one-command orchestration.

`run_v5_suite.py --mode real` refuses campaign entries marked synthetic, preventing demo data from silently entering a paper run.

### Auditing/reproducibility

- Added Pareto policy audit output.
- Expanded run metadata.
- Added real-campaign example configuration.
- Added trace CSVs and self-checking vector generators.
- Updated Linux/PowerShell test runners.

## Result files bundled in this archive

`experiments/results/v5_suite_demo/` is a precomputed **synthetic software-validation suite only**. It is intentionally labelled `NOT FOR MANUSCRIPT` and demonstrates that the new pipeline executes correctly across different cost structures.

No new real CKB result is claimed until `run_trace_intervals.sh` is executed with `ckb-debugger` for the new traces and those measured CSVs are supplied to the real campaign config.

## Environment limitation during this revision

The Python/test/vector pipeline was fully exercised in the build environment. Cargo was not available, so the modified Rust CKB target could not be compiled here. The repository retains the normal Rust preflight/build path for a machine with the CKB/Rust toolchain installed.
