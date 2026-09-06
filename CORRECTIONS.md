# Corrections in this revision

## Code

- Fixed root-level unit-test imports (`python -m unittest discover -s experiments/tests -v` now works directly).
- Added validation for model size, intervals, terminal costs, query costs, split indices, policies, Merkle configuration, duplicate CSV rows, span mismatches, and negative/NaN costs.
- Added numerical tie tolerance to HNDT while preserving settlement-on-tie behavior.
- Made split tie-breaking deterministic: fewer rounds, closer to midpoint, then smaller index.
- Made the `Operator midpoint + ZK` baseline strict: it is included only if every atomic interval has a finite zkVM measurement.
- Corrected summary statistics so leaf-size/backend counts use unique terminal intervals rather than weighting leaves by the number of fault positions they contain.
- Added `fault_position = fault_index + 1` to exported fault CSVs to match manuscript notation.
- Strengthened homogenization: preserve the measured interval residual and fail if the reconstructed cost becomes negative instead of silently clipping it.
- Made CKB parsers strict by default and validated the complete 42-interval LeNet measurement set.
- Cleared stale benchmark logs before new runs.
- Added `ckb_bench/build.rs` so Cargo rebuilds when compile-time benchmark-selection environment variables change.

## Tests

The project now contains **46 Python unit tests**, covering:

- Bellman optimality examples and theorem-level monotonicity;
- homogeneous bisection and non-midpoint optima;
- direct settlement and unavailable queries;
- exhaustive enumeration on many random instances with `n <= 5`;
- fixed-granularity and midpoint baselines;
- strict zkVM baseline availability;
- fault-path traversal and policy validation;
- summary statistics;
- trace, interval-cost, query-cost, and config validation;
- CKB debugger log parsing and 42-interval completeness;
- heterogeneity-ablation reconstruction.

`run_tests.sh` / `run_tests.ps1` also execute a synthetic end-to-end integration smoke test after the unit suite.

## Manuscript hierarchy

The five main sections are unchanged. Numbered subsection count was reduced to:

- Section 2: 3 subsections;
- Section 3: 3 subsections;
- Section 4: 4 subsections.

All former `subsubsection` headings and most small subsection headings are now simple bold lead-ins. Detailed theorem proofs and all blank experimental forms are retained.
