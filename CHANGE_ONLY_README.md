# Change-only patch for `block_chain_paper-main (1)(3).zip`

This archive contains only files that must be added/replaced to make the uploaded
`block_chain_paper-main (1)(3).zip` consistent with the V5 code/results plan.

## Comparison finding

The uploaded repository already contained the V5 documentation/manifests and many
of the trace/CKB support changes, but the core V5 experiment implementation was
partially missing/reverted. In particular, weighted evaluation, Pareto solver
instrumentation, sensitivity/robustness/scalability runners, campaign configs,
and their regression tests were absent or older.

## What this patch restores/adds

- weighted fault-path reporting with uniform/front/back/cost profiles;
- query-cost scaling without changing the Pareto-HNDT Bellman recurrence;
- optional Pareto solver statistics (candidate counts, pruning, peak frontier);
- Pareto policy audit output;
- three-trace campaign configuration (LeNet, conv-heavy, GEMM-heavy);
- deterministic synthetic demo inputs clearly marked NOT FOR MANUSCRIPT;
- sensitivity runner;
- bounded measurement-noise robustness runner;
- solver-scalability runner;
- one-command V5 suite;
- generic trace-log parser regression tests and V5 extension tests.

## Validation

- Uploaded base: 75 Python tests passed.
- Patched repository: 84 Python tests passed.
- V5 demo suite completed for all three traces plus sensitivity, robustness, and
  scalability checks at n=12,24,48.
- Synthetic demo outputs are software checks only and must not be reported as
  CKB manuscript evidence.

## Apply

Copy the contents of the `block_chain_paper-main/` folder over the repository of
the same name, preserving directories and replacing files when prompted.

The patch does **not** contain generated `v5_suite_demo` result folders or cache
files; regenerate outputs from the supplied runners.
