# Test report — V5 code/result revision

Validation completed for this revision:

- `python -m compileall -q experiments ckb_bench`: passed;
- `python -m pytest -q experiments/tests`: **84 passed + 17 subtests**;
- original synthetic end-to-end experiment: passed;
- synthetic three-trace campaign: passed;
- corrected Merkle generator: **13/13** canonical proofs reconstruct one known
  depth-4 root;
- state-chained LeNet generator: all **42** span-1..4 intervals reproduce their
  exact end checkpoints;
- added convolution-heavy generator: all **42** span-1..4 intervals reproduce
  exact end checkpoints;
- added GEMM-heavy generator: all **42** span-1..4 intervals reproduce exact end
  checkpoints;
- generic interval-log parser: strict completeness/duplicate/cycle parsing tests
  passed;
- Pareto instrumentation regression: enabling stats leaves the selected policy
  and both objectives unchanged;
- weighted-mean reporting matches the Pareto objective under non-uniform weights;
- deterministic bounded-noise models preserve exact Pareto/scalar minimax
  recovery in tests.

## What the V5 tests add

In addition to the original scalar/Pareto Bellman, round-budget, baseline,
Merkle, I/O, and parser tests, V5 specifically checks:

- query-cost scaling without terminal-cost mutation;
- uniform/front/back/cost-proportional fault-weight normalization;
- weighted evaluation consistency;
- optional Pareto candidate/pruning/frontier statistics;
- deterministic perturbation for measurement-noise robustness;
- both new state-chained trace generators;
- generic trace-log parsing.

## Environment limitation

The CKB/Rust benchmark binary was **not compiled or executed in this container**
because `cargo` / the RISC-V CKB toolchain is unavailable here. `run_tests.sh`
retains a conditional real RISC-V build guard, and the real measurement scripts
require `ckb-debugger`.

No real cycle numbers were fabricated. All precomputed V5 campaign,
sensitivity, robustness, and scalability files included in this archive are
explicitly synthetic software/scaling evidence and are marked **NOT FOR
MANUSCRIPT**.
