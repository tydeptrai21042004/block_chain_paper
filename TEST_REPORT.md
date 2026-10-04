# Test report

Validation completed for this revision:

- `python -m compileall -q experiments ckb_bench`: passed;
- `python -m unittest discover -s experiments/tests -v`: **75/75 passed**;
- synthetic end-to-end experiment: passed;
- corrected Merkle generator: **13/13** canonical proofs reconstruct one known
  depth-4 root;
- state-chained LeNet generator: all **42** span-1..4 intervals reproduce their
  exact end checkpoints;
- local Markdown link regression test: passed.

## What the 75 tests cover

- exact scalar-HNDT Bellman optimality, including exhaustive small instances;
- exact Pareto-HNDT frontier correctness against exhaustive strategy
  enumeration;
- recovery of the scalar minimax optimum by Pareto-HNDT;
- mean-cost non-regression among minimax-optimal policies;
- nondominance of retained Pareto labels;
- exact round-budgeted HNDT and recovery of the unconstrained optimum;
- Arbitrum/opML/Agatha adapter fidelity and policy equivalence on an ordered
  trace;
- Kirkpatrick--Klawe alphabetic-minimax objective reduction;
- Hu--Tucker weighted-path objective reduction;
- strict zk-OPML atomic-ZK completeness gating;
- adaptive-split, adaptive-stop, fixed-granularity, and heterogeneity
  ablations;
- canonical Merkle paths and generated Rust-vector freshness;
- state-chained neural checkpoints and all 42 interval vectors;
- policy traversal and per-fault evaluation;
- trace/query/interval/config validation and CKB log parsers;
- documentation links.

## Environment limitation

The CKB/Rust benchmark binary was **not compiled or executed in this container**
because the required RISC-V Rust target / `ckb-debugger` toolchain is not
available here. `run_tests.sh` retains a conditional real RISC-V build guard,
and `KAGGLE_ONE_CELL.py` makes that build a mandatory preflight before real
cycle measurement.

No real cycle numbers were fabricated. The bundled demo data are synthetic and
must not be used as manuscript results.
