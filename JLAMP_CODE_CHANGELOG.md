# JLAMP-oriented code revision

This revision strengthens the existing method without changing its stop-or-split
research idea.

## Core numerical/algorithmic changes

- Pareto objective arithmetic is exact rational arithmetic (`Fraction`) rather
  than tolerance-dependent floating-point dominance.
- Root/subinterval second objective is documented as weighted expected-cost
  contribution (expectation mass); it equals expected cost at the root.
- 2-D Pareto pruning uses sorting plus a one-pass antichain scan (`O(C log C)`
  for `C` interval candidates) instead of quadratic all-pairs dominance tests.
- Restriction hooks were added to the Pareto solver for fair mechanism ablation
  without duplicating the solver.

## Fair ablations

- Pareto adaptive-split + atomic-stop.
- Pareto midpoint-split + adaptive-stop.
- Pareto midpoint-split + atomic-stop.
- Legacy scalar ablations are retained so old tables/scripts still run.

## Paper-backed baselines

- RDoC binary-search localization (Canetti--Riva--Rothblum, 2013).
- TrueBit binary verification-game localization.
- Existing Arbitrum, opML, Agatha, Kirkpatrick--Klawe, Hu--Tucker, and gated
  zk-OPML adapters are retained.
- Height-limited alphabetic weighted-path baseline (Larmore--Przytycka, 1994),
  default depth `L=4`, configurable with `--alphabetic-height-limit`.

Close midpoint/binary-search system adaptations are deduplicated in the main
summary instead of presented as falsely independent numerical methods.

## Theorem-driven regression checks

- Exact decimal/rational arithmetic check.
- Sorted Pareto pruning vs the quadratic dominance definition on random labels.
- Ancestor-slack counterexample: a child label that is locally worse in minimax
  cost can give strictly lower parent expected cost without changing parent
  worst-case cost, proving that one local lexicographic label is insufficient.
- Height-budget compliance and infeasibility checks.
- Fair Pareto-ablation action-space checks.

## Validation

- `89/89` Python unit tests pass.
- `bash run_tests.sh` passes all available unit, vector-generation, and synthetic
  integration/campaign smoke tests.
- The environment-dependent CKB RISC-V build regression remains skipped when the
  Cargo/rustup/target toolchain is unavailable; no result is fabricated.
