# Corrected repository changelog

This revision upgrades the proposal from scalar HNDT to **Pareto-HNDT** while
keeping the original solver intact as a regression oracle.

## Proposal-method changes

- Added exact two-objective Pareto dynamic programming over the unchanged
  stop-or-split action space.
- Primary selection preserves the scalar minimax optimum exactly and minimizes
  uniform-mean fault-path cost among all minimax-optimal policies.
- Added exact frontier export, policy reconstruction, dominance pruning, and a
  mean-first frontier endpoint for ablation.

## Baseline changes

- Retained Arbitrum-IVP, opML Phase-1, and Agatha-GPP chain adapters.
- Added Kirkpatrick--Klawe alphabetic-minimax objective adaptation.
- Added Hu--Tucker weighted-path objective adaptation.
- Added conditional zk-OPML operator-dispute adapter that refuses to run when
  any required atomic ZK cost is missing.
- Close midpoint-system adapters are deduplicated only in the main summary;
  individual paper provenance remains available.

## Ablation/output changes

- Added scalar-HNDT secondary-objective ablation.
- Added Pareto mean-first endpoint.
- Expanded mechanism isolation to include the new mean-refinement dimension.
- Added `pareto_frontier.csv`, `proposal_gain.csv`, and
  `proposal_ablation.csv`.
- Kept fixed-`g`, heterogeneity, direct-native, and round-budget diagnostics.

## Validation

- Python/unit tests: **75/75 passed**.
- Merkle vectors: **13/13 passed**.
- State-chained LeNet vectors: **42 intervals passed**.
- Synthetic integration smoke test: passed.
- Documentation links: passed.

The current container cannot perform the real CKB RISC-V build; the Kaggle
reproduction script keeps that build as a mandatory real-measurement preflight.
