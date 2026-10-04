# Change manifest — Pareto-HNDT revision

Base compared: `block_chain_paper-main(4).zip`.

## Scientific changes

1. **Pareto-HNDT proposal**
   - exact nondominated frontier of worst-case and uniform-mean fault-path cost;
   - minimax-safe lexicographic selection: preserve exact scalar-HNDT optimum,
     then minimize mean among all minimax-optimal policies;
   - no weighted-sum coefficient or additional proposal hyperparameter.

2. **Paper-grounded baseline expansion**
   - existing Arbitrum-IVP, opML Phase-1, and Agatha-GPP chain adaptations;
   - Kirkpatrick--Klawe alphabetic-minimax objective adaptation;
   - Hu--Tucker alphabetic weighted-path objective adaptation;
   - conditional zk-OPML operator-dispute baseline requiring complete reproduced
     atomic `zkvm` costs.

3. **Ablation expansion**
   - scalar HNDT (removes the new secondary objective);
   - adaptive split + atomic stop;
   - midpoint split + adaptive stop;
   - Pareto mean-first endpoint;
   - oracle and requested fixed-`g` policies;
   - heterogeneity and round-budget diagnostics.

4. **New outputs**
   - `pareto_frontier.csv`;
   - `proposal_ablation.csv`;
   - `proposal_gain.csv`;
   - Pareto/scalar policy JSON files;
   - updated literature provenance and individual summaries.

## Code changes

- new `experiments/hndt/pareto.py`;
- expanded `experiments/hndt/baselines.py`;
- expanded `experiments/hndt/literature_baselines.py`;
- revised `experiments/run_experiment.py`;
- revised plotting and Kaggle reproduction scripts;
- new exhaustive `experiments/tests/test_pareto.py`;
- updated baseline tests and documentation.

## Validation

- **75/75 Python tests passed**;
- **13/13** canonical Merkle proofs validated;
- **42/42** state-chained LeNet intervals validated;
- synthetic integration run passed;
- documentation-link regression passed.

The RISC-V/CKB binary build is conditionally skipped in this container because
the target toolchain is unavailable. No real cycle values were invented.
