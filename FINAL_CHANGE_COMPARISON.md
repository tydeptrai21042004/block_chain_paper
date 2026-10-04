# Final code-change comparison against `block_chain_paper-main(4).zip`

This revision changes the **scientific center** of the proposal while preserving
the original scalar HNDT solver as an independent regression oracle.

## 1. Proposal method

### Before

Scalar HNDT optimized only

```text
worst-case fault-path cost
```

using the exact stop-or-split recurrence. Mean fault-path cost was reported only
as a diagnostic and scalar-HNDT tie handling was not guaranteed to minimize it.

### After — Pareto-HNDT (minimax-safe)

The new solver computes the exact nondominated frontier

```text
(worst-case fault-path cost, uniform-mean fault-path cost)
```

for every interval. Direct settlement and split composition are exact:

```text
settle: (A, P_ij * A)
split:  (q + max(W_L,W_R), P_ij*q + M_L + M_R)
```

The selected proposal is lexicographic:

1. minimize worst-case cost exactly;
2. among **all** minimax-optimal policies, minimize mean fault-path cost;
3. only then use deterministic tertiary tie breaking.

Therefore the implementation checks at runtime that

```text
W(Pareto-HNDT) == W(scalar HNDT)
mean(Pareto-HNDT) <= mean(scalar HNDT)
```

No weighted-sum coefficient or new method hyperparameter is introduced.

## 2. New theory-backed baselines

Added paper-grounded objective adapters:

- **Kirkpatrick--Klawe Alphabetic Minimax Tree (1985)** — exact forced-atomic,
  constant-query reduction of `max_t(A_t + q d_t)`;
- **Hu--Tucker Optimal Alphabetic Mean Tree (1971)** — exact forced-atomic,
  constant-query weighted-path-length objective, using uniform fault weights by
  default so no fault-frequency prior is invented;
- **zk-OPML (2026)** — binary operator localization followed by strict atomic ZK
  settlement, executed only when every required `zkvm` measurement exists.

Existing Arbitrum/opML/Agatha adapters remain and are deduplicated only inside
the close midpoint/pinpoint system family.

## 3. New and strengthened ablations

The main experiment now separates three proposal ingredients:

| Policy | Adaptive split | Adaptive stop | Minimax-safe mean refinement |
|---|---:|---:|---:|
| Pareto-HNDT | yes | yes | yes |
| Scalar HNDT | yes | yes | no |
| Adaptive split + atomic stop | yes | no | no |
| Midpoint + adaptive stop | no | yes | no |
| Midpoint atomic family | no | no | no |

Also added the **Pareto mean-first endpoint**. It intentionally relaxes minimax
priority and shows the actual worst/mean trade-off rather than hiding it behind
a scalarization constant.

Existing fixed-`g`, oracle fixed-`g`, heterogeneity, direct-full-native, and
round-budget diagnostics are retained.

## 4. New outputs

- `pareto_frontier.csv`
- `pareto_hndt_policy.json`
- `pareto_mean_first_policy.json`
- `proposal_ablation.csv`
- `proposal_gain.csv`
- `hndt_scalar_policy.json`
- `hndt_scalar_policy_audit.csv`

`mechanism_ablation.csv` is retained as a backward-compatible copy of the new
proposal ablation table.

## 5. Correctness guards

Added exhaustive small-instance tests that enumerate all admissible stop/split
strategies and compare the true nondominated objective set against Pareto-HNDT.
The test suite also checks:

- exact minimax recovery from scalar HNDT;
- mean non-regression at the minimax optimum;
- nondominance of every retained frontier label;
- Kirkpatrick--Klawe reduction equivalence;
- Hu--Tucker mean-objective equivalence;
- rejection of classical-tree provenance claims when query costs are not
  constant;
- strict zk-OPML measurement completeness.

## 6. Validation performed in this revision

- **75/75 Python tests passed**;
- 13/13 canonical Merkle proofs passed;
- all 42 state-chained LeNet vector checks passed;
- synthetic integration experiment passed;
- documentation-link test passed;
- CKB RISC-V build regression remains conditionally skipped in this container
  because the target toolchain is unavailable, exactly as before.

The bundled synthetic demo outputs are for software verification only and must
not be used as manuscript results.
