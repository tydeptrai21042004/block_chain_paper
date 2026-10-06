# Paper-grounded comparison policies

All reported methods use the same CellVG trace and measured cost tables. The
repository does **not** pretend to reproduce complete external systems when the
required chain/runtime measurements are absent. Each adapter exposes a precise
fidelity boundary in `experiments/results/.../literature_baselines.csv`.

## Optimistic-system localization family

The following five adapters reproduce only the comparable localization rule
and then use the same measured native CKB-VM atomic adjudicator:

1. **RDoC binary-search dispute** — Canetti, Riva, and Rothblum, 2013.
   Binary-search localization over committed computation configurations.
2. **TrueBit verification game** — Teutsch and Reitwießner. Binary search to
   one disputed execution step.
3. **Arbitrum-IVP** — Kalodner et al., USENIX Security 2018. Recursive
   challenge bisection to one disputed transition.
4. **opML Phase-1** — Conway et al., 2024. Operator-level bisection; the
   lower-level VM/microinstruction phase is excluded because this artifact does
   not measure that trace.
5. **Agatha-GPP chain projection** — Zheng et al., 2021. GPP restricted to the
   ordered-chain case; general-DAG/XCE machinery is excluded.

On the present ordered trace these can induce the same midpoint tree, so the
main table deduplicates them as one prior-work family while preserving separate
provenance and individual summaries.

## Classical optimal-tree baselines

Three additional paper-backed baselines test whether Pareto-HNDT gains are merely
an artifact of comparing against naive midpoint search or unrestricted tree height.

### Alphabetic Minimax Tree objective adaptation

**Source:** David G. Kirkpatrick and Maria M. Klawe, *Alphabetic Minimax
Trees*, SIAM Journal on Computing 14(3), 1985, DOI 10.1137/0214039.

Under forced atomic leaves and constant query cost `q`, a fault at transition
`t` costs

```text
A_t + q * d_t
```

where `d_t` is its tree depth. For `q>0`, minimizing the maximum is exactly the
paper's `max_t(w_t + d_t)` objective after setting `w_t=A_t/q`. The repository
solves the identical objective independently using its exact interval DP.

### Hu-Tucker weighted-path objective adaptation

**Source:** T. C. Hu and A. C. Tucker, *Optimal Computer Search Trees and
Variable-Length Alphabetical Codes*, SIAM Journal on Applied Mathematics
21(4), 1971, DOI 10.1137/0121057.

With forced atomic leaves and constant query cost, minimizing weighted mean
fault-path cost reduces to minimizing alphabetic weighted path length. The
artifact uses uniform fault weights by default, so no fault-frequency model is
invented. An independent interval DP is used rather than copying the historical
construction algorithm.


### Height-limited alphabetic mean objective adaptation

**Source:** L. L. Larmore and T. M. Przytycka, *A Fast Algorithm for Optimum
Height-Limited Alphabetic Binary Trees*, SIAM Journal on Computing 23(6), 1994,
DOI 10.1137/S0097539792231167.

The repository independently solves the corresponding fixed-leaf weighted-path
objective under an explicit maximum split depth.  This is useful when comparing
expected dispute cost under the same interaction-depth budget.

## zk-OPML operator-dispute baseline

**Source:** Vid Keršič and Muhamed Turkanović, *zk-OPML: Using zero-knowledge
proofs to optimize OPML*, 2026, DOI 10.1007/s44443-026-00573-1.

The paper's comparable structure is binary operator localization followed by a
ZK proof of the isolated ONNX operator. This adapter is run **only** when every
atomic interval has an independently reproduced finite `zkvm` cost. Otherwise
it is explicitly marked skipped; native verification is never substituted into
a row labelled ZK.

## Proposal and ablations

The revised proposal is **Pareto-HNDT (minimax-safe)**. It computes the exact
nondominated frontier of

```text
(worst-case fault-path cost, uniform-mean fault-path cost)
```

and then minimizes mean cost subject to preserving the exact scalar HNDT
minimax optimum. No weighted-sum scalarization parameter is introduced.

The experiment also reports:

- scalar HNDT (minimax only) — removes the new secondary objective;
- **Pareto adaptive split + atomic stop** — removes adaptive stopping while keeping the same minimax-safe secondary objective;
- **Pareto midpoint split + adaptive stop** — removes adaptive split placement while keeping the same objective;
- **Pareto midpoint + atomic stop** — removes both mechanisms while keeping the same objective;
- legacy scalar versions of the same restrictions for backward comparison;
- midpoint atomic literature family — paper-backed binary-search localization;
- Pareto mean-first endpoint — shows the worst/mean trade-off if minimax safety
  is deliberately relaxed;
- oracle best fixed-`g` and requested fixed-`g` policies;
- heterogeneity and round-budget diagnostics.

The key machine-readable outputs are `summary.csv`, `proposal_ablation.csv`,
`pareto_frontier.csv`, `literature_individual_summary.csv`, and
`literature_baselines.csv`.


## Exact arithmetic and compositional-theory check

Pareto objective pairs are compared with exact rational arithmetic; no epsilon is
used for equality or dominance.  Pareto pruning uses a sorted two-dimensional
antichain scan.  `python experiments/run_theory_checks.py` exports the small
ancestor-slack counterexample showing why retaining only the locally minimax
child label is not compositionally sufficient.
