# Paper-grounded comparison policies

All reported methods use the same CellVG trace and measured cost tables. The
repository does **not** pretend to reproduce complete external systems when the
required chain/runtime measurements are absent. Each adapter exposes a precise
fidelity boundary in `experiments/results/.../literature_baselines.csv`.

## Optimistic-system localization family

The following three adapters reproduce only the comparable localization rule
and then use the same measured native CKB-VM atomic adjudicator:

1. **Arbitrum-IVP** — Kalodner et al., USENIX Security 2018. Recursive
   challenge bisection to one disputed transition.
2. **opML Phase-1** — Conway et al., 2024. Operator-level bisection; the
   lower-level VM/microinstruction phase is excluded because this artifact does
   not measure that trace.
3. **Agatha-GPP chain projection** — Zheng et al., 2021. GPP restricted to the
   ordered-chain case; general-DAG/XCE machinery is excluded.

On the present ordered trace these can induce the same midpoint tree, so the
main table deduplicates them as one prior-work family while preserving separate
provenance and individual summaries.

## Classical optimal-tree baselines

Two additional paper-backed baselines test whether Pareto-HNDT gains are merely
an artifact of comparing against naive midpoint search.

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
- adaptive split + atomic stop — removes adaptive stopping;
- midpoint split + adaptive stop — removes adaptive split placement;
- midpoint atomic literature family — removes both adaptive mechanisms;
- Pareto mean-first endpoint — shows the worst/mean trade-off if minimax safety
  is deliberately relaxed;
- oracle best fixed-`g` and requested fixed-`g` policies;
- heterogeneity and round-budget diagnostics.

The key machine-readable outputs are `summary.csv`, `proposal_ablation.csv`,
`pareto_frontier.csv`, `literature_individual_summary.csv`, and
`literature_baselines.csv`.
