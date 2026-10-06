# Paper-grounded comparison policies

The proposal core is **Pareto-DPS**, not a CKB-specific optimizer. Baselines are executed against the same ordered trace and admissible cost oracle, but every row is labelled by its actual fidelity. The repository never fabricates missing protocol levels or substitutes one verifier type for another.

See [BASELINE_FIDELITY.md](BASELINE_FIDELITY.md) for the machine-enforced fidelity rules.

## Binary-localization projections

The following papers provide binary localization structures that can be projected onto an ordered trace when an atomic terminal action with capability `one-step` is available:

1. **RDoC** — Canetti, Riva, and Rothblum, 2013.
2. **TrueBit** — Teutsch and Reitwießner.
3. **Arbitrum** — Kalodner et al., USENIX Security 2018.

On the current ordered trace these projections can induce the same binary tree. The main table therefore deduplicates them into one prior-work binary-localization family while preserving separate provenance rows.

These are structural projections only. Refereed-server assumptions, staking, incentives, assertion protocols, chain economics, and other system-specific mechanisms are not assigned synthetic costs.

## opML

The implementation now distinguishes two different comparison levels.

### Single-phase opML

The one-phase protocol bisects **VM microinstructions** until one instruction remains. The adapter executes only when

```text
trace_granularity = vm-microinstruction
```

and a `one-step` terminal action is available. An operator-level trace is rejected rather than silently relabelled as one-phase opML.

### Outer-phase projection

For an operator/high-level trace, the repository can report

```text
opML outer-phase operator projection (not full two-phase opML)
```

This reproduces only the outer/high-level localization structure. The inner VM-microinstruction phase is explicitly excluded unless corresponding data are supplied.

## Agatha

Agatha is reported only as

```text
Agatha GPP ordered-chain projection (not full DAG GPP)
```

because the current optimizer works on ordered executions. The general DAG graph-pinpoint protocol and XCE machinery are not claimed to be reproduced.

## Classical exact-objective baselines

### Kirkpatrick-Klawe alphabetic minimax

Under forced atomic leaves and constant query cost `q`, fault `t` pays

```text
A_t + q * d_t
```

so the objective is exactly the alphabetic minimax objective after rescaling. The artifact checks the required assumptions and solves the objective by an independent exact interval DP.

### Hu-Tucker weighted path length

Under the same fixed-leaf/constant-query assumptions, weighted expected dispute cost differs from weighted path length only by a tree-independent terminal term. The adapter therefore solves the exact objective reduction.

### Height-limited alphabetic mean

The fixed-leaf weighted-path objective is also solved under an explicit maximum split depth, corresponding to the height-limited alphabetic-tree setting.

Small-instance tests independently enumerate all ordered full binary trees and verify the objective values of these classical adapters.

## zk-OPML

The numerical adapter requires an explicit atomic terminal action tagged

```text
zk-proof
```

for every isolated operator. The backend name is irrelevant. If a required ZK terminal cost is unavailable, the baseline is skipped. Native/replay costs are never inserted into a ZK-labelled row.

The row is labelled as an operator-localization/ZK-terminal projection unless all other claimed protocol costs are also reproduced.

## Proposal and ablations

**Pareto-DPS (minimax-safe)** computes the exact nondominated antichain of

```text
(worst-case path cost, weighted expectation mass)
```

and lexicographically minimizes worst-case cost first, expected cost second. At the root, expectation mass equals ordinary expected cost.

The binary CKB experiment also reports:

- scalar binary DPS/HNDT minimax;
- Pareto adaptive split + atomic stop;
- Pareto midpoint split + adaptive stop;
- Pareto midpoint + atomic stop;
- legacy scalar restrictions;
- Pareto mean-first endpoint;
- fixed-granularity controls;
- round-budget, heterogeneity, prior, noise, and scaling diagnostics.

## Exact arithmetic

All Pareto decisions and the scalar/reference minimax decisions use exact rational arithmetic. Historical tolerance arguments are accepted only for API compatibility and do not determine scientific objective equality.
