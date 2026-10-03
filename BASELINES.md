# Paper-grounded comparison policies

The experiment uses one common CellVG cost table and changes only the dispute
policy. This is important: the repository does **not** claim to reproduce three
complete external blockchain stacks. It reproduces the comparable localization
component supported by each cited paper and records the missing components
explicitly in `literature_baselines.csv`.

## 1. Arbitrum-IVP common-trace adaptation

**Source:** Harry Kalodner et al., *Arbitrum: Scalable, Private Smart
Contracts*, USENIX Security 2018.

- https://www.usenix.org/conference/usenixsecurity18/presentation/kalodner
- comparable component: recursively bisect a disputed assertion until one
  atomic transition remains;
- common-testbed terminal action: measured native CKB-VM verification;
- excluded: Arbitrum VM economics, staking, and full assertion protocol.

## 2. opML Phase-1 common-trace adaptation

**Source:** K. D. Conway et al., *opML: Optimistic Machine Learning on
Blockchain*, 2024.

- https://arxiv.org/abs/2401.17555
- comparable component: operator-level bisection/localization;
- common-testbed terminal action: measured native CKB operator verification;
- excluded: opML's second VM-microinstruction dispute phase because this
  artifact has no independently measured microinstruction trace. No cost is
  fabricated for it.

## 3. Agatha-GPP chain projection

**Source:** Zihan Zheng et al., *Agatha: Smart Contract for DNN Computation*,
2021.

- https://arxiv.org/abs/2105.04919
- comparable component: graph-node pinpointing;
- current projection: the HNDT state space is an ordered chain, so the GPP
  comparison is explicitly restricted to the chain case;
- excluded: general-DAG GPP, Cross-evaluator Consistent Execution (XCE), and
  Ethereum-specific arbitration.

## Why the three prior-work adaptations can be numerically identical

On the current ordered 12-transition trace, all three comparable localization
components reduce to midpoint/pinpoint bisection followed by native atomic
settlement. The code detects identical policy trees and reports them **once** in
the main table as:

```text
Prior-work midpoint/pinpoint family
```

The individual policies are still serialized and individually summarized for
provenance. This avoids presenting three duplicate rows as three independent
numerical competitors.

## Mechanism-isolation ablations

The revised experiment adds the two restrictions needed to identify where
HNDT's improvement comes from:

| Policy | Adaptive split | Adaptive stop |
|---|---:|---:|
| Prior-work midpoint/pinpoint family | No | No |
| Adaptive split + atomic stop | Yes | No |
| Midpoint split + adaptive stop | No | Yes |
| **HNDT** | **Yes** | **Yes** |

It also computes an **oracle best fixed-g** policy by evaluating all feasible
global terminal granularities and selecting the best on the same cost table.
This is deliberately labeled an oracle internal ablation, not a published
baseline.

## ZK comparison rule

`operator midpoint + reproduced ZK` is executed only when a finite `zkvm` cost
exists on every required atomic interval. Otherwise the strategy is written to
`optional_policy_status.csv` as skipped. Native verification is never silently
substituted into a strategy labeled `+ ZK`.
