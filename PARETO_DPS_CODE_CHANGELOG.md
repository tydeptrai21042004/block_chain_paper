# Pareto-DPS code revision

This revision generalizes the proposal so that the optimization core is no longer defined by Nervos CKB, CKB-VM, Merkle authentication, neural networks, or a fixed cost unit. CKB-VM remains a measured adapter/case study.

## 1. Platform-independent proposal core

Added `experiments/hndt/dps.py` with:

- `VerificationInstance` — an ordered deterministic execution with admissible terminal and query actions;
- `TerminalActionSpec` — opaque action id, exact cost, optional capability tags;
- `QueryActionSpec` — arbitrary finite ordered partitions using one or more checkpoint cuts;
- `solve_pareto_dps()` — exact Pareto synthesis over finite partitions;
- `solve_scalar_dps()` — independent exact minimax reference over the same action space;
- `reconstruct_dps_policy()` and exact policy evaluation;
- `from_cost_model()` — backward-compatible adapter from the old binary cost model;
- `dps_to_legacy_policy()` — conversion for existing reporting/CKB scripts.

The generalized composition for a query action with children `I_1,...,I_m` is implemented as

`W = q + max_r W_r`

`M = P_I q + sum_r M_r`

so the historical binary split is a strict special case.

## 2. Generic input schema

Added `experiments/hndt/generic_io.py` and `experiments/run_generic_instance.py`.

Generic terminal CSV columns:

`i,j,action_id,cost,capabilities`

Generic query CSV columns:

`i,j,action_id,cuts,cost`

The core has no required columns named `native_cycles`, `zk_cycles`, `depth`, or `cycles`.

A runnable non-CKB example is included under `experiments/data/generic_demo/` and uses `microseconds` as its cost unit.

## 3. CKB is now an adapter

`experiments/hndt/io.py` now exposes `build_ckb_model()`; the historical `build_model()` is retained only as a compatibility alias.

The CKB adapter attaches metadata and capabilities:

- `native` -> `replay`, `one-step`;
- `zkvm` -> `zk-proof`;
- platform -> `ckb-vm`;
- cost unit -> `cycles`;
- authentication -> `merkle`.

The generic solver does not depend on any of these names.

## 4. Exact arithmetic extended to scalar/reference solvers

The scalar minimax solver and round-budget solver now make scientific decisions with `Fraction`, matching the exact Pareto solver. Historical tolerance arguments remain accepted for API compatibility but do not change objective comparisons.

`HNDTResult` now exposes `optimum_exact` and `exact_value` while keeping float reporting fields.

Classical mean/constant-query baselines were also moved to exact comparison logic.

## 5. Capability-based terminal actions

`CostModel` now supports:

- `terminal_capabilities`;
- `metadata`;
- `backend_priority`.

Baseline adapters request capabilities such as `one-step` or `zk-proof` instead of requiring literal backend names. A legacy inference shim preserves old fixtures, but new adapters explicitly supply capabilities.

## 6. Corrected paper-backed baselines

### RDoC / TrueBit / Arbitrum

Implemented as an explicitly labelled ordered-trace binary-localization family requiring a one-step terminal capability. These are structural projections only; protocol economics are not fabricated.

### opML

The previous `opML Phase-1` operator-level row was too permissive.

Now:

- `opml_single_phase_policy()` executes only when `trace_granularity` is `vm-microinstruction`/`microinstruction`;
- on an operator trace it is skipped;
- `opml_outer_phase_projection_policy()` is a separate, explicitly labelled projection of the high-level phase and is never described as full two-phase opML.

### Agatha

Kept only as `Agatha GPP ordered-chain projection`; the code and provenance explicitly state that it is not a reproduction of the full DAG GPP/XCE system.

### zk-OPML

Requires a `zk-proof` terminal capability on every atomic operator. The backend can have any name; missing reproduced ZK costs cause a clean skip.

### Classical tree objectives

Kirkpatrick-Klawe, Hu-Tucker, and the height-limited alphabetic objective retain mechanically checked assumptions. New tests independently enumerate small ordered trees and verify the objective values.

## 7. Platform-independent cost-regime experiment

Added `experiments/run_cost_regimes.py`.

It sweeps dimensionless terminal-cost heterogeneity and relative query cost without any CKB-specific input, reporting frontier size, minimax-safe expected-cost gain, mean-first worst-cost ratio, and depth.

## 8. Proposal path in the existing CKB experiment

`experiments/run_experiment.py` now:

1. constructs the legacy CKB binary model through the adapter;
2. solves the proposal through generic Pareto-DPS;
3. independently solves the legacy binary Pareto specialization;
4. requires exact equality of the selected `(worst, expected)` objectives;
5. uses the generic Pareto-DPS policy for the proposal row.

This guards against accidental divergence while preserving all existing CKB data/reporting scripts.

## 9. Validation

`run_tests.sh` now also runs:

- the generic non-CKB instance smoke test;
- a small platform-independent cost-regime smoke test.

Current validation in this environment:

- 100/100 Python unit tests pass;
- deterministic Merkle and trace-vector checks pass;
- CKB/demo campaign smoke tests pass;
- generic Pareto-DPS smoke test passes;
- cost-regime smoke test passes;
- RISC-V contract build is skipped when the required Rust target is unavailable, as before.
