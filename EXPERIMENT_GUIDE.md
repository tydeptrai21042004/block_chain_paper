# Reproducible CellVG/HNDT experiment

This revision evaluates one narrow claim well: given a committed ordered neural
trace and measured verification/query costs, does exact cost-aware stop-or-split
optimization improve the worst dispute path relative to structurally restricted
localization policies?

## 1. Validate the artifact

```bash
python -m pip install -r experiments/requirements.txt
bash run_tests.sh
```

The test suite checks generated Merkle vectors, generated state-chained LeNet
checkpoints, Bellman optimality, round budgets, restricted policies, parsers,
and the synthetic integration run.

For a software-only smoke test:

```bash
bash run_small_experiment.sh demo
```

Everything under `experiments/data/demo/` is synthetic and is never manuscript
evidence.

## 2. Measure corrected Merkle query cost

Read [ckb_bench/README.md](ckb_bench/README.md), then:

```bash
cd ckb_bench
python generate_merkle_vectors.py --check
bash run_benchmarks.sh
cd ..
python experiments/fill_query_template_from_ckb.py
```

The depth-4 publication configuration corresponds to 13 committed states padded
to 16 leaves. Verification uses supplied sibling digests, actual leaf-index
ordering, one parent hash per level, and a known root.

## 3. Measure state-chained neural intervals

```bash
cd ckb_bench
python generate_lenet_vectors.py --check
bash run_lenet_intervals.sh
cd ..
python experiments/fill_interval_template_from_ckb.py
```

The 42 runs cover all consecutive spans 1--4. Each interval `[i,j]` loads the
exact generated checkpoint `S_i`, re-executes the true sequence
`f_{i+1},...,f_j`, and requires exact equality with `S_j`.

The deterministic weights are synthetic/untrained. They exist only to make the
verification trace reproducible; accuracy is not an outcome.

## 4. Run the real policy experiment

```bash
bash run_small_experiment.sh real
```

The real run fails on missing measurements instead of interpolating or
fabricating values.

### Main comparison

The main table contains:

1. HNDT;
2. one deduplicated prior-work midpoint/pinpoint family;
3. adaptive split + forced atomic stop;
4. midpoint split + adaptive stop;
5. oracle best fixed-`g`;
6. requested fixed-`g` ablations;
7. optional strategies only when their required measurements exist.

See [BASELINES.md](BASELINES.md) for the exact prior-work adaptation boundary.

## 5. Output files

A completed run writes:

- `summary.csv` — main deduplicated comparison;
- `fault_costs.csv` — complete fault-position sweep;
- `literature_baselines.csv` — citation, fidelity, exclusions, status;
- `literature_individual_summary.csv` — individual adapted-paper rows even if
  they collapse numerically;
- `literature_policy_groups.csv` — equivalence-group audit;
- `proposal_ablation.csv` — adaptive split / adaptive stop / minimax-safe mean-refinement isolation;
- `pareto_frontier.csv` — exact worst-case/mean cost frontier;
- `proposal_gain.csv` — Pareto-HNDT versus scalar HNDT;
- `round_budget_frontier.csv` — exact cost vs hard interaction-round budget;
- `hndt_scalar_policy_audit.csv` — scalar-HNDT chosen action, best alternative, and action margin;
- `heterogeneity_ablation.csv`;
- `optional_policy_status.csv` — skipped/available ZK and full-native extremes;
- `policies/*.json`;
- `run_metadata.json`.

## 6. Exact round-constrained HNDT

The original HNDT objective is unchanged. For deployments with a hard
interaction limit, the artifact also solves

```text
F_0(i,j) = A(i,j)
F_r(i,j) = min(
    A(i,j),
    min_k q(i,j,k) + max(F_{r-1}(i,k), F_{r-1}(k,j))
)
```

This is an exact restriction of the admissible strategy set, not a latency
heuristic. `round_budget_frontier.csv` evaluates successive budgets until the
unconstrained optimum is recovered.

To explicitly add one constrained policy to `summary.csv`:

```bash
python experiments/run_experiment.py --demo --round-budget 3
```

## 7. Manuscript rule

The manuscript is in [paper/](paper/README.md). Replace `\resultblank`
placeholders only after a verified **real** run and inspection of the raw
`ckb-debugger` logs.
