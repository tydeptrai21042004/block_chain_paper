# Reproducible CellVG / Pareto-HNDT V5 experiment

V5 keeps the Pareto-HNDT Bellman recurrence unchanged and strengthens the
**evidence** around it.  The primary scientific question is now tested across
multiple ordered traces, query-cost regimes, fault-weight profiles, bounded
measurement noise, and solver sizes.

## 1. Validate the artifact

```bash
python -m pip install -r experiments/requirements.txt
bash run_tests.sh
```

The suite checks the original LeNet trace plus the new convolution-heavy and
GEMM-heavy state-chained traces.  All three contain 12 transitions / 13
checkpoints, so the same depth-4 Merkle configuration is comparable across
traces.

## 2. Fast software-only V5 smoke test

```bash
python experiments/run_v5_suite.py \
  --mode demo \
  --out experiments/results/v5_suite_demo
```

Everything used by demo mode is synthetic and is marked **NOT FOR MANUSCRIPT**.
It validates orchestration only.

## 3. Measure the common Merkle query cost

```bash
cd ckb_bench
python generate_merkle_vectors.py --check
bash run_benchmarks.sh
cd ..
python experiments/fill_query_template_from_ckb.py
```

The publication configuration commits 13 states, padded to 16 leaves, so the
canonical authentication depth is four.

## 4. Measure three real state-chained traces

```bash
cd ckb_bench
python generate_lenet_vectors.py --check
python generate_trace_vectors.py --check

bash run_trace_intervals.sh lenet
bash run_trace_intervals.sh conv_heavy
bash run_trace_intervals.sh gemm_heavy
cd ..
```

Each command measures every consecutive interval of spans 1--4, i.e. 42
intervals per trace.  Raw `ckb-debugger` logs are retained under
`ckb_bench/logs_<trace>/` and parsed into:

- `ckb_bench/lenet_interval_measurements.csv`;
- `ckb_bench/conv_heavy_interval_measurements.csv`;
- `ckb_bench/gemm_heavy_interval_measurements.csv`.

No longer interval is interpolated.

## 5. Run the real multi-trace campaign

After filling the measured query table, run:

```bash
python experiments/run_campaign.py \
  --campaign experiments/config/campaign_real.example.json \
  --out experiments/results/v5_real_campaign
```

Or run the complete V5 suite:

```bash
python experiments/run_v5_suite.py \
  --mode real \
  --campaign experiments/config/campaign_real.example.json \
  --out experiments/results/v5_real_suite
```

Real mode rejects campaign entries marked synthetic. Missing measurements cause
an explicit failure instead of a fallback to demo values.

## 6. Query-cost sensitivity

The experiment runner accepts a multiplicative query-cost scale without
changing any terminal cost or solver rule:

```bash
python experiments/run_experiment.py ... --query-scale 0.5
python experiments/run_experiment.py ... --query-scale 1
python experiments/run_experiment.py ... --query-scale 2
```

The predefined sensitivity runner uses the fixed grid
`0.25, 0.5, 1, 2, 4`:

```bash
python experiments/run_sensitivity.py \
  --trace <trace.csv> \
  --interval-costs <measured.csv> \
  --query-costs experiments/data/templates/query_costs_template.csv \
  --config experiments/config/experiment.json \
  --out experiments/results/sensitivity
```

No query scale is selected from the observed gain.

## 7. Fault-weight sensitivity

The primary result remains **uniform** fault weights.  Three transparent
counterfactual profiles are available:

```text
uniform : p_t = 1/n
front   : p_t proportional to n-t
back    : p_t proportional to t+1
cost    : p_t proportional to measured atomic terminal cost
```

Example:

```bash
python experiments/run_experiment.py ... --fault-weights front
```

The same weights are passed to Pareto-HNDT, weighted reporting, and the
Hu--Tucker objective adaptation.

## 8. Measurement-noise robustness

```bash
python experiments/run_robustness.py \
  --trace <trace.csv> \
  --interval-costs <measured.csv> \
  --query-costs experiments/data/templates/query_costs_template.csv \
  --config experiments/config/experiment.json \
  --noise-levels 0.01,0.025,0.05 \
  --trials 100 \
  --out experiments/results/robustness
```

This never overwrites measured data.  Each trial constructs a perturbed in-memory
cost model and checks that Pareto-HNDT still exactly recovers the scalar minimax
projection.

## 9. Solver scalability / frontier instrumentation

```bash
python experiments/run_scalability.py \
  --sizes 12,24,48 \
  --families homogeneous,mild,strong \
  --out experiments/results/scalability/scalability.csv
```

The default is deliberately modest because the exact Pareto solver is
output-sensitive. Larger sizes can be requested explicitly, e.g.
`--sizes 12,24,48,96,192`. The scalability costs are synthetic and are used
**only** for runtime/frontier analysis, never for CKB performance claims.

Recorded diagnostics include:

- scalar and Pareto wall-clock solver time;
- Python peak memory during Pareto solving;
- root and peak frontier size;
- generated/retained label counts;
- duplicate and dominated-label pruning;
- pruning ratio.

Instrumentation is optional and does not enter the optimization objective.

## 10. Per-run outputs

A completed `run_experiment.py` execution writes:

- `summary.csv`;
- `fault_costs.csv` including the applied fault weight;
- `pareto_frontier.csv`;
- `proposal_gain.csv`;
- `proposal_ablation.csv` / `mechanism_ablation.csv`;
- `round_budget_frontier.csv`;
- `heterogeneity_ablation.csv`;
- `hndt_scalar_policy_audit.csv`;
- `pareto_policy_audit.csv`;
- literature baseline provenance/grouping tables;
- `policies/*.json`;
- `run_metadata.json`, including query scale, fault weights, and optional Pareto
  pruning statistics.

## 11. Scientific guardrails

- Do not change the Bellman recurrence to increase a reported gain.
- Do not select traces, query scales, or fault profiles after looking at the
  answer and report only favorable cases.
- Do not fabricate unavailable zkVM costs.
- Do not interpolate unmeasured native intervals.
- Keep synthetic scalability/demo results separate from real CKB evidence.
- Preserve raw `ckb-debugger` logs and exact tool versions for every real run.
