# V5 result status

## What is validated now

- Full Python regression suite: **84 passed + 17 subtests**.
- Both added trace generators reproduce deterministic checkpoints and verify all 42 span-1..4 intervals internally.
- The original/default experiment path remains compatible.
- Pareto statistics are observational only: enabling them does not alter the exact frontier or minimax-safe selected policy.
- Sensitivity, robustness, campaign, and scalability runners execute end-to-end.

## Bundled demo output

`experiments/results/v5_suite_demo/` uses synthetic costs and is **NOT FOR MANUSCRIPT**. Its purpose is software validation only. It demonstrates that:

- the multi-trace campaign produces different policies for different cost structures;
- query/fault-weight sweeps are wired correctly;
- bounded perturbation runs preserve exact Pareto/scalar minimax recovery checks;
- frontier/pruning statistics are emitted;
- solver-scaling measurements can be generated reproducibly.

Do not quote the demo percentage gains as experimental evidence in the paper.

## Scalability precomputation

The bundled demo scaling grid is limited to `n = 12, 24, 48`, which completes reliably in this environment. The runner accepts larger explicit sizes, including 96 and 192. A broad strong-heterogeneity run at the larger sizes exceeded this execution environment's time budget, so no runtime for those sizes is fabricated or recorded as completed.

## What must be run for manuscript-quality V5 results

1. Measure `lenet`, `conv_heavy`, and `gemm_heavy` with `ckb-debugger` using `ckb_bench/run_trace_intervals.sh`.
2. Parse each raw-log directory with `ckb_bench/parse_trace_logs.py`.
3. Populate a real campaign config from `experiments/config/campaign_real.example.json`.
4. Run `python experiments/run_v5_suite.py --mode real ...`.
5. Use only those real outputs in the manuscript comparison tables.

This separation is deliberate: missing real measurements are never replaced by synthetic or interpolated values.
