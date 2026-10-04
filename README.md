# CellVG / HNDT artifact

This repository contains the **Pareto-HNDT** optimizer, the original scalar HNDT
solver as an independent regression oracle, the exact constrained extension,
state-chained CKB-VM verification benchmark, paper-grounded comparison-policy
adapters, tests, experiment pipeline, and manuscript-integration notes.

## Start here

- [Experiment guide](EXPERIMENT_GUIDE.md)
- [Paper-grounded comparisons](BASELINES.md)
- [CKB benchmark semantics](ckb_bench/README.md)
- [Corrections and changes](CORRECTIONS.md)
- [Test report](TEST_REPORT.md)
- [Bundled manuscript](paper/README.md)


## V5 code/result revision

The V5 revision does **not** change the Pareto-HNDT objective or Bellman recurrence.
It expands the evidence pipeline with:

- two additional deterministic 12-transition traces (`conv_heavy`, `gemm_heavy`);
- a generic CKB interval benchmark/parser (`run_trace_intervals.sh`);
- multi-trace campaigns (`run_campaign.py`);
- fixed-grid query-cost and fault-weight sensitivity (`run_sensitivity.py`);
- bounded measurement-noise stability analysis (`run_robustness.py`);
- optional Pareto candidate/frontier/pruning instrumentation;
- synthetic solver-scalability experiments (`run_scalability.py`);
- a one-command orchestrator (`run_v5_suite.py`);
- a Pareto policy audit showing local alternatives and objective deltas.

Synthetic V5 outputs are explicitly marked **NOT FOR MANUSCRIPT**. Real mode
never substitutes demo costs for missing measurements.

## Quick validation

```bash
bash run_tests.sh
```

The current suite validates:

- exact Pareto-HNDT frontier and minimax-preserving mean selector;
- exact unconstrained scalar HNDT recovery;
- exact round-budgeted HNDT;
- mechanism-isolation ablations;
- literature-adapter provenance/fidelity guards;
- corrected 13-state Merkle proofs;
- generated **state-chained** LeNet, convolution-heavy, and GEMM-heavy checkpoints (42 span-1..4 intervals per trace);
- parsers, CSV/model validation, backend completeness, and local documentation
  links;
- a synthetic end-to-end experiment.

## Demo sanity run

```bash
python -m pip install -r experiments/requirements.txt
bash run_small_experiment.sh demo
```

Demo data are synthetic and must never be copied into the manuscript.

## Real measurement workflow

```bash
cd ckb_bench
bash run_benchmarks.sh
bash run_trace_intervals.sh lenet
bash run_trace_intervals.sh conv_heavy
bash run_trace_intervals.sh gemm_heavy
cd ..

python experiments/fill_query_template_from_ckb.py
python experiments/run_v5_suite.py --mode real \
  --campaign experiments/config/campaign_real.example.json \
  --out experiments/results/v5_real_suite
```

The main run reports:

- **Pareto-HNDT (minimax-safe)** as the primary proposal;
- scalar HNDT as a proposal ablation/regression oracle;
- one deduplicated Arbitrum/opML/Agatha midpoint family;
- Kirkpatrick--Klawe alphabetic-minimax and Hu--Tucker weighted-path objective baselines;
- conditional zk-OPML operator-level ZK comparison when complete reproduced ZK costs exist;
- adaptive-split, adaptive-stop, mean-first, fixed-`g`, heterogeneity, and round-budget ablations.

Additional outputs include the exact worst/mean Pareto frontier, a proposal-gain table,
an exact scalar round-budget frontier, and a scalar-HNDT policy audit with the cost
margin to the best alternative action.

## Scope statement

The neural interval benchmark is now genuinely state chained: each measured
interval starts from a generated checkpoint `S_i`, executes the real ordered
operator sequence, and must exactly reproduce `S_j`. The weights are fixed
synthetic integers, not a trained model, because the research outcome is
verification/dispute cost rather than predictive accuracy.

The benchmark still does not claim complete transaction cost: witness syscalls,
Cell serialization, and production model/economic plumbing must be measured
separately.
