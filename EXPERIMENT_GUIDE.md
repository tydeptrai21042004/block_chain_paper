# Small experiment guide for CellVG/HNDT

This package implements the minimum experiment set designed for the manuscript:

1. CKB-VM primitive and Merkle-query measurements.
2. One deterministic 12-transition LeNet-5 trace.
3. Direct measurement of every LeNet interval with span 1--4.
4. Exact HNDT versus midpoint and fixed-granularity policies.
5. One fault at every operator position.
6. One heterogeneous-versus-homogeneous cost ablation.

No large neural-network training or accuracy benchmark is required.

## 1. Verify the optimizer before measuring anything

Install the plotting dependency:

```bash
python -m pip install -r experiments/requirements.txt
```

Run the complete code check directly from the repository root:

```bash
./run_tests.sh
```

Or run only the 46 Python unit tests:

```bash
python -m unittest discover -s experiments/tests -v
```

The suite includes an exhaustive strategy-enumeration cross-check for many deterministic random instances with $n\le5$, in addition to input validation, baseline, policy-path, parser, and ablation tests.

Run the synthetic software sanity check:

```bash
./run_small_experiment.sh demo
```

Everything under `experiments/data/demo/` is synthetic and exists only to test the pipeline. **Never copy demo numbers into the paper.**

## 2. Measure primitive and Merkle-query costs on CKB-VM

Read `ckb_bench/README.md`, then run:

```bash
cd ckb_bench
./run_benchmarks.sh
cd ..
```

This produces `ckb_bench/ckb_primitive_measurements.csv` and raw logs.

Populate the query-cost template from the measured Merkle rows:

```bash
python experiments/fill_query_template_from_ckb.py
```

For the 12-transition trace there are 13 committed states, so the default authentication depth is

```text
ceil(log2(13)) = 4.
```

The code retains depths 1--8 in the template so the experiment can be changed later.

## 3. Measure the canonical LeNet intervals directly

The included CKB script can execute the arithmetic dimensions of each LeNet operator and selected consecutive blocks. Run:

```bash
cd ckb_bench
./run_lenet_intervals.sh
cd ..
```

This measures only 42 intervals:

- 12 atomic intervals;
- 11 span-2 intervals;
- 10 span-3 intervals;
- 9 span-4 intervals.

Measuring every span up to four is important for fairness: the fixed `g=2` and `g=4` baselines then have a measured terminal action whenever they reach an interval at or below their advertised granularity.

Then copy them into the blank HNDT interval form:

```bash
python experiments/fill_interval_template_from_ckb.py
```

The destination is:

```text
experiments/data/templates/interval_costs_template.csv
```

All 78 possible intervals are listed there, but **you do not need to measure all 78**. Unmeasured rows remain blank and are unavailable to HNDT.

### Important cost interpretation

The supplied CKB LeNet benchmark creates deterministic integer operands inside the contract. Thus its cycles isolate the arithmetic/kernel execution cost. They are not automatically a complete transaction cost including tensor witness loading, serialization, commitment lookup, and protocol logic.

For the small paper experiment this is acceptable if described exactly as a **kernel-level CKB-VM calibration**. If you later add witness/protocol code, report those extra components separately and update the interval costs accordingly.

## 4. Optional manual edits

You may edit:

```text
experiments/data/templates/interval_costs_template.csv
experiments/data/templates/query_costs_template.csv
```

Rules:

- every atomic interval `[0,1]`, ..., `[11,12]` must have at least one available backend;
- only enter costs you actually measured/reproduced;
- leave unimplemented/unmeasured direct-settlement actions blank;
- leave `zk_cycles` blank unless you independently reproduce a zkVM verifier configuration;
- do not interpolate missing values.

## 5. Run the real experiment

```bash
./run_small_experiment.sh real
```

Outputs:

- `experiments/results/current/summary.csv` — main strategy comparison;
- `experiments/results/current/fault_costs.csv` — cost for each faulty operator and policy;
- `experiments/results/current/hndt_policy.json` — exact optimal policy;
- `experiments/results/current/heterogeneity_ablation.csv` — required ablation;
- `experiments/results/current/hndt_policy_homogenized.json` — homogenized policy;
- `figures/fault_position_cost.pdf` — fault-position plot;
- `figures/heterogeneity_ablation.pdf` — heterogeneity plot.

If required measurements are missing, the real run fails rather than silently inventing values.

## 6. Fill the manuscript

`sections/04_analysis.tex` contains deliberately blank forms:

- `tab:primitive-form` — selected CKB primitive measurements;
- `tab:lenet-trace` — all 12 LeNet atomic operator measurements;
- `tab:validation` — HNDT/baseline summary;
- `tab:fault-position` — per-fault-position costs;
- `tab:heterogeneity-ablation` — measured vs homogenized costs;
- `fig:fault-position-placeholder` — figure placeholder.

Replace only the `\resultblank` rules with real measured values.

After the real run, replace the figure placeholder with:

```latex
\includegraphics[width=\linewidth]{figures/fault_position_cost.pdf}
```

Use these sources:

| Manuscript item | Source |
|---|---|
| Primitive table | `ckb_bench/ckb_primitive_measurements.csv` + raw logs |
| LeNet atomic rows | `ckb_bench/lenet_interval_measurements.csv` |
| Main comparison | `experiments/results/current/summary.csv` |
| Fault-position table/figure | `experiments/results/current/fault_costs.csv` |
| Heterogeneity ablation | `experiments/results/current/heterogeneity_ablation.csv` |

## Cost semantics implemented by the solver

For interval `[i,j]`, the CSV defines the terminal backends that are actually available. HNDT computes

```text
F(i,j) = min(
    available terminal costs on [i,j],
    min_{i<k<j} q(i,j,k) + max(F(i,k), F(k,j))
)
```

Blank entries are not assigned infinity by guesswork at data-generation time; they are simply absent from the available terminal action set.

## Baselines

The real run evaluates:

- atomic midpoint;
- fixed `g=2`;
- fixed `g=4`;
- exact HNDT/CellVG.

An operator-midpoint-plus-zkVM baseline is included only when the cost table contains a reproducible zk backend.

## Fault model

For fault position `t`, the evaluator follows the unique policy branch containing transition `t`, sums every query cost along the path, and adds the final terminal cost. This is exactly the quantity `C_T(t)` appearing in the minimax formulation; no stochastic fault distribution is introduced.

## Heterogeneity ablation

The ablation replaces heterogeneous atomic native costs by their mean while preserving interval availability and estimated non-atomic overhead. It therefore tests whether location-dependent neural cost, rather than merely the checkpoint topology, causes HNDT to depart from midpoint localization.

## Minimal command sequence

After the toolchain is installed, the complete real workflow is:

```bash
cd ckb_bench
./run_benchmarks.sh
./run_lenet_intervals.sh
cd ..

python experiments/fill_query_template_from_ckb.py
python experiments/fill_interval_template_from_ckb.py
./run_small_experiment.sh real
```

Then copy only real results into the manuscript blank cells.
