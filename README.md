# CellVG / HNDT experiment code

This folder contains only implementation, tests, CKB-VM benchmarks, templates, and generated experiment outputs.

## Quick validation

```bash
./run_tests.sh
```

The test suite checks the Bellman solver, baselines, policy traversal, input/CSV validation, CKB parsers, backend completeness, and heterogeneity ablation.

## Demo sanity run

```bash
python -m pip install -r experiments/requirements.txt
./run_small_experiment.sh demo
```

Demo values are synthetic and must not be copied into the manuscript.

## Real measurement workflow

```bash
cd ckb_bench
./run_benchmarks.sh
./run_lenet_intervals.sh
cd ..
python experiments/fill_query_template_from_ckb.py
python experiments/fill_interval_template_from_ckb.py
./run_small_experiment.sh real
```

See `EXPERIMENT_GUIDE.md` for the full protocol.

Generated plots remain inside this code folder. After reviewing the real logs/CSVs, copy only the final verified figure(s) and numerical values into `../latex/`.
