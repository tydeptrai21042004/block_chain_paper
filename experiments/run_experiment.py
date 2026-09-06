#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hndt.baselines import (
    fixed_g_policy,
    has_complete_atomic_backend,
    midpoint_atomic_policy,
    midpoint_operator_zk_policy,
)
from hndt.core import CostModel, solve_hndt
from hndt.evaluate import evaluate_policy, summarize
from hndt.io import build_model, validate_atomic_coverage
from hndt.reporting import first_split, write_csv, write_fault_paths, write_policy_json


def parse_args():
    p = argparse.ArgumentParser(description="Run the small HNDT/CellVG LeNet-5 dispute experiment")
    p.add_argument("--demo", action="store_true", help="Use synthetic costs only for code sanity checks")
    p.add_argument("--trace", default=str(ROOT / "data" / "templates" / "lenet5_trace.csv"))
    p.add_argument("--interval-costs", default=None)
    p.add_argument("--query-costs", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=str(ROOT / "results" / "current"))
    p.add_argument("--fixed-g", default="2,4", help="Comma-separated fixed terminal sizes")
    return p.parse_args()


def homogenized_model(model: CostModel) -> CostModel:
    atomic = []
    for i in range(model.n):
        c, _ = model.best_terminal(i, i + 1)
        if not math.isfinite(c):
            raise ValueError("Cannot homogenize without complete atomic terminal costs")
        atomic.append(c)
    mean_atomic = sum(atomic) / len(atomic)

    terminal = {}
    for (i, j), backends in model.terminal_costs.items():
        new_backends = dict(backends)
        if "native" in new_backends:
            # Preserve the interval's non-atomic overhead while replacing heterogeneous
            # atomic work by a uniform mean. This avoids changing availability.
            measured_atomic_sum = sum(atomic[i:j])
            residual = float(new_backends["native"]) - measured_atomic_sum
            homogenized = residual + (j - i) * mean_atomic
            if homogenized < 0:
                raise ValueError(
                    f"Homogenization would make native cost negative on [{i},{j}]. "
                    "The measured interval table is incompatible with the additive-residual ablation."
                )
            new_backends["native"] = homogenized
        terminal[(i, j)] = new_backends
    return CostModel(model.n, terminal, model.query_cost)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        interval = ROOT / "data" / "demo" / "interval_costs_demo.csv"
        query = ROOT / "data" / "demo" / "query_costs_demo.csv"
        config = ROOT / "config" / "demo.json"
        marker = "SYNTHETIC DEMO - NOT FOR MANUSCRIPT"
    else:
        interval = Path(args.interval_costs or ROOT / "data" / "templates" / "interval_costs_template.csv")
        query = Path(args.query_costs or ROOT / "data" / "templates" / "query_costs_template.csv")
        config = Path(args.config or ROOT / "config" / "experiment.json")
        marker = "REAL/USER-SUPPLIED MEASUREMENTS"

    try:
        model = build_model(args.trace, interval, query, config)
        missing = validate_atomic_coverage(model)
        if missing:
            raise ValueError(
                "Missing terminal measurements for atomic intervals: "
                + ", ".join(f"[{i},{j}]" for i, j in missing)
            )
    except Exception as exc:
        out.joinpath("MISSING_MEASUREMENTS.txt").write_text(
            f"Experiment not executed.\nReason: {exc}\n\n"
            "Fill experiments/data/templates/interval_costs_template.csv and "
            "query_costs_template.csv, or run `python experiments/run_experiment.py --demo` "
            "to test the software with clearly synthetic data.\n",
            encoding="utf-8",
        )
        raise

    h = solve_hndt(model)
    policies = {"HNDT": h.action, "Atomic midpoint": midpoint_atomic_policy(model)}

    for raw_g in args.fixed_g.split(","):
        g = int(raw_g.strip())
        policies[f"Fixed g={g}"] = fixed_g_policy(model, g)

    if has_complete_atomic_backend(model, "zkvm"):
        policies["Operator midpoint + ZK"] = midpoint_operator_zk_policy(model)

    paths = {name: evaluate_policy(model, policy) for name, policy in policies.items()}
    summaries = []
    for name, path_list in paths.items():
        row = summarize(name, path_list, h.optimum)
        row["first_split"] = first_split(policies[name], model.n)
        summaries.append(row)

    write_csv(out / "summary.csv", summaries)
    write_fault_paths(out / "fault_costs.csv", paths)
    write_policy_json(out / "hndt_policy.json", h.action)

    # Heterogeneity ablation.
    hom = homogenized_model(model)
    hh = solve_hndt(hom)
    ablation = [
        {
            "cost_model": "heterogeneous",
            "optimal_worst_case_cost": h.optimum,
            "first_action": first_split(h.action, model.n),
            "max_rounds": h.max_rounds[(0, model.n)],
        },
        {
            "cost_model": "homogenized",
            "optimal_worst_case_cost": hh.optimum,
            "first_action": first_split(hh.action, model.n),
            "max_rounds": hh.max_rounds[(0, model.n)],
        },
    ]
    write_csv(out / "heterogeneity_ablation.csv", ablation)
    write_policy_json(out / "hndt_policy_homogenized.json", hh.action)

    metadata = {
        "data_status": marker,
        "trace": str(args.trace),
        "interval_costs": str(interval),
        "query_costs": str(query),
        "config": str(config),
        "n": model.n,
        "optimum": h.optimum,
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
