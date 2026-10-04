#!/usr/bin/env python3
from __future__ import annotations

"""Predefined query-cost and fault-weight sensitivity experiments.

No sweep parameter is selected from the observed result.  The default grid is
fixed in advance and the reference point is query_scale=1, uniform weights.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hndt.baselines import hu_tucker_atomic_policy
from hndt.core import solve_hndt
from hndt.evaluate import evaluate_policy, summarize
from hndt.io import build_model, validate_atomic_coverage
from hndt.pareto import solve_pareto_hndt
from hndt.reporting import first_split, write_csv
from hndt.sensitivity import fault_weight_profile, scale_query_costs


def _parse_scales(value: str) -> list[float]:
    scales = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not scales or any(x < 0 for x in scales):
        raise ValueError("query scales must be a non-empty list of non-negative values")
    return scales


def _row(model, *, label: str, profile: str, query_scale: float) -> dict:
    weights = fault_weight_profile(model, profile)
    scalar = solve_hndt(model)
    pareto = solve_pareto_hndt(model, fault_weights=weights, collect_stats=True)
    if abs(scalar.optimum - pareto.worst_optimum) > 1e-9:
        raise AssertionError("Pareto minimax projection differs from scalar HNDT")

    p_paths = evaluate_policy(model, pareto.policy)
    s_paths = evaluate_policy(model, scalar.action)
    p = summarize("Pareto-HNDT", p_paths, scalar.optimum, fault_weights=weights)
    s = summarize("Scalar HNDT", s_paths, scalar.optimum, fault_weights=weights)

    try:
        ht_policy = hu_tucker_atomic_policy(model, fault_weights=weights)
        ht = summarize(
            "Hu-Tucker objective",
            evaluate_policy(model, ht_policy),
            scalar.optimum,
            fault_weights=weights,
        )
        ht_worst, ht_mean = ht["worst_case_cost"], ht["mean_cost"]
    except ValueError:
        ht_worst, ht_mean = "", ""

    return {
        "experiment": label,
        "fault_weight_profile": profile,
        "query_scale": query_scale,
        "pareto_worst": p["worst_case_cost"],
        "scalar_worst": s["worst_case_cost"],
        "pareto_mean": p["mean_cost"],
        "scalar_mean": s["mean_cost"],
        "mean_improvement": s["mean_cost"] - p["mean_cost"],
        "mean_improvement_percent": (
            100.0 * (s["mean_cost"] - p["mean_cost"]) / s["mean_cost"]
            if s["mean_cost"]
            else 0.0
        ),
        "pareto_max_rounds": p["max_rounds"],
        "scalar_max_rounds": s["max_rounds"],
        "pareto_first_split": first_split(pareto.policy, model.n),
        "scalar_first_split": first_split(scalar.action, model.n),
        "root_frontier_size": len(pareto.root_frontier),
        "peak_frontier_size": pareto.stats.peak_frontier_size if pareto.stats else "",
        "candidates_generated": pareto.stats.candidates_generated if pareto.stats else "",
        "pruning_ratio": pareto.stats.pruning_ratio if pareto.stats else "",
        "hu_tucker_worst": ht_worst,
        "hu_tucker_mean": ht_mean,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--interval-costs", required=True)
    ap.add_argument("--query-costs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--query-scales", default="0.25,0.5,1,2,4")
    ap.add_argument("--fault-profiles", default="uniform,front,back,cost")
    args = ap.parse_args()

    base = build_model(args.trace, args.interval_costs, args.query_costs, args.config)
    missing = validate_atomic_coverage(base)
    if missing:
        raise ValueError(f"missing atomic costs: {missing}")

    scales = _parse_scales(args.query_scales)
    profiles = [x.strip() for x in args.fault_profiles.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    query_rows = []
    for scale in scales:
        model = scale_query_costs(base, scale)
        query_rows.append(_row(model, label="query_cost", profile="uniform", query_scale=scale))
    write_csv(out / "query_sensitivity.csv", query_rows)

    weight_rows = []
    for profile in profiles:
        weight_rows.append(_row(base, label="fault_weights", profile=profile, query_scale=1.0))
    write_csv(out / "fault_weight_sensitivity.csv", weight_rows)

    (out / "sensitivity_metadata.json").write_text(
        json.dumps(
            {
                "trace": args.trace,
                "interval_costs": args.interval_costs,
                "query_costs": args.query_costs,
                "config": args.config,
                "query_scales": scales,
                "fault_profiles": profiles,
                "note": "Sensitivity only; no parameter is selected to maximize the proposal gain.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
