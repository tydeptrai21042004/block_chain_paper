#!/usr/bin/env python3
from __future__ import annotations

"""Bounded measurement-noise stability analysis for Pareto-HNDT."""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hndt.core import solve_hndt
from hndt.evaluate import evaluate_policy, weighted_mean_cost
from hndt.io import build_model, validate_atomic_coverage
from hndt.literature_baselines import policy_signature
from hndt.pareto import solve_pareto_hndt
from hndt.reporting import first_split, write_csv
from hndt.sensitivity import fault_weight_profile, perturb_cost_model


def _parse_levels(value: str) -> list[float]:
    levels = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not levels or any(x < 0 or x >= 1 for x in levels):
        raise ValueError("noise levels must satisfy 0 <= level < 1")
    return levels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--interval-costs", required=True)
    ap.add_argument("--query-costs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise-levels", default="0.01,0.025,0.05")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--fault-weights", choices=["uniform", "front", "back", "cost"], default="uniform")
    ap.add_argument("--terminal-only", action="store_true", help="Perturb terminal costs but leave query costs fixed")
    args = ap.parse_args()
    if args.trials <= 0:
        raise ValueError("trials must be positive")

    base = build_model(args.trace, args.interval_costs, args.query_costs, args.config)
    missing = validate_atomic_coverage(base)
    if missing:
        raise ValueError(f"missing atomic costs: {missing}")
    base_weights = fault_weight_profile(base, args.fault_weights)
    ref_scalar = solve_hndt(base)
    ref_pareto = solve_pareto_hndt(base, fault_weights=base_weights)
    ref_signature = policy_signature(ref_pareto.policy)
    ref_first = first_split(ref_pareto.policy, base.n)

    rows = []
    levels = _parse_levels(args.noise_levels)
    for level_index, noise in enumerate(levels):
        for trial in range(args.trials):
            seed = args.seed + level_index * 100000 + trial
            model = perturb_cost_model(
                base,
                relative_noise=noise,
                seed=seed,
                perturb_queries=not args.terminal_only,
            )
            weights = fault_weight_profile(model, args.fault_weights)
            scalar = solve_hndt(model)
            pareto = solve_pareto_hndt(model, fault_weights=weights)
            p_paths = evaluate_policy(model, pareto.policy)
            s_paths = evaluate_policy(model, scalar.action)
            p_mean = weighted_mean_cost(p_paths, weights)
            s_mean = weighted_mean_cost(s_paths, weights)
            exact_recovery = abs(pareto.worst_optimum - scalar.optimum) <= 1e-8
            rows.append(
                {
                    "noise_fraction": noise,
                    "noise_percent": noise * 100.0,
                    "trial": trial,
                    "seed": seed,
                    "pareto_worst": pareto.worst_optimum,
                    "scalar_worst": scalar.optimum,
                    "minimax_recovered": exact_recovery,
                    "pareto_mean": p_mean,
                    "scalar_mean": s_mean,
                    "mean_improvement_percent": 100.0 * (s_mean - p_mean) / s_mean if s_mean else 0.0,
                    "same_root_action": first_split(pareto.policy, model.n) == ref_first,
                    "same_full_policy": policy_signature(pareto.policy) == ref_signature,
                    "root_frontier_size": len(pareto.root_frontier),
                }
            )
            if not exact_recovery:
                raise AssertionError("Pareto minimax projection failed under perturbed costs")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "robustness_trials.csv", rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["noise_fraction"]].append(row)
    summary = []
    for noise in levels:
        subset = grouped[noise]
        gains = [float(r["mean_improvement_percent"]) for r in subset]
        summary.append(
            {
                "noise_fraction": noise,
                "noise_percent": noise * 100.0,
                "trials": len(subset),
                "minimax_recovery_rate": sum(bool(r["minimax_recovered"]) for r in subset) / len(subset),
                "same_root_action_rate": sum(bool(r["same_root_action"]) for r in subset) / len(subset),
                "same_full_policy_rate": sum(bool(r["same_full_policy"]) for r in subset) / len(subset),
                "mean_gain_percent_average": statistics.fmean(gains),
                "mean_gain_percent_min": min(gains),
                "mean_gain_percent_max": max(gains),
            }
        )
    write_csv(out / "robustness_summary.csv", summary)
    (out / "robustness_metadata.json").write_text(
        json.dumps(
            {
                "noise_levels": levels,
                "trials_per_level": args.trials,
                "seed": args.seed,
                "fault_weight_profile": args.fault_weights,
                "perturb_queries": not args.terminal_only,
                "reference_first_action": ref_first,
                "reference_worst": ref_scalar.optimum,
                "note": "Sensitivity analysis only; original measured inputs are never overwritten.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
