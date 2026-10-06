#!/usr/bin/env python3
from __future__ import annotations

"""Platform-independent cost-regime study for Pareto-DPS.

This is not a CKB simulation.  It uses dimensionless normalized verification
costs to expose when heterogeneous terminal costs and interaction costs create a
non-trivial Pareto opportunity.
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hndt.core import CostModel, solve_hndt
from hndt.dps import from_cost_model, solve_pareto_dps
from hndt.evaluate import evaluate_policy, weighted_mean_cost
from hndt.pareto import solve_pareto_hndt


def parse_args():
    p = argparse.ArgumentParser(description="Run normalized Pareto-DPS cost-regime sweeps")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--max-terminal-span", type=int, default=4)
    p.add_argument("--heterogeneity", default="0,0.25,0.5,1,2")
    p.add_argument("--query-ratio", default="0.1,0.25,0.5,1,2,4,8")
    p.add_argument("--out", required=True)
    return p.parse_args()


def make_model(n: int, h: float, rho: float, max_span: int) -> CostModel:
    if n <= 1:
        raise ValueError("n must be > 1")
    # Deterministic centered shape; h=0 gives homogeneous terminal work.
    shape = [((7 * t + 3) % n) / max(1, n - 1) for t in range(n)]
    atom = [1.0 + h * x for x in shape]
    typical = sorted(atom)[n // 2]
    q = rho * typical

    terminal = {}
    for span in range(1, min(max_span, n) + 1):
        for i in range(n - span + 1):
            j = i + span
            # Generic replay/check cost: additive work plus a small interval
            # setup term.  No platform-specific quantity is assumed.
            terminal[(i, j)] = {"replay": sum(atom[i:j]) + 0.05 * span}

    return CostModel(
        n,
        terminal,
        lambda i, j, k: q,
        terminal_capabilities={"replay": {"replay", "one-step"}},
        metadata={
            "platform": "normalized-generic",
            "cost_unit": "normalized-unit",
            "trace_granularity": "primitive-transition",
        },
    )


def main():
    args = parse_args()
    if args.n <= 1:
        raise ValueError("--n must be > 1")
    hs = [float(x) for x in args.heterogeneity.split(",") if x.strip()]
    rhos = [float(x) for x in args.query_ratio.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for h in hs:
        for rho in rhos:
            model = make_model(args.n, h, rho, args.max_terminal_span)
            scalar = solve_hndt(model)
            pareto = solve_pareto_hndt(model)
            generic = solve_pareto_dps(from_cost_model(model))
            if generic.worst_optimum_exact != pareto.worst_optimum_exact:
                raise AssertionError("generic and binary Pareto solvers disagree")
            scalar_paths = evaluate_policy(model, scalar.action)
            scalar_mean = weighted_mean_cost(scalar_paths)
            pareto_paths = evaluate_policy(model, pareto.policy)
            pareto_mean = weighted_mean_cost(pareto_paths)
            gain = 0.0 if scalar_mean == 0 else 100.0 * (scalar_mean - pareto_mean) / scalar_mean
            mean_first = min(pareto.root_frontier, key=lambda x: (x.mean_exact, x.worst_exact))
            rows.append(
                {
                    "n": args.n,
                    "heterogeneity": h,
                    "query_ratio": rho,
                    "root_frontier_size": len(pareto.root_frontier),
                    "scalar_worst": scalar.optimum,
                    "pareto_worst": pareto.worst_optimum,
                    "scalar_expected": scalar_mean,
                    "pareto_expected": pareto_mean,
                    "minimax_safe_expected_gain_pct": gain,
                    "mean_first_worst_ratio": mean_first.worst_cost / pareto.worst_optimum,
                    "pareto_max_rounds": pareto.max_rounds,
                }
            )

    path = out / "cost_regimes.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
