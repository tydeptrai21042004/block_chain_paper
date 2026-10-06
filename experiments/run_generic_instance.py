#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hndt.dps import (
    expected_cost_of_dps_policy,
    reconstruct_dps_policy,
    solve_pareto_dps,
    solve_scalar_dps,
)
from hndt.generic_io import build_verification_instance


def parse_args():
    p = argparse.ArgumentParser(
        description="Run platform-independent Pareto-DPS on a generic verification instance"
    )
    p.add_argument("--trace", required=True)
    p.add_argument("--terminal-actions", required=True)
    p.add_argument("--query-actions", required=True)
    p.add_argument("--metadata", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    instance = build_verification_instance(
        args.trace,
        args.terminal_actions,
        args.query_actions,
        args.metadata,
    )
    pareto = solve_pareto_dps(instance)
    scalar = solve_scalar_dps(instance)
    pareto_policy = reconstruct_dps_policy(pareto.selected)
    scalar_policy = reconstruct_dps_policy(scalar.selected)
    scalar_expected = expected_cost_of_dps_policy(instance, scalar_policy)

    with (out / "pareto_frontier.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["worst_cost", "expected_cost", "max_rounds", "root_action", "root_cuts"],
        )
        writer.writeheader()
        for label in pareto.root_frontier:
            writer.writerow(
                {
                    "worst_cost": float(label.worst_exact),
                    "expected_cost": float(label.mass_exact),
                    "max_rounds": label.max_rounds,
                    "root_action": label.action.action_id,
                    "root_cuts": ";".join(str(x) for x in label.action.cuts),
                }
            )

    summary = {
        "method": "Pareto-DPS",
        "n": instance.n,
        "metadata": dict(instance.metadata),
        "pareto_worst": float(pareto.worst_optimum_exact),
        "pareto_expected": float(pareto.expected_at_worst_exact),
        "scalar_worst": float(scalar.optimum_exact),
        "scalar_expected": float(scalar_expected),
        "minimax_preserved": pareto.worst_optimum_exact == scalar.optimum_exact,
        "root_frontier_size": len(pareto.root_frontier),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def dump_policy(path, policy):
        rows = {}
        for (i, j), action in sorted(policy.items()):
            rows[f"{i}:{j}"] = {
                "kind": action.kind,
                "action_id": action.action_id,
                "cuts": list(action.cuts),
            }
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    dump_policy(out / "pareto_policy.json", pareto_policy)
    dump_policy(out / "scalar_policy.json", scalar_policy)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
