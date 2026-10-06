#!/usr/bin/env python3
from __future__ import annotations

"""Executable checks for the compositional claims used in the JLAMP revision.

This script is intentionally measurement-free.  It exports small exact examples
that explain why a single locally lexicographic label is not a sufficient DP
summary and why the nondominated antichain is required.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

from hndt.core import Action
from hndt.pareto import ParetoLabel, prune_nondominated


def _pair(label: ParetoLabel) -> tuple[Fraction, Fraction]:
    return label.worst_exact, label.mean_exact


def ancestor_slack_counterexample() -> dict:
    left = ParetoLabel(
        interval=(0, 1),
        worst_exact=Fraction(10),
        mean_exact=Fraction(0),
        max_rounds=0,
        action=Action.settle("left"),
    )
    right_local_minimax = ParetoLabel(
        interval=(1, 2),
        worst_exact=Fraction(1),
        mean_exact=Fraction(100),
        max_rounds=0,
        action=Action.settle("right_minimax"),
    )
    right_slack_exploiting = ParetoLabel(
        interval=(1, 2),
        worst_exact=Fraction(9),
        mean_exact=Fraction(0),
        max_rounds=0,
        action=Action.settle("right_slack"),
    )
    frontier = prune_nondominated([right_local_minimax, right_slack_exploiting])
    local = min(frontier, key=lambda x: (x.worst_exact, x.mean_exact))

    def compose(right: ParetoLabel) -> tuple[Fraction, Fraction]:
        return max(left.worst_exact, right.worst_exact), left.mean_exact + right.mean_exact

    local_parent = compose(local)
    full_parent = min(compose(right) for right in frontier)
    assert local_parent == (Fraction(10), Fraction(100))
    assert full_parent == (Fraction(10), Fraction(0))

    return {
        "claim": "local lexicographic minimization is not compositionally sufficient",
        "left_bottleneck": [str(x) for x in _pair(left)],
        "right_frontier": [[str(x) for x in _pair(label)] for label in frontier],
        "parent_using_local_minimax": [str(x) for x in local_parent],
        "parent_using_full_frontier": [str(x) for x in full_parent],
        "same_parent_worst_case": local_parent[0] == full_parent[0],
        "strict_parent_expected_improvement": full_parent[1] < local_parent[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="experiments/results/theory_checks.json",
        help="Output JSON path",
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arithmetic": "exact rational",
        "dominance_tolerance": 0,
        "ancestor_slack_counterexample": ancestor_slack_counterexample(),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
