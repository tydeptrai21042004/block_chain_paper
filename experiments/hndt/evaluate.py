from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite, isnan
from statistics import median
from typing import Dict, List, Mapping, Tuple

from .core import Action, CostModel, policy_reachable_intervals

Interval = Tuple[int, int]


@dataclass
class PathResult:
    fault_index: int
    total_cost: float
    rounds: int
    terminal_i: int
    terminal_j: int
    backend: str


def simulate_fault(
    model: CostModel,
    policy: Mapping[Interval, Action],
    fault_index: int,
) -> PathResult:
    """Follow the unique policy branch containing ``fault_index``.

    Transitions are indexed ``0,...,n-1`` and interval ``[i,j]`` contains
    transitions ``i,...,j-1``. A split at checkpoint ``k`` sends faults with
    index ``< k`` left and all remaining faults right.
    """

    if not isinstance(fault_index, int) or isinstance(fault_index, bool):
        raise TypeError("fault_index must be an integer")
    if not 0 <= fault_index < model.n:
        raise ValueError(f"fault_index must be in 0..{model.n - 1}")

    i, j = 0, model.n
    total = 0.0
    rounds = 0
    visited = set()

    while True:
        interval = (i, j)
        if interval in visited:
            raise RuntimeError(f"policy cycle detected at interval {interval}")
        visited.add(interval)
        if interval not in policy:
            raise ValueError(f"policy is missing action for interval {interval}")

        act = policy[interval]
        if act.kind == "settle":
            if act.backend is None:
                raise ValueError(f"settlement action on [{i},{j}] has no backend")
            backend = str(act.backend)
            raw_cost = model.backends(i, j).get(backend, inf)
            cost = float(raw_cost)
            if not isfinite(cost):
                raise ValueError(
                    f"Policy requests unavailable backend {backend!r} on [{i},{j}]"
                )
            total += cost
            if not (i <= fault_index < j):
                raise AssertionError(
                    f"policy terminated on [{i},{j}], which does not contain fault {fault_index}"
                )
            return PathResult(fault_index, total, rounds, i, j, backend)

        if act.kind != "split" or act.split is None:
            raise ValueError(f"invalid policy action {act!r} on interval [{i},{j}]")
        k = int(act.split)
        if not (i < k < j):
            raise ValueError(f"invalid split {k} on interval [{i},{j}]")
        q = model.checked_query_cost(i, j, k)
        if not isfinite(q):
            raise ValueError(f"policy uses unavailable infinite query on [{i},{j}] at {k}")
        if isnan(q):
            raise ValueError("query cost cannot be NaN")
        total += q
        rounds += 1
        if fault_index < k:
            j = k
        else:
            i = k


def evaluate_policy(
    model: CostModel,
    policy: Mapping[Interval, Action],
) -> List[PathResult]:
    # Validate the reachable policy structure once before the sweep.
    list(policy_reachable_intervals(policy, (0, model.n)))
    return [simulate_fault(model, policy, t) for t in range(model.n)]


def summarize(
    name: str,
    paths: List[PathResult],
    optimum: float | None = None,
) -> dict:
    if not paths:
        raise ValueError("paths must be non-empty")
    worst = max(p.total_cost for p in paths)
    best = min(p.total_cost for p in paths)
    mean = sum(p.total_cost for p in paths) / len(paths)
    worst_rounds = max(p.rounds for p in paths)
    mean_rounds = sum(p.rounds for p in paths) / len(paths)

    unique_leaves = {(p.terminal_i, p.terminal_j, p.backend) for p in paths}
    leaf_sizes = [j - i for i, j, _ in unique_leaves]
    native_leaves = sum(1 for _, _, b in unique_leaves if b == "native")
    zk_leaves = sum(1 for _, _, b in unique_leaves if b == "zkvm")
    zk_fault_paths = sum(1 for p in paths if p.backend == "zkvm")

    ratio = worst / optimum if optimum not in (None, 0) else None
    return {
        "strategy": name,
        "worst_case_cost": worst,
        "best_case_cost": best,
        "mean_cost": mean,
        "max_rounds": worst_rounds,
        "mean_rounds": mean_rounds,
        "median_leaf_size": median(leaf_sizes),
        "min_leaf_size": min(leaf_sizes),
        "max_leaf_size": max(leaf_sizes),
        "num_terminal_intervals": len(unique_leaves),
        "native_leaf_count": native_leaves,
        "zk_leaf_count": zk_leaves,
        "zk_fault_paths": zk_fault_paths,
        "ratio_to_optimum": ratio,
        "first_split": None,
    }
