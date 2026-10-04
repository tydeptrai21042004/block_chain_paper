from __future__ import annotations

"""Exact Pareto extension of HNDT.

The original HNDT dynamic program minimizes only the adversarial worst-case
fault-path cost.  This module keeps the same stop-or-split action space but
computes the nondominated frontier of

    (worst-case fault-path cost, weighted mean fault-path cost).

No scalarization coefficient is introduced.  The primary proposal is selected
lexicographically: first preserve the exact HNDT minimax optimum, then minimize
weighted mean cost among *all* minimax-optimal policies.

For normalized fault weights p_t and interval mass P_ij = sum_{t=i}^{j-1} p_t,
terminal settlement with cost A contributes

    (A, P_ij A),

while a split with query cost q composes child labels (W_L, M_L) and
(W_R, M_R) as

    (q + max(W_L, W_R), P_ij q + M_L + M_R).

Dominated labels can be discarded exactly because both composition operators
are monotone in each child objective.
"""

from dataclasses import dataclass
from math import inf, isfinite, isnan
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .core import Action, CostModel

Interval = Tuple[int, int]


@dataclass(frozen=True)
class ParetoLabel:
    """One achievable nondominated objective pair for an interval."""

    interval: Interval
    worst_cost: float
    mean_cost: float
    max_rounds: int
    action: Action
    left: Optional["ParetoLabel"] = None
    right: Optional["ParetoLabel"] = None


@dataclass
class ParetoHNDTResult:
    """Exact Pareto-HNDT result for all intervals."""

    n: int
    weights: Tuple[float, ...]
    frontier: Dict[Interval, Tuple[ParetoLabel, ...]]
    selected: ParetoLabel
    policy: Dict[Interval, Action]

    @property
    def worst_optimum(self) -> float:
        return self.selected.worst_cost

    @property
    def mean_at_worst_optimum(self) -> float:
        return self.selected.mean_cost

    @property
    def max_rounds(self) -> int:
        return self.selected.max_rounds

    @property
    def root_frontier(self) -> Tuple[ParetoLabel, ...]:
        return self.frontier[(0, self.n)]


def normalize_fault_weights(n: int, weights: Optional[Sequence[float]] = None) -> Tuple[float, ...]:
    """Validate and normalize nonnegative fault weights to sum to one.

    ``None`` means a uniform distribution, so the second objective is the
    arithmetic mean over all possible single-fault positions.  This is the
    manuscript's distribution-free diagnostic: every position receives equal
    weight and no learned fault model is required.
    """

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    if weights is None:
        return tuple(1.0 / n for _ in range(n))
    if len(weights) != n:
        raise ValueError(f"fault weights must have length {n}, got {len(weights)}")
    values = []
    for index, raw in enumerate(weights):
        value = float(raw)
        if isnan(value) or not isfinite(value) or value < 0:
            raise ValueError(
                f"fault weight at position {index} must be finite and non-negative"
            )
        values.append(value)
    total = sum(values)
    if total <= 0:
        raise ValueError("at least one fault weight must be positive")
    return tuple(v / total for v in values)


def _action_tie_key(label: ParetoLabel) -> tuple:
    """Deterministic tertiary ordering that never changes the (W, M) pair."""

    act = label.action
    if act.kind == "settle":
        backend = act.backend or ""
        return (
            label.max_rounds,
            0,
            0 if backend == "native" else 1,
            backend,
            -1,
        )
    split = -1 if act.split is None else int(act.split)
    i, j = label.interval
    midpoint = (i + j) / 2.0
    return (label.max_rounds, 1, 2, "", abs(split - midpoint), split)


def _equivalent(a: ParetoLabel, b: ParetoLabel, tol: float) -> bool:
    return (
        abs(a.worst_cost - b.worst_cost) <= tol
        and abs(a.mean_cost - b.mean_cost) <= tol
    )


def _dominates(a: ParetoLabel, b: ParetoLabel, tol: float) -> bool:
    no_worse = (
        a.worst_cost <= b.worst_cost + tol
        and a.mean_cost <= b.mean_cost + tol
    )
    strictly_better = (
        a.worst_cost < b.worst_cost - tol
        or a.mean_cost < b.mean_cost - tol
    )
    return no_worse and strictly_better


def prune_nondominated(
    labels: Iterable[ParetoLabel],
    tie_tolerance: float = 1e-12,
) -> Tuple[ParetoLabel, ...]:
    """Return a deterministic exact 2-D nondominated frontier.

    Equal objective pairs are collapsed using only tertiary policy preferences
    (fewer rounds, then deterministic action ordering).  Dominance itself is
    based solely on the two scientific objectives.
    """

    if tie_tolerance < 0 or isnan(float(tie_tolerance)):
        raise ValueError("tie_tolerance must be non-negative and not NaN")

    finite = [
        x
        for x in labels
        if isfinite(x.worst_cost) and isfinite(x.mean_cost)
    ]
    finite.sort(key=lambda x: (x.worst_cost, x.mean_cost, _action_tie_key(x)))

    unique: list[ParetoLabel] = []
    for label in finite:
        duplicate_index = next(
            (idx for idx, kept in enumerate(unique) if _equivalent(label, kept, tie_tolerance)),
            None,
        )
        if duplicate_index is None:
            unique.append(label)
        elif _action_tie_key(label) < _action_tie_key(unique[duplicate_index]):
            unique[duplicate_index] = label

    frontier = []
    for label in unique:
        if any(
            other is not label and _dominates(other, label, tie_tolerance)
            for other in unique
        ):
            continue
        frontier.append(label)

    frontier.sort(key=lambda x: (x.worst_cost, x.mean_cost, _action_tie_key(x)))
    return tuple(frontier)


def reconstruct_policy(label: ParetoLabel) -> Dict[Interval, Action]:
    """Reconstruct the reachable policy induced by one root label."""

    policy: Dict[Interval, Action] = {}

    def visit(node: ParetoLabel) -> None:
        previous = policy.get(node.interval)
        if previous is not None and previous != node.action:
            raise RuntimeError(
                f"inconsistent reconstructed actions for interval {node.interval}: "
                f"{previous!r} vs {node.action!r}"
            )
        policy[node.interval] = node.action
        if node.action.kind == "split":
            if node.left is None or node.right is None:
                raise RuntimeError(f"split label {node.interval} is missing child labels")
            visit(node.left)
            visit(node.right)

    visit(label)
    return policy


def select_frontier_label(
    labels: Sequence[ParetoLabel],
    *,
    mode: str = "minimax_safe",
) -> ParetoLabel:
    """Select one deterministic policy from a Pareto frontier.

    ``minimax_safe`` minimizes worst-case cost first and mean cost second.
    ``mean_first`` is an ablation endpoint that minimizes mean cost first and
    worst-case cost second.
    """

    if not labels:
        raise ValueError("cannot select from an empty Pareto frontier")
    if mode == "minimax_safe":
        key = lambda x: (x.worst_cost, x.mean_cost, _action_tie_key(x))
    elif mode == "mean_first":
        key = lambda x: (x.mean_cost, x.worst_cost, _action_tie_key(x))
    else:
        raise ValueError("mode must be 'minimax_safe' or 'mean_first'")
    return min(labels, key=key)


def solve_pareto_hndt(
    model: CostModel,
    fault_weights: Optional[Sequence[float]] = None,
    tie_tolerance: float = 1e-12,
) -> ParetoHNDTResult:
    """Compute the exact Pareto frontier for HNDT's stop-or-split strategy space.

    The straightforward algorithm is output-sensitive.  If ``P`` is the
    largest interval-frontier size, candidate generation is O(n^3 P^2) in the
    worst case before dominance pruning; storage is O(n^2 P).
    """

    if tie_tolerance < 0 or isnan(float(tie_tolerance)):
        raise ValueError("tie_tolerance must be non-negative and not NaN")

    n = model.n
    weights = normalize_fault_weights(n, fault_weights)
    prefix = [0.0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)

    def mass(i: int, j: int) -> float:
        return prefix[j] - prefix[i]

    frontier: Dict[Interval, Tuple[ParetoLabel, ...]] = {}

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            candidates: list[ParetoLabel] = []
            interval_mass = mass(i, j)

            for backend, raw_cost in model.backends(i, j).items():
                cost = float(raw_cost)
                if not isfinite(cost):
                    continue
                candidates.append(
                    ParetoLabel(
                        interval=(i, j),
                        worst_cost=cost,
                        mean_cost=interval_mass * cost,
                        max_rounds=0,
                        action=Action.settle(backend),
                    )
                )

            if length > 1:
                for k in range(i + 1, j):
                    q = model.checked_query_cost(i, j, k)
                    if not isfinite(q):
                        continue
                    left_frontier = frontier.get((i, k), ())
                    right_frontier = frontier.get((k, j), ())
                    for left in left_frontier:
                        for right in right_frontier:
                            candidates.append(
                                ParetoLabel(
                                    interval=(i, j),
                                    worst_cost=q + max(left.worst_cost, right.worst_cost),
                                    mean_cost=(
                                        interval_mass * q
                                        + left.mean_cost
                                        + right.mean_cost
                                    ),
                                    max_rounds=1 + max(left.max_rounds, right.max_rounds),
                                    action=Action.split_at(k),
                                    left=left,
                                    right=right,
                                )
                            )

            current = prune_nondominated(candidates, tie_tolerance=tie_tolerance)
            if not current:
                raise ValueError(
                    f"Interval [{i},{j}] is unsolvable in Pareto-HNDT: no finite "
                    "settlement and no finite split with solvable children"
                )
            frontier[(i, j)] = current

    root_frontier = frontier[(0, n)]
    selected = select_frontier_label(root_frontier, mode="minimax_safe")
    policy = reconstruct_policy(selected)
    return ParetoHNDTResult(
        n=n,
        weights=weights,
        frontier=frontier,
        selected=selected,
        policy=policy,
    )


def mean_first_policy(result: ParetoHNDTResult) -> Dict[Interval, Action]:
    """Return the mean-first endpoint of an already computed Pareto frontier."""

    label = select_frontier_label(result.root_frontier, mode="mean_first")
    return reconstruct_policy(label)


def frontier_rows(result: ParetoHNDTResult) -> list[dict]:
    """Machine-readable root frontier for plotting/auditing."""

    selected = result.selected
    mean_first = select_frontier_label(result.root_frontier, mode="mean_first")
    rows = []
    for index, label in enumerate(result.root_frontier, start=1):
        action = label.action
        rows.append(
            {
                "frontier_id": index,
                "worst_case_cost": label.worst_cost,
                "mean_cost": label.mean_cost,
                "max_rounds": label.max_rounds,
                "root_action": action.kind,
                "root_backend": action.backend or "",
                "root_split": "" if action.split is None else action.split,
                "selected_minimax_safe": label is selected,
                "selected_mean_first": label is mean_first,
            }
        )
    return rows
