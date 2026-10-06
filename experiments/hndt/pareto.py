from __future__ import annotations

"""Exact Pareto extension of HNDT.

The stop-or-split action space is unchanged.  The solver computes the exact
nondominated frontier of two objectives:

    (worst-case fault-path cost, weighted expected-cost contribution).

All Pareto comparisons use rational arithmetic.  No epsilon/tolerance is used
for scientific objective equality or dominance.  Public reporting properties
remain floats for compatibility with the existing CSV/plotting pipeline.

For normalized fault weights p_t and interval mass P_ij, terminal settlement
with cost A contributes

    (A, P_ij A),

and a split with query cost q composes child labels (W_L, M_L), (W_R, M_R) as

    (q + max(W_L, W_R), P_ij q + M_L + M_R).

The second coordinate is deliberately an *unnormalized expectation mass* on a
subinterval.  At the root P_0n = 1, so it equals the ordinary expected cost.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isnan
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

from .core import Action, CostModel
from .exact import as_fraction, normalize_weight_fractions

Interval = Tuple[int, int]
SplitFilter = Callable[[int, int, int], bool]
TerminalFilter = Callable[[int, int, str], bool]


@dataclass(frozen=True)
class ParetoLabel:
    """One achievable nondominated objective pair for an interval."""

    interval: Interval
    worst_exact: Fraction
    mean_exact: Fraction
    max_rounds: int
    action: Action
    left: Optional["ParetoLabel"] = None
    right: Optional["ParetoLabel"] = None

    @property
    def worst_cost(self) -> float:
        return float(self.worst_exact)

    @property
    def mean_cost(self) -> float:
        return float(self.mean_exact)


@dataclass
class ParetoStats:
    """Optional solver instrumentation; never participates in policy decisions."""

    intervals: int = 0
    candidates_generated: int = 0
    labels_retained: int = 0
    duplicate_pruned: int = 0
    dominated_pruned: int = 0
    peak_frontier_size: int = 0
    peak_candidates_per_interval: int = 0

    @property
    def pruning_ratio(self) -> float:
        if self.candidates_generated == 0:
            return 0.0
        return 1.0 - (self.labels_retained / self.candidates_generated)


@dataclass
class ParetoHNDTResult:
    """Exact Pareto-HNDT result for all intervals."""

    n: int
    weights: Tuple[float, ...]
    exact_weights: Tuple[Fraction, ...]
    frontier: Dict[Interval, Tuple[ParetoLabel, ...]]
    selected: ParetoLabel
    policy: Dict[Interval, Action]
    stats: Optional[ParetoStats] = None

    @property
    def worst_optimum(self) -> float:
        return self.selected.worst_cost

    @property
    def worst_optimum_exact(self) -> Fraction:
        return self.selected.worst_exact

    @property
    def mean_at_worst_optimum(self) -> float:
        return self.selected.mean_cost

    @property
    def mean_at_worst_optimum_exact(self) -> Fraction:
        return self.selected.mean_exact

    @property
    def max_rounds(self) -> int:
        return self.selected.max_rounds

    @property
    def root_frontier(self) -> Tuple[ParetoLabel, ...]:
        return self.frontier[(0, self.n)]


def normalize_fault_weights(
    n: int,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[float, ...]:
    """Backward-compatible float view of the exactly normalized weights."""

    return tuple(float(x) for x in normalize_weight_fractions(n, weights))


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
    # Compare twice the distance to the midpoint using integers.
    midpoint_distance2 = abs(2 * split - (i + j))
    return (label.max_rounds, 1, 2, "", midpoint_distance2, split)


def _equivalent(a: ParetoLabel, b: ParetoLabel) -> bool:
    return a.worst_exact == b.worst_exact and a.mean_exact == b.mean_exact


def _dominates(a: ParetoLabel, b: ParetoLabel) -> bool:
    return (
        a.worst_exact <= b.worst_exact
        and a.mean_exact <= b.mean_exact
        and (a.worst_exact < b.worst_exact or a.mean_exact < b.mean_exact)
    )


def _validate_legacy_tolerance(tie_tolerance: float) -> None:
    # Retain the old argument so external scripts do not break, but objective
    # comparisons are exact and do not use this tolerance.
    if tie_tolerance < 0 or isnan(float(tie_tolerance)):
        raise ValueError("tie_tolerance must be non-negative and not NaN")


def _prune_nondominated_with_counts(
    labels: Iterable[ParetoLabel],
    tie_tolerance: float = 0.0,
) -> tuple[Tuple[ParetoLabel, ...], int, int, int]:
    """Exact O(C log C) 2-D Pareto pruning for ``C`` candidate labels.

    Candidates are sorted by increasing worst-case cost and then expected-cost
    mass.  Exact duplicate objective pairs are collapsed by tertiary policy
    preference.  A single scan then retains a label iff its second coordinate
    is strictly smaller than every earlier label's second coordinate.
    """

    _validate_legacy_tolerance(tie_tolerance)
    ordered = sorted(
        labels,
        key=lambda x: (x.worst_exact, x.mean_exact, _action_tie_key(x)),
    )
    finite_candidates = len(ordered)

    unique: list[ParetoLabel] = []
    for label in ordered:
        if unique and _equivalent(label, unique[-1]):
            if _action_tie_key(label) < _action_tie_key(unique[-1]):
                unique[-1] = label
        else:
            unique.append(label)

    frontier: list[ParetoLabel] = []
    best_mean: Fraction | None = None
    for label in unique:
        if best_mean is None or label.mean_exact < best_mean:
            frontier.append(label)
            best_mean = label.mean_exact

    duplicate_pruned = finite_candidates - len(unique)
    dominated_pruned = len(unique) - len(frontier)
    return tuple(frontier), duplicate_pruned, dominated_pruned, finite_candidates


def prune_nondominated(
    labels: Iterable[ParetoLabel],
    tie_tolerance: float = 0.0,
) -> Tuple[ParetoLabel, ...]:
    """Return a deterministic exact 2-D nondominated frontier."""

    frontier, _, _, _ = _prune_nondominated_with_counts(labels, tie_tolerance)
    return frontier


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
    """Select one deterministic policy from a Pareto frontier."""

    if not labels:
        raise ValueError("cannot select from an empty Pareto frontier")
    if mode == "minimax_safe":
        key = lambda x: (x.worst_exact, x.mean_exact, _action_tie_key(x))
    elif mode == "mean_first":
        key = lambda x: (x.mean_exact, x.worst_exact, _action_tie_key(x))
    else:
        raise ValueError("mode must be 'minimax_safe' or 'mean_first'")
    return min(labels, key=key)


def solve_pareto_hndt(
    model: CostModel,
    fault_weights: Optional[Sequence[float]] = None,
    tie_tolerance: float = 0.0,
    collect_stats: bool = False,
    *,
    split_filter: Optional[SplitFilter] = None,
    terminal_filter: Optional[TerminalFilter] = None,
) -> ParetoHNDTResult:
    """Compute the exact Pareto frontier for a stop-or-split strategy space.

    ``split_filter`` and ``terminal_filter`` are restriction hooks used only by
    fair mechanism ablations.  The proposal leaves both unset and therefore
    searches the complete HNDT action space.

    If ``P`` is the largest interval-frontier size, candidate generation remains
    O(n^3 P^2).  Each interval frontier is pruned in O(C log C) time for ``C``
    generated candidates, rather than by a quadratic all-pairs dominance scan.
    """

    _validate_legacy_tolerance(tie_tolerance)

    n = model.n
    exact_weights = normalize_weight_fractions(n, fault_weights)
    prefix = [Fraction(0, 1)]
    for weight in exact_weights:
        prefix.append(prefix[-1] + weight)

    def mass(i: int, j: int) -> Fraction:
        return prefix[j] - prefix[i]

    frontier: Dict[Interval, Tuple[ParetoLabel, ...]] = {}
    stats = ParetoStats() if collect_stats else None

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            candidates: list[ParetoLabel] = []
            interval_mass = mass(i, j)

            for backend, raw_cost in model.backends(i, j).items():
                if terminal_filter is not None and not terminal_filter(i, j, backend):
                    continue
                try:
                    cost = as_fraction(raw_cost)
                except ValueError:
                    continue
                candidates.append(
                    ParetoLabel(
                        interval=(i, j),
                        worst_exact=cost,
                        mean_exact=interval_mass * cost,
                        max_rounds=0,
                        action=Action.settle(backend),
                    )
                )

            if length > 1:
                for k in range(i + 1, j):
                    if split_filter is not None and not split_filter(i, j, k):
                        continue
                    try:
                        q = as_fraction(model.checked_query_cost(i, j, k))
                    except ValueError:
                        continue
                    left_frontier = frontier.get((i, k), ())
                    right_frontier = frontier.get((k, j), ())
                    for left in left_frontier:
                        for right in right_frontier:
                            candidates.append(
                                ParetoLabel(
                                    interval=(i, j),
                                    worst_exact=q + max(left.worst_exact, right.worst_exact),
                                    mean_exact=(
                                        interval_mass * q
                                        + left.mean_exact
                                        + right.mean_exact
                                    ),
                                    max_rounds=1 + max(left.max_rounds, right.max_rounds),
                                    action=Action.split_at(k),
                                    left=left,
                                    right=right,
                                )
                            )

            current, duplicate_pruned, dominated_pruned, finite_candidates = (
                _prune_nondominated_with_counts(candidates, tie_tolerance=tie_tolerance)
            )
            if stats is not None:
                stats.intervals += 1
                stats.candidates_generated += finite_candidates
                stats.labels_retained += len(current)
                stats.duplicate_pruned += duplicate_pruned
                stats.dominated_pruned += dominated_pruned
                stats.peak_frontier_size = max(stats.peak_frontier_size, len(current))
                stats.peak_candidates_per_interval = max(
                    stats.peak_candidates_per_interval, finite_candidates
                )
            if not current:
                raise ValueError(
                    f"Interval [{i},{j}] is unsolvable in Pareto-HNDT: no finite "
                    "settlement and no allowed finite split with solvable children"
                )
            frontier[(i, j)] = current

    root_frontier = frontier[(0, n)]
    selected = select_frontier_label(root_frontier, mode="minimax_safe")
    policy = reconstruct_policy(selected)
    return ParetoHNDTResult(
        n=n,
        weights=tuple(float(x) for x in exact_weights),
        exact_weights=exact_weights,
        frontier=frontier,
        selected=selected,
        policy=policy,
        stats=stats,
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
                "worst_case_cost_exact": str(label.worst_exact),
                "mean_cost_exact": str(label.mean_exact),
                "max_rounds": label.max_rounds,
                "root_action": action.kind,
                "root_backend": action.backend or "",
                "root_split": "" if action.split is None else action.split,
                "selected_minimax_safe": label is selected,
                "selected_mean_first": label is mean_first,
            }
        )
    return rows
