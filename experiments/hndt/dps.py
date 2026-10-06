from __future__ import annotations

"""Platform-independent Pareto dispute-policy synthesis (Pareto-DPS).

The model is intentionally independent of CKB, blockchains, Merkle trees,
neural networks, and any particular cost unit.  An instance supplies:

* an ordered deterministic execution with ``n`` primitive transitions;
* admissible terminal verification actions for each interval;
* admissible finite ordered partition/query actions for each interval;
* a fault-location weight distribution.

A query action may split an interval into two or more contiguous children.  The
legacy binary HNDT model is therefore a strict special case.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from math import isfinite
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .core import Action, CostModel
from .exact import as_fraction, normalize_weight_fractions

Interval = Tuple[int, int]


@dataclass(frozen=True)
class TerminalActionSpec:
    action_id: str
    cost: object
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id.strip():
            raise ValueError("terminal action_id must be a non-empty string")
        c = as_fraction(self.cost)
        if c < 0:
            raise ValueError("terminal action cost must be non-negative")
        object.__setattr__(self, "action_id", self.action_id.strip())
        object.__setattr__(self, "cost", c)
        object.__setattr__(
            self,
            "capabilities",
            frozenset(str(x).strip() for x in self.capabilities if str(x).strip()),
        )


@dataclass(frozen=True)
class QueryActionSpec:
    action_id: str
    cuts: Tuple[int, ...]
    cost: object

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id.strip():
            raise ValueError("query action_id must be a non-empty string")
        cuts = tuple(int(x) for x in self.cuts)
        if tuple(sorted(set(cuts))) != cuts:
            raise ValueError("query cuts must be strictly increasing and unique")
        c = as_fraction(self.cost)
        if c < 0:
            raise ValueError("query action cost must be non-negative")
        object.__setattr__(self, "action_id", self.action_id.strip())
        object.__setattr__(self, "cuts", cuts)
        object.__setattr__(self, "cost", c)

    def children(self, i: int, j: int) -> Tuple[Interval, ...]:
        if not self.cuts:
            raise ValueError("query action must contain at least one cut")
        if any(not (i < k < j) for k in self.cuts):
            raise ValueError(
                f"query cuts {self.cuts!r} must lie strictly inside interval [{i},{j}]"
            )
        points = (i,) + self.cuts + (j,)
        return tuple((points[r], points[r + 1]) for r in range(len(points) - 1))


@dataclass
class VerificationInstance:
    n: int
    terminal_actions: Mapping[Interval, Sequence[TerminalActionSpec]]
    query_actions: Callable[[int, int], Sequence[QueryActionSpec]]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n <= 0:
            raise ValueError("n must be a positive integer")
        if not callable(self.query_actions):
            raise TypeError("query_actions must be callable")
        self.metadata = dict(self.metadata)
        for (i, j), actions in self.terminal_actions.items():
            self.validate_interval(i, j)
            for action in actions:
                if not isinstance(action, TerminalActionSpec):
                    raise TypeError("terminal_actions must contain TerminalActionSpec values")

    def validate_interval(self, i: int, j: int) -> None:
        if not (0 <= i < j <= self.n):
            raise ValueError(f"invalid interval [{i},{j}] for n={self.n}")

    def terminals(self, i: int, j: int) -> Tuple[TerminalActionSpec, ...]:
        self.validate_interval(i, j)
        return tuple(self.terminal_actions.get((i, j), ()))

    def queries(self, i: int, j: int) -> Tuple[QueryActionSpec, ...]:
        self.validate_interval(i, j)
        out = []
        seen_ids = set()
        for query in self.query_actions(i, j):
            if not isinstance(query, QueryActionSpec):
                raise TypeError("query_actions must return QueryActionSpec values")
            if query.action_id in seen_ids:
                raise ValueError(
                    f"duplicate query action id {query.action_id!r} on interval [{i},{j}]"
                )
            seen_ids.add(query.action_id)
            children = query.children(i, j)
            if any(b - a >= j - i for a, b in children):
                raise ValueError("every query child must be strictly shorter than its parent")
            out.append(query)
        return tuple(out)


@dataclass(frozen=True)
class DPSAction:
    kind: str  # settle | query
    action_id: str
    cuts: Tuple[int, ...] = ()


@dataclass(frozen=True)
class DPSLabel:
    interval: Interval
    worst_exact: Fraction
    mass_exact: Fraction
    max_rounds: int
    action: DPSAction
    children: Tuple["DPSLabel", ...] = ()

    @property
    def worst_cost(self) -> float:
        return float(self.worst_exact)

    @property
    def expected_mass(self) -> float:
        return float(self.mass_exact)

    @property
    def mean_cost(self) -> float:
        # At the root the mass is exactly ordinary expected cost.  This alias
        # keeps the historical reporting vocabulary convenient.
        return float(self.mass_exact)


@dataclass
class ParetoDPSResult:
    n: int
    exact_weights: Tuple[Fraction, ...]
    frontier: Dict[Interval, Tuple[DPSLabel, ...]]
    selected: DPSLabel

    @property
    def root_frontier(self) -> Tuple[DPSLabel, ...]:
        return self.frontier[(0, self.n)]

    @property
    def worst_optimum_exact(self) -> Fraction:
        return self.selected.worst_exact

    @property
    def expected_at_worst_exact(self) -> Fraction:
        return self.selected.mass_exact

    @property
    def worst_optimum(self) -> float:
        return float(self.worst_optimum_exact)

    @property
    def expected_at_worst(self) -> float:
        return float(self.expected_at_worst_exact)


@dataclass
class ScalarDPSResult:
    n: int
    exact_value: Dict[Interval, Fraction]
    selected: DPSLabel

    @property
    def optimum_exact(self) -> Fraction:
        return self.exact_value[(0, self.n)]

    @property
    def optimum(self) -> float:
        return float(self.optimum_exact)


def _action_tie_key(label: DPSLabel) -> tuple:
    return (
        label.max_rounds,
        0 if label.action.kind == "settle" else 1,
        label.action.action_id,
        label.action.cuts,
    )


def prune_dps_frontier(labels: Iterable[DPSLabel]) -> Tuple[DPSLabel, ...]:
    """Exact ``O(C log C)`` pruning of a two-objective antichain."""

    ordered = sorted(
        labels,
        key=lambda x: (x.worst_exact, x.mass_exact, _action_tie_key(x)),
    )
    unique: list[DPSLabel] = []
    for label in ordered:
        if (
            unique
            and label.worst_exact == unique[-1].worst_exact
            and label.mass_exact == unique[-1].mass_exact
        ):
            if _action_tie_key(label) < _action_tie_key(unique[-1]):
                unique[-1] = label
        else:
            unique.append(label)

    frontier: list[DPSLabel] = []
    best_mass: Optional[Fraction] = None
    for label in unique:
        if best_mass is None or label.mass_exact < best_mass:
            frontier.append(label)
            best_mass = label.mass_exact
    return tuple(frontier)


def _prefix_weights(weights: Tuple[Fraction, ...]) -> list[Fraction]:
    prefix = [Fraction(0, 1)]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    return prefix


def solve_pareto_dps(
    instance: VerificationInstance,
    fault_weights: Optional[Sequence[float]] = None,
) -> ParetoDPSResult:
    """Compute the exact Pareto antichain for a finite-partition instance."""

    weights = normalize_weight_fractions(instance.n, fault_weights)
    prefix = _prefix_weights(weights)

    def mass(i: int, j: int) -> Fraction:
        return prefix[j] - prefix[i]

    frontier: Dict[Interval, Tuple[DPSLabel, ...]] = {}

    for length in range(1, instance.n + 1):
        for i in range(0, instance.n - length + 1):
            j = i + length
            interval_mass = mass(i, j)
            candidates: list[DPSLabel] = []

            for terminal in instance.terminals(i, j):
                cost = as_fraction(terminal.cost)
                candidates.append(
                    DPSLabel(
                        interval=(i, j),
                        worst_exact=cost,
                        mass_exact=interval_mass * cost,
                        max_rounds=0,
                        action=DPSAction("settle", terminal.action_id),
                    )
                )

            if length > 1:
                for query in instance.queries(i, j):
                    child_intervals = query.children(i, j)
                    if any(child not in frontier for child in child_intervals):
                        continue
                    child_frontiers = [frontier[child] for child in child_intervals]
                    if any(not values for values in child_frontiers):
                        continue
                    q = as_fraction(query.cost)
                    for child_labels in product(*child_frontiers):
                        candidates.append(
                            DPSLabel(
                                interval=(i, j),
                                worst_exact=q + max(x.worst_exact for x in child_labels),
                                mass_exact=(
                                    interval_mass * q
                                    + sum((x.mass_exact for x in child_labels), Fraction(0, 1))
                                ),
                                max_rounds=1 + max(x.max_rounds for x in child_labels),
                                action=DPSAction("query", query.action_id, query.cuts),
                                children=tuple(child_labels),
                            )
                        )

            frontier[(i, j)] = prune_dps_frontier(candidates) if candidates else tuple()

    root = frontier[(0, instance.n)]
    if not root:
        raise ValueError("root interval is unsolvable")
    selected = min(
        root,
        key=lambda x: (x.worst_exact, x.mass_exact, _action_tie_key(x)),
    )
    return ParetoDPSResult(instance.n, weights, frontier, selected)


def solve_scalar_dps(
    instance: VerificationInstance,
    fault_weights: Optional[Sequence[float]] = None,
) -> ScalarDPSResult:
    """Exact minimax-only reference solver over the same finite action space."""

    weights = normalize_weight_fractions(instance.n, fault_weights)
    prefix = _prefix_weights(weights)

    def mass(i: int, j: int) -> Fraction:
        return prefix[j] - prefix[i]

    value: Dict[Interval, Fraction] = {}
    chosen: Dict[Interval, DPSLabel] = {}

    for length in range(1, instance.n + 1):
        for i in range(0, instance.n - length + 1):
            j = i + length
            interval_mass = mass(i, j)
            candidates: list[DPSLabel] = []
            for terminal in instance.terminals(i, j):
                cost = as_fraction(terminal.cost)
                candidates.append(
                    DPSLabel(
                        (i, j),
                        cost,
                        interval_mass * cost,
                        0,
                        DPSAction("settle", terminal.action_id),
                    )
                )
            if length > 1:
                for query in instance.queries(i, j):
                    child_intervals = query.children(i, j)
                    if any(child not in chosen for child in child_intervals):
                        continue
                    children = tuple(chosen[child] for child in child_intervals)
                    q = as_fraction(query.cost)
                    candidates.append(
                        DPSLabel(
                            (i, j),
                            q + max(x.worst_exact for x in children),
                            interval_mass * q
                            + sum((x.mass_exact for x in children), Fraction(0, 1)),
                            1 + max(x.max_rounds for x in children),
                            DPSAction("query", query.action_id, query.cuts),
                            children,
                        )
                    )
            if not candidates:
                continue
            # Minimax only.  Exact ties prefer settlement, then fewer rounds,
            # then deterministic action identity.  Expected mass is deliberately
            # NOT used to choose among minimax ties.
            best = min(
                candidates,
                key=lambda x: (
                    x.worst_exact,
                    0 if x.action.kind == "settle" else 1,
                    x.max_rounds,
                    x.action.action_id,
                    x.action.cuts,
                ),
            )
            value[(i, j)] = best.worst_exact
            chosen[(i, j)] = best

    root = (0, instance.n)
    if root not in chosen:
        raise ValueError("root interval is unsolvable")
    return ScalarDPSResult(instance.n, value, chosen[root])


def reconstruct_dps_policy(label: DPSLabel) -> Dict[Interval, DPSAction]:
    policy: Dict[Interval, DPSAction] = {}

    def visit(node: DPSLabel) -> None:
        old = policy.get(node.interval)
        if old is not None and old != node.action:
            raise RuntimeError(f"inconsistent actions for interval {node.interval}")
        policy[node.interval] = node.action
        for child in node.children:
            visit(child)

    visit(label)
    return policy


def evaluate_dps_fault(
    instance: VerificationInstance,
    policy: Mapping[Interval, DPSAction],
    fault_index: int,
) -> tuple[Fraction, int, Interval, str]:
    if not 0 <= fault_index < instance.n:
        raise ValueError("fault_index out of range")
    interval = (0, instance.n)
    total = Fraction(0, 1)
    rounds = 0
    seen = set()
    while True:
        if interval in seen:
            raise RuntimeError("policy cycle detected")
        seen.add(interval)
        action = policy.get(interval)
        if action is None:
            raise ValueError(f"policy is missing interval {interval}")
        i, j = interval
        if action.kind == "settle":
            matches = [x for x in instance.terminals(i, j) if x.action_id == action.action_id]
            if len(matches) != 1:
                raise ValueError(f"terminal action {action.action_id!r} unavailable on {interval}")
            total += as_fraction(matches[0].cost)
            return total, rounds, interval, action.action_id
        if action.kind != "query":
            raise ValueError(f"invalid DPS action kind {action.kind!r}")
        matches = [x for x in instance.queries(i, j) if x.action_id == action.action_id]
        if len(matches) != 1:
            raise ValueError(f"query action {action.action_id!r} unavailable on {interval}")
        query = matches[0]
        total += as_fraction(query.cost)
        rounds += 1
        containing = [child for child in query.children(i, j) if child[0] <= fault_index < child[1]]
        if len(containing) != 1:
            raise AssertionError("query partition does not identify exactly one fault child")
        interval = containing[0]


def expected_cost_of_dps_policy(
    instance: VerificationInstance,
    policy: Mapping[Interval, DPSAction],
    fault_weights: Optional[Sequence[float]] = None,
) -> Fraction:
    weights = normalize_weight_fractions(instance.n, fault_weights)
    return sum(
        (
            weights[t] * evaluate_dps_fault(instance, policy, t)[0]
            for t in range(instance.n)
        ),
        Fraction(0, 1),
    )


def from_cost_model(model: CostModel) -> VerificationInstance:
    """Adapt the legacy binary :class:`CostModel` to generic Pareto-DPS."""

    terminals: dict[Interval, tuple[TerminalActionSpec, ...]] = {}
    for interval, backend_map in model.terminal_costs.items():
        actions = []
        for backend, raw in backend_map.items():
            if raw is None or not isfinite(float(raw)):
                continue
            actions.append(
                TerminalActionSpec(
                    backend,
                    raw,
                    capabilities=model.capabilities(backend),
                )
            )
        if actions:
            terminals[interval] = tuple(actions)

    def queries(i: int, j: int) -> tuple[QueryActionSpec, ...]:
        out = []
        for k in range(i + 1, j):
            q = model.checked_query_cost_exact(i, j, k)
            if q is not None:
                out.append(QueryActionSpec(f"binary@{k}", (k,), q))
        return tuple(out)

    metadata = dict(model.metadata)
    metadata.setdefault("source_model", "binary-cost-model")
    return VerificationInstance(model.n, terminals, queries, metadata)


def dps_to_legacy_policy(label: DPSLabel) -> Dict[Interval, Action]:
    """Convert a binary Pareto-DPS policy to the historical policy format."""

    dps_policy = reconstruct_dps_policy(label)
    legacy: Dict[Interval, Action] = {}
    for interval, action in dps_policy.items():
        if action.kind == "settle":
            legacy[interval] = Action.settle(action.action_id)
        elif action.kind == "query" and len(action.cuts) == 1:
            legacy[interval] = Action.split_at(action.cuts[0])
        else:
            raise ValueError(
                "legacy policy conversion is defined only for binary query actions"
            )
    return legacy
