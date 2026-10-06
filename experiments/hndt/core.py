from __future__ import annotations

"""Platform-neutral binary dispute-policy primitives.

This module intentionally contains no blockchain-, VM-, Merkle-, neural-, or
unit-specific assumptions.  A :class:`CostModel` is only an ordered trace with
finite terminal actions and a binary checkpoint-query cost oracle.

All scientific minimax decisions are made with exact rational arithmetic.
Float-valued views are retained solely for backwards-compatible reporting.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from math import inf, isfinite, isnan
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from .exact import as_fraction

Interval = Tuple[int, int]


@dataclass(frozen=True)
class Action:
    """One policy action for a disputed interval."""

    kind: str  # "settle" or "split"
    backend: Optional[str] = None
    split: Optional[int] = None

    @staticmethod
    def settle(backend: str) -> "Action":
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("settlement backend must be a non-empty string")
        return Action(kind="settle", backend=backend.strip())

    @staticmethod
    def split_at(k: int) -> "Action":
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("split index must be an integer")
        return Action(kind="split", split=k)


@dataclass
class CostModel:
    """Ordered binary verification-cost model.

    Parameters
    ----------
    n:
        Number of primitive transitions in the committed execution.
    terminal_costs:
        Mapping ``(i,j) -> {action_id: cost}``.  The action identifiers are
        opaque to the optimizer.  Missing actions are unavailable; ``+inf`` is
        also accepted as an explicit unavailable value.
    query_cost:
        Function ``q(i,j,k)`` returning the cost of querying checkpoint ``k``
        inside active interval ``[i,j]``.
    terminal_capabilities:
        Optional platform-neutral tags attached to terminal action identifiers,
        for example ``{"replay", "one-step"}`` or ``{"zk-proof"}``.
        Baseline adapters use these tags instead of hard-coded backend names.
    metadata:
        Free-form experiment metadata.  It never participates in optimization.
    backend_priority:
        Optional deterministic tie-break priority for equal-cost terminal
        actions.  Smaller numbers win; lexical action id is the final tie-break.

    The legacy class name ``CostModel`` is retained because the public scripts
    already import it.  The general finite-partition proposal lives in
    :mod:`hndt.dps` and adapts this binary model without platform assumptions.
    """

    n: int
    terminal_costs: Mapping[Interval, Mapping[str, float]]
    query_cost: Callable[[int, int, int], float]
    terminal_capabilities: Mapping[str, Iterable[str]] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    backend_priority: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n <= 0:
            raise ValueError("n must be a positive integer")
        if not callable(self.query_cost):
            raise TypeError("query_cost must be callable")

        normalized_caps: dict[str, frozenset[str]] = {}
        for name, tags in dict(self.terminal_capabilities).items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("terminal capability action ids must be non-empty strings")
            if isinstance(tags, str):
                values = [tags]
            else:
                values = list(tags)
            clean = frozenset(str(x).strip() for x in values if str(x).strip())
            normalized_caps[name.strip()] = clean
        self.terminal_capabilities = normalized_caps
        self.metadata = dict(self.metadata)
        self.backend_priority = {str(k): int(v) for k, v in dict(self.backend_priority).items()}

        for interval, backends in self.terminal_costs.items():
            if not (isinstance(interval, tuple) and len(interval) == 2):
                raise ValueError(f"invalid interval key: {interval!r}")
            i, j = interval
            self.validate_interval(i, j)
            if not isinstance(backends, Mapping):
                raise TypeError(f"backends for [{i},{j}] must be a mapping")
            for name, raw_cost in backends.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"backend name for [{i},{j}] must be non-empty")
                value = float(raw_cost)
                if isnan(value) or value < 0:
                    raise ValueError(
                        f"terminal cost for backend {name!r} on [{i},{j}] "
                        "must be non-negative and not NaN"
                    )

    def validate_interval(self, i: int, j: int) -> None:
        if not (isinstance(i, int) and isinstance(j, int)):
            raise TypeError("interval endpoints must be integers")
        if not (0 <= i < j <= self.n):
            raise ValueError(f"invalid interval [{i},{j}] for n={self.n}")

    def backends(self, i: int, j: int) -> Mapping[str, float]:
        self.validate_interval(i, j)
        return self.terminal_costs.get((i, j), {})

    def capabilities(self, backend: str) -> frozenset[str]:
        """Return platform-neutral capabilities for a terminal action id.

        Explicit capability metadata is authoritative.  A tiny legacy shim is
        retained for older experiment fixtures that predate capability tags:
        identifiers named ``native`` are treated as replay/one-step actions and
        identifiers containing ``zk`` are treated as ZK-proof actions.  New
        platform adapters should always supply capabilities explicitly.
        """

        if backend in self.terminal_capabilities:
            return frozenset(self.terminal_capabilities[backend])
        name = str(backend).strip().lower()
        inferred = {"terminal"}
        if name in {"native", "replay", "one-step", "onestep"}:
            inferred.update({"replay", "one-step"})
        if "zk" in name or "proof" in name:
            inferred.add("zk-proof")
        return frozenset(inferred)

    def backend_supports(self, backend: str, required: Iterable[str]) -> bool:
        return frozenset(str(x) for x in required).issubset(self.capabilities(backend))

    def _terminal_candidates_exact(
        self,
        i: int,
        j: int,
        required_capabilities: Iterable[str] = (),
    ) -> list[tuple[Fraction, str]]:
        required = frozenset(str(x) for x in required_capabilities)
        candidates: list[tuple[Fraction, str]] = []
        for name, raw_cost in self.backends(i, j).items():
            if raw_cost is None or not isfinite(float(raw_cost)):
                continue
            if required and not self.backend_supports(name, required):
                continue
            candidates.append((as_fraction(raw_cost), name))
        candidates.sort(
            key=lambda x: (
                x[0],
                self.backend_priority.get(x[1], 10**9),
                x[1],
            )
        )
        return candidates

    def best_terminal_exact(
        self,
        i: int,
        j: int,
        required_capabilities: Iterable[str] = (),
    ) -> Tuple[Optional[Fraction], Optional[str]]:
        candidates = self._terminal_candidates_exact(i, j, required_capabilities)
        if not candidates:
            return None, None
        return candidates[0]

    def best_terminal(
        self,
        i: int,
        j: int,
        required_capabilities: Iterable[str] = (),
    ) -> Tuple[float, Optional[str]]:
        cost, name = self.best_terminal_exact(i, j, required_capabilities)
        return (inf, None) if cost is None else (float(cost), name)

    def best_terminal_with_capabilities(
        self,
        i: int,
        j: int,
        required_capabilities: Iterable[str],
    ) -> Tuple[float, Optional[str]]:
        return self.best_terminal(i, j, required_capabilities)

    def checked_query_cost_exact(self, i: int, j: int, k: int) -> Optional[Fraction]:
        self.validate_interval(i, j)
        if not isinstance(k, int) or isinstance(k, bool) or not (i < k < j):
            raise ValueError(f"split {k!r} must satisfy {i} < k < {j}")
        raw = self.query_cost(i, j, k)
        value = float(raw)
        if isnan(value) or value < 0:
            raise ValueError(
                f"query cost q({i},{j},{k}) must be non-negative and not NaN"
            )
        if not isfinite(value):
            return None
        return as_fraction(raw)

    def checked_query_cost(self, i: int, j: int, k: int) -> float:
        exact = self.checked_query_cost_exact(i, j, k)
        return inf if exact is None else float(exact)


@dataclass
class HNDTResult:
    n: int
    value: Dict[Interval, float]
    exact_value: Dict[Interval, Fraction]
    action: Dict[Interval, Action]
    max_rounds: Dict[Interval, int]

    @property
    def optimum(self) -> float:
        return self.value[(0, self.n)]

    @property
    def optimum_exact(self) -> Fraction:
        return self.exact_value[(0, self.n)]


def _validate_legacy_tolerance(tie_tolerance: float) -> None:
    """Validate but deliberately do not use historical epsilon tie handling."""

    if tie_tolerance < 0 or isnan(float(tie_tolerance)):
        raise ValueError("tie_tolerance must be non-negative and not NaN")


def solve_hndt(model: CostModel, tie_tolerance: float = 0.0) -> HNDTResult:
    """Compute the exact binary stop-or-split minimax policy in ``O(n^3)``.

    Scientific comparisons use :class:`fractions.Fraction`.  ``tie_tolerance``
    is accepted only for backwards API compatibility and does not affect policy
    selection.  Equal objective values are resolved by fewer worst-case rounds,
    then midpoint proximity, then smaller split index; direct settlement is
    retained when exactly tied with a split.
    """

    _validate_legacy_tolerance(tie_tolerance)

    n = model.n
    exact_value: Dict[Interval, Fraction] = {}
    action: Dict[Interval, Action] = {}
    rounds: Dict[Interval, int] = {}

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            terminal, backend = model.best_terminal_exact(i, j)
            best: Optional[Fraction] = terminal
            best_action = Action.settle(backend) if backend is not None else None
            best_rounds = 0 if backend is not None else 10**9
            best_split_key: tuple[int, int, int] | None = None

            if length > 1:
                for k in range(i + 1, j):
                    left = exact_value.get((i, k))
                    right = exact_value.get((k, j))
                    if left is None or right is None:
                        continue
                    q = model.checked_query_cost_exact(i, j, k)
                    if q is None:
                        continue
                    candidate = q + max(left, right)
                    candidate_rounds = 1 + max(rounds[(i, k)], rounds[(k, j)])
                    split_key = (candidate_rounds, abs(2 * k - (i + j)), k)

                    improve = best is None or candidate < best
                    tie_better_split = (
                        best is not None
                        and candidate == best
                        and best_action is not None
                        and best_action.kind == "split"
                        and (best_split_key is None or split_key < best_split_key)
                    )
                    if improve or tie_better_split:
                        best = candidate
                        best_action = Action.split_at(k)
                        best_rounds = candidate_rounds
                        best_split_key = split_key

            if best_action is None or best is None:
                raise ValueError(
                    f"Interval [{i},{j}] is unsolvable: no settlement action and no "
                    "finite split whose children are solvable. Ensure every atomic "
                    "interval has at least one terminal action."
                )
            exact_value[(i, j)] = best
            action[(i, j)] = best_action
            rounds[(i, j)] = best_rounds

    value = {interval: float(v) for interval, v in exact_value.items()}
    return HNDTResult(
        n=n,
        value=value,
        exact_value=exact_value,
        action=action,
        max_rounds=rounds,
    )


@dataclass
class RoundBudgetResult:
    """Exact minimax policy constrained by a worst-case split-round budget."""

    n: int
    round_budget: int
    optimum: float
    optimum_exact: Fraction
    action: Dict[Interval, Action]
    max_rounds: int


def solve_hndt_round_budget(
    model: CostModel,
    round_budget: int,
    tie_tolerance: float = 0.0,
) -> RoundBudgetResult:
    """Minimize worst-case cost subject to at most ``round_budget`` splits."""

    if not isinstance(round_budget, int) or isinstance(round_budget, bool) or round_budget < 0:
        raise ValueError("round_budget must be a non-negative integer")
    _validate_legacy_tolerance(tie_tolerance)

    n = model.n
    values: list[Dict[Interval, Optional[Fraction]]] = []
    actions: list[Dict[Interval, Action | None]] = []
    rounds_used: list[Dict[Interval, int]] = []

    for budget in range(round_budget + 1):
        value_b: Dict[Interval, Optional[Fraction]] = {}
        action_b: Dict[Interval, Action | None] = {}
        rounds_b: Dict[Interval, int] = {}

        for length in range(1, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                terminal, backend = model.best_terminal_exact(i, j)
                best = terminal
                best_action: Action | None = (
                    Action.settle(backend) if backend is not None else None
                )
                best_rounds = 0 if backend is not None else 10**9
                best_split_key: tuple[int, int, int] | None = None

                if budget > 0 and length > 1:
                    previous_values = values[budget - 1]
                    previous_rounds = rounds_used[budget - 1]
                    for k in range(i + 1, j):
                        left = previous_values[(i, k)]
                        right = previous_values[(k, j)]
                        if left is None or right is None:
                            continue
                        q = model.checked_query_cost_exact(i, j, k)
                        if q is None:
                            continue

                        candidate = q + max(left, right)
                        candidate_rounds = 1 + max(
                            previous_rounds[(i, k)],
                            previous_rounds[(k, j)],
                        )
                        split_key = (candidate_rounds, abs(2 * k - (i + j)), k)

                        improve = best is None or candidate < best
                        tie_better_split = (
                            best is not None
                            and candidate == best
                            and best_action is not None
                            and best_action.kind == "split"
                            and (best_split_key is None or split_key < best_split_key)
                        )
                        if improve or tie_better_split:
                            best = candidate
                            best_action = Action.split_at(k)
                            best_rounds = candidate_rounds
                            best_split_key = split_key

                value_b[(i, j)] = best
                action_b[(i, j)] = best_action
                rounds_b[(i, j)] = best_rounds

        values.append(value_b)
        actions.append(action_b)
        rounds_used.append(rounds_b)

    root = (0, n)
    root_value = values[round_budget][root]
    if actions[round_budget][root] is None or root_value is None:
        raise ValueError(
            f"No feasible policy for n={n} within round_budget={round_budget}. "
            "Increase the budget or provide larger-interval terminal measurements."
        )

    policy: Dict[Interval, Action] = {}
    stack: list[tuple[int, int, int]] = [(0, n, round_budget)]
    while stack:
        i, j, budget = stack.pop()
        act = actions[budget][(i, j)]
        if act is None:
            raise AssertionError("reachable budgeted state is unexpectedly infeasible")
        previous = policy.get((i, j))
        if previous is not None and previous != act:
            raise AssertionError(
                f"interval [{i},{j}] reached with inconsistent budgeted actions"
            )
        policy[(i, j)] = act
        if act.kind == "split":
            if budget == 0 or act.split is None:
                raise AssertionError("invalid split in zero-budget state")
            k = int(act.split)
            stack.append((i, k, budget - 1))
            stack.append((k, j, budget - 1))

    return RoundBudgetResult(
        n=n,
        round_budget=round_budget,
        optimum=float(root_value),
        optimum_exact=root_value,
        action=policy,
        max_rounds=rounds_used[round_budget][root],
    )


def policy_reachable_intervals(
    action: Mapping[Interval, Action], root: Interval
) -> Iterable[Interval]:
    """Yield all intervals reachable from ``root`` and validate the policy tree."""

    stack = [root]
    seen = set()
    while stack:
        interval = stack.pop()
        if interval in seen:
            continue
        if interval not in action:
            raise ValueError(f"policy is missing action for interval {interval}")
        seen.add(interval)
        yield interval
        act = action[interval]
        i, j = interval
        if act.kind == "settle":
            if act.backend is None:
                raise ValueError(f"settlement action on {interval} has no backend")
            continue
        if act.kind != "split" or act.split is None:
            raise ValueError(f"invalid action {act!r} on interval {interval}")
        k = int(act.split)
        if not (i < k < j):
            raise ValueError(f"invalid split {k} on interval [{i},{j}]")
        stack.append((i, k))
        stack.append((k, j))
