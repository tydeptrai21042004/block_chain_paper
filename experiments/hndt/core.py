from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite, isnan
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

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
        if not isinstance(k, int):
            raise TypeError("split index must be an integer")
        return Action(kind="split", split=k)


@dataclass
class CostModel:
    """Finite terminal costs plus a query-cost function.

    ``terminal_costs`` maps ``(i, j)`` to ``{backend: cycles}``. Missing
    intervals/backends are unavailable. ``+inf`` is accepted as an explicit
    unavailable cost, while NaN and negative costs are rejected.
    """

    n: int
    terminal_costs: Mapping[Interval, Mapping[str, float]]
    query_cost: Callable[[int, int, int], float]

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n <= 0:
            raise ValueError("n must be a positive integer")
        if not callable(self.query_cost):
            raise TypeError("query_cost must be callable")

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

    def best_terminal(self, i: int, j: int) -> Tuple[float, Optional[str]]:
        candidates = [
            (float(cost), name)
            for name, cost in self.backends(i, j).items()
            if cost is not None and isfinite(float(cost))
        ]
        if not candidates:
            return inf, None
        # Deterministic tie break: native first, then lexical backend name.
        candidates.sort(key=lambda x: (x[0], 0 if x[1] == "native" else 1, x[1]))
        return candidates[0]

    def checked_query_cost(self, i: int, j: int, k: int) -> float:
        self.validate_interval(i, j)
        if not isinstance(k, int) or not (i < k < j):
            raise ValueError(f"split {k!r} must satisfy {i} < k < {j}")
        value = float(self.query_cost(i, j, k))
        if isnan(value) or value < 0:
            raise ValueError(
                f"query cost q({i},{j},{k}) must be non-negative and not NaN"
            )
        return value


@dataclass
class HNDTResult:
    n: int
    value: Dict[Interval, float]
    action: Dict[Interval, Action]
    max_rounds: Dict[Interval, int]

    @property
    def optimum(self) -> float:
        return self.value[(0, self.n)]


def solve_hndt(model: CostModel, tie_tolerance: float = 1e-12) -> HNDTResult:
    """Compute the exact stop-or-split minimax policy in ``O(n^3)`` time.

    Tie handling is deterministic. Direct settlement is retained when a split
    differs by at most ``tie_tolerance``. Between effectively equal splits, the
    solver prefers fewer worst-case rounds, then the split closest to the
    interval midpoint, and finally the smaller split index.
    """

    if tie_tolerance < 0 or isnan(float(tie_tolerance)):
        raise ValueError("tie_tolerance must be non-negative and not NaN")

    n = model.n
    value: Dict[Interval, float] = {}
    action: Dict[Interval, Action] = {}
    rounds: Dict[Interval, int] = {}

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            best, backend = model.best_terminal(i, j)
            best_action = Action.settle(backend) if backend is not None else None
            best_rounds = 0 if backend is not None else 10**9
            best_split_key: tuple[int, float, int] | None = None
            midpoint = (i + j) / 2.0

            if length > 1:
                for k in range(i + 1, j):
                    left = value[(i, k)]
                    right = value[(k, j)]
                    if not (isfinite(left) and isfinite(right)):
                        continue
                    q = model.checked_query_cost(i, j, k)
                    if not isfinite(q):
                        # +inf is a clean way to mark a query as unavailable.
                        continue
                    candidate = q + max(left, right)
                    candidate_rounds = 1 + max(rounds[(i, k)], rounds[(k, j)])
                    split_key = (candidate_rounds, abs(k - midpoint), k)

                    improve = candidate < best - tie_tolerance
                    tied_with_best = abs(candidate - best) <= tie_tolerance
                    tie_better_split = (
                        tied_with_best
                        and best_action is not None
                        and best_action.kind == "split"
                        and (best_split_key is None or split_key < best_split_key)
                    )
                    # If best_action is settlement, a tied split is deliberately
                    # not selected: settlement uses no additional interaction.
                    if improve or tie_better_split:
                        best = candidate
                        best_action = Action.split_at(k)
                        best_rounds = candidate_rounds
                        best_split_key = split_key

            if best_action is None:
                raise ValueError(
                    f"Interval [{i},{j}] is unsolvable: no settlement backend and no "
                    "finite split whose children are solvable. Ensure every atomic "
                    "interval has at least one terminal cost."
                )
            value[(i, j)] = best
            action[(i, j)] = best_action
            rounds[(i, j)] = best_rounds

    return HNDTResult(n=n, value=value, action=action, max_rounds=rounds)



@dataclass
class RoundBudgetResult:
    """Exact HNDT policy constrained by a worst-case split-round budget."""

    n: int
    round_budget: int
    optimum: float
    action: Dict[Interval, Action]
    max_rounds: int


def solve_hndt_round_budget(
    model: CostModel,
    round_budget: int,
    tie_tolerance: float = 1e-12,
) -> RoundBudgetResult:
    """Minimize worst-case cost subject to at most ``round_budget`` splits.

    This is a minimal latency-constrained extension of HNDT.  Let F_r(i,j)
    denote the cheapest worst-case dispute cost on interval [i,j] with at most
    r further split rounds.  Settlement consumes no additional split round,
    while a split at k consumes one and gives each child budget r-1:

        F_r(i,j) = min(D(i,j),
                       min_k q(i,j,k) + max(F_{r-1}(i,k), F_{r-1}(k,j))).

    The algorithm is exact in O(round_budget * n^3) time and does not require
    any new benchmark measurements.
    """

    if not isinstance(round_budget, int) or isinstance(round_budget, bool) or round_budget < 0:
        raise ValueError("round_budget must be a non-negative integer")
    if tie_tolerance < 0 or isnan(float(tie_tolerance)):
        raise ValueError("tie_tolerance must be non-negative and not NaN")

    n = model.n
    values: list[Dict[Interval, float]] = []
    actions: list[Dict[Interval, Action | None]] = []
    rounds_used: list[Dict[Interval, int]] = []

    for budget in range(round_budget + 1):
        value_b: Dict[Interval, float] = {}
        action_b: Dict[Interval, Action | None] = {}
        rounds_b: Dict[Interval, int] = {}

        for length in range(1, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                best, backend = model.best_terminal(i, j)
                best_action: Action | None = (
                    Action.settle(backend) if backend is not None else None
                )
                best_rounds = 0 if backend is not None else 10**9
                best_split_key: tuple[int, float, int] | None = None
                midpoint = (i + j) / 2.0

                if budget > 0 and length > 1:
                    previous_values = values[budget - 1]
                    previous_rounds = rounds_used[budget - 1]
                    for k in range(i + 1, j):
                        left = previous_values[(i, k)]
                        right = previous_values[(k, j)]
                        if not (isfinite(left) and isfinite(right)):
                            continue
                        q = model.checked_query_cost(i, j, k)
                        if not isfinite(q):
                            continue

                        candidate = q + max(left, right)
                        candidate_rounds = 1 + max(
                            previous_rounds[(i, k)],
                            previous_rounds[(k, j)],
                        )
                        split_key = (candidate_rounds, abs(k - midpoint), k)

                        improve = candidate < best - tie_tolerance
                        tied = abs(candidate - best) <= tie_tolerance
                        tie_better_split = (
                            tied
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
    if actions[round_budget][root] is None or not isfinite(values[round_budget][root]):
        raise ValueError(
            f"No feasible HNDT policy for n={n} within round_budget={round_budget}. "
            "Increase the budget or provide larger-interval terminal measurements."
        )

    # Materialize only the policy states reachable from the root.  The remaining
    # budget is part of the DP state, but an interval in the resulting tree has
    # one deterministic budget inherited from its depth.
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
        optimum=values[round_budget][root],
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
