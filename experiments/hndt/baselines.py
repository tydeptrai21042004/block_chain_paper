from __future__ import annotations

from math import isfinite
from typing import Dict, Optional, Tuple

from .core import Action, CostModel

Interval = Tuple[int, int]


def _midpoint(i: int, j: int) -> int:
    if j - i < 2:
        raise ValueError("midpoint split requires a non-atomic interval")
    return i + (j - i) // 2


def _terminal_action(
    model: CostModel,
    i: int,
    j: int,
    *,
    required_backend: Optional[str] = None,
) -> Optional[Action]:
    backends = model.backends(i, j)
    if required_backend is not None:
        value = backends.get(required_backend)
        if value is None or not isfinite(float(value)):
            return None
        return Action.settle(required_backend)
    _, backend = model.best_terminal(i, j)
    return None if backend is None else Action.settle(backend)


def midpoint_atomic_policy(model: CostModel) -> Dict[Interval, Action]:
    """Fixed midpoint localization until one transition remains.

    At each atomic interval the cheapest admissible terminal backend is used.
    """

    policy: Dict[Interval, Action] = {}

    def build(i: int, j: int) -> None:
        if j - i == 1:
            act = _terminal_action(model, i, j)
            if act is None:
                raise ValueError(f"Atomic interval [{i},{j}] has no terminal backend")
            policy[(i, j)] = act
            return
        k = _midpoint(i, j)
        policy[(i, j)] = Action.split_at(k)
        build(i, k)
        build(k, j)

    build(0, model.n)
    return policy


def midpoint_operator_zk_policy(model: CostModel) -> Dict[Interval, Action]:
    """Midpoint localization to one operator, then *require* zkVM settlement.

    This strict behavior avoids silently turning the ``+ ZK`` baseline into a
    mixed-backend baseline when one atomic zkVM measurement is missing.
    """

    policy: Dict[Interval, Action] = {}

    def build(i: int, j: int) -> None:
        if j - i == 1:
            act = _terminal_action(model, i, j, required_backend="zkvm")
            if act is None:
                raise ValueError(
                    f"Atomic interval [{i},{j}] has no finite zkvm terminal cost"
                )
            policy[(i, j)] = act
            return
        k = _midpoint(i, j)
        policy[(i, j)] = Action.split_at(k)
        build(i, k)
        build(k, j)

    build(0, model.n)
    return policy


def fixed_g_policy(model: CostModel, g: int) -> Dict[Interval, Action]:
    """Midpoint localization with target terminal span ``<= g``.

    If an interval of size at most ``g`` has no directly measured/admissible
    terminal backend, localization continues until a measured settlement is
    available. This keeps the baseline well-defined with sparse interval tables
    without inventing missing costs.
    """

    if not isinstance(g, int) or isinstance(g, bool) or g < 1:
        raise ValueError("g must be a positive integer")
    policy: Dict[Interval, Action] = {}

    def build(i: int, j: int) -> None:
        if j - i <= g:
            act = _terminal_action(model, i, j)
            if act is not None:
                policy[(i, j)] = act
                return
        if j - i == 1:
            raise ValueError(f"Atomic interval [{i},{j}] has no terminal backend")
        k = _midpoint(i, j)
        policy[(i, j)] = Action.split_at(k)
        build(i, k)
        build(k, j)

    build(0, model.n)
    return policy


def has_complete_atomic_backend(model: CostModel, backend: str) -> bool:
    """Return True iff ``backend`` is finite on every atomic interval."""

    for i in range(model.n):
        raw = model.backends(i, i + 1).get(backend)
        if raw is None or not isfinite(float(raw)):
            return False
    return True


def optimal_split_atomic_policy(model: CostModel) -> Dict[Interval, Action]:
    """Exact split optimization with *forced atomic* settlement.

    This mechanism-isolation ablation retains HNDT's adaptive split selection
    but removes its adaptive stopping action on every non-atomic interval.  It
    therefore isolates the value of choosing *where to split* from the value of
    choosing *when to stop*.
    """

    from .core import CostModel as _CostModel, solve_hndt

    terminal = {
        (i, i + 1): dict(model.backends(i, i + 1))
        for i in range(model.n)
    }
    restricted = _CostModel(model.n, terminal, model.query_cost)
    return solve_hndt(restricted).action


def midpoint_adaptive_stop_policy(
    model: CostModel,
    tie_tolerance: float = 1e-12,
) -> Dict[Interval, Action]:
    """Midpoint-only localization with exact adaptive stopping.

    Each interval can either settle using its cheapest measured backend or split
    *only* at the midpoint.  This is the complementary mechanism-isolation
    ablation to :func:`optimal_split_atomic_policy`.
    """

    from math import inf, isfinite, isnan
    from .core import policy_reachable_intervals

    if tie_tolerance < 0 or isnan(float(tie_tolerance)):
        raise ValueError("tie_tolerance must be non-negative and not NaN")

    value: Dict[Interval, float] = {}
    actions: Dict[Interval, Action] = {}
    for length in range(1, model.n + 1):
        for i in range(0, model.n - length + 1):
            j = i + length
            best, backend = model.best_terminal(i, j)
            best_action = Action.settle(backend) if backend is not None else None

            if length > 1:
                k = _midpoint(i, j)
                left = value[(i, k)]
                right = value[(k, j)]
                q = model.checked_query_cost(i, j, k)
                candidate = q + max(left, right) if all(map(isfinite, [left, right, q])) else inf
                # Ties deliberately retain direct settlement because it uses no
                # additional interaction round.
                if candidate < best - tie_tolerance:
                    best = candidate
                    best_action = Action.split_at(k)

            if best_action is None or not isfinite(best):
                raise ValueError(
                    f"Midpoint-adaptive-stop policy is infeasible on [{i},{j}]"
                )
            value[(i, j)] = best
            actions[(i, j)] = best_action

    reachable = list(policy_reachable_intervals(actions, (0, model.n)))
    return {interval: actions[interval] for interval in reachable}


def direct_native_policy(model: CostModel) -> Dict[Interval, Action]:
    """Settle the complete trace directly with the measured native backend."""

    raw = model.backends(0, model.n).get("native")
    if raw is None or not isfinite(float(raw)):
        raise ValueError("full-trace native verification cost is unavailable")
    return {(0, model.n): Action.settle("native")}
