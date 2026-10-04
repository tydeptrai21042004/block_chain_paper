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


def constant_query_cost(model: CostModel, tie_tolerance: float = 1e-12) -> float:
    """Return the common finite split-query cost, or raise if it is not constant.

    The classical alphabetic-tree reductions are exact only when each tree edge
    has the same additive query cost.  Models built by ``hndt.io.build_model``
    satisfy this because they use one Merkle depth for the committed trace.
    """

    from math import isnan

    if tie_tolerance < 0 or isnan(float(tie_tolerance)):
        raise ValueError("tie_tolerance must be non-negative and not NaN")
    seen = None
    for span in range(2, model.n + 1):
        for i in range(0, model.n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                q = model.checked_query_cost(i, j, k)
                if not isfinite(q):
                    raise ValueError("classical alphabetic-tree baseline requires finite query costs")
                if seen is None:
                    seen = q
                elif abs(q - seen) > tie_tolerance:
                    raise ValueError(
                        "classical alphabetic-tree baseline requires a constant query cost"
                    )
    return 0.0 if seen is None else float(seen)


def kirkpatrick_klawe_atomic_policy(model: CostModel) -> Dict[Interval, Action]:
    """Alphabetic-minimax objective adaptation (Kirkpatrick--Klawe, 1985).

    With forced atomic leaves and constant query cost q, an atomic fault at t
    pays A_t + q d_t.  Dividing by q (for q>0) yields the classical objective
    max_t(A_t/q + d_t).  For q=0 the same exact atomic minimax DP remains the
    continuous zero-edge-cost limit.  The implementation uses the repository's
    exact forced-atomic HNDT solver rather than reimplementing the historical
    construction, but optimizes the same binary alphabetic minimax objective.
    """

    constant_query_cost(model)
    return optimal_split_atomic_policy(model)


def optimal_mean_atomic_policy(
    model: CostModel,
    fault_weights=None,
    tie_tolerance: float = 1e-12,
) -> Dict[Interval, Action]:
    """Exact forced-atomic policy minimizing weighted mean fault-path cost.

    This is kept independent from Pareto-HNDT so it can serve as an external
    objective baseline.  Under constant query cost, fixed atomic leaves, and
    weights p_t, the tree-dependent term is q * sum_t p_t d_t, exactly the
    optimal alphabetic weighted-path-length objective.
    """

    from math import inf, isnan
    from .pareto import normalize_fault_weights

    if tie_tolerance < 0 or isnan(float(tie_tolerance)):
        raise ValueError("tie_tolerance must be non-negative and not NaN")

    weights = normalize_fault_weights(model.n, fault_weights)
    prefix = [0.0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)

    def mass(i: int, j: int) -> float:
        return prefix[j] - prefix[i]

    value: Dict[Interval, float] = {}
    actions: Dict[Interval, Action] = {}
    rounds: Dict[Interval, int] = {}

    for length in range(1, model.n + 1):
        for i in range(0, model.n - length + 1):
            j = i + length
            if length == 1:
                cost, backend = model.best_terminal(i, j)
                if backend is None or not isfinite(cost):
                    raise ValueError(f"Atomic interval [{i},{j}] has no terminal backend")
                value[(i, j)] = mass(i, j) * cost
                actions[(i, j)] = Action.settle(backend)
                rounds[(i, j)] = 0
                continue

            best = inf
            best_action = None
            best_rounds = 10**9
            midpoint = (i + j) / 2.0
            best_key = None
            for k in range(i + 1, j):
                q = model.checked_query_cost(i, j, k)
                if not isfinite(q):
                    continue
                candidate = mass(i, j) * q + value[(i, k)] + value[(k, j)]
                candidate_rounds = 1 + max(rounds[(i, k)], rounds[(k, j)])
                key = (candidate_rounds, abs(k - midpoint), k)
                if candidate < best - tie_tolerance or (
                    abs(candidate - best) <= tie_tolerance
                    and (best_key is None or key < best_key)
                ):
                    best = candidate
                    best_action = Action.split_at(k)
                    best_rounds = candidate_rounds
                    best_key = key

            if best_action is None:
                raise ValueError(f"Mean-optimal atomic policy is infeasible on [{i},{j}]")
            value[(i, j)] = best
            actions[(i, j)] = best_action
            rounds[(i, j)] = best_rounds

    # All intervals can be retained; evaluate_policy only traverses reachable ones.
    return actions


def hu_tucker_atomic_policy(model: CostModel, fault_weights=None) -> Dict[Interval, Action]:
    """Hu--Tucker weighted-path-length objective adaptation.

    The historical Hu--Tucker problem assumes ordered fixed leaves and constant
    per-edge search cost.  This adapter enforces those conditions, then solves
    the same objective exactly by interval DP.  Uniform weights are used unless
    explicit fault weights are provided; no fault-frequency model is invented.
    """

    constant_query_cost(model)
    return optimal_mean_atomic_policy(model, fault_weights=fault_weights)
