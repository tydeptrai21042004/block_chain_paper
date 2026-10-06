from __future__ import annotations

from fractions import Fraction
from math import isfinite
from typing import Dict, Iterable, Optional, Tuple

from .core import Action, CostModel
from .exact import as_fraction, normalize_weight_fractions

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
    required_capabilities: Iterable[str] = (),
) -> Optional[Action]:
    """Choose a finite terminal action, optionally by generic capability.

    ``required_backend`` is retained only for backwards compatibility.  New
    baseline adapters should use ``required_capabilities`` so they are not tied
    to platform-specific action names.
    """

    if required_backend is not None:
        value = model.backends(i, j).get(required_backend)
        if value is None or not isfinite(float(value)):
            return None
        if required_capabilities and not model.backend_supports(
            required_backend, required_capabilities
        ):
            return None
        return Action.settle(required_backend)

    _, backend = model.best_terminal_with_capabilities(i, j, required_capabilities)
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
            act = _terminal_action(model, i, j, required_capabilities={"zk-proof"})
            if act is None:
                raise ValueError(
                    f"Atomic interval [{i},{j}] has no finite terminal action with capability zk-proof"
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


def has_complete_atomic_capability(
    model: CostModel, required_capabilities: Iterable[str]
) -> bool:
    """Return True iff every atomic interval has a matching terminal action."""

    for i in range(model.n):
        _, backend = model.best_terminal_with_capabilities(
            i, i + 1, required_capabilities
        )
        if backend is None:
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
    restricted = _CostModel(
        model.n,
        terminal,
        model.query_cost,
        terminal_capabilities=model.terminal_capabilities,
        metadata=model.metadata,
        backend_priority=model.backend_priority,
    )
    return solve_hndt(restricted).action


def midpoint_adaptive_stop_policy(
    model: CostModel,
    tie_tolerance: float = 0.0,
) -> Dict[Interval, Action]:
    """Midpoint-only localization with exact adaptive stopping."""

    from .core import _validate_legacy_tolerance, policy_reachable_intervals

    _validate_legacy_tolerance(tie_tolerance)
    value: Dict[Interval, Fraction] = {}
    actions: Dict[Interval, Action] = {}
    for length in range(1, model.n + 1):
        for i in range(0, model.n - length + 1):
            j = i + length
            best, backend = model.best_terminal_exact(i, j)
            best_action = Action.settle(backend) if backend is not None else None

            if length > 1:
                k = _midpoint(i, j)
                left = value[(i, k)]
                right = value[(k, j)]
                q = model.checked_query_cost_exact(i, j, k)
                if q is not None:
                    candidate = q + max(left, right)
                    # Exact ties retain direct settlement because it uses no
                    # additional interaction round.
                    if best is None or candidate < best:
                        best = candidate
                        best_action = Action.split_at(k)

            if best_action is None or best is None:
                raise ValueError(
                    f"Midpoint-adaptive-stop policy is infeasible on [{i},{j}]"
                )
            value[(i, j)] = best
            actions[(i, j)] = best_action

    reachable = list(policy_reachable_intervals(actions, (0, model.n)))
    return {interval: actions[interval] for interval in reachable}


def direct_verification_policy(
    model: CostModel,
    required_capabilities: Iterable[str] = (),
) -> Dict[Interval, Action]:
    """Settle the complete trace with the cheapest admissible terminal action."""

    _, backend = model.best_terminal_with_capabilities(
        0, model.n, required_capabilities
    )
    if backend is None:
        required = ",".join(required_capabilities) or "any"
        raise ValueError(
            f"full-trace verification action is unavailable (required={required})"
        )
    return {(0, model.n): Action.settle(backend)}


def direct_native_policy(model: CostModel) -> Dict[Interval, Action]:
    """Backward-compatible CKB helper; new code should use direct_verification_policy."""

    raw = model.backends(0, model.n).get("native")
    if raw is None or not isfinite(float(raw)):
        raise ValueError("full-trace native verification cost is unavailable")
    return {(0, model.n): Action.settle("native")}


def constant_query_cost_exact(model: CostModel) -> Fraction:
    """Return the common finite query cost exactly, or raise if non-constant."""

    seen: Fraction | None = None
    for span in range(2, model.n + 1):
        for i in range(0, model.n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                q = model.checked_query_cost_exact(i, j, k)
                if q is None:
                    raise ValueError(
                        "classical alphabetic-tree baseline requires finite query costs"
                    )
                if seen is None:
                    seen = q
                elif q != seen:
                    raise ValueError(
                        "classical alphabetic-tree baseline requires a constant query cost"
                    )
    return Fraction(0, 1) if seen is None else seen


def constant_query_cost(model: CostModel, tie_tolerance: float = 0.0) -> float:
    """Backward-compatible float view of :func:`constant_query_cost_exact`."""

    from .core import _validate_legacy_tolerance
    _validate_legacy_tolerance(tie_tolerance)
    return float(constant_query_cost_exact(model))


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
    tie_tolerance: float = 0.0,
) -> Dict[Interval, Action]:
    """Exact forced-atomic policy minimizing weighted expected fault-path cost."""

    from .core import _validate_legacy_tolerance

    _validate_legacy_tolerance(tie_tolerance)
    weights = normalize_weight_fractions(model.n, fault_weights)
    prefix = [Fraction(0, 1)]
    for weight in weights:
        prefix.append(prefix[-1] + weight)

    def mass(i: int, j: int) -> Fraction:
        return prefix[j] - prefix[i]

    value: Dict[Interval, Fraction] = {}
    actions: Dict[Interval, Action] = {}
    rounds: Dict[Interval, int] = {}

    for length in range(1, model.n + 1):
        for i in range(0, model.n - length + 1):
            j = i + length
            if length == 1:
                cost, backend = model.best_terminal_exact(i, j)
                if backend is None or cost is None:
                    raise ValueError(f"Atomic interval [{i},{j}] has no terminal backend")
                value[(i, j)] = mass(i, j) * cost
                actions[(i, j)] = Action.settle(backend)
                rounds[(i, j)] = 0
                continue

            best: Fraction | None = None
            best_action = None
            best_rounds = 10**9
            best_key = None
            for k in range(i + 1, j):
                q = model.checked_query_cost_exact(i, j, k)
                if q is None:
                    continue
                candidate = mass(i, j) * q + value[(i, k)] + value[(k, j)]
                candidate_rounds = 1 + max(rounds[(i, k)], rounds[(k, j)])
                key = (candidate_rounds, abs(2 * k - (i + j)), k)
                if best is None or candidate < best or (
                    candidate == best and (best_key is None or key < best_key)
                ):
                    best = candidate
                    best_action = Action.split_at(k)
                    best_rounds = candidate_rounds
                    best_key = key

            if best_action is None or best is None:
                raise ValueError(f"Mean-optimal atomic policy is infeasible on [{i},{j}]")
            value[(i, j)] = best
            actions[(i, j)] = best_action
            rounds[(i, j)] = best_rounds

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


def pareto_atomic_policy(model: CostModel, fault_weights=None) -> Dict[Interval, Action]:
    """Fair ablation: adaptive split + forced atomic stop + Pareto refinement.

    Unlike :func:`optimal_split_atomic_policy`, this keeps the proposal's exact
    minimax-safe secondary expected-cost objective.  Therefore the comparison
    isolates adaptive stopping rather than conflating stopping with objective
    selection.
    """

    from .pareto import solve_pareto_hndt

    result = solve_pareto_hndt(
        model,
        fault_weights=fault_weights,
        terminal_filter=lambda i, j, backend: j - i == 1,
    )
    return result.policy


def pareto_midpoint_adaptive_stop_policy(
    model: CostModel,
    fault_weights=None,
) -> Dict[Interval, Action]:
    """Fair ablation: midpoint-only split + adaptive stop + Pareto refinement."""

    from .pareto import solve_pareto_hndt

    result = solve_pareto_hndt(
        model,
        fault_weights=fault_weights,
        split_filter=lambda i, j, k: k == _midpoint(i, j),
    )
    return result.policy


def pareto_midpoint_atomic_policy(
    model: CostModel,
    fault_weights=None,
) -> Dict[Interval, Action]:
    """Fully restricted factorial endpoint with the same Pareto objective."""

    from .pareto import solve_pareto_hndt

    result = solve_pareto_hndt(
        model,
        fault_weights=fault_weights,
        split_filter=lambda i, j, k: k == _midpoint(i, j),
        terminal_filter=lambda i, j, backend: j - i == 1,
    )
    return result.policy


def height_limited_mean_atomic_policy(
    model: CostModel,
    max_rounds: int,
    fault_weights=None,
) -> Dict[Interval, Action]:
    """Height-limited optimal alphabetic mean tree objective adaptation.

    This is the fixed-leaf, constant-query-cost objective studied by the
    height-limited alphabetic-tree literature (e.g. Larmore--Przytycka), solved
    here by an independent exact interval DP.  ``max_rounds`` is the maximum
    root-to-leaf number of split queries.
    """

    from fractions import Fraction
    from .exact import as_fraction, normalize_weight_fractions

    if not isinstance(max_rounds, int) or isinstance(max_rounds, bool) or max_rounds < 0:
        raise ValueError("max_rounds must be a non-negative integer")
    constant_query_cost(model)

    weights = normalize_weight_fractions(model.n, fault_weights)
    prefix = [Fraction(0, 1)]
    for weight in weights:
        prefix.append(prefix[-1] + weight)

    def mass(i: int, j: int) -> Fraction:
        return prefix[j] - prefix[i]

    # DP state: (budget, i, j) -> exact weighted contribution / action / rounds.
    values: dict[tuple[int, int, int], Fraction | None] = {}
    actions: dict[tuple[int, int, int], Action | None] = {}
    rounds_used: dict[tuple[int, int, int], int] = {}

    for budget in range(max_rounds + 1):
        for length in range(1, model.n + 1):
            for i in range(0, model.n - length + 1):
                j = i + length
                key = (budget, i, j)

                if length == 1:
                    cost, backend = model.best_terminal(i, j)
                    if backend is None or not isfinite(cost):
                        values[key] = None
                        actions[key] = None
                        rounds_used[key] = 10**9
                    else:
                        values[key] = mass(i, j) * as_fraction(cost)
                        actions[key] = Action.settle(backend)
                        rounds_used[key] = 0
                    continue

                if budget == 0:
                    values[key] = None
                    actions[key] = None
                    rounds_used[key] = 10**9
                    continue

                best_value: Fraction | None = None
                best_action: Action | None = None
                best_rounds = 10**9
                best_tie = None
                for k in range(i + 1, j):
                    left = values[(budget - 1, i, k)]
                    right = values[(budget - 1, k, j)]
                    if left is None or right is None:
                        continue
                    try:
                        q = as_fraction(model.checked_query_cost(i, j, k))
                    except ValueError:
                        continue
                    candidate = mass(i, j) * q + left + right
                    candidate_rounds = 1 + max(
                        rounds_used[(budget - 1, i, k)],
                        rounds_used[(budget - 1, k, j)],
                    )
                    tie = (candidate_rounds, abs(2 * k - (i + j)), k)
                    if best_value is None or candidate < best_value or (
                        candidate == best_value and (best_tie is None or tie < best_tie)
                    ):
                        best_value = candidate
                        best_action = Action.split_at(k)
                        best_rounds = candidate_rounds
                        best_tie = tie

                values[key] = best_value
                actions[key] = best_action
                rounds_used[key] = best_rounds

    root_key = (max_rounds, 0, model.n)
    if values[root_key] is None or actions[root_key] is None:
        raise ValueError(
            f"No height-limited atomic alphabetic tree is feasible for n={model.n} "
            f"with max_rounds={max_rounds}"
        )

    policy: Dict[Interval, Action] = {}
    stack = [(0, model.n, max_rounds)]
    while stack:
        i, j, budget = stack.pop()
        act = actions[(budget, i, j)]
        if act is None:
            raise AssertionError("reachable height-limited state is infeasible")
        previous = policy.get((i, j))
        if previous is not None and previous != act:
            raise AssertionError("interval reached with inconsistent height-limited action")
        policy[(i, j)] = act
        if act.kind == "split":
            if budget == 0 or act.split is None:
                raise AssertionError("invalid split in zero-budget height-limited state")
            k = int(act.split)
            stack.append((i, k, budget - 1))
            stack.append((k, j, budget - 1))

    return policy
