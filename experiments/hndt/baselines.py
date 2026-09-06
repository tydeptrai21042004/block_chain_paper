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
