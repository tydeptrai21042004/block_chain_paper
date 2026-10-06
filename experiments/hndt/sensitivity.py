from __future__ import annotations

"""Controlled sensitivity helpers that do not modify HNDT/Pareto-HNDT logic."""

import math
import random
from typing import Sequence

from .core import CostModel


def normalize_weights(values: Sequence[float]) -> tuple[float, ...]:
    data = [float(x) for x in values]
    if not data or any(not math.isfinite(x) or x < 0 for x in data):
        raise ValueError("weights must be a non-empty finite non-negative sequence")
    total = sum(data)
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return tuple(x / total for x in data)


def fault_weight_profile(model: CostModel, profile: str) -> tuple[float, ...]:
    """Return a predefined transparent fault-weight sensitivity profile."""

    key = profile.strip().lower().replace("-", "_")
    n = model.n
    if key == "uniform":
        return tuple(1.0 / n for _ in range(n))
    if key in {"front", "front_skewed"}:
        return normalize_weights([n - i for i in range(n)])
    if key in {"back", "back_skewed"}:
        return normalize_weights([i + 1 for i in range(n)])
    if key in {"cost", "cost_proportional"}:
        atomic = []
        for i in range(n):
            value, backend = model.best_terminal(i, i + 1)
            if backend is None or not math.isfinite(value):
                raise ValueError("cost-proportional weights require complete finite atomic costs")
            atomic.append(value)
        return normalize_weights(atomic)
    raise ValueError("fault-weight profile must be one of: uniform, front, back, cost")


def scale_query_costs(model: CostModel, scale: float) -> CostModel:
    factor = float(scale)
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("query scale must be finite and non-negative")

    def q(i: int, j: int, k: int) -> float:
        return factor * model.checked_query_cost(i, j, k)

    return CostModel(
        model.n, model.terminal_costs, q,
        terminal_capabilities=model.terminal_capabilities,
        metadata=model.metadata,
        backend_priority=model.backend_priority,
    )


def perturb_cost_model(
    model: CostModel,
    *,
    relative_noise: float,
    seed: int,
    perturb_queries: bool = True,
) -> CostModel:
    """Apply deterministic bounded multiplicative noise for stability analysis.

    Each finite terminal cost is multiplied by ``1 + u`` for
    ``u ~ Uniform[-relative_noise, relative_noise]``. Query costs can be
    perturbed independently with the same bounded distribution.  This is a
    sensitivity experiment only; it is never used to choose the reference
    policy or alter measured CSVs.
    """

    eps = float(relative_noise)
    if not math.isfinite(eps) or eps < 0 or eps >= 1:
        raise ValueError("relative_noise must satisfy 0 <= noise < 1")
    rng = random.Random(int(seed))
    terminal = {}
    for interval in sorted(model.terminal_costs):
        backend_map = {}
        for backend in sorted(model.terminal_costs[interval]):
            raw = float(model.terminal_costs[interval][backend])
            if math.isfinite(raw):
                raw *= 1.0 + rng.uniform(-eps, eps)
            backend_map[backend] = raw
        terminal[interval] = backend_map

    query_factors: dict[tuple[int, int, int], float] = {}
    if perturb_queries:
        for span in range(2, model.n + 1):
            for i in range(0, model.n - span + 1):
                j = i + span
                for k in range(i + 1, j):
                    query_factors[(i, j, k)] = 1.0 + rng.uniform(-eps, eps)

    def q(i: int, j: int, k: int) -> float:
        base = model.checked_query_cost(i, j, k)
        return base * query_factors.get((i, j, k), 1.0)

    return CostModel(
        model.n, terminal, q,
        terminal_capabilities=model.terminal_capabilities,
        metadata=model.metadata,
        backend_priority=model.backend_priority,
    )
