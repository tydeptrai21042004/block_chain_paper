from __future__ import annotations

"""Exact numeric helpers for discrete verification-cost optimization.

The measured CKB quantities are integer cycle counts, while sensitivity studies
may supply finite decimal values.  Converting through the human-readable decimal
representation makes comparisons exact with respect to the values supplied to
this repository and avoids tolerance-dependent Pareto pruning.
"""

from decimal import Decimal
from fractions import Fraction
from math import isfinite
from numbers import Integral, Real
from typing import Sequence

Exact = Fraction


def as_fraction(value) -> Fraction:
    """Convert a finite numeric value to an exact :class:`Fraction`.

    Floats are converted through ``str(value)`` rather than their binary IEEE-754
    representation.  Thus ``1.5`` becomes exactly ``3/2`` and measured integer
    cycles stored as ``37382.0`` become exactly ``37382``.
    """

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid costs or weights")
    if isinstance(value, Integral):
        return Fraction(int(value), 1)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("value must be finite")
        return Fraction(value)
    if isinstance(value, Real):
        number = float(value)
        if not isfinite(number):
            raise ValueError("value must be finite")
        return Fraction(str(number))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("numeric string must be non-empty")
        decimal = Decimal(text)
        if not decimal.is_finite():
            raise ValueError("value must be finite")
        return Fraction(decimal)
    raise TypeError(f"unsupported exact numeric type: {type(value).__name__}")


def normalize_weight_fractions(
    n: int,
    weights: Sequence[float] | None = None,
) -> tuple[Fraction, ...]:
    """Return exact non-negative normalized weights summing to one."""

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    if weights is None:
        return tuple(Fraction(1, n) for _ in range(n))
    if len(weights) != n:
        raise ValueError(f"fault weights must have length {n}, got {len(weights)}")

    values = tuple(as_fraction(x) for x in weights)
    if any(x < 0 for x in values):
        raise ValueError("fault weights must be non-negative")
    total = sum(values, Fraction(0, 1))
    if total <= 0:
        raise ValueError("at least one fault weight must be positive")
    return tuple(x / total for x in values)
