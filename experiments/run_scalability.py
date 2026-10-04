#!/usr/bin/env python3
from __future__ import annotations

"""Synthetic solver scalability/frontier-size experiment.

This benchmark studies algorithmic scaling only.  Costs are deterministic
synthetic families and outputs are marked NOT FOR MANUSCRIPT REAL-COST CLAIMS.
"""

import argparse
import math
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hndt.core import CostModel, solve_hndt
from hndt.pareto import solve_pareto_hndt
from hndt.reporting import write_csv


def _parse_ints(value: str) -> list[int]:
    data = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not data or any(x <= 0 for x in data):
        raise ValueError("sizes must be positive integers")
    return data


def atomic_costs(n: int, family: str) -> list[float]:
    if family == "homogeneous":
        return [100.0] * n
    if family == "mild":
        pattern = [82.0, 91.0, 103.0, 117.0, 109.0, 96.0]
        return [pattern[i % len(pattern)] for i in range(n)]
    if family == "strong":
        pattern = [18.0, 35.0, 70.0, 140.0, 280.0, 560.0, 45.0, 95.0]
        return [pattern[i % len(pattern)] for i in range(n)]
    raise ValueError("family must be homogeneous, mild, or strong")


def synthetic_model(n: int, family: str, max_span: int = 4) -> CostModel:
    atomic = atomic_costs(n, family)
    terminal = {}
    for span in range(1, min(max_span, n) + 1):
        for i in range(0, n - span + 1):
            j = i + span
            raw_sum = sum(atomic[i:j])
            # Interval verification receives a deterministic amortisation benefit.
            # This creates a nontrivial stop/split choice while remaining transparent.
            native = 12.0 + (0.90 if span > 1 else 1.0) * raw_sum
            terminal[(i, j)] = {"native": native}

    query = 22.0
    return CostModel(n, terminal, lambda i, j, k: query)


def run_one(n: int, family: str, max_span: int) -> dict:
    model = synthetic_model(n, family, max_span=max_span)

    t0 = time.perf_counter()
    scalar = solve_hndt(model)
    scalar_seconds = time.perf_counter() - t0

    tracemalloc.start()
    t0 = time.perf_counter()
    pareto = solve_pareto_hndt(model, collect_stats=True)
    pareto_seconds = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if abs(pareto.worst_optimum - scalar.optimum) > 1e-9:
        raise AssertionError("Pareto solver failed to recover scalar minimax optimum")
    stats = pareto.stats
    assert stats is not None
    num_intervals = n * (n + 1) // 2
    return {
        "data_status": "SYNTHETIC SCALABILITY - NOT REAL CKB COSTS",
        "family": family,
        "n": n,
        "max_measured_span": max_span,
        "subproblem_intervals": num_intervals,
        "scalar_seconds": scalar_seconds,
        "pareto_seconds": pareto_seconds,
        "pareto_to_scalar_time_ratio": pareto_seconds / scalar_seconds if scalar_seconds else math.inf,
        "peak_python_memory_mb": peak / (1024 * 1024),
        "scalar_worst": scalar.optimum,
        "pareto_worst": pareto.worst_optimum,
        "root_frontier_size": len(pareto.root_frontier),
        "peak_frontier_size": stats.peak_frontier_size,
        "candidates_generated": stats.candidates_generated,
        "labels_retained": stats.labels_retained,
        "duplicate_pruned": stats.duplicate_pruned,
        "dominated_pruned": stats.dominated_pruned,
        "pruning_ratio": stats.pruning_ratio,
        "peak_candidates_per_interval": stats.peak_candidates_per_interval,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="12,24,48")
    ap.add_argument("--families", default="homogeneous,mild,strong")
    ap.add_argument("--max-span", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "scalability" / "scalability.csv"))
    args = ap.parse_args()
    if args.max_span <= 0:
        raise ValueError("max-span must be positive")
    sizes = _parse_ints(args.sizes)
    families = [x.strip() for x in args.families.split(",") if x.strip()]

    rows = []
    for family in families:
        for n in sizes:
            row = run_one(n, family, args.max_span)
            rows.append(row)
            print(
                f"family={family} n={n} pareto={row['pareto_seconds']:.4f}s "
                f"peakP={row['peak_frontier_size']} prune={row['pruning_ratio']:.3f}"
            )
    out = Path(args.out)
    write_csv(out, rows)
    print(out)


if __name__ == "__main__":
    main()
