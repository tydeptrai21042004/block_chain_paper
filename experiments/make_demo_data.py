#!/usr/bin/env python3
"""Create deterministic synthetic data for software tests only.

These numbers are intentionally artificial and MUST NOT be copied into the paper.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "data" / "templates"
DEMO = ROOT / "data" / "demo"
DEMO.mkdir(parents=True, exist_ok=True)

kind_cost = {
    "Conv2D": 90.0,
    "ReLU": 8.0,
    "MaxPool2D": 18.0,
    "Flatten": 3.0,
    "GEMM": 55.0,
}

with (TEMPLATES / "lenet5_trace.csv").open(newline="", encoding="utf-8") as f:
    trace = list(csv.DictReader(f))
atomic = [kind_cost[r["kind"]] for r in trace]
n = len(atomic)

with (DEMO / "interval_costs_demo.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["i", "j", "span", "native_cycles", "zk_cycles", "source", "notes"])
    for span in range(1, n + 1):
        for i in range(n - span + 1):
            j = i + span
            # Synthetic availability: native can settle up to four operators; a fixed
            # overhead makes stop/split nontrivial. ZK is available for spans >= 3.
            native = 12.0 + sum(atomic[i:j]) if span <= 4 else ""
            zk = 190.0 if span >= 3 else ""
            w.writerow([i, j, span, native, zk, "synthetic-demo", "NOT FOR MANUSCRIPT"])

with (DEMO / "query_costs_demo.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["depth", "cycles", "witness_bytes", "source", "notes"])
    for depth in range(1, 9):
        w.writerow([depth, 6.0 + 2.0 * depth, 32 * depth, "synthetic-demo", "NOT FOR MANUSCRIPT"])

print(f"Synthetic demo data written under {DEMO}")
