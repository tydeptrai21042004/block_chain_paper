#!/usr/bin/env python3
"""Create deterministic synthetic data for software tests only.

These numbers are artificial and MUST NOT be copied into the manuscript.  The
multi-trace demo exists to exercise the V5 campaign/sensitivity/scalability code
before real ckb-debugger measurements are supplied.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "data" / "templates"
TRACES = ROOT / "data" / "traces"
DEMO = ROOT / "data" / "demo"
DEMO.mkdir(parents=True, exist_ok=True)

kind_cost = {
    "Conv2D": 90.0,
    "ReLU": 8.0,
    "MaxPool2D": 18.0,
    "Flatten": 3.0,
    "GEMM": 55.0,
}

trace_paths = {
    "lenet": TEMPLATES / "lenet5_trace.csv",
    "conv_heavy": TRACES / "conv_heavy_trace.csv",
    "gemm_heavy": TRACES / "gemm_heavy_trace.csv",
}


def write_interval_demo(key: str, trace_path: Path) -> Path:
    with trace_path.open(newline="", encoding="utf-8") as f:
        trace = list(csv.DictReader(f))
    atomic = [kind_cost[row["kind"]] for row in trace]
    n = len(atomic)
    out = DEMO / f"interval_costs_{key}_demo.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j", "span", "native_cycles", "zk_cycles", "source", "notes"])
        for span in range(1, n + 1):
            for i in range(n - span + 1):
                j = i + span
                native = 12.0 + sum(atomic[i:j]) if span <= 4 else ""
                zk = 190.0 if span >= 3 else ""
                writer.writerow([i, j, span, native, zk, "synthetic-demo", "NOT FOR MANUSCRIPT"])
    return out


for key, trace_path in trace_paths.items():
    write_interval_demo(key, trace_path)

# Backward-compatible filename used by the original --demo mode.
shutil.copyfile(DEMO / "interval_costs_lenet_demo.csv", DEMO / "interval_costs_demo.csv")

with (DEMO / "query_costs_demo.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["depth", "cycles", "witness_bytes", "source", "notes"])
    for depth in range(1, 9):
        writer.writerow([depth, 6.0 + 2.0 * depth, 32 * depth, "synthetic-demo", "NOT FOR MANUSCRIPT"])

print(f"Synthetic multi-trace demo data written under {DEMO}")
