#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "templates"
DATA.mkdir(parents=True, exist_ok=True)

TRACE = [
    (0, "conv1", "Conv2D", "1x32x32", "6x28x28"),
    (1, "relu1", "ReLU", "6x28x28", "6x28x28"),
    (2, "pool1", "MaxPool2D", "6x28x28", "6x14x14"),
    (3, "conv2", "Conv2D", "6x14x14", "16x10x10"),
    (4, "relu2", "ReLU", "16x10x10", "16x10x10"),
    (5, "pool2", "MaxPool2D", "16x10x10", "16x5x5"),
    (6, "flatten", "Flatten", "16x5x5", "400"),
    (7, "fc1", "GEMM", "400", "120"),
    (8, "relu3", "ReLU", "120", "120"),
    (9, "fc2", "GEMM", "120", "84"),
    (10, "relu4", "ReLU", "84", "84"),
    (11, "fc3", "GEMM", "84", "10"),
]

with (DATA / "lenet5_trace.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["index", "name", "kind", "input_shape", "output_shape", "notes"])
    for row in TRACE:
        w.writerow([*row, "Canonical deterministic LeNet-5 dispute trace; accuracy is not evaluated."])

with (DATA / "interval_costs_template.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["i", "j", "span", "native_cycles", "zk_cycles", "source", "notes"])
    n = len(TRACE)
    for span in range(1, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            w.writerow([i, j, span, "", "", "", "Fill only measured/admissible backends; blank = unavailable."])

with (DATA / "query_costs_template.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["depth", "cycles", "witness_bytes", "source", "notes"])
    for depth in range(1, 9):
        w.writerow([depth, "", "", "", "Merkle/checkpoint authentication cost at this depth."])

print(f"Templates written under {DATA}")
