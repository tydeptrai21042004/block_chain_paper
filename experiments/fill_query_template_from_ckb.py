#!/usr/bin/env python3
"""Copy measured Merkle-path cycles into the query-cost publication template."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measurements", default="ckb_bench/ckb_primitive_measurements.csv")
    p.add_argument("--template", default="experiments/data/templates/query_costs_template.csv")
    p.add_argument("--required-depth", type=int, default=4)
    args = p.parse_args()

    measured = {}
    with open(args.measurements, newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            if row.get("primitive") != "merkle":
                continue
            raw_cycles = (row.get("cycles") or "").strip()
            if not raw_cycles:
                continue
            depth = int(row["size"])
            if depth in measured:
                raise ValueError(f"Duplicate Merkle measurement at depth {depth}")
            cycles = float(raw_cycles)
            if cycles < 0:
                raise ValueError(f"Negative Merkle cycles at line {line_no}")
            measured[depth] = row["cycles"]

    if args.required_depth not in measured:
        raise ValueError(
            f"Required Merkle depth {args.required_depth} is missing from {args.measurements}"
        )

    path = Path(args.template)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Template is empty: {path}")

    for row in rows:
        depth = int(row["depth"])
        if depth in measured:
            row["cycles"] = measured[depth]
            row["source"] = "ckb-debugger"
            row["notes"] = (
                "Filled automatically from the CKB Merkle-path benchmark; verify raw log."
            )

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
