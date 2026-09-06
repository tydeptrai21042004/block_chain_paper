#!/usr/bin/env python3
"""Merge measured LeNet interval cycles into the publication template."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def expected_intervals(n: int, max_span: int) -> set[tuple[int, int]]:
    return {
        (i, i + span)
        for span in range(1, min(max_span, n) + 1)
        for i in range(0, n - span + 1)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", default="ckb_bench/lenet_interval_measurements.csv")
    ap.add_argument("--template", default="experiments/data/templates/interval_costs_template.csv")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-span", type=int, default=4)
    args = ap.parse_args()

    measured = {}
    with open(args.measurements, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, r in enumerate(reader, start=2):
            key = (int(r["i"]), int(r["j"]))
            if key in measured:
                raise ValueError(f"Duplicate measured interval {key} at line {line_no}")
            raw = (r.get("native_cycles") or "").strip()
            if not raw:
                raise ValueError(f"Missing native_cycles for measured interval {key}")
            cycles = float(raw)
            if cycles < 0:
                raise ValueError(f"Negative native_cycles for interval {key}")
            measured[key] = r

    expected = expected_intervals(args.n, args.max_span)
    missing = sorted(expected - set(measured))
    if missing:
        raise ValueError(
            "Measurement CSV is incomplete for the required fair comparison set: "
            + ", ".join(f"[{i},{j}]" for i, j in missing)
        )

    path = Path(args.template)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Template is empty: {path}")

    template_keys = {(int(r["i"]), int(r["j"])) for r in rows}
    absent = sorted(expected - template_keys)
    if absent:
        raise ValueError(f"Template is missing required intervals: {absent}")

    for r in rows:
        key = (int(r["i"]), int(r["j"]))
        if key in measured:
            m = measured[key]
            r["native_cycles"] = m["native_cycles"]
            r["source"] = "ckb-debugger"
            r["notes"] = m.get("notes", "")

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
