#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional

PATTERNS = [
    re.compile(r"Total cycles consumed:\s*([0-9]+)", re.I),
    re.compile(r"All cycles:\s*([0-9]+)", re.I),
    re.compile(r"total cycles[^0-9]*([0-9]+)", re.I),
]


def parse_cycles(text: str) -> Optional[int]:
    for pattern in PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def expected_intervals(n: int = 12, max_span: int = 4) -> set[tuple[int, int]]:
    if n <= 0 or max_span <= 0:
        raise ValueError("n and max_span must be positive")
    return {
        (i, i + span)
        for span in range(1, min(max_span, n) + 1)
        for i in range(0, n - span + 1)
    }


def parse_lenet_directory(
    log_dir: Path,
    *,
    n: int = 12,
    max_span: int = 4,
    strict: bool = True,
) -> list[dict]:
    rows = []
    seen: set[tuple[int, int]] = set()
    missing_cycles = []
    for path in sorted(log_dir.glob("lenet_*_*.log")):
        parts = path.stem.split("_")
        if len(parts) != 3:
            if strict:
                raise ValueError(f"Unexpected LeNet log filename: {path.name}")
            continue
        _, raw_i, raw_j = parts
        try:
            i, j = int(raw_i), int(raw_j)
        except ValueError as exc:
            raise ValueError(f"Invalid interval in filename: {path.name}") from exc
        if not (0 <= i < j <= n):
            raise ValueError(f"Out-of-range LeNet interval [{i},{j}] in {path.name}")
        key = (i, j)
        if key in seen:
            raise ValueError(f"Duplicate LeNet interval log [{i},{j}]")
        seen.add(key)
        c = parse_cycles(path.read_text(encoding="utf-8", errors="replace"))
        if c is None:
            missing_cycles.append(path.name)
        rows.append(
            {
                "i": i,
                "j": j,
                "span": j - i,
                "native_cycles": "" if c is None else c,
                "zk_cycles": "",
                "source": "ckb-debugger",
                "notes": f"state-chained deterministic LeNet interval; raw log {path.name}",
            }
        )

    expected = expected_intervals(n=n, max_span=max_span)
    missing_intervals = sorted(expected - seen)
    extra_intervals = sorted(seen - expected)
    if strict and missing_intervals:
        raise ValueError(
            "Missing expected LeNet logs: "
            + ", ".join(f"[{i},{j}]" for i, j in missing_intervals)
        )
    if strict and extra_intervals:
        raise ValueError(
            "Unexpected LeNet intervals beyond configured span: "
            + ", ".join(f"[{i},{j}]" for i, j in extra_intervals)
        )
    if strict and missing_cycles:
        raise ValueError("Cycle count not found in: " + ", ".join(missing_cycles))

    rows.sort(key=lambda r: (r["span"], r["i"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-span", type=int, default=4)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    rows = parse_lenet_directory(
        Path(args.logs),
        n=args.n,
        max_span=args.max_span,
        strict=not args.allow_incomplete,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "i",
                "j",
                "span",
                "native_cycles",
                "zk_cycles",
                "source",
                "notes",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
