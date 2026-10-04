#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from parse_trace_logs import expected_intervals, parse_cycles, parse_trace_directory


def parse_lenet_directory(log_dir: Path, *, n: int = 12, max_span: int = 4, strict: bool = True) -> list[dict]:
    return parse_trace_directory(log_dir, trace="lenet", n=n, max_span=max_span, strict=strict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-span", type=int, default=4)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    rows = parse_lenet_directory(Path(args.logs), n=args.n, max_span=args.max_span, strict=not args.allow_incomplete)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["i", "j", "span", "native_cycles", "zk_cycles", "source", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
