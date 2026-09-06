#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional

CYCLE_PATTERNS = [
    re.compile(r"Total cycles consumed:\s*([0-9]+)", re.I),
    re.compile(r"All cycles:\s*([0-9]+)", re.I),
    re.compile(r"total cycles[^0-9]*([0-9]+)", re.I),
]


def parse_cycles(text: str) -> Optional[int]:
    """Extract a ckb-debugger cycle count from supported output formats."""
    for pat in CYCLE_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return None


def parse_log_directory(log_dir: Path, strict: bool = True) -> list[dict]:
    rows = []
    paths = sorted(log_dir.glob("*.log"))
    if not paths:
        raise ValueError(f"No .log files found in {log_dir}")
    seen = set()
    missing = []
    for path in paths:
        stem = path.stem
        if "_" not in stem:
            if strict:
                raise ValueError(f"Unexpected benchmark log filename: {path.name}")
            continue
        kind, raw_size = stem.rsplit("_", 1)
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise ValueError(f"Invalid benchmark size in filename: {path.name}") from exc
        key = (kind, size)
        if key in seen:
            raise ValueError(f"Duplicate benchmark log for {kind} size {size}")
        seen.add(key)
        text = path.read_text(encoding="utf-8", errors="replace")
        cycles = parse_cycles(text)
        if cycles is None:
            missing.append(path.name)
        rows.append(
            {
                "primitive": kind,
                "size": size,
                "cycles": "" if cycles is None else cycles,
                "log_file": path.name,
                "status": "ok" if cycles is not None else "cycle-line-not-found",
            }
        )
    if strict and missing:
        raise ValueError("Cycle count not found in: " + ", ".join(missing))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write rows with blank cycles instead of failing when a log has no cycle line",
    )
    args = p.parse_args()

    rows = parse_log_directory(Path(args.logs), strict=not args.allow_missing)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["primitive", "size", "cycles", "log_file", "status"]
        )
        w.writeheader()
        w.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
