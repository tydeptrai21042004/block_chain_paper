from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

from .core import Action
from .evaluate import PathResult

Interval = Tuple[int, int]


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_fault_paths(
    path: Path,
    strategy_to_paths: Mapping[str, Iterable[PathResult]],
) -> None:
    rows = []
    for strategy, paths in strategy_to_paths.items():
        for p in paths:
            rows.append(
                {
                    "strategy": strategy,
                    "fault_index": p.fault_index,
                    "fault_position": p.fault_index + 1,
                    "total_cost": p.total_cost,
                    "rounds": p.rounds,
                    "terminal_i": p.terminal_i,
                    "terminal_j": p.terminal_j,
                    "leaf_size": p.terminal_j - p.terminal_i,
                    "backend": p.backend,
                }
            )
    write_csv(path, rows)


def write_policy_json(path: Path, policy: Mapping[Interval, Action]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for (i, j), act in sorted(policy.items()):
        key = f"{i}:{j}"
        data[key] = {"kind": act.kind, "backend": act.backend, "split": act.split}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def first_split(policy: Mapping[Interval, Action], n: int):
    root = (0, n)
    if root not in policy:
        raise ValueError(f"policy is missing root interval {root}")
    act = policy[root]
    return act.split if act.kind == "split" else "settle"
