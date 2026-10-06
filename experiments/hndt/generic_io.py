from __future__ import annotations

"""Generic CSV/JSON input for platform-independent Pareto-DPS instances."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .dps import QueryActionSpec, TerminalActionSpec, VerificationInstance

Interval = Tuple[int, int]


def _require_columns(fieldnames, required, path: Path) -> None:
    present = set(fieldnames or [])
    missing = [x for x in required if x not in present]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def read_generic_trace(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_columns(reader.fieldnames, ["index"], path)
        rows = list(reader)
    if not rows:
        raise ValueError(f"trace is empty: {path}")
    indices = [int(row["index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError(f"trace indices must be consecutive 0..n-1; got {indices}")
    return rows


def _parse_capabilities(text: str | None) -> frozenset[str]:
    if text is None:
        return frozenset()
    raw = str(text).strip()
    if not raw:
        return frozenset()
    return frozenset(x.strip() for x in raw.replace(";", ",").split(",") if x.strip())


def read_terminal_actions(path: str | Path) -> Dict[Interval, Tuple[TerminalActionSpec, ...]]:
    """Read long-format terminal actions.

    Required columns: ``i,j,action_id,cost``.
    Optional column: ``capabilities`` (comma/semicolon separated tags).
    """

    path = Path(path)
    grouped: dict[Interval, list[TerminalActionSpec]] = defaultdict(list)
    seen = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_columns(reader.fieldnames, ["i", "j", "action_id", "cost"], path)
        for line_no, row in enumerate(reader, start=2):
            i, j = int(row["i"]), int(row["j"])
            action_id = str(row["action_id"]).strip()
            key = (i, j, action_id)
            if key in seen:
                raise ValueError(f"duplicate terminal action {key!r} at {path}:{line_no}")
            seen.add(key)
            grouped[(i, j)].append(
                TerminalActionSpec(
                    action_id,
                    row["cost"],
                    _parse_capabilities(row.get("capabilities")),
                )
            )
    return {interval: tuple(actions) for interval, actions in grouped.items()}


def _parse_cuts(text: str) -> tuple[int, ...]:
    raw = str(text).strip()
    if not raw:
        raise ValueError("query cuts must be non-empty")
    return tuple(int(x.strip()) for x in raw.replace(",", ";").split(";") if x.strip())


def read_query_actions(path: str | Path) -> Dict[Interval, Tuple[QueryActionSpec, ...]]:
    """Read long-format query actions.

    Required columns: ``i,j,action_id,cuts,cost``.  ``cuts`` uses ``;`` between
    absolute checkpoint indices, e.g. ``4`` for binary or ``4;8`` for ternary.
    """

    path = Path(path)
    grouped: dict[Interval, list[QueryActionSpec]] = defaultdict(list)
    seen = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_columns(reader.fieldnames, ["i", "j", "action_id", "cuts", "cost"], path)
        for line_no, row in enumerate(reader, start=2):
            i, j = int(row["i"]), int(row["j"])
            action_id = str(row["action_id"]).strip()
            key = (i, j, action_id)
            if key in seen:
                raise ValueError(f"duplicate query action {key!r} at {path}:{line_no}")
            seen.add(key)
            grouped[(i, j)].append(
                QueryActionSpec(action_id, _parse_cuts(row["cuts"]), row["cost"])
            )
    return {interval: tuple(actions) for interval, actions in grouped.items()}


def load_metadata(path: str | Path | None) -> dict:
    if path is None:
        return {}
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("metadata JSON must contain an object")
    return value


def build_verification_instance(
    trace_path: str | Path,
    terminal_actions_path: str | Path,
    query_actions_path: str | Path,
    metadata_path: str | Path | None = None,
) -> VerificationInstance:
    trace = read_generic_trace(trace_path)
    n = len(trace)
    terminals = read_terminal_actions(terminal_actions_path)
    queries_by_interval = read_query_actions(query_actions_path)

    for i, j in list(terminals) + list(queries_by_interval):
        if not (0 <= i < j <= n):
            raise ValueError(f"action interval [{i},{j}] exceeds trace length n={n}")

    def queries(i: int, j: int):
        return queries_by_interval.get((i, j), ())

    return VerificationInstance(
        n=n,
        terminal_actions=terminals,
        query_actions=queries,
        metadata=load_metadata(metadata_path),
    )
