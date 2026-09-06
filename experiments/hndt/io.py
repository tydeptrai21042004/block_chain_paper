from __future__ import annotations

import csv
import json
import math
from math import isfinite, isnan
from pathlib import Path
from typing import Dict, List, Tuple

from .core import CostModel


def _require_columns(fieldnames, required, path: Path) -> None:
    present = set(fieldnames or [])
    missing = [x for x in required if x not in present]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def read_trace(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_columns(reader.fieldnames, ["index", "name", "kind"], path)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Trace file is empty: {path}")
    try:
        indices = [int(r["index"]) for r in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Trace indices in {path} must be integers") from exc
    expected = list(range(len(rows)))
    if indices != expected:
        raise ValueError(f"Trace indices must be consecutive 0..n-1; got {indices}")
    names = [(r.get("name") or "").strip() for r in rows]
    if any(not x for x in names):
        raise ValueError(f"Trace operator names must be non-empty: {path}")
    if len(set(names)) != len(names):
        raise ValueError(f"Trace operator names must be unique: {path}")
    return rows


def _optional_float(value: str | None):
    if value is None or str(value).strip() == "":
        return None
    out = float(value)
    if isnan(out) or out < 0:
        raise ValueError("cost values must be non-negative and not NaN")
    return out


def read_interval_costs(path: str | Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    path = Path(path)
    costs: Dict[Tuple[int, int], Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_columns(reader.fieldnames, ["i", "j", "native_cycles", "zk_cycles"], path)
        for line_no, row in enumerate(reader, start=2):
            try:
                i, j = int(row["i"]), int(row["j"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid interval endpoints at {path}:{line_no}") from exc
            if i < 0 or j <= i:
                raise ValueError(f"Invalid interval [{i},{j}] at {path}:{line_no}")
            if row.get("span") not in (None, ""):
                try:
                    span = int(row["span"])
                except ValueError as exc:
                    raise ValueError(f"Invalid span at {path}:{line_no}") from exc
                if span != j - i:
                    raise ValueError(
                        f"Span mismatch at {path}:{line_no}: span={span}, interval=[{i},{j}]"
                    )
            key = (i, j)
            if key in costs:
                raise ValueError(f"Duplicate interval [{i},{j}] in {path}")
            backends: Dict[str, float] = {}
            native = _optional_float(row.get("native_cycles"))
            zk = _optional_float(row.get("zk_cycles"))
            if native is not None and isfinite(native):
                backends["native"] = native
            if zk is not None and isfinite(zk):
                backends["zkvm"] = zk
            if backends:
                costs[key] = backends
    return costs


def read_query_depth_costs(path: str | Path) -> Dict[int, float]:
    path = Path(path)
    data: Dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_columns(reader.fieldnames, ["depth", "cycles"], path)
        for line_no, row in enumerate(reader, start=2):
            raw = (row.get("depth") or "").strip()
            cycles = _optional_float(row.get("cycles"))
            if not raw and cycles is None:
                continue
            try:
                depth = int(raw)
            except ValueError as exc:
                raise ValueError(f"Invalid Merkle depth at {path}:{line_no}") from exc
            if depth < 1:
                raise ValueError(f"Merkle depth must be >= 1 at {path}:{line_no}")
            if depth in data:
                raise ValueError(f"Duplicate Merkle depth {depth} in {path}")
            if cycles is not None and isfinite(cycles):
                data[depth] = cycles
    return data


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    return data


def build_model(
    trace_path: str | Path,
    interval_path: str | Path,
    query_path: str | Path,
    config_path: str | Path,
) -> CostModel:
    trace = read_trace(trace_path)
    n = len(trace)
    terminal = read_interval_costs(interval_path)
    for i, j in terminal:
        if j > n:
            raise ValueError(
                f"Terminal interval [{i},{j}] exceeds trace length n={n}: {interval_path}"
            )

    query_by_depth = read_query_depth_costs(query_path)
    config = load_config(config_path)

    constant_query = config.get("constant_query_cycles")
    query_overhead = float(config.get("query_fixed_overhead_cycles", 0.0))
    if isnan(query_overhead) or query_overhead < 0 or not isfinite(query_overhead):
        raise ValueError("query_fixed_overhead_cycles must be finite and non-negative")

    merkle_leaf_count = int(config.get("merkle_leaf_count", n + 1))
    if merkle_leaf_count < n + 1:
        raise ValueError(
            f"merkle_leaf_count={merkle_leaf_count} is too small for {n + 1} committed states"
        )
    required_depth = int(math.ceil(math.log2(max(2, merkle_leaf_count))))

    if constant_query is not None:
        q_base = float(constant_query)
        if isnan(q_base) or q_base < 0 or not isfinite(q_base):
            raise ValueError("constant_query_cycles must be finite and non-negative")
        q_value = q_base + query_overhead
    else:
        if required_depth not in query_by_depth:
            raise ValueError(
                f"No query cost for Merkle depth {required_depth}. Fill depth={required_depth} "
                f"in {query_path}, or set constant_query_cycles in {config_path}."
            )
        q_value = float(query_by_depth[required_depth]) + query_overhead

    def q(i: int, j: int, k: int) -> float:
        del i, j, k
        return q_value

    return CostModel(n=n, terminal_costs=terminal, query_cost=q)


def validate_atomic_coverage(model: CostModel) -> List[Tuple[int, int]]:
    missing = []
    for i in range(model.n):
        cost, backend = model.best_terminal(i, i + 1)
        if backend is None or not math.isfinite(cost):
            missing.append((i, i + 1))
    return missing
