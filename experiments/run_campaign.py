#!/usr/bin/env python3
from __future__ import annotations

"""Run the same Pareto-HNDT experiment over multiple ordered traces.

The campaign runner never invents missing real measurements.  A campaign entry
must point to explicit trace/interval/query/config files.  Synthetic campaign
files must set ``synthetic: true`` and are marked NOT FOR MANUSCRIPT by
``run_experiment.py``.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=str(ROOT / "config" / "campaign_demo.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "campaign_current"))
    ap.add_argument("--fault-weights", choices=["uniform", "front", "back", "cost"], default="uniform")
    ap.add_argument("--query-scale", type=float, default=1.0)
    args = ap.parse_args()

    campaign_path = Path(args.campaign)
    config = json.loads(campaign_path.read_text(encoding="utf-8"))
    runs = config.get("runs", [])
    if not runs:
        raise ValueError("campaign must define a non-empty runs array")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    combined_summary: list[dict] = []
    combined_gain: list[dict] = []
    metadata_rows: list[dict] = []

    for entry in runs:
        name = str(entry["name"])
        run_out = out / "per_trace" / name
        cmd = [
            sys.executable,
            str(ROOT / "run_experiment.py"),
            "--trace", str(_resolve(entry["trace"])),
            "--interval-costs", str(_resolve(entry["interval_costs"])),
            "--query-costs", str(_resolve(entry["query_costs"])),
            "--config", str(_resolve(entry["config"])),
            "--out", str(run_out),
            "--fault-weights", args.fault_weights,
            "--query-scale", str(args.query_scale),
            "--collect-pareto-stats",
        ]
        if bool(entry.get("synthetic", False)):
            cmd.append("--synthetic-data")
        subprocess.run(cmd, check=True, cwd=ROOT.parent)

        for row in _read_csv(run_out / "summary.csv"):
            combined_summary.append({"trace": name, **row})
        for row in _read_csv(run_out / "proposal_gain.csv"):
            combined_gain.append({"trace": name, **row})
        meta = json.loads((run_out / "run_metadata.json").read_text(encoding="utf-8"))
        metadata_rows.append(
            {
                "trace": name,
                "data_status": meta["data_status"],
                "n": meta["n"],
                "pareto_worst": meta["pareto_hndt_worst_optimum"],
                "pareto_mean": meta["pareto_hndt_mean_at_optimum"],
                "frontier_size": meta["pareto_root_frontier_size"],
                "pareto_rounds": meta["pareto_hndt_max_rounds"],
                "scalar_rounds": meta["scalar_hndt_max_rounds"],
                "query_scale": meta["query_scale"],
                "fault_weight_profile": meta["fault_weight_profile"],
            }
        )

    _write_csv(out / "combined_summary.csv", combined_summary)
    _write_csv(out / "combined_gain.csv", combined_gain)
    _write_csv(out / "combined_metadata.csv", metadata_rows)
    (out / "campaign_metadata.json").write_text(
        json.dumps(
            {
                "campaign": str(campaign_path),
                "fault_weight_profile": args.fault_weights,
                "query_scale": args.query_scale,
                "runs": [entry["name"] for entry in runs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
