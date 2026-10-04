#!/usr/bin/env python3
from __future__ import annotations

"""One-command V5 evidence suite.

Demo mode uses only clearly labelled synthetic data. Real mode requires an
explicit campaign JSON whose entries point to real ckb-debugger measurements;
missing measurements are never replaced by synthetic values.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT.parent)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "real"], default="demo")
    ap.add_argument("--campaign", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "v5_suite"))
    ap.add_argument("--robustness-trials", type=int, default=100)
    ap.add_argument("--scalability-sizes", default="12,24,48")
    args = ap.parse_args()

    if args.robustness_trials <= 0:
        raise ValueError("robustness-trials must be positive")

    campaign = Path(
        args.campaign
        or (
            ROOT / "config" / "campaign_demo.json"
            if args.mode == "demo"
            else ROOT / "config" / "campaign_real.example.json"
        )
    )
    cfg = json.loads(campaign.read_text(encoding="utf-8"))
    runs = cfg.get("runs", [])
    if not runs:
        raise ValueError("campaign contains no runs")
    if args.mode == "real" and any(bool(entry.get("synthetic", False)) for entry in runs):
        raise ValueError("real mode refuses campaign entries marked synthetic")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable,
        str(ROOT / "run_campaign.py"),
        "--campaign", str(campaign),
        "--out", str(out / "campaign"),
    ])

    for entry in runs:
        name = str(entry["name"])
        common = [
            "--trace", str(resolve(entry["trace"])),
            "--interval-costs", str(resolve(entry["interval_costs"])),
            "--query-costs", str(resolve(entry["query_costs"])),
            "--config", str(resolve(entry["config"])),
        ]
        run([
            sys.executable,
            str(ROOT / "run_sensitivity.py"),
            *common,
            "--out", str(out / "sensitivity" / name),
        ])
        run([
            sys.executable,
            str(ROOT / "run_robustness.py"),
            *common,
            "--out", str(out / "robustness" / name),
            "--trials", str(args.robustness_trials),
        ])

    run([
        sys.executable,
        str(ROOT / "run_scalability.py"),
        "--sizes", args.scalability_sizes,
        "--out", str(out / "scalability" / "scalability.csv"),
    ])

    (out / "SUITE_STATUS.txt").write_text(
        "V5 evidence suite completed.\n"
        f"Mode: {args.mode}\n"
        f"Campaign: {campaign}\n"
        + (
            "All campaign inputs were required to be real/user-supplied measurements.\n"
            if args.mode == "real"
            else "Synthetic demo mode: outputs are NOT FOR MANUSCRIPT.\n"
        ),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
