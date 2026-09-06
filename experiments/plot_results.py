#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="experiments/results/current")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_csv(results / "fault_costs.csv")
    series = defaultdict(list)
    for r in rows:
        series[r["strategy"]].append((int(r["fault_index"]), float(r["total_cost"])))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name, pts in series.items():
        pts.sort()
        ax.plot([x + 1 for x, _ in pts], [y for _, y in pts], marker="o", linewidth=1.4, label=name)
    ax.set_xlabel("Injected fault position (operator index)")
    ax.set_ylabel("Dispute cost (CKB-VM cycles or chosen cost unit)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fault_position_cost.pdf")
    fig.savefig(out / "fault_position_cost.png", dpi=180)
    plt.close(fig)

    ab = read_csv(results / "heterogeneity_ablation.csv")
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    labels = [r["cost_model"] for r in ab]
    values = [float(r["optimal_worst_case_cost"]) for r in ab]
    ax.bar(labels, values)
    ax.set_ylabel("Optimal worst-case cost")
    ax.set_xlabel("Cost model")
    fig.tight_layout()
    fig.savefig(out / "heterogeneity_ablation.pdf")
    fig.savefig(out / "heterogeneity_ablation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
