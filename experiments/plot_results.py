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

    pareto_path = results / "pareto_frontier.csv"
    if pareto_path.exists():
        pf = read_csv(pareto_path)
        if pf:
            fig, ax = plt.subplots(figsize=(5.8, 4.2))
            xs = [float(r["mean_cost"]) for r in pf]
            ys = [float(r["worst_case_cost"]) for r in pf]
            ax.plot(xs, ys, marker="o", linewidth=1.2)
            for r, x, y in zip(pf, xs, ys):
                if r.get("selected_minimax_safe", "").lower() == "true":
                    ax.annotate("minimax-safe", (x, y), xytext=(5, 5), textcoords="offset points")
                elif r.get("selected_mean_first", "").lower() == "true":
                    ax.annotate("mean-first", (x, y), xytext=(5, -12), textcoords="offset points")
            ax.set_xlabel("Mean fault-path cost")
            ax.set_ylabel("Worst-case fault-path cost")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out / "pareto_frontier.pdf")
            fig.savefig(out / "pareto_frontier.png", dpi=180)
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
