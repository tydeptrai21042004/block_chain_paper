#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hndt.baselines import (
    direct_native_policy,
    fixed_g_policy,
    has_complete_atomic_backend,
    midpoint_adaptive_stop_policy,
    midpoint_operator_zk_policy,
    optimal_split_atomic_policy,
)
from hndt.core import CostModel, solve_hndt, solve_hndt_round_budget
from hndt.evaluate import evaluate_policy, summarize
from hndt.io import build_model, validate_atomic_coverage
from hndt.literature_baselines import (
    literature_baselines,
    policy_signature,
    provenance_rows,
)
from hndt.reporting import first_split, write_csv, write_fault_paths, write_policy_json


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the HNDT/CellVG ordered-trace dispute experiment"
    )
    p.add_argument("--demo", action="store_true", help="Use synthetic costs only for code sanity checks")
    p.add_argument("--trace", default=str(ROOT / "data" / "templates" / "lenet5_trace.csv"))
    p.add_argument("--interval-costs", default=None)
    p.add_argument("--query-costs", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=str(ROOT / "results" / "current"))
    p.add_argument(
        "--fixed-g",
        default="2,4",
        help="Comma-separated fixed granularities to report in addition to the oracle best fixed-g ablation",
    )
    p.add_argument(
        "--round-budget",
        type=int,
        default=None,
        help="Optionally add one latency-constrained HNDT policy with at most this many split rounds",
    )
    return p.parse_args()


def homogenized_model(model: CostModel) -> CostModel:
    atomic = []
    for i in range(model.n):
        c, _ = model.best_terminal(i, i + 1)
        if not math.isfinite(c):
            raise ValueError("Cannot homogenize without complete atomic terminal costs")
        atomic.append(c)
    mean_atomic = sum(atomic) / len(atomic)

    terminal = {}
    for (i, j), backends in model.terminal_costs.items():
        new_backends = dict(backends)
        if "native" in new_backends:
            # Preserve measured interval overhead while replacing heterogeneous
            # atomic work by a uniform mean. Availability is unchanged.
            measured_atomic_sum = sum(atomic[i:j])
            residual = float(new_backends["native"]) - measured_atomic_sum
            homogenized = residual + (j - i) * mean_atomic
            if homogenized < 0:
                raise ValueError(
                    f"Homogenization would make native cost negative on [{i},{j}]. "
                    "The interval table is incompatible with the additive-residual ablation."
                )
            new_backends["native"] = homogenized
        terminal[(i, j)] = new_backends
    return CostModel(model.n, terminal, model.query_cost)


def _slug(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return value or "policy"


def _worst(paths) -> float:
    return max(p.total_cost for p in paths)


def choose_best_fixed_g(model: CostModel):
    """Oracle-tune one global terminal granularity on the evaluated cost table.

    This is intentionally labelled an oracle internal ablation, not prior work.
    It gives fixed-granularity localization its strongest possible global choice
    without allowing per-interval adaptation.
    """

    candidates = []
    for g in range(1, model.n + 1):
        try:
            policy = fixed_g_policy(model, g)
            paths = evaluate_policy(model, policy)
        except ValueError:
            continue
        candidates.append((_worst(paths), g, policy, paths))
    if not candidates:
        raise ValueError("No feasible fixed-g policy")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0]


def build_literature_groups(model: CostModel):
    """Execute literature adapters and deduplicate identical policy trees."""

    executed = []
    status = []
    for spec in literature_baselines():
        try:
            policy = spec.builder(model)
            executed.append((spec, policy))
            status.append({"strategy": spec.display_name, "status": "executed", "reason": ""})
        except ValueError as exc:
            status.append({"strategy": spec.display_name, "status": "skipped", "reason": str(exc)})

    grouped = {}
    for spec, policy in executed:
        grouped.setdefault(policy_signature(policy), []).append((spec, policy))

    groups = []
    for index, members in enumerate(grouped.values(), start=1):
        specs = [spec for spec, _ in members]
        representative = members[0][1]
        if len(specs) > 1:
            name = f"Prior-work midpoint/pinpoint family ({len(specs)} adapted policies)"
        else:
            name = specs[0].display_name
        groups.append(
            {
                "group_id": index,
                "name": name,
                "members": specs,
                "policy": representative,
                "member_names": "; ".join(spec.display_name for spec in specs),
            }
        )
    return executed, groups, status


def build_policy_audit(model: CostModel, h) -> list[dict]:
    rows = []
    for (i, j), act in sorted(h.action.items()):
        terminal_cost, terminal_backend = model.best_terminal(i, j)
        candidates = []
        if terminal_backend is not None and math.isfinite(terminal_cost):
            candidates.append((terminal_cost, f"settle:{terminal_backend}"))

        split_details = {}
        if j - i > 1:
            for k in range(i + 1, j):
                q = model.checked_query_cost(i, j, k)
                left = h.value[(i, k)]
                right = h.value[(k, j)]
                if not all(map(math.isfinite, [q, left, right])):
                    continue
                cost = q + max(left, right)
                candidates.append((cost, f"split:{k}"))
                split_details[k] = (q, left, right, cost)

        chosen_label = (
            f"settle:{act.backend}" if act.kind == "settle" else f"split:{int(act.split)}"
        )
        alternatives = sorted(
            (cost, label) for cost, label in candidates if label != chosen_label
        )
        alt_cost, alt_label = alternatives[0] if alternatives else (math.inf, "")
        margin = alt_cost - h.value[(i, j)] if math.isfinite(alt_cost) else math.inf
        rel_margin = (
            margin / h.value[(i, j)]
            if math.isfinite(margin) and h.value[(i, j)] > 0
            else math.inf
        )

        row = {
            "i": i,
            "j": j,
            "span": j - i,
            "chosen_action": act.kind,
            "chosen_backend": act.backend or "",
            "chosen_split": act.split if act.split is not None else "",
            "optimal_subproblem_cost": h.value[(i, j)],
            "best_terminal_backend": terminal_backend or "",
            "best_terminal_cost": terminal_cost if math.isfinite(terminal_cost) else "",
            "best_alternative_action": alt_label,
            "best_alternative_cost": alt_cost if math.isfinite(alt_cost) else "",
            "action_margin": margin if math.isfinite(margin) else "",
            "relative_action_margin": rel_margin if math.isfinite(rel_margin) else "",
            "query_cost": "",
            "left_cost": "",
            "right_cost": "",
            "chosen_split_cost": "",
        }
        if act.kind == "split" and act.split is not None:
            k = int(act.split)
            q, left, right, split_cost = split_details[k]
            row["query_cost"] = q
            row["left_cost"] = left
            row["right_cost"] = right
            row["chosen_split_cost"] = split_cost
        rows.append(row)
    return rows


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        interval = ROOT / "data" / "demo" / "interval_costs_demo.csv"
        query = ROOT / "data" / "demo" / "query_costs_demo.csv"
        config = ROOT / "config" / "demo.json"
        marker = "SYNTHETIC DEMO - NOT FOR MANUSCRIPT"
    else:
        interval = Path(args.interval_costs or ROOT / "data" / "templates" / "interval_costs_template.csv")
        query = Path(args.query_costs or ROOT / "data" / "templates" / "query_costs_template.csv")
        config = Path(args.config or ROOT / "config" / "experiment.json")
        marker = "REAL/USER-SUPPLIED MEASUREMENTS"

    try:
        model = build_model(args.trace, interval, query, config)
        missing = validate_atomic_coverage(model)
        if missing:
            raise ValueError(
                "Missing terminal measurements for atomic intervals: "
                + ", ".join(f"[{i},{j}]" for i, j in missing)
            )
    except Exception as exc:
        out.joinpath("MISSING_MEASUREMENTS.txt").write_text(
            f"Experiment not executed.\nReason: {exc}\n\n"
            "Fill experiments/data/templates/interval_costs_template.csv and "
            "query_costs_template.csv, or run `python experiments/run_experiment.py --demo` "
            "to test the software with clearly synthetic data.\n",
            encoding="utf-8",
        )
        raise

    h = solve_hndt(model)

    policies = {"HNDT": h.action}
    categories = {"HNDT": "proposal"}

    # Paper-grounded adaptations are executed individually for provenance, then
    # deduplicated in the main table when they induce the same tree on this
    # ordered trace. This prevents three identical rows from looking like three
    # independent numerical competitors.
    literature_executed, literature_groups, literature_status = build_literature_groups(model)
    for group in literature_groups:
        policies[group["name"]] = group["policy"]
        categories[group["name"]] = "paper-supported prior-work family"

    # Mechanism-isolation ablations.
    policies["Ablation: adaptive split + atomic stop"] = optimal_split_atomic_policy(model)
    categories["Ablation: adaptive split + atomic stop"] = "mechanism ablation"
    policies["Ablation: midpoint split + adaptive stop"] = midpoint_adaptive_stop_policy(model)
    categories["Ablation: midpoint split + adaptive stop"] = "mechanism ablation"

    # Strongest globally fixed granularity, selected with oracle access to this
    # evaluation cost table. This is deliberately not called prior work.
    best_g_cost, best_g, best_g_policy, _ = choose_best_fixed_g(model)
    best_g_name = f"Ablation: oracle best fixed-g (g={best_g})"
    policies[best_g_name] = best_g_policy
    categories[best_g_name] = "oracle internal ablation"

    requested_g = []
    for raw_g in args.fixed_g.split(","):
        raw_g = raw_g.strip()
        if not raw_g:
            continue
        g = int(raw_g)
        if g in requested_g or g == best_g:
            continue
        requested_g.append(g)
        name = f"Ablation: Fixed g={g}"
        policies[name] = fixed_g_policy(model, g)
        categories[name] = "internal ablation"

    optional_status = []
    try:
        policies["Ablation: direct full native"] = direct_native_policy(model)
        categories["Ablation: direct full native"] = "extreme internal ablation"
        optional_status.append({"strategy": "Ablation: direct full native", "status": "executed", "reason": ""})
    except ValueError as exc:
        optional_status.append({"strategy": "Ablation: direct full native", "status": "skipped", "reason": str(exc)})

    if has_complete_atomic_backend(model, "zkvm"):
        name = "Ablation: operator midpoint + reproduced ZK"
        policies[name] = midpoint_operator_zk_policy(model)
        categories[name] = "cross-backend ablation"
        optional_status.append({"strategy": name, "status": "executed", "reason": ""})
    else:
        optional_status.append(
            {
                "strategy": "Ablation: operator midpoint + reproduced ZK",
                "status": "skipped",
                "reason": "complete reproduced zkvm cost is unavailable on one or more atomic intervals",
            }
        )

    if args.round_budget is not None:
        rb = solve_hndt_round_budget(model, args.round_budget)
        name = f"HNDT-RB (R={args.round_budget})"
        policies[name] = rb.action
        categories[name] = "proposal extension"

    paths = {name: evaluate_policy(model, policy) for name, policy in policies.items()}
    summaries = []
    for name, path_list in paths.items():
        row = summarize(name, path_list, h.optimum)
        row["category"] = categories[name]
        row["first_split"] = first_split(policies[name], model.n)
        summaries.append(row)
    write_csv(out / "summary.csv", summaries)
    write_fault_paths(out / "fault_costs.csv", paths)
    write_policy_json(out / "hndt_policy.json", h.action)

    # Individual literature rows remain available for audit/provenance, even if
    # the main summary collapses identical policy trees.
    literature_individual_rows = []
    for spec, policy in literature_executed:
        path_list = evaluate_policy(model, policy)
        row = summarize(spec.display_name, path_list, h.optimum)
        row["category"] = "paper-supported common-testbed adaptation"
        row["first_split"] = first_split(policy, model.n)
        literature_individual_rows.append(row)
    write_csv(out / "literature_individual_summary.csv", literature_individual_rows)

    group_rows = []
    for group in literature_groups:
        group_rows.append(
            {
                "group_id": group["group_id"],
                "main_summary_name": group["name"],
                "members": group["member_names"],
                "member_count": len(group["members"]),
                "identical_policy_on_current_trace": "yes" if len(group["members"]) > 1 else "n/a",
                "reason": (
                    "The current CellVG/HNDT state space is an ordered chain; midpoint-based "
                    "localization components therefore collapse to the same tree."
                ),
            }
        )
    write_csv(out / "literature_policy_groups.csv", group_rows)

    provenance = provenance_rows()
    status_by_name = {row["strategy"]: row for row in literature_status}
    group_by_member = {}
    for group in literature_groups:
        for spec in group["members"]:
            group_by_member[spec.display_name] = group["name"]
    for row in provenance:
        status = status_by_name[row["strategy"]]
        row["status"] = status["status"]
        row["status_reason"] = status["reason"]
        row["main_summary_group"] = group_by_member.get(row["strategy"], "")
    write_csv(out / "literature_baselines.csv", provenance)
    write_csv(out / "optional_policy_status.csv", optional_status)

    policy_dir = out / "policies"
    for name, policy in policies.items():
        write_policy_json(policy_dir / f"{_slug(name)}.json", policy)
    for spec, policy in literature_executed:
        write_policy_json(policy_dir / f"literature_{spec.key}.json", policy)

    write_csv(out / "hndt_policy_audit.csv", build_policy_audit(model, h))

    # 2x2 mechanism isolation table: adaptive split vs adaptive stop.
    mechanism_names = {
        "HNDT": (1, 1),
        "Ablation: adaptive split + atomic stop": (1, 0),
        "Ablation: midpoint split + adaptive stop": (0, 1),
    }
    if literature_groups:
        mechanism_names[literature_groups[0]["name"]] = (0, 0)
    mechanism_rows = []
    for name, (adaptive_split, adaptive_stop) in mechanism_names.items():
        if name not in paths:
            continue
        path_list = paths[name]
        mechanism_rows.append(
            {
                "strategy": name,
                "adaptive_split": adaptive_split,
                "adaptive_stop": adaptive_stop,
                "worst_case_cost": _worst(path_list),
                "ratio_to_hndt": _worst(path_list) / h.optimum,
                "max_rounds": max(p.rounds for p in path_list),
                "first_split": first_split(policies[name], model.n),
            }
        )
    write_csv(out / "mechanism_ablation.csv", mechanism_rows)

    # Exact cost-vs-round Pareto frontier. Infeasible small budgets are retained
    # explicitly rather than silently omitted.
    root_rounds = h.max_rounds[(0, model.n)]
    max_frontier_budget = max(root_rounds, args.round_budget or 0)
    frontier_rows = []
    for budget in range(max_frontier_budget + 1):
        try:
            rb = solve_hndt_round_budget(model, budget)
            frontier_rows.append(
                {
                    "round_budget": budget,
                    "status": "feasible",
                    "optimal_worst_case_cost": rb.optimum,
                    "ratio_to_unconstrained": rb.optimum / h.optimum,
                    "actual_max_rounds": rb.max_rounds,
                    "recovers_unconstrained_optimum": abs(rb.optimum - h.optimum) <= 1e-12,
                }
            )
        except ValueError as exc:
            frontier_rows.append(
                {
                    "round_budget": budget,
                    "status": "infeasible",
                    "optimal_worst_case_cost": "",
                    "ratio_to_unconstrained": "",
                    "actual_max_rounds": "",
                    "recovers_unconstrained_optimum": False,
                    "reason": str(exc),
                }
            )
    write_csv(out / "round_budget_frontier.csv", frontier_rows)

    # Heterogeneity ablation.
    hom = homogenized_model(model)
    hh = solve_hndt(hom)
    ablation = [
        {
            "cost_model": "heterogeneous",
            "optimal_worst_case_cost": h.optimum,
            "first_action": first_split(h.action, model.n),
            "max_rounds": h.max_rounds[(0, model.n)],
        },
        {
            "cost_model": "homogenized",
            "optimal_worst_case_cost": hh.optimum,
            "first_action": first_split(hh.action, model.n),
            "max_rounds": hh.max_rounds[(0, model.n)],
        },
    ]
    write_csv(out / "heterogeneity_ablation.csv", ablation)
    write_policy_json(out / "hndt_policy_homogenized.json", hh.action)

    metadata = {
        "data_status": marker,
        "trace": str(args.trace),
        "interval_costs": str(interval),
        "query_costs": str(query),
        "config": str(config),
        "n": model.n,
        "hndt_optimum": h.optimum,
        "hndt_max_rounds": root_rounds,
        "oracle_best_fixed_g": best_g,
        "oracle_best_fixed_g_cost": best_g_cost,
        "paper_supported_adaptations": [
            row["strategy"] for row in literature_status if row["status"] == "executed"
        ],
        "paper_supported_policy_groups": [group["name"] for group in literature_groups],
        "skipped_paper_adaptations": [
            {"strategy": row["strategy"], "reason": row["reason"]}
            for row in literature_status
            if row["status"] != "executed"
        ],
        "round_budget": args.round_budget,
        "method_note": (
            "HNDT remains the exact stop-or-split minimax solver. HNDT-RB is an exact "
            "round-constrained extension; mechanism ablations restrict either split or stop actions."
        ),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
