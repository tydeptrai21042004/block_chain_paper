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
    midpoint_adaptive_stop_policy,
    optimal_split_atomic_policy,
)
from hndt.core import CostModel, solve_hndt, solve_hndt_round_budget
from hndt.evaluate import evaluate_policy, summarize, weighted_mean_cost
from hndt.io import build_model, validate_atomic_coverage
from hndt.literature_baselines import (
    literature_baselines,
    policy_signature,
    provenance_rows,
)
from hndt.pareto import (
    frontier_rows as pareto_frontier_rows,
    mean_first_policy,
    solve_pareto_hndt,
)
from hndt.reporting import first_split, write_csv, write_fault_paths, write_policy_json
from hndt.sensitivity import fault_weight_profile, scale_query_costs


PROPOSAL_NAME = "Pareto-HNDT (minimax-safe)"
SCALAR_HNDT_NAME = "Ablation: scalar HNDT (minimax only)"
MEAN_FIRST_NAME = "Ablation: Pareto mean-first endpoint"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the Pareto-HNDT/CellVG ordered-trace dispute experiment"
    )
    p.add_argument("--demo", action="store_true", help="Use the bundled LeNet synthetic costs for code sanity checks")
    p.add_argument("--synthetic-data", action="store_true", help="Mark explicitly supplied input files as synthetic/non-manuscript data")
    p.add_argument("--trace", default=str(ROOT / "data" / "templates" / "lenet5_trace.csv"))
    p.add_argument("--interval-costs", default=None)
    p.add_argument("--query-costs", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=str(ROOT / "results" / "current"))
    p.add_argument(
        "--query-scale",
        type=float,
        default=1.0,
        help="Multiply every measured query cost by this non-negative sensitivity factor.",
    )
    p.add_argument(
        "--fault-weights",
        choices=["uniform", "front", "back", "cost"],
        default="uniform",
        help="Transparent fault-position weighting used only for the mean objective/reporting.",
    )
    p.add_argument(
        "--collect-pareto-stats",
        action="store_true",
        help="Record candidate/pruning/frontier statistics without changing solver decisions.",
    )
    p.add_argument(
        "--fixed-g",
        default="2,4",
        help="Comma-separated fixed granularities to report in addition to the oracle best fixed-g ablation",
    )
    p.add_argument(
        "--round-budget",
        type=int,
        default=None,
        help=(
            "Optionally add one scalar-HNDT latency-constrained policy with at most this many "
            "split rounds. The Pareto proposal itself remains unconstrained."
        ),
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


def _mean(paths, fault_weights=None) -> float:
    return weighted_mean_cost(paths, fault_weights)


def choose_best_fixed_g(model: CostModel, fault_weights=None):
    """Oracle-tune one global terminal granularity on the evaluated cost table."""

    candidates = []
    for g in range(1, model.n + 1):
        try:
            policy = fixed_g_policy(model, g)
            paths = evaluate_policy(model, policy)
        except ValueError:
            continue
        candidates.append((_worst(paths), _mean(paths, fault_weights), g, policy, paths))
    if not candidates:
        raise ValueError("No feasible fixed-g policy")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0]


def build_literature_groups(model: CostModel, fault_weights=None):
    """Execute paper adapters and deduplicate only the close system family.

    Arbitrum/opML/Agatha can collapse to the same midpoint policy on an ordered
    trace, so they are grouped to avoid pseudo-replication. Classical objective
    baselines and zk-OPML remain individually visible even if a particular
    trace happens to yield the same tree.
    """

    executed = []
    status = []
    for spec in literature_baselines():
        try:
            policy = spec.builder(model, fault_weights=fault_weights) if spec.key == "hu_tucker_mean" else spec.builder(model)
            executed.append((spec, policy))
            status.append({"strategy": spec.display_name, "status": "executed", "reason": ""})
        except ValueError as exc:
            status.append({"strategy": spec.display_name, "status": "skipped", "reason": str(exc)})

    grouped = {}
    for spec, policy in executed:
        is_midpoint_system = spec.baseline_class in {"optimistic-system", "optimistic-ml"}
        domain = "midpoint-system-family" if is_midpoint_system else spec.key
        grouped.setdefault((domain, policy_signature(policy)), []).append((spec, policy))

    groups = []
    for index, members in enumerate(grouped.values(), start=1):
        specs = [spec for spec, _ in members]
        representative = members[0][1]
        if len(specs) > 1 and all(
            s.baseline_class in {"optimistic-system", "optimistic-ml"} for s in specs
        ):
            name = f"Prior-work midpoint/pinpoint family ({len(specs)} adapted policies)"
        elif len(specs) > 1:
            name = f"Equivalent literature policy family ({len(specs)} adaptations)"
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


def build_pareto_policy_audit(pareto) -> list[dict]:
    """Audit selected Pareto labels and their closest lexicographic alternatives."""

    selected_nodes = {}

    def visit(node):
        selected_nodes[node.interval] = node
        if node.action.kind == "split":
            visit(node.left)
            visit(node.right)

    visit(pareto.selected)
    rows = []
    for interval, chosen in sorted(selected_nodes.items()):
        alternatives = [label for label in pareto.frontier[interval] if label is not chosen]
        alt = min(alternatives, key=lambda x: (x.worst_cost, x.mean_cost, x.max_rounds)) if alternatives else None
        rows.append({
            "i": interval[0],
            "j": interval[1],
            "span": interval[1] - interval[0],
            "selected_action": chosen.action.kind,
            "selected_backend": chosen.action.backend or "",
            "selected_split": "" if chosen.action.split is None else chosen.action.split,
            "selected_worst": chosen.worst_cost,
            "selected_mean": chosen.mean_cost,
            "selected_max_rounds": chosen.max_rounds,
            "alternative_action": "" if alt is None else alt.action.kind,
            "alternative_backend": "" if alt is None else (alt.action.backend or ""),
            "alternative_split": "" if alt is None or alt.action.split is None else alt.action.split,
            "alternative_worst": "" if alt is None else alt.worst_cost,
            "alternative_mean": "" if alt is None else alt.mean_cost,
            "delta_worst": "" if alt is None else alt.worst_cost - chosen.worst_cost,
            "delta_mean": "" if alt is None else alt.mean_cost - chosen.mean_cost,
            "frontier_size": len(pareto.frontier[interval]),
        })
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
        marker = (
            "SYNTHETIC DEMO - NOT FOR MANUSCRIPT"
            if args.synthetic_data
            else "REAL/USER-SUPPLIED MEASUREMENTS"
        )

    try:
        model = build_model(args.trace, interval, query, config)
        model = scale_query_costs(model, args.query_scale)
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

    fault_weights = fault_weight_profile(model, args.fault_weights)

    # Original scalar minimax solver remains an independent regression oracle.
    h = solve_hndt(model)

    # Revised proposal: exact nondominated (worst, mean) frontier, followed by
    # minimax-preserving mean minimization. Uniform fault weights are used, so
    # no learned prior or tunable scalarization coefficient is introduced.
    pareto = solve_pareto_hndt(model, fault_weights=fault_weights, collect_stats=args.collect_pareto_stats)
    if abs(pareto.worst_optimum - h.optimum) > 1e-9:
        raise AssertionError(
            "Pareto-HNDT must recover the scalar HNDT minimax optimum; "
            f"got {pareto.worst_optimum} vs {h.optimum}"
        )

    policies = {
        PROPOSAL_NAME: pareto.policy,
        SCALAR_HNDT_NAME: h.action,
        MEAN_FIRST_NAME: mean_first_policy(pareto),
    }
    categories = {
        PROPOSAL_NAME: "proposal",
        SCALAR_HNDT_NAME: "proposal ablation",
        MEAN_FIRST_NAME: "proposal ablation",
    }

    literature_executed, literature_groups, literature_status = build_literature_groups(model, fault_weights=fault_weights)
    for group in literature_groups:
        policies[group["name"]] = group["policy"]
        categories[group["name"]] = "paper-supported baseline"

    # Mechanism-isolation ablations around the same action space.
    atomic_name = "Ablation: adaptive split + atomic stop"
    midpoint_stop_name = "Ablation: midpoint split + adaptive stop"
    policies[atomic_name] = optimal_split_atomic_policy(model)
    categories[atomic_name] = "mechanism ablation"
    policies[midpoint_stop_name] = midpoint_adaptive_stop_policy(model)
    categories[midpoint_stop_name] = "mechanism ablation"

    # Strongest globally fixed granularity under oracle access to this table.
    best_g_cost, best_g_mean, best_g, best_g_policy, _ = choose_best_fixed_g(model, fault_weights=fault_weights)
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
        name = "Ablation: direct full native"
        policies[name] = direct_native_policy(model)
        categories[name] = "extreme internal ablation"
        optional_status.append({"strategy": name, "status": "executed", "reason": ""})
    except ValueError as exc:
        optional_status.append(
            {"strategy": "Ablation: direct full native", "status": "skipped", "reason": str(exc)}
        )

    if args.round_budget is not None:
        rb = solve_hndt_round_budget(model, args.round_budget)
        name = f"Ablation: scalar HNDT round budget R={args.round_budget}"
        policies[name] = rb.action
        categories[name] = "latency ablation"

    paths = {name: evaluate_policy(model, policy) for name, policy in policies.items()}

    # Numerical invariant: Pareto selection cannot make the scalar minimax
    # objective worse, and it cannot have a larger mean than the old scalar
    # tie-selected HNDT policy at the same minimax value.
    proposal_worst = _worst(paths[PROPOSAL_NAME])
    scalar_worst = _worst(paths[SCALAR_HNDT_NAME])
    proposal_mean = _mean(paths[PROPOSAL_NAME], fault_weights)
    scalar_mean = _mean(paths[SCALAR_HNDT_NAME], fault_weights)
    if abs(proposal_worst - scalar_worst) > 1e-9:
        raise AssertionError("Pareto-HNDT changed the exact minimax optimum")
    if proposal_mean > scalar_mean + 1e-9:
        raise AssertionError("Pareto-HNDT must not worsen mean cost among minimax-optimal policies")
    if abs(proposal_mean - pareto.mean_at_worst_optimum) > 1e-8:
        raise AssertionError(
            "evaluated weighted mean must match Pareto objective: "
            f"{proposal_mean} vs {pareto.mean_at_worst_optimum}"
        )

    summaries = []
    for name, path_list in paths.items():
        row = summarize(name, path_list, pareto.worst_optimum, fault_weights=fault_weights)
        row["category"] = categories[name]
        row["first_split"] = first_split(policies[name], model.n)
        row["mean_delta_vs_proposal"] = row["mean_cost"] - proposal_mean
        row["worst_delta_vs_proposal"] = row["worst_case_cost"] - proposal_worst
        summaries.append(row)
    write_csv(out / "summary.csv", summaries)
    write_fault_paths(out / "fault_costs.csv", paths, fault_weights=fault_weights)
    write_policy_json(out / "pareto_hndt_policy.json", pareto.policy)
    write_policy_json(out / "hndt_scalar_policy.json", h.action)
    write_policy_json(out / "pareto_mean_first_policy.json", policies[MEAN_FIRST_NAME])
    write_csv(out / "pareto_frontier.csv", pareto_frontier_rows(pareto))

    # Individual literature rows remain available for audit/provenance even if
    # the close midpoint systems are deduplicated in the main summary.
    literature_individual_rows = []
    for spec, policy in literature_executed:
        path_list = evaluate_policy(model, policy)
        row = summarize(spec.display_name, path_list, pareto.worst_optimum, fault_weights=fault_weights)
        row["category"] = f"paper-supported {spec.baseline_class} adaptation"
        row["first_split"] = first_split(policy, model.n)
        row["policy_signature"] = repr(policy_signature(policy))
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
                    "Only close midpoint-system adapters are deduplicated. Classical objective "
                    "baselines remain individually visible even when one trace yields the same tree."
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

    # The old scalar audit remains useful as a sensitivity diagnostic.
    write_csv(out / "hndt_scalar_policy_audit.csv", build_policy_audit(model, h))
    write_csv(out / "pareto_policy_audit.csv", build_pareto_policy_audit(pareto))

    # Proposal-component ablation: split adaptivity, stopping adaptivity, and
    # minimax-safe secondary mean optimization are separated explicitly.
    mechanism_names = {
        PROPOSAL_NAME: (1, 1, 1),
        SCALAR_HNDT_NAME: (1, 1, 0),
        atomic_name: (1, 0, 0),
        midpoint_stop_name: (0, 1, 0),
    }
    midpoint_group = next(
        (
            group["name"]
            for group in literature_groups
            if any(s.key == "arbitrum_ivp" for s in group["members"])
        ),
        None,
    )
    if midpoint_group is not None:
        mechanism_names[midpoint_group] = (0, 0, 0)

    mechanism_rows = []
    for name, (adaptive_split, adaptive_stop, secondary_mean) in mechanism_names.items():
        if name not in paths:
            continue
        path_list = paths[name]
        mechanism_rows.append(
            {
                "strategy": name,
                "adaptive_split": adaptive_split,
                "adaptive_stop": adaptive_stop,
                "minimax_safe_mean_refinement": secondary_mean,
                "worst_case_cost": _worst(path_list),
                "mean_cost": _mean(path_list, fault_weights),
                "worst_delta_vs_proposal": _worst(path_list) - proposal_worst,
                "mean_delta_vs_proposal": _mean(path_list, fault_weights) - proposal_mean,
                "ratio_to_proposal_worst": _worst(path_list) / proposal_worst,
                "max_rounds": max(p.rounds for p in path_list),
                "first_split": first_split(policies[name], model.n),
            }
        )
    write_csv(out / "proposal_ablation.csv", mechanism_rows)
    # Backward-compatible filename for older manuscript scripts.
    write_csv(out / "mechanism_ablation.csv", mechanism_rows)

    # Scalar cost-vs-round frontier remains an exact latency diagnostic.
    root_rounds = h.max_rounds[(0, model.n)]
    max_frontier_budget = max(root_rounds, args.round_budget or 0)
    round_rows = []
    for budget in range(max_frontier_budget + 1):
        try:
            rb = solve_hndt_round_budget(model, budget)
            round_rows.append(
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
            round_rows.append(
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
    write_csv(out / "round_budget_frontier.csv", round_rows)

    # Heterogeneity ablation now reports both Pareto objectives.
    hom = homogenized_model(model)
    hh = solve_hndt(hom)
    ph = solve_pareto_hndt(hom, fault_weights=fault_weights)
    hom_paths = evaluate_policy(hom, ph.policy)
    ablation = [
        {
            "cost_model": "heterogeneous",
            "optimal_worst_case_cost": proposal_worst,
            "mean_cost_at_minimax": proposal_mean,
            "first_action": first_split(pareto.policy, model.n),
            "max_rounds": max(p.rounds for p in paths[PROPOSAL_NAME]),
            "root_frontier_size": len(pareto.root_frontier),
        },
        {
            "cost_model": "homogenized",
            "optimal_worst_case_cost": ph.worst_optimum,
            "mean_cost_at_minimax": _mean(hom_paths, fault_weights),
            "first_action": first_split(ph.policy, model.n),
            "max_rounds": max(p.rounds for p in hom_paths),
            "root_frontier_size": len(ph.root_frontier),
        },
    ]
    write_csv(out / "heterogeneity_ablation.csv", ablation)
    write_policy_json(out / "pareto_hndt_policy_homogenized.json", ph.policy)
    write_policy_json(out / "hndt_scalar_policy_homogenized.json", hh.action)

    gain_rows = [
        {
            "comparison": "Pareto-HNDT vs scalar HNDT",
            "pareto_worst": proposal_worst,
            "scalar_worst": scalar_worst,
            "worst_improvement": scalar_worst - proposal_worst,
            "pareto_mean": proposal_mean,
            "scalar_mean": scalar_mean,
            "mean_improvement": scalar_mean - proposal_mean,
            "mean_improvement_percent": (
                100.0 * (scalar_mean - proposal_mean) / scalar_mean if scalar_mean else 0.0
            ),
            "guarantee": (
                "worst costs must be equal; Pareto mean must be <= scalar-HNDT mean"
            ),
        }
    ]
    write_csv(out / "proposal_gain.csv", gain_rows)

    metadata = {
        "data_status": marker,
        "trace": str(args.trace),
        "interval_costs": str(interval),
        "query_costs": str(query),
        "config": str(config),
        "n": model.n,
        "query_scale": args.query_scale,
        "fault_weight_profile": args.fault_weights,
        "fault_weights": list(fault_weights),
        "proposal": PROPOSAL_NAME,
        "pareto_hndt_worst_optimum": pareto.worst_optimum,
        "pareto_hndt_mean_at_optimum": pareto.mean_at_worst_optimum,
        "pareto_root_frontier_size": len(pareto.root_frontier),
        "pareto_hndt_max_rounds": pareto.max_rounds,
        "scalar_hndt_optimum": h.optimum,
        "scalar_hndt_max_rounds": root_rounds,
        "oracle_best_fixed_g": best_g,
        "oracle_best_fixed_g_cost": best_g_cost,
        "oracle_best_fixed_g_mean": best_g_mean,
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
        "pareto_stats": (
            None
            if pareto.stats is None
            else {
                "intervals": pareto.stats.intervals,
                "candidates_generated": pareto.stats.candidates_generated,
                "labels_retained": pareto.stats.labels_retained,
                "duplicate_pruned": pareto.stats.duplicate_pruned,
                "dominated_pruned": pareto.stats.dominated_pruned,
                "peak_frontier_size": pareto.stats.peak_frontier_size,
                "peak_candidates_per_interval": pareto.stats.peak_candidates_per_interval,
                "pruning_ratio": pareto.stats.pruning_ratio,
            }
        ),
        "method_note": (
            "Pareto-HNDT computes the exact nondominated frontier of worst-case and weighted-mean "
            "fault-path cost, then preserves the exact scalar HNDT minimax optimum while minimizing "
            "mean cost among all minimax-optimal policies. No scalarization hyperparameter is used."
        ),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
