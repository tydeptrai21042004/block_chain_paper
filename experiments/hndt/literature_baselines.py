from __future__ import annotations

"""Paper-grounded comparison policies adapted to CellVG's common cost model.

Every adapter has an explicit fidelity boundary.  System papers contribute only
localization/terminal behavior that can be reproduced on the current ordered
trace; classical tree papers contribute only the optimization objective that is
mathematically identical under the adapter's stated assumptions.  Missing
full-system components are never assigned synthetic costs.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from .baselines import (
    _midpoint,
    _terminal_action,
    hu_tucker_atomic_policy,
    kirkpatrick_klawe_atomic_policy,
    midpoint_operator_zk_policy,
)
from .core import Action, CostModel

Interval = Tuple[int, int]
Policy = Dict[Interval, Action]


@dataclass(frozen=True)
class LiteratureBaselineSpec:
    key: str
    display_name: str
    paper: str
    year: int
    url: str
    fidelity: str
    comparable_component: str
    excluded_components: str
    builder: Callable[[CostModel], Policy]
    baseline_class: str = "system"


def _midpoint_to_native_atomic(model: CostModel, *, label: str) -> Policy:
    """Midpoint localization followed by a measured native atomic verifier."""

    policy: Policy = {}

    def build(i: int, j: int) -> None:
        if j - i == 1:
            act = _terminal_action(model, i, j, required_backend="native")
            if act is None:
                raise ValueError(
                    f"{label} requires a finite native CKB-VM cost on atomic "
                    f"interval [{i},{j}]"
                )
            policy[(i, j)] = act
            return

        k = _midpoint(i, j)
        policy[(i, j)] = Action.split_at(k)
        build(i, k)
        build(k, j)

    build(0, model.n)
    return policy


def arbitrum_ivp_policy(model: CostModel) -> Policy:
    return _midpoint_to_native_atomic(model, label="Arbitrum-IVP")


def opml_phase1_policy(model: CostModel) -> Policy:
    return _midpoint_to_native_atomic(model, label="opML Phase-1")


def agatha_gpp_chain_policy(model: CostModel) -> Policy:
    return _midpoint_to_native_atomic(model, label="Agatha-GPP chain projection")


def zkopml_operator_policy(model: CostModel) -> Policy:
    """zk-OPML comparable operator-level adaptation.

    The paper localizes by binary search to one ONNX operator and then resolves
    that operator with a ZK proof.  The adapter is executable only when a finite
    reproduced ``zkvm`` cost exists for every atomic operator; otherwise it is
    skipped rather than silently using native CKB verification.
    """

    return midpoint_operator_zk_policy(model)


# Backward-compatible aliases for earlier revised archives.
opml_bisection_policy = opml_phase1_policy
agatha_gpp_policy = agatha_gpp_chain_policy


def literature_baselines() -> tuple[LiteratureBaselineSpec, ...]:
    return (
        LiteratureBaselineSpec(
            key="arbitrum_ivp",
            display_name="Arbitrum-IVP (common-trace adaptation)",
            paper="Kalodner et al., Arbitrum: Scalable, Private Smart Contracts",
            year=2018,
            url="https://www.usenix.org/conference/usenixsecurity18/presentation/kalodner",
            fidelity="structural adaptation",
            comparable_component="Binary challenge localization to one disputed transition.",
            excluded_components="Arbitrum VM economics, staking, and full assertion protocol.",
            builder=arbitrum_ivp_policy,
            baseline_class="optimistic-system",
        ),
        LiteratureBaselineSpec(
            key="opml_phase1",
            display_name="opML Phase-1 (common-trace adaptation)",
            paper="Conway et al., opML: Optimistic Machine Learning on Blockchain",
            year=2024,
            url="https://arxiv.org/abs/2401.17555",
            fidelity="phase-1 adaptation",
            comparable_component="Operator-level bisection to one disputed DNN operator.",
            excluded_components=(
                "Second VM-microinstruction dispute phase and original-chain economics; "
                "not measured, therefore not fabricated."
            ),
            builder=opml_phase1_policy,
            baseline_class="optimistic-ml",
        ),
        LiteratureBaselineSpec(
            key="agatha_gpp_chain",
            display_name="Agatha-GPP chain projection (common-trace adaptation)",
            paper="Zheng et al., Agatha: Smart Contract for DNN Computation",
            year=2021,
            url="https://arxiv.org/abs/2105.04919",
            fidelity="chain projection of GPP",
            comparable_component="Graph-node pinpointing restricted to the ordered chain case.",
            excluded_components="General DAG GPP, XCE machinery, and Ethereum-specific arbitration.",
            builder=agatha_gpp_chain_policy,
            baseline_class="optimistic-ml",
        ),
        LiteratureBaselineSpec(
            key="kirkpatrick_klawe_minimax",
            display_name="Alphabetic Minimax Tree (Kirkpatrick-Klawe objective adaptation)",
            paper="Kirkpatrick and Klawe, Alphabetic Minimax Trees",
            year=1985,
            url="https://doi.org/10.1137/0214039",
            fidelity="exact objective reduction under forced atomic leaves and constant query cost",
            comparable_component=(
                "Ordered binary tree minimizing max_t(A_t + q d_t), equivalent after "
                "scaling to the paper's max_t(w_t + d_t) objective."
            ),
            excluded_components=(
                "Historical construction algorithm itself; this repository uses an independent "
                "exact interval DP for the mathematically identical objective."
            ),
            builder=kirkpatrick_klawe_atomic_policy,
            baseline_class="classical-tree-theory",
        ),
        LiteratureBaselineSpec(
            key="hu_tucker_mean",
            display_name="Optimal Alphabetic Mean Tree (Hu-Tucker objective adaptation)",
            paper="Hu and Tucker, Optimal Computer Search Trees and Variable-Length Alphabetical Codes",
            year=1971,
            url="https://doi.org/10.1137/0121057",
            fidelity="exact objective reduction under forced atomic leaves and constant query cost",
            comparable_component=(
                "Ordered fixed-leaf tree minimizing weighted path length; uniform fault weights "
                "are used unless an explicit distribution is supplied."
            ),
            excluded_components=(
                "Historical Hu-Tucker construction algorithm itself; this repository uses an "
                "independent exact interval DP for the same objective."
            ),
            builder=hu_tucker_atomic_policy,
            baseline_class="classical-tree-theory",
        ),
        LiteratureBaselineSpec(
            key="zkopml_operator",
            display_name="zk-OPML operator dispute (reproduced-ZK adaptation)",
            paper="Kersic and Turkanovic, zk-OPML: Using zero-knowledge proofs to optimize OPML",
            year=2026,
            url="https://doi.org/10.1007/s44443-026-00573-1",
            fidelity="operator-localization plus reproduced atomic-ZK adaptation",
            comparable_component=(
                "Binary search over the operator sequence followed by ZK verification of the "
                "isolated operator."
            ),
            excluded_components=(
                "Original smart-contract/finality economics and any ZK cost not independently "
                "reproduced in this artifact."
            ),
            builder=zkopml_operator_policy,
            baseline_class="hybrid-optimistic-zk",
        ),
    )


def policy_signature(policy: Policy) -> tuple:
    """Canonical signature used to detect numerically duplicate policy trees."""

    return tuple(
        (i, j, action.kind, action.backend or "", -1 if action.split is None else action.split)
        for (i, j), action in sorted(policy.items())
    )


def provenance_rows() -> list[dict]:
    rows = []
    for spec in literature_baselines():
        rows.append(
            {
                "strategy": spec.display_name,
                "key": spec.key,
                "baseline_class": spec.baseline_class,
                "paper": spec.paper,
                "year": spec.year,
                "url": spec.url,
                "fidelity": spec.fidelity,
                "comparable_component": spec.comparable_component,
                "excluded_components": spec.excluded_components,
                "common_environment": (
                    "same ordered trace, same measured terminal costs when required, same corrected "
                    "Merkle query cost, same fault positions as Pareto-HNDT"
                ),
                "claim_guard": "common-testbed/objective adaptation; not a full-system reproduction",
            }
        )
    return rows
