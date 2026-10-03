from __future__ import annotations

"""Paper-grounded localization policies adapted to CellVG's common cost model.

The code below is deliberately conservative about reproduction claims.  It
implements only the comparable dispute-localization component supported by the
cited papers, while keeping *all* terminal CKB costs, Merkle-query costs, trace
positions, and fault positions common with HNDT.  Components not measured in
this repository (for example opML's lower-level VM phase or Agatha's full DAG
and XCE machinery) are explicitly excluded rather than assigned invented costs.

Because the present HNDT model is an ordered trace, these three adaptations can
collapse to the same midpoint tree.  The experiment reports that equivalence
instead of manufacturing artificial differences.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from .baselines import _midpoint, _terminal_action
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
    """Arbitrum IVP common-trace adaptation.

    Reproduced component: recursive bisection of a disputed assertion until one
    atomic transition remains.  CellVG's measured native CKB-VM verifier plays
    the role of the one-step adjudicator in the common testbed.
    """

    return _midpoint_to_native_atomic(model, label="Arbitrum-IVP")


def opml_phase1_policy(model: CostModel) -> Policy:
    """opML Phase-1/operator-localization adaptation.

    The repository measures operator-level CKB verification, not opML's second
    VM-microinstruction trace.  The implementation therefore stops after the
    paper's operator-localization phase and uses the same measured native CKB
    terminal verifier as all other common-testbed methods.  No Phase-2 cost is
    fabricated.
    """

    return _midpoint_to_native_atomic(model, label="opML Phase-1")


def agatha_gpp_chain_policy(model: CostModel) -> Policy:
    """Agatha GPP projected onto CellVG's ordered chain.

    Agatha's full GPP is graph-based.  HNDT's current state space is an ordered
    trace, so the only faithful comparable projection is the chain case, where
    graph pinpointing reduces to ordered bisection.  Full DAG scheduling, XCE,
    and Ethereum-specific arbitration are intentionally outside this adapter.
    """

    return _midpoint_to_native_atomic(model, label="Agatha-GPP chain projection")


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
                "paper": spec.paper,
                "year": spec.year,
                "url": spec.url,
                "fidelity": spec.fidelity,
                "comparable_component": spec.comparable_component,
                "excluded_components": spec.excluded_components,
                "common_environment": (
                    "same ordered trace, same native CKB terminal costs, same corrected "
                    "Merkle query cost, same fault positions as HNDT"
                ),
                "claim_guard": "common-testbed adaptation; not a full-system reproduction",
            }
        )
    return rows
