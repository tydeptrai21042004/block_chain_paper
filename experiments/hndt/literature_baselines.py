from __future__ import annotations

"""Paper-grounded baseline adapters with explicit fidelity boundaries.

The adapters compare only components that can be represented faithfully on the
current ordered-trace cost model.  They use terminal *capabilities* rather than
platform-specific names.  Missing protocol levels or verifier measurements are
skipped instead of fabricated.
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
    claim_guard: str = "adaptation; not a full-system reproduction"


def _midpoint_to_atomic_capability(
    model: CostModel,
    *,
    label: str,
    required_capabilities=frozenset({"one-step"}),
) -> Policy:
    """Binary midpoint localization followed by a capability-matched atomic check."""

    policy: Policy = {}

    def build(i: int, j: int) -> None:
        if j - i == 1:
            act = _terminal_action(
                model,
                i,
                j,
                required_capabilities=required_capabilities,
            )
            if act is None:
                caps = ",".join(sorted(required_capabilities))
                raise ValueError(
                    f"{label} requires a finite atomic terminal action with "
                    f"capabilities {{{caps}}} on [{i},{j}]"
                )
            policy[(i, j)] = act
            return
        k = _midpoint(i, j)
        policy[(i, j)] = Action.split_at(k)
        build(i, k)
        build(k, j)

    build(0, model.n)
    return policy


def rdoc_binary_search_policy(model: CostModel) -> Policy:
    return _midpoint_to_atomic_capability(model, label="RDoC binary-search localization")


def truebit_verification_game_policy(model: CostModel) -> Policy:
    return _midpoint_to_atomic_capability(model, label="TrueBit verification game")


def arbitrum_ivp_policy(model: CostModel) -> Policy:
    return _midpoint_to_atomic_capability(model, label="Arbitrum interactive verification")


def opml_single_phase_policy(model: CostModel) -> Policy:
    """Faithful one-phase opML projection, only for VM-microinstruction traces.

    One-phase opML bisects a VM execution until one microinstruction remains.
    Applying that baseline directly to an operator-level trace would silently
    replace a microinstruction game with operator replay, so it is rejected.
    """

    granularity = str(model.metadata.get("trace_granularity", "")).strip().lower()
    if granularity not in {"vm-microinstruction", "microinstruction"}:
        raise ValueError(
            "faithful one-phase opML requires a VM-microinstruction trace; "
            f"current trace_granularity={granularity or 'unspecified'}"
        )
    return _midpoint_to_atomic_capability(model, label="opML single-phase")


def opml_outer_phase_projection_policy(model: CostModel) -> Policy:
    """Outer/high-level phase of multi-phase opML on an ordered operator trace.

    This is intentionally labelled a *projection*: the inner VM-microinstruction
    phase is not represented unless corresponding microtrace measurements exist.
    """

    return _midpoint_to_atomic_capability(
        model,
        label="opML outer-phase projection",
        required_capabilities=frozenset({"one-step"}),
    )


def agatha_gpp_chain_policy(model: CostModel) -> Policy:
    """Ordered-chain projection of Agatha's graph-based pinpoint protocol."""

    return _midpoint_to_atomic_capability(
        model,
        label="Agatha GPP chain projection",
        required_capabilities=frozenset({"one-step"}),
    )


def zkopml_operator_policy(model: CostModel) -> Policy:
    """Operator localization followed by a measured ZK-capable terminal action."""

    return midpoint_operator_zk_policy(model)


# Backward-compatible aliases used by older scripts.
opml_phase1_policy = opml_single_phase_policy
opml_bisection_policy = opml_outer_phase_projection_policy
agatha_gpp_policy = agatha_gpp_chain_policy


def literature_baselines() -> tuple[LiteratureBaselineSpec, ...]:
    return (
        LiteratureBaselineSpec(
            key="rdoc_binary_search",
            display_name="RDoC binary-search localization (ordered-trace projection)",
            paper="Canetti, Riva, and Rothblum, Refereed Delegation of Computation",
            year=2013,
            url="https://doi.org/10.1016/j.ic.2013.03.003",
            fidelity="structural projection",
            comparable_component="Binary search over committed computation configurations to isolate disagreement.",
            excluded_components="Refereed-delegation protocol, cryptographic game, and server economics.",
            builder=rdoc_binary_search_policy,
            baseline_class="binary-localization",
        ),
        LiteratureBaselineSpec(
            key="truebit_verification_game",
            display_name="TrueBit binary verification game (ordered-trace projection)",
            paper="Teutsch and Reitwiessner, A Scalable Verification Solution for Blockchains",
            year=2023,
            url="https://doi.org/10.1142/9789811278631_0015",
            fidelity="structural projection",
            comparable_component="Binary localization of an incorrect execution to one disputed transition.",
            excluded_components="Incentive layer, deposits, task market, and protocol economics.",
            builder=truebit_verification_game_policy,
            baseline_class="binary-localization",
        ),
        LiteratureBaselineSpec(
            key="arbitrum_ivp",
            display_name="Arbitrum binary challenge localization (ordered-trace projection)",
            paper="Kalodner et al., Arbitrum: Scalable, Private Smart Contracts",
            year=2018,
            url="https://www.usenix.org/conference/usenixsecurity18/presentation/kalodner",
            fidelity="structural projection",
            comparable_component="Binary challenge localization to one disputed transition.",
            excluded_components="VM economics, staking, assertion protocol, and chain-specific one-step proof plumbing.",
            builder=arbitrum_ivp_policy,
            baseline_class="binary-localization",
        ),
        LiteratureBaselineSpec(
            key="opml_single_phase",
            display_name="opML single-phase VM dispute (requires microinstruction trace)",
            paper="Conway et al., opML: Optimistic Machine Learning on Blockchain",
            year=2024,
            url="https://arxiv.org/abs/2401.17555",
            fidelity="faithful localization level when microinstruction trace is supplied",
            comparable_component="Bisection over VM microinstructions followed by one-step arbitration.",
            excluded_components="Chain economics and any microinstruction measurements not supplied to the artifact.",
            builder=opml_single_phase_policy,
            baseline_class="conditional-full-level",
            claim_guard="execute only on a VM-microinstruction trace; otherwise skip",
        ),
        LiteratureBaselineSpec(
            key="opml_outer_phase_projection",
            display_name="opML outer-phase operator projection (not full two-phase opML)",
            paper="Conway et al., opML: Optimistic Machine Learning on Blockchain",
            year=2024,
            url="https://arxiv.org/abs/2401.17555",
            fidelity="outer-phase structural projection",
            comparable_component="High-level/operator bisection of multi-phase opML.",
            excluded_components="Inner VM-microinstruction bisection/arbitration and chain economics.",
            builder=opml_outer_phase_projection_policy,
            baseline_class="protocol-projection",
            claim_guard="projection only; never label as full opML",
        ),
        LiteratureBaselineSpec(
            key="agatha_gpp_chain",
            display_name="Agatha GPP ordered-chain projection (not full DAG GPP)",
            paper="Zheng et al., Agatha: Smart Contract for DNN Computation",
            year=2021,
            url="https://arxiv.org/abs/2105.04919",
            fidelity="ordered-chain structural projection",
            comparable_component="Pinpoint localization when the computation graph is restricted to a chain.",
            excluded_components="General DAG GPP, XCE machinery, and Ethereum-specific arbitration.",
            builder=agatha_gpp_chain_policy,
            baseline_class="protocol-projection",
            claim_guard="projection only; never label as full Agatha reproduction",
        ),
        LiteratureBaselineSpec(
            key="kirkpatrick_klawe_minimax",
            display_name="Alphabetic minimax objective (Kirkpatrick-Klawe)",
            paper="Kirkpatrick and Klawe, Alphabetic Minimax Trees",
            year=1985,
            url="https://doi.org/10.1137/0214039",
            fidelity="exact objective reduction",
            comparable_component=(
                "Ordered binary fixed-leaf tree minimizing max_t(A_t + q d_t) under constant query cost."
            ),
            excluded_components="Historical construction algorithm; the artifact uses an independent exact interval DP.",
            builder=kirkpatrick_klawe_atomic_policy,
            baseline_class="classical-tree-theory",
            claim_guard="exact objective adaptation under mechanically checked preconditions",
        ),
        LiteratureBaselineSpec(
            key="hu_tucker_mean",
            display_name="Alphabetic weighted-path objective (Hu-Tucker)",
            paper="Hu and Tucker, Optimal Computer Search Trees and Variable-Length Alphabetical Codes",
            year=1971,
            url="https://doi.org/10.1137/0121057",
            fidelity="exact objective reduction",
            comparable_component="Ordered fixed-leaf tree minimizing weighted path length under constant query cost.",
            excluded_components="Historical Hu-Tucker construction algorithm; independent exact interval DP is used.",
            builder=hu_tucker_atomic_policy,
            baseline_class="classical-tree-theory",
            claim_guard="exact objective adaptation under mechanically checked preconditions",
        ),
        LiteratureBaselineSpec(
            key="zkopml_operator",
            display_name="zk-OPML operator localization + measured ZK terminal projection",
            paper="Kersic and Turkanovic, zk-OPML: Using zero-knowledge proofs to optimize OPML",
            year=2026,
            url="https://doi.org/10.1007/s44443-026-00573-1",
            fidelity="operator-localization plus measured ZK-terminal projection",
            comparable_component="Binary/operator localization followed by ZK verification of the isolated operator.",
            excluded_components="Any proving, finality, gas, or protocol cost not explicitly reproduced in the input cost oracle.",
            builder=zkopml_operator_policy,
            baseline_class="hybrid-optimistic-zk",
            claim_guard="skip unless every atomic operator has an explicit zk-proof-capable terminal cost",
        ),
    )


def policy_signature(policy: Policy) -> tuple:
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
                    "same ordered trace, same admissible terminal/query cost oracle, "
                    "same fault positions as Pareto-DPS"
                ),
                "claim_guard": spec.claim_guard,
            }
        )
    return rows
