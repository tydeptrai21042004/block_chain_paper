"""Platform-neutral dispute-policy synthesis utilities.

``Pareto-DPS`` is the general finite-partition formulation.  The historical
``HNDT``/``CostModel`` names remain available as a binary specialization for
backwards compatibility and the CKB-VM case study.
"""

from .core import Action, CostModel, HNDTResult, solve_hndt
from .dps import (
    DPSAction,
    DPSLabel,
    ParetoDPSResult,
    QueryActionSpec,
    TerminalActionSpec,
    VerificationInstance,
    from_cost_model,
    solve_pareto_dps,
    solve_scalar_dps,
)
from .pareto import ParetoHNDTResult, ParetoLabel, solve_pareto_hndt

__all__ = [
    "Action",
    "CostModel",
    "HNDTResult",
    "ParetoLabel",
    "ParetoHNDTResult",
    "DPSAction",
    "DPSLabel",
    "ParetoDPSResult",
    "TerminalActionSpec",
    "QueryActionSpec",
    "VerificationInstance",
    "from_cost_model",
    "solve_hndt",
    "solve_pareto_hndt",
    "solve_pareto_dps",
    "solve_scalar_dps",
]
