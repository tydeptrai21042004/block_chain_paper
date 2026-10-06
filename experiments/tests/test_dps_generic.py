import csv
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from hndt.core import CostModel, solve_hndt
from hndt.dps import (
    QueryActionSpec,
    TerminalActionSpec,
    VerificationInstance,
    dps_to_legacy_policy,
    expected_cost_of_dps_policy,
    from_cost_model,
    reconstruct_dps_policy,
    solve_pareto_dps,
    solve_scalar_dps,
)
from hndt.generic_io import build_verification_instance
from hndt.pareto import solve_pareto_hndt


class GenericDPSTests(unittest.TestCase):
    def test_binary_adapter_matches_legacy_pareto_exactly(self):
        terminal = {
            (0, 1): {"replayA": 10},
            (1, 2): {"replayA": 3},
            (2, 3): {"replayA": 7},
            (0, 2): {"replayA": 9},
        }
        model = CostModel(
            3,
            terminal,
            lambda i, j, k: 2,
            terminal_capabilities={"replayA": {"replay", "one-step"}},
        )
        legacy = solve_pareto_hndt(model)
        generic = solve_pareto_dps(from_cost_model(model))
        self.assertEqual(generic.worst_optimum_exact, legacy.worst_optimum_exact)
        self.assertEqual(generic.expected_at_worst_exact, legacy.mean_at_worst_optimum_exact)
        self.assertEqual(dps_to_legacy_policy(generic.selected), legacy.policy)

    def test_finite_partition_query_can_be_ternary(self):
        n = 3
        terminals = {
            (0, 1): (TerminalActionSpec("replay", 5),),
            (1, 2): (TerminalActionSpec("replay", 5),),
            (2, 3): (TerminalActionSpec("replay", 5),),
        }

        def queries(i, j):
            if (i, j) == (0, 3):
                return (QueryActionSpec("ternary", (1, 2), 1),)
            return ()

        instance = VerificationInstance(n, terminals, queries)
        result = solve_pareto_dps(instance)
        self.assertEqual(result.worst_optimum_exact, Fraction(6))
        self.assertEqual(result.selected.action.cuts, (1, 2))
        policy = reconstruct_dps_policy(result.selected)
        self.assertEqual(expected_cost_of_dps_policy(instance, policy), Fraction(6))

    def test_positive_cost_scaling_preserves_policy(self):
        terminals = {
            (0, 1): (TerminalActionSpec("a", 2),),
            (1, 2): (TerminalActionSpec("a", 9),),
            (0, 2): (TerminalActionSpec("a", 12),),
        }
        base = VerificationInstance(
            2,
            terminals,
            lambda i, j: (QueryActionSpec("q", (1,), 1),) if (i, j) == (0, 2) else (),
        )
        scale = Fraction(7, 3)
        terminals_scaled = {
            k: tuple(TerminalActionSpec(x.action_id, x.cost * scale, x.capabilities) for x in v)
            for k, v in terminals.items()
        }
        scaled = VerificationInstance(
            2,
            terminals_scaled,
            lambda i, j: (QueryActionSpec("q", (1,), scale),) if (i, j) == (0, 2) else (),
        )
        a = solve_pareto_dps(base)
        b = solve_pareto_dps(scaled)
        self.assertEqual(reconstruct_dps_policy(a.selected), reconstruct_dps_policy(b.selected))
        self.assertEqual(b.worst_optimum_exact, scale * a.worst_optimum_exact)
        self.assertEqual(b.expected_at_worst_exact, scale * a.expected_at_worst_exact)

    def test_scalar_dps_preserves_minimax_but_can_have_larger_expectation(self):
        terminals = {
            (0, 1): (TerminalActionSpec("a", 10),),
            (1, 2): (TerminalActionSpec("a", 1), TerminalActionSpec("b", 9)),
        }

        def queries(i, j):
            return (QueryActionSpec("split", (1,), 0),) if (i, j) == (0, 2) else ()

        instance = VerificationInstance(2, terminals, queries)
        pareto = solve_pareto_dps(instance, fault_weights=[0, 1])
        scalar = solve_scalar_dps(instance, fault_weights=[0, 1])
        self.assertEqual(pareto.worst_optimum_exact, scalar.optimum_exact)
        p_mean = pareto.expected_at_worst_exact
        s_mean = expected_cost_of_dps_policy(instance, reconstruct_dps_policy(scalar.selected), [0, 1])
        self.assertLessEqual(p_mean, s_mean)

    def test_generic_csv_has_no_platform_specific_columns(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            trace = td / "trace.csv"
            terminals = td / "terminals.csv"
            queries = td / "queries.csv"
            metadata = td / "metadata.json"
            trace.write_text("index,name\n0,a\n1,b\n", encoding="utf-8")
            terminals.write_text(
                "i,j,action_id,cost,capabilities\n"
                "0,1,replayA,10,one-step;replay\n"
                "1,2,replayA,3,one-step;replay\n",
                encoding="utf-8",
            )
            queries.write_text(
                "i,j,action_id,cuts,cost\n0,2,binary,1,2\n",
                encoding="utf-8",
            )
            metadata.write_text(json.dumps({"cost_unit": "microseconds"}), encoding="utf-8")
            instance = build_verification_instance(trace, terminals, queries, metadata)
            result = solve_pareto_dps(instance)
            self.assertEqual(instance.metadata["cost_unit"], "microseconds")
            self.assertEqual(result.worst_optimum_exact, Fraction(12))

    def test_scalar_binary_solver_uses_exact_decimal_comparisons(self):
        model = CostModel(
            2,
            {
                (0, 1): {"a": 0.1},
                (1, 2): {"a": 0.2},
                (0, 2): {"a": 0.3000000000001},
            },
            lambda i, j, k: 0,
        )
        result = solve_hndt(model)
        self.assertEqual(result.optimum_exact, Fraction("0.2"))
        self.assertEqual(result.action[(0, 2)].kind, "split")


if __name__ == "__main__":
    unittest.main()
