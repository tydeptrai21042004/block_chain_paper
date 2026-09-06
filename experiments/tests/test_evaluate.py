import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from hndt.core import Action, CostModel, solve_hndt
from hndt.evaluate import evaluate_policy, simulate_fault, summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.model = CostModel(
            4,
            {(i, i + 1): {"native": float(i + 1)} for i in range(4)},
            lambda i, j, k: 2.0,
        )

    def test_fault_branch_boundary_uses_left_for_fault_before_k(self):
        policy = {
            (0, 4): Action.split_at(2),
            (0, 2): Action.settle("native"),
            (2, 4): Action.settle("native"),
        }
        # Add direct interval costs used by this manually constructed policy.
        model = CostModel(
            4,
            {
                (0, 1): {"native": 1.0},
                (1, 2): {"native": 1.0},
                (2, 3): {"native": 1.0},
                (3, 4): {"native": 1.0},
                (0, 2): {"native": 5.0},
                (2, 4): {"native": 7.0},
            },
            lambda i, j, k: 2.0,
        )
        self.assertEqual(simulate_fault(model, policy, 1).terminal_i, 0)
        self.assertEqual(simulate_fault(model, policy, 2).terminal_i, 2)

    def test_hndt_sweep_worst_equals_bellman_optimum(self):
        result = solve_hndt(self.model)
        paths = evaluate_policy(self.model, result.action)
        self.assertAlmostEqual(max(p.total_cost for p in paths), result.optimum)
        self.assertEqual(max(p.rounds for p in paths), result.max_rounds[(0, 4)])

    def test_invalid_fault_index_rejected(self):
        policy = solve_hndt(self.model).action
        for bad in [-1, 4, 1.2, True]:
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError)):
                    simulate_fault(self.model, policy, bad)

    def test_missing_policy_interval_rejected(self):
        policy = {(0, 4): Action.split_at(2)}
        with self.assertRaises(ValueError):
            evaluate_policy(self.model, policy)

    def test_invalid_split_rejected(self):
        policy = {(0, 4): Action(kind="split", split=0)}
        with self.assertRaises(ValueError):
            simulate_fault(self.model, policy, 0)

    def test_summary_uses_unique_terminal_intervals(self):
        terminal = {
            (0, 1): {"native": 1.0},
            (1, 2): {"native": 1.0},
            (2, 3): {"native": 1.0},
            (3, 4): {"native": 1.0},
            (0, 2): {"native": 4.0},
            (2, 4): {"native": 4.0},
        }
        model = CostModel(4, terminal, lambda i, j, k: 1.0)
        policy = {
            (0, 4): Action.split_at(2),
            (0, 2): Action.settle("native"),
            (2, 4): Action.settle("native"),
        }
        row = summarize("x", evaluate_policy(model, policy), optimum=5.0)
        self.assertEqual(row["num_terminal_intervals"], 2)
        self.assertEqual(row["median_leaf_size"], 2)
        self.assertEqual(row["ratio_to_optimum"], 1.0)


if __name__ == "__main__":
    unittest.main()
