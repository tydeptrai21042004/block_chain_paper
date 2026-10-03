import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from hndt.core import CostModel, solve_hndt, solve_hndt_round_budget
from hndt.evaluate import evaluate_policy


class RoundBudgetTests(unittest.TestCase):
    def atomic_model(self, n=4, q=1.0):
        terminal = {(i, i + 1): {"native": float(i + 2)} for i in range(n)}
        return CostModel(n, terminal, lambda i, j, k: q)

    def test_rejects_invalid_budget(self):
        model = self.atomic_model()
        for bad in [-1, 1.5, True]:
            with self.subTest(budget=bad):
                with self.assertRaises(ValueError):
                    solve_hndt_round_budget(model, bad)

    def test_zero_budget_requires_root_settlement(self):
        model = self.atomic_model()
        with self.assertRaises(ValueError):
            solve_hndt_round_budget(model, 0)

        terminal = dict(model.terminal_costs)
        terminal[(0, 4)] = {"native": 20.0}
        result = solve_hndt_round_budget(
            CostModel(4, terminal, model.query_cost), 0
        )
        self.assertEqual(result.optimum, 20.0)
        self.assertEqual(result.action[(0, 4)].kind, "settle")
        self.assertEqual(result.max_rounds, 0)

    def test_more_round_budget_cannot_worsen_cost(self):
        terminal = {(i, i + 1): {"native": 10.0} for i in range(4)}
        terminal[(0, 2)] = {"native": 17.0}
        terminal[(2, 4)] = {"native": 16.0}
        terminal[(0, 4)] = {"native": 40.0}
        model = CostModel(4, terminal, lambda i, j, k: 2.0)
        costs = [
            solve_hndt_round_budget(model, r).optimum
            for r in range(0, 4)
        ]
        self.assertTrue(all(b <= a for a, b in zip(costs, costs[1:])))

    def test_sufficient_budget_matches_unconstrained_hndt(self):
        model = self.atomic_model(n=5, q=3.0)
        unconstrained = solve_hndt(model)
        budgeted = solve_hndt_round_budget(model, 4)
        self.assertAlmostEqual(budgeted.optimum, unconstrained.optimum)
        self.assertEqual(
            max(p.total_cost for p in evaluate_policy(model, budgeted.action)),
            budgeted.optimum,
        )

    def test_budget_is_respected_on_every_fault_path(self):
        terminal = {(i, i + 1): {"native": 5.0} for i in range(8)}
        terminal[(0, 4)] = {"native": 15.0}
        terminal[(4, 8)] = {"native": 15.0}
        model = CostModel(8, terminal, lambda i, j, k: 1.0)
        result = solve_hndt_round_budget(model, 1)
        paths = evaluate_policy(model, result.action)
        self.assertLessEqual(max(p.rounds for p in paths), 1)
        self.assertLessEqual(result.max_rounds, 1)


if __name__ == "__main__":
    unittest.main()
