import math
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from hndt.core import Action, CostModel, policy_reachable_intervals, solve_hndt


class CoreTests(unittest.TestCase):
    def atomic_model(self, costs, q=1.0):
        terminal = {(i, i + 1): {"native": float(c)} for i, c in enumerate(costs)}
        return CostModel(len(costs), terminal, lambda i, j, k: q)

    def test_rejects_nonpositive_n(self):
        with self.assertRaises(ValueError):
            CostModel(0, {}, lambda i, j, k: 1.0)

    def test_rejects_out_of_range_terminal_interval(self):
        with self.assertRaises(ValueError):
            CostModel(2, {(0, 3): {"native": 1.0}}, lambda i, j, k: 1.0)

    def test_rejects_negative_terminal_cost(self):
        with self.assertRaises(ValueError):
            CostModel(1, {(0, 1): {"native": -1.0}}, lambda i, j, k: 1.0)

    def test_native_wins_backend_tie(self):
        model = CostModel(
            1,
            {(0, 1): {"zkvm": 5.0, "native": 5.0, "other": 5.0}},
            lambda i, j, k: 1.0,
        )
        self.assertEqual(model.best_terminal(0, 1), (5.0, "native"))

    def test_lexical_tie_break_for_non_native_backends(self):
        model = CostModel(
            1,
            {(0, 1): {"zeta": 5.0, "alpha": 5.0}},
            lambda i, j, k: 1.0,
        )
        self.assertEqual(model.best_terminal(0, 1), (5.0, "alpha"))

    def test_homogeneous_eight_leaves_recovers_log_depth(self):
        n, a, r = 8, 10.0, 3.0
        result = solve_hndt(self.atomic_model([a] * n, q=r))
        self.assertAlmostEqual(result.optimum, a + r * math.ceil(math.log2(n)))
        self.assertEqual(result.max_rounds[(0, n)], 3)

    def test_settlement_is_preferred_on_value_tie(self):
        terminal = {
            (0, 1): {"native": 10.0},
            (1, 2): {"native": 10.0},
            (0, 2): {"native": 12.0},
        }
        model = CostModel(2, terminal, lambda i, j, k: 2.0)
        result = solve_hndt(model)
        self.assertEqual(result.optimum, 12.0)
        self.assertEqual(result.action[(0, 2)].kind, "settle")

    def test_nonmidpoint_root_can_be_optimal(self):
        model = self.atomic_model([100.0, 0.0, 0.0, 0.0], q=1.0)
        result = solve_hndt(model)
        self.assertEqual(result.action[(0, 4)], Action.split_at(1))
        self.assertEqual(result.optimum, 101.0)

    def test_direct_settlement_beats_forced_atomic(self):
        terminal = {
            (0, 1): {"native": 1.0},
            (1, 2): {"native": 1.0},
            (0, 2): {"native": 1.0},
        }
        result = solve_hndt(CostModel(2, terminal, lambda i, j, k: 100.0))
        self.assertEqual(result.action[(0, 2)].kind, "settle")
        self.assertEqual(result.optimum, 1.0)

    def test_adding_backend_cannot_worsen_optimum(self):
        terminal_a = {(i, i + 1): {"native": c} for i, c in enumerate([8, 7, 6, 5])}
        terminal_b = {k: dict(v) for k, v in terminal_a.items()}
        terminal_b[(0, 4)] = {"zkvm": 9.0}
        a = solve_hndt(CostModel(4, terminal_a, lambda i, j, k: 2.0)).optimum
        b = solve_hndt(CostModel(4, terminal_b, lambda i, j, k: 2.0)).optimum
        self.assertLessEqual(b, a)

    def test_increasing_all_costs_cannot_improve_optimum(self):
        low = self.atomic_model([3, 4, 5, 6], q=1.0)
        high = self.atomic_model([4, 5, 6, 7], q=2.0)
        self.assertGreaterEqual(solve_hndt(high).optimum, solve_hndt(low).optimum)

    def test_infinite_query_is_treated_as_unavailable(self):
        terminal = {
            (0, 1): {"native": 1.0},
            (1, 2): {"native": 1.0},
            (0, 2): {"native": 9.0},
        }
        model = CostModel(2, terminal, lambda i, j, k: math.inf)
        result = solve_hndt(model)
        self.assertEqual(result.action[(0, 2)].kind, "settle")
        self.assertEqual(result.optimum, 9.0)

    def test_negative_query_cost_is_rejected(self):
        model = self.atomic_model([1.0, 1.0], q=-1.0)
        with self.assertRaises(ValueError):
            solve_hndt(model)

    def test_unsolvable_atomic_interval_raises(self):
        model = CostModel(2, {(0, 1): {"native": 1.0}}, lambda i, j, k: 1.0)
        with self.assertRaises(ValueError):
            solve_hndt(model)

    def test_reachable_interval_validation_detects_missing_child(self):
        policy = {(0, 2): Action.split_at(1), (0, 1): Action.settle("native")}
        with self.assertRaises(ValueError):
            list(policy_reachable_intervals(policy, (0, 2)))

    def test_random_small_instances_match_exhaustive_strategy_enumeration(self):
        import random

        def exhaustive(model, i, j):
            values = []
            for cost in model.backends(i, j).values():
                c = float(cost)
                if math.isfinite(c):
                    values.append(c)
            for k in range(i + 1, j):
                q = model.checked_query_cost(i, j, k)
                if not math.isfinite(q):
                    continue
                left = exhaustive(model, i, k)
                right = exhaustive(model, k, j)
                for lv in left:
                    for rv in right:
                        values.append(q + max(lv, rv))
            return values

        rng = random.Random(20260906)
        for n in range(2, 6):
            for case in range(8):
                atomic = [float(rng.randint(1, 20)) for _ in range(n)]
                terminal = {(i, i + 1): {"native": atomic[i]} for i in range(n)}
                # Randomly expose a few non-atomic settlement actions.
                for span in range(2, n + 1):
                    for i in range(n - span + 1):
                        j = i + span
                        if rng.random() < 0.35:
                            terminal[(i, j)] = {"native": float(rng.randint(1, 35))}
                q_table = {
                    (i, j, k): float(rng.randint(0, 8))
                    for span in range(2, n + 1)
                    for i in range(n - span + 1)
                    for j in [i + span]
                    for k in range(i + 1, j)
                }
                model = CostModel(n, terminal, lambda i, j, k, qt=q_table: qt[(i, j, k)])
                dp = solve_hndt(model).optimum
                brute = min(exhaustive(model, 0, n))
                self.assertAlmostEqual(dp, brute, msg=f"n={n}, case={case}")


if __name__ == "__main__":
    unittest.main()
