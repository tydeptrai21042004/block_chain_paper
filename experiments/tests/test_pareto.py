import math
from fractions import Fraction
import random
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from hndt.core import Action, CostModel, solve_hndt
from hndt.evaluate import evaluate_policy
from hndt.pareto import (
    prune_nondominated,
    reconstruct_policy,
    select_frontier_label,
    solve_pareto_hndt,
)


class ParetoHNDTTests(unittest.TestCase):
    def test_minimax_projection_matches_scalar_hndt(self):
        terminal = {
            (0, 1): {"native": 10.0},
            (1, 2): {"native": 2.0},
            (2, 3): {"native": 9.0},
            (3, 4): {"native": 1.0},
            (0, 2): {"native": 9.0},
            (2, 4): {"native": 8.0},
        }
        model = CostModel(4, terminal, lambda i, j, k: 1.5)
        scalar = solve_hndt(model)
        pareto = solve_pareto_hndt(model)
        self.assertAlmostEqual(pareto.worst_optimum, scalar.optimum)

    def test_selected_policy_never_worsens_scalar_mean_at_same_worst(self):
        rng = random.Random(20261004)
        for n in range(2, 6):
            for _ in range(10):
                terminal = {(i, i + 1): {"native": float(rng.randint(1, 20))} for i in range(n)}
                for span in range(2, n + 1):
                    for i in range(n - span + 1):
                        j = i + span
                        if rng.random() < 0.45:
                            terminal[(i, j)] = {"native": float(rng.randint(2, 30))}
                model = CostModel(n, terminal, lambda i, j, k: 2.0)
                scalar = solve_hndt(model)
                pareto = solve_pareto_hndt(model)
                scalar_paths = evaluate_policy(model, scalar.action)
                pareto_paths = evaluate_policy(model, pareto.policy)
                scalar_worst = max(p.total_cost for p in scalar_paths)
                pareto_worst = max(p.total_cost for p in pareto_paths)
                scalar_mean = sum(p.total_cost for p in scalar_paths) / n
                pareto_mean = sum(p.total_cost for p in pareto_paths) / n
                self.assertAlmostEqual(pareto_worst, scalar_worst)
                self.assertLessEqual(pareto_mean, scalar_mean + 1e-10)

    def test_root_frontier_is_nondominated(self):
        terminal = {
            (0, 1): {"native": 1.0},
            (1, 2): {"native": 9.0},
            (2, 3): {"native": 1.0},
            (0, 2): {"native": 7.0},
            (1, 3): {"native": 7.0},
        }
        model = CostModel(3, terminal, lambda i, j, k: 1.0)
        result = solve_pareto_hndt(model)
        labels = result.root_frontier
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i == j:
                    continue
                dominates = (
                    a.worst_cost <= b.worst_cost
                    and a.mean_cost <= b.mean_cost
                    and (a.worst_cost < b.worst_cost or a.mean_cost < b.mean_cost)
                )
                self.assertFalse(dominates)

    def test_frontier_matches_exhaustive_small_strategy_pairs(self):
        def all_pairs(model, i, j, weights):
            mass = sum(weights[i:j])
            pairs = []
            for raw in model.backends(i, j).values():
                c = float(raw)
                if math.isfinite(c):
                    pairs.append((c, mass * c))
            for k in range(i + 1, j):
                q = model.checked_query_cost(i, j, k)
                if not math.isfinite(q):
                    continue
                for lw, lm in all_pairs(model, i, k, weights):
                    for rw, rm in all_pairs(model, k, j, weights):
                        pairs.append((q + max(lw, rw), mass * q + lm + rm))
            return pairs

        def brute_frontier(pairs):
            out = set()
            for p in pairs:
                dominated = any(
                    q[0] <= p[0]
                    and q[1] <= p[1]
                    and (q[0] < p[0] or q[1] < p[1])
                    for q in pairs
                )
                if not dominated:
                    out.add((round(p[0], 10), round(p[1], 10)))
            return out

        rng = random.Random(77)
        for n in range(2, 5):
            for _ in range(5):
                terminal = {(i, i + 1): {"native": float(rng.randint(1, 9))} for i in range(n)}
                for span in range(2, n + 1):
                    for i in range(n - span + 1):
                        j = i + span
                        if rng.random() < 0.5:
                            terminal[(i, j)] = {"native": float(rng.randint(2, 15))}
                q_table = {
                    (i, i + span, k): float(rng.randint(0, 4))
                    for span in range(2, n + 1)
                    for i in range(n - span + 1)
                    for k in range(i + 1, i + span)
                }
                model = CostModel(n, terminal, lambda i, j, k, qt=q_table: qt[(i, j, k)])
                result = solve_pareto_hndt(model)
                weights = [1.0 / n] * n
                expected = brute_frontier(all_pairs(model, 0, n, weights))
                actual = {
                    (round(x.worst_cost, 10), round(x.mean_cost, 10))
                    for x in result.root_frontier
                }
                self.assertEqual(actual, expected)


    def test_exact_decimal_arithmetic_has_no_tolerance_collapse(self):
        terminal = {
            (0, 1): {"native": 0.1},
            (1, 2): {"native": 0.2},
            (0, 2): {"native": 0.3000000000001},
        }
        model = CostModel(2, terminal, lambda i, j, k: 0.0)
        result = solve_pareto_hndt(model)
        self.assertTrue(all(isinstance(x.worst_exact, Fraction) for x in result.root_frontier))
        self.assertEqual(sum(result.exact_weights, Fraction(0, 1)), Fraction(1, 1))

    def test_local_lexicographic_summary_is_not_compositionally_sufficient(self):
        # Root split at k=1 has a left bottleneck W=10.  The right interval has
        # two nondominated labels: local minimax (1,100) and (9,0).  Retaining
        # only the local minimax label produces parent (10,100), while retaining
        # the full frontier exposes (10,0).  This is the ancestor-slack reason
        # that Pareto-HNDT must keep nondominated subpolicies.
        from hndt.pareto import ParetoLabel, prune_nondominated
        left = ParetoLabel((0, 1), Fraction(10), Fraction(0), 0, Action.settle("native"))
        right_a = ParetoLabel((1, 2), Fraction(1), Fraction(100), 0, Action.settle("a"))
        right_b = ParetoLabel((1, 2), Fraction(9), Fraction(0), 0, Action.settle("b"))
        right_frontier = prune_nondominated([right_a, right_b])
        self.assertEqual(len(right_frontier), 2)
        local_minimax = min(right_frontier, key=lambda x: (x.worst_exact, x.mean_exact))
        parent_local = (max(left.worst_exact, local_minimax.worst_exact), left.mean_exact + local_minimax.mean_exact)
        parent_full = min(
            (max(left.worst_exact, y.worst_exact), left.mean_exact + y.mean_exact)
            for y in right_frontier
        )
        self.assertEqual(parent_local, (Fraction(10), Fraction(100)))
        self.assertEqual(parent_full, (Fraction(10), Fraction(0)))

    def test_sorted_pruning_matches_quadratic_definition_randomly(self):
        from hndt.pareto import ParetoLabel
        rng = random.Random(20261006)
        for _ in range(50):
            labels = []
            for i in range(40):
                labels.append(
                    ParetoLabel(
                        (0, 1),
                        Fraction(rng.randint(0, 20)),
                        Fraction(rng.randint(0, 20)),
                        rng.randint(0, 3),
                        Action.settle(f"b{i}"),
                    )
                )
            actual = {(x.worst_exact, x.mean_exact) for x in prune_nondominated(labels)}
            expected = set()
            for x in labels:
                dominated = any(
                    y.worst_exact <= x.worst_exact
                    and y.mean_exact <= x.mean_exact
                    and (y.worst_exact < x.worst_exact or y.mean_exact < x.mean_exact)
                    for y in labels
                )
                if not dominated:
                    expected.add((x.worst_exact, x.mean_exact))
            self.assertEqual(actual, expected)

    def test_mean_first_endpoint_can_trade_worst_for_mean(self):
        terminal = {
            (0, 1): {"native": 100.0},
            (1, 2): {"native": 1.0},
            (2, 3): {"native": 1.0},
            (0, 3): {"native": 60.0},
        }
        model = CostModel(3, terminal, lambda i, j, k: 1.0)
        result = solve_pareto_hndt(model)
        minimax = result.selected
        mean_first = select_frontier_label(result.root_frontier, mode="mean_first")
        self.assertLessEqual(mean_first.mean_cost, minimax.mean_cost + 1e-12)
        self.assertGreaterEqual(mean_first.worst_cost + 1e-12, minimax.worst_cost)
        self.assertTrue(reconstruct_policy(mean_first))


if __name__ == "__main__":
    unittest.main()
