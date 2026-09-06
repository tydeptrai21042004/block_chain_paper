import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from hndt.baselines import (
    fixed_g_policy,
    has_complete_atomic_backend,
    midpoint_atomic_policy,
    midpoint_operator_zk_policy,
)
from hndt.core import CostModel
from hndt.evaluate import evaluate_policy


class BaselineTests(unittest.TestCase):
    def test_midpoint_atomic_root_for_odd_n(self):
        terminal = {(i, i + 1): {"native": 1.0} for i in range(5)}
        model = CostModel(5, terminal, lambda i, j, k: 1.0)
        policy = midpoint_atomic_policy(model)
        self.assertEqual(policy[(0, 5)].split, 2)
        self.assertEqual(len(evaluate_policy(model, policy)), 5)

    def test_fixed_g_stops_on_available_span(self):
        terminal = {(i, i + 1): {"native": 10.0} for i in range(4)}
        terminal[(0, 2)] = {"native": 7.0}
        terminal[(2, 4)] = {"native": 8.0}
        model = CostModel(4, terminal, lambda i, j, k: 1.0)
        policy = fixed_g_policy(model, 2)
        self.assertEqual(policy[(0, 2)].kind, "settle")
        self.assertEqual(policy[(2, 4)].kind, "settle")

    def test_fixed_g_continues_when_span_cost_missing(self):
        terminal = {(i, i + 1): {"native": 2.0} for i in range(4)}
        model = CostModel(4, terminal, lambda i, j, k: 1.0)
        policy = fixed_g_policy(model, 4)
        self.assertEqual(policy[(0, 4)].kind, "split")
        self.assertTrue(all(p.terminal_j - p.terminal_i == 1 for p in evaluate_policy(model, policy)))

    def test_fixed_g_rejects_invalid_g(self):
        model = CostModel(1, {(0, 1): {"native": 1.0}}, lambda i, j, k: 1.0)
        for bad in [0, -1, 1.5, True]:
            with self.subTest(g=bad):
                with self.assertRaises(ValueError):
                    fixed_g_policy(model, bad)

    def test_zk_baseline_requires_complete_atomic_zk(self):
        terminal = {
            (0, 1): {"native": 1.0, "zkvm": 2.0},
            (1, 2): {"native": 1.0},
        }
        model = CostModel(2, terminal, lambda i, j, k: 1.0)
        self.assertFalse(has_complete_atomic_backend(model, "zkvm"))
        with self.assertRaises(ValueError):
            midpoint_operator_zk_policy(model)

    def test_zk_baseline_uses_zk_on_every_atomic_leaf(self):
        terminal = {
            (0, 1): {"native": 1.0, "zkvm": 3.0},
            (1, 2): {"native": 1.0, "zkvm": 3.0},
        }
        model = CostModel(2, terminal, lambda i, j, k: 1.0)
        self.assertTrue(has_complete_atomic_backend(model, "zkvm"))
        paths = evaluate_policy(model, midpoint_operator_zk_policy(model))
        self.assertTrue(all(p.backend == "zkvm" for p in paths))


if __name__ == "__main__":
    unittest.main()
