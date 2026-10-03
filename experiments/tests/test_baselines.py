import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from hndt.baselines import (
    direct_native_policy,
    fixed_g_policy,
    has_complete_atomic_backend,
    midpoint_adaptive_stop_policy,
    midpoint_atomic_policy,
    midpoint_operator_zk_policy,
    optimal_split_atomic_policy,
)
from hndt.core import CostModel, solve_hndt
from hndt.evaluate import evaluate_policy
from hndt.literature_baselines import (
    agatha_gpp_chain_policy,
    arbitrum_ivp_policy,
    literature_baselines,
    opml_phase1_policy,
    policy_signature,
)


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

    def test_paper_grounded_adaptations_have_explicit_fidelity_guards(self):
        specs = literature_baselines()
        self.assertEqual(
            [s.key for s in specs],
            ["arbitrum_ivp", "opml_phase1", "agatha_gpp_chain"],
        )
        for spec in specs:
            self.assertTrue(spec.paper)
            self.assertTrue(spec.url)
            self.assertTrue(spec.fidelity)
            self.assertTrue(spec.comparable_component)
            self.assertTrue(spec.excluded_components)

    def test_literature_adaptations_collapse_honestly_on_ordered_chain(self):
        terminal = {(i, i + 1): {"native": float(i + 1)} for i in range(5)}
        model = CostModel(5, terminal, lambda i, j, k: 2.0)
        policies = [
            arbitrum_ivp_policy(model),
            opml_phase1_policy(model),
            agatha_gpp_chain_policy(model),
        ]
        signatures = [policy_signature(p) for p in policies]
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[1], signatures[2])
        for policy in policies:
            self.assertEqual(policy[(0, 5)].split, 2)
            paths = evaluate_policy(model, policy)
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.backend == "native" for path in paths))

    def test_literature_adaptations_require_native_atomic_arbitration(self):
        terminal = {(0, 1): {"native": 1.0}, (1, 2): {"zkvm": 2.0}}
        model = CostModel(2, terminal, lambda i, j, k: 1.0)
        for builder in [arbitrum_ivp_policy, opml_phase1_policy, agatha_gpp_chain_policy]:
            with self.subTest(builder=builder.__name__):
                with self.assertRaises(ValueError):
                    builder(model)

    def test_split_only_ablation_forces_atomic_leaves(self):
        terminal = {(i, i + 1): {"native": 5.0} for i in range(4)}
        terminal[(0, 2)] = {"native": 1.0}
        terminal[(2, 4)] = {"native": 1.0}
        terminal[(0, 4)] = {"native": 1.0}
        model = CostModel(4, terminal, lambda i, j, k: 0.5)
        policy = optimal_split_atomic_policy(model)
        paths = evaluate_policy(model, policy)
        self.assertTrue(all(p.terminal_j - p.terminal_i == 1 for p in paths))

    def test_midpoint_stop_ablation_can_stop_non_atomic(self):
        terminal = {(i, i + 1): {"native": 100.0} for i in range(4)}
        terminal[(0, 2)] = {"native": 2.0}
        terminal[(2, 4)] = {"native": 2.0}
        model = CostModel(4, terminal, lambda i, j, k: 1.0)
        policy = midpoint_adaptive_stop_policy(model)
        self.assertEqual(policy[(0, 4)].kind, "split")
        self.assertEqual(policy[(0, 2)].kind, "settle")
        self.assertEqual(policy[(2, 4)].kind, "settle")

    def test_hndt_never_worse_than_both_restricted_mechanism_ablations(self):
        terminal = {(i, i + 1): {"native": float(5 + i)} for i in range(5)}
        terminal.update({(0, 2): {"native": 8.0}, (2, 5): {"native": 13.0}})
        model = CostModel(5, terminal, lambda i, j, k: 2.0)
        optimum = solve_hndt(model).optimum
        for policy in [optimal_split_atomic_policy(model), midpoint_adaptive_stop_policy(model)]:
            worst = max(p.total_cost for p in evaluate_policy(model, policy))
            self.assertLessEqual(optimum, worst + 1e-12)

    def test_direct_native_requires_full_interval_measurement(self):
        model = CostModel(2, {(0, 1): {"native": 1.0}, (1, 2): {"native": 1.0}}, lambda i, j, k: 1.0)
        with self.assertRaises(ValueError):
            direct_native_policy(model)
        full = CostModel(
            2,
            {(0, 1): {"native": 1.0}, (1, 2): {"native": 1.0}, (0, 2): {"native": 4.0}},
            lambda i, j, k: 1.0,
        )
        self.assertEqual(direct_native_policy(full)[(0, 2)].backend, "native")


if __name__ == "__main__":
    unittest.main()
