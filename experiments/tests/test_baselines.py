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
    hu_tucker_atomic_policy,
    kirkpatrick_klawe_atomic_policy,
    optimal_mean_atomic_policy,
    pareto_atomic_policy,
    pareto_midpoint_adaptive_stop_policy,
    pareto_midpoint_atomic_policy,
    height_limited_mean_atomic_policy,
)
from hndt.core import CostModel, solve_hndt
from hndt.evaluate import evaluate_policy
from hndt.literature_baselines import (
    agatha_gpp_chain_policy,
    arbitrum_ivp_policy,
    literature_baselines,
    opml_phase1_policy,
    policy_signature,
    rdoc_binary_search_policy,
    truebit_verification_game_policy,
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
            [
                "rdoc_binary_search",
                "truebit_verification_game",
                "arbitrum_ivp",
                "opml_phase1",
                "agatha_gpp_chain",
                "kirkpatrick_klawe_minimax",
                "hu_tucker_mean",
                "zkopml_operator",
            ],
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
            rdoc_binary_search_policy(model),
            truebit_verification_game_policy(model),
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
        for builder in [
            rdoc_binary_search_policy,
            truebit_verification_game_policy,
            arbitrum_ivp_policy,
            opml_phase1_policy,
            agatha_gpp_chain_policy,
        ]:
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

    def test_kirkpatrick_klawe_adapter_matches_forced_atomic_minimax(self):
        terminal = {(i, i + 1): {"native": c} for i, c in enumerate([20.0, 1.0, 1.0, 1.0])}
        model = CostModel(4, terminal, lambda i, j, k: 2.0)
        self.assertEqual(
            policy_signature(kirkpatrick_klawe_atomic_policy(model)),
            policy_signature(optimal_split_atomic_policy(model)),
        )

    def test_classical_adapters_reject_nonconstant_query_cost(self):
        terminal = {(i, i + 1): {"native": 1.0} for i in range(3)}
        model = CostModel(
            3,
            terminal,
            lambda i, j, k: 1.0 if (i, j, k) == (0, 3, 1) else 2.0,
        )
        with self.assertRaises(ValueError):
            kirkpatrick_klawe_atomic_policy(model)
        with self.assertRaises(ValueError):
            hu_tucker_atomic_policy(model)

    def test_hu_tucker_adapter_minimizes_uniform_mean_over_atomic_trees(self):
        terminal = {(i, i + 1): {"native": float(i + 1)} for i in range(5)}
        model = CostModel(5, terminal, lambda i, j, k: 3.0)
        hu = hu_tucker_atomic_policy(model)
        general = optimal_mean_atomic_policy(model)
        hu_mean = sum(p.total_cost for p in evaluate_policy(model, hu)) / model.n
        general_mean = sum(p.total_cost for p in evaluate_policy(model, general)) / model.n
        self.assertAlmostEqual(hu_mean, general_mean)

    def test_zkopml_literature_adapter_is_explicitly_gated(self):
        specs = {s.key: s for s in literature_baselines()}
        zkopml = specs["zkopml_operator"]
        model = CostModel(
            2,
            {(0, 1): {"native": 1.0, "zkvm": 2.0}, (1, 2): {"native": 1.0}},
            lambda i, j, k: 1.0,
        )
        with self.assertRaises(ValueError):
            zkopml.builder(model)


    def test_pareto_mechanism_ablations_preserve_restrictions(self):
        terminal = {(i, i + 1): {"native": float(10 + i)} for i in range(4)}
        terminal.update({(0, 2): {"native": 3.0}, (2, 4): {"native": 4.0}})
        model = CostModel(4, terminal, lambda i, j, k: 1.0)

        atomic = pareto_atomic_policy(model)
        self.assertTrue(all(p.terminal_j - p.terminal_i == 1 for p in evaluate_policy(model, atomic)))

        midpoint_stop = pareto_midpoint_adaptive_stop_policy(model)
        self.assertEqual(midpoint_stop[(0, 4)].split, 2)

        midpoint_atomic = pareto_midpoint_atomic_policy(model)
        self.assertEqual(midpoint_atomic[(0, 4)].split, 2)
        self.assertTrue(
            all(p.terminal_j - p.terminal_i == 1 for p in evaluate_policy(model, midpoint_atomic))
        )

    def test_height_limited_alphabetic_policy_respects_depth(self):
        terminal = {(i, i + 1): {"native": float(i + 1)} for i in range(8)}
        model = CostModel(8, terminal, lambda i, j, k: 2.0)
        policy = height_limited_mean_atomic_policy(model, max_rounds=3)
        paths = evaluate_policy(model, policy)
        self.assertLessEqual(max(p.rounds for p in paths), 3)
        with self.assertRaises(ValueError):
            height_limited_mean_atomic_policy(model, max_rounds=2)

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
