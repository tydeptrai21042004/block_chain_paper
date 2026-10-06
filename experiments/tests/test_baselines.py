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
    opml_single_phase_policy,
    opml_outer_phase_projection_policy,
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
                "opml_single_phase",
                "opml_outer_phase_projection",
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
            opml_outer_phase_projection_policy(model),
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

    def test_literature_adaptations_use_capabilities_not_backend_names(self):
        terminal = {(0, 1): {"replayA": 1.0}, (1, 2): {"replayA": 2.0}}
        model = CostModel(
            2, terminal, lambda i, j, k: 1.0,
            terminal_capabilities={"replayA": {"one-step", "replay"}},
        )
        for builder in [
            rdoc_binary_search_policy,
            truebit_verification_game_policy,
            arbitrum_ivp_policy,
            opml_outer_phase_projection_policy,
            agatha_gpp_chain_policy,
        ]:
            with self.subTest(builder=builder.__name__):
                policy = builder(model)
                self.assertTrue(
                    all(p.backend == "replayA" for p in evaluate_policy(model, policy))
                )

    def test_opml_single_phase_refuses_operator_trace(self):
        terminal = {(0, 1): {"step": 1.0}, (1, 2): {"step": 2.0}}
        model = CostModel(
            2, terminal, lambda i, j, k: 1.0,
            terminal_capabilities={"step": {"one-step"}},
            metadata={"trace_granularity": "operator"},
        )
        with self.assertRaises(ValueError):
            opml_single_phase_policy(model)

    def test_zk_baseline_uses_capability_not_literal_backend_name(self):
        terminal = {
            (0, 1): {"proofA": 3.0},
            (1, 2): {"proofA": 4.0},
        }
        model = CostModel(
            2, terminal, lambda i, j, k: 1.0,
            terminal_capabilities={"proofA": {"zk-proof"}},
        )
        paths = evaluate_policy(model, midpoint_operator_zk_policy(model))
        self.assertTrue(all(p.backend == "proofA" for p in paths))

    def test_opml_single_phase_runs_on_microinstruction_trace(self):
        terminal = {(i, i + 1): {"judge-step": 1.0} for i in range(4)}
        model = CostModel(
            4, terminal, lambda i, j, k: 1.0,
            terminal_capabilities={"judge-step": {"one-step"}},
            metadata={"trace_granularity": "vm-microinstruction"},
        )
        policy = opml_single_phase_policy(model)
        self.assertEqual(policy[(0, 4)].split, 2)
        self.assertTrue(all(p.backend == "judge-step" for p in evaluate_policy(model, policy)))

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

    def test_classical_objective_adapters_match_exhaustive_small_trees(self):
        # Independent exhaustive oracle over all ordered full binary trees.
        costs = [7.0, 1.0, 9.0, 2.0, 5.0]
        q = 2.0
        terminal = {(i, i + 1): {"step": costs[i]} for i in range(len(costs))}
        model = CostModel(
            len(costs), terminal, lambda i, j, k: q,
            terminal_capabilities={"step": {"one-step"}},
        )

        def all_depths(i, j):
            if j - i == 1:
                return [(0,)]
            out = []
            for k in range(i + 1, j):
                for left in all_depths(i, k):
                    for right in all_depths(k, j):
                        out.append(tuple(d + 1 for d in left + right))
            return out

        depths = all_depths(0, len(costs))
        exhaustive_worst = min(max(c + q * d for c, d in zip(costs, ds)) for ds in depths)
        exhaustive_mean = min(
            sum(c + q * d for c, d in zip(costs, ds)) / len(costs)
            for ds in depths
        )

        kk_paths = evaluate_policy(model, kirkpatrick_klawe_atomic_policy(model))
        ht_paths = evaluate_policy(model, hu_tucker_atomic_policy(model))
        self.assertAlmostEqual(max(p.total_cost for p in kk_paths), exhaustive_worst)
        self.assertAlmostEqual(
            sum(p.total_cost for p in ht_paths) / len(ht_paths), exhaustive_mean
        )

    def test_height_limited_objective_matches_exhaustive_small_trees(self):
        costs = [4.0, 9.0, 2.0, 8.0]
        q = 1.5
        terminal = {(i, i + 1): {"step": costs[i]} for i in range(len(costs))}
        model = CostModel(
            len(costs), terminal, lambda i, j, k: q,
            terminal_capabilities={"step": {"one-step"}},
        )

        def all_depths(i, j):
            if j - i == 1:
                return [(0,)]
            out = []
            for k in range(i + 1, j):
                for left in all_depths(i, k):
                    for right in all_depths(k, j):
                        out.append(tuple(d + 1 for d in left + right))
            return out

        feasible = [ds for ds in all_depths(0, 4) if max(ds) <= 2]
        exhaustive = min(
            sum(c + q * d for c, d in zip(costs, ds)) / 4
            for ds in feasible
        )
        policy = height_limited_mean_atomic_policy(model, max_rounds=2)
        paths = evaluate_policy(model, policy)
        self.assertAlmostEqual(sum(p.total_cost for p in paths) / 4, exhaustive)

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
