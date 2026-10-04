import math
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
CKB = ROOT / "ckb_bench"
for path in (str(EXP), str(CKB)):
    if path not in sys.path:
        sys.path.insert(0, path)

from hndt.core import CostModel, solve_hndt
from hndt.evaluate import evaluate_policy, summarize, weighted_mean_cost
from hndt.pareto import solve_pareto_hndt
from hndt.sensitivity import fault_weight_profile, perturb_cost_model, scale_query_costs
import generate_trace_vectors
import parse_trace_logs


class V5ExtensionTests(unittest.TestCase):
    def setUp(self):
        terminal = {
            (0, 1): {"native": 10.0},
            (1, 2): {"native": 2.0},
            (2, 3): {"native": 8.0},
            (3, 4): {"native": 3.0},
            (0, 2): {"native": 9.0},
            (2, 4): {"native": 8.5},
        }
        self.model = CostModel(4, terminal, lambda i, j, k: 1.5)

    def test_pareto_stats_do_not_change_solution(self):
        a = solve_pareto_hndt(self.model)
        b = solve_pareto_hndt(self.model, collect_stats=True)
        self.assertEqual(a.worst_optimum, b.worst_optimum)
        self.assertEqual(a.mean_at_worst_optimum, b.mean_at_worst_optimum)
        self.assertEqual(a.policy, b.policy)
        self.assertIsNotNone(b.stats)
        self.assertGreater(b.stats.candidates_generated, 0)
        self.assertGreaterEqual(b.stats.peak_frontier_size, 1)
        self.assertGreaterEqual(b.stats.pruning_ratio, 0.0)
        self.assertLessEqual(b.stats.pruning_ratio, 1.0)

    def test_query_scaling_is_exact(self):
        scaled = scale_query_costs(self.model, 2.5)
        self.assertEqual(scaled.checked_query_cost(0, 4, 2), 3.75)
        self.assertEqual(scaled.terminal_costs, self.model.terminal_costs)

    def test_fault_weight_profiles_normalize(self):
        for profile in ("uniform", "front", "back", "cost"):
            weights = fault_weight_profile(self.model, profile)
            self.assertEqual(len(weights), 4)
            self.assertAlmostEqual(sum(weights), 1.0)
            self.assertTrue(all(x >= 0 for x in weights))

    def test_weighted_reporting_matches_pareto_objective(self):
        weights = fault_weight_profile(self.model, "front")
        result = solve_pareto_hndt(self.model, fault_weights=weights)
        paths = evaluate_policy(self.model, result.policy)
        self.assertAlmostEqual(weighted_mean_cost(paths, weights), result.mean_at_worst_optimum)
        row = summarize("p", paths, result.worst_optimum, fault_weights=weights)
        self.assertAlmostEqual(row["mean_cost"], result.mean_at_worst_optimum)

    def test_perturbation_is_deterministic_and_minimax_safe(self):
        a = perturb_cost_model(self.model, relative_noise=0.05, seed=7)
        b = perturb_cost_model(self.model, relative_noise=0.05, seed=7)
        self.assertEqual(a.terminal_costs, b.terminal_costs)
        self.assertEqual(a.checked_query_cost(0, 4, 2), b.checked_query_cost(0, 4, 2))
        scalar = solve_hndt(a)
        pareto = solve_pareto_hndt(a)
        self.assertAlmostEqual(scalar.optimum, pareto.worst_optimum)


    def test_rust_trace_selection_is_compile_time_isolated(self):
        main_rs = (ROOT / "ckb_bench" / "src" / "main.rs").read_text(encoding="utf-8")
        build_rs = (ROOT / "ckb_bench" / "build.rs").read_text(encoding="utf-8")
        self.assertIn("BENCH_TRACE", build_rs)
        self.assertIn("trace_conv_heavy", build_rs)
        self.assertIn("trace_gemm_heavy", build_rs)
        self.assertIn("#[cfg(any(trace_conv_heavy, trace_gemm_heavy))]", main_rs)
        self.assertIn("#[cfg(not(any(trace_conv_heavy, trace_gemm_heavy)))]", main_rs)

    def test_extra_trace_bundles_have_42_measured_intervals(self):
        for bundle in (generate_trace_vectors.build_conv_heavy(), generate_trace_vectors.build_gemm_heavy()):
            self.assertEqual(len(bundle.states), 13)
            self.assertEqual(len(bundle.ops), 12)
            self.assertTrue(generate_trace_vectors.self_check_bundle(bundle))
            self.assertEqual(len(parse_trace_logs.expected_intervals(12, 4)), 42)


if __name__ == "__main__":
    unittest.main()
