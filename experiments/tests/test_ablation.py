import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from hndt.core import CostModel
from run_experiment import homogenized_model


class AblationTests(unittest.TestCase):
    def test_homogenization_preserves_interval_availability(self):
        terminal = {
            (0, 1): {"native": 2.0},
            (1, 2): {"native": 6.0},
            (0, 2): {"native": 10.0, "zkvm": 9.0},
        }
        model = CostModel(2, terminal, lambda i, j, k: 1.0)
        hom = homogenized_model(model)
        self.assertEqual(set(hom.terminal_costs), set(model.terminal_costs))
        self.assertEqual(hom.backends(0, 2)["zkvm"], 9.0)
        self.assertEqual(hom.backends(0, 1)["native"], 4.0)
        self.assertEqual(hom.backends(1, 2)["native"], 4.0)
        self.assertEqual(hom.backends(0, 2)["native"], 10.0)

    def test_homogenization_rejects_negative_reconstructed_cost(self):
        terminal = {
            (0, 1): {"native": 100.0},
            (1, 2): {"native": 1.0},
            (2, 3): {"native": 1.0},
            (0, 1): {"native": 100.0},
            (0, 2): {"native": 1.0},
        }
        # Atomic costs 100,1,1 => mean 34.  Interval [0,2] residual is -100,
        # so reconstructed homogeneous cost is -32 and must not be silently clamped.
        terminal[(2, 3)] = {"native": 1.0}
        model = CostModel(3, terminal, lambda i, j, k: 1.0)
        with self.assertRaises(ValueError):
            homogenized_model(model)


if __name__ == "__main__":
    unittest.main()
