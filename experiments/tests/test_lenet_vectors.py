import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "ckb_bench" / "generate_lenet_vectors.py"

spec = importlib.util.spec_from_file_location("generate_lenet_vectors", GEN)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class LenetVectorTests(unittest.TestCase):
    def test_checkpoint_dimensions(self):
        states = module.build_states()
        self.assertEqual([len(s) for s in states], module.STATE_LENS)
        self.assertEqual(len(states), 13)

    def test_all_measured_intervals_are_genuinely_state_chained(self):
        states = module.build_states()
        checked = 0
        for span in range(1, 5):
            for i in range(0, 12 - span + 1):
                j = i + span
                self.assertEqual(module.execute_interval(states, i, j), states[j])
                checked += 1
        self.assertEqual(checked, 42)

    def test_generated_rust_is_current(self):
        subprocess.run(
            [sys.executable, str(GEN), "--check"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
