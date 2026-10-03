import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "ckb_bench" / "generate_merkle_vectors.py"

spec = importlib.util.spec_from_file_location("cellvg_merkle_vectors", GEN)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class MerkleVectorTests(unittest.TestCase):
    def test_all_13_canonical_proofs_reconstruct_same_root(self):
        leaves = module.canonical_leaves()
        levels = module.build_levels(leaves)
        expected = levels[-1][0]
        self.assertEqual(len(leaves), 16)
        self.assertEqual(module.LEAF_COUNT, 13)
        self.assertEqual(module.DEPTH, 4)
        for index in range(module.LEAF_COUNT):
            proof = module.proof_for(levels, index)
            self.assertEqual(len(proof), 4)
            actual = module.verify_proof(leaves[index], index, proof)
            self.assertEqual(actual, expected)

    def test_leaf_index_controls_left_right_order(self):
        leaves = module.canonical_leaves()
        levels = module.build_levels(leaves)
        expected = levels[-1][0]
        proof = module.proof_for(levels, 1)
        self.assertEqual(module.verify_proof(leaves[1], 1, proof), expected)
        self.assertNotEqual(module.verify_proof(leaves[1], 0, proof), expected)

    def test_generated_rust_vectors_are_current(self):
        generated = module.render_rust()
        actual = (ROOT / "ckb_bench" / "src" / "merkle_vectors.rs").read_text(
            encoding="utf-8"
        )
        self.assertEqual(actual, generated)


    def test_rust_benchmark_uses_supplied_siblings_and_leaf_index_bits(self):
        source = (ROOT / "ckb_bench" / "src" / "main.rs").read_text(encoding="utf-8")
        self.assertNotIn("sibling_seed", source)
        self.assertIn("((leaf_index >> level) & 1)", source)
        self.assertIn("CANONICAL_ROOT", source)
        self.assertIn("bench_merkle_access", source)



if __name__ == "__main__":
    unittest.main()
