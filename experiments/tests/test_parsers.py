import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKB = ROOT / "ckb_bench"
if str(CKB) not in sys.path:
    sys.path.insert(0, str(CKB))

import parse_lenet_logs
import parse_logs


class ParserTests(unittest.TestCase):
    def test_cycle_patterns(self):
        examples = [
            "Total cycles consumed: 12345",
            "All cycles: 54321",
            "some total cycles = 777 end",
        ]
        expected = [12345, 54321, 777]
        for text, value in zip(examples, expected):
            with self.subTest(text=text):
                self.assertEqual(parse_logs.parse_cycles(text), value)
                self.assertEqual(parse_lenet_logs.parse_cycles(text), value)

    def test_cycle_parser_returns_none_when_absent(self):
        self.assertIsNone(parse_logs.parse_cycles("no cycle line"))

    def test_primitive_parser_strictly_rejects_missing_cycle_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "relu_16.log"
            p.write_text("nothing useful", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_logs.parse_log_directory(Path(tmp), strict=True)

    def test_lenet_expected_interval_count_is_42(self):
        self.assertEqual(len(parse_lenet_logs.expected_intervals(12, 4)), 42)

    def test_lenet_parser_detects_missing_expected_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "lenet_0_1.log"
            p.write_text("Total cycles consumed: 100", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_lenet_logs.parse_lenet_directory(Path(tmp), n=12, max_span=4, strict=True)

    def test_lenet_parser_accepts_complete_small_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, j in sorted(parse_lenet_logs.expected_intervals(3, 2)):
                (root / f"lenet_{i}_{j}.log").write_text(
                    f"Total cycles consumed: {100 + i + j}", encoding="utf-8"
                )
            rows = parse_lenet_logs.parse_lenet_directory(root, n=3, max_span=2, strict=True)
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r["native_cycles"] != "" for r in rows))


if __name__ == "__main__":
    unittest.main()
