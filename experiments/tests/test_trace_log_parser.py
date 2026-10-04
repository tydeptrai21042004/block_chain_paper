import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKB = ROOT / "ckb_bench"
if str(CKB) not in sys.path:
    sys.path.insert(0, str(CKB))

import parse_trace_logs


class TraceLogParserTests(unittest.TestCase):
    def test_generic_parser_accepts_complete_small_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, j in sorted(parse_trace_logs.expected_intervals(3, 2)):
                (root / f"conv_heavy_{i}_{j}.log").write_text(
                    f"Total cycles consumed: {100 + i + j}", encoding="utf-8"
                )
            rows = parse_trace_logs.parse_trace_directory(
                root, trace="conv_heavy", n=3, max_span=2, strict=True
            )
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["source"] == "ckb-debugger" for row in rows))

    def test_generic_parser_rejects_missing_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gemm_heavy_0_1.log").write_text(
                "Total cycles consumed: 100", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                parse_trace_logs.parse_trace_directory(
                    root, trace="gemm_heavy", n=3, max_span=2, strict=True
                )


if __name__ == "__main__":
    unittest.main()
