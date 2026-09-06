import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from hndt.io import (
    build_model,
    read_interval_costs,
    read_query_depth_costs,
    read_trace,
    validate_atomic_coverage,
)


class IOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def valid_trace(self):
        return self.write("trace.csv", "index,name,kind\n0,a,ReLU\n1,b,GEMM\n")

    def test_read_trace_validates_consecutive_indices(self):
        path = self.write("bad.csv", "index,name,kind\n0,a,ReLU\n2,b,GEMM\n")
        with self.assertRaises(ValueError):
            read_trace(path)

    def test_read_trace_rejects_duplicate_names(self):
        path = self.write("bad.csv", "index,name,kind\n0,a,ReLU\n1,a,GEMM\n")
        with self.assertRaises(ValueError):
            read_trace(path)

    def test_interval_reader_rejects_duplicate_interval(self):
        path = self.write(
            "interval.csv",
            "i,j,span,native_cycles,zk_cycles\n0,1,1,1,\n0,1,1,2,\n",
        )
        with self.assertRaises(ValueError):
            read_interval_costs(path)

    def test_interval_reader_rejects_span_mismatch(self):
        path = self.write("interval.csv", "i,j,span,native_cycles,zk_cycles\n0,2,1,3,\n")
        with self.assertRaises(ValueError):
            read_interval_costs(path)

    def test_interval_reader_rejects_negative_cycles(self):
        path = self.write("interval.csv", "i,j,span,native_cycles,zk_cycles\n0,1,1,-3,\n")
        with self.assertRaises(ValueError):
            read_interval_costs(path)

    def test_query_reader_rejects_duplicate_depth(self):
        path = self.write("query.csv", "depth,cycles\n2,10\n2,11\n")
        with self.assertRaises(ValueError):
            read_query_depth_costs(path)

    def test_build_model_requires_depth_measurement(self):
        trace = self.valid_trace()
        interval = self.write(
            "interval.csv",
            "i,j,span,native_cycles,zk_cycles\n0,1,1,1,\n1,2,1,1,\n",
        )
        query = self.write("query.csv", "depth,cycles\n1,10\n")
        cfg = self.write("cfg.json", json.dumps({"merkle_leaf_count": 3}))
        with self.assertRaises(ValueError):
            build_model(trace, interval, query, cfg)

    def test_build_model_accepts_constant_query(self):
        trace = self.valid_trace()
        interval = self.write(
            "interval.csv",
            "i,j,span,native_cycles,zk_cycles\n0,1,1,1,\n1,2,1,1,\n",
        )
        query = self.write("query.csv", "depth,cycles\n")
        cfg = self.write(
            "cfg.json",
            json.dumps({"merkle_leaf_count": 3, "constant_query_cycles": 7, "query_fixed_overhead_cycles": 2}),
        )
        model = build_model(trace, interval, query, cfg)
        self.assertEqual(model.checked_query_cost(0, 2, 1), 9.0)
        self.assertEqual(validate_atomic_coverage(model), [])

    def test_build_model_rejects_too_small_merkle_tree(self):
        trace = self.valid_trace()
        interval = self.write(
            "interval.csv",
            "i,j,span,native_cycles,zk_cycles\n0,1,1,1,\n1,2,1,1,\n",
        )
        query = self.write("query.csv", "depth,cycles\n1,10\n")
        cfg = self.write("cfg.json", json.dumps({"merkle_leaf_count": 2, "constant_query_cycles": 1}))
        with self.assertRaises(ValueError):
            build_model(trace, interval, query, cfg)

    def test_atomic_coverage_reports_missing(self):
        trace = self.valid_trace()
        interval = self.write("interval.csv", "i,j,span,native_cycles,zk_cycles\n0,1,1,1,\n")
        query = self.write("query.csv", "depth,cycles\n")
        cfg = self.write("cfg.json", json.dumps({"merkle_leaf_count": 3, "constant_query_cycles": 1}))
        model = build_model(trace, interval, query, cfg)
        self.assertEqual(validate_atomic_coverage(model), [(1, 2)])


if __name__ == "__main__":
    unittest.main()
