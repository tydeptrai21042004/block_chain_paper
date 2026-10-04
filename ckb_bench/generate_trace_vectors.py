#!/usr/bin/env python3
"""Generate deterministic state-chained vectors for all publication traces.

The built-in traces intentionally reuse the integer kernels already implemented
by the CKB-VM benchmark.  They are not trained networks and are never used for
predictive-accuracy claims.  Their only purpose is to create reproducible,
ordered verification traces with different operator-cost structures.

Traces:
* lenet: existing 12-transition LeNet-5-shaped reference trace.
* conv_heavy: 12-transition convolution-dominated sequential trace.
* gemm_heavy: 12-transition fully-connected/GEMM-dominated sequential trace.

All traces contain 12 transitions / 13 checkpoints, so the same canonical
Merkle depth-4 query benchmark is comparable across them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate_lenet_vectors as lenet


@dataclass(frozen=True)
class OpSpec:
    name: str
    kind: str
    input_shape: str
    output_shape: str


@dataclass
class TraceBundle:
    key: str
    state_lens: list[int]
    states: list[list[int]]
    ops: list[OpSpec]
    arrays: list[tuple[str, list[int], str]]
    apply_op: Callable[[int, list[int]], list[int]]


def _make_initial(count: int, seed: int) -> list[int]:
    return [((i * (7 + seed) + 3 * seed) % 23) - 11 for i in range(count)]


def _fingerprint(states: list[list[int]]) -> str:
    h = hashlib.sha256()
    for state in states:
        h.update(len(state).to_bytes(4, "little"))
        for value in state:
            h.update(int(value).to_bytes(4, "little", signed=True))
    return h.hexdigest()


def _build_states(initial: list[int], apply_op: Callable[[int, list[int]], list[int]], n: int = 12) -> list[list[int]]:
    states = [initial]
    for op in range(1, n + 1):
        states.append(apply_op(op, states[-1]))
    return states


def build_conv_heavy() -> TraceBundle:
    # Three genuine convolutions are placed at different locations from LeNet.
    c1_w = lenet.make_weights(4 * 1 * 3 * 3, 21)
    c1_b = lenet.make_bias(4, 21)
    c2_w = lenet.make_weights(4 * 4 * 3 * 3, 22)
    c2_b = lenet.make_bias(4, 22)
    c3_w = lenet.make_weights(8 * 4 * 3 * 3, 23)
    c3_b = lenet.make_bias(8, 23)
    f1_w = lenet.make_weights(64 * 288, 24)
    f1_b = lenet.make_bias(64, 24)
    f2_w = lenet.make_weights(10 * 64, 25)
    f2_b = lenet.make_bias(10, 25)

    def apply(op: int, inp: list[int]) -> list[int]:
        if op == 1:
            return lenet.conv2d(inp, 1, 32, 32, 4, 3, c1_w, c1_b, 2)
        if op == 2:
            return lenet.relu(inp)
        if op == 3:
            return lenet.conv2d(inp, 4, 30, 30, 4, 3, c2_w, c2_b, 4)
        if op == 4:
            return lenet.relu(inp)
        if op == 5:
            return lenet.maxpool2x2(inp, 4, 28, 28)
        if op == 6:
            return lenet.conv2d(inp, 4, 14, 14, 8, 3, c3_w, c3_b, 4)
        if op == 7:
            return lenet.relu(inp)
        if op == 8:
            return lenet.maxpool2x2(inp, 8, 12, 12)
        if op == 9:
            if len(inp) != 288:
                raise ValueError("conv-heavy flatten expects 288 elements")
            return list(inp)
        if op == 10:
            return lenet.fully_connected(inp, 64, f1_w, f1_b, 6)
        if op == 11:
            return lenet.relu(inp)
        if op == 12:
            return lenet.fully_connected(inp, 10, f2_w, f2_b, 5)
        raise ValueError(f"unknown conv-heavy operator {op}")

    ops = [
        OpSpec("conv1", "Conv2D", "1x32x32", "4x30x30"),
        OpSpec("relu1", "ReLU", "4x30x30", "4x30x30"),
        OpSpec("conv2", "Conv2D", "4x30x30", "4x28x28"),
        OpSpec("relu2", "ReLU", "4x28x28", "4x28x28"),
        OpSpec("pool1", "MaxPool2D", "4x28x28", "4x14x14"),
        OpSpec("conv3", "Conv2D", "4x14x14", "8x12x12"),
        OpSpec("relu3", "ReLU", "8x12x12", "8x12x12"),
        OpSpec("pool2", "MaxPool2D", "8x12x12", "8x6x6"),
        OpSpec("flatten", "Flatten", "8x6x6", "288"),
        OpSpec("fc1", "GEMM", "288", "64"),
        OpSpec("relu4", "ReLU", "64", "64"),
        OpSpec("fc2", "GEMM", "64", "10"),
    ]
    states = _build_states(_make_initial(32 * 32, 2), apply)
    return TraceBundle(
        key="conv_heavy",
        state_lens=[len(s) for s in states],
        states=states,
        ops=ops,
        arrays=[
            ("CONV_HEAVY_C1_W", c1_w, "i8"), ("CONV_HEAVY_C1_B", c1_b, "i32"),
            ("CONV_HEAVY_C2_W", c2_w, "i8"), ("CONV_HEAVY_C2_B", c2_b, "i32"),
            ("CONV_HEAVY_C3_W", c3_w, "i8"), ("CONV_HEAVY_C3_B", c3_b, "i32"),
            ("CONV_HEAVY_F1_W", f1_w, "i8"), ("CONV_HEAVY_F1_B", f1_b, "i32"),
            ("CONV_HEAVY_F2_W", f2_w, "i8"), ("CONV_HEAVY_F2_B", f2_b, "i32"),
        ],
        apply_op=apply,
    )


def build_gemm_heavy() -> TraceBundle:
    dims = [192, 160, 128, 96, 64, 48, 32, 10]
    # FC layers occur at operators 1,3,5,7,9,11,12; ReLUs fill the alternating gaps.
    layer_pairs = list(zip(dims[:-1], dims[1:]))
    weights = [lenet.make_weights(out_dim * in_dim, 40 + idx) for idx, (in_dim, out_dim) in enumerate(layer_pairs)]
    biases = [lenet.make_bias(out_dim, 40 + idx) for idx, (_, out_dim) in enumerate(layer_pairs)]

    fc_ops = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 12: 6}

    def apply(op: int, inp: list[int]) -> list[int]:
        if op in fc_ops:
            idx = fc_ops[op]
            return lenet.fully_connected(inp, dims[idx + 1], weights[idx], biases[idx], 6)
        if op in {2, 4, 6, 8, 10}:
            return lenet.relu(inp)
        raise ValueError(f"unknown gemm-heavy operator {op}")

    ops = [
        OpSpec("fc1", "GEMM", "192", "160"),
        OpSpec("relu1", "ReLU", "160", "160"),
        OpSpec("fc2", "GEMM", "160", "128"),
        OpSpec("relu2", "ReLU", "128", "128"),
        OpSpec("fc3", "GEMM", "128", "96"),
        OpSpec("relu3", "ReLU", "96", "96"),
        OpSpec("fc4", "GEMM", "96", "64"),
        OpSpec("relu4", "ReLU", "64", "64"),
        OpSpec("fc5", "GEMM", "64", "48"),
        OpSpec("relu5", "ReLU", "48", "48"),
        OpSpec("fc6", "GEMM", "48", "32"),
        OpSpec("fc7", "GEMM", "32", "10"),
    ]
    states = _build_states(_make_initial(dims[0], 5), apply)
    arrays: list[tuple[str, list[int], str]] = []
    for idx, (w, b) in enumerate(zip(weights, biases), start=1):
        arrays.append((f"GEMM_HEAVY_F{idx}_W", w, "i8"))
        arrays.append((f"GEMM_HEAVY_F{idx}_B", b, "i32"))
    return TraceBundle(
        key="gemm_heavy",
        state_lens=[len(s) for s in states],
        states=states,
        ops=ops,
        arrays=arrays,
        apply_op=apply,
    )


def _rust_int_array(values: list[int], width: int = 16) -> list[str]:
    lines: list[str] = []
    for start in range(0, len(values), width):
        chunk = values[start:start + width]
        lines.append("    " + ", ".join(str(v) for v in chunk) + ",")
    return lines


def _emit_array(name: str, values: list[int], rust_type: str) -> list[str]:
    out = [f"pub static {name}: [{rust_type}; {len(values)}] = ["]
    out.extend(_rust_int_array(values))
    out.append("];\n")
    return out


def _rust_prefix(key: str) -> str:
    return key.upper()


def render_extra_rust(bundles: list[TraceBundle]) -> str:
    lines = [
        "// Generated by ../generate_trace_vectors.py; do not edit by hand.",
        "// Deterministic untrained traces for verification-cost experiments.",
        "",
    ]
    for bundle in bundles:
        prefix = _rust_prefix(bundle.key)
        lines.append(f"pub const {prefix}_STATE_LENS: [usize; 13] = [" + ", ".join(map(str, bundle.state_lens)) + "];\n")
        for name, values, rust_type in bundle.arrays:
            lines.extend(_emit_array(name, values, rust_type))
        for idx, state in enumerate(bundle.states):
            lines.extend(_emit_array(f"{prefix}_STATE_{idx}", state, "i32"))

        func = bundle.key
        lines.extend([
            f"pub fn load_{func}_checkpoint(index: usize, out: &mut [i32]) -> usize {{",
            "    match index {",
        ])
        for idx, state in enumerate(bundle.states):
            lines.append(
                f"        {idx} => {{ if out.len() < {len(state)} {{ return 0; }} out[..{len(state)}].copy_from_slice(&{prefix}_STATE_{idx}); {len(state)} }},"
            )
        lines.extend(["        _ => 0,", "    }", "}\n"])
        lines.extend([
            f"pub fn {func}_checkpoint_matches(index: usize, data: &[i32], len: usize) -> bool {{",
            "    match index {",
        ])
        for idx, state in enumerate(bundle.states):
            lines.append(f"        {idx} => len == {len(state)} && data == &{prefix}_STATE_{idx}[..],")
        lines.extend(["        _ => false,", "    }", "}\n"])
    return "\n".join(lines)


def render_trace_csv(bundle: TraceBundle) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["index", "name", "kind", "input_shape", "output_shape", "notes"], lineterminator="\n")
    writer.writeheader()
    for index, op in enumerate(bundle.ops):
        writer.writerow({
            "index": index,
            "name": op.name,
            "kind": op.kind,
            "input_shape": op.input_shape,
            "output_shape": op.output_shape,
            "notes": f"Deterministic {bundle.key} verification trace; accuracy is not evaluated.",
        })
    return buffer.getvalue()


def write_trace_csv(bundle: TraceBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_trace_csv(bundle), encoding="utf-8")


def self_check_bundle(bundle: TraceBundle) -> str:
    if len(bundle.states) != 13 or len(bundle.ops) != 12:
        raise AssertionError(f"{bundle.key} must have 13 states and 12 transitions")
    for i in range(12):
        if bundle.apply_op(i + 1, list(bundle.states[i])) != bundle.states[i + 1]:
            raise AssertionError(f"{bundle.key}: atomic mismatch at {i + 1}")
    for i in range(13):
        for j in range(i + 1, min(12, i + 4) + 1):
            current = list(bundle.states[i])
            for op in range(i + 1, j + 1):
                current = bundle.apply_op(op, current)
            if current != bundle.states[j]:
                raise AssertionError(f"{bundle.key}: interval mismatch [{i},{j}]")
    return _fingerprint(bundle.states)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rust-out", default=str(HERE / "src" / "extra_trace_vectors.rs"))
    ap.add_argument("--trace-dir", default=str(ROOT / "experiments" / "data" / "traces"))
    args = ap.parse_args()

    conv = build_conv_heavy()
    gemm = build_gemm_heavy()
    for bundle in (conv, gemm):
        fp = self_check_bundle(bundle)
        print(f"{bundle.key}: {fp}")

    # Existing reference generator remains the source of truth for LeNet.
    print(f"lenet: {lenet.self_check()}")
    rust_path = Path(args.rust_out)
    trace_dir = Path(args.trace_dir)
    generated_rust = render_extra_rust([conv, gemm])
    expected_csv = {
        trace_dir / "conv_heavy_trace.csv": render_trace_csv(conv),
        trace_dir / "gemm_heavy_trace.csv": render_trace_csv(gemm),
    }
    if args.check:
        if not rust_path.exists() or rust_path.read_text(encoding="utf-8") != generated_rust:
            raise SystemExit("generated extra trace vectors are stale; run `python ckb_bench/generate_trace_vectors.py`")
        for path, expected in expected_csv.items():
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"generated trace CSV is stale: {path}")
        print("Extra trace vectors OK: conv_heavy + gemm_heavy, 42 intervals each (spans 1-4)")
        return

    rust_path.parent.mkdir(parents=True, exist_ok=True)
    rust_path.write_text(generated_rust, encoding="utf-8")

    write_trace_csv(conv, trace_dir / "conv_heavy_trace.csv")
    write_trace_csv(gemm, trace_dir / "gemm_heavy_trace.csv")
    # Keep a copy of the canonical LeNet trace under the common trace directory.
    source_lenet = ROOT / "experiments" / "data" / "templates" / "lenet5_trace.csv"
    target_lenet = trace_dir / "lenet_trace.csv"
    target_lenet.parent.mkdir(parents=True, exist_ok=True)
    target_lenet.write_text(source_lenet.read_text(encoding="utf-8"), encoding="utf-8")
    print(rust_path)
    print(trace_dir)


if __name__ == "__main__":
    main()
