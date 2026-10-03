#!/usr/bin/env python3
"""Generate a state-chained deterministic quantized LeNet-5 trace.

The generated vectors are *not trained weights* and are not used for predictive
accuracy. They provide a reproducible integer execution trace whose operators
have the canonical LeNet-5 dimensions. Every checkpoint S_t is the exact output
of operator t applied to S_{t-1}. CKB interval benchmarks load S_i, re-execute
operators i+1..j, and require an exact match with the generated S_j.

This fixes the earlier benchmark semantics where each operator generated an
independent synthetic workload and therefore did not represent an execution
trace at all.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

STATE_LENS = [1024, 4704, 4704, 1176, 1600, 1600, 400, 400, 120, 120, 84, 84, 10]
MAX_STATE = max(STATE_LENS)


def make_weights(count: int, seed: int) -> list[int]:
    return [((i * 17 + seed * 13 + 5) % 7) - 3 for i in range(count)]


def make_bias(count: int, seed: int) -> list[int]:
    return [((i * 11 + seed * 7 + 3) % 17) - 8 for i in range(count)]


def quantize(acc: int, shift: int) -> int:
    value = acc >> shift
    return max(-32768, min(32767, value))


def conv2d(
    inp: list[int],
    cin: int,
    h: int,
    w: int,
    cout: int,
    kernel: int,
    weights: list[int],
    bias: list[int],
    shift: int,
) -> list[int]:
    oh, ow = h - kernel + 1, w - kernel + 1
    if len(inp) != cin * h * w:
        raise ValueError("invalid convolution input length")
    if len(weights) != cout * cin * kernel * kernel:
        raise ValueError("invalid convolution weight length")
    out = [0] * (cout * oh * ow)
    for oc in range(cout):
        for y in range(oh):
            for x in range(ow):
                acc = bias[oc]
                for ic in range(cin):
                    for ky in range(kernel):
                        for kx in range(kernel):
                            src = ic * h * w + (y + ky) * w + (x + kx)
                            wi = (((oc * cin + ic) * kernel + ky) * kernel + kx)
                            acc += inp[src] * weights[wi]
                out[oc * oh * ow + y * ow + x] = quantize(acc, shift)
    return out


def relu(inp: list[int]) -> list[int]:
    return [v if v > 0 else 0 for v in inp]


def maxpool2x2(inp: list[int], channels: int, h: int, w: int) -> list[int]:
    if len(inp) != channels * h * w or h % 2 or w % 2:
        raise ValueError("invalid pool input")
    oh, ow = h // 2, w // 2
    out: list[int] = []
    for c in range(channels):
        for y in range(oh):
            for x in range(ow):
                base = c * h * w
                out.append(
                    max(
                        inp[base + (2 * y) * w + 2 * x],
                        inp[base + (2 * y) * w + 2 * x + 1],
                        inp[base + (2 * y + 1) * w + 2 * x],
                        inp[base + (2 * y + 1) * w + 2 * x + 1],
                    )
                )
    return out


def fully_connected(
    inp: list[int],
    outputs: int,
    weights: list[int],
    bias: list[int],
    shift: int,
) -> list[int]:
    inputs = len(inp)
    if len(weights) != outputs * inputs or len(bias) != outputs:
        raise ValueError("invalid fully-connected dimensions")
    out = []
    for o in range(outputs):
        acc = bias[o]
        base = o * inputs
        for i, value in enumerate(inp):
            acc += value * weights[base + i]
        out.append(quantize(acc, shift))
    return out


W1 = make_weights(6 * 1 * 5 * 5, 1)
B1 = make_bias(6, 1)
W2 = make_weights(16 * 6 * 5 * 5, 2)
B2 = make_bias(16, 2)
W3 = make_weights(120 * 400, 3)
B3 = make_bias(120, 3)
W4 = make_weights(84 * 120, 4)
B4 = make_bias(84, 4)
W5 = make_weights(10 * 84, 5)
B5 = make_bias(10, 5)


def initial_state() -> list[int]:
    return [((i * 7 + 3) % 17) - 8 for i in range(32 * 32)]


def apply_op(op: int, inp: list[int]) -> list[int]:
    if op == 1:
        return conv2d(inp, 1, 32, 32, 6, 5, W1, B1, 2)
    if op == 2:
        return relu(inp)
    if op == 3:
        return maxpool2x2(inp, 6, 28, 28)
    if op == 4:
        return conv2d(inp, 6, 14, 14, 16, 5, W2, B2, 4)
    if op == 5:
        return relu(inp)
    if op == 6:
        return maxpool2x2(inp, 16, 10, 10)
    if op == 7:
        if len(inp) != 400:
            raise ValueError("flatten expects 400 elements")
        return list(inp)
    if op == 8:
        return fully_connected(inp, 120, W3, B3, 7)
    if op == 9:
        return relu(inp)
    if op == 10:
        return fully_connected(inp, 84, W4, B4, 6)
    if op == 11:
        return relu(inp)
    if op == 12:
        return fully_connected(inp, 10, W5, B5, 5)
    raise ValueError(f"unknown operator {op}")


def build_states() -> list[list[int]]:
    states = [initial_state()]
    for op in range(1, 13):
        states.append(apply_op(op, states[-1]))
    if [len(s) for s in states] != STATE_LENS:
        raise AssertionError("generated checkpoint dimensions do not match LeNet trace")
    return states


def execute_interval(states: list[list[int]], i: int, j: int) -> list[int]:
    if not 0 <= i < j <= 12:
        raise ValueError("interval must satisfy 0 <= i < j <= 12")
    current = list(states[i])
    for op in range(i + 1, j + 1):
        current = apply_op(op, current)
    return current


def _rust_int_array(values: list[int], rust_type: str, width: int = 16) -> list[str]:
    lines: list[str] = []
    for start in range(0, len(values), width):
        chunk = values[start : start + width]
        lines.append("    " + ", ".join(str(v) for v in chunk) + ",")
    return lines


def _emit_array(name: str, values: list[int], rust_type: str) -> list[str]:
    lines = [f"pub static {name}: [{rust_type}; {len(values)}] = ["]
    lines.extend(_rust_int_array(values, rust_type))
    lines.append("];\n")
    return lines


def render_rust() -> str:
    states = build_states()
    arrays: list[str] = [
        "// Generated by ../generate_lenet_vectors.py; do not edit by hand.",
        "// Deterministic integer LeNet-5-shaped trace; weights are synthetic and untrained.",
        f"pub const LENET_MAX_STATE: usize = {MAX_STATE};",
        "pub const LENET_STATE_LENS: [usize; 13] = [" + ", ".join(map(str, STATE_LENS)) + "];\n",
    ]
    for name, values, ty in [
        ("CONV1_W", W1, "i8"), ("CONV1_B", B1, "i32"),
        ("CONV2_W", W2, "i8"), ("CONV2_B", B2, "i32"),
        ("FC1_W", W3, "i8"), ("FC1_B", B3, "i32"),
        ("FC2_W", W4, "i8"), ("FC2_B", B4, "i32"),
        ("FC3_W", W5, "i8"), ("FC3_B", B5, "i32"),
    ]:
        arrays.extend(_emit_array(name, values, ty))
    for idx, state in enumerate(states):
        arrays.extend(_emit_array(f"LENET_STATE_{idx}", state, "i32"))

    arrays.extend([
        "pub fn load_checkpoint(index: usize, out: &mut [i32; LENET_MAX_STATE]) -> usize {",
        "    match index {",
    ])
    for idx, state in enumerate(states):
        arrays.append(f"        {idx} => {{ out[..{len(state)}].copy_from_slice(&LENET_STATE_{idx}); {len(state)} }},")
    arrays.extend([
        "        _ => 0,",
        "    }",
        "}\n",
        "pub fn checkpoint_matches(index: usize, data: &[i32], len: usize) -> bool {",
        "    match index {",
    ])
    for idx, state in enumerate(states):
        arrays.append(f"        {idx} => len == {len(state)} && data == &LENET_STATE_{idx}[..],")
    arrays.extend([
        "        _ => false,",
        "    }",
        "}\n",
    ])
    return "\n".join(arrays)


def fingerprint(states: list[list[int]]) -> str:
    h = hashlib.sha256()
    for state in states:
        h.update(len(state).to_bytes(4, "little"))
        for v in state:
            h.update(int(v).to_bytes(4, "little", signed=True))
    return h.hexdigest()


def self_check() -> str:
    states = build_states()
    for i in range(12):
        if execute_interval(states, i, i + 1) != states[i + 1]:
            raise AssertionError(f"atomic transition mismatch at op {i + 1}")
    for i in range(13):
        for j in range(i + 1, min(12, i + 4) + 1):
            if execute_interval(states, i, j) != states[j]:
                raise AssertionError(f"interval mismatch [{i},{j}]")
    return fingerprint(states)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "src" / "lenet_vectors.rs"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    fp = self_check()
    output = Path(args.output)
    generated = render_rust()
    if args.check:
        if not output.exists():
            raise SystemExit(f"missing generated file: {output}")
        if output.read_text(encoding="utf-8") != generated:
            raise SystemExit(
                "generated LeNet vectors are stale; run "
                "`python ckb_bench/generate_lenet_vectors.py`"
            )
        print(f"LeNet vectors OK: 42 intervals (spans 1-4), fingerprint={fp}")
        return

    output.write_text(generated, encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Trace fingerprint: {fp}")


if __name__ == "__main__":
    main()
