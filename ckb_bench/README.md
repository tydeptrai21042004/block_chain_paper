# CKB-VM microbenchmarks for CellVG/HNDT

This directory provides the small CKB-VM measurement layer used by the manuscript. It has two purposes:

1. measure a few deterministic primitive kernels and Merkle-query costs; and
2. measure the **actual arithmetic dimensions of the canonical 12-operator LeNet-5 trace**, including selected consecutive blocks, so that HNDT can use measured interval costs rather than guessed costs.

The benchmark is intentionally not a neural-accuracy benchmark. It calibrates the terminal/query cost model used by the dispute optimizer.

## Benchmarked kernels

The CKB `no_std` Rust script supports:

- `relu`: integer ReLU loop;
- `dot`: integer dot product;
- `gemm`: square integer GEMM checksum;
- `conv`: single-channel valid 3x3 integer convolution;
- `merkle`: one CKB-Blake2b hash per authentication level;
- `lenet`: the canonical LeNet-5 arithmetic workload, addressed by operator interval `[start,end)`.

The LeNet mode uses the manuscript dimensions:

- Conv1: `1 x 32 x 32 -> 6 x 28 x 28`, 5x5 kernel;
- ReLU1: `6 x 28 x 28`;
- Pool1: `6 x 28 x 28 -> 6 x 14 x 14`;
- Conv2: `6 x 14 x 14 -> 16 x 10 x 10`, 5x5 kernels;
- ReLU2;
- Pool2: `16 x 10 x 10 -> 16 x 5 x 5`;
- Flatten: 400 elements;
- FC1: `400 -> 120`;
- ReLU3;
- FC2: `120 -> 84`;
- ReLU4;
- FC3: `84 -> 10`.

## What the cycle numbers mean

The benchmark generates deterministic integer operands **inside the contract** and keeps a volatile checksum so the arithmetic is not optimized away. Therefore the reported CKB-VM cycles are **kernel-level execution calibrations**.

They do **not** by themselves include a complete production transaction's witness decoding, tensor serialization/deserialization, commitment lookup, or other protocol overhead. If you want to claim full transaction cost, measure those components separately and add/report them explicitly. The manuscript leaves witness-byte cells blank for this reason.

This separation is deliberate: the small experiment is intended to test the HNDT cost-aware policy with reproducible VM costs, not to pretend that a microkernel is already a complete deployment.

## Prerequisites

Linux/WSL2 is recommended. Install Rust, the CKB RISC-V target, and `ckb-debugger`.

```bash
rustup target add riscv64imac-unknown-none-elf
cargo install ckb-debugger --version 1.1.1
```

The included `.cargo/config.toml` targets `riscv64imac-unknown-none-elf` and applies Rust's atomic-lowering pass for the CKB-VM toolchain. `build.rs` declares `BENCH_KIND`, `BENCH_SIZE`, `BENCH_START`, and `BENCH_END` as rebuild dependencies because the benchmark selector is compiled through `option_env!`.

## 1. Primitive and Merkle measurements

Run:

```bash
cd ckb_bench
./run_benchmarks.sh
```

Outputs:

- `logs/*.log`: retained raw debugger logs (for example `relu_16.log` and `merkle_4.log`);
- `ckb_primitive_measurements.csv`: parsed cycle counts.

The default sweep contains:

- ReLU: 16, 64, 256, 1024 elements;
- dot: 16, 32, 64, 128, 256 elements;
- GEMM: 8, 16, 24, 32;
- valid 3x3 convolution: side lengths 8, 16, 24, 32;
- Merkle/Blake2b: depths 3--8.

From the repository root, copy the measured Merkle values into the HNDT query-cost form:

```bash
python experiments/fill_query_template_from_ckb.py
```

## 2. LeNet atomic and short-interval measurements

Run:

```bash
cd ckb_bench
./run_lenet_intervals.sh
```

The script measures:

- all 12 atomic intervals (span 1);
- all 11 consecutive span-2 intervals;
- all 10 consecutive span-3 intervals;
- all 9 consecutive span-4 intervals.

That is only 42 CKB runs, rather than all 78 possible intervals. Measuring every span up to four also makes the fixed-$g$ baselines fair: any interval of size at most their advertised granularity has a measured terminal action. It is enough to give HNDT a sparse set of direct-settlement choices while keeping the experiment small.

Outputs:

- `logs_lenet/lenet_*.log`: raw debugger logs;
- `lenet_interval_measurements.csv`: parsed interval cycles.

From the repository root, populate the blank interval-cost form automatically:

```bash
python experiments/fill_interval_template_from_ckb.py
```

Unmeasured intervals remain blank and are treated as **unavailable** settlement actions by the solver. No interpolation is performed.

## 3. Run HNDT

After both fill scripts have completed:

```bash
./run_small_experiment.sh real
```

The optimizer then uses only the measured/admissible entries.

## Measurement rules for the paper

1. Use the `ckb-debugger` CKB-VM cycle count, never native desktop elapsed time.
2. Keep the raw debugger logs.
3. Do not fill manuscript cells with demo/synthetic values.
4. Do not invent costs for unmeasured intervals.
5. State clearly whether a reported number is kernel-only or includes witness/protocol overhead.
6. If a zkVM/SP1 backend is added later, keep it blank until it is independently reproduced under a clearly documented configuration.

## Reliability checks added in this revision

- old logs are cleared before each benchmark sweep, preventing stale measurements from entering a new CSV;
- parsers fail by default when a cycle line is missing;
- the LeNet parser requires exactly the configured interval set (42 intervals for `n=12`, `max_span=4`);
- duplicate or out-of-range interval logs are rejected;
- fill scripts refuse incomplete required measurement sets instead of partially populating the publication templates.
