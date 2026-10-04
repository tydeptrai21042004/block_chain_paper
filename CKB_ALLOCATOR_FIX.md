# CKB `no_std` allocator build fix

The real CKB benchmark uses `ckb-std` with default features disabled. The previous manifest therefore disabled the allocator while the contract dependency graph required allocation, causing:

```text
error: no global memory allocator found but one is required
```

This revision makes three minimal changes:

1. Enables only the `allocator` feature on `ckb-std` while keeping `default-features = false`.
2. Instantiates the official `ckb_std::default_alloc!()` allocator in the CKB binary.
3. Adds a RISC-V `no_std` build regression guard to `run_tests.sh` when the target is installed.

The HNDT algorithm, baselines, Merkle/state-chain data, cost model, experiment policies, and result calculations are unchanged. `-C passes=lower-atomic` is retained for the modern CKB/Rust toolchain.
