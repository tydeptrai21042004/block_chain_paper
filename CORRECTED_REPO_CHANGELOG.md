# Corrected repository changes

This patch fixes the Kaggle failure without changing the CellVG/HNDT scientific method.

## Functional fixes

- `ckb_bench/Cargo.toml`: enables the `ckb-std` `allocator` feature while retaining `default-features = false`.
- `ckb_bench/src/main.rs`: imports and instantiates `default_alloc!()` for the `#![no_std]` CKB binary.
- `ckb_bench/src/main.rs`: removes one harmless redundant-parentheses warning.
- `run_tests.sh`: adds an early RISC-V `no_std` contract build regression check when the target is available.
- `paper/README.md`: supplies the documentation target referenced by `README.md` and `EXPERIMENT_GUIDE.md`, so the documentation-link unit test passes without an ad-hoc notebook patch.
- `KAGGLE_ONE_CELL.py`: complete one-cell Kaggle reproduction script. It is idempotent and can also patch an older GitHub checkout before running.

## Validation performed here

- Python byte-compilation: passed.
- Merkle vector validation: 13/13 passed.
- State-chained LeNet vector validation: 42 intervals passed.
- Repository unit tests: 66/66 passed.
- Synthetic integration smoke test: passed.
- Shell syntax checks: passed.

The local execution container does not include Rust/Cargo, so the RISC-V binary was not built here. The Kaggle script performs the real RISC-V build as a mandatory preflight before any CKB cycle measurement.
