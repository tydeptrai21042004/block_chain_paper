# Apply the Pareto-DPS code revision

This package contains only files changed or added relative to the uploaded `block_chain_paper-main(9).zip`, plus `PARETO_DPS.patch`.

Recommended options:

1. Replace the matching repository files with the files in this package; or
2. Review/apply `PARETO_DPS.patch` from the repository parent directory.

After applying, run:

```bash
bash run_tests.sh
```

Expected Python unit-test result for this revision: **100 tests passed**. The RISC-V build regression is environment-dependent and is skipped when the required Rust target/toolchain is unavailable.
