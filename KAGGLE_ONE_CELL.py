# CellVG / HNDT — corrected one-cell Kaggle reproduction
# CPU only; real CKB-VM cycle measurements; no --demo for paper results.

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

REPO_URL = "https://github.com/tydeptrai21042004/block_chain_paper.git"
ROOT = Path("/kaggle/working/block_chain_paper")
CKB_DIR = ROOT / "ckb_bench"
EXP_DIR = ROOT / "experiments"
RESULT_DIR = EXP_DIR / "results" / "current"
FIGURE_DIR = ROOT / "figures" / "final"
ARTIFACT_DIR = ROOT / "final_paper_artifacts"
CKB_DEBUGGER_VERSION = "1.1.1"
FIXED_G = "1,2,3,4"
ROUND_BUDGET = 3
TARGET = "riscv64imac-unknown-none-elf"


def run(cmd, cwd=None, check=True, capture=False, env=None):
    print("\n" + "=" * 100)
    print("$", cmd)
    print("=" * 100)
    kwargs = dict(shell=True, cwd=cwd, env=env, text=True, check=check)
    if capture:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        result = subprocess.run(cmd, **kwargs)
        print(result.stdout)
        return result.stdout
    return subprocess.run(cmd, **kwargs)


def command_exists(name):
    return shutil.which(name) is not None


def capture(cmd, cwd=None):
    try:
        return subprocess.check_output(
            cmd, shell=True, cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def patch_repo_if_needed(root: Path):
    """Idempotent compatibility patch, safe whether GitHub is old or updated."""
    cargo_toml = root / "ckb_bench" / "Cargo.toml"
    text = cargo_toml.read_text(encoding="utf-8")
    old = 'ckb-std = { version = "1.1.0", default-features = false }'
    new = 'ckb-std = { version = "1.1.0", default-features = false, features = ["allocator"] }'
    if old in text:
        text = text.replace(old, new)
        cargo_toml.write_text(text, encoding="utf-8")
        print("[PATCH] Enabled ckb-std allocator feature.")
    elif 'features = ["allocator"]' not in text:
        raise RuntimeError("Unexpected ckb-std dependency line; refusing to guess a patch.")

    main_rs = root / "ckb_bench" / "src" / "main.rs"
    text = main_rs.read_text(encoding="utf-8")
    if "use ckb_std::entry;" in text:
        text = text.replace(
            "use ckb_std::entry;",
            "use ckb_std::{default_alloc, entry};",
            1,
        )
    elif "default_alloc" not in text:
        raise RuntimeError("Unexpected ckb_std import; allocator macro not found.")

    if "default_alloc!();" not in text:
        marker = "entry!(program_entry);"
        if marker not in text:
            raise RuntimeError("Could not locate CKB entry macro.")
        text = text.replace(marker, marker + "\ndefault_alloc!();", 1)

    text = text.replace(
        "let wi = (((oc * cin + ic) * kernel + ky) * kernel + kx);",
        "let wi = ((oc * cin + ic) * kernel + ky) * kernel + kx;",
    )
    main_rs.write_text(text, encoding="utf-8")

    # The original repo had two documentation links to this missing file.
    paper_dir = root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper_readme = paper_dir / "README.md"
    if not paper_readme.exists():
        paper_readme.write_text(
            "# CellVG / HNDT manuscript directory\n\n"
            "Documentation-only file for the manuscript path referenced by the root docs.\n"
            "It does not alter algorithms, benchmarks, measurements, or results.\n",
            encoding="utf-8",
        )
        print("[PATCH] Added missing paper/README.md.")

    # Make the regression guard present even when cloning the older GitHub commit.
    run_tests = root / "run_tests.sh"
    rt = run_tests.read_text(encoding="utf-8")
    if "[TEST] CKB RISC-V no_std contract build" not in rt:
        needle = "python3 ckb_bench/generate_lenet_vectors.py --check\n"
        guard = r'''python3 ckb_bench/generate_lenet_vectors.py --check

# Regression guard: catch no_std/allocator/toolchain failures before benchmarking.
if command -v cargo >/dev/null 2>&1 && command -v rustup >/dev/null 2>&1 && \
   rustup target list --installed | grep -qx 'riscv64imac-unknown-none-elf'; then
  echo "[TEST] CKB RISC-V no_std contract build"
  export RUSTFLAGS="${RUSTFLAGS:-} -C passes=lower-atomic"
  BENCH_KIND="relu" BENCH_SIZE="16" cargo build \
    --manifest-path ckb_bench/Cargo.toml \
    --release \
    --target riscv64imac-unknown-none-elf
  test -x ckb_bench/target/riscv64imac-unknown-none-elf/release/cellvg-ckb-bench
  echo "[PASS] CKB RISC-V no_std contract build"
else
  echo "[SKIP] CKB RISC-V build regression test (Cargo/rustup/target unavailable)."
fi
'''
        if needle not in rt:
            raise RuntimeError("Could not patch run_tests.sh safely.")
        rt = rt.replace(needle, guard, 1)
        run_tests.write_text(rt, encoding="utf-8")
        print("[PATCH] Added RISC-V contract build regression guard.")


print("\n### 1. System dependencies")
run(
    "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
    "build-essential pkg-config libssl-dev clang cmake git curl ca-certificates"
)

print("\n### 2. Rust toolchain")
cargo_bin = Path.home() / ".cargo" / "bin"
if not command_exists("cargo"):
    run(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal"
    )
os.environ["PATH"] = f"{cargo_bin}:{os.environ['PATH']}"
os.environ["CARGO_NET_GIT_FETCH_WITH_CLI"] = "true"
os.environ["MPLBACKEND"] = "Agg"
run("rustup toolchain install stable --profile minimal")
run("rustup default stable")
run(f"rustup target add {TARGET}")
run("rustc --version")
run("cargo --version")

print("\n### 3. ckb-debugger")
need_debugger = True
if command_exists("ckb-debugger"):
    version = capture("ckb-debugger --version")
    print("Existing:", version)
    need_debugger = CKB_DEBUGGER_VERSION not in version
if need_debugger:
    locked = subprocess.run(
        f"cargo install ckb-debugger --version {CKB_DEBUGGER_VERSION} --locked --force",
        shell=True,
        text=True,
    )
    if locked.returncode != 0:
        print("Locked install failed (for example because of a yanked locked crate); retrying unlocked.")
        run(f"cargo install ckb-debugger --version {CKB_DEBUGGER_VERSION} --force")
run("ckb-debugger --version")

print("\n### 4. Fresh clone + idempotent compatibility patch")
if ROOT.exists():
    shutil.rmtree(ROOT)
run(f"git clone --depth 1 {REPO_URL} {ROOT}")
COMMIT = capture("git rev-parse HEAD", cwd=ROOT)
print("Git commit:", COMMIT)
patch_repo_if_needed(ROOT)

required = [
    "experiments/run_experiment.py",
    "experiments/hndt/core.py",
    "experiments/hndt/baselines.py",
    "experiments/hndt/literature_baselines.py",
    "experiments/fill_query_template_from_ckb.py",
    "experiments/fill_interval_template_from_ckb.py",
    "ckb_bench/generate_merkle_vectors.py",
    "ckb_bench/generate_lenet_vectors.py",
    "ckb_bench/run_benchmarks.sh",
    "ckb_bench/run_lenet_intervals.sh",
    "ckb_bench/src/main.rs",
]
missing = [p for p in required if not (ROOT / p).exists()]
if missing:
    raise RuntimeError(f"Missing required repository files: {missing}")

print("\n### 5. Python dependencies")
requirements = EXP_DIR / "requirements.txt"
if requirements.exists():
    run(f"{sys.executable} -m pip install -q -r {requirements}", cwd=ROOT)
run(f"{sys.executable} -m pip install -q pandas matplotlib", cwd=ROOT)

print("\n### 6. Vectors + all tests + real RISC-V build preflight")
run(f"{sys.executable} ckb_bench/generate_merkle_vectors.py --check", cwd=ROOT)
run(f"{sys.executable} ckb_bench/generate_lenet_vectors.py --check", cwd=ROOT)
run("bash run_tests.sh", cwd=ROOT)

# Extra hard assertion: do not trust a skipped build guard in the paper run.
env = os.environ.copy()
env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " -C passes=lower-atomic").strip()
env["BENCH_KIND"] = "relu"
env["BENCH_SIZE"] = "16"
run(
    f"cargo build --manifest-path ckb_bench/Cargo.toml --release --target {TARGET}",
    cwd=ROOT,
    env=env,
)
binary = CKB_DIR / "target" / TARGET / "release" / "cellvg-ckb-bench"
if not binary.exists():
    raise RuntimeError("RISC-V CKB binary was not produced.")
print("✅ RISC-V allocator/build preflight passed:", binary)

print("\n### 7. Real primitive + Merkle CKB measurements")
run("bash run_benchmarks.sh", cwd=CKB_DIR)
primitive_csv = CKB_DIR / "ckb_primitive_measurements.csv"
if not primitive_csv.exists():
    raise RuntimeError("Primitive benchmark CSV was not generated.")
primitive_df = pd.read_csv(primitive_csv)
display(primitive_df)
merkle4 = primitive_df[
    (primitive_df["primitive"] == "merkle")
    & (primitive_df["size"].astype(int) == 4)
]
if len(merkle4) != 1 or pd.isna(merkle4.iloc[0]["cycles"]):
    raise RuntimeError("Expected one valid Merkle depth-4 measurement.")
print("✅ Merkle depth-4 cycles:", int(merkle4.iloc[0]["cycles"]))
run(f"{sys.executable} experiments/fill_query_template_from_ckb.py", cwd=ROOT)

print("\n### 8. All 42 real state-chained LeNet intervals")
run("bash run_lenet_intervals.sh", cwd=CKB_DIR)
interval_csv = CKB_DIR / "lenet_interval_measurements.csv"
if not interval_csv.exists():
    raise RuntimeError("LeNet interval benchmark CSV was not generated.")
interval_df = pd.read_csv(interval_csv)
display(interval_df)
expected = {
    (i, i + span)
    for span in (1, 2, 3, 4)
    for i in range(0, 12 - span + 1)
}
actual = set(zip(interval_df["i"].astype(int), interval_df["j"].astype(int)))
missing_intervals = sorted(expected - actual)
if missing_intervals or len(actual) != 42:
    raise RuntimeError(f"Incomplete interval set. Missing={missing_intervals}; unique={len(actual)}")
if interval_df["native_cycles"].isna().any() or (interval_df["native_cycles"] <= 0).any():
    raise RuntimeError("Invalid/non-positive native cycle count detected.")
print("✅ All 42 real state-chained intervals measured.")
run(f"{sys.executable} experiments/fill_interval_template_from_ckb.py", cwd=ROOT)

print("\n### 9. Final HNDT + baselines + ablations")
if RESULT_DIR.exists():
    shutil.rmtree(RESULT_DIR)
if FIGURE_DIR.exists():
    shutil.rmtree(FIGURE_DIR)
run(
    f"{sys.executable} experiments/run_experiment.py "
    f"--out {RESULT_DIR} --fixed-g {FIXED_G} --round-budget {ROUND_BUDGET}",
    cwd=ROOT,
)
metadata_file = RESULT_DIR / "run_metadata.json"
if not metadata_file.exists():
    raise RuntimeError("run_metadata.json is missing.")
metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
print(json.dumps(metadata, indent=2))
data_status = str(metadata.get("data_status", "")).upper()
if "SYNTHETIC" in data_status or "DEMO" in data_status:
    raise RuntimeError("STOP: final experiment used demo/synthetic cost input.")
run(
    f"{sys.executable} experiments/plot_results.py --results {RESULT_DIR} --out {FIGURE_DIR}",
    cwd=ROOT,
)

summary_file = RESULT_DIR / "summary.csv"
if not summary_file.exists():
    raise RuntimeError("summary.csv was not generated.")
summary = pd.read_csv(summary_file)
print("\nFINAL PAPER RESULT — HNDT + BASELINES")
display(summary.sort_values("worst_case_cost").reset_index(drop=True))

for filename, title in [
    ("mechanism_ablation.csv", "ADAPTIVE SPLIT / ADAPTIVE STOP ABLATION"),
    ("literature_individual_summary.csv", "PAPER-SUPPORTED BASELINE ADAPTATIONS"),
    ("literature_policy_groups.csv", "PRIOR-WORK POLICY EQUIVALENCE"),
    ("fault_costs.csv", "FAULT AT ALL 12 OPERATOR POSITIONS"),
    ("heterogeneity_ablation.csv", "HETEROGENEITY ABLATION"),
    ("round_budget_frontier.csv", "HNDT ROUND-BUDGET FRONTIER"),
    ("hndt_policy_audit.csv", "HNDT POLICY AUDIT"),
    ("optional_policy_status.csv", "OPTIONAL BACKEND STATUS"),
]:
    path = RESULT_DIR / filename
    if path.exists():
        print("\n" + title)
        display(pd.read_csv(path))

hndt = summary[summary["strategy"] == "HNDT"]
if len(hndt) != 1:
    raise RuntimeError("Expected exactly one HNDT summary row.")
if "ratio_to_optimum" in hndt.columns:
    ratio = float(hndt.iloc[0]["ratio_to_optimum"])
    if abs(ratio - 1.0) > 1e-9:
        raise RuntimeError(f"HNDT exact-optimum sanity check failed: ratio={ratio}")
print("✅ HNDT final-result sanity check passed.")

print("\n### 10. Package reproducibility artifacts")
if ARTIFACT_DIR.exists():
    shutil.rmtree(ARTIFACT_DIR)
ARTIFACT_DIR.mkdir(parents=True)
shutil.copytree(RESULT_DIR, ARTIFACT_DIR / "results")
if FIGURE_DIR.exists():
    shutil.copytree(FIGURE_DIR, ARTIFACT_DIR / "figures")
shutil.copy2(primitive_csv, ARTIFACT_DIR / "ckb_primitive_measurements.csv")
shutil.copy2(interval_csv, ARTIFACT_DIR / "lenet_interval_measurements.csv")

query_table = ROOT / "experiments/data/templates/query_costs_template.csv"
interval_table = ROOT / "experiments/data/templates/interval_costs_template.csv"
if query_table.exists():
    shutil.copy2(query_table, ARTIFACT_DIR / "query_costs_final.csv")
if interval_table.exists():
    shutil.copy2(interval_table, ARTIFACT_DIR / "interval_costs_final.csv")
for src, dst in [
    (CKB_DIR / "logs", ARTIFACT_DIR / "raw_logs_primitives"),
    (CKB_DIR / "logs_lenet", ARTIFACT_DIR / "raw_logs_intervals"),
]:
    if src.exists():
        shutil.copytree(src, dst)
for name in [
    "README.md",
    "BASELINES.md",
    "CORRECTIONS.md",
    "EXPERIMENT_GUIDE.md",
    "TEST_REPORT.md",
    "CKB_ALLOCATOR_FIX.md",
]:
    src = ROOT / name
    if src.exists():
        shutil.copy2(src, ARTIFACT_DIR / name)

(ARTIFACT_DIR / "environment.txt").write_text(
    f"""CellVG / HNDT FINAL PAPER REPRODUCTION
======================================
Repository: {REPO_URL}
Git commit: {COMMIT}
OS: {platform.platform()}
Machine: {platform.machine()}
Python: {sys.version}
Rust: {capture('rustc --version')}
Cargo: {capture('cargo --version')}
CKB debugger: {capture('ckb-debugger --version')}
Rust target: {TARGET}
GPU: NOT USED
Training epochs: NOT APPLICABLE
Reported primary metric: CKB-VM cycles
Merkle authentication depth: 4
State-chained neural transitions: 12
Measured consecutive interval spans: 1,2,3,4
Total measured neural intervals: 42
HNDT: exact minimax dynamic programming
Fixed-g policies: {FIXED_G}
Round-constrained HNDT budget: {ROUND_BUDGET}
Data status: {metadata.get('data_status')}
Local compatibility patch: allocator + missing documentation target only; scientific method unchanged.
""",
    encoding="utf-8",
)
(ARTIFACT_DIR / "git_status.txt").write_text(
    capture("git status --short", cwd=ROOT) + "\n", encoding="utf-8"
)
zip_path = shutil.make_archive(
    "/kaggle/working/CellVG_HNDT_FINAL_PAPER_RESULTS",
    "zip",
    root_dir=ARTIFACT_DIR,
)

print("\n" + "=" * 100)
print("FINAL PAPER EXPERIMENT COMPLETE")
print("=" * 100)
print("Git commit:", COMMIT)
print("Primitive CSV:", primitive_csv)
print("42 intervals CSV:", interval_csv)
print("Summary:", summary_file)
print("Figures:", FIGURE_DIR)
print("Final ZIP:", zip_path)
print("✅ NO --demo values used for final paper results.")
print("✅ NO GPU / NO training epochs.")
print("✅ Real CKB-VM cycles used.")
print("✅ 42 state-chained intervals validated.")
print("✅ Allocator/RISC-V contract build validated before measurement.")
