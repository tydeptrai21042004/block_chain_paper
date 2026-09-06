# Test report

Validation performed for this revision:

- `python -m unittest discover -s experiments/tests -v`: **46/46 passed**.
- `python -m compileall -q experiments ckb_bench`: passed.
- synthetic end-to-end experiment (`experiments/run_experiment.py --demo`): passed.
- LaTeX manuscript: compiled twice with `pdflatex -halt-on-error`; no LaTeX errors, undefined citations/references, or overfull boxes were reported.
- final PDF: rendered to 29 pages and visually checked, including the compact section hierarchy and blank result tables.

The CKB/Rust benchmark was not executed in this environment because a CKB benchmark toolchain (`cargo` plus `ckb-debugger`) is not available here. The benchmark scripts and strict parsers are included for execution on the target Linux/WSL2 setup.
