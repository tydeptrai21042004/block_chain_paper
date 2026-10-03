# Test report

Validation completed for this revision:

- `python -m compileall -q experiments ckb_bench`: passed;
- `python -m unittest discover -s experiments/tests -v`: **66/66 passed**;
- synthetic end-to-end experiment: passed;
- corrected Merkle generator: **13/13** canonical proofs reconstruct one known
  depth-4 root;
- state-chained LeNet generator: all **42** measured intervals of spans 1--4
  reproduce their exact end checkpoints;
- local Markdown link regression test: passed;
- manuscript: compiled twice with `pdflatex -halt-on-error`; current snapshot is
  **31 pages** and the modified method/results pages were rendered and visually
  checked.

## What the 66 tests cover

- exact HNDT Bellman optimality, including exhaustive enumeration on random
  small instances;
- monotonicity and backend-action properties;
- exact round-budgeted HNDT and recovery of the unconstrained optimum;
- Arbitrum/opML-Phase-1/Agatha-chain adaptation fidelity metadata and expected
  policy equivalence on an ordered trace;
- adaptive-split-only and adaptive-stop-only mechanism restrictions;
- strict ZK-backend completeness;
- all 13 canonical Merkle paths and generated Rust vector freshness;
- leaf-index left/right ordering and supplied-sibling regression checks;
- generated state-chained neural checkpoints and all 42 measured intervals;
- policy traversal and per-fault evaluation;
- trace/query/interval/config validation;
- CKB log parsers;
- heterogeneity ablation;
- documentation links.

## Environment limitation

The CKB/Rust benchmark binary itself was **not compiled or executed here**
because this environment does not contain `cargo`, `rustc`, or `ckb-debugger`.
The Python generators, generated Rust constants, parser layer, runner scripts,
and benchmark source were validated, but final CKB cycle measurements must be
run on the target Linux/WSL2 setup described in [ckb_bench/README.md](ckb_bench/README.md).

No real cycle numbers were invented and the manuscript result placeholders
remain unfilled until those measurements are available.
