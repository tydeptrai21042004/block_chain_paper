# Revision change log

This revision addresses both reviewer comments and the experimental weaknesses
identified during artifact inspection.

## P0 benchmark corrections

### Merkle authentication

- supplied sibling digests are no longer hashed inside the measured proof loop;
- verification performs one parent CKB-Blake2b hash per level;
- left/right ordering uses the actual leaf-index bits;
- the publication tree contains 13 real leaves padded to 16 with zero digests;
- all 13 real proofs reconstruct one known root;
- proof-data memory access is measured separately from parent hashing.

### Neural interval verification

The previous `bench_lenet_block()` summed independent operator-shaped workloads.
That did not satisfy the paper's trace semantics `S_t=f_t(S_{t-1})`.

The replacement:

- generates one deterministic integer input and fixed quantized weights;
- constructs exact checkpoints `S0...S12` with canonical LeNet-5 dimensions;
- versions the generated checkpoint/weight source in the repository;
- verifies all 42 measured span-1..4 intervals in the generator tests;
- makes the CKB benchmark load `S_i`, execute the real operator chain, and
  require exact equality with `S_j`.

## Paper-grounded comparison corrections

The repository now avoids overstating baseline reproduction:

- **Arbitrum-IVP** is explicitly a common-trace structural adaptation;
- **opML** is explicitly **Phase-1/operator localization only**; no unmeasured
  microinstruction phase is fabricated;
- **Agatha-GPP** is explicitly a **chain projection** of the graph-based
  pinpoint protocol; general DAG/XCE behavior is not claimed.

Because the three comparable localization components collapse to the same tree
on the current ordered trace, the main table deduplicates them into one
`Prior-work midpoint/pinpoint family`. Individual provenance and summaries are
still exported.

## Method improvement

The original HNDT Bellman equation is retained. The extension is intentionally
minimal and exact:

```text
F_0(i,j) = A(i,j)
F_r(i,j) = min(
    A(i,j),
    min_k q(i,j,k) + max(F_{r-1}(i,k), F_{r-1}(k,j))
)
```

`F_r` is the exact optimum under a hard worst-case split-round budget. It yields
an interpretable cost--interaction frontier without inventing a latency weight.

The experiment also adds two exact mechanism-isolation restrictions:

- adaptive split + atomic stop;
- midpoint split + adaptive stop.

Together with HNDT and the prior-work family, these form a clean 2x2 test of the
two novel decisions.

## Auditability improvements

- oracle best fixed-`g` comparator;
- duplicate literature-policy grouping;
- explicit skipped-strategy status;
- policy JSON for every executed strategy;
- HNDT action margin to the best alternative decision;
- exact round-budget frontier;
- synchronized manuscript terminology and validation design.
