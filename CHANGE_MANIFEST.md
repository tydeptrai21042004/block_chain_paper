# Change-only revision manifest

Base compared: `block_chain_paper-main (2).zip`.

This revision is designed as an overlay: copy these changed/new files over the
base repository while preserving the directory structure.

## Main scientific changes

1. **Correct Merkle semantics**
   - supplied sibling digests;
   - one parent hash per proof level;
   - actual leaf-index ordering;
   - real 13-state / padded-16 tree with known-root validation;
   - separate proof-access benchmark.

2. **Real state-chained neural interval verification**
   - deterministic generated integer weights and input;
   - exact checkpoints `S0...S12` with canonical LeNet-5 dimensions;
   - every measured interval starts from `S_i`, executes the true operator
     chain, and must exactly equal `S_j`;
   - all 42 span-1..4 intervals are generator-tested.

3. **Corrected paper-grounded comparison layer**
   - Arbitrum-IVP common-trace adaptation;
   - opML explicitly limited to Phase-1/operator localization;
   - Agatha-GPP explicitly limited to the ordered-chain projection;
   - duplicate prior-work trees are deduplicated in the main table while
     retaining individual provenance and summaries.

4. **Mechanism-isolation evaluation**
   - adaptive split + atomic stop;
   - midpoint split + adaptive stop;
   - oracle best fixed-`g`;
   - strict optional ZK/full-native availability status.

5. **Proposal-method improvement**
   - exact round-constrained HNDT `F_r(i,j)`;
   - monotonic cost-vs-round frontier;
   - recovery of unconstrained HNDT with sufficient budget;
   - policy audit with best-alternative action and action margin.

6. **Paper synchronization**
   - manuscript now describes the true benchmark semantics;
   - Agatha added to related work/bibliography;
   - method includes round-constrained HNDT and proposition;
   - validation table matches the new baseline/ablation design;
   - compiled 31-page PDF snapshot included.

## Validation

- 66/66 Python tests passed.
- 13/13 canonical Merkle proofs validated.
- 42/42 measured neural intervals validated as state chained.
- synthetic integration run passed.
- manuscript compiled twice with `pdflatex -halt-on-error` and rendered pages
  were visually checked.

## Remaining external measurement step

This environment does not contain `cargo`, `rustc`, or `ckb-debugger`, so the
final CKB binary and real cycle counts were not executed here. No cycle values
were fabricated; manuscript result placeholders remain blank until a target
CKB measurement run is completed.
