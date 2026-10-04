# Manuscript integration note

The executable artifact is self-contained, but the manuscript source is not
required for the test or benchmark pipeline. The revised code exposes the
results needed for the paper in `experiments/results/current/`.

For the proposal-method revision, use:

- `pareto_frontier.csv` for the exact worst/mean frontier;
- `proposal_gain.csv` for Pareto-HNDT versus scalar HNDT;
- `proposal_ablation.csv` for mechanism isolation;
- `literature_individual_summary.csv` and `literature_baselines.csv` for
  paper-grounded comparison provenance.

Do not copy synthetic demo values into a manuscript.
