# Results directory

- `demo/` — synthetic pipeline sanity data; never manuscript evidence.
- `current/` — real/user-supplied measurement run.

A completed run may contain:

- `summary.csv` — deduplicated main comparison;
- `fault_costs.csv` — all strategy/fault-position paths;
- `literature_baselines.csv` — provenance, fidelity, exclusions, run status;
- `literature_individual_summary.csv` — individual paper-adaptation rows;
- `literature_policy_groups.csv` — identical-policy grouping;
- `mechanism_ablation.csv` — adaptive split vs adaptive stop;
- `round_budget_frontier.csv` — exact cost/interaction frontier;
- `hndt_policy_audit.csv` — action alternatives and margins;
- `optional_policy_status.csv` — explicitly skipped unavailable strategies;
- `policies/*.json`;
- `heterogeneity_ablation.csv`;
- `run_metadata.json`.

The repository intentionally ships without real numerical results.

See [the experiment guide](../../EXPERIMENT_GUIDE.md).
