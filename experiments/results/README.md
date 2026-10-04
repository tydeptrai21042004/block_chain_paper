# Results directory

The repository separates **real measurement evidence** from synthetic software
and scalability checks.

- `current/` — one real/user-supplied measurement run, when available;
- `v5_demo_campaign/` — synthetic three-trace campaign; **NOT FOR MANUSCRIPT**;
- `v5_demo_sensitivity/` — synthetic fixed-grid sensitivity; **NOT FOR MANUSCRIPT**;
- `v5_demo_robustness/` — synthetic bounded-noise stability checks; **NOT FOR MANUSCRIPT**;
- `v5_demo_scalability/` — synthetic solver-scaling results; not CKB cost evidence.

A real V5 campaign should be written to a separate directory such as
`v5_real_campaign/` or `v5_real_suite/` after the raw CKB measurements have been
collected.

Per-run files include `summary.csv`, `fault_costs.csv`, `pareto_frontier.csv`,
`proposal_gain.csv`, `proposal_ablation.csv`, `round_budget_frontier.csv`,
`heterogeneity_ablation.csv`, `hndt_scalar_policy_audit.csv`,
`pareto_policy_audit.csv`, literature provenance tables, policy JSON files, and
`run_metadata.json`.

See [the experiment guide](../../EXPERIMENT_GUIDE.md).
