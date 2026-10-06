# Generic Pareto-DPS input schema

Pareto-DPS itself does not require CKB.

## Trace

CSV with at least:

```csv
index,name
0,step_0
1,step_1
```

Only `index` is required and must be consecutive `0..n-1`.

## Terminal actions

```csv
i,j,action_id,cost,capabilities
0,1,replay,10,one-step;replay
1,2,proof,7,zk-proof
```

`action_id` is opaque. `cost` may be any non-negative finite numeric quantity. `capabilities` is optional metadata used by baseline adapters.

## Query actions

```csv
i,j,action_id,cuts,cost
0,4,binary_mid,2,3
0,4,ternary,1;3,4
```

`cuts` are absolute checkpoint indices. One cut is binary; multiple cuts produce a finite ordered partition.

## Metadata

Optional JSON, for example:

```json
{
  "platform": "wasm-runtime",
  "cost_unit": "fuel",
  "trace_granularity": "vm-microinstruction"
}
```

Metadata never affects Pareto optimality unless a specific baseline explicitly requires a fidelity condition such as trace granularity.

## Run

```bash
python experiments/run_generic_instance.py \
  --trace trace.csv \
  --terminal-actions terminal_actions.csv \
  --query-actions query_actions.csv \
  --metadata metadata.json \
  --out results/generic
```
