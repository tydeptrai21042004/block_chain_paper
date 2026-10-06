$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

python -m compileall -q experiments ckb_bench
python ckb_bench/generate_merkle_vectors.py --check
python ckb_bench/generate_lenet_vectors.py --check
python ckb_bench/generate_trace_vectors.py --check
python -m unittest discover -s experiments/tests -v

$TmpOut = Join-Path $env:TEMP ("cellvg_hndt_demo_smoke_" + [guid]::NewGuid().ToString("N"))
try {
    python experiments/run_experiment.py --demo --out $TmpOut --collect-pareto-stats | Out-Null
    python experiments/run_campaign.py --campaign experiments/config/campaign_demo.json --out (Join-Path $TmpOut "campaign") | Out-Null
    python experiments/run_generic_instance.py `
      --trace experiments/data/generic_demo/trace.csv `
      --terminal-actions experiments/data/generic_demo/terminal_actions.csv `
      --query-actions experiments/data/generic_demo/query_actions.csv `
      --metadata experiments/data/generic_demo/metadata.json `
      --out (Join-Path $TmpOut "generic") | Out-Null
    python experiments/run_cost_regimes.py --n 6 --heterogeneity 0,1 --query-ratio 0.25,1 --out (Join-Path $TmpOut "cost_regimes") | Out-Null
}
finally {
    if (Test-Path $TmpOut) { Remove-Item -Recurse -Force $TmpOut }
}
Write-Host "All unit tests, trace-vector checks, CKB adapter smoke tests, and generic Pareto-DPS smoke tests passed."
Write-Host "The smoke test uses synthetic data only and does not modify manuscript result tables."
