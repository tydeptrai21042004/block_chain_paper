$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

python -m compileall -q experiments ckb_bench
python ckb_bench/generate_merkle_vectors.py --check
python ckb_bench/generate_lenet_vectors.py --check
python -m unittest discover -s experiments/tests -v

$TmpOut = Join-Path $env:TEMP ("cellvg_hndt_demo_smoke_" + [guid]::NewGuid().ToString("N"))
try {
    python experiments/run_experiment.py --demo --out $TmpOut | Out-Null
}
finally {
    if (Test-Path $TmpOut) { Remove-Item -Recurse -Force $TmpOut }
}
Write-Host "All unit tests and the synthetic integration smoke test passed."
Write-Host "The smoke test uses synthetic data only and does not modify manuscript result tables."
