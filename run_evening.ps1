# ==== run_evening.ps1 ====
$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path ".\logs\evening_$ts.log" | Out-Null

. ".\.venv\Scripts\Activate.ps1"
$env:PYTHONPATH = "$ROOT\src;$env:PYTHONPATH"

# 1) Update raw data (catch today's result if published)
python data/scrape_update.py

# 2) Use latest CSV for answer merge
$latest = Get-ChildItem ".\data\raw\*_Numbers3features.csv" | Sort-Object Name | Select-Object -Last 1
if (-not $latest) { throw "No *_Numbers3features.csv found under .\data\raw" }
$env:N3_MASTER_CSV = $latest.FullName
Write-Host ("N3_MASTER_CSV = {0}" -f $env:N3_MASTER_CSV)

# 3) Merge answers into prediction_history.csv
python -m n3.merge_answers

# 4) (optional) backfill last 22 business days
python -m n3.backfill_last_month --history "$($latest.FullName)" --models_dir artifacts\models --out artifacts\outputs\prediction_history.csv

Stop-Transcript | Out-Null
