# ==== run_noon.ps1 ====
$ErrorActionPreference = "Stop"

# Work in project root
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

# Start log
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Transcript -Path ".\logs\noon_$ts.log" | Out-Null

# Activate venv
. ".\.venv\Scripts\Activate.ps1"

# Ensure src is on PYTHONPATH
$env:PYTHONPATH = "$ROOT\src;$env:PYTHONPATH"

# 1) Update raw data (this month + last month)
python data/scrape_update.py

# 2) Pick the latest *_Numbers3features.csv
$latest = Get-ChildItem ".\data\raw\*_Numbers3features.csv" | Sort-Object Name | Select-Object -Last 1
if (-not $latest) { throw "No *_Numbers3features.csv found under .\data\raw" }
$HIST = $latest.FullName
Write-Host ("Using history: {0}" -f $HIST)

# 3) Train models
python -m n3.train_evaluate --history "$HIST" --models_dir artifacts\models

# 4) Predict next draw
python -m n3.predict_next --history "$HIST" --models_dir artifacts\models --out artifacts\outputs\next_prediction.csv

# 5) Append to prediction history
python -m n3.append_history

Stop-Transcript | Out-Null
