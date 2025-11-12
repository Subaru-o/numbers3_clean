# scripts/refresh_ev.ps1
# 使い方:
#   .\scripts\refresh_ev.ps1 -ProjectRoot "." -Price 200 -Payout 90000
param(
  [string]$ProjectRoot = "$PSScriptRoot\..",
  [int]$Price = 200,
  [int]$Payout = 90000,
  [int]$Retry = 1  # スクレイピングと予測の各ステップでのリトライ回数
)

$ErrorActionPreference = "Stop"

function Invoke-Cmd {
  param([string]$Cmd, [string]$LogFile, [int]$Retry = 0, [int]$DelaySec = 5)
  $attempt = 0
  $lastExit = 0
  while ($true) {
    $attempt++
    "ATTEMPT $attempt: $Cmd" | Tee-Object -FilePath $LogFile -Append
    cmd /c $Cmd | Tee-Object -FilePath $LogFile -Append
    $lastExit = $LASTEXITCODE
    if ($lastExit -eq 0) { break }
    if ($attempt -gt $Retry) { break }
    "RETRY after ${DelaySec}s (exit=$lastExit)" | Tee-Object -FilePath $LogFile -Append
    Start-Sleep -Seconds $DelaySec
  }
  return $lastExit
}

# --- パス解決
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
Set-Location $ProjectRoot

$venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  Write-Host "[ERR] .venv の python が見つかりません → $venvPy"
  exit 10
}

$srcDir   = Join-Path $ProjectRoot "src"
$dataRaw  = Join-Path $ProjectRoot "data\raw"
$outDir   = Join-Path $ProjectRoot "artifacts\outputs"
$modelsV5 = Join-Path $ProjectRoot "artifacts\models_V5_joint"
$nextCsv  = Join-Path $outDir "next_prediction.csv"
$histTmp  = Join-Path $outDir "prediction_history.tmp.csv"

# --- ログ
$logsDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$logFile = Join-Path $logsDir "$stamp`_refresh.log"

"=== START refresh_ev.ps1 ===" | Tee-Object -FilePath $logFile -Append
"ROOT=$ProjectRoot"            | Tee-Object -FilePath $logFile -Append

# --- 環境変数（EV計算で参照）
$env:N3_PRICE  = "$Price"
$env:N3_PAYOUT = "$Payout"

# --- スクレイピング: モジュール優先、なければスクリプトにフォールバック
"=== UPDATE START ===" | Tee-Object -FilePath $logFile -Append

# 1) python -m n3.scrape_update を試す
$updateCmd1 = "`"$venvPy`" -m n3.scrape_update"
$rc = Invoke-Cmd -Cmd $updateCmd1 -LogFile $logFile -Retry $Retry
if ($rc -ne 0) {
  # 2) 代表的な配置にフォールバック
  $cand = @(
    (Join-Path $srcDir   "n3\scrape_update.py"),
    (Join-Path $ProjectRoot "data\scrape_update.py"),
    (Join-Path $ProjectRoot "scrape_update.py"),
    (Join-Path $srcDir   "n3\scrape_all.py"),
    (Join-Path $ProjectRoot "data\scrape_all.py"),
    (Join-Path $ProjectRoot "scrape_all.py")
  ) | Where-Object { Test-Path $_ } | Select-Object -First 1

  if ($null -ne $cand) {
    $updateCmd2 = "`"$venvPy`" `"$cand`""
    "FALLBACK: $cand" | Tee-Object -FilePath $logFile -Append
    $rc = Invoke-Cmd -Cmd $updateCmd2 -LogFile $logFile -Retry $Retry
  } else {
    "[ERR] スクレイピングスクリプトが見つかりません。" | Tee-Object -FilePath $logFile -Append
    exit 11
  }
}

# --- 最新 history の検出
$hist = Get-ChildItem -Path $dataRaw -Filter "*_Numbers3features.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $hist) {
  "[ERR] *_Numbers3features.csv がありません（スクレイピングに失敗？）" | Tee-Object -FilePath $logFile -Append
  exit 12
}
"USE HISTORY: $($hist.FullName)" | Tee-Object -FilePath $logFile -Append

# --- 予測（joint 1000-class）
"=== PREDICT START ===" | Tee-Object -FilePath $logFile -Append
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$predictCmd = @(
  "`"$venvPy`"", "-m", "n3.prediction.predict_next_joint",
  "--models_dir", "`"$modelsV5`"",
  "--history",    "`"$($hist.FullName)`"",
  "--out",        "`"$nextCsv`"",
  "--hist_out",   "`"$histTmp`"",
  "--price",      "$Price",
  "--payout",     "$Payout",
  "--topn",       "1000"
) -join " "
$rc = Invoke-Cmd -Cmd $predictCmd -LogFile $logFile -Retry $Retry
if ($rc -ne 0 -or -not (Test-Path $nextCsv)) {
  "[ERR] 予測に失敗、もしくは next_prediction.csv がありません。" | Tee-Object -FilePath $logFile -Append
  exit 13
}

# --- EV ビルド（Pythonヘルパー）
"=== BUILD_EV START ===" | Tee-Object -FilePath $logFile -Append
$buildEv = Join-Path $ProjectRoot "scripts\build_ev.py"
if (-not (Test-Path $buildEv)) {
  "[ERR] scripts\build_ev.py が見つかりません。" | Tee-Object -FilePath $logFile -Append
  exit 14
}
$buildCmd = "`"$venvPy`" `"$buildEv`""
$rc = Invoke-Cmd -Cmd $buildCmd -LogFile $logFile -Retry 0
if ($rc -ne 0) {
  "[ERR] EV ビルドに失敗しました。" | Tee-Object -FilePath $logFile -Append
  exit 15
}

"=== DONE (OK) ===" | Tee-Object -FilePath $logFile -Append
exit 0
