# auto_refresh_and_push.ps1
# Numbers3: スクレイピング → 予測 → EV生成 → GitHub 自動 push

param(
    [int]$Price  = 200,     # 1口あたり購入金額
    [int]$Payout = 90000,   # 1口あたり払戻金額
    [int]$TopN   = 1000,    # joint予測の候補数
    [string]$Branch = "main" # push先ブランチ
)

# ===== 設定 =====
$Root = "C:\Users\subar\Desktop\NumbersAI_Ver2\numbers3_clean"
$VenvActivate = ".venv\Scripts\activate"   # venv を使わないなら後でコメントアウト

# Git で管理したいファイル（必要に応じて増減OK）
$TargetFiles = @(
    "artifacts\outputs\ev_report.csv",
    "artifacts\outputs\next_prediction.csv",
    "artifacts\outputs\prediction_history.csv"
)

Write-Host "==== Numbers3 自動更新 ＋ GitHub push 開始 ====" -ForegroundColor Cyan

# ルートへ移動
Set-Location $Root

# ===== 仮想環境の有効化（使っている場合のみ） =====
if (Test-Path $VenvActivate) {
    Write-Host "[INFO] venv をアクティベートします: $VenvActivate"
    # PowerShell から .bat を呼ぶ感じで
    cmd /c "call $VenvActivate && python --version"
} else {
    Write-Host "[WARN] venv が見つかりませんでした。グローバル環境の python を使用します。" -ForegroundColor Yellow
}

# ===== 1) 最新化スクリプトの実行 =====
Write-Host "[STEP1] local_refresh_joint.py を実行中..." -ForegroundColor Cyan

$pythonCmd = "python local_refresh_joint.py --price $Price --payout $Payout --topn $TopN"
Write-Host "[RUN] $pythonCmd"

# ログも保存しておくと便利
$LogDir = Join-Path $Root "artifacts\logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}
$LogPath = Join-Path $LogDir "latest_run.log"

# 実行＆ログ保存
cmd /c "$pythonCmd > `"$LogPath`" 2>&1"
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host "[FATAL] local_refresh_joint.py がエラー終了しました。ログを確認してください。" -ForegroundColor Red
    Write-Host "  ログファイル: $LogPath"
    exit $exitCode
}

Write-Host "[OK] 最新化スクリプトは正常終了しました。" -ForegroundColor Green
Write-Host "  ログファイル: $LogPath"

# ===== 2) Git 変更有無の確認 =====
Write-Host "[STEP2] Git 変更状況を確認中..." -ForegroundColor Cyan

# 念のため最新を取得（衝突ケアしたいなら、ここで git pull も可）
# git pull origin $Branch

# まず対象ファイルが存在するか確認
$ExistingTargets = @()
foreach ($f in $TargetFiles) {
    if (Test-Path $f) {
        $ExistingTargets += $f
    } else {
        Write-Host "[WARN] 指定ファイルが見つかりません: $f" -ForegroundColor Yellow
    }
}

if ($ExistingTargets.Count -eq 0) {
    Write-Host "[WARN] 追加対象のファイルが 1つも存在しません。push はスキップします。" -ForegroundColor Yellow
    exit 0
}

# 現在の差分をチェック（porcelain形式で機械的に見やすい）
$gitStatus = git status --porcelain
if ([string]::IsNullOrWhiteSpace($gitStatus)) {
    Write-Host "[INFO] 作業ツリーに変更はありません。push は不要です。" -ForegroundColor Green
    exit 0
}

Write-Host "[INFO] 変更が検出されました。対象CSVを git add します。"

# 対象ファイルだけ add
foreach ($f in $ExistingTargets) {
    Write-Host "  git add $f"
    git add $f
}

# status 再チェック（対象外の変更をコミットしたくない場合の簡易チェック）
$gitStatusAfterAdd = git status --porcelain
if ([string]::IsNullOrWhiteSpace($gitStatusAfterAdd)) {
    Write-Host "[INFO] 対象CSV以外に変更はなく、結果的にコミット対象もありませんでした。" -ForegroundColor Green
    exit 0
}

# ===== 3) 自動コミット =====
$now = Get-Date
$timestamp = $now.ToString("yyyy-MM-dd HH:mm:ss")
$shortDate = $now.ToString("yyyyMMdd")

$commitMessage = "auto: update Numbers3 predictions ($shortDate)"

Write-Host "[STEP3] git commit を実行します..." -ForegroundColor Cyan
Write-Host "  メッセージ: $commitMessage"

git commit -m "$commitMessage"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERR] git commit でエラーが発生しました。手動で確認してください。" -ForegroundColor Red
    exit 1
}

# ===== 4) GitHub へ push =====
Write-Host "[STEP4] git push origin $Branch を実行します..." -ForegroundColor Cyan

git push origin $Branch

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERR] git push でエラーが発生しました。ネットワークや認証情報を確認してください。" -ForegroundColor Red
    exit 1
}

Write-Host "==== 自動更新 ＋ GitHub push 完了 ✅ ====" -ForegroundColor Green
exit 0
