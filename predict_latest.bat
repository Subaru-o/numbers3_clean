@echo off
REM ============================
REM  Numbers3: EV再生成 & predict_next (最新予測) 一括実行
REM ============================
chcp 65001 >NUL
setlocal EnableExtensions DisableDelayedExpansion

set "PROJ=C:\Users\subar\Desktop\NumbersAI_Ver2\numbers3_clean"
set "VENV=%PROJ%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "ACT=%VENV%\Scripts\activate.bat"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "MODELDIR=artifacts\models_V4_XGB"
set "OUTDIR=artifacts\outputs"
set "LOG=%PROJ%\logs\predict_latest.log"

cd /d "%PROJ%"
if not exist "logs" mkdir "logs"
echo.>> "%LOG%"
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd HH:mm:ss\")"') do set STAMP=%%I
echo [%STAMP%] ===== start predict_latest ===== >> "%LOG%"

call "%ACT%"
if errorlevel 1 (
  echo [%STAMP%] ERROR: venv activate failed >> "%LOG%"
  goto :END
)
set "PYTHONPATH=%PROJ%\src"
echo PYTHONPATH=%PYTHONPATH% >> "%LOG%"

REM = 最新features CSV を取得 =
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command ^
  "(Get-ChildItem -Path '%PROJ%\data\raw' -Filter '*_Numbers3features.csv' | Sort-Object LastWriteTime | Select-Object -Last 1).FullName"`) do set "HIST=%%P"
if "%HIST%"=="" (
  echo [%STAMP%] ERROR: features CSV not found >> "%LOG%"
  goto :END
)
echo history=%HIST% >> "%LOG%"

REM = EV再生成 =
"%PY%" -m n3.cli --make-ev --out "%OUTDIR%\ev_report.csv" --price 200 --payout 100000 >> "%LOG%" 2>&1

REM = EVバックフィル（履歴構築） =
"%PY%" -m n3.backfill_from_ev --topk 5 --ev-th 0 --max-per-day 20 --price 200 --payout 100000 >> "%LOG%" 2>&1

REM = 次回予測（学習モデルを使った1本の予測） =
"%PY%" -m n3.predict_next --history "%HIST%" --models_dir "%MODELDIR%" --out "%OUTDIR%\next_prediction.csv" >> "%LOG%" 2>&1

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd HH:mm:ss\")"') do set STAMP_END=%%I
echo [%STAMP_END%] done. >> "%LOG%"

:END
endlocal
