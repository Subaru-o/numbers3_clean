@echo off
REM ==========================================
REM Numbers3 Daily Pipeline (V4 + EV + Buylist Top3)
REM ==========================================
chcp 65001 >NUL
setlocal EnableExtensions DisableDelayedExpansion

REM ---- プロジェクトルート ----
set "PROJ=C:\Users\subar\Desktop\NumbersAI_Ver2\numbers3_clean"

REM ---- venv / python ----
set "VENV=%PROJ%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "ACT=%VENV%\Scripts\activate.bat"

REM ---- パス系 ----
set "SRC=%PROJ%\src"
set "MODELDIR=%PROJ%\artifacts\models_V4_XGB"
set "OUTDIR=%PROJ%\artifacts\outputs"
set "LOGDIR=%PROJ%\logs"
set "LOG=%LOGDIR%\daily_numbers3.log"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM ---- タイムスタンプ ----
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd HH:mm:ss\")"') do set "STAMP=%%I"

echo.>> "%LOG%"
echo =========================================================>> "%LOG%"
echo [%STAMP%] Start DAILY pipeline >> "%LOG%"

REM ---- venv activate ----
call "%ACT%"
if errorlevel 1 (
  echo [%STAMP%] ERROR: venv activate failed >> "%LOG%"
  goto :END
)

REM ---- PYTHONPATH ----
set "PYTHONPATH=%SRC%;%PYTHONPATH%"
echo PYTHONPATH=%PYTHONPATH%>> "%LOG%"

REM =========================================================
REM 1) データ更新（scrape_update or scrape_all）
REM =========================================================
echo [%STAMP%] Step1: scrape_update >> "%LOG%"
REM まず -m n3.scrape_update を試す
"%PY%" -m n3.scrape_update >> "%LOG%" 2>&1
if errorlevel 1 (
  REM モジュール無い場合はファイル直叩きのフォールバック
  if exist "%PROJ%\src\n3\scrape_update.py" (
    "%PY%" "%PROJ%\src\n3\scrape_update.py" >> "%LOG%" 2>&1
  ) else if exist "%PROJ%\data\scrape_update.py" (
    "%PY%" "%PROJ%\data\scrape_update.py" >> "%LOG%" 2>&1
  ) else if exist "%PROJ%\scrape_update.py" (
    "%PY%" "%PROJ%\scrape_update.py" >> "%LOG%" 2>&1
  ) else if exist "%PROJ%\src\n3\scrape_all.py" (
    "%PY%" "%PROJ%\src\n3\scrape_all.py" >> "%LOG%" 2>&1
  ) else (
    echo [%STAMP%] WARN: scrape_update系スクリプトが見つかりません >> "%LOG%"
  )
)

REM =========================================================
REM 2) 学習（V4 XGB + キャリブレーション）
REM =========================================================
echo [%STAMP%] Step2: train_evaluate_v4 >> "%LOG%"
"%PY%" -m n3.train_evaluate_v4 ^
  --history "%PROJ%\data\raw\*_Numbers3features.csv" ^
  --models_dir "%MODELDIR%" ^
  --use_xgb 1 ^
  --calibrate 1 ^
  --calib_method isotonic ^
  --valid_ratio 0.10 ^
  --test_ratio 0.20 >> "%LOG%" 2>&1

REM 失敗時フォールバック（旧版V3）
if errorlevel 1 (
  echo [%STAMP%] train_evaluate_v4 failed. fallback to train_evaluate(V3) >> "%LOG%"
  if exist "%PROJ%\src\n3\features_v3.py" (
    "%PY%" -m n3.train_evaluate ^
      --history "%PROJ%\data\raw\*_Numbers3features.csv" ^
      --models_dir "%MODELDIR%" ^
      --feature_set V3 ^
      --use_xgb 1 ^
      --test_ratio 0.2 >> "%LOG%" 2>&1
  ) else (
    echo [%STAMP%] ERROR: features_v3.py not found. skip training. >> "%LOG%"
  )
)

REM =========================================================
REM 3) 次回予測（next_prediction.csv）
REM =========================================================
echo [%STAMP%] Step3: predict_next >> "%LOG%"
del /f /q "%OUTDIR%\next_prediction.csv" >NUL 2>&1
"%PY%" -m n3.predict_next ^
  --history "%PROJ%\data\raw\*_Numbers3features.csv" ^
  --models_dir "%MODELDIR%" ^
  --out "%OUTDIR%\next_prediction.csv" >> "%LOG%" 2>&1

REM =========================================================
REM 4) EVレポート生成（ev_report.csv）
REM =========================================================
echo [%STAMP%] Step4: make-ev >> "%LOG%"
del /f /q "%OUTDIR%\ev_report.csv" >NUL 2>&1
"%PY%" -m n3.cli ^
  --make-ev ^
  --out "%OUTDIR%\ev_report.csv" ^
  --price 200 ^
  --payout 90000 >> "%LOG%" 2>&1

REM =========================================================
REM 5) 購入リスト（Top3）
REM =========================================================
echo [%STAMP%] Step5: buylist Top3 >> "%LOG%"
"%PY%" -m n3.make_buylist ^
  --ev "%OUTDIR%\ev_report.csv" ^
  --outdir "%OUTDIR%" ^
  --topn 3 >> "%LOG%" 2>&1

echo [%STAMP%] DONE >> "%LOG%"

:END
endlocal
