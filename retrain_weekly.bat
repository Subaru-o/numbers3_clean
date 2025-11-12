@echo off
REM ===== Weekly retrain (final clean) + EV =====
chcp 65001 >NUL
setlocal EnableExtensions DisableDelayedExpansion

REM ==== Settings ====
set "PROJ=C:\Users\subar\Desktop\NumbersAI_Ver2\numbers3_clean"
set "VENV=%PROJ%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "ACT_CMD=%VENV%\Scripts\activate.bat"
set "MODELDIR=artifacts\models_V4_XGB"
set "LOGDIR=logs"
set "LOG=%LOGDIR%\weekly_retrain.log"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJ%"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd_HHmmss\")"') do set "STAMP=%%I"
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyyMMdd_HHmmss\")"') do set "STAMP2=%%I"

echo.>> "%LOG%"
echo ================================================>> "%LOG%"
echo [%STAMP%] Start WEEKLY retrain job >> "%LOG%"

call "%ACT_CMD%"
if errorlevel 1 (
  echo [%STAMP%] ERROR: venv activate failed. >> "%LOG%"
  goto :END
)
set "PYTHONPATH=%PROJ%\src"
echo PYTHONPATH=%PYTHONPATH%>> "%LOG%"
echo VENV activated>> "%LOG%"

REM ---- Backup ----
set "BKBASE=artifacts\models_V4_XGB_backup"
set "BKDIR=%BKBASE%\%STAMP2%"
if not exist "%BKBASE%" mkdir "%BKBASE%"
if exist "%MODELDIR%" (
  mkdir "%BKDIR%" 2>NUL
  robocopy "%MODELDIR%" "%BKDIR%" /E /NFL /NDL /NJH /NJS /NP >> "%LOG%" 2>&1
  echo [%STAMP%] Backup done -> %BKDIR% >> "%LOG%"
) else (
  echo [%STAMP%] No existing model dir. Skip backup. >> "%LOG%"
)

REM ---- Train ----
echo [%STAMP%] Step1: train_evaluate_v4 >> "%LOG%"
"%PY%" -m n3.train_evaluate_v4 ^
  --history "data/raw/Numbers3features_master.csv" ^
  --models_dir "%MODELDIR%" ^
  --use_xgb 1 ^
  --calibrate 1 ^
  --calib_method isotonic ^
  --valid_ratio 0.10 ^
  --test_ratio 0.20 >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [%STAMP%] ERROR: training failed. See log. >> "%LOG%"
  goto :END
)

REM ---- Backtest & Candidates (期間は必要に応じて調整) ----
set "START=2025-07-01"
set "END=2025-10-10"

echo [%STAMP%] Step2: backtest_candidates_v4 (K=5, Top50, joint) >> "%LOG%"
"%PY%" -m n3.backtest_candidates_v4 ^
  --hist "data/raw/Numbers3features_master.csv" ^
  --models_dir "%MODELDIR%" ^
  --start %START% --end %END% ^
  --topk 5 --topn 50 ^
  --score joint ^
  --out "artifacts/outputs/cback_joint_k5_n50_weekly.csv" ^
  --predhist "artifacts/outputs/prediction_history.csv" ^
  --snapshot_dir "artifacts/outputs/csnap_joint_k5_n50_weekly" >> "%LOG%" 2>&1

echo [%STAMP%] Step3: generate_candidates (K=5, Top64, joint) >> "%LOG%"
"%PY%" -m n3.generate_candidates ^
  --history "data/raw/Numbers3features_master.csv" ^
  --models_dir "%MODELDIR%" ^
  --candidates_out "artifacts/outputs/next_candidates_k5_n64_joint.csv" ^
  --topk 5 --topn 64 ^
  --score joint >> "%LOG%" 2>&1

REM ---- Step4: predict_next（保険的に分布も更新） ----
echo [%STAMP%] Step4: predict_next >> "%LOG%"
"%PY%" -m n3.predict_next ^
  --history "data/raw/Numbers3features_master.csv" ^
  --models_dir "%MODELDIR%" ^
  --out "artifacts/outputs/next_prediction.csv" >> "%LOG%" 2>&1

REM ---- Step5: EV report ----
echo [%STAMP%] Step5: make_ev_report >> "%LOG%"
"%PY%" -m n3.make_ev_report ^
  --next_csv "artifacts/outputs/next_prediction.csv" ^
  --candidates "artifacts/outputs/next_candidates_k5_n64_joint.csv" ^
  --out "artifacts/outputs/ev_report.csv" ^
  --payout 90000 --cost 200 --topn 128 >> "%LOG%" 2>&1

REM ---- Output checks ----
setlocal EnableDelayedExpansion
set "MISS=0"
for %%F in (
  "artifacts\models_V4_XGB\model_一の位.joblib"
  "artifacts\models_V4_XGB\model_十の位.joblib"
  "artifacts\models_V4_XGB\model_百の位.joblib"
  "artifacts\outputs\cback_joint_k5_n50_weekly.csv"
  "artifacts\outputs\next_candidates_k5_n64_joint.csv"
  "artifacts\outputs\ev_report.csv"
) do (
  dir /b %%F >NUL 2>&1
  if errorlevel 1 (
    echo MISSING: %%F>> "%LOG%"
    set /a MISS+=1
  ) else (
    for %%I in (%%F) do echo OUTPUT OK: %%I updated=%%~tI>> "%LOG%"
  )
)
endlocal

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd HH:mm:ss\")"') do set "STAMP_END=%%I"
if "%MISS%"=="0" (
  echo [%STAMP_END%] SUCCESS: weekly retrain finished >> "%LOG%"
) else (
  echo [%STAMP_END%] DONE WITH WARNINGS: %MISS% missing outputs >> "%LOG%"
)

:END
endlocal
