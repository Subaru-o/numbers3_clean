@echo off
cd /d C:\Users\subar\Desktop\NumbersAI_Ver2\numbers3_clean

call .venv\Scripts\activate

python local_refresh_joint.py --price 200 --payout 90000 --topn 1000

REM ==== Git 反映 ====
git add artifacts/outputs/*.csv
git commit -m "Auto update: prediction + EV + history"
git push

echo.
echo ==== 完了しました ====
pause
