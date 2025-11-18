@echo off
cd /d C:\Users\subar\Desktop\NumbersAI_Ver2\numbers3_clean

echo ==== Python 仮想環境をアクティベート ====
call .venv\Scripts\activate

echo ==== 予測 + EV + 履歴のローカル更新 ====
python local_refresh_joint.py --price 200 --payout 90000 --topn 1000

echo ==== Git 反映準備 ====

REM ★ artifacts/outputs（公開用CSV）
git add artifacts/outputs/*.csv

REM ★ data/raw 配下の Numbers3features（history 更新）
git add data/raw/*_Numbers3features.csv

REM 変更がない日でもエラーにならないように — allow empty 設定
git commit -m "Auto update: prediction + EV + history" --allow-empty

echo ==== Git Push ====
git push

echo.
echo ==== 完了しました ====
pause
