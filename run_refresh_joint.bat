@echo off
REM Numbers3 ローカル最新化バッチ

cd /d C:\Users\subar\Desktop\NumbersAI_Ver2\numbers3_clean

REM 仮想環境を使っている場合
call .venv\Scripts\activate

python local_refresh_joint.py --price 200 --payout 90000 --topn 1000

REM 仮想環境を抜ける（必要なら）
REM deactivate

echo.
echo ==== 完了しました。ログを確認してください ====
pause
