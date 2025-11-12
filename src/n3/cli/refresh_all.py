# src/n3/refresh_all.py
# 目的:
# - data/raw の *_Numbers3features.csv から最新版を検出
# - artifacts/outputs/next_prediction.csv と ev_report.csv を強制再生成（既存を削除）
# - 生成結果が最新版の日付に揃っているか検証
#
# 使い方:
#   python -m n3.refresh_all --models-dir artifacts/models_V4_XGB \
#       --next-out artifacts/outputs/next_prediction.csv \
#       --ev-out artifacts/outputs/ev_report.csv \
#       --price 200 --payout 90000
#
# 返り値:
#   成功で exit code 0。失敗時は非0で標準出力に理由を表示。

from __future__ import annotations
import sys
import subprocess
from pathlib import Path
from typing import Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repo ルート（.../numbers3_clean）
DATA_RAW = ROOT / "data" / "raw"

def run_py(args: list[str]) -> Tuple[int, str]:
    proc = subprocess.run(
        [sys.executable] + args, cwd=str(ROOT),
        text=True, capture_output=True
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out

def find_latest_raw_csv() -> Path | None:
    if not DATA_RAW.exists():
        return None
    files = list(DATA_RAW.glob("*_Numbers3features.csv"))
    if not files:
        return None

    def _key(p: Path):
        # 例: 20201102-20251017_Numbers3features.csv → 20251017 をキー化
        stem = p.stem
        try:
            head = stem.split("_")[0]
            last = head.split("-")[-1]
            score = int(last)
        except Exception:
            score = 0
        return (score, p.stat().st_mtime)

    return sorted(files, key=_key)[-1]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--next-out", required=True)
    ap.add_argument("--ev-out", required=True)
    ap.add_argument("--price", type=int, default=200)
    ap.add_argument("--payout", type=int, default=90000)
    args = ap.parse_args()

    models_dir = Path(args.models_dir).resolve()
    next_out   = Path(args.next_out).resolve()
    ev_out     = Path(args.ev_out).resolve()

    latest = find_latest_raw_csv()
    if not latest:
        print("[ERROR] data/raw に *_Numbers3features.csv が見つかりません。先に scrape_update を実行してください。")
        sys.exit(2)

    print(f"[INFO] Detected latest history CSV: {latest}")

    # 既存出力を削除（中身の持ち越し防止）
    for p in [next_out, ev_out]:
        try:
            if p.exists():
                p.unlink()
                print(f"[INFO] Removed old: {p}")
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")

    # 予測を実行（公式モジュールを引数付きで強制）
    print("[INFO] Running predict_next...")
    rc1, out1 = run_py([
        "-m", "n3.predict_next",
        "--history", str(latest),
        "--models_dir", str(models_dir),
        "--out", str(next_out),
    ])
    print(out1)
    if rc1 != 0:
        print("[ERROR] predict_next failed.")
        sys.exit(3)

    # EV 作成
    print("[INFO] Running make-ev...")
    rc2, out2 = run_py([
        "-m", "n3.cli",
        "--make-ev",
        "--out", str(ev_out),
        "--price", str(args.price),
        "--payout", str(args.payout),
    ])
    print(out2)
    if rc2 != 0:
        print("[ERROR] make-ev failed.")
        sys.exit(4)

    # ====== 検証：履歴の最終日 と 出力の抽せん日 が揃っているか ======
    try:
        hist = pd.read_csv(latest, encoding="utf-8-sig")
        last_day = pd.to_datetime(hist["抽せん日"], errors="coerce").max().date()
    except Exception:
        last_day = None

    try:
        df_next = pd.read_csv(next_out, encoding="utf-8-sig")
        # next 側に抽選日っぽいカラムが複数ある可能性に対応
        dcol = None
        for c in ["抽せん日","対象日","target_day","抽選日"]:
            if c in df_next.columns:
                dcol = c
                break
        next_day = pd.to_datetime(df_next[dcol], errors="coerce").max().date() if dcol else None
    except Exception:
        next_day = None

    print(f"[INFO] last_day in history = {last_day}, next_day in next_prediction = {next_day}")
    if last_day and next_day and next_day < last_day:
        print("[ERROR] 出力が古い日付を指しています。スクリプト内部で固定CSV/キャッシュを参照している可能性があります。")
        sys.exit(5)

    print("[OK] refresh_all completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    import sys
    sys.exit(main())
