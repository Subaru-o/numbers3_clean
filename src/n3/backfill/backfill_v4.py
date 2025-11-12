# -*- coding: utf-8 -*-
"""
n3.backfill_v4 — V4モデルで過去日付の「その時点までの履歴→翌回予測」を作成し、
prediction_history に順次追記 → 正解マージ → 直近評価まで実行するユーティリティ。

既定では 2025-09-29 ～ 2025-10-10 の平日（営業日）を対象に処理します。
--start / --end で期間指定、または --dates で日付リストを直接指定できます。
"""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
from typing import Iterable, List

import pandas as pd


def run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def trim_history_until(hist: Path, until_yyyy_mm_dd: str, out_path: Path) -> None:
    """元の履歴CSVを until 日付以下にトリムして out_path へ保存。"""
    df = pd.read_csv(hist, encoding="utf-8-sig")
    if "抽せん日" not in df.columns:
        raise SystemExit("column not found: 抽せん日")

    df["抽せん日"] = pd.to_datetime(df["抽せん日"])
    cut = pd.to_datetime(until_yyyy_mm_dd)
    df = df[df["抽せん日"] <= cut].sort_values("抽せん日")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {out_path} rows={len(df)} until={until_yyyy_mm_dd}")


def build_dates(args) -> List[str]:
    if args.dates:
        # カンマ区切りの明示指定
        return [s.strip() for s in args.dates.split(",") if s.strip()]
    # start～end の平日（営業日）のみを自動生成
    rng = pd.bdate_range(args.start, args.end).strftime("%Y-%m-%d").tolist()
    return rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="data/raw/Numbers3features_master.csv")
    ap.add_argument("--models_dir", default="artifacts/models_V4_XGB")
    ap.add_argument("--out_pred", default="artifacts/outputs/next_prediction.csv")
    ap.add_argument("--start", default="2025-09-29")
    ap.add_argument("--end", default="2025-10-10")
    ap.add_argument("--dates", default="", help="カンマ区切りで日付列挙（指定時は start/end 無視）")
    ap.add_argument("--skip_merge_eval", action="store_true", help="最後の merge/eval をスキップ")
    args = ap.parse_args()

    hist = Path(args.hist)
    models_dir = Path(args.models_dir)
    out_pred = Path(args.out_pred)
    tmp_hist = Path("data/raw/_tmp_history_until.csv")

    ensure_parent(hist)
    ensure_parent(out_pred)
    ensure_parent(tmp_hist)

    dates = build_dates(args)
    print("[INFO] targets:", dates)

    for d in dates:
        print(f"\n== Backfill as of {d} ==")
        # 1) 指定日までの履歴を作成
        trim_history_until(hist, d, tmp_hist)

        # 2) その履歴を用いて“次回予測”を出力
        run([sys.executable, "-m", "n3.predict_next",
             "--history", str(tmp_hist),
             "--models_dir", str(models_dir),
             "--out", str(out_pred)])

        # 3) 予測を履歴に追記
        run([sys.executable, "-m", "n3.append_history"])

    if not args.skip_merge_eval:
        # 4) 正解マージ
        run([sys.executable, "-m", "n3.merge_answers"])
        # 5) 直近30回の評価
        run([sys.executable, "-m", "n3.short_term_eval",
             "--features", str(hist),
             "--pred", "artifacts/outputs/prediction_history.csv",
             "--window", "30"])

    print("\n[INFO] backfill_v4 finished.")


if __name__ == "__main__":
    main()
