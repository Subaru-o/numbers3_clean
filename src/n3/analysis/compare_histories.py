# -*- coding: utf-8 -*-
# src/n3/compare_histories.py
# 予測履歴CSVを複数受け取り、桁一致・2桁一致・ストレートを横並び比較

from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def load_hist(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 日付
    if "抽せん日" in df.columns:
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    # 数値化
    for c in ["予測_百の位","予測_十の位","予測_一の位","正解_百の位","正解_十の位","正解_一の位"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def metrics(df: pd.DataFrame) -> dict:
    # 位置一致（桁ごと）
    hit_b = (df["予測_百の位"] == df["正解_百の位"])
    hit_j = (df["予測_十の位"] == df["正解_十の位"])
    hit_i = (df["予測_一の位"] == df["正解_一の位"])
    hit_count = hit_b.astype(int) + hit_j.astype(int) + hit_i.astype(int)

    # 2桁一致（位置一致ベース）
    hit2 = (hit_count == 2)
    # ストレート
    hit3 = (hit_count == 3)

    # ローリング（200回窓、末尾の値を代表値に）
    def last_roll(series: pd.Series, win=200):
        if len(series) < win:
            return float(series.mean())
        return float(series.rolling(win).mean().iloc[-1])

    return {
        "件数": int(len(df)),
        "期間_from": df["抽せん日"].min().date().isoformat() if df["抽せん日"].notna().any() else "",
        "期間_to": df["抽せん日"].max().date().isoformat() if df["抽せん日"].notna().any() else "",
        "百一致率": float(hit_b.mean()),
        "十一致率": float(hit_j.mean()),
        "一一致率": float(hit_i.mean()),
        "2桁一致率": float(hit2.mean()),
        "ストレート一致率": float(hit3.mean()),
        "2桁一致_直近200": last_roll(hit2.astype(float)),
        "桁一致平均": float(np.mean([hit_b.mean(), hit_j.mean(), hit_i.mean()])),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="比較する予測履歴CSV（複数指定）")
    ap.add_argument("--out", default="artifacts/outputs/ev_compare.csv", help="比較結果の出力CSV")
    args = ap.parse_args()

    rows = []
    for p in args.files:
        name = Path(p).stem
        df = load_hist(p)
        m = metrics(df)
        m["ファイル"] = name
        # 使用特徴セット（あれば拾う）
        if "使用特徴セット" in df.columns and df["使用特徴セット"].notna().any():
            m["使用特徴セット(推定)"] = df["使用特徴セット"].dropna().iloc[-1]
        rows.append(m)

    out = pd.DataFrame(rows).set_index("ファイル").sort_values("2桁一致率", ascending=False)
    print("\n=== 比較サマリー ===")
    with pd.option_context("display.max_columns", None, "display.max_colwidth", 120):
        print(out.to_string())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, encoding="utf-8-sig")
    print(f"\n[INFO] 書き出し: {args.out}")

if __name__ == "__main__":
    main()
