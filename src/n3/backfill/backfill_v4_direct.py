# -*- coding: utf-8 -*-
"""
n3.backfill_v4_direct
V4モデルを使い、指定した営業日ごとに「その前営業日までの履歴で学習済みモデルに投入→
当日の予測（百/十/一）」を1行ずつ prediction_history.csv に直接追記します。

append_history を介さないため、複数日をきちんと埋められます。
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np
from joblib import load

from n3.features_v4 import add_features_v4

TARGETS = ["百の位", "十の位", "一の位"]

def load_model_and_feats(models_dir: Path):
    models = {}
    feat_cols = None
    for tgt in TARGETS:
        mpath = models_dir / f"model_{tgt}.joblib"
        fpath = models_dir / f"features_{tgt}.json"
        if not mpath.exists() or not fpath.exists():
            raise FileNotFoundError(f"missing: {mpath} or {fpath}")
        models[tgt] = load(mpath)
        with open(fpath, "r", encoding="utf-8") as f:
            j = json.load(f)
        # 全桁で同じ想定（異なる場合も最初のを採用）
        if feat_cols is None:
            feat_cols = j.get("features", [])
    return models, feat_cols

def bday_prev(d: pd.Timestamp) -> pd.Timestamp:
    # 前営業日（日本の祝日考慮なし、平日ベース）
    return pd.bdate_range(end=d, periods=2)[0]

def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"feature columns missing: {miss[:6]}{'...' if len(miss)>6 else ''}")
    return df[cols]

def predict_one_day(d: pd.Timestamp, whole_hist: pd.DataFrame,
                    models: dict, feat_cols: list[str]) -> dict:
    """
    予測対象 = d（営業日）。特徴は「d の前営業日までの履歴」を使う。
    戻り値は prediction_history.csv に入る1行の dict （主要列のみ）
    """
    prev = bday_prev(d)
    hist_until_prev = whole_hist[whole_hist["抽せん日"] <= prev].copy()
    if hist_until_prev.empty:
        raise ValueError(f"no history until {prev.date()}")

    feat_df = add_features_v4(hist_until_prev)
    if feat_df.empty:
        raise ValueError(f"no features until {prev.date()}")
    x = ensure_cols(feat_df, feat_cols).iloc[[-1]]  # 1行DataFrame

    preds = {}
    for tgt in TARGETS:
        yhat = int(models[tgt].predict(x)[0])
        preds[tgt] = yhat

    num = preds["百の位"] * 100 + preds["十の位"] * 10 + preds["一の位"]
    row = {
        "回号": np.nan,
        "抽せん日": d.strftime("%Y-%m-%d"),
        "予測番号": num,
        "予測_百の位": preds["百の位"],
        "予測_十の位": preds["十の位"],
        "予測_一の位": preds["一の位"],
        # 正解系は後で merge_answers が埋める
        "正解_百の位": np.nan,
        "正解_十の位": np.nan,
        "正解_一の位": np.nan,
        "使用特徴セット": "V4",
        "モデル種別_百": "XGBClassifier",
        "モデル種別_十": "XGBClassifier",
        "モデル種別_一": "XGBClassifier",
    }
    return row

def append_rows(csv_path: Path, rows: list[dict]):
    out_cols_order = [
        "回号","抽せん日","抽せん日_表示","予測番号",
        "予測_百の位","予測_十の位","予測_一の位",
        "正解_百の位","正解_十の位","正解_一の位",
        "ストレート_金額","回号_推定","使用特徴セット",
        "モデル種別_百","モデル種別_十","モデル種別_一",
    ]

    if csv_path.exists():
        base = pd.read_csv(csv_path, encoding="utf-8-sig")
    else:
        base = pd.DataFrame(columns=out_cols_order)

    add = pd.DataFrame(rows)
    # 既存にある列は維持、足りない列は作る
    for c in out_cols_order:
        if c not in add.columns:
            add[c] = np.nan
    add = add[out_cols_order]

    # 同じ「抽せん日」の重複行は新しい方を残す
    merged = pd.concat([base, add], ignore_index=True)
    merged = (merged
              .sort_values(["抽せん日"])
              .drop_duplicates(subset=["抽せん日"], keep="last")
              .reset_index(drop=True))

    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {csv_path} rows={len(merged)} (+{len(add)})")

def build_dates(args) -> list[str]:
    if args.dates:
        return [s.strip() for s in args.dates.split(",") if s.strip()]
    return pd.bdate_range(args.start, args.end).strftime("%Y-%m-%d").tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="data/raw/Numbers3features_master.csv")
    ap.add_argument("--models_dir", default="artifacts/models_V4_XGB")
    ap.add_argument("--pred_hist", default="artifacts/outputs/prediction_history.csv")
    ap.add_argument("--start", default="2025-09-29")
    ap.add_argument("--end", default="2025-10-10")
    ap.add_argument("--dates", default="", help="カンマ区切りで日付列挙（指定時は start/end 無視）")
    args = ap.parse_args()

    hist = Path(args.hist)
    models_dir = Path(args.models_dir)
    pred_csv = Path(args.pred_hist)

    # 入力
    whole = pd.read_csv(hist, encoding="utf-8-sig")
    whole["抽せん日"] = pd.to_datetime(whole["抽せん日"])
    whole = whole.sort_values("抽せん日").reset_index(drop=True)

    models, feat_cols = load_model_and_feats(models_dir)
    dates = build_dates(args)
    print("[INFO] target days:", dates)

    new_rows: list[dict] = []
    for s in dates:
        d = pd.to_datetime(s)
        try:
            row = predict_one_day(d, whole, models, feat_cols)
            new_rows.append(row)
            print(f"[OK] {s} -> {row['予測番号']} ({row['予測_百の位']}{row['予測_十の位']}{row['予測_一の位']})")
        except Exception as e:
            print(f"[SKIP] {s}: {e}")

    if new_rows:
        pred_csv.parent.mkdir(parents=True, exist_ok=True)
        append_rows(pred_csv, new_rows)
    else:
        print("[INFO] no rows to append.")

    # 仕上げは従来通り（正解付与＆評価）
    try:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "n3.merge_answers"], check=True)
        subprocess.run([sys.executable, "-m", "n3.short_term_eval",
                        "--features", str(hist),
                        "--pred", str(pred_csv),
                        "--window", "30"], check=True)
    except Exception as e:
        print("[WARN] merge/eval failed:", e)

if __name__ == "__main__":
    main()
