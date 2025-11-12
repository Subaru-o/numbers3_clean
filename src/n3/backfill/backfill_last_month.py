# -*- coding: utf-8 -*-
# src/n3/backfill_last_month.py
#
# 目的:
#   ★ 抽せん日 d を評価するときは、必ず d の前日までの履歴だけで学習し、
#     学習は「当日の特徴 → 翌日の桁 (shift(-1))」で行う。
#     予測入力は「d の前日」の特徴1行 → 出力は d の予測。
#
# 使い方:
#   python -m n3.backfill_last_month --history <*_Numbers3features.csv> \
#       --models_dir artifacts/models --out artifacts/outputs/prediction_history.csv \
#       --days 22
#
# 引数:
#   --history     マスター履歴CSV
#   --models_dir  モデル保存用ディレクトリ（任意。逐次学習した最新だけ保存）
#   --out         追記先の履歴CSV (prediction_history.csv)
#   --days        さかのぼる評価対象の「抽せん日」件数（末尾から何回ぶん）
#   --skip_existing  既存履歴に同じ抽せん日があればスキップ（既定: True）

from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from joblib import dump
from .features import add_features
from .model import build_rf

TARGETS = ["百の位", "十の位", "一の位"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="履歴CSV (*_Numbers3features.csv)")
    ap.add_argument("--models_dir", required=True, help="モデル保存先ディレクトリ")
    ap.add_argument("--out", required=True, help="出力CSV (prediction_history.csv)")
    ap.add_argument("--days", type=int, default=22, help="さかのぼる抽せん日件数")
    ap.add_argument("--skip_existing", action="store_true", default=True, help="既存履歴があればスキップ")
    return ap.parse_args()

def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_keywords = ["当せん番号", "抽せん日", "抽せん日_表示", "抽選日表示", "回号", "予測", "正解"]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feats = []
    for c in num_cols:
        if any(k in c for k in drop_keywords):
            continue
        feats.append(c)
    feats = [c for c in feats if not c.endswith("_lag1")]
    return sorted(feats)

def main():
    args = parse_args()
    history_path = Path(args.history)
    models_dir = Path(args.models_dir)
    out_path = Path(args.out)

    models_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 履歴読み込み
    src = pd.read_csv(history_path, encoding="utf-8-sig")
    if "抽せん日" not in src.columns:
        raise KeyError("履歴CSVに 抽せん日 列がありません。")
    src["抽せん日"] = pd.to_datetime(src["抽せん日"], errors="coerce")
    src = src.sort_values("抽せん日").reset_index(drop=True)

    # 対象日
    unique_days = src["抽せん日"].dropna().sort_values().unique()
    if len(unique_days) == 0:
        print("[ERROR] 抽せん日が空です。")
        return
    days = max(1, int(args.days))
    target_days = unique_days[-days:]

    # 既存の prediction_history の抽せん日
    existing_dates = set()
    if out_path.exists():
        try:
            hist = pd.read_csv(out_path, encoding="utf-8-sig")
            if "抽せん日" in hist.columns:
                ex = pd.to_datetime(hist["抽せん日"], errors="coerce").dropna().dt.strftime("%Y-%m-%d")
                existing_dates = set(ex.tolist())
        except Exception:
            pass

    outputs = []

    # 日付ごとにウォークフォワード
    for d in target_days:
        train_src = src[src["抽せん日"] < d].copy()
        if train_src.empty:
            print(f"[WARN] 学習データ不足のためスキップ: {pd.Timestamp(d).date()}")
            continue

        d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        if args.skip_existing and d_str in existing_dates:
            print(f"[INFO] 既存履歴のためスキップ: {d_str}")
            continue

        # 特徴量（学習用）
        df_train = add_features(train_src)
        feature_columns = _select_feature_columns(df_train)

        # 翌日ラベルにずらす
        df_shift = df_train.copy()
        for tgt in TARGETS:
            df_shift[f"{tgt}_next"] = df_shift[tgt].shift(-1)

        use_cols = feature_columns + [f"{t}_next" for t in TARGETS]
        df_tr = df_shift.dropna(subset=use_cols).copy()
        if len(df_tr) < 50:
            print(f"[WARN] 有効学習行が少ないためスキップ: {pd.Timestamp(d).date()} rows={len(df_tr)}")
            continue

        X_tr = df_tr[feature_columns]
        models = {}
        for tgt in TARGETS:
            y_tr = df_tr[f"{tgt}_next"].astype(int)
            clf = build_rf().fit(X_tr, y_tr)
            models[tgt] = clf

        # 直近モデルを保存（最後のイテレーションのみ上書き）
        try:
            from joblib import dump
            dump(models["百の位"], models_dir / "rf_百の位.joblib")
            dump(models["十の位"], models_dir / "rf_十の位.joblib")
            dump(models["一の位"], models_dir / "rf_一の位.joblib")
            meta = {
                "feature_columns": feature_columns,
                "targets": TARGETS,
                "history_path": str(history_path.resolve()),
                "trained_until": pd.to_datetime(train_src["抽せん日"]).max().strftime("%Y-%m-%d"),
                "label_def": "next_day",
                "mode": "walkforward_backfill",
            }
            with open(models_dir / "model_meta.json", "w", encoding="utf-8") as f:
                import json
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # 予測入力 = 「d の前日」の特徴 1 行
        last_feat = df_train.iloc[[-1]][feature_columns]

        preds = []
        for tgt in TARGETS:
            digit = int(models[tgt].predict(last_feat)[0])
            preds.append(digit)

        pred_number = int("".join(map(str, preds)))
        outputs.append({
            "抽せん日": d_str,
            "抽せん日_表示": "",
            "回号_推定": "",
            "予測番号": f"{pred_number:03d}",
            "予測_百の位": preds[0],
            "予測_十の位": preds[1],
            "予測_一の位": preds[2],
        })

        print(f"[OK] {d_str} pred={pred_number:03d}")

    if not outputs:
        print("[INFO] 追加するバックフィル結果がありません。")
        return

    df_out = pd.DataFrame(outputs)
    if out_path.exists():
        try:
            base = pd.read_csv(out_path, encoding="utf-8-sig")
        except Exception:
            base = pd.DataFrame()
        merged = pd.concat([base, df_out], ignore_index=True)
    else:
        merged = df_out

    if "抽せん日" in merged.columns:
        merged["抽せん日"] = pd.to_datetime(merged["抽せん日"], errors="coerce")
        merged = merged.sort_values("抽せん日")
        merged = merged.drop_duplicates(subset=["抽せん日"], keep="last")

    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("OK: backfill written ->", out_path)

if __name__ == "__main__":
    main()
