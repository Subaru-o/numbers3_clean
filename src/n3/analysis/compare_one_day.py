# -*- coding: utf-8 -*-
# src/n3/compare_one_day.py
# 指定日までのデータで特徴量を作り、RFとXGBのモデルで翌日(=ラベル)を予測して比較

import argparse
from pathlib import Path
import joblib
import pandas as pd
from n3.features import add_features

TARGETS = ["百の位", "十の位", "一の位"]

def load_feats_until(hist_path: Path, until_date: str) -> pd.DataFrame:
    df = pd.read_csv(hist_path, encoding="utf-8-sig")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df[df["抽せん日"] <= pd.to_datetime(until_date)].copy()
    d = add_features(df).sort_values("抽せん日").reset_index(drop=True)
    latest = d.iloc[-1:].copy()
    return latest, d

def load_model_and_cols(models_dir: Path, target: str):
    import json
    model = joblib.load(models_dir / f"model_{target}.joblib")
    info = json.load(open(models_dir / f"features_{target}.json", "r", encoding="utf-8"))
    return model, info["features"], info.get("feature_set")

def predict_for(models_dir: Path, latest: pd.DataFrame):
    preds = {}
    kinds = {}
    for tgt in TARGETS:
        model, cols, fset = load_model_and_cols(models_dir, tgt)
        X = latest[cols].astype(float)
        preds[tgt] = int(model.predict(X)[0])
        kinds[tgt] = type(model).__name__
    s = f"{preds['百の位']}{preds['十の位']}{preds['一の位']}".zfill(3)
    return s, preds, kinds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--until", required=True, help="YYYY-MM-DD（この日までの情報で予測）")
    ap.add_argument("--rf_dir", required=True)
    ap.add_argument("--xgb_dir", required=True)
    args = ap.parse_args()

    latest, _ = load_feats_until(Path(args.history), args.until)
    s_rf, p_rf, k_rf = predict_for(Path(args.rf_dir), latest)
    s_xg, p_xg, k_xg = predict_for(Path(args.xgb_dir), latest)

    print(f"[INFO] until={args.until}")
    print(f"[RF]  model={k_rf}  pred={s_rf}  detail={p_rf}")
    print(f"[XGB] model={k_xg}  pred={s_xg}  detail={p_xg}")

if __name__ == "__main__":
    main()
