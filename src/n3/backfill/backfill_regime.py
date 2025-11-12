# -*- coding: utf-8 -*-
# src/n3/backfill_regime.py
# 曜日レジームごとに保存したモデルを、各日の曜日で自動切替してバックフィル
# 出力: prediction_history_*.csv（抽せん日 / 予測_百/十/一 / 予測番号 など）

from __future__ import annotations
import argparse
from pathlib import Path
import json
import joblib
import pandas as pd
from n3.features import add_features

TARGETS = ["百の位", "十の位", "一の位"]

def load_model_pack(models_dir: Path, target: str):
    model = joblib.load(models_dir / f"model_{target}.joblib")
    meta = {"feature_set": None, "model_name": None, "features": []}
    fp = models_dir / f"features_{target}.json"
    if fp.exists():
        with open(fp, "r", encoding="utf-8") as f:
            j = json.load(f)
        meta.update(j)
    return model, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="*_Numbers3features.csv")
    ap.add_argument("--models_base", required=True, help="train_regime.pyで作ったベース（例: artifacts/models_D_v2_regime）")
    ap.add_argument("--out", required=True, help="出力CSV（例: artifacts/outputs/prediction_history_D_v2_regime.csv）")
    ap.add_argument("--days", type=int, default=999999, help="末尾からの件数で限定（デフォルト全期間）")
    args = ap.parse_args()

    hist = Path(args.history)
    models_base = Path(args.models_base)
    out_path = Path(args.out)

    # 2つのレジーム・モデルをロード
    dir_TF = models_base / "models_D_v2_regime_TF"
    dir_MON = models_base / "models_D_v2_regime_MON"
    if not dir_TF.exists() or not dir_MON.exists():
        raise RuntimeError("モデルフォルダが見つかりません。train_regime.py の out_base と feature_set_name を確認してください。")

    models = {}
    featlists = {}
    modelnames = {}
    for reg_name, mdir in [("TF", dir_TF), ("MON", dir_MON)]:
        models[reg_name] = {}
        featlists[reg_name] = {}
        modelnames[reg_name] = {}
        for tgt in TARGETS:
            mdl, meta = load_model_pack(mdir, tgt)
            models[reg_name][tgt] = mdl
            featlists[reg_name][tgt] = meta.get("features", [])
            modelnames[reg_name][tgt] = meta.get("model_name", "RandomForestClassifier")

    # 入力→特徴量
    src = pd.read_csv(hist, encoding="utf-8-sig")
    src["抽せん日"] = pd.to_datetime(src["抽せん日"], errors="coerce")
    src = src.dropna(subset=["抽せん日"])
    d = add_features(src.copy()).sort_values("抽せん日").reset_index(drop=True)
    d["dow"] = d["抽せん日"].dt.dayofweek

    # 予測の作り方：日付 i を当てるために、特徴は前日 i-1 の行を使う（未来情報禁止）
    # よって i は 1..N-1 を巡回（最初の日は予測不可）
    rows = []
    n = len(d)
    start = max(1, n - args.days)  # 直近 days 件ぶん
    for i in range(start, n):
        date_i = d.loc[i, "抽せん日"]
        dow = int(d.loc[i, "dow"])
        # レジーム選択
        regime = "TF" if dow in [1,2,3,4] else "MON"
        # 特徴は前日
        latest = d.iloc[[i-1]]

        pred_digits = {}
        for tgt in TARGETS:
            cols = featlists[regime][tgt]
            if not cols:
                raise RuntimeError(f"{regime} {tgt}: features_*.json が不正です。")
            X = latest[cols].astype(float)
            mdl = models[regime][tgt]
            pred_digits[tgt] = int(mdl.predict(X)[0])

        s = f"{pred_digits['百の位']}{pred_digits['十の位']}{pred_digits['一の位']}".zfill(3)

        rows.append({
            "抽せん日": date_i,
            "予測_百の位": pred_digits["百の位"],
            "予測_十の位": pred_digits["十の位"],
            "予測_一の位": pred_digits["一の位"],
            "予測番号": s,
            "使用レジーム": regime,
        })

    out_df = pd.DataFrame(rows).sort_values("抽せん日").reset_index(drop=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[INFO] backfill written ->", str(out_path))

if __name__ == "__main__":
    main()
