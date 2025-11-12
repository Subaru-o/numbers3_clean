# -*- coding: utf-8 -*-
# src/n3/train_regime.py
# 曜日レジーム分割で学習（火〜金 / 月）し、それぞれ別ディレクトリに保存
# 既存パイプラインに影響しない独立スクリプト

from __future__ import annotations
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from n3.features import add_features

TARGETS = ["百の位", "十の位", "一の位"]

REGIMES = {
    "TF": [1, 2, 3, 4],  # 火(1)〜金(4)
    "MON": [0],          # 月(0)
}

def build_xy(df_feat: pd.DataFrame, feat_cols: list[str], target_col: str):
    # y は「翌日ラベル」（次回抽せんの桁）。学習時点で未来情報は使わない。
    y = df_feat[target_col].shift(-1)  # 翌日の正解
    X = df_feat[feat_cols].copy()
    data = pd.concat([X, y.rename(f"{target_col}_next")], axis=1).dropna()
    X = data[feat_cols].astype(float)
    y = data[f"{target_col}_next"].astype(int)
    return X, y

def select_feature_columns(d: pd.DataFrame) -> list[str]:
    # 使ってはダメな列（リーク・非数）
    ban = set(["抽せん日", "当せん番号"])
    # 目的変数になる桁（当日の桁は使ってOK：翌日を当てるためリークにはならない）
    # ただし、将来情報（shift(-1)など）はこの関数では付与しない方針
    cols = []
    for c in d.columns:
        if c in ban:
            continue
        if d[c].dtype.kind in "biufc":  # 数値系のみ
            cols.append(c)
    return cols

def chronological_split(X, y, test_ratio=0.2):
    n = len(X)
    n_tr = int(np.floor(n * (1.0 - test_ratio)))
    return (X.iloc[:n_tr], X.iloc[n_tr:], y.iloc[:n_tr], y.iloc[n_tr:])

def train_one_target(d_reg: pd.DataFrame, feat_cols: list[str], target: str):
    X, y = build_xy(d_reg, feat_cols, target)
    if len(X) < 50:
        raise RuntimeError(f"学習データが不足しています（{target} rows={len(X)}）")
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, test_ratio=0.2)
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    return model, acc, feat_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="*_Numbers3features.csv")
    ap.add_argument("--out_base", required=True, help="保存ベースディレクトリ（例: artifacts/models_D_v2_regime）")
    ap.add_argument("--feature_set_name", default="D_v2_regime", help="features_*.jsonに書き出す名称")
    args = ap.parse_args()

    hist_path = Path(args.history)
    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    # 履歴→特徴量
    src = pd.read_csv(hist_path, encoding="utf-8-sig")
    src["抽せん日"] = pd.to_datetime(src["抽せん日"], errors="coerce")
    src = src.dropna(subset=["抽せん日"])
    d = add_features(src.copy()).sort_values("抽せん日").reset_index(drop=True)
    d["dow"] = d["抽せん日"].dt.dayofweek  # 月=0, 火=1, ...

    # 共通の特徴列を一度決める（数値列全体）
    feat_cols_all = select_feature_columns(d)

    for reg_name, dows in REGIMES.items():
        print(f"\n========== Regime: {reg_name} (dow in {dows}) ==========")
        d_reg = d[d["dow"].isin(dows)].copy()
        d_reg = d_reg.reset_index(drop=True)

        # 学習
        models_dir = out_base / f"models_{args.feature_set_name}_{reg_name}"
        models_dir.mkdir(parents=True, exist_ok=True)

        for tgt in TARGETS:
            try:
                model, acc, feat_cols = train_one_target(d_reg, feat_cols_all, tgt)
                joblib.dump(model, models_dir / f"model_{tgt}.joblib")
                with open(models_dir / f"features_{tgt}.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "feature_set": f"{args.feature_set_name}_{reg_name}",
                            "model_name": "RandomForestClassifier",
                            "features": feat_cols,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"[INFO] Saved {reg_name} {tgt}: acc={acc:.4f}, dir={models_dir}")
            except Exception as e:
                print(f"[ERROR] {reg_name} {tgt}: {e}")

    print("\n[INFO] レジーム学習完了:", str(out_base))

if __name__ == "__main__":
    main()
