# -*- coding: utf-8 -*-
# src/n3/retrain_top_features.py
# 役割:
# - 既存モデル(= base_models_dir) の features_{桁}.json を参照し、
#   学習時に使った feature_set を取得（V3 or 旧来）。
# - 同じ特徴生成器で特徴を再生成し、Permutation重要度CSVの上位TopN列で
#   モデル(RF/XGB)を再学習。
#
# 使い方:
#   python -m n3.retrain_top_features \
#     --history "$HIST" \
#     --base_models_dir artifacts\models_D_v3_XGB \
#     --imp_dir artifacts\shap\D_v3 \
#     --out_models_dir artifacts\models_D_v3_top50_perm_RF \
#     --from perm \
#     --topn 50 \
#     --use_xgb 0

from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None  # xgboost未導入環境でも読み込めるように

# 旧来(A–D)の特徴生成
from n3.features import add_features
# V3特徴生成（存在しない場合の保険）
try:
    from n3.features_v3 import add_features_v3
except Exception:
    add_features_v3 = None

TARGETS = ["百の位", "十の位", "一の位"]


def _load_base_meta(base_dir: Path, target: str) -> dict:
    meta_path = base_dir / f"features_{target}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"特徴定義が見つかりません: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta  # {"feature_set": "...", "features": [...]}


def _build_features_by_set(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "V3":
        if add_features_v3 is None:
            raise RuntimeError("V3特徴が要求されていますが features_v3.py が読み込めません。")
        d = add_features_v3(df.copy())
    else:
        d = add_features(df.copy())
    if "抽せん日" not in d.columns:
        raise KeyError("抽せん日 列が見つかりません。入力CSV/前処理をご確認ください。")
    return d.sort_values("抽せん日").reset_index(drop=True)


def _load_imp_csv(imp_dir: Path, imp_from: str, target: str) -> pd.DataFrame:
    if imp_from != "perm":
        raise ValueError("--from には perm のみ対応しています。")
    path = imp_dir / f"perm_importance_{target}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。まず feature_importance.py で出力してください。")
    d = pd.read_csv(path, encoding="utf-8-sig")
    if not {"feature", "importance"}.issubset(d.columns):
        raise ValueError(f"{path} のカラムに feature / importance が見つかりません。")
    d = d.sort_values("importance", ascending=False).reset_index(drop=True)
    return d


def _make_xy(feat: pd.DataFrame, target: str, feat_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    # 翌日ラベル
    y = feat[target].shift(-1)
    X = feat[feat_cols]
    # 末尾1行は y NaN → 除外・欠損埋め
    X = X.iloc[:-1].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.iloc[:-1].astype(int)
    return X, y


def _time_split(X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2):
    n = len(X)
    n_test = max(1, int(np.floor(n * test_ratio)))
    n_train = n - n_test
    return X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]


def _train_rf(X_tr, y_tr, X_te, y_te) -> Tuple[RandomForestClassifier, float]:
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    acc = (model.predict(X_te) == y_te).mean()
    return model, float(acc)


def _train_xgb(X_tr, y_tr, X_te, y_te) -> Tuple["XGBClassifier", float]:
    if XGBClassifier is None:
        raise RuntimeError("xgboost がインポートできません。`pip install xgboost` を実行してください。")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=10,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    acc = (model.predict(X_te) == y_te).mean()
    return model, float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="履歴CSV（*_Numbers3features.csv）")
    ap.add_argument("--base_models_dir", required=True, help="元モデルの保存ディレクトリ（features_*.json を参照）")
    ap.add_argument("--imp_dir", required=True, help="重要度CSVのディレクトリ（perm_importance_*.csv）")
    ap.add_argument("--out_models_dir", required=True, help="再学習モデルの出力先")
    ap.add_argument("--from", dest="imp_from", default="perm", help="重要度種別（perm のみ対応）")
    ap.add_argument("--topn", type=int, default=50, help="上位何本の特徴で再学習するか")
    ap.add_argument("--use_xgb", type=int, default=0, help="0: RandomForest, 1: XGBoost")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    args = ap.parse_args()

    hist_path = Path(args.history)
    base_dir = Path(args.base_models_dir)
    imp_dir = Path(args.imp_dir)
    out_dir = Path(args.out_models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(hist_path, encoding="utf-8-sig")
    print(f"[INFO] loaded: {hist_path} rows={len(df_raw)}")
    print(f"[INFO] importance_from={args.imp_from}, topn={args.topn}, use_xgb={args.use_xgb}")

    for tgt in TARGETS:
        print("\n" + "="*60)
        print(f"[INFO] Retrain target: {tgt}")

        # 1) 学習時の feature_set を取得（V3か否か）
        base_meta = _load_base_meta(base_dir, tgt)
        feature_set = base_meta.get("feature_set", "A")  # 既定は旧来
        print(f"[INFO] base feature_set = {feature_set}")

        # 2) 同じ生成器で特徴再生成
        feat = _build_features_by_set(df_raw, feature_set)

        # 3) Permutation重要度CSVをロード → TopN選抜
        imp_df = _load_imp_csv(imp_dir, args.imp_from, tgt)
        top_features = imp_df["feature"].tolist()[: args.topn]

        # 4) データに存在しない列が混ざっていたら除去（＝安全）
        missing_in_data = [c for c in top_features if c not in feat.columns]
        if missing_in_data:
            # ここで落とすのではなく、警告して除外して進める
            print(f"[WARN] {tgt}: データに存在しない列を除外します: {missing_in_data[:10]} ...")
        feat_cols = [c for c in top_features if c in feat.columns]
        if len(feat_cols) == 0:
            raise RuntimeError(f"{tgt}: 使用可能な特徴が0です。重要度CSVと特徴生成の不整合を確認してください。")
        print(f"[INFO] use features = {len(feat_cols)} / {args.topn}")

        # 5) 学習用データ作成・分割
        X, y = _make_xy(feat, tgt, feat_cols)
        X_tr, X_te, y_tr, y_te = _time_split(X, y, test_ratio=args.test_ratio)

        # 6) 学習
        if args.use_xgb == 1:
            model, acc = _train_xgb(X_tr, y_tr, X_te, y_te)
            model_name = "XGBClassifier"
        else:
            model, acc = _train_rf(X_tr, y_tr, X_te, y_te)
            model_name = "RandomForestClassifier"

        print(f"[INFO] Split done, test_acc={acc:.4f}")

        # 7) 保存（モデル＋“今回使った列”＋feature_set名）
        joblib.dump(model, out_dir / f"model_{tgt}.joblib")
        meta = {
            "feature_set": "Top{}_perm({})".format(args.topn, feature_set),
            "features": feat_cols,
            "base_feature_set": feature_set,
            "model": model_name,
        }
        with open(out_dir / f"features_{tgt}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved: {out_dir / ('model_' + tgt + '.joblib')}")
        print(f"[INFO] Features saved: {out_dir / ('features_' + tgt + '.json')}")

    print(f"\n[INFO] 再学習完了: {out_dir}")
