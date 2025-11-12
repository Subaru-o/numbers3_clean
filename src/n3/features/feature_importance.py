# -*- coding: utf-8 -*-
# src/n3/feature_importance.py
# 学習済みモデルの features_{桁}.json を読み、保存時と同じ“特徴セット/列”で
# 時系列分割 → Permutation Importance を算出（V3/従来どちらも可）

from __future__ import annotations
import argparse
from pathlib import Path
import json
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# 旧来(A–D)の特徴
from n3.features import add_features
# V3拡張特徴（存在しない場合の保険）
try:
    from n3.features_v3 import add_features_v3
except Exception:
    add_features_v3 = None

TARGETS = ["百の位", "十の位", "一の位"]


def load_model_and_meta(models_dir: Path, target: str):
    model_path = models_dir / f"model_{target}.joblib"
    meta_path  = models_dir / f"features_{target}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"特徴定義が見つかりません: {meta_path}")
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feat_cols = meta.get("features", [])
    feature_set = meta.get("feature_set", None)  # "V3" or "A".."D"
    if not feat_cols:
        raise RuntimeError(f"{meta_path} の features が空です。")
    return model, feat_cols, feature_set


def build_features_by_set(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """feature_set に応じて適切な特徴生成を実行"""
    if feature_set == "V3":
        if add_features_v3 is None:
            raise RuntimeError("V3特徴が要求されていますが features_v3.py を読み込めません。")
        d = add_features_v3(df.copy())
    else:
        d = add_features(df.copy())
    if "抽せん日" not in d.columns:
        raise KeyError("抽せん日 列が見つかりません。CSV/前処理をご確認ください。")
    return d.sort_values("抽せん日").reset_index(drop=True)


def make_xy(feat: pd.DataFrame, target: str, feat_cols: List[str]):
    # 翌日ラベル: y(t) = target.shift(-1)
    y = feat[target].shift(-1)
    X = feat[feat_cols]
    # 末尾1行は y が NaN → 除外
    X = X.iloc[:-1].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.iloc[:-1].astype(int)
    return X, y


def time_split(X: pd.DataFrame, y: pd.Series, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n = len(X)
    n_test = max(1, int(np.floor(n * test_ratio)))
    n_train = n - n_test
    return X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]


def run_perm_importance(model, X_te: pd.DataFrame, y_te: pd.Series, n_repeats: int = 8, random_state: int = 42):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_acc = accuracy_score(y_te, model.predict(X_te))
        r = permutation_importance(
            model, X_te, y_te,
            scoring="accuracy",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
    return base_acc, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="履歴CSV（*_Numbers3features.csv）")
    ap.add_argument("--models_dir", required=True, help="モデル格納ディレクトリ（model_*.joblib / features_*.json）")
    ap.add_argument("--out_dir", required=True, help="出力先ディレクトリ（perm_importance_*.csv）")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--perm_repeats", type=int, default=8)
    args = ap.parse_args()

    hist_path = Path(args.history)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(hist_path, encoding="utf-8-sig")
    print(f"[INFO] loaded: {hist_path} rows={len(df)}")
    print(f"[INFO] test_ratio={args.test_ratio}, perm_repeats={args.perm_repeats}")

    for tgt in TARGETS:
        print("\n" + "="*70)
        print(f"### Target: {tgt}")

        try:
            # 1) モデルと“学習時の”特徴リスト・セットを読む
            model, feat_cols, feature_set = load_model_and_meta(models_dir, tgt)

            # 2) 特徴を“同じセット”で再生成（ここがV3対応の肝）
            feat = build_features_by_set(df, feature_set)

            # 3) 必要列が揃っているか確認
            missing = [c for c in feat_cols if c not in feat.columns]
            if missing:
                raise RuntimeError(f"データに存在しない列: {missing[:10]} ...")

            # 4) 時系列分割 & Permutation
            X, y = make_xy(feat, tgt, feat_cols)
            X_tr, X_te, y_tr, y_te = time_split(X, y, test_ratio=args.test_ratio)

            base_acc, r = run_perm_importance(model, X_te, y_te, n_repeats=args.perm_repeats)

            imp = pd.DataFrame({
                "feature": X_te.columns,
                "importance": r.importances_mean,
                "std": r.importances_std,
            }).sort_values("importance", ascending=False).reset_index(drop=True)

            out_csv = out_dir / f"perm_importance_{tgt}.csv"
            imp.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"[INFO] Permutation saved: {out_csv}")
            print(f"[INFO] test_acc={base_acc:.4f}")

        except Exception as e:
            print(f"[ERROR] {tgt}: {e}")


if __name__ == "__main__":
    main()
