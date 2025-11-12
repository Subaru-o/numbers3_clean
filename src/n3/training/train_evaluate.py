# src/n3/train_evaluate.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .features_v4 import build_v4   # 新V4
from .calibrate import WeekdayIsotonic

def load_history(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, encoding="utf-8-sig")
    if "抽せん日" not in df.columns:
        raise RuntimeError("history CSV に『抽せん日』がありません。")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    return df.sort_values("抽せん日").reset_index(drop=True)

def _fit_one(X: pd.DataFrame, y: pd.Series):
    # 2モデルを学習
    xgb = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        objective="multi:softprob", num_class=10, random_state=42,
        tree_method="hist"
    )
    rf  = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42
    )
    xgb.fit(X, y)
    rf.fit(X, y)
    return xgb, rf

def _pred_avg(models, X: pd.DataFrame) -> pd.DataFrame:
    xgb, rf = models
    p1 = xgb.predict_proba(X)
    p2 = rf.predict_proba(X)
    P = 0.6*p1 + 0.4*p2   # 重みは簡易（後で最適化可）
    cols = [f"p_{i}" for i in range(10)]
    return pd.DataFrame(P, columns=cols, index=X.index)

def _pack_model(models, calib: WeekdayIsotonic):
    # 簡易に保存（pickleにしてもよいがjsonに可読化メタのみ）
    return {"type": "XGB+RF (weekday-calib)", "calib": True}

def build_dataset(df: pd.DataFrame, feature_set: str):
    if feature_set.upper() == "V4":
        return build_v4(df)
    raise RuntimeError("対応していない feature_set です（V4 を指定してください）")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--feature_set", default="V4")
    args = ap.parse_args()

    hist = load_history(Path(args.history))
    a = build_dataset(hist, args.feature_set)

    outdir = Path(args.models_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for digit in ["百","十","一"]:
        X, y, meta = a[digit]
        # 学習/検証分割（保持データが少ない想定なので holdout で簡易）
        tr_idx, te_idx = train_test_split(np.arange(len(X)), test_size=0.2, shuffle=False)
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        dow_tr, dow_te = hist.loc[tr_idx, "抽せん日"].dt.weekday, hist.loc[te_idx, "抽せん日"].dt.weekday

        # 学習
        models = _fit_one(Xtr, ytr)

        # 予測確率の平均
        p_tr = _pred_avg(models, Xtr)
        p_te = _pred_avg(models, Xte)

        # 曜日別アイソトニックで校正
        calib = WeekdayIsotonic().fit(p_tr, ytr, dow_tr)
        p_te_cal = calib.transform(p_te, dow_te)

        # 検証スコア（対数損失の代替として平均正解確率）
        correct_prob = p_te_cal.to_numpy()[np.arange(len(yte)), yte.astype(int)]
        results[digit] = float(np.mean(correct_prob))

        # メタだけ保存（本番はモデルpickleを保存して n3.predict_next 側でも使用）
        meta_json = {
            "digit": digit,
            "model_name": "XGB+RF (weekday-calib)",
            "feature_set": args.feature_set,
            "valid_mean_true_prob": results[digit],
        }
        (outdir / f"model_{digit}_meta.json").write_text(json.dumps(meta_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # ダンプ
    summary = {
        "feature_set": args.feature_set,
        "valid_mean_true_prob": results,
    }
    print("[OK] trained with V4. per-digit mean(true_prob) =", results)
    (outdir / f"summary_V4.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
