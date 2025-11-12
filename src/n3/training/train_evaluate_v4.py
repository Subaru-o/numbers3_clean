# -*- coding: utf-8 -*-
"""
V4 学習スクリプト（時系列分割 + 確率校正 + 特徴量列順の恒久保存）
- 抽せん日で時系列ソートして train/valid/test を作成（shuffleしない）
- 漏洩列は除外
- ターゲットごとに使用した特徴量の『列順』を保存（features_v4_cols.json）
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

LEAK_DROP = {
    "当せん番号","正解3","正解_百の位","正解_十の位","正解_一の位",
    "百の位","十の位","一の位",
    "予測番号","予測_百の位","予測_十の位","予測_一の位",
    "候補_3桁","候補番号","候補番号3",
}

TARGETS = ["百の位","十の位","一の位"]


def find_latest_history_csv(root: Path) -> Path | None:
    dr = root / "data" / "raw"
    if not dr.exists(): return None
    c = sorted(dr.glob("*_Numbers3features.csv"))
    return c[-1] if c else None


def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf,-np.inf], np.nan).fillna(0)
    return X


def build_Xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(df[target_col], errors="coerce")
    drop_cols = [c for c in LEAK_DROP if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    X = coerce_numeric_df(X)
    return X, y


def timeseries_slices(n: int, valid_ratio: float, test_ratio: float):
    n_test  = max(1, int(round(n*test_ratio)))
    n_valid = max(1, int(round(n*valid_ratio)))
    if n_valid + n_test >= n:
        n_test  = max(1, min(n-2, n_test))
        n_valid = max(1, min(n-n_test-1, n_valid))
    return slice(0, n-n_valid-n_test), slice(n-n_valid-n_test, n-n_test), slice(n-n_test, n)


def make_base_model(use_xgb: int):
    if use_xgb and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            objective="multi:softprob", num_class=10, tree_method="hist",
            random_state=42,
        )
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=300, multi_class="multinomial")


def train_one_target(df_sorted: pd.DataFrame, tgt: str,
                     use_xgb:int, calibrate:int, calib_method:str,
                     valid_ratio:float, test_ratio:float):
    X_all, y_all = build_Xy(df_sorted, tgt)
    n = len(X_all)
    i_tr, i_va, i_te = timeseries_slices(n, valid_ratio, test_ratio)
    X_tr, y_tr = X_all.iloc[i_tr], y_all.iloc[i_tr]
    X_va, y_va = X_all.iloc[i_va], y_all.iloc[i_va]
    X_te, y_te = X_all.iloc[i_te], y_all.iloc[i_te]

    base = make_base_model(use_xgb)
    base.fit(X_tr, y_tr)

    model = base
    if calibrate:
        # sklearn 1.4+ は estimator= を使う
        model = CalibratedClassifierCV(estimator=base, cv="prefit", method=calib_method)
        model.fit(X_va, y_va)

    acc = float(accuracy_score(y_te, model.predict(X_te)))
    # ★ この順序が命：学習時に渡した X_all.columns を保存する
    used_cols_in_order = list(X_all.columns)
    return model, acc, used_cols_in_order


def save_model(model, out_dir: Path, name: str):
    import joblib
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f"model_{name}.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)   # 'auto' か パス
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--use_xgb", type=int, default=1)
    ap.add_argument("--calibrate", type=int, default=1)
    ap.add_argument("--calib_method", choices=["isotonic","sigmoid"], default="isotonic")
    ap.add_argument("--valid_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio",  type=float, default=0.20)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    if args.history == "auto":
        hist = find_latest_history_csv(ROOT)
        if hist is None:
            raise SystemExit("[ERR] data/raw の *_Numbers3features.csv が見つかりません。")
        print(f"[INFO] history CSV: {hist}")
    else:
        hist = Path(args.history)
        print(f"[INFO] history CSV: {hist}")

    df = pd.read_csv(hist, encoding="utf-8-sig")
    if "抽せん日" in df.columns:
        _d = pd.to_datetime(df["抽せん日"], errors="coerce")
        df = df.loc[_d.notna()].assign(_date=_d[_d.notna()]).sort_values("_date").drop(columns=["_date"])

    models_dir = Path(args.models_dir)
    feature_meta = {"feature_cols_by_target": {}}

    for tgt in TARGETS:
        if tgt not in df.columns:
            print(f"[WARN] '{tgt}' が無いためスキップ")
            continue
        model, acc, used_cols_in_order = train_one_target(
            df, tgt, args.use_xgb, args.calibrate, args.calib_method,
            args.valid_ratio, args.test_ratio,
        )
        print(f"[INFO] test_acc({tgt})={acc:.4f}")
        save_model(model, models_dir, tgt)
        feature_meta["feature_cols_by_target"][tgt] = used_cols_in_order

    if not feature_meta["feature_cols_by_target"]:
        raise SystemExit("[ERR] 学習対象がありません。")

    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "features_v4_cols.json").write_text(
        json.dumps(feature_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] saved feature meta: {models_dir/'features_v4_cols.json'}")
    print(f"[INFO] 学習完了: {models_dir}")
