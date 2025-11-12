# src/n3/train_joint.py
# Numbers3 ジョイント（1000クラス）学習：
# 1) RFで学習 → 2) 可能なら CalibratedClassifierCV(sigmoid, prefit)
# 3) 失敗時はグローバル温度スケーリング（power calibration）で確率補正
#    p_cal = normalize(p ** alpha) を検証NLLで最適化
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List
from collections import Counter

import numpy as np
import pandas as pd
from joblib import dump
from dataclasses import dataclass

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

# -------- utils --------
def log(msg: str) -> None:
    print(f"[INFO] {msg}")

def fmt3(v) -> str:
    try:
        return f"{int(float(v))%1000:03d}"
    except Exception:
        return "000"

def detect_target_series(df: pd.DataFrame) -> pd.Series:
    cols = df.columns
    if "当選番号" in cols:
        s = pd.to_numeric(df["当選番号"], errors="coerce")
        if s.notna().any():
            return s.fillna(0).astype(int).astype(str).str.zfill(3)
    if "当せん番号" in cols:
        s = pd.to_numeric(df["当せん番号"], errors="coerce")
        if s.notna().any():
            return s.fillna(0).astype(int).astype(str).str.zfill(3)
    need = ["百の位","十の位","一の位"]
    if all(c in cols for c in need):
        b = pd.to_numeric(df["百の位"], errors="coerce").fillna(0).astype(int).astype(str)
        t = pd.to_numeric(df["十の位"], errors="coerce").fillna(0).astype(int).astype(str)
        o = pd.to_numeric(df["一の位"], errors="coerce").fillna(0).astype(int).astype(str)
        return (b + t + o).str.zfill(3)
    raise ValueError("ターゲット（当選番号/当せん番号 or 百/十/一）が見つかりません。")

def build_feature_matrix(df: pd.DataFrame, target_col_name: str) -> pd.DataFrame:
    drop_like = {
        target_col_name,
        "当選番号","当せん番号","百の位","十の位","一の位",
        "抽せん日","抽選日","date","draw_date","曜日","回号"
    }
    num_df = df.select_dtypes(include=[np.number]).copy()
    use_cols = [c for c in num_df.columns if c not in drop_like]
    X = num_df[use_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    if X.empty:
        raise ValueError("特徴量が空です（数値列が無い/除外し過ぎの可能性）。")
    return X

def make_label_encoder_with_all_1000() -> LabelEncoder:
    le = LabelEncoder()
    le.fit([f"{i:03d}" for i in range(1000)])
    return le

def choose_calibration(y_encoded: np.ndarray) -> Tuple[str, int]:
    cnt = Counter(y_encoded.tolist())
    if not cnt:
        return "sigmoid", 3
    m = min(cnt.values())
    if m >= 8:
        return "isotonic", 5
    elif m >= 2:
        return "sigmoid", 3
    else:
        return "sigmoid", 2

def temporal_split(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, val_ratio: float = 0.15):
    """
    抽せん日/抽選日があれば“時間順”で末尾を検証に。無ければ単純ホールドアウト。
    """
    date_col = None
    for c in ["抽せん日","抽選日","date","draw_date"]:
        if c in df.columns:
            date_col = c; break
    n = len(df)
    if n < 10:
        raise ValueError("行数が少なすぎます。")
    idx_all = np.arange(n)
    if date_col is None:
        cut = int(n * (1 - val_ratio))
        tr_idx, va_idx = idx_all[:cut], idx_all[cut:]
    else:
        dd = pd.to_datetime(df[date_col], errors="coerce")
        order = np.argsort(dd.fillna(pd.Timestamp(1970,1,1)).values)
        cut = int(n * (1 - val_ratio))
        tr_idx = order[:cut]
        va_idx = order[cut:]
    return tr_idx, va_idx

# --------- power calibration (global temperature) ----------
def _power_calibrate(P: np.ndarray, alpha: float, eps: float = 1e-12) -> np.ndarray:
    # P: (n_samples, n_classes)  すべて >=0 かつ 行ごとに 1 に正規化されていること
    P = np.clip(P, eps, 1.0)
    Q = P ** alpha
    Q = Q / Q.sum(axis=1, keepdims=True)
    return Q

def _nll(P: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    # y: 0..C-1 の整数クラス
    P = np.clip(P, eps, 1.0)
    return float(-np.log(P[np.arange(len(y)), y]).mean())

def search_best_alpha(P_val: np.ndarray, y_val: np.ndarray) -> float:
    # ざっくりグリッド → 簡単な微調整
    grid = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
    best_a, best_l = 1.0, _nll(P_val, y_val)
    for a in grid:
        l = _nll(_power_calibrate(P_val, a), y_val)
        if l < best_l:
            best_a, best_l = a, l
    # 近傍微調整
    lo, hi = max(0.2, best_a*0.5), min(5.0, best_a*1.5)
    for a in np.linspace(lo, hi, 13):
        l = _nll(_power_calibrate(P_val, a), y_val)
        if l < best_l:
            best_a, best_l = a, l
    return float(best_a)

# --------- 保存用の軽量ラッパ（joblibで読み出し→predict_probaで即補正可能に） ---------
@dataclass
class PowerCalibratedWrapper:
    base_path: str
    alpha: float

    def predict_proba(self, X):
        from joblib import load
        base = load(self.base_path)
        P = base.predict_proba(X)
        return _power_calibrate(P, self.alpha)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Numbers3 ジョイント1000クラス学習（堅牢キャリブレーション付）")
    ap.add_argument("--history", required=True, help="特徴量CSV")
    ap.add_argument("--models_dir", required=True, help="保存先ディレクトリ")
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    args = ap.parse_args()

    history = Path(args.history)
    out_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load
    df = pd.read_csv(history, encoding="utf-8-sig")
    log(f"history loaded: rows={len(df):,}, cols={len(df.columns)}")

    # 2) target
    y_3 = detect_target_series(df).astype(str).str.zfill(3)
    # 3) features
    X = build_feature_matrix(df, target_col_name=y_3.name if y_3.name else "target")
    log(f"X shape: {X.shape}, feature_cols={X.columns.tolist()[:5]}{'...' if X.shape[1] > 5 else ''}")

    # 4) label encoding to 0..999
    le = make_label_encoder_with_all_1000()
    y = le.transform(y_3)
    present = len(np.unique(y))
    log(f"y classes present: {present} / 1000")

    # 5) temporal split
    tr_idx, va_idx = temporal_split(df, X, y, val_ratio=args.val_ratio)
    X_tr, y_tr = X.iloc[tr_idx].values, y[tr_idx]
    X_va, y_va = X.iloc[va_idx].values, y[va_idx]

    # 6) base estimator
    base = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=args.random_state
    )

    # 7) fit base
    log("fitting base model...")
    base.fit(X_tr, y_tr)
    log("base fit done.")

    # 8) try calibrated CV (prefit)
    calib_meta = {"type": None}
    calibrated_ok = False
    method, nfold = choose_calibration(y_tr)
    try:
        log(f"try CalibratedClassifierCV(prefit): method={method}, folds={nfold}")
        clf = CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
        # 注意：cv="prefit" のとき fit(X_va, y_va) は「検証データ」を渡す
        clf.fit(X_va, y_va)
        calibrated_ok = True
        model_obj = clf
        calib_meta = {"type": "sklearn_calibrated", "method": method, "cv": "prefit"}
        log("calibration (sklearn) succeeded.")
    except Exception as e:
        log(f"calibration failed: {e}")
        # 9) fallback: power calibration
        log("fallback to power calibration (global temperature).")
        P_va = base.predict_proba(X_va)
        alpha = search_best_alpha(P_va, y_va)
        log(f"best alpha = {alpha:.4f}")
        # wrapper オブジェクトを保存して、predict_proba で補正をかけられるようにする
        base_path = out_dir / "base_model.joblib"
        dump(base, base_path)
        wrapper = PowerCalibratedWrapper(base_path=str(base_path), alpha=float(alpha))
        model_obj = wrapper
        calib_meta = {"type": "power", "alpha": float(alpha)}

    # 10) save artifacts
    model_path = out_dir / "joint_model.joblib"
    le_path    = out_dir / "label_encoder.joblib"
    meta_path  = out_dir / "model_meta.json"

    dump(model_obj, model_path)
    dump(le, le_path)

    meta = {
        "task": "numbers3_joint_1000class",
        "history_csv": str(history),
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "feature_names": X.columns.tolist(),
        "label_space": 1000,
        "present_classes": int(present),
        "base_estimator": "RandomForestClassifier",
        "calibration": calib_meta,
        "split": {
            "val_ratio": args.val_ratio,
            "train_size": int(len(tr_idx)),
            "val_size": int(len(va_idx))
        },
        "random_state": args.random_state,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"saved: {model_path}")
    log(f"saved: {le_path}")
    log(f"saved: {meta_path}")
    log("done.")
    

if __name__ == "__main__":
    main()
