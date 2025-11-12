# -*- coding: utf-8 -*-
"""
train_joint_force1000.py — 安全ラグ特徴で 1000クラス学習
- 当せん番号(0..999) をラベルに採用 (--label-col 省略可: 百/十/一から合成)
- features_safe.build_safe_features(window=200) を内部で実行
- 危険列や識別子は自然に除外（特徴は生成列が中心）
- 欠けラベルはゼロ特徴ダミーを極小重みで補完 -> classes_=1000 保証
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from .features_safe import build_safe_features

def _pick(df: pd.DataFrame, names: list[str]) -> str|None:
    for n in names:
        if n in df.columns: return n
    return None

def _build_label(df: pd.DataFrame, label_col: str|None) -> np.ndarray:
    if label_col:
        if label_col not in df.columns:
            raise ValueError(f"label column '{label_col}' not found")
        y = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
        if y.isna().any(): raise ValueError(f"label '{label_col}' has NaN")
        y = y.astype(int).values
    else:
        h = pd.to_numeric(df[_pick(df,["百","百の位","百位"])], errors="coerce").fillna(0).astype(int).clip(0,9)
        t = pd.to_numeric(df[_pick(df,["十","十の位","十位"])], errors="coerce").fillna(0).astype(int).clip(0,9)
        o = pd.to_numeric(df[_pick(df,["一","一の位","一位"])], errors="coerce").fillna(0).astype(int).clip(0,9)
        y = (100*h + 10*t + o).values
    if (y<0).any() or (y>999).any(): raise ValueError("y outside 0..999")
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--models_out", required=True)
    ap.add_argument("--label-col", default="当せん番号")
    ap.add_argument("--window", type=int, default=200, help="rolling window for safe features")
    ap.add_argument("--allow-pattern", action="store_true", help="パターンもone-hotに含める")
    args = ap.parse_args()

    hist = Path(args.history); outdir = Path(args.models_out); outdir.mkdir(parents=True, exist_ok=True)
    df0 = pd.read_csv(hist, encoding="utf-8-sig")
    df0["抽せん日"] = pd.to_datetime(df0["抽せん日"], errors="coerce")
    df0 = df0[df0["抽せん日"].notna()].sort_values("抽せん日").reset_index(drop=True)

    # 1) ラベル
    y_raw = _build_label(df0, args.label_col)

    # 2) 安全特徴の自動生成
    df_feat = build_safe_features(df0, window=args.window,
                                  use_weekday_onehot=True,
                                  use_pattern_onehot=args.allow_pattern)

    # 特徴は生成列のみ（元の桁や当選などは含まれない）
    reserved = {"抽せん日","百の位","十の位","一の位","曜日","パターン"}
    feat_cols = [c for c in df_feat.columns if c not in reserved and c.startswith(("h_freq_","t_freq_","o_freq_","wd_","pt_"))]
    if not feat_cols: raise RuntimeError("no generated features found")

    X = df_feat[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0.0).astype(float)

    # 3) 1000クラス保証
    le = LabelEncoder(); le.classes_ = np.arange(1000)
    y = le.transform(y_raw)
    present = set(np.unique(y_raw)); missing = sorted(set(range(1000)) - present)
    if missing:
        print(f"[AUG] add dummy for {len(missing)} missing labels")
        X_dummy = pd.DataFrame(np.zeros((len(missing), X.shape[1]), dtype=float), columns=X.columns)
        y_dummy = le.transform(np.array(missing, dtype=int))
        X = pd.concat([X, X_dummy], axis=0, ignore_index=True)
        y = np.concatenate([y, y_dummy], axis=0)
        w = np.concatenate([np.ones(len(y_raw)), np.full(len(missing), 1e-6)], axis=0)
    else:
        w = None

    # 4) 学習
    clf = RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=-1, random_state=42)
    clf.fit(X, y, sample_weight=w)

    # 5) 保存
    meta = {"features": feat_cols, "feature_names": feat_cols, "window": args.window, "allow_pattern": bool(args.allow_pattern)}
    joblib.dump({"model": clf, "meta": meta}, outdir/"joint_model.joblib")
    joblib.dump(le, outdir/"label_encoder.joblib")
    (outdir/"model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6) 診断
    classes = getattr(clf, "classes_", None)
    if classes is not None:
        print(f"[OK] trained. classes_ size = {len(np.unique(classes))} (expected 1000)")
    print(f"[OK] saved to: {outdir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
