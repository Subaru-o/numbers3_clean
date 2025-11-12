# -*- coding: utf-8 -*-
"""
predict_v4_next.py — V4モデルで次回予測（桁別確率を出力）
- artifacts/models_V4_XGB/model_百の位.joblib 等を読み込み
- features_v4.build_v4() で直近1行の特徴量ベクトルを構成
- 各桁の確率分布を推論 → next_prediction.csv を出力
"""

from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .features_v4 import build_v4

def _load_model(models_dir: Path, target_name: str):
    p = models_dir / f"model_{target_name}.joblib"
    if not p.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {p}")
    return joblib.load(p)

def _expected_feature_names(model) -> list[str] | None:
    # CalibratedClassifierCV の場合など
    for attr in ("feature_names_in_",):
        if hasattr(model, attr):
            return list(getattr(model, attr))
    # CalibratedClassifierCV で base_estimator に入っている場合
    if hasattr(model, "base_estimator") and hasattr(model.base_estimator, "feature_names_in_"):
        return list(model.base_estimator.feature_names_in_)
    return None

def _to3(x) -> str:
    try:
        return f"{int(pd.to_numeric(x)):03d}"
    except Exception:
        return ""

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history",    required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--out",        required=True)
    args = ap.parse_args()

    history   = Path(args.history)
    modelsdir = Path(args.models_dir)
    out_csv   = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 3 モデル読み込み
    mdl_h = _load_model(modelsdir, "百の位")
    mdl_t = _load_model(modelsdir, "十の位")
    mdl_o = _load_model(modelsdir, "一の位")

    # それぞれが要求する特徴量名を取得（優先）
    names_h = _expected_feature_names(mdl_h)
    names_t = _expected_feature_names(mdl_t)
    names_o = _expected_feature_names(mdl_o)

    # 直近1行の X を構成（モデルに合わせて列を用意）
    Xh, meta = build_v4(history, required_columns=names_h)
    Xt, _    = build_v4(history, required_columns=names_t)
    Xo, _    = build_v4(history, required_columns=names_o)

    # 予測
    proba_h = mdl_h.predict_proba(Xh)[0]
    proba_t = mdl_t.predict_proba(Xt)[0]
    proba_o = mdl_o.predict_proba(Xo)[0]

    # 念のため 0-9 にそろえる（万一ラベル順が崩れている場合の保険）
    def to_fixed10(model, proba):
        # classes_ が [0..9] でない順のときに並び替える
        if hasattr(model, "classes_"):
            idx = np.argsort(model.classes_)
            proba = proba[idx]
            # classes_ が 0..9 以外を含むなら補正
            classes = np.array(model.classes_)[idx]
            fixed = np.zeros(10, dtype=float)
            for cls, p in zip(classes, proba):
                if 0 <= int(cls) <= 9:
                    fixed[int(cls)] = float(p)
            s = fixed.sum()
            return fixed / s if s > 0 else np.ones(10)/10.0
        return proba
    proba_h = to_fixed10(mdl_h, proba_h)
    proba_t = to_fixed10(mdl_t, proba_t)
    proba_o = to_fixed10(mdl_o, proba_o)

    # 最尤の桁
    d_h = int(np.argmax(proba_h))
    d_t = int(np.argmax(proba_t))
    d_o = int(np.argmax(proba_o))
    y3  = f"{d_h}{d_t}{d_o}"

    # 出力テーブル作成
    rec = {
        "抽せん日": "",  # 日付の最終確定は n3.predict_next 側で「次の平日」に上書き
        "予測番号": int(y3),
        "百": d_h, "十": d_t, "一": d_o,
        "model_百": type(mdl_h).__name__,
        "model_十": type(mdl_t).__name__,
        "model_一": type(mdl_o).__name__,
        "feature_set": meta.get("feature_set", "V4"),
        "n_features": meta.get("n_features", 0),
    }
    # 桁別確率
    for i in range(10):
        rec[f"p百の位_{i}"] = float(proba_h[i])
        rec[f"p十の位_{i}"] = float(proba_t[i])
        rec[f"p一の位_{i}"] = float(proba_o[i])

    df_out = pd.DataFrame([rec])
    # ヘルパー列
    df_out["予測番号_3桁"] = df_out["予測番号"].map(_to3)

    # 既存があっても上書き保存（predict_next が抽せん日を上書きし直す）
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] V4 next_prediction を作成: {out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
