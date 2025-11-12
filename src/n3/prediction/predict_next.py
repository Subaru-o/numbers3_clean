# -*- coding: utf-8 -*-
"""
predict_next.py — 安定運用版
- features_v4_cols.json に完全準拠して列を reindex
- 漏洩列を明示除去
- joint_prob = 各桁 argmax の確率の積（predict_proba）
- prediction_history は日付で一意管理（update_modeで append/overwrite/skip）
"""

import argparse
import json
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import pandas as pd
import joblib

TARGETS = ["百の位","十の位","一の位"]
LEAK_DROP = {
    "当せん番号","正解3","正解_百の位","正解_十の位","正解_一の位",
    "百の位","十の位","一の位",
    "予測番号","予測_百の位","予測_十の位","予測_一の位",
    "候補_3桁","候補番号","候補番号3",
}

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

def next_weekday(d: date) -> date:
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d += timedelta(days=1)
    return d

def _load_feature_cols(models_dir: Path) -> dict:
    meta_path = models_dir / "features_v4_cols.json"
    if not meta_path.exists():
        raise SystemExit("[ERR] features_v4_cols.json が見つかりません。再学習してください。")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cols_by_tgt: dict = meta.get("feature_cols_by_target", {})
    if not all(t in cols_by_tgt for t in TARGETS):
        raise SystemExit("[ERR] features_v4_cols.json に列情報がありません。再学習してください。")
    return cols_by_tgt

def _predict_per_digit(model_path: Path, cols: list[str], last: pd.DataFrame) -> np.ndarray:
    model = joblib.load(model_path)
    X_all = last.drop(columns=[c for c in LEAK_DROP if c in last.columns], errors="ignore")
    X_all = coerce_numeric_df(X_all)
    for c in cols:
        if c not in X_all.columns:
            X_all[c] = 0
    X_all = X_all.reindex(columns=cols)
    prob = model.predict_proba(X_all)[0]  # shape (10,)
    return prob

def predict_one(models_dir: Path, df_hist: pd.DataFrame) -> dict:
    cols_by_tgt = _load_feature_cols(models_dir)

    # 履歴を日時ソート → 最新行のみで予測
    df = df_hist.copy()
    if "抽せん日" in df.columns:
        d = pd.to_datetime(df["抽せん日"], errors="coerce")
        df = df.loc[d.notna()].assign(_date=d[d.notna()]).sort_values("_date").drop(columns=["_date"])

    last = df.tail(1).copy()

    probs = {}
    top_digits = {}
    for tgt in TARGETS:
        model_path = models_dir / f"model_{tgt}.joblib"
        if not model_path.exists():
            raise SystemExit(f"[ERR] {model_path} が見つかりません。学習してください。")
        prob = _predict_per_digit(model_path, cols_by_tgt[tgt], last)
        probs[tgt] = prob
        top_digits[tgt] = int(np.argmax(prob))

    p_h, p_t, p_o = probs["百の位"], probs["十の位"], probs["一の位"]
    h, t, o = top_digits["百の位"], top_digits["十の位"], top_digits["一の位"]
    joint_prob = float(p_h[h] * p_t[t] * p_o[o])

    num3 = f"{h}{t}{o}"
    return {"digits": (h,t,o), "number": num3, "joint_prob": joint_prob}

def safe_to3(x) -> str:
    s = pd.to_numeric(pd.Series([x]), errors="coerce")
    if s.isna().iloc[0]: return ""
    return f"{int(s.iloc[0]):03d}"

def _date_iso(x) -> str:
    d = pd.to_datetime(x, errors="coerce")
    return (d.date().isoformat() if pd.notna(d) else "")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)  # 'auto' か パス
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hist_out", required=False)
    ap.add_argument("--price", type=float, default=200.0)
    ap.add_argument("--payout", type=float, default=90000.0)
    ap.add_argument("--update_mode", choices=["skip","overwrite","append"], default="skip",
                    help="prediction_history の同日行が存在する場合の動作（既定: skip）")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    if args.history == "auto":
        hist = find_latest_history_csv(ROOT)
        if hist is None:
            raise SystemExit("[ERR] data/raw の *_Numbers3features.csv が見つかりません。")
    else:
        hist = Path(args.history)

    df_hist = pd.read_csv(hist, encoding="utf-8-sig")

    # 次の平日（履歴の最大日 + 平日補正）
    if "抽せん日" in df_hist.columns:
        dmax = pd.to_datetime(df_hist["抽せん日"], errors="coerce").max()
        next_draw = next_weekday((dmax + pd.Timedelta(days=1)).date()) if pd.notna(dmax) else date.today()
    else:
        next_draw = next_weekday(date.today())

    # 予測
    models_dir = Path(args.models_dir)
    out_path   = Path(args.out)
    pred = predict_one(models_dir, df_hist)

    out = pd.DataFrame([{
        "抽せん日": next_draw.isoformat(),
        "予測番号": int(pred["number"]),
        "百": pred["digits"][0], "十": pred["digits"][1], "一": pred["digits"][2],
        "joint_prob": pred["joint_prob"],
        "expected_payout": float(args.payout) * pred["joint_prob"],
        "price": float(args.price),
        "EV_gross": float(args.payout) * pred["joint_prob"],
        "EV_net": float(args.payout) * pred["joint_prob"] - float(args.price),
    }])
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] next_prediction 書き出し: {out_path}")

    # 履歴追記（安定運用: 既存日付は既定でスキップ）
    if args.hist_out:
        hist_out = Path(args.hist_out)
        new_row = out.copy()
        new_row["候補_3桁"] = new_row["予測番号"].map(safe_to3)
        new_row["抽せん日"] = new_row["抽せん日"].map(_date_iso)

        if hist_out.exists():
            old = pd.read_csv(hist_out, encoding="utf-8-sig")
        else:
            old = pd.DataFrame(columns=[
                "抽せん日","候補_3桁","joint_prob","EV_net","EV_gross",
                "price","expected_payout","予測番号","百","十","一"
            ])

        key = new_row["抽せん日"].iloc[0]
        mask_same = (old["抽せん日"].astype(str) == key) if not old.empty else pd.Series([], dtype=bool)
        exists = (mask_same.any() if not old.empty else False)

        if exists and args.update_mode == "skip":
            print(f"[SKIP] {key} は既に履歴に存在します（update_mode=skip）。")
            merged = old
        elif exists and args.update_mode == "overwrite":
            merged = pd.concat([old.loc[~mask_same], new_row], ignore_index=True)
            print(f"[OK] 履歴を上書きしました: {key}")
        else:  # append または 未存在
            merged = pd.concat([old, new_row], ignore_index=True)

        merged = merged.sort_values("抽せん日").reset_index(drop=True)
        cols = ["抽せん日","候補_3桁","joint_prob","EV_net","EV_gross","price","expected_payout","予測番号","百","十","一"]
        merged = merged[cols]
        merged.to_csv(hist_out, index=False, encoding="utf-8-sig")
        print(f"[OK] prediction_history 更新: {hist_out}")
