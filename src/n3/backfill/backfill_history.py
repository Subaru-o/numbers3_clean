# -*- coding: utf-8 -*-
"""
backfill_history.py
  履歴CSV（features）を1行ずつモデルに通して
  ・その日の最も確からしい3桁（候補_3桁）
  ・結合確率（joint_prob）
  ・EV_net（= payout*joint_prob - price）
  ・hit（正解があれば判定）
を作成し、prediction_history.csv に（再計算 or 追記）します。

- 「抽せん日」は必須。『回号』は任意（無くても動く）。
- 学習時の列順（features_v4_cols.json）で予測。欠損列は0で補完。
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from joblib import load

TARGETS = ["百の位", "十の位", "一の位"]

LEAK_DROP = {
    "当せん番号","当選番号","正解3","正解_百の位","正解_十の位","正解_一の位",
    "百の位","十の位","一の位",
    "予測番号","予測_百の位","予測_十の位","予測_一の位",
    "候補_3桁","候補番号","候補番号3",
}

def _mkdir_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, encoding="utf-8-sig")

def _find_latest_history(data_raw: Path) -> Path | None:
    if not data_raw.exists():
        return None
    cands = sorted(data_raw.glob("*_Numbers3features.csv"))
    return cands[-1] if cands else None

def _load_models(models_dir: Path):
    return (
        load(models_dir / "model_百の位.joblib"),
        load(models_dir / "model_十の位.joblib"),
        load(models_dir / "model_一の位.joblib"),
    )

def _load_feature_meta(models_dir: Path):
    meta_p = models_dir / "features_v4_cols.json"
    if not meta_p.exists():
        return None
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if isinstance(meta, dict) and "feature_cols_by_target" in meta:
        by_tgt = meta["feature_cols_by_target"]
        if all(k in by_tgt for k in TARGETS):
            return by_tgt
    return None

def _to_num_frame(row: pd.Series, cols: list[str]) -> pd.DataFrame:
    # row から指定列だけ取り出し、欠けていれば 0 列を追加、すべて数値化→NaNを0に
    x = {}
    for c in cols:
        if c in row.index:
            x[c] = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
        else:
            x[c] = 0.0
    df = pd.DataFrame([x])[cols]
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def _argmax_and_prob(proba1d: np.ndarray) -> Tuple[int, float]:
    cls = int(np.argmax(proba1d))
    return cls, float(proba1d[cls])

def _safe_fmt3(s) -> str:
    v = pd.to_numeric(pd.Series([s]), errors="coerce")
    if v.isna().iloc[0]:
        return ""
    return f"{int(v.iloc[0])%1000:03d}"

def _compute_one_row(row: pd.Series,
                     models: Tuple[Any, Any, Any],
                     feat_by_target: Dict[str, list] | None,
                     price: float,
                     payout: float) -> Dict[str, Any]:
    m_h, m_t, m_o = models

    # 特徴列の決定
    if feat_by_target is not None:
        cols_h = feat_by_target["百の位"]
        cols_t = feat_by_target["十の位"]
        cols_o = feat_by_target["一の位"]
    else:
        # メタが無い場合は大まかに推定（危険だがフォールバック）
        drop = {"回号","抽せん日","当せん番号","当選番号",
                "百の位","十の位","一の位","正解_百の位","正解_十の位","正解_一の位"}
        base_cols = [c for c in row.index if c not in drop]
        cols_h = cols_t = cols_o = base_cols

    xh = _to_num_frame(row, cols_h)
    xt = _to_num_frame(row, cols_t)
    xo = _to_num_frame(row, cols_o)

    ph = m_h.predict_proba(xh)[0]
    pt = m_t.predict_proba(xt)[0]
    po = m_o.predict_proba(xo)[0]

    dh, p_h = _argmax_and_prob(ph)
    dt, p_t = _argmax_and_prob(pt)
    do, p_o = _argmax_and_prob(po)

    cand = f"{dh}{dt}{do}"
    joint = p_h * p_t * p_o
    evnet = payout * joint - price

    # 正解が取れるなら判定（任意）
    hit = False
    if all(k in row.index for k in ["百の位","十の位","一の位"]):
        try:
            ans = f"{int(row['百の位'])}{int(row['十の位'])}{int(row['一の位'])}"
            hit = (cand == _safe_fmt3(ans))
        except Exception:
            hit = False

    out = {
        "抽せん日": pd.to_datetime(row["抽せん日"]).date().isoformat(),
        "候補_3桁": cand,
        "joint_prob": float(joint),
        "EV_net": float(evnet),
        "hit": bool(hit),
    }
    # 回号は任意で付与（無ければ空）
    if "回号" in row.index:
        out["回号"] = row["回号"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="auto")
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--hist_out", required=True)
    ap.add_argument("--price", type=float, default=200.0)
    ap.add_argument("--payout", type=float, default=90000.0)
    ap.add_argument("--data_raw", default="data/raw")
    ap.add_argument("--mode", choices=["recompute","append"], default="recompute",
                   help="recompute: 全期間を再計算し出力を置換 / append: 既存を保ち不足日だけ追加")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    hist_out   = Path(args.hist_out)
    data_raw   = Path(args.data_raw)

    # 履歴CSVの決定
    if args.history == "auto":
        p_hist = _find_latest_history(data_raw)
        if p_hist is None:
            print("[ERR] history CSV not found in", data_raw)
            return 1
    else:
        p_hist = Path(args.history)

    print(f"[INFO] history: {p_hist}")
    print(f"[INFO] models : {models_dir}")

    # モデル & メタ
    try:
        models = _load_models(models_dir)
    except Exception as e:
        print(f"[ERR] load models failed: {e}")
        return 1
    feat_by_target = _load_feature_meta(models_dir)
    if feat_by_target is None:
        print("[WARN] features_v4_cols.json not found; fallback column inference.")

    # 履歴読込（抽せん日は必須 / 回号は任意）
    df = _read_csv(p_hist)
    if "抽せん日" not in df.columns:
        print("[ERR] expected 抽せん日 in input data")
        return 1
    if "回号" not in df.columns:
        print("[INFO] 回号列が無いので空で進めます。")

    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df[df["抽せん日"].notna()].sort_values("抽せん日").reset_index(drop=True)

    # 1行ずつ推論
    rows = []
    for _, row in df.iterrows():
        try:
            rows.append(_compute_one_row(row, models, feat_by_target, args.price, args.payout))
        except Exception as e:
            print(f"[WARN] failed at {row.get('抽せん日')}: {e}")
            dstr = row.get("抽せん日")
            dstr = pd.to_datetime(dstr, errors="coerce")
            dstr = dstr.date().isoformat() if pd.notna(dstr) else ""
            r = {"抽せん日": dstr, "候補_3桁": "", "joint_prob": 0.0,
                 "EV_net": -float(args.price), "hit": False}
            if "回号" in row.index:
                r["回号"] = row["回号"]
            rows.append(r)

    new_df = pd.DataFrame(rows)
    # 値の整形
    new_df["候補_3桁"] = new_df["候補_3桁"].map(_safe_fmt3)
    if "回号" in new_df.columns:
        new_df["回号"] = pd.to_numeric(new_df["回号"], errors="coerce").astype("Int64")

    # 既存とマージ（append モード）
    out_df = new_df.copy()
    if args.mode == "append" and hist_out.exists():
        try:
            old = _read_csv(hist_out)
            # 既存の「抽せん日」型揃え
            if "抽せん日" in old.columns:
                old["抽せん日"] = pd.to_datetime(old["抽せん日"], errors="coerce").dt.date.astype("string")
            # 新規も同様
            out_df["抽せん日"] = pd.to_datetime(out_df["抽せん日"]).dt.date.astype("string")
            # 既存日に対しては old を優先（=上書きしない）
            combined = (
                pd.concat([old, out_df], ignore_index=True)
                  .sort_values("抽せん日")
                  .drop_duplicates(subset=["抽せん日"], keep="first")
            )
            out_df = combined
        except Exception as e:
            print(f"[WARN] append merge failed ({e}); overwrite with recompute result.")

    # 保存
    _mkdir_parent(hist_out)
    out_df.to_csv(hist_out, index=False, encoding="utf-8-sig")
    print(f"[OK] prediction_history updated: {hist_out} (rows={len(out_df)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
