# backfill_history.py — prediction_history.csv を全期間バックフィル（学習メタに厳密アライン）
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from joblib import load

def _to_yyyymmdd_int(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.strftime("%Y%m%d").fillna("0").astype(int)

def _weekday_from_date(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.weekday.fillna(0).astype(int)

def _weekday_from_ja(s: pd.Series) -> pd.Series:
    m = {
        "月":0,"月曜":0,"月曜日":0,
        "火":1,"火曜":1,"火曜日":1,
        "水":2,"水曜":2,"水曜日":2,
        "木":3,"木曜":3,"木曜日":3,
        "金":4,"金曜":4,"金曜日":4,
        "土":5,"土曜":5,"土曜日":5,
        "日":6,"日曜":6,"日曜日":6,
        "Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6,
    }
    return s.astype(str).map(lambda x: m.get(x, np.nan)).fillna(0).astype(int)

def _numericize(df: pd.DataFrame, cols_meta: list[str]) -> pd.DataFrame:
    dfc = df.copy()

    # 特殊列を数値化
    if "抽せん日" in dfc.columns:
        # 学習に使うときは int(YYYYMMDD)、学習メタに無いなら後で無視される
        dfc["抽せん日"] = _to_yyyymmdd_int(dfc["抽せん日"])

    if "曜日" in dfc.columns and not np.issubdtype(dfc["曜日"].dtype, np.number):
        tmp = _weekday_from_ja(dfc["曜日"])
        # もし全NaNなら抽せん日から算出
        if "抽せん日" in dfc.columns and (tmp == 0).all() and (dfc["曜日"] != "月").any():
            tmp = _weekday_from_date(dfc["抽せん日"])
        dfc["曜日"] = tmp.astype(int)

    out = pd.DataFrame(index=dfc.index)
    for c in cols_meta:
        if c in dfc.columns:
            out[c] = pd.to_numeric(dfc[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0
    return out.astype(float)

def _argmax_digits(p: np.ndarray) -> int:
    return int(np.argmax(p))

def _ensure_outdir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--price", type=float, default=200)
    ap.add_argument("--payout", type=float, default=90000)
    args = ap.parse_args()

    history = Path(args.history)
    models_dir = Path(args.models_dir)
    out_csv = Path(args.out)

    if not history.exists():
        raise FileNotFoundError(f"history not found: {history}")
    if not models_dir.exists():
        raise FileNotFoundError(f"models_dir not found: {models_dir}")

    # モデル
    m_h = load(models_dir / "model_百の位.joblib")
    m_t = load(models_dir / "model_十の位.joblib")
    m_o = load(models_dir / "model_一の位.joblib")

    # 学習メタ
    meta_path = models_dir / "features_v4_cols.json"
    cols_meta: list[str] | None = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cols_meta = meta.get("feature_cols", meta if isinstance(meta, list) else None)

    # 履歴ロード
    df = pd.read_csv(history, encoding="utf-8-sig")

    # ★ 修正ポイント：まず日付列を datetime にして、列名で sort_values(by=...) する
    if "抽せん日" in df.columns:
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
        df = df.sort_values(by="抽せん日").reset_index(drop=True)

    # 特徴量行列
    if cols_meta is None:
        X_all = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        # 数値化のために文字列の抽せん日を再度含めたいので、X 用に別コピーを作る
        df_for_X = df.copy()
        # df は上で datetime 型にしているので、_numericize 内で YYYYMMDD int へ変換される
        X_all = _numericize(df_for_X, cols_meta)

    rows = []
    for i in range(len(X_all)):
        x = X_all.iloc[[i]]
        # 予測
        try:
            p_h = m_h.predict_proba(x)[0]
            p_t = m_t.predict_proba(x)[0]
            p_o = m_o.predict_proba(x)[0]
        except Exception:
            x = x.astype(float)
            p_h = m_h.predict_proba(x)[0]
            p_t = m_t.predict_proba(x)[0]
            p_o = m_o.predict_proba(x)[0]

        dh, dt, do = _argmax_digits(p_h), _argmax_digits(p_t), _argmax_digits(p_o)
        num3 = int(f"{dh}{dt}{do}")
        joint = float(p_h[dh] * p_t[dt] * p_o[do])
        ev_net = args.payout * joint - args.price

        # 出力用の抽せん日は yyyy-mm-dd 文字列で
        draw_str = ""
        if "抽せん日" in df.columns and pd.notna(df.loc[i, "抽せん日"]):
            draw_str = pd.to_datetime(df.loc[i, "抽せん日"]).date().isoformat()

        rows.append({
            "抽せん日": draw_str,
            "予測番号": num3,
            "百": dh, "十": dt, "一": do,
            "joint_prob": joint,
            "EV_net": ev_net,
            "feature_set": "V4",
            "n_features": X_all.shape[1],
            "model_百": type(m_h).__name__,
            "model_十": type(m_t).__name__,
            "model_一": type(m_o).__name__,
        })

    outdf = pd.DataFrame(rows)

    if args.overwrite or (not out_csv.exists()):
        _ensure_outdir(out_csv)
        outdf.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] wrote: {out_csv}  ({len(outdf)} rows)")
    else:
        old = pd.read_csv(out_csv, encoding="utf-8-sig") if out_csv.exists() else pd.DataFrame()
        merged = pd.concat([old, outdf], ignore_index=True)
        if "抽せん日" in merged.columns:
            merged = merged.drop_duplicates(subset=["抽せん日", "予測番号"], keep="last")
        else:
            merged = merged.drop_duplicates(keep="last")
        _ensure_outdir(out_csv)
        merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] appended: {out_csv}  (total {len(merged)} rows)")

if __name__ == "__main__":
    main()
