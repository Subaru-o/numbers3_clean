# src/n3/build_ev_backfill.py
# prediction_history.csv から joint_prob を復元して ev_backfill.csv を作る
# 使い方:
#   python -m n3.build_ev_backfill --hist artifacts/outputs/prediction_history.csv \
#       --out artifacts/outputs/ev_backfill.csv --price 200 --payout 90000

from __future__ import annotations
import argparse
from pathlib import Path
import re

import pandas as pd

def _num(x) -> int:
    s = pd.to_numeric(pd.Series([x]), errors="coerce").fillna(0)
    return int(s.iloc[0])

def _to3(x) -> str:
    return f"{_num(x):03d}"

def _find_col(cols: list[str], pat: str) -> str | None:
    """列名をゆるく探す（全角・半角スペース/余計な空白を無視、大小も無視）"""
    rx = re.compile(pat, re.IGNORECASE)
    norm = lambda c: re.sub(r"\s+", "", c)  # 空白除去
    for c in cols:
        if rx.fullmatch(norm(c)):
            return c
    return None

def _proba_col(cols: list[str], head: str, d: int) -> str | None:
    """
    例: head='p百の位', d=4 -> 'p百の位_4' を探す（空白のブレにも対応）
    許容パターン: 'p百の位_4', 'p 百 の 位 _4', 'p百位_4' など空白を除去してマッチ
    """
    pat = rf"{re.escape(head)}_{d}"
    return _find_col(cols, pat)

def build_backfill(hist_path: Path, out_path: Path, price: float, payout: float) -> tuple[pd.DataFrame, str]:
    if not hist_path.exists():
        return pd.DataFrame(), f"[ERR] history not found: {hist_path}"

    df = pd.read_csv(hist_path, encoding="utf-8-sig")
    if df.empty:
        return pd.DataFrame(), "[ERR] history file is empty"

    cols = df.columns.tolist()

    # 列候補の検出
    c_date = None
    for c in ["抽せん日","date","draw_date"]:
        if c in cols: c_date = c; break
    if not c_date:
        return pd.DataFrame(), "[ERR] no date column (抽せん日) in history"

    # 予測された各桁（百/十/一）
    c_pred_b = next((c for c in ["予測_百の位","百","pred_b"] if c in cols), None)
    c_pred_t = next((c for c in ["予測_十の位","十","pred_t"] if c in cols), None)
    c_pred_o = next((c for c in ["予測_一の位","一","pred_o"] if c in cols), None)
    if not all([c_pred_b, c_pred_t, c_pred_o]):
        return pd.DataFrame(), "[ERR] predicted digits columns not found (予測_百/十/一の位 など)"

    # 各桁の確率ベクトル p百の位_0..9 / p十の位_0..9 / p一の位_0..9 が必要
    # 列名のブレ（空白）を許容するため、見つけるたびに実列名へ解決
    real_cols_cache: dict[tuple[str,int], str] = {}

    def get_pcol(head: str, d: int) -> str | None:
        key = (head, d)
        if key in real_cols_cache:
            return real_cols_cache[key]
        c = _proba_col(cols, head, d)
        if c: real_cols_cache[key] = c
        return c

    # 出力フレームを作成
    out = pd.DataFrame()
    out["抽せん日"] = pd.to_datetime(df[c_date], errors="coerce").dt.date
    out["候補_百の位"] = df[c_pred_b].map(_num)
    out["候補_十の位"] = df[c_pred_t].map(_num)
    out["候補_一の位"] = df[c_pred_o].map(_num)
    out["候補_3桁"] = (out["候補_百の位"].astype(int).astype(str) +
                     out["候補_十の位"].astype(int).astype(str) +
                     out["候補_一の位"].astype(int).astype(str)).str.zfill(3)

    # 各桁の確率を抽出して joint_prob = pB * pT * pO
    pB, pT, pO = [], [], []
    for i, row in out.iterrows():
        b = int(row["候補_百の位"]); t = int(row["候補_十の位"]); o = int(row["候補_一の位"])

        cb = get_pcol("p百の位", b)
        ct = get_pcol("p十の位", t)
        co = get_pcol("p一の位", o)

        # fallback: 一部のCSVで “p 百の位_4” のような空白が入っている場合に備え、
        #           正規化して再探索（_find_col が空白除去で対応）
        if cb is None:
            cb = _proba_col(cols, "p 百の位", b)
        if ct is None:
            ct = _proba_col(cols, "p 十の位", t)
        if co is None:
            co = _proba_col(cols, "p 一の位", o)

        # 見つからなければ 0 とみなす
        pb = pd.to_numeric(df[cb], errors="coerce").fillna(0).iloc[i] if cb else 0.0
        pt = pd.to_numeric(df[ct], errors="coerce").fillna(0).iloc[i] if ct else 0.0
        po = pd.to_numeric(df[co], errors="coerce").fillna(0).iloc[i] if co else 0.0

        pB.append(float(pb)); pT.append(float(pt)); pO.append(float(po))

    out["prob_百"] = pB
    out["prob_十"] = pT
    out["prob_一"] = pO
    out["joint_prob"] = out["prob_百"] * out["prob_十"] * out["prob_一"]

    # EV を付加
    out["price"]   = float(price)
    out["EV_gross"] = out["joint_prob"] * float(payout)
    out["EV_net"]   = out["EV_gross"] - out["price"]

    # 表示に使う番号など
    out["番号"] = out["候補_3桁"]
    out["feature_set"] = df["feature_set"] if "feature_set" in df.columns else "V4"
    out["model_name"]  = (
        df["model_百"].fillna("") + "/" + df["model_十"].fillna("") + "/" + df["model_一"].fillna("")
        if set(["model_百","model_十","model_一"]).issubset(df.columns)
        else ""
    )

    # 並びを調える
    keep = [c for c in [
        "抽せん日","番号","候補_3桁","候補_百の位","候補_十の位","候補_一の位",
        "prob_百","prob_十","prob_一","joint_prob","EV_gross","EV_net","price",
        "feature_set","model_name"
    ] if c in out.columns]
    out = out[keep].copy()

    # 日付昇順→番号で一応並べる
    out = out.sort_values(["抽せん日","番号"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out, f"[OK] wrote backfill: {out_path} (rows={len(out)})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True, help="path to prediction_history.csv")
    ap.add_argument("--out",  required=True, help="path to write ev_backfill.csv")
    ap.add_argument("--price", type=float, default=200)
    ap.add_argument("--payout", type=float, default=90000)
    args = ap.parse_args()

    df, msg = build_backfill(Path(args.hist), Path(args.out), args.price, args.payout)
    print(msg)
    if not df.empty:
        print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
