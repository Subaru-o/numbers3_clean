# -*- coding: utf-8 -*-
"""
make_ev_from_next.py — next_prediction.csv をそのまま EV に変換
- joint_prob の“モデル出力”をそのまま使う（固定の 0.016/0.216 にしない）
- 候補_3桁→番号/百十一 を補完
- 抽選日は history から「翌営業日」を推定して付与（なければ today）

使い方:
  python -m n3.make_ev_from_next \
    --next_pred artifacts/outputs/next_prediction.csv \
    --history   data/raw/XXXX_Numbers3features.csv \
    --out       artifacts/outputs/ev_report.csv \
    --price 200 --payout 90000
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import timedelta, date

import numpy as np
import pandas as pd


def fmt3(x) -> str:
    try:
        return f"{int(float(x))%1000:03d}"
    except Exception:
        return ""


def _next_weekday_from_history(history_csv: Path | None) -> date:
    if not history_csv or not history_csv.exists():
        return date.today()
    df = pd.read_csv(history_csv, encoding="utf-8-sig", usecols=lambda c: c == "抽せん日")
    if "抽せん日" not in df.columns or df.empty:
        return date.today()
    dmax = pd.to_datetime(df["抽せん日"], errors="coerce").max()
    if pd.isna(dmax):
        return date.today()
    d = dmax.date()
    while True:
        d = d + timedelta(days=1)
        if d.weekday() < 5:
            return d


def _ensure_num3_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "候補_3桁" in out.columns:
        out["候補_3桁"] = out["候補_3桁"].map(fmt3)
    elif "番号" in out.columns:
        out["候補_3桁"] = out["番号"].map(fmt3)
    elif "候補番号3" in out.columns:
        out["候補_3桁"] = out["候補番号3"].map(fmt3)
    elif all(c in out.columns for c in ["百","十","一"]):
        out["候補_3桁"] = (
            pd.to_numeric(out["百"], errors="coerce").fillna(0).astype(int).astype(str) +
            pd.to_numeric(out["十"], errors="coerce").fillna(0).astype(int).astype(str) +
            pd.to_numeric(out["一"], errors="coerce").fillna(0).astype(int).astype(str)
        )
        out["候補_3桁"] = out["候補_3桁"].map(fmt3)
    else:
        out["候補_3桁"] = ""

    # 百/十/一/番号 も整えておく
    out["番号"] = pd.to_numeric(out["候補_3桁"], errors="coerce").fillna(0).astype(int)
    out["百"] = out["候補_3桁"].str[0].fillna("0").astype(int)
    out["十"] = out["候補_3桁"].str[1].fillna("0").astype(int)
    out["一"] = out["候補_3桁"].str[2].fillna("0").astype(int)
    return out


def _ensure_joint_prob(df: pd.DataFrame) -> pd.Series:
    # p_hundred×p_ten×p_one があれば優先、無ければ joint_prob、score の順
    if all(c in df.columns for c in ["p_hundred","p_ten","p_one"]):
        p = (pd.to_numeric(df["p_hundred"], errors="coerce").clip(0,1) *
             pd.to_numeric(df["p_ten"],     errors="coerce").clip(0,1) *
             pd.to_numeric(df["p_one"],     errors="coerce").clip(0,1))
    elif "joint_prob" in df.columns:
        p = pd.to_numeric(df["joint_prob"], errors="coerce").clip(0,1)
    elif "score" in df.columns:
        p = pd.to_numeric(df["score"], errors="coerce").clip(0,1)
    else:
        p = pd.Series(0.0, index=df.index)
    return p.fillna(0.0).clip(0,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--next_pred", required=True)
    ap.add_argument("--history", required=False, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--price", type=float, default=200)
    ap.add_argument("--payout", type=float, default=90000)
    args = ap.parse_args()

    next_csv   = Path(args.next_pred)
    history_csv= Path(args.history) if args.history else None
    out_csv    = Path(args.out); out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not next_csv.exists():
        raise SystemExit(f"[ERR] next_prediction がありません: {next_csv}")

    df = pd.read_csv(next_csv, encoding="utf-8-sig")
    df = _ensure_num3_cols(df)
    p  = _ensure_joint_prob(df)

    # 抽選日の推定（履歴の翌営業日）
    draw_date = _next_weekday_from_history(history_csv)

    ev_gross = p * float(args.payout)
    ev_net   = ev_gross - float(args.price)

    out = pd.DataFrame({
        "抽せん日": draw_date,
        "番号": df["番号"],
        "joint_prob": p,
        "EV_net": ev_net,
        "EV_gross": ev_gross,
        "百": df["百"], "十": df["十"], "一": df["一"],
        "候補_3桁": df["候補_3桁"],
        "price": float(args.price),
        "expected_payout": p * float(args.payout),
    }).sort_values(["EV_net","joint_prob"], ascending=[False, False])

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] EV を作成: {out_csv}")
    print(out.head(5)[["抽せん日","番号","EV_net","joint_prob"]].to_string(index=False))


if __name__ == "__main__":
    main()
