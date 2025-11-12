# scripts/build_ev.py
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
DATA_RAW = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "artifacts" / "outputs"

EV_CSV        = OUT_DIR / "ev_report.csv"
NEXT_CSV      = OUT_DIR / "next_prediction.csv"
PRED_HISTORY  = OUT_DIR / "prediction_history.csv"
PRED_HISTORY_TMP = OUT_DIR / "prediction_history.tmp.csv"

DEFAULT_PRICE  = int(os.environ.get("N3_PRICE", "200"))
DEFAULT_PAYOUT = int(os.environ.get("N3_PAYOUT", "90000"))

JST = timezone(timedelta(hours=9))

def fmt3(v: object) -> str:
    s = str(v).strip()
    if s in ("", "None", "nan", "<NA>"):
        return ""
    try:
        return f"{int(float(s))%1000:03d}"
    except Exception:
        return ""

def read_csv_safe(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        return None

def find_latest_history() -> Path | None:
    if not DATA_RAW.exists():
        return None
    cands = list(DATA_RAW.glob("*_Numbers3features.csv"))
    if not cands:
        return None
    return max(cands, key=lambda x: x.stat().st_mtime)

def _next_weekday(d: date) -> date:
    while d.weekday() >= 5:  # 5=Sat 6=Sun
        d += timedelta(days=1)
    return d

def compute_target_draw_date(hist_last_date_str: str) -> str:
    last_d = datetime.strptime(hist_last_date_str, "%Y-%m-%d").date()
    base = _next_weekday(last_d + timedelta(days=1))
    today = datetime.now(JST).date()
    target = base
    if target < today:
        target = _next_weekday(today)
    while target <= last_d:
        target = _next_weekday(target + timedelta(days=1))
    return target.isoformat()

def ensure_joint_prob(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if all(c in df.columns for c in ["p_hundred","p_ten","p_one"]):
        p = (
            pd.to_numeric(df["p_hundred"], errors="coerce").clip(0,1) *
            pd.to_numeric(df["p_ten"],     errors="coerce").clip(0,1) *
            pd.to_numeric(df["p_one"],     errors="coerce").clip(0,1)
        )
    elif "joint_prob" in df.columns:
        p = pd.to_numeric(df["joint_prob"], errors="coerce")
    elif "score" in df.columns:
        p = pd.to_numeric(df["score"], errors="coerce")
    else:
        p = pd.Series([0.0]*len(df), index=df.index)
    df["joint_prob"] = p.fillna(0.0).clip(0,1)
    return df

def _make_date_key(df: pd.DataFrame, col: str = "抽せん日") -> pd.DataFrame:
    if col not in df.columns:
        df[col] = pd.NaT
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df["date_key"] = df[col].dt.date
    return df

def _stable_merge_history(new_hist: pd.DataFrame) -> pd.DataFrame:
    base = read_csv_safe(PRED_HISTORY)
    if base is None or base.empty:
        if "抽せん日" in new_hist.columns:
            new_hist = _make_date_key(new_hist, "抽せん日")
        return new_hist.copy()
    if "抽せん日" in base.columns:     base = _make_date_key(base, "抽せん日")
    if "抽せん日" in new_hist.columns: new_hist = _make_date_key(new_hist, "抽せん日")
    all_cols = list(dict.fromkeys(list(base.columns) + list(new_hist.columns)))
    base2 = base.reindex(columns=all_cols); new2 = new_hist.reindex(columns=all_cols)
    exist_keys = set(base2["date_key"].dropna().unique())
    add_rows = new2[~new2["date_key"].isin(exist_keys)].copy()
    merged = pd.concat([base2, add_rows], ignore_index=True)
    if "抽せん日" in merged.columns:
        merged = merged.sort_values("抽せん日", ascending=False)
    return merged

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hist = find_latest_history()
    if hist is None:
        print("[ERR] *_Numbers3features.csv が見つかりません。先にデータ更新を実行してください。")
        return 2

    # history の最終抽せん日→ターゲット抽せん日
    df_hist = pd.read_csv(hist, encoding="utf-8-sig")
    dmax = pd.to_datetime(df_hist["抽せん日"], errors="coerce").max()
    if pd.isna(dmax):
        print("[ERR] history の抽せん日が読み取れません。")
        return 3
    hist_last_iso = dmax.date().isoformat()
    target_str = compute_target_draw_date(hist_last_iso)
    print(f"[INFO] history last : {hist_last_iso}")
    print(f"[INFO] target draw  : {target_str} (JST today={datetime.now(JST).date()})")

    # TMP 履歴の抽せん日補正 → 安定マージ
    if PRED_HISTORY_TMP.exists():
        tmp = read_csv_safe(PRED_HISTORY_TMP)
        if tmp is not None and not tmp.empty:
            if "抽せん日" not in tmp.columns:
                tmp["抽せん日"] = target_str
            else:
                tmp["抽せん日"] = pd.to_datetime(target_str)
            tmp.to_csv(PRED_HISTORY_TMP, index=False, encoding="utf-8-sig")
            merged = _stable_merge_history(tmp)
            merged.to_csv(PRED_HISTORY, index=False, encoding="utf-8-sig")
            print(f"[OK] prediction_history を安定マージしました: rows={len(merged)}")

    # EV 生成
    df_next = read_csv_safe(NEXT_CSV)
    if df_next is None or df_next.empty:
        print("[ERR] NEXT_CSV が空です。予測ステップで失敗の可能性。")
        return 4

    df_next["抽せん日"] = target_str

    if "候補_3桁" not in df_next.columns:
        if all(c in df_next.columns for c in ["百","十","一"]):
            df_next["候補_3桁"] = (
                pd.to_numeric(df_next["百"], errors="coerce").fillna(0).astype(int).astype(str) +
                pd.to_numeric(df_next["十"], errors="coerce").fillna(0).astype(int).astype(str) +
                pd.to_numeric(df_next["一"], errors="coerce").fillna(0).astype(int).astype(str)
            )
        elif "候補番号" in df_next.columns:
            df_next["候補_3桁"] = pd.to_numeric(df_next["候補番号"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(3)
        elif "番号" in df_next.columns:
            df_next["候補_3桁"] = pd.to_numeric(df_next["番号"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(3)
        else:
            df_next["候補_3桁"] = ""
    df_next["候補_3桁"] = df_next["候補_3桁"].map(fmt3)

    df_next = ensure_joint_prob(df_next)
    jp = pd.to_numeric(df_next["joint_prob"], errors="coerce").fillna(0.0).clip(0, 1)
    price  = float(DEFAULT_PRICE)
    payout = float(DEFAULT_PAYOUT)
    df_next["EV_gross"] = jp * payout
    df_next["EV_net"]   = df_next["EV_gross"] - price
    df_next = df_next.sort_values(["EV_net","EV_gross","joint_prob"], ascending=[False,False,False]).reset_index(drop=True)
    df_next.to_csv(EV_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] EV を生成しました → {EV_CSV} rows={len(df_next)}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
