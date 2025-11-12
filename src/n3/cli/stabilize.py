# src/n3/stabilize.py
from __future__ import annotations
import os, random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ========== 基本ユーティリティ ==========
def set_all_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # xgboost等にも seed パラメータを渡すと完全再現性が上がる


def fmt3(v: object) -> str:
    s = str(v).strip()
    if s in ("", "None", "nan", "<NA>"):
        return ""
    try:
        return f"{int(float(s))%1000:03d}"
    except Exception:
        return ""


def ensure_candidate3(df: pd.DataFrame) -> pd.DataFrame:
    if "候補_3桁" not in df.columns:
        if "候補番号" in df.columns:
            df["候補_3桁"] = df["候補番号"].map(fmt3)
        elif all(c in df.columns for c in ["百","十","一"]):
            df["候補_3桁"] = (
                pd.to_numeric(df["百"], errors="coerce").fillna(0).astype(int).astype(str) +
                pd.to_numeric(df["十"], errors="coerce").fillna(0).astype(int).astype(str) +
                pd.to_numeric(df["一"], errors="coerce").fillna(0).astype(int).astype(str)
            ).str.zfill(3)
        elif "番号" in df.columns:
            df["候補_3桁"] = pd.to_numeric(df["番号"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(3)
        else:
            df["候補_3桁"] = ""
    df["候補_3桁"] = df["候補_3桁"].map(fmt3)
    return df


def ensure_joint_prob(df: pd.DataFrame) -> pd.DataFrame:
    if "joint_prob" in df.columns:
        s = pd.to_numeric(df["joint_prob"], errors="coerce")
    elif "score" in df.columns:
        s = pd.to_numeric(df["score"], errors="coerce")
    else:
        s = pd.Series([0.0] * len(df))
    df["joint_prob"] = s.fillna(0.0).clip(0, 1)
    return df


def compute_ev(df: pd.DataFrame, price: float, payout: float) -> pd.DataFrame:
    # 期待値列を必ず統一名で付与
    df = ensure_joint_prob(df)
    df["EV_gross"] = df["joint_prob"] * float(payout)
    df["EV_net"]   = df["EV_gross"] - float(price)
    return df


def stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    # 安定ソート（同点時は候補文字列昇順で決定的に）
    df["EV_net"] = pd.to_numeric(df.get("EV_net", 0), errors="coerce").fillna(0.0)
    df = ensure_joint_prob(df)
    df = ensure_candidate3(df)
    return df.sort_values(
        by=["EV_net", "joint_prob", "候補_3桁"],
        ascending=[False, False, True],
        kind="mergesort"  # 安定
    ).reset_index(drop=True)


def load_yesterday_pick(hist_out: str) -> Optional[str]:
    try:
        h = pd.read_csv(hist_out, encoding="utf-8-sig")
        if "抽せん日" not in h.columns:
            return None
        h["抽せん日"] = pd.to_datetime(h["抽せん日"], errors="coerce")
        h = h[h["抽せん日"].notna()].sort_values("抽せん日")
        if "候補_3桁" in h.columns and not h.empty:
            return str(h.iloc[-1]["候補_3桁"]).zfill(3)
    except Exception:
        pass
    return None


def avoid_consecutive_duplicate(df_sorted: pd.DataFrame, prev_pick: Optional[str]) -> pd.DataFrame:
    if not prev_pick:
        return df_sorted
    prev_pick = fmt3(prev_pick)
    alt = df_sorted[df_sorted["候補_3桁"] != prev_pick]
    if alt.empty:
        return df_sorted
    # Top1 を前日と異なるものに差し替える（以降はそのまま）
    new_top = alt.head(1)
    body    = df_sorted.drop(index=new_top.index, errors="ignore")
    out = pd.concat([new_top, body], ignore_index=True)
    return out.reset_index(drop=True)


def _add_date_key(df: pd.DataFrame) -> pd.DataFrame:
    if "抽せん日" in df.columns:
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
        df["date_key"] = df["抽せん日"].dt.date
    else:
        df["date_key"] = pd.NaT
    return df


def append_history_first_write_wins(hist_out: str, df_new: pd.DataFrame) -> None:
    """同じ date_key は上書きしない = 既存を真実として新規だけ追記。"""
    df_new = _add_date_key(df_new.copy())

    try:
        base = pd.read_csv(hist_out, encoding="utf-8-sig")
        base = _add_date_key(base)
    except Exception:
        base = pd.DataFrame(columns=df_new.columns)

    exist = set(
        str(k) for k in base.get("date_key", pd.Series([], dtype=object)).dropna().astype(str).unique()
    )
    add = df_new[~df_new["date_key"].astype(str).isin(exist)].copy()

    all_cols = list(dict.fromkeys(list(base.columns) + list(add.columns)))
    base = base.reindex(columns=all_cols)
    add  = add.reindex(columns=all_cols)

    out = pd.concat([base, add], ignore_index=True)
    out.to_csv(hist_out, index=False, encoding="utf-8-sig")


# ========== まとめてかけるポストプロセス ==========
@dataclass
class PostProcessConfig:
    price: float
    payout: float
    hist_out: str
    avoid_consec_dup: bool = True
    seed: int = 42
    # 追記する固定メタ（任意）
    feature_set: Optional[str] = None
    model_name: Optional[str]  = None
    model_version: Optional[str] = None

def postprocess_predictions(df_scores: pd.DataFrame, cfg: PostProcessConfig) -> pd.DataFrame:
    """予測結果（候補リスト）に対し、確率・EV統一、安定ソート、連続同番号抑止まで実施。"""
    set_all_seeds(cfg.seed)
    df = df_scores.copy()
    df = ensure_candidate3(df)
    df = ensure_joint_prob(df)
    df = compute_ev(df, cfg.price, cfg.payout)
    df = stable_sort(df)

    if cfg.avoid_consec_dup:
        prev = load_yesterday_pick(cfg.hist_out)
        df = avoid_consecutive_duplicate(df, prev)

    # メタ列（無ければ付与）
    if cfg.feature_set is not None and "feature_set" not in df.columns:
        df["feature_set"] = cfg.feature_set
    if cfg.model_name is not None and "model_name" not in df.columns:
        df["model_name"]  = cfg.model_name
    if cfg.model_version is not None and "model_version" not in df.columns:
        df["model_version"] = cfg.model_version

    return df.reset_index(drop=True)
