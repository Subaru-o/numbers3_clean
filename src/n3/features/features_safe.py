# -*- coding: utf-8 -*-
"""
features_safe.py — 当日情報を使わない安全なラグ特徴（window=200デフォ）
- 必須: 抽せん日, 百の位, 十の位, 一の位
- オプション: 曜日, パターン（そのままone-hot）
- すべて shift(1) して当日リークを遮断
- 生成物: 各桁(百/十/一)×各数字0..9の移動出現率 (30列)
"""
from __future__ import annotations
import pandas as pd
import numpy as np

DIGITS = list(range(10))
POS_COLS = [("百の位","h"), ("十の位","t"), ("一の位","o")]

def _one_hot_counts(s: pd.Series) -> pd.DataFrame:
    # s: (0..9) の整数列
    d = {i: (s==i).astype(int) for i in DIGITS}
    return pd.DataFrame(d)

def build_safe_features(df_raw: pd.DataFrame, window: int = 200,
                        use_weekday_onehot: bool = True,
                        use_pattern_onehot: bool = True) -> pd.DataFrame:
    df = df_raw.copy()
    if "抽せん日" not in df.columns:
        raise ValueError("build_safe_features: '抽せん日' が必要です。")
    for base in ["百の位","十の位","一の位"]:
        if base not in df.columns:
            raise ValueError(f"build_safe_features: '{base}' が必要です。")

    # ソート & index整形
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df[df["抽せん日"].notna()].sort_values("抽せん日").reset_index(drop=True)

    # ===== 過去に限定するため全て shift(1) =====
    # 各桁の one-hot を作って -> rolling(window).mean() で移動出現率
    feat_blocks = []
    for col, tag in POS_COLS:
        s = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
        s = s.where(s.between(0,9), other=-1)
        oh = _one_hot_counts(s.clip(lower=0))  # -1は0扱い
        oh = oh.shift(1)  # 当日情報は見ない
        roll = oh.rolling(window=window, min_periods=1).mean()
        roll.columns = [f"{tag}_freq_d{d}_w{window}" for d in DIGITS]
        feat_blocks.append(roll)

    feats = pd.concat(feat_blocks, axis=1)

    # 曜日 / パターン を必要に応じて one-hot（shiftは不要：属性として扱う）
    if use_weekday_onehot and ("曜日" in df.columns):
        wd = pd.get_dummies(df["曜日"].astype(str), prefix="wd", dummy_na=False)
        feats = pd.concat([feats, wd], axis=1)

    if use_pattern_onehot and ("パターン" in df.columns):
        pt = pd.get_dummies(df["パターン"].astype(str), prefix="pt", dummy_na=False)
        feats = pd.concat([feats, pt], axis=1)

    # 数値化・NaN埋め
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # 出力を結合
    out = df.copy()
    out = pd.concat([out, feats], axis=1)
    return out
