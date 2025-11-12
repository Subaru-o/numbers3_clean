# -*- coding: utf-8 -*-
"""
n3.features_v3 — Numbers3 用 前処理・特徴量生成（Feature Set V3）
- 目的：Feature Set D を拡張し、短期の“当たりやすさ”を捕まえる追加特徴を提供
- 追加点：
  1) 各桁 lag を 1..20 に拡張
  2) 平滑化系：各桁の rolling mean/std（w=5,10）※ラベルリーク防止で shift(1)
  3) 桁ペア差分：|百-十|, |十-一|, |百-一| の lag1
  4) mod3 系：合計mod3 の lag1、各桁の mod3 ダミー（0/1/2）※lag1
  5) 既存派生：sum_digits_lag1, num_even_lag1, has_consecutive_lag1
  6) 既存頻度：last_freq_w(10,50,200)
  7) 曜日ダミー

呼び出し側：
  - train_evaluate.py で --feature_set V3 を指定して利用
  - predict_next.py でも自動で V3 を選択可能
"""

from __future__ import annotations
from typing import Iterable, Tuple, Dict
import numpy as np
import pandas as pd

# 既存ユーティリティを再利用
from n3.features import (
    detect_schema,          # 日付列 / 数字列（3桁 or 桁別） / 賞金列 自動判定
    add_base_columns,       # 抽せん日・百/十/一 を正規化して付与
    TARGETS,
    WEEKDAY_PREFIX,
)

# ====== 内部ユーティリティ ======
def _add_lag_features(o: pd.DataFrame, cols=TARGETS, lags=range(1, 21)) -> pd.DataFrame:
    df = o.copy()
    for lag in lags:
        for c in cols:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
    return df

def _add_diff_features(o: pd.DataFrame, cols=TARGETS) -> pd.DataFrame:
    """diff1: x - lag1, adiff1: |x - lag1|"""
    df = o.copy()
    for c in cols:
        lag1 = df[f"{c}_lag1"] if f"{c}_lag1" in df.columns else df[c].shift(1)
        df[f"{c}_diff1"] = df[c] - lag1
        df[f"{c}_adiff1"] = (df[f"{c}_diff1"]).abs()
    return df

def _last_value_freq(x: pd.Series) -> float:
    if len(x) == 0:
        return 0.0
    last = x.iloc[-1]
    return float((x == last).mean())

def _add_freq_features(o: pd.DataFrame, cols=TARGETS, windows=(10, 50, 200)) -> pd.DataFrame:
    df = o.copy()
    for w in windows:
        for c in cols:
            df[f"{c}_last_freq_w{w}"] = df[c].rolling(w, min_periods=1).apply(_last_value_freq, raw=False)
    return df

def _add_rolling_features(o: pd.DataFrame, cols=TARGETS, windows=(5, 10)) -> pd.DataFrame:
    """
    各桁の rolling mean/std（w=5,10）を追加。
    ラベルリーク防止のため、統計値は shift(1) して前日までの情報に限定。
    """
    df = o.copy()
    for w in windows:
        for c in cols:
            df[f"{c}_rollmean_w{w}_lag1"] = df[c].rolling(w, min_periods=1).mean().shift(1)
            df[f"{c}_rollstd_w{w}_lag1"]  = df[c].rolling(w, min_periods=1).std().fillna(0.0).shift(1)
    return df

def _add_pair_features(o: pd.DataFrame) -> pd.DataFrame:
    """
    ペア差分（絶対値）の lag1。数字の“離れ具合”を事前情報として使う。
      - |百-十|_lag1, |十-一|_lag1, |百-一|_lag1
    """
    df = o.copy()
    df["pair_diff_百十"] = (df["百の位"] - df["十の位"]).abs().shift(1)
    df["pair_diff_十一"] = (df["十の位"] - df["一の位"]).abs().shift(1)
    df["pair_diff_百一"] = (df["百の位"] - df["一の位"]).abs().shift(1)
    return df

def _add_mod_features(o: pd.DataFrame) -> pd.DataFrame:
    """
    mod3 系の特徴：
      - sum_mod3_lag1：各桁合計の mod3（lag1）
      - 各桁 mod3 の One-Hot（lag1）→ 百/十/一 それぞれ 3 値のダミー（0/1/2）
    """
    df = o.copy()

    # 合計 mod3（lag1）
    sum_mod3 = ((df["百の位"] + df["十の位"] + df["一の位"]) % 3).shift(1)
    df["sum_mod3_lag1"] = sum_mod3

    # 各桁 mod3（lag1）
    for c in TARGETS:
        m = (df[c] % 3).shift(1).astype("Int64")
        for k in (0, 1, 2):
            df[f"{c}_mod3_{k}_lag1"] = (m == k).astype(float)

    return df

def _add_derived(o: pd.DataFrame) -> pd.DataFrame:
    """
    既存の有効派生（Feature Set D と同様）：
      - sum_digits_lag1
      - num_even_lag1
      - has_consecutive_lag1
    """
    df = o.copy()
    df["sum_digits_lag1"] = (df["百の位"] + df["十の位"] + df["一の位"]).shift(1)
    df["num_even_lag1"] = ((df[["百の位","十の位","一の位"]] % 2 == 0).sum(axis=1)).shift(1)
    df["has_consecutive_lag1"] = (
        (abs(df["百の位"] - df["十の位"]) == 1) |
        (abs(df["十の位"] - df["一の位"]) == 1)
    ).shift(1).astype(float)
    return df

def _add_weekday_dummies(o: pd.DataFrame) -> pd.DataFrame:
    """
    曜日ダミー（0..6）を追加。欠落しないように全7列を保証。
    """
    df = o.copy()
    # add_base_columns で weekday 列（0=Mon..6=Sun）を既に作成済み
    dummies = pd.get_dummies(df["weekday"], prefix=WEEKDAY_PREFIX.rstrip("_"))
    for i in range(7):
        col = f"{WEEKDAY_PREFIX}{i}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"{WEEKDAY_PREFIX}{i}" for i in range(7)]]
    return pd.concat([df, dummies], axis=1)

# ====== メイン：V3 特徴量生成 ======
def add_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Set V3：D をベースに短期安定化・連動性捕捉のための特徴を拡張。
    戻り値は「抽せん日」昇順で返す。
    """
    # ベース（抽せん日・百/十/一・weekday 等）
    o = add_base_columns(df)

    # lag を 1..20 まで拡張
    o = _add_lag_features(o, TARGETS, range(1, 21))

    # diff/adiff（lag1）
    o = _add_diff_features(o, TARGETS)

    # 頻度特徴（last value frequency）
    o = _add_freq_features(o, TARGETS, (10, 50, 200))

    # 平滑化（rolling mean/std w=5,10 → shift(1)）
    o = _add_rolling_features(o, TARGETS, (5, 10))

    # ペア差分（lag1）
    o = _add_pair_features(o)

    # mod3 系（合計mod3のlag1、各桁mod3のlag1ダミー）
    o = _add_mod_features(o)

    # 派生（D と同様の3種）
    o = _add_derived(o)

    # 曜日ダミー
    o = _add_weekday_dummies(o)

    # 時系列順
    if "抽せん日" not in o.columns:
        raise KeyError("抽せん日 列が見つかりません。入力CSV/前処理をご確認ください。")
    o = o.sort_values("抽せん日").reset_index(drop=True)

    return o
