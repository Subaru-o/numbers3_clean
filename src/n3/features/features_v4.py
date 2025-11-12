# src/n3/features_v4.py
# V4 特徴量: 安全に作れる軽量版
# - add_features_v4(df) の提供（train_evaluate_v4 が import）
# - 互換のため build_v4 = add_features_v4 もエクスポート
# - リーク回避のため、すべて lag=1 以上にしてからロール計算

from __future__ import annotations
import pandas as pd
import numpy as np

def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    need = ["抽せん日", "百の位", "十の位", "一の位"]
    for c in need:
        if c not in df.columns:
            # 列がない場合でも動くように0で埋める
            df[c] = 0
    return df

def _weekday_dummies(s_dt: pd.Series) -> pd.DataFrame:
    # 月=0..日=6 を one-hot（Numbers3は平日開催が多いので 0..4 が主）
    wd = s_dt.dt.weekday
    out = pd.get_dummies(wd, prefix="weekday", dtype=int)
    # 欠ける列を追加しておく（0..6）
    for i in range(7):
        col = f"weekday_{i}"
        if col not in out.columns:
            out[col] = 0
    return out

def _add_digit_lags(df: pd.DataFrame, col: str, lags=(1,2,3,5,8,10,12,15,17,18,19,20)) -> pd.DataFrame:
    for L in lags:
        df[f"{col}_lag{L}"] = _to_int(df[col]).shift(L)
    return df

def _add_rolls(df: pd.DataFrame, col: str, windows=(5,10,20)) -> pd.DataFrame:
    s = _to_int(df[col]).shift(1)  # lag=1 してから roll（リーク回避）
    for w in windows:
        df[f"{col}_rollmean_w{w}_lag1"] = s.rolling(w, min_periods=max(2, w//2)).mean()
        df[f"{col}_rollstd_w{w}_lag1"]  = s.rolling(w, min_periods=max(2, w//2)).std()
    return df

def _safe_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return _to_int(a) - _to_int(b)

def add_features_v4(df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    入力: 履歴テーブル（少なくとも 抽せん日, 百の位, 十の位, 一の位 があるのが望ましい）
    出力: 上記＋各種特徴量列を付与した DataFrame
    """
    df = df_hist.copy()
    df = _ensure_columns(df)

    # 基本整形
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df.sort_values("抽せん日").reset_index(drop=True)

    # 便利な数値列
    df["百"] = _to_int(df["百の位"])
    df["十"] = _to_int(df["十の位"])
    df["一"] = _to_int(df["一の位"])

    # 集約的な原始特徴
    df["sum_digits"]   = df["百"] + df["十"] + df["一"]
    df["max_digit"]    = df[["百","十","一"]].max(axis=1)
    df["min_digit"]    = df[["百","十","一"]].min(axis=1)
    df["最大-最小"]      = df["max_digit"] - df["min_digit"]
    df["sum_mod3"]     = (df["sum_digits"] % 3).astype(int)

    # weekday dummies
    wkd = _weekday_dummies(df["抽せん日"])
    df = pd.concat([df, wkd], axis=1)

    # ラグ・ロール（各桁）
    for col in ["百","十","一"]:
        df = _add_digit_lags(df, col)
        df = _add_rolls(df, col)

    # ペア差分（lag1）
    df["pair_diff_百十_lag1"] = _safe_diff(df["百"], df["十"]).shift(1)
    df["pair_diff_百一_lag1"] = _safe_diff(df["百"], df["一"]).shift(1)
    df["pair_diff_十一_lag1"] = _safe_diff(df["十"], df["一"]).shift(1)

    # “同じかどうか”系（lag1）
    df["十_same_as_一_lag1"] = (df["十"].shift(1) == df["一"].shift(1)).astype(int)
    df["百_same_as_十_lag1"] = (df["百"].shift(1) == df["十"].shift(1)).astype(int)
    df["百_same_as_一_lag1"] = (df["百"].shift(1) == df["一"].shift(1)).astype(int)

    # よく参照されていた列名（過去との互換）
    # - “回号”があればそのまま。なければ連番を作る
    if "回号" not in df.columns:
        df["回号"] = np.arange(1, len(df) + 1)

    # 欠損は学習側で無視できるように float に
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

# 互換: 以前のコードが build_v4 を import していた場合にも対応
build_v4 = add_features_v4

# 参考: 学習に使う候補（あくまで目安。学習側が select してもOK）
def list_feature_columns(df: pd.DataFrame) -> list[str]:
    blacklist = {"抽せん日","回号","百の位","十の位","一の位","百","十","一"}
    feats = [c for c in df.columns if c not in blacklist]
    return feats
