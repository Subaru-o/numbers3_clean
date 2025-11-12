# -*- coding: utf-8 -*-
# src/n3/features.py
# - 基本整形（抽せん日/当せん番号→百/十/一）
# - A/B/C の既存特徴を維持
# - D: 追加特徴 強化版（中期窓30/90、未出現間隔=100/200/300、桁ペア共起=30/50/90/200、合計バケット、曜日×フラグ、月/四半期）

from __future__ import annotations
import pandas as pd
import numpy as np

# 既存窓
WINS_BASE = [10, 50, 200]
# D の追加（中期窓）
WINS_D_EXTRA = [30, 90]
# D: 桁ペア共起の窓（強化）
PAIR_WINS = [30, 50, 90, 200]
# D: 未出現間隔の対象 lookback
GAP_LOOKBACKS = [100, 200, 300]

TRIPLE_COLS = ["百の位", "十の位", "一の位"]

# -------------------------------
# 基本整形
# -------------------------------
def _ensure_base(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # 日付
    if "抽せん日" not in d.columns:
        raise KeyError("抽せん日 列が見つかりません。")
    d["抽せん日"] = pd.to_datetime(d["抽せん日"], errors="coerce")

    # 桁（当せん番号→百/十/一）
    if "当せん番号" in d.columns and not all(c in d.columns for c in TRIPLE_COLS):
        s = d["当せん番号"].astype(str).str.zfill(3)
        d["百の位"] = s.str[0].astype(int)
        d["十の位"] = s.str[1].astype(int)
        d["一の位"] = s.str[2].astype(int)

    # 派生（合計/最大/最小/差）
    if not all(c in d.columns for c in ["合計", "最大", "最小", "最大-最小"]):
        d["合計"] = d[TRIPLE_COLS].sum(axis=1)
        d["最大"] = d[TRIPLE_COLS].max(axis=1)
        d["最小"] = d[TRIPLE_COLS].min(axis=1)
        d["最大-最小"] = d["最大"] - d["最小"]

    # 偶奇・mod・ゾロ目/ダブル
    d["偶数カウント"] = (d[TRIPLE_COLS] % 2 == 0).sum(axis=1)
    d["合計偶奇"] = (d["合計"] % 2 == 0).astype(int)  # 1=偶数, 0=奇数
    d["合計mod3"] = (d["合計"] % 3).astype(int)
    d["ゾロ目"] = ((d["百の位"] == d["十の位"]) & (d["十の位"] == d["一の位"])).astype(int)
    d["ダブル"] = (
        ((d["百の位"] == d["十の位"]) | (d["百の位"] == d["一の位"]) | (d["十の位"] == d["一の位"])) &
        (d["ゾロ目"] == 0)
    ).astype(int)

    # 曜日・月・四半期
    try:
        d["曜日"] = d["抽せん日"].dt.day_name(locale="ja_JP")
    except Exception:
        d["曜日"] = d["抽せん日"].dt.day_name()

    d["月"] = d["抽せん日"].dt.month
    d["四半期"] = ((d["月"] - 1) // 3 + 1).astype(int)

    return d

# -------------------------------
# 既存：直前値と同値の出現回数（過去window）
# -------------------------------
def _rolling_freq_same(s: pd.Series, window: int) -> pd.Series:
    arr = s.to_numpy()
    out = np.full(len(s), np.nan, dtype=float)
    for i in range(1, len(s)):
        j0 = max(0, i - window)
        hist = arr[j0:i]
        out[i] = float((hist == arr[i - 1]).sum())
    return pd.Series(out, index=s.index)

# 既存：直前値と同値の指数減衰重み総和
def _exp_decay_freq(s: pd.Series, lookback: int = 100, alpha: float = 0.06) -> pd.Series:
    arr = s.to_numpy()
    out = np.full(len(s), np.nan, dtype=float)
    for i in range(1, len(s)):
        j0 = max(0, i - lookback)
        hist = arr[j0:i]
        if hist.size == 0:
            continue
        ages = np.arange(hist.size)[::-1]
        w = np.exp(-alpha * ages)
        out[i] = float(w[hist == arr[i - 1]].sum())
    return pd.Series(out, index=s.index)

# -------------------------------
# D: 直前値と同値の「未出現間隔」（最後に出たのは何回前？）
# -------------------------------
def _last_seen_gap_same(s: pd.Series, lookback: int) -> pd.Series:
    arr = s.to_numpy()
    out = np.full(len(s), np.nan, dtype=float)
    for i in range(1, len(s)):
        j0 = max(0, i - lookback)
        hist = arr[j0:i]
        val = arr[i - 1]
        idx = np.where(hist == val)[0]
        if idx.size == 0:
            out[i] = np.nan
        else:
            last_pos = idx[-1]
            out[i] = (len(hist) - 1) - last_pos
    return pd.Series(out, index=s.index)

# -------------------------------
# D: 桁ペアの共起回数（過去windowで「直前ペア」と一致）
# -------------------------------
def _rolling_pair_freq_same(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    pa = a.to_numpy()
    pb = b.to_numpy()
    out = np.full(len(a), np.nan, dtype=float)
    for i in range(1, len(a)):
        j0 = max(0, i - window)
        ha = pa[j0:i]
        hb = pb[j0:i]
        out[i] = float(((ha == pa[i - 1]) & (hb == pb[i - 1])).sum())
    return pd.Series(out, index=a.index)

# -------------------------------
# D: 合計のバケット（one-hot）
# -------------------------------
def _sum_bucket_onehot(sum_series: pd.Series) -> pd.DataFrame:
    s = sum_series
    # 0〜27 を想定：   <10, 10-14, 15-19, 20-24, >=25
    bins = pd.cut(
        s,
        bins=[-1, 9, 14, 19, 24, 100],
        labels=["sum_lt10", "sum_10_14", "sum_15_19", "sum_20_24", "sum_ge25"]
    )
    return pd.get_dummies(bins, prefix="", prefix_sep="").astype(int)

# -------------------------------
# 特徴量生成本体
# -------------------------------
def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_base(df_in)

    # ラグ・差分
    for col in TRIPLE_COLS:
        for k in range(1, 11):
            d[f"{col}_lag{k}"] = d[col].shift(k)
        d[f"{col}_diff1"] = d[col] - d[col].shift(1)
        d[f"{col}_adiff1"] = d[f"{col}_diff1"].abs()

    # 既存の移動頻度（last_freq_w{10,50,200}）
    for col in TRIPLE_COLS:
        for w in WINS_BASE:
            d[f"{col}_last_freq_w{w}"] = _rolling_freq_same(d[col], window=w)

    # 既存の指数減衰頻度
    for col in TRIPLE_COLS:
        d[f"{col}_expfreq_l100_a006"] = _exp_decay_freq(d[col], lookback=100, alpha=0.06)

    # D: 追加の中期窓（30, 90）
    for col in TRIPLE_COLS:
        for w in WINS_D_EXTRA:
            d[f"{col}_last_freq_w{w}"] = _rolling_freq_same(d[col], window=w)

    # D: 未出現間隔（直前値ベース, lookback=100/200/300）
    for col in TRIPLE_COLS:
        for g in GAP_LOOKBACKS:
            d[f"{col}_gap_same_w{g}"] = _last_seen_gap_same(d[col], lookback=g)

    # D: 桁ペア共起（百-十, 十-一, 百-一） for windows 30/50/90/200
    pairs = [("百の位", "十の位"), ("十の位", "一の位"), ("百の位", "一の位")]
    for a, b in pairs:
        for w in PAIR_WINS:
            d[f"{a}_{b}_pair_last_w{w}"] = _rolling_pair_freq_same(d[a], d[b], window=w)

    # D: 合計バケット one-hot
    sum_oh = _sum_bucket_onehot(d["合計"])
    d = pd.concat([d, sum_oh], axis=1)

    # 曜日ダミー
    dow = pd.get_dummies(d["曜日"], prefix="曜日")
    d = pd.concat([d, dow], axis=1)

    # D: 月・四半期ダミー
    mon_oh = pd.get_dummies(d["月"], prefix="月")
    qtr_oh = pd.get_dummies(d["四半期"], prefix="四半期")
    d = pd.concat([d, mon_oh, qtr_oh], axis=1)

    # D: 曜日×フラグの交互作用（軽量）
    if "ゾロ目" in d.columns:
        for c in dow.columns:
            d[f"{c}_x_ゾロ目"] = d[c] * d["ゾロ目"]
    if "合計偶奇" in d.columns:
        for c in dow.columns:
            d[f"{c}_x_合計偶奇"] = d[c] * d["合計偶奇"]

    return d

# 表示用：自然な日本語日付
def format_jp_date(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    ts = pd.Timestamp(ts)
    try:
        wd = ts.day_name(locale="ja_JP")
    except Exception:
        wd = ts.day_name()
    return f"{ts.year}年{ts.month}月{ts.day}日({wd})"
