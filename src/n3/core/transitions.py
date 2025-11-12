# -*- coding: utf-8 -*-
# src/n3/transitions.py
# - 桁ごとのマルコフ遷移を推定（曜日別）
# - RF確率とブレンドする補助

from __future__ import annotations
import numpy as np
import pandas as pd

def estimate_transition(df: pd.DataFrame, col: str, by_weekday: bool = True):
    s = pd.to_numeric(df[col], errors="coerce")
    if "抽せん日" in df.columns:
        wd = pd.to_datetime(df["抽せん日"], errors="coerce").dt.weekday
    else:
        wd = pd.Series([None] * len(df), index=df.index)

    if by_weekday:
        out = {}
        for w in range(7):
            idx = (wd == w)
            prev = s.shift(1)[idx]
            curr = s[idx]
            mat = np.ones((10, 10)) * 1e-3  # ラプラス平滑化
            for p, c in zip(prev, curr):
                if np.isnan(p) or np.isnan(c):
                    continue
                mat[int(p), int(c)] += 1.0
            mat = (mat.T / mat.sum(axis=1)).T
            out[w] = mat
        return out
    else:
        prev = s.shift(1)
        curr = s
        mat = np.ones((10, 10)) * 1e-3
        for p, c in zip(prev, curr):
            if np.isnan(p) or np.isnan(c):
                continue
            mat[int(p), int(c)] += 1.0
        mat = (mat.T / mat.sum(axis=1)).T
        return mat

def blend_with_transition(proba_rf: np.ndarray, last_digit: int, weekday: int, trans, w: float = 0.30) -> np.ndarray:
    if isinstance(trans, dict):
        mat = trans.get(weekday)
        vec = mat[last_digit] if mat is not None else np.full(10, 0.1)
    else:
        vec = trans[last_digit]
    p = (1.0 - w) * proba_rf + w * vec
    p = p / p.sum()
    return p
