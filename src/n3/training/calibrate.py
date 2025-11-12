# src/n3/calibrate.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

class WeekdayIsotonic:
    """dow(0..6) ごとに Isotonic を持つ簡易校正器"""
    def __init__(self):
        self.models = {}

    def fit(self, df_prob: pd.DataFrame, y_true: pd.Series, dow: pd.Series):
        # df_prob: shape (n,10) それぞれ値0..9の確率列
        for d in range(7):
            idx = (dow==d)
            if idx.sum() < 30:
                # データが少ない曜日は平均でスキップ
                continue
            p = df_prob[idx].values.ravel()
            # 正解の one-hot
            y = np.eye(10)[y_true[idx].astype(int)].ravel()
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)
            self.models[d] = iso
        return self

    def transform(self, df_prob: pd.DataFrame, dow: pd.Series) -> pd.DataFrame:
        out = df_prob.copy()
        pflat = df_prob.values
        for i in range(len(df_prob)):
            d = int(dow.iloc[i]) if not pd.isna(dow.iloc[i]) else -1
            if d in self.models:
                # 同一スケールでまとめて校正（0..9の並びを保ったまま）
                row = pflat[i]
                iso = self.models[d]
                out.iloc[i,:] = iso.predict(row)
        # 再正規化
        out = out.div(out.sum(axis=1), axis=0).fillna(0)
        return out
