# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import pandas as pd

DRAW_WEEKDAYS = {0, 1, 2, 3, 4}  # Mon=0 ... Fri=4

def is_draw_day(d: dt.date | pd.Timestamp) -> bool:
    wd = int(pd.Timestamp(d).weekday())
    return wd in DRAW_WEEKDAYS

def next_draw_day(d: dt.date | pd.Timestamp) -> dt.date:
    """d 当日が土日なら、次の月曜まで進める。平日ならそのまま。"""
    ts = pd.Timestamp(d).normalize()
    while int(ts.weekday()) not in DRAW_WEEKDAYS:
        ts += pd.Timedelta(days=1)
    return ts.date()

def next_draw_day_after(d: dt.date | pd.Timestamp) -> dt.date:
    """d の翌日以降で最初の平日（抽選日）を返す。"""
    ts = pd.Timestamp(d).normalize() + pd.Timedelta(days=1)
    return next_draw_day(ts)

def last_n_draw_days(end_date: dt.date | pd.Timestamp, n: int) -> list[dt.date]:
    """end_date を上限として、直近の抽選日（平日）n個を後ろ向きに取得（昇順）"""
    ts = pd.Timestamp(end_date).normalize()
    # end_date が土日なら直近の金曜まで戻す
    while int(ts.weekday()) not in DRAW_WEEKDAYS:
        ts -= pd.Timedelta(days=1)

    out = []
    while len(out) < n:
        if int(ts.weekday()) in DRAW_WEEKDAYS:
            out.append(ts.date())
        ts -= pd.Timedelta(days=1)
    out.reverse()
    return out

def bday_range(start: dt.date | pd.Timestamp, end: dt.date | pd.Timestamp) -> pd.DatetimeIndex:
    """営業日（平日）だけの日付レンジ"""
    # 日本の祝日は考慮せず「月〜金」のシンプル営業日
    return pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end), freq="B")
