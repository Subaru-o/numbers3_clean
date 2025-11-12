# tools/repair_features.py
from __future__ import annotations
import pandas as pd, unicodedata as u
from pathlib import Path

SRC = Path(r"data/raw/20201102-20251031_Numbers3features.csv")
DST = Path(r"data/raw_fixed/20201102-20251031_Numbers3features.fixed.csv")
DST.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC, encoding="utf-8-sig")

# ---- 正規化関数（数値候補・文字列候補を一括整形） ----
def normalize_series(s: pd.Series) -> pd.Series:
    # 文字正規化（全角→半角、互換分解を含む）
    s2 = s.astype(str).map(lambda x: u.normalize("NFKC", x))
    # よく混ざる記号の吸収
    s2 = (
        s2.str.replace("−", "-", regex=False)  # 全角マイナス
          .str.replace("￥", "", regex=False)
          .str.replace("¥", "", regex=False)
          .str.replace(",", "", regex=False)
          .str.replace("％", "", regex=False)
          .str.replace("%", "", regex=False)
          .str.replace("/", "-", regex=False)  # 日付系の区切り揺れ
          .str.strip()
    )
    return s2

# ---- 対象列 ----
money_cols = ["ボックス_金額","セットS_金額","セットB_金額"]
count_cols = ["ボックス_口数","セットS_口数","セットB_口数"]
cat_cols   = ["曜日","パターン"]
date_col   = "抽せん日"

# まず存在する列だけに限定
money_cols = [c for c in money_cols if c in df.columns]
count_cols = [c for c in count_cols if c in df.columns]
cat_cols   = [c for c in cat_cols   if c in df.columns]
has_date   = date_col in df.columns

# 口数・金額は数値化（欠損は0埋め）※将来リークを避けるなら学習では使わない想定
for c in money_cols + count_cols:
    s = normalize_series(df[c])
    df[c] = pd.to_numeric(s, errors="coerce").fillna(0).astype("Int64")

# 抽せん日→datetime化、曜日を再計算（もともと曜日列が空でも再生成する）
if has_date:
    s = normalize_series(df[date_col])
    # よくある表記ゆれに対応（YYYY-MM-DD or YYYY/MM/DD）
    df[date_col] = pd.to_datetime(s, errors="coerce", format=None, dayfirst=False, utc=False)
    # 新しい曜日を生成（0=月 … 6=日）
    df["曜日_int"] = df[date_col].dt.weekday  # モデル用の数値
    df["曜日"] = df[date_col].dt.day_name(locale="ja_JP").fillna(df.get("曜日", ""))

# パターン列が空なら再生成（000〜999の重複タイプ）
# 前提：百の位/十の位/一の位 が数値で存在
for col in ["百の位","十の位","一の位"]:
    if col not in df.columns:
        raise SystemExit(f"[ERR] 必要な列がありません: {col}")

def pattern_of_row(a: int,b: int,c: int) -> str:
    if a==b==c: return "ゾロ目"     # 3桁同一
    if (a==b) or (b==c) or (a==c): return "ダブル"   # どれか2桁同一
    return "バラ"                  # 全て異なる

if "パターン" in df.columns:
    # 既存値が全欠損なら上書き再生成
    if df["パターン"].notna().sum()==0:
        df["パターン"] = [pattern_of_row(a,b,c) for a,b,c in df[["百の位","十の位","一の位"]].itertuples(index=False)]
else:
    df["パターン"] = [pattern_of_row(a,b,c) for a,b,c in df[["百の位","十の位","一の位"]].itertuples(index=False)]

# 確認ダンプ（先頭数行）
print(df[["抽せん日","曜日","曜日_int","パターン"] + money_cols + count_cols].head(10).to_string(index=False))

# 保存
df.to_csv(DST, index=False, encoding="utf-8-sig")
print(f"[OK] Fixed CSV saved -> {DST}")
