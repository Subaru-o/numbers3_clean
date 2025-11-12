# leak_check_quick.py — 履歴CSVの即席リーク検査（要注意列＆同日ジャンプ）
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    args = ap.parse_args()

    hist = Path(args.history)
    df = pd.read_csv(hist, encoding="utf-8-sig")

    # ① 結果由来の“危険キーワード”を含む列名を洗い出し
    bad_kw = re.compile(r"(当|中|hit|result|label|target|payout|的中|予測|rank|num3|正解|当選|抽選|実績)", re.I)
    sus = [c for c in df.columns if bad_kw.search(c)]
    print("▼要注意列:", sus)

    # ② 同日“最後の行”だけが強くなっていないか（頻度・カウント系のジャンプ検査）
    if "抽せん日" not in df.columns:
        print("※ '抽せん日' 列が無いため、同日ジャンプ検査はスキップします。")
        return

    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df[df["抽せん日"].notna()].copy()
    df["date_key"] = df["抽せん日"].dt.date

    # 頻度・回数・rolling など“当日結果が紛れやすい”列を自動推定
    cand_cols = [c for c in df.columns
                 if any(k in c.lower() for k in ["freq","count","rolling","近傍","出現","回数","履歴","累積"])]
    # 数値のみ
    num_cols = [c for c in cand_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        print("※ 頻度/回数系の数値列が見当たらないため、同日ジャンプ検査はスキップします。")
        return

    def day_last_prev_jump(g):
        g = g.sort_values("抽せん日")
        if len(g) < 2:
            return 0.0
        last = g.tail(1)[num_cols].astype(float)
        prev = g.tail(2).head(1)[num_cols].astype(float)
        # “最後の行 − 直前の行” の平均差
        return float((last.values - prev.values).mean())

    jump = df.groupby("date_key").apply(day_last_prev_jump)
    print("▼同日 最後行−直前行 の平均差（+が大きいほど怪しい）")
    print(jump.describe())
    print("上位疑義日TOP5:\n", jump.sort_values(ascending=False).head())

if __name__ == "__main__":
    main()
