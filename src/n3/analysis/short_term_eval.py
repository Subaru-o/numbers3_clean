# -*- coding: utf-8 -*-
# src/n3/short_term_eval.py
# 予測履歴（RF / XGB など）を正解CSVと突き合わせ、直近N回の一致率を評価（リーク無し）
# - 入力: 各モデルの prediction_history_*.csv と Numbers3features.csv
# - 出力: 端末表示 + （任意）詳細CSV

from __future__ import annotations
import argparse
import pandas as pd
from typing import Dict, Tuple

def load_truth(features_path: str) -> pd.DataFrame:
    feat = pd.read_csv(features_path, encoding="utf-8-sig")
    feat["抽せん日"] = pd.to_datetime(feat["抽せん日"], errors="coerce")
    feat["当せん番号"] = feat["当せん番号"].astype(str).str.zfill(3)
    ans = pd.DataFrame({
        "抽せん日": feat["抽せん日"],
        "正解_百の位": feat["当せん番号"].str[0].astype(int),
        "正解_十の位": feat["当せん番号"].str[1].astype(int),
        "正解_一の位": feat["当せん番号"].str[2].astype(int),
        "曜日": feat["抽せん日"].dt.day_name(),  # 可視化用（日本語化は環境依存なので英語名）
    })
    return ans

def prep_pred(pred_path: str) -> pd.DataFrame:
    d = pd.read_csv(pred_path, encoding="utf-8-sig")
    d["抽せん日"] = pd.to_datetime(d["抽せん日"], errors="coerce")
    d = d.sort_values("抽せん日").reset_index(drop=True)
    # 互換: 予測番号を持つ/持たないに関わらず、桁ごと列で評価する
    req = ["予測_百の位","予測_十の位","予測_一の位"]
    missing = [c for c in req if c not in d.columns]
    if missing:
        raise SystemExit(f"[ERROR] {pred_path} に必要列がありません: {missing}")
    return d[["抽せん日","予測_百の位","予測_十の位","予測_一の位"]].copy()

def eval_window(df_pred: pd.DataFrame, ans: pd.DataFrame, window: int) -> Tuple[Dict, pd.DataFrame]:
    d = df_pred.merge(ans, on="抽せん日", how="inner").sort_values("抽せん日").reset_index(drop=True)
    if len(d) == 0:
        raise SystemExit("[ERROR] 予測と正解の突合結果が空です。日付レンジを確認してください。")
    # 直近ウィンドウ
    tail = d.tail(window).copy() if len(d) >= window else d.copy()
    hb = (tail["予測_百の位"] == tail["正解_百の位"])
    hj = (tail["予測_十の位"] == tail["正解_十の位"])
    hi = (tail["予測_一の位"] == tail["正解_一の位"])
    hit3 = hb.astype(int) + hj.astype(int) + hi.astype(int)

    summary = {
        "件数": int(len(tail)),
        "百一致率": float(hb.mean()),
        "十一致率": float(hj.mean()),
        "一一致率": float(hi.mean()),
        "桁一致平均": float((hb.mean()+hj.mean()+hi.mean())/3),
        "2桁一致率": float((hit3==2).mean()),
        "ストレート一致率": float((hit3==3).mean()),
        "期間_from": tail["抽せん日"].min().date().isoformat(),
        "期間_to": tail["抽せん日"].max().date().isoformat(),
    }

    # 参考: 曜日別（短期ウィンドウ内）
    weekday_tbl = (
        tail.assign(
            百一致=hb.astype(int),
            十一致=hj.astype(int),
            一一致=hi.astype(int),
            二桁=(hit3==2).astype(int),
            三桁=(hit3==3).astype(int),
        )
        .groupby("曜日")[["百一致","十一致","一一致","二桁","三桁"]]
        .mean()
        .sort_index()
    ).reset_index()

    return summary, weekday_tbl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Numbers3features.csv（正解）")
    ap.add_argument("--pred", action="append", required=True,
                    help="比較したい予測履歴CSV。--pred をモデルごとに複数指定（例: RF / XGB）。")
    ap.add_argument("--label", action="append", required=False,
                    help="各 --pred に対応する表示名。未指定ならファイル名を採用。")
    ap.add_argument("--window", type=int, default=200, help="直近ウィンドウ幅（デフォルト200）")
    ap.add_argument("--out", required=False, help="詳細を書き出すCSV（任意）")
    args = ap.parse_args()

    ans = load_truth(args.features)

    preds = [prep_pred(p) for p in args.pred]
    labels = args.label if args.label and len(args.label)==len(preds) else [p.split("/")[-1].split("\\")[-1] for p in args.pred]

    # 共通の抽せん日に合わせる（双方に存在する日だけ残す）
    common_days = set(preds[0]["抽せん日"])
    for d in preds[1:]:
        common_days &= set(d["抽せん日"])
    if not common_days:
        raise SystemExit("[ERROR] 指定した予測CSVの間で共通の抽せん日がありません。")

    common_days = sorted(list(common_days))
    preds = [d[d["抽せん日"].isin(common_days)].sort_values("抽せん日").reset_index(drop=True) for d in preds]

    # 各モデルを評価
    summaries = []
    weekday_blocks = []
    for lbl, dfp in zip(labels, preds):
        s, w = eval_window(dfp, ans, window=args.window)
        s = {"モデル": lbl, **s}
        summaries.append(s)
        w = w.assign(モデル=lbl)
        weekday_blocks.append(w)

    sum_df = pd.DataFrame(summaries).set_index("モデル").round(4)
    print(f"\n=== 直近{args.window}回の評価（共通日付）===")
    print(sum_df.to_string())

    # 差分率（先頭と2番目のみ比較。3本以上指定されたら先頭 vs 各を出す）
    base = preds[0].rename(columns={"予測_百の位":"Base_百","予測_十の位":"Base_十","予測_一の位":"Base_一"})
    for lbl, dfp in zip(labels[1:], preds[1:]):
        merged = base.merge(
            dfp[["抽せん日","予測_百の位","予測_十の位","予測_一の位"]].rename(
                columns={"予測_百の位":"Cmp_百","予測_十の位":"Cmp_十","予測_一の位":"Cmp_一"}),
            on="抽せん日", how="inner"
        )
        diff_mask = (merged[["Base_百","Base_十","Base_一"]].values != merged[["Cmp_百","Cmp_十","Cmp_一"]].values).any(axis=1)
        diff_rate = float(diff_mask.mean())
        print(f"\n差分率（{labels[0]} vs {lbl}）: {diff_rate:.2%}")

    # 任意の詳細出力
    if args.out:
        wd = pd.concat(weekday_blocks, ignore_index=True)
        wd = wd[["モデル","曜日","百一致","十一致","一一致","二桁","三桁"]].sort_values(["モデル","曜日"])
        wd.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] 曜日別の短期評価を出力しました: {args.out}")

if __name__ == "__main__":
    main()
