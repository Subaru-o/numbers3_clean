# -*- coding: utf-8 -*-
# src/n3/compare_from_histories.py
# 既存の予測履歴CSV（RF版/XGB版）を使って、リーク無しで比較する

import argparse
import pandas as pd

def load_truth(features_path: str) -> pd.DataFrame:
    feat = pd.read_csv(features_path, encoding="utf-8-sig")
    feat["抽せん日"] = pd.to_datetime(feat["抽せん日"], errors="coerce")
    feat["当せん番号"] = feat["当せん番号"].astype(str).str.zfill(3)
    ans = pd.DataFrame({
        "抽せん日": feat["抽せん日"],
        "正解_百の位": feat["当せん番号"].str[0].astype(int),
        "正解_十の位": feat["当せん番号"].str[1].astype(int),
        "正解_一の位": feat["当せん番号"].str[2].astype(int),
    })
    return ans

def eval_df(df: pd.DataFrame, ans: pd.DataFrame) -> dict:
    d = df.merge(ans, on="抽せん日", how="left")
    hb = (d["予測_百の位"] == d["正解_百の位"])
    hj = (d["予測_十の位"] == d["正解_十の位"])
    hi = (d["予測_一の位"] == d["正解_一の位"])
    hit_count = hb.astype(int) + hj.astype(int) + hi.astype(int)
    return {
        "件数": int(len(d)),
        "百一致率": float(hb.mean()),
        "十一致率": float(hj.mean()),
        "一一致率": float(hi.mean()),
        "桁一致平均": float((hb.mean()+hj.mean()+hi.mean())/3),
        "2桁一致率": float((hit_count==2).mean()),
        "ストレート一致率": float((hit_count==3).mean()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf_hist", required=True, help="RF版 prediction_history.csv")
    ap.add_argument("--xgb_hist", required=True, help="XGB版 prediction_history.csv")
    ap.add_argument("--features", required=True, help="Numbers3features.csv（正解）")
    args = ap.parse_args()

    df_rf = pd.read_csv(args.rf_hist, encoding="utf-8-sig")
    df_xg = pd.read_csv(args.xgb_hist, encoding="utf-8-sig")
    df_rf["抽せん日"] = pd.to_datetime(df_rf["抽せん日"], errors="coerce")
    df_xg["抽せん日"] = pd.to_datetime(df_xg["抽せん日"], errors="coerce")

    ans = load_truth(args.features)

    # 同一日のみに揃える（両者に存在する抽せん日だけを対象）
    common_days = sorted(set(df_rf["抽せん日"]).intersection(set(df_xg["抽せん日"])))
    if not common_days:
        raise SystemExit("[ERROR] 共通の抽せん日がありません。履歴CSVを見直してください。")

    df_rf = df_rf[df_rf["抽せん日"].isin(common_days)].copy()
    df_xg = df_xg[df_xg["抽せん日"].isin(common_days)].copy()

    res_rf = eval_df(df_rf, ans)
    res_xg = eval_df(df_xg, ans)

    print("=== RF（履歴CSVから評価） ===")
    print({k: (round(v,4) if isinstance(v, float) else v) for k, v in res_rf.items()})
    print("=== XGB（履歴CSVから評価） ===")
    print({k: (round(v,4) if isinstance(v, float) else v) for k, v in res_xg.items()})

    # 差分率（同日の予測が異なる割合）
    merged = df_rf[["抽せん日","予測_百の位","予測_十の位","予測_一の位"]].rename(
        columns={"予測_百の位":"RF_百","予測_十の位":"RF_十","予測_一の位":"RF_一"}).merge(
        df_xg[["抽せん日","予測_百の位","予測_十の位","予測_一の位"]].rename(
            columns={"予測_百の位":"XGB_百","予測_十の位":"XGB_十","予測_一の位":"XGB_一"}),
        on="抽せん日", how="inner"
    )
    diff_mask = (merged[["RF_百","RF_十","RF_一"]].values != merged[["XGB_百","XGB_十","XGB_一"]].values).any(axis=1)
    print(f"差分率: {diff_mask.mean():.2%}")
    if diff_mask.any():
        print("差分例（末尾10件）:")
        print(merged.loc[diff_mask].tail(10).to_string(index=False))

if __name__ == "__main__":
    main()
