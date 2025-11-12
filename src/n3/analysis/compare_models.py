# -*- coding: utf-8 -*-
# src/n3/compare_models.py
# RF版とXGB版など複数モデルの予測履歴を比較するスクリプト

import pandas as pd
import argparse

def load_truth(features_path):
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

def eval_df(df, ans):
    d = df.merge(ans, on="抽せん日", how="left")
    hb = (d["予測_百の位"] == d["正解_百の位"])
    hj = (d["予測_十の位"] == d["正解_十の位"])
    hi = (d["予測_一の位"] == d["正解_一の位"])
    hit_count = hb.astype(int)+hj.astype(int)+hi.astype(int)
    return {
        "件数": len(d),
        "百一致率": hb.mean(),
        "十一致率": hj.mean(),
        "一一致率": hi.mean(),
        "桁一致平均": (hb.mean()+hj.mean()+hi.mean())/3,
        "2桁一致率": (hit_count==2).mean(),
        "ストレート一致率": (hit_count==3).mean(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf", required=True, help="RF版のprediction_history.csv")
    ap.add_argument("--xgb", required=True, help="XGB版のprediction_history.csv")
    ap.add_argument("--features", required=True, help="Numbers3features.csv")
    args = ap.parse_args()

    # load
    df_rf = pd.read_csv(args.rf, encoding="utf-8-sig")
    df_xgb = pd.read_csv(args.xgb, encoding="utf-8-sig")
    df_rf["抽せん日"] = pd.to_datetime(df_rf["抽せん日"], errors="coerce")
    df_xgb["抽せん日"] = pd.to_datetime(df_xgb["抽せん日"], errors="coerce")
    ans = load_truth(args.features)

    # evaluate
    res_rf = eval_df(df_rf, ans)
    res_xgb = eval_df(df_xgb, ans)

    print("=== RF版 ===")
    print(res_rf)
    print("=== XGB版 ===")
    print(res_xgb)

    # diff
    merged = df_rf[["抽せん日","予測_百の位","予測_十の位","予測_一の位"]].rename(
        columns={"予測_百の位":"RF_百","予測_十の位":"RF_十","予測_一の位":"RF_一"}).merge(
        df_xgb[["抽せん日","予測_百の位","予測_十の位","予測_一の位"]].rename(
            columns={"予測_百の位":"XGB_百","予測_十の位":"XGB_十","予測_一の位":"XGB_一"}),
        on="抽せん日", how="inner"
    )

    diff_mask = (merged[["RF_百","RF_十","RF_一"]].values != merged[["XGB_百","XGB_十","XGB_一"]].values).any(axis=1)
    print(f"差分率: {diff_mask.mean():.2%}")
    print("差分例（最新10件）:")
    print(merged.loc[diff_mask].tail(10))

if __name__ == "__main__":
    main()
