# -*- coding: utf-8 -*-
# src/n3/compare_many_days.py
# 連続日付レンジで「RF vs XGB」予測の不一致率と精度（翌日正解ベース）を比較

from __future__ import annotations
import argparse
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from n3.features import add_features as make_features  # 既存の特徴量関数を使用

TARGETS = ["百の位", "十の位", "一の位"]

def add_truth(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["抽せん日"] = pd.to_datetime(d["抽せん日"], errors="coerce")
    s = d["当せん番号"].astype(str).str.zfill(3)
    d["百の位"] = s.str[0].astype(int)
    d["十の位"] = s.str[1].astype(int)
    d["一の位"] = s.str[2].astype(int)
    return d.sort_values("抽せん日").reset_index(drop=True)

def load_model_and_cols(models_dir: Path, target: str):
    model = joblib.load(models_dir / f"model_{target}.joblib")
    info = json.load(open(models_dir / f"features_{target}.json", "r", encoding="utf-8"))
    return model, info["features"]

def predict_next_for_date(history: pd.DataFrame, date_str: str, models_dir: Path):
    """指定日までの履歴で最新1行の特徴を作り、その“翌日”を予測する。"""
    cutoff = pd.to_datetime(date_str)
    hist_upto = history[history["抽せん日"] <= cutoff].copy()
    feats = make_features(hist_upto).sort_values("抽せん日").reset_index(drop=True)
    latest = feats.iloc[-1:].copy()

    preds = {}
    for tgt in TARGETS:
        model, cols = load_model_and_cols(models_dir, tgt)
        X = latest[cols].astype(float)
        preds[tgt] = int(model.predict(X)[0])
    pred_str = f"{preds['百の位']}{preds['十の位']}{preds['一の位']}".zfill(3)
    return preds, pred_str

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="*_Numbers3features.csv")
    ap.add_argument("--rf_dir", required=True, help="RFモデルの保存ディレクトリ")
    ap.add_argument("--xgb_dir", required=True, help="XGBモデルの保存ディレクトリ")
    ap.add_argument("--start", required=False, help="範囲開始日 YYYY-MM-DD（省略時は全期間）")
    ap.add_argument("--end", required=False, help="範囲終了日 YYYY-MM-DD（省略時は全期間）")
    args = ap.parse_args()

    hist = pd.read_csv(args.history, encoding="utf-8-sig")
    hist = add_truth(hist)

    # 評価対象となる「基準日」は、翌日データが存在する日まで（最後の1日は除外）
    dates = pd.to_datetime(hist["抽せん日"].unique())
    # ✅ 修正: 並び替えは Index にして sort_values（または np.sort でも可）
    dates = pd.Index(dates).sort_values()

    if args.start:
        dates = dates[dates >= pd.to_datetime(args.start)]
    if args.end:
        dates = dates[dates <= pd.to_datetime(args.end)]
    if len(dates) <= 1:
        raise SystemExit("[ERROR] 日付レンジが短すぎます（翌日が存在しません）。--start/--end を見直してください。")

    dates = dates[:-1]  # 最終日は翌日正解が無いので除外

    rows = []
    diff_cnt = 0

    for dt in dates:
        ds = dt.strftime("%Y-%m-%d")
        preds_rf, s_rf = predict_next_for_date(hist, ds, Path(args.rf_dir))
        preds_xg, s_xg = predict_next_for_date(hist, ds, Path(args.xgb_dir))

        # 翌日の正解（基準日dtの次の抽せん日）
        nxt = hist.loc[hist["抽せん日"] > dt].iloc[0]
        ans = {"百の位": int(nxt["百の位"]), "十の位": int(nxt["十の位"]), "一の位": int(nxt["一の位"])}

        # 桁一致数
        def hit3(p): return sum(int(p[k] == ans[k]) for k in TARGETS)
        h_rf = hit3(preds_rf)
        h_xg = hit3(preds_xg)

        diff = int(s_rf != s_xg)
        diff_cnt += diff

        rows.append({
            "基準日": ds,
            "翌日": pd.to_datetime(nxt["抽せん日"]).date().isoformat(),
            "正解": f"{ans['百の位']}{ans['十の位']}{ans['一の位']}",
            "RF_pred": s_rf, "XGB_pred": s_xg,
            "RF_hit": h_rf, "XGB_hit": h_xg,
            "RF_2桁": int(h_rf == 2), "XGB_2桁": int(h_xg == 2),
            "RF_3桁": int(h_rf == 3), "XGB_3桁": int(h_xg == 3),
            "異なる予測": diff,
        })

    out = pd.DataFrame(rows)

    # 集計
    def agg(prefix: str):
        m = {
            "件数": int(len(out)),
            "桁一致平均": float((out[f"{prefix}_hit"] / 3).mean()),
            "2桁一致率": float(out[f"{prefix}_2桁"].mean()),
            "ストレート一致率": float(out[f"{prefix}_3桁"].mean()),
        }
        return m

    res = pd.DataFrame({
        "RF": agg("RF"),
        "XGB": agg("XGB"),
    }).T

    diff_rate = float(out["異なる予測"].mean())

    print("=== 集計（基準日→翌日の的中評価）===")
    print(res.round(4).to_string())
    print(f"\n不一致率（RFとXGBの予測が異なる割合）: {diff_rate:.2%}")

    # 直近200件の比較（短期傾向）
    if len(out) >= 200:
        tail = out.tail(200)
        rf_2 = float(tail["RF_2桁"].mean())
        xg_2 = float(tail["XGB_2桁"].mean())
        print("\n=== 直近200件の2桁一致率 ===")
        print(f"RF:  {rf_2:.2%}")
        print(f"XGB: {xg_2:.2%}")

    # 直近で“勝敗が分かれる”例（末尾20件）
    diff_tail = out[out["異なる予測"] == 1].tail(20)
    if not diff_tail.empty:
        print("\n=== 直近で予測が食い違った例（末尾20件）===")
        print(diff_tail[["基準日","翌日","正解","RF_pred","XGB_pred","RF_hit","XGB_hit"]].to_string(index=False))

if __name__ == "__main__":
    main()
