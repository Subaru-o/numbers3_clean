# -*- coding: utf-8 -*-
"""
EVレポートからEV起点のバックフィル履歴を生成するスクリプト（堅牢版）
- --ev-th / --max-per-day（ハイフン）と --ev_th / --max_per_day（アンダースコア）両対応
- EVレポートの列ゆらぎを吸収し、欠けていたら補完（候補番号3, EV列, payout/cost 等）
- groupby.apply の将来挙動に備え include_groups=False を使用
- reset_index 時の重複カラム回避
- 「抽せん日」を必ず復元してから groupby
- サマリー列名 "return" を使わず "revenue" を使用（今回のエラー修正）
出力:
  artifacts/outputs/ev_backfill_history.csv
  artifacts/outputs/ev_backfill_summary.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # .../numbers3_clean
OUTDIR = ROOT / "artifacts" / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

EV_REPORT = OUTDIR / "ev_report.csv"
HIST_OUT = OUTDIR / "ev_backfill_history.csv"
SUMM_OUT = OUTDIR / "ev_backfill_summary.csv"


# -----------------------------
# ユーティリティ
# -----------------------------
def _to3(x) -> str:
    """数値/文字を3桁ゼロ埋め。欠損は空文字。"""
    if pd.isna(x):
        return ""
    try:
        n = int(float(x))
        return f"{n:03d}"
    except Exception:
        s = str(x)
        digits = "".join(ch for ch in s if ch.isdigit())
        return digits.zfill(3) if digits else ""


def _normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    """日付列のゆらぎを吸収して '抽せん日' を保証する。"""
    df = df.copy()
    if "抽せん日" not in df.columns:
        cand = next((c for c in ["抽選日", "date", "抽選日表示", "抽せん日_表示"] if c in df.columns), None)
        if cand:
            df["抽せん日"] = pd.to_datetime(df[cand], errors="coerce").dt.date
        else:
            df["抽せん日"] = pd.NaT
    else:
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce").dt.date
    return df


def _ensure_cols(ev: pd.DataFrame, price_default: int, payout_default: int) -> pd.DataFrame:
    """EVレポートの必須列を揃える。足りなければ生成。"""
    df = ev.copy()

    # 列名ゆらぎ → 正規化
    rename_map = {
        "候補番号": "候補番号3",
        "candidate": "候補番号3",
        "rank": "rank_ev",
        "rank_ev": "rank_ev",
        "抽選日": "抽せん日",
        "date": "抽せん日",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 候補番号3 を作る
    if "候補番号3" not in df.columns:
        can_build = all(c in df.columns for c in ["候補_百の位", "候補_十の位", "候補_一の位"])
        if can_build:
            df["候補番号3"] = (
                pd.to_numeric(df["候補_百の位"], errors="coerce").fillna(-1).astype(int) * 100 +
                pd.to_numeric(df["候補_十の位"], errors="coerce").fillna(-1).astype(int) * 10 +
                pd.to_numeric(df["候補_一の位"], errors="coerce").fillna(-1).astype(int)
            ).astype(int).map(lambda n: f"{n:03d}" if n >= 0 else "")
        else:
            raise KeyError("EVレポートに '候補番号3' または（候補_百の位/十/一）がありません。")
    df["候補番号3"] = df["候補番号3"].map(_to3)

    # 日付保証
    df = _normalize_date(df)

    # EV関連の列補完
    if "EV_net" not in df.columns:
        df["EV_net"] = 0.0
    if "EV_gross" not in df.columns:
        df["EV_gross"] = 0.0
    if "joint_prob" not in df.columns:
        df["joint_prob"] = 0.0

    # payout / cost の補完
    if "cost" not in df.columns:
        df["cost"] = price_default
    if "payout" not in df.columns:
        df["payout"] = payout_default

    # 文字数正規化
    for c in ["EV_net", "EV_gross", "joint_prob", "cost", "payout"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


# -----------------------------
# メイン処理
# -----------------------------
def backfill_from_ev(
    ev_csv: Path = EV_REPORT,
    ev_topk: int = 5,
    ev_threshold: float = 0.0,
    max_per_day: int = 20,
    price: int = 200,
    payout: int = 90000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ev_report.csv を読み、日付ごとに EV 上位をピックして履歴を作成。
    - ev_threshold を超えるものを優先（なければ上位 ev_topk）
    - 1日あたり max_per_day まで
    - EV_net/EV_gross がなければ 0 扱い
    戻り値: (history_df, summary_df)
    """
    if not ev_csv.exists():
        raise FileNotFoundError(f"{ev_csv} が見つかりません。先に ev_report.csv を作成してください。")

    ev = pd.read_csv(ev_csv, encoding="utf-8-sig")
    ev = _ensure_cols(ev, price_default=price, payout_default=payout)

    # 日付で選抜
    def _pick(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["_score"] = pd.to_numeric(g["EV_net"], errors="coerce").fillna(0.0)
        cand = g[g["_score"] >= ev_threshold].sort_values(["_score", "EV_gross", "joint_prob"], ascending=False)
        if cand.empty:
            cand = g.sort_values(["_score", "EV_gross", "joint_prob"], ascending=False)
        # 閾値が高すぎて0件でも、最低限 ev_topk 分は拾う（安全ネット）
        topk_extra = g.sort_values(["_score", "EV_gross", "joint_prob"], ascending=False).head(ev_topk)
        out = pd.concat([cand.head(max_per_day), topk_extra], axis=0).drop_duplicates().head(max_per_day)
        return out.drop(columns=["_score"], errors="ignore")

    df_pick = (
        ev.groupby("抽せん日", group_keys=False)
          .apply(_pick, include_groups=False)
          .reset_index(drop=True)
    )

    # 念のため '抽せん日' を再保証
    df_pick = _normalize_date(df_pick)

    # rank_ev が無ければ再計算
    if "rank_ev" not in df_pick.columns:
        df_pick["rank_ev"] = (
            df_pick.groupby("抽せん日")["EV_net"]
                   .rank(method="first", ascending=False)
                   .astype(int)
        )

    # 正解がある場合のヒット判定
    if "正解3" not in df_pick.columns and all(c in df_pick.columns for c in ["正解_百の位", "正解_十の位", "正解_一の位"]):
        df_pick["正解3"] = (
            pd.to_numeric(df_pick["正解_百の位"], errors="coerce").fillna(-1).astype(int)*100 +
            pd.to_numeric(df_pick["正解_十の位"], errors="coerce").fillna(-1).astype(int)*10 +
            pd.to_numeric(df_pick["正解_一の位"], errors="coerce").fillna(-1).astype(int)
        ).astype(int).map(lambda n: f"{n:03d}" if n >= 0 else "")
    if "正解3" not in df_pick.columns:
        df_pick["正解3"] = ""

    df_pick["hit"] = (df_pick["候補番号3"] == df_pick["正解3"]).astype(int)

    # 出力列
    base_keep = [
        "抽せん日","候補番号3","rank_ev","EV_net","EV_gross","joint_prob",
        "payout","cost","feature_set","model_name","score","正解3","hit"
    ]
    keep_cols = [c for c in base_keep if c in df_pick.columns]
    if "抽せん日" not in keep_cols:
        keep_cols = ["抽せん日"] + keep_cols

    hist = df_pick[keep_cols].copy()
    hist = _normalize_date(hist)

    # ---- サマリー（日付単位） ----
    def _summary(g: pd.DataFrame) -> pd.Series:
        spent = float(pd.to_numeric(g.get("cost", 0), errors="coerce").fillna(0).sum())
        revenue = float(
            pd.to_numeric(g.get("payout", 0), errors="coerce").fillna(0) *
            pd.to_numeric(g.get("hit", 0), errors="coerce").fillna(0)
        .sum())
        return pd.Series({
            "num_bets": int(len(g)),
            "hits": int(pd.to_numeric(g.get("hit", 0), errors="coerce").fillna(0).sum()),
            "spent": spent,
            "revenue": revenue,
        })

    if len(hist) == 0:
        # 空でも列は揃えて返す
        summ = pd.DataFrame(columns=["抽せん日","num_bets","hits","spent","revenue","profit"])
    else:
        summ = (
            hist.groupby("抽せん日", as_index=False)
                .apply(_summary, include_groups=False)
                .reset_index(drop=True)
        )
        summ = _normalize_date(summ)
        for c in ["num_bets","hits","spent","revenue"]:
            if c not in summ.columns:
                summ[c] = 0
        summ["profit"] = pd.to_numeric(summ["revenue"], errors="coerce").fillna(0) - pd.to_numeric(summ["spent"], errors="coerce").fillna(0)

    # 保存
    hist.to_csv(HIST_OUT, index=False, encoding="utf-8-sig")
    summ.to_csv(SUMM_OUT, index=False, encoding="utf-8-sig")
    return hist, summ


def main():
    p = argparse.ArgumentParser(description="EVレポート起点のバックフィル生成")
    p.add_argument("--ev-csv", type=str, default=str(EV_REPORT), help="ev_report.csv のパス")
    p.add_argument("--topk", type=int, default=5, help="閾値未満のとき上位から拾う個数")
    # ハイフン/アンダースコア両対応
    p.add_argument("--ev-th", "--ev_th", dest="ev_th", type=float, default=0.0, help="EV_net 閾値")
    p.add_argument("--max-per-day", "--max_per_day", dest="max_per_day", type=int, default=20, help="1日あたりの最大選定数")
    p.add_argument("--price", type=int, default=200, help="1口の購入額（円）")
    p.add_argument("--payout", type=int, default=90000, help="ストレート当選時の払戻（円）")
    args = p.parse_args()

    ev_csv = Path(args.ev_csv)
    hist, summ = backfill_from_ev(
        ev_csv=ev_csv,
        ev_topk=args.topk,
        ev_threshold=args.ev_th,
        max_per_day=args.max_per_day,
        price=args.price,
        payout=args.payout,
    )
    print(f"[OK] history -> {HIST_OUT}")
    print(f"[OK] summary -> {SUMM_OUT}")
    if len(hist):
        print(hist.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
