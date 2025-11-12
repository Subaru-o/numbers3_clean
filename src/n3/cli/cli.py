# src/n3/cli.py  — EVレポート生成（恒久対応：必ず最大1000行に展開）
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np


def _read_next_pred(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"next_prediction.csv が見つかりません: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _pick_date(df_next: pd.DataFrame) -> str:
    # next_prediction の抽せん日をそのまま継承
    for col in ["抽せん日", "date", "draw_date"]:
        if col in df_next.columns and df_next[col].notna().any():
            return str(pd.to_datetime(df_next[col].iloc[0]).date())
    # どれも無ければ今日の日付
    return str(pd.Timestamp.today().date())


def _load_digit_probs(df_next: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    next_prediction.csv から各桁(百/十/一)の確率分布（長さ10のベクトル）を取り出す。
    p百の位_0..9 / p十の位_0..9 / p一の位_0..9 が理想。
    無い場合は、予測番号（予測_百の位 等 or 百/十/一）から「ピーク+薄く分散」の仮分布を作る。
    """
    def _extract(prefix: str) -> np.ndarray | None:
        cols = [c for c in df_next.columns if c.startswith(prefix)]
        if len(cols) == 10:
            arr = df_next[cols].iloc[0].astype(float).to_numpy()
            s = arr.sum()
            if s > 0:
                return arr / s
        return None

    p_h = _extract("p百の位_")
    p_t = _extract("p十の位_")
    p_o = _extract("p一の位_")

    if p_h is not None and p_t is not None and p_o is not None:
        return p_h, p_t, p_o

    # ---- フォールバック：予測数字があれば「0.6 を当たり目に、残り均等に」
    def _peak_vec(peak_digit: int, peak: float = 0.6) -> np.ndarray:
        v = np.full(10, (1.0 - peak) / 9.0, dtype=float)
        if 0 <= peak_digit <= 9:
            v[peak_digit] = peak
        return v / v.sum()

    # 候補となる桁
    cand_h = None
    cand_t = None
    cand_o = None

    # 1) 予測_百/十/一の位
    for name, var in [("予測_百の位", "h"), ("予測_十の位", "t"), ("予測_一の位", "o")]:
        if name in df_next.columns and pd.notna(df_next[name].iloc[0]):
            val = int(float(df_next[name].iloc[0]))
            if var == "h": cand_h = val
            if var == "t": cand_t = val
            if var == "o": cand_o = val

    # 2) 百/十/一 そのもの
    for name, var in [("百", "h"), ("十", "t"), ("一", "o")]:
        if (locals().get(f"cand_{var}") is None) and name in df_next.columns and pd.notna(df_next[name].iloc[0]):
            val = int(float(df_next[name].iloc[0]))
            if var == "h": cand_h = val
            if var == "t": cand_t = val
            if var == "o": cand_o = val

    # 3) 予測番号（例: 729）から分解
    if (cand_h is None or cand_t is None or cand_o is None):
        for name in ["予測番号", "番号", "pred", "num"]:
            if name in df_next.columns and isinstance(df_next[name].iloc[0], (int, float, str)):
                s = str(df_next[name].iloc[0]).strip()
                s = ''.join([c for c in s if c.isdigit()])
                if len(s) == 3:
                    cand_h = cand_h if cand_h is not None else int(s[0])
                    cand_t = cand_t if cand_t is not None else int(s[1])
                    cand_o = cand_o if cand_o is not None else int(s[2])
                break

    # 最後の手段：一様分布
    if cand_h is None: cand_h = 0
    if cand_t is None: cand_t = 0
    if cand_o is None: cand_o = 0

    return _peak_vec(cand_h), _peak_vec(cand_t), _peak_vec(cand_o)


def _expand_all_numbers(draw_date: str,
                        p_h: np.ndarray, p_t: np.ndarray, p_o: np.ndarray,
                        price: float, payout: float) -> pd.DataFrame:
    """000〜999 を展開し、joint_prob と EV を計算して返す"""
    rows: List[dict] = []
    for h in range(10):
        for t in range(10):
            for o in range(10):
                joint = float(p_h[h] * p_t[t] * p_o[o])
                num = h * 100 + t * 10 + o
                ev_gross = payout * joint
                ev_net = ev_gross - price
                rows.append({
                    "抽せん日": draw_date,
                    "番号": f"{num:03d}",
                    "百": h, "十": t, "一": o,
                    "joint_prob": joint,
                    "EV_gross": ev_gross,
                    "EV_net": ev_net,
                })
    df = pd.DataFrame(rows)
    # 降順ソート
    df = df.sort_values(["EV_net", "EV_gross", "joint_prob"], ascending=[False, False, False])
    # 期待値の見やすい列
    df["expected_payout"] = df["EV_gross"]
    df["price"] = price
    # 順位列
    df["rank"] = np.arange(1, len(df) + 1)
    # 表示用（互換）
    df["候補_3桁"] = df["番号"]
    return df


def build_ev(next_pred: Path, out_path: Path, price: float, payout: float, limit: int | None) -> pd.DataFrame:
    df_next = _read_next_pred(next_pred)
    draw_date = _pick_date(df_next)

    # 桁ごとの確率を取得（無ければフォールバックで作る）
    p_h, p_t, p_o = _load_digit_probs(df_next)

    # 1000通りへ展開
    df = _expand_all_numbers(draw_date, p_h, p_t, p_o, float(price), float(payout))

    if limit is not None and limit > 0:
        df = df.head(limit).copy()

    # メタ情報（あれば流用）
    for meta_col in ["feature_set", "model_name", "n_features"]:
        if meta_col in df_next.columns:
            df[meta_col] = df_next[meta_col].iloc[0]

    # 最終カラム順
    base_cols = ["抽せん日", "番号", "rank", "joint_prob",
                 "expected_payout", "price", "EV_gross", "EV_net",
                 "百", "十", "一", "候補_3桁"]
    meta_cols = [c for c in ["feature_set", "model_name", "n_features"] if c in df.columns]
    df = df[base_cols + meta_cols]

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def main():
    ap = argparse.ArgumentParser(description="EVレポート生成（000〜999を展開）")
    ap.add_argument("--make-ev", action="store_true", help="EV を生成")
    ap.add_argument("--next_pred", type=str, default="artifacts/outputs/next_prediction.csv",
                    help="next_prediction.csv のパス")
    ap.add_argument("--out", type=str, default="artifacts/outputs/ev_report.csv",
                    help="出力CSVのパス")
    ap.add_argument("--price", type=float, default=200, help="購入金額（1口）")
    ap.add_argument("--payout", type=float, default=90000, help="払戻（単勝ち）")
    ap.add_argument("--limit", type=int, default=1000, help="最大出力行数（デフォルト1000）")
    args = ap.parse_args()

    if not args.make_ev:
        ap.print_help()
        return 0

    next_pred = Path(args.next_pred)
    out_path = Path(args.out)
    if not next_pred.exists():
        print("[ERR] next_prediction.csv が見つかりません。--next_pred で明示してください。")
        return 1

    df = build_ev(next_pred, out_path, args.price, args.payout, args.limit)
    # 先頭だけログ
    head = df.head(5)[["抽せん日", "番号", "EV_net", "joint_prob"]]
    print(f"[OK] EV を作成: {out_path}")
    with pd.option_context("display.max_rows", 10, "display.width", 200):
        print(head.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
