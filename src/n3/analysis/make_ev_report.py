# src/n3/make_ev_report.py
# 次回の候補/分布から EV テーブルを作成して ev_report.csv を出力
# 使い方例:
#   python -m n3.make_ev_report --next_csv artifacts/outputs/next_prediction.csv \
#       --candidates artifacts/outputs/next_candidates_k5_n64_joint.csv \
#       --out artifacts/outputs/ev_report.csv --payout 90000 --cost 200 --topn 128

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import itertools

def _to_3digits(x: int | str) -> str:
    try:
        i = int(str(x).strip())
        return f"{i:03d}"
    except Exception:
        s = "".join([c for c in str(x) if c.isdigit()])
        return s.zfill(3)[:3] if s else ""

def _load_csv(p: Path) -> pd.DataFrame | None:
    if not p or not p.exists():
        return None
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    return None

def _ensure_cols_from_candidates(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # 候補番号
    cand_cols = [c for c in d.columns if c in ("候補番号","予測番号","番号","comb","candidate","number","候補_番号")]
    if cand_cols:
        d.rename(columns={cand_cols[0]: "候補番号"}, inplace=True)
    elif {"候補_百の位","候補_十の位","候補_一の位"}.issubset(d.columns):
        d["候補番号"] = (d["候補_百の位"].astype(int)*100 + d["候補_十の位"].astype(int)*10 + d["候補_一の位"].astype(int)).astype(int).astype(str).str.zfill(3)
    elif {"候補_百","候補_十","候補_一"}.issubset(d.columns):
        d["候補番号"] = (d["候補_百"].astype(int)*100 + d["候補_十"].astype(int)*10 + d["候補_一"].astype(int)).astype(int).astype(str).str.zfill(3)

    # joint_prob 推定
    prob_cols = ["joint_prob","prob_joint","joint","score","得点","スコア"]
    jp = None
    for c in prob_cols:
        if c in d.columns:
            jp = c; break
    if jp is None and {"prob_百","prob_十","prob_一"}.issubset(d.columns):
        d["joint_prob"] = pd.to_numeric(d["prob_百"], errors="coerce") * \
                          pd.to_numeric(d["prob_十"], errors="coerce") * \
                          pd.to_numeric(d["prob_一"], errors="coerce")
    elif jp and jp != "joint_prob":
        d.rename(columns={jp:"joint_prob"}, inplace=True)

    # 抽せん日
    if "抽せん日" not in d.columns:
        # 候補スナップショット等は抽せん日を持つことが多いが無ければ空に
        d["抽せん日"] = ""

    # 桁列
    if "候補番号" in d.columns:
        s = d["候補番号"].astype(str).apply(_to_3digits)
        d["百"] = pd.to_numeric(s.str[0], errors="coerce")
        d["十"] = pd.to_numeric(s.str[1], errors="coerce")
        d["一"] = pd.to_numeric(s.str[2], errors="coerce")

    return d

def _build_from_next(next_df: pd.DataFrame, topk_each: int = 5, topn: int = 128) -> pd.DataFrame:
    """next_prediction.csv の各桁確率から単純直積で候補を作る（上位 topk_each のみ使用）"""
    row = next_df.iloc[0]
    def pick(prefix: str):
        cols = [c for c in next_df.columns if c.startswith(prefix)]
        # p百の位_0..9 → (digit, prob) に整形
        pairs = []
        for c in cols:
            try:
                d = int(c.split("_")[-1])
            except Exception:
                continue
            pairs.append((d, float(row[c])))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:topk_each]
        return pairs

    H = pick("p百") or pick("p_hyaku")
    T = pick("p十") or pick("p_juu")
    O = pick("p一") or pick("p_ichi")
    combos = []
    for (h, ph), (t, pt), (o, po) in itertools.product(H, T, O):
        jp = float(ph) * float(pt) * float(po)
        combos.append({"百":h,"十":t,"一":o,"joint_prob":jp})
    d = pd.DataFrame(combos).sort_values("joint_prob", ascending=False).head(topn).reset_index(drop=True)
    d["候補番号"] = (d["百"]*100 + d["十"]*10 + d["一"]).astype(int).astype(str).str.zfill(3)
    d["抽せん日"] = next_df["抽せん日"].iloc[0] if "抽せん日" in next_df.columns else ""
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--next_csv", type=str, required=False, default="artifacts/outputs/next_prediction.csv")
    ap.add_argument("--candidates", type=str, required=False, default="")
    ap.add_argument("--out", type=str, required=False, default="artifacts/outputs/ev_report.csv")
    ap.add_argument("--payout", type=float, default=90000.0, help="当せん配当（ストレート想定）")
    ap.add_argument("--cost", type=float, default=200.0, help="1口あたり購入額")
    ap.add_argument("--topn", type=int, default=128, help="出力上位件数")
    ap.add_argument("--topk_each", type=int, default=6, help="next分布から作る場合の各桁の上位採用数")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cand_df = _load_csv(Path(args.candidates)) if args.candidates else None
    next_df = _load_csv(Path(args.next_csv))

    if cand_df is not None and not cand_df.empty:
        d = _ensure_cols_from_candidates(cand_df)
        d = d.dropna(subset=["候補番号"]).copy()
        if "joint_prob" not in d.columns:
            raise SystemExit("ERROR: candidatesにjoint_probが見つかりません（prob_百×prob_十×prob_一 で作る仕様に合わせてください）。")
    elif next_df is not None and not next_df.empty:
        d = _build_from_next(next_df, topk_each=int(args.topk_each), topn=int(args.topn))
    else:
        raise SystemExit("ERROR: 候補CSVも次回分布CSVも読み込めませんでした。")

    # EV計算
    d["payout"] = float(args.payout)
    d["cost"] = float(args.cost)
    d["EV_gross"] = d["joint_prob"] * d["payout"]
    d["EV_net"] = d["EV_gross"] - d["cost"]

    # 表示整形
    cols_first = [c for c in ["抽せん日","候補番号","百","十","一","joint_prob","payout","cost","EV_gross","EV_net"] if c in d.columns]
    other = [c for c in d.columns if c not in cols_first]
    d = d[cols_first + other].sort_values("EV_gross", ascending=False).head(int(args.topn))

    d.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] EV report -> {out_path} rows={len(d)} payout={args.payout} cost={args.cost}")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    import sys
    sys.exit(main())
