# -*- coding: utf-8 -*-
"""
n3.backtest_candidates_v4 (pairboost/adaptive対応版)
- 期間の各営業日について、前営業日までの情報で特徴生成→Top-K直積→Top-N判定
- 並べ替えスコア: joint / pairboost
- 低確信の桁だけKを自動増やす adaptive に対応
- features_master優先→prediction_historyの正解列にフォールバック

出力: 日別に true_num, top1_num, rank_if_hit, hit_top1, hit_topN など
"""

from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import load

from n3.features_v4 import add_features_v4

TARGETS = ["百の位", "十の位", "一の位"]


def bday_prev(d: pd.Timestamp) -> pd.Timestamp:
    return pd.bdate_range(end=d, periods=2)[0]


def as_int(val) -> int:
    if pd.isna(val): raise ValueError("nan")
    try: return int(val)
    except Exception:
        return int(round(float(val)))


def get_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def load_model_and_feats(models_dir: Path):
    models = {}
    feat_cols = None
    feature_set = "V4"
    model_name  = "XGBClassifier"
    for tgt in TARGETS:
        mpath = models_dir / f"model_{tgt}.joblib"
        fpath = models_dir / f"features_{tgt}.json"
        if not mpath.exists() or not fpath.exists():
            raise FileNotFoundError(f"missing: {mpath} or {fpath}")
        models[tgt] = load(mpath)
        if feat_cols is None:
            with open(fpath, "r", encoding="utf-8") as f:
                j = json.load(f)
            feat_cols = j.get("features", [])
            feature_set = j.get("feature_set", "V4")
            model_name  = j.get("model_name", "XGBClassifier")
    if not feat_cols:
        raise RuntimeError("feature list not found in JSON")
    return models, feat_cols, feature_set, model_name


def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    miss = [c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"feature columns missing: {miss[:6]}{'...' if len(miss)>6 else ''}")
    return df[cols]


def topk_from_proba(model, x_one: pd.DataFrame, k: int):
    proba = model.predict_proba(x_one)[0]
    classes = getattr(model, "classes_", np.arange(len(proba), dtype=int))
    pairs = list(zip(classes.astype(int), proba.astype(float)))
    pairs.sort(key=lambda t: t[1], reverse=True)
    return pairs[:k]


def adaptive_expand(k_base: int, p_max: float, thresh: float, kmax: int) -> int:
    return k_base + 1 if (p_max < thresh and k_base < kmax) else k_base


def score_joint(ph: float, pt: float, po: float) -> float:
    return float(ph * pt * po)


def score_pairboost(ph: float, pt: float, po: float, lam: float) -> float:
    pair_gm = (ph*pt * ph*po * pt*po) ** (1.0/3.0)
    return float((ph*pt*po) * (pair_gm ** lam))


def gen_candidates(models: Dict[str, object],
                   x_one: pd.DataFrame,
                   topk: int, topn: int,
                   score_mode: str, pair_lambda: float,
                   adaptive: bool, thresh: float, kmax: int) -> List[Dict]:
    proba_h = models["百の位"].predict_proba(x_one)[0]
    proba_t = models["十の位"].predict_proba(x_one)[0]
    proba_o = models["一の位"].predict_proba(x_one)[0]
    max_h, max_t, max_o = float(np.max(proba_h)), float(np.max(proba_t)), float(np.max(proba_o))

    k_h = adaptive_expand(topk, max_h, thresh, kmax) if adaptive else topk
    k_t = adaptive_expand(topk, max_t, thresh, kmax) if adaptive else topk
    k_o = adaptive_expand(topk, max_o, thresh, kmax) if adaptive else topk

    h_list = topk_from_proba(models["百の位"], x_one, k_h)
    t_list = topk_from_proba(models["十の位"], x_one, k_t)
    o_list = topk_from_proba(models["一の位"], x_one, k_o)

    combos = []
    for (h, ph), (t, pt), (o, po) in itertools.product(h_list, t_list, o_list):
        joint = score_joint(ph, pt, po)
        s = score_pairboost(ph, pt, po, pair_lambda) if score_mode == "pairboost" else joint
        num = int(h) * 100 + int(t) * 10 + int(o)
        combos.append({
            "候補番号": num, "候補_百の位": int(h), "候補_十の位": int(t), "候補_一の位": int(o),
            "prob_百": float(ph), "prob_十": float(pt), "prob_一": float(po),
            "joint_prob": float(joint), "score": float(s),
        })
    combos.sort(key=lambda d: (-d["score"], d["候補番号"]))

    out, seen = [], set()
    for c in combos:
        if c["候補番号"] in seen: continue
        out.append(c); seen.add(c["候補番号"])
        if len(out) >= topn: break
    return out


def get_truth_from_features(whole: pd.DataFrame, day: pd.Timestamp) -> Optional[Tuple[int,int,int]]:
    row = whole[whole["抽せん日"] == day]
    if row.empty: return None
    r = row.iloc[0]
    def pick(prefix_opts):
        for n in prefix_opts:
            if n in whole.columns: return as_int(r[n])
        return None
    h = pick(["百の位","本数字_百の位","当せん_百の位"])
    t = pick(["十の位","本数字_十の位","当せん_十の位"])
    o = pick(["一の位","本数字_一の位","当せん_一の位"])
    if h is None or t is None or o is None: return None
    return h, t, o


def get_truth_from_predhist(predhist_path: Path, day: pd.Timestamp) -> Optional[Tuple[int,int,int]]:
    if not predhist_path.exists(): return None
    ph = pd.read_csv(predhist_path, encoding="utf-8-sig")
    if "抽せん日" not in ph.columns: return None
    ph["抽せん日"] = pd.to_datetime(ph["抽せん日"], errors="coerce")
    row = ph[ph["抽せん日"] == day]
    if row.empty: return None
    r = row.iloc[0]
    try:
        h = as_int(r.get("正解_百の位"))
        t = as_int(r.get("正解_十の位"))
        o = as_int(r.get("正解_一の位"))
        return h, t, o
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="data/raw/Numbers3features_master.csv")
    ap.add_argument("--models_dir", default="artifacts/models_V4_XGB")
    ap.add_argument("--start", default="2025-09-29")
    ap.add_argument("--end", default="2025-10-10")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--score", choices=["joint","pairboost"], default="joint")
    ap.add_argument("--pair_lambda", type=float, default=0.5)
    ap.add_argument("--adaptive", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.25)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--out", default="artifacts/outputs/candidate_backtest.csv")
    ap.add_argument("--predhist", default="artifacts/outputs/prediction_history.csv")
    ap.add_argument("--snapshot_dir", default="")
    args = ap.parse_args()

    hist_path = Path(args.hist)
    models_dir = Path(args.models_dir)
    out_path = Path(args.out)
    predhist_path = Path(args.predhist)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    whole = pd.read_csv(hist_path, encoding="utf-8-sig")
    if "抽せん日" not in whole.columns:
        raise SystemExit("『抽せん日』列が見つかりません。")
    whole["抽せん日"] = pd.to_datetime(whole["抽せん日"])
    whole = whole.sort_values("抽せん日").reset_index(drop=True)

    models, feat_cols, feature_set, model_name = load_model_and_feats(models_dir)

    days = pd.bdate_range(args.start, args.end)
    results = []

    snap_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
    if snap_dir: snap_dir.mkdir(parents=True, exist_ok=True)

    for day in days:
        prev = bday_prev(day)
        hist_until_prev = whole[whole["抽せん日"] <= prev].copy()
        if hist_until_prev.empty: continue

        feats = add_features_v4(hist_until_prev)
        if feats.empty: continue

        try:
            x_one = ensure_cols(feats, feat_cols).iloc[[-1]]
        except Exception:
            continue

        cands = gen_candidates(
            models, x_one,
            topk=args.topk, topn=args.topn,
            score_mode=args.score, pair_lambda=args.pair_lambda,
            adaptive=bool(args.adaptive), thresh=args.thresh, kmax=args.kmax
        )

        truth = get_truth_from_features(whole, day) or get_truth_from_predhist(predhist_path, day)
        if truth is None:  # 正解が無ければ評価不可
            continue

        h, t, o = truth
        true_num = h*100 + t*10 + o
        rank = next((i+1 for i,c in enumerate(cands) if c["候補番号"]==true_num), None)

        results.append({
            "抽せん日": day.strftime("%Y-%m-%d"),
            "true_num": true_num,
            "top1_num": cands[0]["候補番号"] if cands else None,
            "rank_if_hit": rank,
            "hit_top1": int(rank == 1) if rank is not None else 0,
            "hit_topN": int(rank is not None),
            "K": args.topk, "N": args.topn,
            "score": args.score, "pair_lambda": args.pair_lambda,
            "adaptive": int(bool(args.adaptive)), "thresh": args.thresh, "kmax": args.kmax,
            "feature_set": feature_set, "model_name": model_name,
        })

        if snap_dir is not None:
            df = pd.DataFrame(cands)
            df.insert(0, "順位", np.arange(1, len(df)+1))
            df.insert(0, "抽せん日", day.strftime("%Y-%m-%d"))
            df.to_csv(snap_dir / f"cands_{day.strftime('%Y%m%d')}.csv",
                      index=False, encoding="utf-8-sig")

    if not results:
        print("[INFO] no evaluable days.")
        return

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_path, index=False, encoding="utf-8-sig")

    topN_rate = df_res["hit_topN"].mean()
    top1_rate = df_res["hit_top1"].mean()

    print("=== Candidate Backtest Summary ===")
    print(f"期間: {args.start} .. {args.end} (評価日 {len(df_res)})")
    print(f"Top-1 命中率 : {top1_rate:.3f}")
    print(f"Top-{args.topn} 命中率: {topN_rate:.3f}")
    print(f"score={args.score}, pair_lambda={args.pair_lambda}, adaptive={bool(args.adaptive)}")
    print(f"出力: {out_path}")
    if snap_dir:
        print(f"候補スナップショット: {snap_dir}")


if __name__ == "__main__":
    main()
