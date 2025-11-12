# -*- coding: utf-8 -*-
"""
n3.generate_candidates — Top-K候補生成（V4対応・後方互換）
- 既存の predict_next / append_history は触りません
- K拡張（K=5等）、低確信時の適応拡張、ペアブースト順位付けに対応
- デフォルトは従来どおり joint（積）/ K=4 / TopN=20

出力（ロング形式）:
  抽せん日, 順位, 候補番号, 候補_百の位, 候補_十の位, 候補_一の位,
  prob_百, prob_十, prob_一, joint_prob, score, feature_set, model_name
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import itertools
import numpy as np
import pandas as pd
from joblib import load

from n3.features_v4 import add_features_v4

# オプショナル（無ければ無視）
try:
    from n3.features_v3 import add_features_v3  # type: ignore
except Exception:
    add_features_v3 = None  # type: ignore

add_features_legacy = None
try:
    from n3.features import add_features as _legacy_add_features  # type: ignore
    add_features_legacy = _legacy_add_features
except Exception:
    try:
        from n3.features import build_features as _legacy_build_features  # type: ignore
        add_features_legacy = _legacy_build_features
    except Exception:
        add_features_legacy = None

TARGETS = ["百の位", "十の位", "一の位"]


def build_frames(history_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    if add_features_legacy is not None:
        try:
            frames["LEGACY"] = add_features_legacy(history_df).copy()
        except Exception:
            frames["LEGACY"] = history_df.copy()
    else:
        frames["LEGACY"] = history_df.copy()

    if add_features_v3 is not None:
        try:
            frames["V3"] = add_features_v3(history_df).copy()
        except Exception:
            pass

    try:
        frames["V4"] = add_features_v4(history_df).copy()
    except Exception:
        pass

    return frames


def pick_frame_by_set(frames: Dict[str, pd.DataFrame], feature_set: str | None) -> pd.DataFrame:
    if feature_set and feature_set in frames:
        return frames[feature_set]
    for cand in ("V4", "V3", "LEGACY"):
        if cand in frames:
            return frames[cand]
    raise KeyError(f"no frame for feature_set={feature_set}, available={list(frames.keys())}")


def get_next_bday(last_date: pd.Timestamp) -> pd.Timestamp:
    start = last_date + pd.Timedelta(days=1)
    return pd.bdate_range(start=start, periods=1)[0]


def load_model_and_features(models_dir: Path, target: str):
    model_path = models_dir / f"model_{target}.joblib"
    feat_path  = models_dir / f"features_{target}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"特徴定義が見つかりません: {feat_path}")
    model = load(model_path)
    with open(feat_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    fset = j.get("feature_set", "V4")
    mname = j.get("model_name", "XGBClassifier")
    feats = j.get("features", [])
    if not isinstance(feats, list) or not feats:
        raise ValueError(f"特徴リストが不正です: {feat_path}")
    return model, feats, fset, mname


def topk_from_proba(model, x_one: pd.DataFrame, k: int) -> List[Tuple[int, float]]:
    proba = model.predict_proba(x_one)[0]
    classes = getattr(model, "classes_", np.arange(len(proba)))
    pairs = list(zip(classes.astype(int), proba.astype(float)))
    pairs.sort(key=lambda t: t[1], reverse=True)
    return pairs[:k]


def adaptive_expand(k_base: int, p_max: float, thresh: float, kmax: int) -> int:
    """
    その桁の最大確率が低ければKを+1（上限kmax）する簡易適応拡張
    """
    if p_max < thresh and k_base < kmax:
        return k_base + 1
    return k_base


def score_joint(ph: float, pt: float, po: float) -> float:
    return float(ph * pt * po)


def score_pairboost(ph: float, pt: float, po: float, lam: float) -> float:
    """
    2桁一致を押し上げるブースト：
      joint * (GM(pairwise))^lam
    ここで GM(pairwise) = ((ph*pt)*(ph*po)*(pt*po))^(1/3)
    lam=0 で joint と同じ、lam>0 で2桁が強い候補を相対的に上げる
    """
    pair_gm = (ph*pt * ph*po * pt*po) ** (1.0/3.0)
    return float((ph*pt*po) * (pair_gm ** lam))


def generate_candidates(
    models: Dict[str, object],
    x_one: pd.DataFrame,
    topk: int,
    topn: int,
    score_mode: str = "joint",
    pair_lambda: float = 0.5,
    adaptive: bool = False,
    thresh: float = 0.25,
    kmax: int = 6,
) -> List[Dict]:
    # まず各桁のTopK候補（確信度に応じてKを増減）
    proba_h = models["百の位"].predict_proba(x_one)[0]
    proba_t = models["十の位"].predict_proba(x_one)[0]
    proba_o = models["一の位"].predict_proba(x_one)[0]
    max_h, max_t, max_o = float(np.max(proba_h)), float(np.max(proba_t)), float(np.max(proba_o))

    k_h = adaptive_expand(topk, max_h, thresh, kmax) if adaptive else topk
    k_t = adaptive_expand(topk, max_t, thresh, kmax) if adaptive else topk
    k_o = adaptive_expand(topk, max_o, thresh, kmax) if adaptive else topk

    cand_h = topk_from_proba(models["百の位"], x_one, k_h)
    cand_t = topk_from_proba(models["十の位"], x_one, k_t)
    cand_o = topk_from_proba(models["一の位"], x_one, k_o)

    combos = []
    for (h, ph), (t, pt), (o, po) in itertools.product(cand_h, cand_t, cand_o):
        joint = score_joint(ph, pt, po)
        if score_mode == "pairboost":
            s = score_pairboost(ph, pt, po, pair_lambda)
        else:
            s = joint
        num = int(h) * 100 + int(t) * 10 + int(o)
        combos.append({
            "候補番号": num,
            "候補_百の位": int(h),
            "候補_十の位": int(t),
            "候補_一の位": int(o),
            "prob_百": float(ph),
            "prob_十": float(pt),
            "prob_一": float(po),
            "joint_prob": float(joint),
            "score": float(s),
        })

    combos.sort(key=lambda d: (-d["score"], d["候補番号"]))

    out, seen = [], set()
    for d in combos:
        if d["候補番号"] in seen:
            continue
        out.append(d); seen.add(d["候補番号"])
        if len(out) >= topn:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="data/raw/Numbers3features_master.csv")
    ap.add_argument("--models_dir", default="artifacts/models_V4_XGB")
    ap.add_argument("--candidates_out", default="artifacts/outputs/next_candidates.csv")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--score", choices=["joint","pairboost"], default="joint")
    ap.add_argument("--pair_lambda", type=float, default=0.5)
    ap.add_argument("--adaptive", type=int, default=0, help="0/1: 低確信時にKを+1する")
    ap.add_argument("--thresh", type=float, default=0.25, help="adaptive用しきい値")
    ap.add_argument("--kmax", type=int, default=6, help="adaptiveでのK上限")
    args = ap.parse_args()

    history = Path(args.history)
    models_dir = Path(args.models_dir)
    out_path = Path(args.candidates_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history_df = pd.read_csv(history, encoding="utf-8-sig")
    if "抽せん日" not in history_df.columns:
        raise SystemExit("『抽せん日』列が見つかりません。")
    history_df["抽せん日"] = pd.to_datetime(history_df["抽せん日"])
    history_df = history_df.sort_values("抽せん日").reset_index(drop=True)
    last_day = pd.to_datetime(history_df["抽せん日"].iloc[-1])
    target_day = get_next_bday(last_day)

    frames = build_frames(history_df)

    models: Dict[str, object] = {}
    feat_cols: List[str] | None = None
    feature_set: str | None = None
    model_name: str | None = None

    for tgt in TARGETS:
        m, feats, fset, mname = load_model_and_features(models_dir, tgt)
        models[tgt] = m
        if feat_cols is None:
            feat_cols = feats; feature_set = fset; model_name = mname

    if feat_cols is None:
        raise RuntimeError("特徴リストを取得できませんでした。")

    d = pick_frame_by_set(frames, feature_set)
    x_one = d[feat_cols].iloc[[-1]]

    cands = generate_candidates(
        models, x_one,
        topk=args.topk, topn=args.topn,
        score_mode=args.score, pair_lambda=args.pair_lambda,
        adaptive=bool(args.adaptive), thresh=args.thresh, kmax=args.kmax,
    )

    out_df = pd.DataFrame(cands)
    out_df.insert(0, "順位", np.arange(1, len(out_df) + 1))
    out_df.insert(0, "抽せん日", target_day.strftime("%Y-%m-%d"))
    out_df["feature_set"] = feature_set
    out_df["model_name"] = model_name
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] feature_set={feature_set}, model={model_name}, feats={len(feat_cols)}")
    print(f"[INFO] target_day={target_day.date()}, rows={len(out_df)}")
    print(f"[INFO] score={args.score}, topk={args.topk}, adaptive={bool(args.adaptive)} (th={args.thresh}, kmax={args.kmax})")
    print(f"[INFO] candidates -> {out_path}")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(out_df.head(10))


if __name__ == "__main__":
    main()
