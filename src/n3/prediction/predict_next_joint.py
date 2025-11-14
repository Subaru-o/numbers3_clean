# -*- coding: utf-8 -*-
"""
predict_next_joint.py — 安全ラグ特徴で TopN 予測（アプリ互換）
- 抽せん日は「historyの最新日」ではなく「その次の営業日」に設定
- 毎回のTopNを履歴CSVへ first-write-wins で追記（オプション）
- hist_out には当日TopN全件を書き出し（アプリ側の安定マージに対応）
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import date, timedelta, datetime

import joblib, numpy as np, pandas as pd

from n3.features.features_safe import build_safe_features


# ========= ユーティリティ =========
def _fmt3(x) -> str:
    try:
        return f"{int(x) % 1000:03d}"
    except Exception:
        s = "".join([c for c in str(x) if c.isdigit()])
        return (s[-3:]).zfill(3) if s else ""

def _next_business_day(d: date) -> date:
    nd = d + timedelta(days=1)
    while nd.weekday() >= 5:
        nd += timedelta(days=1)
    return nd

def _now_iso() -> str:
    try:
        return datetime.now().isoformat(timespec="seconds")
    except Exception:
        return str(datetime.now())


# ========= モデル ローダ（探索強化版） =========
def _load_joint(models_dir: Path):
    p = Path(models_dir)
    if not p.exists():
        raise FileNotFoundError(f"models_dir not found: {p}")

    # デバッグ用に中身をざっと表示
    try:
        entries = [f"{q.relative_to(p)} | {q.stat().st_size}B" for q in p.rglob("*") if q.is_file()]
        entries.sort()
        print("[models] listing:")
        for e in entries[:200]:
            print("  -", e)
    except Exception as _e:
        print(f"[models] list error: {_e}")

    # まずは既知名
    candidates = []
    for name in ("joint_model.joblib", "model.joblib"):
        q = p / name
        if q.exists():
            candidates.append(q)

    # 見つからなければ再帰で *.joblib を探索
    if not candidates:
        all_jobs = list(p.rglob("*.joblib"))
        # “joint” を含むものを優先
        joint_like = [q for q in all_jobs if "joint" in q.name.lower()]
        if joint_like:
            candidates = joint_like
        elif all_jobs:
            # 最もサイズが大きいもの（モデル本体っぽい）を採用
            candidates = [max(all_jobs, key=lambda q: q.stat().st_size)]

    if not candidates:
        raise FileNotFoundError(f"joint model not found in: {p}")

    model_path = candidates[0]
    print(f"[models] using: {model_path}")

    payload = joblib.load(model_path)

    if isinstance(payload, dict):
        model = payload.get("model")
        meta = payload.get("meta", {})
    else:
        model = payload
        meta = {}
        for meta_name in ("meta.json", "model_meta.json"):
            mp = p / meta_name
            if mp.exists():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
                break
    if model is None:
        raise RuntimeError("failed to load joint model")

    # feature names
    feat_cols: list[str] = []
    fn_json = p / "feature_names.json"
    if fn_json.exists():
        try:
            feat_cols = list(json.loads(fn_json.read_text(encoding="utf-8")))
        except Exception:
            feat_cols = []
    if not feat_cols and hasattr(model, "feature_names_in_"):
        try:
            feat_cols = [str(c) for c in model.feature_names_in_]
        except Exception:
            feat_cols = []
    if not feat_cols and hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            if booster is not None and booster.feature_names:
                feat_cols = list(booster.feature_names)
        except Exception:
            pass
    meta_feats = meta.get("feature_names") or meta.get("features") or []
    if meta_feats:
        feat_cols = list(dict.fromkeys(list(feat_cols) + list(meta_feats)))

    # classes
    classes = None
    cjson = p / "classes_1000.json"
    if cjson.exists():
        try:
            classes = list(json.loads(cjson.read_text(encoding="utf-8")))
        except Exception:
            classes = None
    if classes is None:
        le_path = p / "label_encoder.joblib"
        if le_path.exists():
            try:
                le = joblib.load(le_path)
                if hasattr(le, "classes_"):
                    classes = le.classes_.tolist()
            except Exception:
                classes = None
    if classes is None:
        classes = [f"{i:03d}" for i in range(1000)]

    # window
    window = meta.get("window", None)
    if window is None:
        wjson = p / "window.json"
        if wjson.exists():
            try:
                window = json.loads(wjson.read_text(encoding="utf-8"))
            except Exception:
                window = None
    try:
        window = int(window) if window is not None else 200
    except Exception:
        window = 200

    uniq = len(np.unique(classes)) if classes is not None else None
    print(f"[DIAG] classes_ size = {uniq}")
    return model, feat_cols, classes, window, meta


# ========= 前処理ヘルパ =========
def _ensure_columns_for_X(df_feat: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    d = df_feat.copy()
    if not feat_cols:
        drop_like = {"抽せん日", "date", "draw_date", "date_key", "回号"}
        num_cols = [c for c in d.columns if c not in drop_like]
        feat_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(d[c])]
        feat_cols = list(dict.fromkeys(feat_cols))
    for c in feat_cols:
        if c not in d.columns:
            d[c] = 0.0
    X = d[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in feat_cols:
        try:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype(float)
        except Exception:
            X[c] = 0.0
    return X, feat_cols


def _first_write_wins_append(csv_path: Path, new_rows: pd.DataFrame, key_cols: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        try:
            old = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            old = pd.DataFrame()
    else:
        old = pd.DataFrame()

    if old is None or old.empty:
        merged = new_rows.copy()
    else:
        all_cols = list(dict.fromkeys(list(old.columns) + list(new_rows.columns)))
        old2 = old.reindex(columns=all_cols)
        new2 = new_rows.reindex(columns=all_cols)

        def _key_tuple(df_: pd.DataFrame) -> pd.Series:
            return df_.apply(lambda r: tuple(r.get(c) for c in key_cols), axis=1)

        existing = set(_key_tuple(old2))
        add_mask = ~_key_tuple(new2).isin(existing)
        to_add = new2[add_mask].copy()
        merged = pd.concat([old2, to_add], ignore_index=True)

    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")


# ========= 本体 =========
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--topn", type=int, default=1000)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--pad-missing-classes", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hist_out", default=None)
    ap.add_argument("--append_history", default=None)
    ap.add_argument("--price", type=float, default=None)
    ap.add_argument("--payout", type=float, default=None)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    model, feat_cols, classes, window, meta = _load_joint(models_dir)

    df0 = pd.read_csv(args.history, encoding="utf-8-sig")
    if "抽せん日" not in df0.columns:
        raise RuntimeError("history に『抽せん日』列がありません。")

    df0["抽せん日"] = pd.to_datetime(df0["抽せん日"], errors="coerce")
    df0 = df0[df0["抽せん日"].notna()].sort_values("抽せん日").reset_index(drop=True)
    if df0.empty:
        raise RuntimeError("history が空（有効行なし）です。")

    df_feat = build_safe_features(
        df0,
        window=window,
        use_weekday_onehot=True,
        use_pattern_onehot=True
    )
    df_feat["date_key"] = df_feat["抽せん日"].dt.date
    last_day: date = df_feat["date_key"].max()
    target_day: date = _next_business_day(last_day)

    latest_row = (
        df_feat[df_feat["date_key"] == last_day]
        .sort_values("抽せん日")
        .head(1)
        .iloc[0]
    )

    X_all, feat_cols = _ensure_columns_for_X(df_feat, feat_cols)
    X = pd.DataFrame([latest_row])[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    proba = model.predict_proba(X)[0].astype(float)
    proba = np.clip(proba, 0, 1)
    cls = classes if (classes is not None and len(classes) == len(proba)) else np.arange(len(proba))

    if abs(args.temp - 1.0) > 1e-9:
        eps = 1e-12
        logits = np.log(np.clip(proba, eps, 1.0)) / float(args.temp)
        x = np.exp(logits - np.max(logits))
        proba = x / np.sum(x)

    if args.pad_missing_classes and len(np.unique(cls)) != 1000:
        full = np.zeros(1000, dtype=float)
        for p, lab in zip(proba, cls):
            li = int(lab)
            if 0 <= li < 1000:
                full[li] += float(p)
        proba = full
        cls = np.arange(1000)

    topn = int(args.topn)
    idx = np.argsort(-proba)[:topn]
    labs = [int(cls[i]) for i in idx]
    probs = [float(proba[i]) for i in idx]
    nums = [_fmt3(v) for v in labs]

    rows = []
    gen_at = _now_iso()
    model_name = meta.get("model_name") or getattr(model, "__class__", type("X", (), {})).__name__
    feature_set = meta.get("feature_set", f"safe_w{window}")
    for n, pv in zip(nums, probs):
        rows.append({
            "抽せん日": pd.to_datetime(target_day),
            "候補_3桁": n,
            "予測番号": int(n),
            "百": int(n[0]),
            "十": int(n[1]),
            "一": int(n[2]),
            "joint_prob": pv,
            "model_name": model_name,
            "feature_set": feature_set,
            "gen_at": gen_at,
            "source": "predict_next_joint",
        })
    df_topn = pd.DataFrame(rows)

    if args.price is not None and args.payout is not None:
        price = float(args.price)
        payout = float(args.payout)
        df_topn["EV_gross"] = df_topn["joint_prob"].clip(0, 1) * payout
        df_topn["EV_net"] = df_topn["EV_gross"] - price
        df_topn["price"] = price
        df_topn["expected_payout"] = df_topn["EV_gross"]

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_topn.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[OK] wrote next_prediction: {out} rows={len(df_topn)}")

    if args.hist_out:
        hist_out = Path(args.hist_out)
        hist_out.parent.mkdir(parents=True, exist_ok=True)
        df_topn.to_csv(hist_out, index=False, encoding="utf-8-sig")
        print(f"[OK] wrote hist_out: {hist_out} rows={len(df_topn)}")

    if args.append_history:
        append_path = Path(args.append_history)
        _first_write_wins_append(append_path, df_topn, key_cols=["抽せん日", "候補_3桁"])
        print(f"[OK] appended to history (first-write-wins): {append_path}")

    print("[TOPN]")
    show_cols = ["抽せん日", "候補_3桁", "予測番号", "百", "十", "一",
                 "joint_prob", "EV_gross", "EV_net", "price", "expected_payout",
                 "model_name", "feature_set", "gen_at"]
    show_cols = [c for c in show_cols if c in df_topn.columns]
    print(df_topn[show_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
