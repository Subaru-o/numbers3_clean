# -*- coding: utf-8 -*-
"""
backfill_history_joint.py — V5 joint backfill (leak-guard + diagnostics + scaling)
- Use the FIRST row per day (head(1)) to avoid same-day leakage.
- Drop dangerous columns (winning/result/label/target/num3).
- Do NOT re-normalize predict_proba.
- Print classes_ size for diagnostics.
- Options:
    --renorm-to-1000      : scale Top1 prob by len(classes_)/1000 (display-only fix)
    --temp T              : temperature scaling (T>1 -> flatter)
    --pad-missing-classes : expand proba to 1000 dims (0..999) with zeros for missing labels.

Usage:
python -m n3.backfill_history_joint \
  --history data/raw/XXXX_Numbers3features.csv \
  --models_dir artifacts/models_V5_joint \
  --hist_out artifacts/outputs/prediction_history.tmp.csv \
  --price 200 --payout 90000 \
  [--renorm-to-1000] [--temp 1.0] [--pad-missing-classes]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

# Columns that MUST NOT be used as features (leak sources)
DANGER_PATTERNS = [
    "当せん", "当選", "的中", "正解", "抽選", "結果", "配当",
    "hit", "label", "target", "result", "payout", "rank", "num3",
    "三桁", "3桁"
]

def _is_danger_col(colname: str) -> bool:
    lc = str(colname).lower()
    return any(p.lower() in lc for p in DANGER_PATTERNS)

def _drop_danger_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    drops = [c for c in df.columns if _is_danger_col(c)]
    if drops:
        print(f"[LEAK-GUARD] dropped dangerous columns: {drops}")
    return df.drop(columns=drops, errors="ignore")

def _assert_no_leak_columns(df: pd.DataFrame):
    bad = [c for c in df.columns if _is_danger_col(c)]
    if bad:
        raise RuntimeError(f"[LEAK BLOCKED] dangerous columns remain: {bad}")

def _load_joint(models_dir: Path):
    # joint_model.joblib or model.joblib
    payload = None
    for name in ["joint_model.joblib", "model.joblib"]:
        p = models_dir / name
        if p.exists():
            payload = joblib.load(p)
            break
    if payload is None:
        raise FileNotFoundError(f"joint model not found in: {models_dir}")

    if isinstance(payload, dict):
        model = payload.get("model", None)
        meta  = payload.get("meta", {})
    else:
        model = payload
        meta = {}
        meta_path = models_dir / "model_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

    if model is None:
        raise RuntimeError("failed to load joint model")

    feat_cols = meta.get("feature_names") or meta.get("features") or []
    feat_cols = [c for c in feat_cols if not _is_danger_col(c)]

    classes = getattr(model, "classes_", None)
    if classes is None and (models_dir / "label_encoder.joblib").exists():
        try:
            le = joblib.load(models_dir / "label_encoder.joblib")
            classes = getattr(le, "classes_", None)
        except Exception:
            classes = None

    # Diagnostics (ASCII only)
    if classes is None:
        print("[DIAG] classes_ is None (cannot validate class coverage)")
    else:
        uniq = np.unique(classes)
        print(f"[DIAG] classes_ size = {len(uniq)} (expected: 1000)")
        try:
            sm = sorted(map(int, uniq))
            if len(sm) > 20:
                print(f"[DIAG] classes_ sample: {sm[:10]} ... {sm[-10:]}")
            else:
                print(f"[DIAG] classes_: {sm}")
        except Exception:
            pass
        if len(uniq) != 1000:
            print("[WARN] classes_ size is not 1000. Mean Top1 prob tends to ~ 1/len(classes_).")

    return model, feat_cols, classes

def _prep_X(row_df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    X = _drop_danger_cols(row_df.copy())
    if feat_cols:
        safe_feat = [c for c in feat_cols if not _is_danger_col(c)]
        if len(safe_feat) != len(feat_cols):
            print(f"[LEAK-GUARD] dropped dangerous names from meta features: {set(feat_cols) - set(safe_feat)}")
        for c in safe_feat:
            if c not in X.columns:
                X[c] = 0
        X = X.reindex(columns=safe_feat)
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    _assert_no_leak_columns(X)
    return X

def _fmt3(x) -> str:
    try:
        return f"{int(x)%1000:03d}"
    except Exception:
        s = "".join([c for c in str(x) if c.isdigit()])
        return (s[-3:]).zfill(3) if s else ""

def _temperature_scale(proba: np.ndarray, temp: float) -> np.ndarray:
    if temp is None or abs(temp - 1.0) < 1e-9:
        return proba
    # work in log space for stability
    eps = 1e-12
    logits = np.log(np.clip(proba, eps, 1.0))
    logits = logits / float(temp)
    x = np.exp(logits - np.max(logits))
    x = x / np.sum(x)
    return x

def _pad_to_1000(proba: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Expand to 1000 dims (0..999). Missing classes get prob=0. Sum is unchanged."""
    full = np.zeros(1000, dtype=float)
    # assume labels are integers 0..999 (as intended in this project)
    for p, lab in zip(proba, cls):
        try:
            idx = int(lab)
        except Exception:
            continue
        if 0 <= idx < 1000:
            full[idx] = full[idx] + float(p)
    return full

def build_backfill_joint(history_csv: Path, models_dir: Path,
                         hist_out: Path, price: float, payout: float,
                         renorm_to_1000: bool, temp: float,
                         pad_missing: bool) -> pd.DataFrame:
    model, feat_cols, classes = _load_joint(models_dir)

    df = pd.read_csv(history_csv, encoding="utf-8-sig")
    if "抽せん日" not in df.columns:
        raise SystemExit("[ERR] '抽せん日' column is missing in history CSV")

    # drop dangerous columns early
    df = _drop_danger_cols(df)

    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df[df["抽せん日"].notna()].copy()
    if df.empty:
        raise SystemExit("[ERR] no valid rows in history CSV")

    # use the first row per day (avoid picking post-result rows)
    df["date_key"] = df["抽せん日"].dt.date
    per_day_first = (df.sort_values("抽せん日")
                       .groupby("date_key", as_index=False)
                       .head(1)
                       .copy())

    cls_count = None
    if classes is not None:
        cls_count = int(len(np.unique(classes)))

    rows = []
    for _, row in per_day_first.iterrows():
        day = pd.to_datetime(row["抽せん日"]).date()
        X = _prep_X(pd.DataFrame([row]), feat_cols)

        proba = model.predict_proba(X)[0].astype(float)
        proba = np.clip(proba, 0, 1)

        # class labels aligned with proba
        if classes is None or len(classes) != len(proba):
            cls = np.arange(len(proba))
        else:
            cls = classes

        # temperature scaling (soften/harden) within available classes
        proba = _temperature_scale(proba, temp)

        # optional: pad to 1000-dim (rank unchanged, scale unchanged)
        if pad_missing:
            proba_full = _pad_to_1000(proba, cls)
            cls_full = np.arange(1000)
            proba = proba_full
            cls = cls_full

        # pick top1 after all transforms
        i_top = int(np.argmax(proba))
        label = int(cls[i_top])
        num3 = _fmt3(label)
        h, t, o = int(num3[0]), int(num3[1]), int(num3[2])

        p = float(proba[i_top])

        # optional: scale to 1000-class display (does not change ranking)
        if renorm_to_1000 and cls_count and cls_count > 0:
            p = p * (float(cls_count) / 1000.0)

        ev_gross = payout * p
        ev_net = ev_gross - price

        rows.append({
            "抽せん日": pd.to_datetime(day),
            "候補_3桁": num3,
            "joint_prob": p,
            "EV_net": ev_net,
            "EV_gross": ev_gross,
            "price": price,
            "expected_payout": ev_gross,
            "予測番号": int(num3),
            "百": h, "十": t, "一": o,
            "feature_set": meta_val(feat_cols),
            "model_name": "CalibratedJoint"
        })

    out = pd.DataFrame(rows).sort_values("抽せん日")
    hist_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(hist_out, index=False, encoding="utf-8-sig")

    mean_p = float(out["joint_prob"].mean())
    print(f"[OK] wrote: {hist_out} rows={len(out)}")
    print(f"[SANITY] mean Top1 prob: {mean_p:.6f} (theoretical ~ 1/K, K=class count)")
    if cls_count:
        print(f"[SANITY] K={cls_count} -> 1/K ~ {1.0/cls_count:.6f}")
    return out

def meta_val(feat_cols: list[str]) -> str:
    return f"V5_joint({len(feat_cols)}f)" if feat_cols else "V5_joint"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--hist_out", required=True)
    ap.add_argument("--price", type=float, default=200)
    ap.add_argument("--payout", type=float, default=90000)
    ap.add_argument("--renorm-to-1000", action="store_true",
                    help="scale Top1 prob by len(classes_)/1000 (display-only fix)")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="temperature scaling (>1 flattens, <1 sharpens; default 1.0)")
    ap.add_argument("--pad-missing-classes", action="store_true",
                    help="expand proba to 1000 dims (0..999) with zeros for missing labels")
    args = ap.parse_args()

    build_backfill_joint(
        Path(args.history), Path(args.models_dir), Path(args.hist_out),
        args.price, args.payout,
        renorm_to_1000=args.renorm_to_1000,
        temp=args.temp,
        pad_missing=args.pad_missing_classes
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
