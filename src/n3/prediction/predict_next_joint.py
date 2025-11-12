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
    """月〜金のみを抽せん日とする前提。祝日は考慮しない（簡易）。"""
    nd = d + timedelta(days=1)
    while nd.weekday() >= 5:  # 5=土, 6=日
        nd += timedelta(days=1)
    return nd

def _now_iso() -> str:
    try:
        return datetime.now().isoformat(timespec="seconds")
    except Exception:
        return str(datetime.now())

def _load_joint(models_dir: Path):
    payload = None
    for name in ["joint_model.joblib", "model.joblib"]:
        p = models_dir / name
        if p.exists():
            payload = joblib.load(p)
            break
    if payload is None:
        raise FileNotFoundError(f"joint model not found in: {models_dir}")

    if isinstance(payload, dict):
        model = payload.get("model")
        meta = payload.get("meta", {})
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
    window = int(meta.get("window", 200))

    classes = getattr(model, "classes_", None)
    if classes is None and (models_dir / "label_encoder.joblib").exists():
        try:
            classes = joblib.load(models_dir / "label_encoder.joblib").classes_
        except Exception:
            classes = None

    uniq = len(np.unique(classes)) if classes is not None else None
    print(f"[DIAG] classes_ size = {uniq}")
    return model, feat_cols, classes, window, meta


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = 0.0
    return d


def _first_write_wins_append(csv_path: Path, new_rows: pd.DataFrame,
                             key_cols: list[str]) -> None:
    """
    既存CSVに対し、key_cols（例: 抽せん日, 候補_3桁）で未登録の行のみ追記。
    既存が優先（first-write-wins）。列は両者の和集合で揃える。
    """
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
        # 列そろえ（和集合）
        all_cols = list(dict.fromkeys(list(old.columns) + list(new_rows.columns)))
        old2 = old.reindex(columns=all_cols)
        new2 = new_rows.reindex(columns=all_cols)

        # 既存キー集合
        def _key_tuple(df_: pd.DataFrame) -> pd.Series:
            return df_.apply(lambda r: tuple(r[c] for c in key_cols), axis=1)

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
    ap.add_argument("--out", default=None)         # next_prediction.csv
    ap.add_argument("--hist_out", default=None)    # 当日TopNの一時履歴（アプリが安定マージ）
    ap.add_argument("--append_history", default=None)  # 指定時はここに追記（任意）
    ap.add_argument("--price", type=float, default=None)
    ap.add_argument("--payout", type=float, default=None)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    model, feat_cols, classes, window, meta = _load_joint(models_dir)

    # --- history 読み込み
    df0 = pd.read_csv(args.history, encoding="utf-8-sig")
    if "抽せん日" not in df0.columns:
        raise RuntimeError("history に『抽せん日』列がありません。")

    df0["抽せん日"] = pd.to_datetime(df0["抽せん日"], errors="coerce")
    df0 = df0[df0["抽せん日"].notna()].sort_values("抽せん日").reset_index(drop=True)
    if df0.empty:
        raise RuntimeError("history が空（有効行なし）です。")

    # --- 特徴生成（学習時と揃える）
    df_feat = build_safe_features(
        df0,
        window=window,
        use_weekday_onehot=True,
        use_pattern_onehot=True
    )
    df_feat["date_key"] = df_feat["抽せん日"].dt.date
    last_day: date = df_feat["date_key"].max()
    target_day: date = _next_business_day(last_day)  # ← 来日（次営業日）に設定

    # 最新日の最初の行をベースに予測（※ historyの「将来行」は作らない）
    latest_row = (
        df_feat[df_feat["date_key"] == last_day]
        .sort_values("抽せん日")
        .head(1)
        .iloc[0]
    )

    # --- フィーチャ行列
    df_feat = _ensure_columns(df_feat, feat_cols)
    X = pd.DataFrame([latest_row])[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # --- 予測確率
    proba = model.predict_proba(X)[0].astype(float)
    proba = np.clip(proba, 0, 1)
    cls = classes if (classes is not None and len(classes) == len(proba)) else np.arange(len(proba))

    # --- 温度スケーリング（任意）
    if abs(args.temp - 1.0) > 1e-9:
        eps = 1e-12
        logits = np.log(np.clip(proba, eps, 1.0)) / float(args.temp)
        x = np.exp(logits - np.max(logits))
        proba = x / np.sum(x)

    # --- 1000次元化（必要なら）
    if args.pad_missing_classes and len(np.unique(cls)) != 1000:
        full = np.zeros(1000, dtype=float)
        for p, lab in zip(proba, cls):
            li = int(lab)
            if 0 <= li < 1000:
                full[li] += float(p)
        proba = full
        cls = np.arange(1000)

    # --- TopN抽出
    topn = int(args.topn)
    idx = np.argsort(-proba)[:topn]
    labs = [int(cls[i]) for i in idx]
    probs = [float(proba[i]) for i in idx]
    nums = [_fmt3(v) for v in labs]

    # --- 出力DataFrame
    rows = []
    gen_at = _now_iso()
    model_name = meta.get("model_name") or getattr(model, "__class__", type("X", (), {})).__name__
    feature_set = meta.get("feature_set", f"safe_w{window}")
    for n, pv in zip(nums, probs):
        rows.append({
            "抽せん日": pd.to_datetime(target_day),     # ← 予測対象は「次営業日」
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

    # --- EV 計算（任意）
    if args.price is not None and args.payout is not None:
        price = float(args.price)
        payout = float(args.payout)
        df_topn["EV_gross"] = df_topn["joint_prob"].clip(0, 1) * payout
        df_topn["EV_net"] = df_topn["EV_gross"] - price
        df_topn["price"] = price
        df_topn["expected_payout"] = df_topn["EV_gross"]

    # --- next_prediction.csv
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_topn.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[OK] wrote next_prediction: {out} rows={len(df_topn)}")

    # --- hist_out（一時履歴：アプリの安定マージ用）
    if args.hist_out:
        hist_out = Path(args.hist_out)
        hist_out.parent.mkdir(parents=True, exist_ok=True)
        df_topn.to_csv(hist_out, index=False, encoding="utf-8-sig")
        print(f"[OK] wrote hist_out: {hist_out} rows={len(df_topn)}")

    # --- 直接履歴に追記（任意：指定時のみ）
    if args.append_history:
        append_path = Path(args.append_history)
        # first-write-wins: (抽せん日, 候補_3桁) が鍵
        _first_write_wins_append(append_path, df_topn, key_cols=["抽せん日", "候補_3桁"])
        print(f"[OK] appended to history (first-write-wins): {append_path}")

    # --- コンソール
    print("[TOPN]")
    # 表示列の順序を軽く整える
    show_cols = ["抽せん日", "候補_3桁", "予測番号", "百", "十", "一",
                 "joint_prob", "EV_gross", "EV_net", "price", "expected_payout",
                 "model_name", "feature_set", "gen_at"]
    show_cols = [c for c in show_cols if c in df_topn.columns]
    print(df_topn[show_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
