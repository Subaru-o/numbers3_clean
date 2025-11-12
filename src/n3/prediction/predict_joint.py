# src/n3/predict_joint.py
# 学習済みジョイントモデル（1000クラス）の確率で next_prediction.csv を作る
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load

def log(m): print(f"[INFO] {m}")

def fmt3(v):
    try: return f"{int(float(v))%1000:03d}"
    except Exception: return "000"

EXCLUDE_COLS = {
    "当選番号","当せん番号","百の位","十の位","一の位",
    "抽せん日","抽選日","date","draw_date","曜日","回号"
}

def build_X(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    use = [c for c in num.columns if c not in EXCLUDE_COLS]
    X = num[use].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if X.empty:
        raise ValueError("数値特徴量が見つかりません。")
    return X, use

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help='"auto" または特徴量CSVパス')
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--out", required=True, help="next_prediction.csv の出力先")
    ap.add_argument("--price", type=float, default=200)
    ap.add_argument("--payout", type=float, default=90000)
    ap.add_argument("--topk", type=int, default=200)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    model_path = models_dir / "joint_model.joblib"
    le_path    = models_dir / "label_encoder.joblib"
    meta_path  = models_dir / "model_meta.json"

    if args.history == "auto":
        # data/raw 内の *_Numbers3features.csv のうち更新日時が最新
        data_raw = Path("data/raw")
        cands = list(data_raw.glob("*_Numbers3features.csv"))
        if not cands:
            raise FileNotFoundError("data/raw に *_Numbers3features.csv がありません。")
        history = max(cands, key=lambda p: p.stat().st_mtime)
    else:
        history = Path(args.history)
    df = pd.read_csv(history, encoding="utf-8-sig")
    log(f"history: {history} rows={len(df):,}")

    # 直近（末尾）レコードで推論（必要に応じて切替可）
    row = df.tail(1).copy()
    X, feat_names = build_X(row)

    # モデル読込
    clf = load(model_path)
    le  = load(le_path)
    labels = le.classes_  # '000'..'999'

    # 確率
    P = clf.predict_proba(X.values)  # shape (1, 1000)
    probs = P[0]
    # 念のため正規化
    probs = probs / probs.sum()

    # 上位Kを候補化
    k = int(max(1, min(args.topk, len(labels))))
    idx = np.argsort(-probs)[:k]
    cand = pd.DataFrame({
        "候補_3桁": labels[idx],
        "joint_prob": probs[idx],
    }).reset_index(drop=True)

    # EV計算
    cand["EV_gross"] = cand["joint_prob"] * float(args.payout)
    cand["EV_net"]   = cand["EV_gross"] - float(args.price)
    cand["feature_set"] = "V5_joint"
    # キャリブ種別をメタから拝借（任意）
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        calib = meta.get("calibration", {}).get("type", "calibrated")
    except Exception:
        calib = "calibrated"
    cand["model_name"]  = f"JointRF_{calib}"

    # 出力
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(out, index=False, encoding="utf-8-sig")
    log(f"saved next_prediction: {out} (rows={len(cand)})")

if __name__ == "__main__":
    main()
