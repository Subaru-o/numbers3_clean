# src/n3/merge_answers.py — ASCII-safe logs + latest CSV auto-detect
# - Normalizes "回号" to Int64
# - Resolves column conflicts (回号_x/回号_y, 抽選日表示→抽せん日_表示)
# - Drops old 正解_* before re-merge

from __future__ import annotations
import os
import glob
from pathlib import Path
import pandas as pd
from .config import PATHS

def _detect_latest_master_csv() -> str | None:
    """Project root配下から最新の *_Numbers3features.csv を検出して返す。"""
    root = PATHS.root
    cands = sorted(
        glob.glob(str(root / "**" / "*_Numbers3features.csv"), recursive=True)
    )
    return cands[-1] if cands else None

def _normalize_history(df_hist: pd.DataFrame) -> pd.DataFrame:
    """履歴CSVの列ゆらぎを吸収し、再マージ準備を整える。"""
    df = df_hist.copy()

    # 表示列の正規化: 抽選日表示 → 抽せん日_表示
    if "抽選日表示" in df.columns and "抽せん日_表示" not in df.columns:
        df = df.rename(columns={"抽選日表示": "抽せん日_表示"})

    # 過去のマージ結果をクリーンアップ
    # - 正解_* はいったん全削除（再マージで付け直す）
    # - ストレート_金額 も再付与するので削除
    df = df.loc[:, ~df.columns.str.contains(r"^正解_|^ストレート_金額")]

    # 回号はマージ前に一旦除去（_x/_y の競合を避ける）
    for c in ["回号", "回号_x", "回号_y"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df

def merge(master_csv_path: str | None = None):
    # 1) マスターCSVの決定: 引数 > 環境変数 > 自動検出
    master_csv = (master_csv_path or os.getenv("N3_MASTER_CSV", "").strip()) or _detect_latest_master_csv()
    if not master_csv:
        raise ValueError("Master CSV not found. Set N3_MASTER_CSV or place *_Numbers3features.csv under the project.")
    master_csv = str(master_csv)

    # 2) パス決定
    hist_path = PATHS.outputs_dir / "prediction_history.csv"
    if not hist_path.exists():
        raise FileNotFoundError(f"prediction_history.csv not found: {hist_path}")

    # 3) 読み込み
    df_hist = pd.read_csv(hist_path, encoding="utf-8-sig")
    df_master = pd.read_csv(master_csv, encoding="utf-8-sig")

    # 4) 日付を datetime 化
    for df in (df_hist, df_master):
        if "抽せん日" in df.columns:
            df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")

    # 5) 履歴の正規化（列名ゆらぎの吸収、旧マージ列の削除など）
    df_hist = _normalize_history(df_hist)

    # 6) 必須列チェック（マスター）
    need_cols = ["抽せん日", "回号", "百の位", "十の位", "一の位", "ストレート_金額"]
    miss = [c for c in need_cols if c not in df_master.columns]
    if miss:
        raise KeyError(f"Columns missing in master CSV: {miss}")

    # 7) 正解データの整形
    df_answer = df_master[need_cols].rename(columns={
        "百の位": "正解_百の位",
        "十の位": "正解_十の位",
        "一の位": "正解_一の位",
    })

    # 8) 左外部結合（履歴の抽せん日を軸に正解と回号を付与）
    out = df_hist.merge(df_answer, on="抽せん日", how="left")

    # 9) 回号の正規化（_x/_y がもし残っていれば救済し、Int64に）
    if "回号_y" in out.columns:
        out["回号"] = out["回号_y"]
    elif "回号" in out.columns:
        # そのまま
        pass
    elif "回号_x" in out.columns:
        out["回号"] = out["回号_x"]

    # 不要な競合列を削除
    for c in ["回号_x", "回号_y"]:
        if c in out.columns:
            out = out.drop(columns=[c])

    # 数値として正規化（欠損は <NA> のまま保持）
    if "回号" in out.columns:
        out["回号"] = pd.to_numeric(out["回号"], errors="coerce").astype("Int64")

    # 10) 列の並び（見やすさ優先）
    first_cols = [
        "回号", "抽せん日", "抽せん日_表示", "予測番号",
        "予測_百の位", "予測_十の位", "予測_一の位",
        "正解_百の位", "正解_十の位", "正解_一の位",
        "ストレート_金額",
    ]
    ordered = [c for c in first_cols if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    out = out[ordered + rest]

    # 11) 保存
    out.to_csv(hist_path, index=False, encoding="utf-8-sig")

    # 12) サマリ（ASCIIログ）
    merged = out["正解_百の位"].notna().sum() if "正解_百の位" in out.columns else 0
    total = len(out)
    print("OK: merge_answers completed ->", hist_path)
    print(f"  merged rows: {merged} / total: {total}")

if __name__ == "__main__":
    merge()
