# local_refresh_joint.py
# ローカル用「最新化」スクリプト
# スクレイピング → joint予測 → EV計算 → 履歴マージ まで一括実行

from __future__ import annotations
import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ========= パス/定数 ========= #
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"

DATA_RAW = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "artifacts" / "outputs"

EV_CSV           = OUT_DIR / "ev_report.csv"
NEXT_CSV         = OUT_DIR / "next_prediction.csv"
PRED_HISTORY     = OUT_DIR / "prediction_history.csv"
PRED_HISTORY_TMP = OUT_DIR / "prediction_history.tmp.csv"

MODELS_V5_JOINT  = ROOT / "artifacts" / "models_V5_joint"

PREDICT_MOD      = "n3.prediction.predict_next_joint"
SCRAPE_MOD       = "n3.scrape_update"  # あればこれを優先的に使う

JST = timezone(timedelta(hours=9))


# ========= ユーティリティ ========= #

def fmt3(v: object) -> str:
    s = str(v).strip()
    if s in ("", "None", "nan", "<NA>"):
        return ""
    try:
        return f"{int(float(s)) % 1000:03d}"
    except Exception:
        return ""

def ensure_joint_prob(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if all(c in df.columns for c in ["p_hundred", "p_ten", "p_one"]):
        p = (
            pd.to_numeric(df["p_hundred"], errors="coerce").clip(0, 1) *
            pd.to_numeric(df["p_ten"],     errors="coerce").clip(0, 1) *
            pd.to_numeric(df["p_one"],     errors="coerce").clip(0, 1)
        )
    elif "joint_prob" in df.columns:
        p = pd.to_numeric(df["joint_prob"], errors="coerce")
    elif "score" in df.columns:
        p = pd.to_numeric(df["score"], errors="coerce")
    else:
        p = pd.Series([0.0] * len(df), index=df.index)
    df["joint_prob"] = p.fillna(0.0).clip(0, 1)
    return df

def _env_with_src() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return env

def module_available(modname: str) -> bool:
    try:
        return importlib.util.find_spec(modname) is not None
    except Exception:
        return False

def run_py_module(module: str, args: list[str]) -> tuple[int, str]:
    """python -m module args... をサブプロセスで実行"""
    cmd = [sys.executable, "-m", module, *args]
    print(f"[RUN] {' '.join(cmd)}")
    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            shell=False,
            env=_env_with_src(),
        )
        out = (p.stdout or "") + (p.stderr or "")
        print(out)
        return p.returncode, out
    except Exception as e:
        msg = f"[runner-error] {e}"
        print(msg)
        return 1, msg

def run_py_script(path: Path, args: list[str]) -> tuple[int, str]:
    """python path args... をサブプロセスで実行"""
    cmd = [sys.executable, str(path), *args]
    print(f"[RUN] {' '.join(cmd)}")
    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            shell=False,
            env=_env_with_src(),
        )
        out = (p.stdout or "") + (p.stderr or "")
        print(out)
        return p.returncode, out
    except Exception as e:
        msg = f"[runner-error] {e}"
        print(msg)
        return 1, msg

def read_csv_safe(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] read_csv失敗: {p} ({e})")
        return None

def find_latest_history() -> Path | None:
    """data/raw から最新の *_Numbers3features.csv を返す"""
    if not DATA_RAW.exists():
        print(f"[ERR] DATA_RAW が存在しません: {DATA_RAW}")
        return None
    cands = list(DATA_RAW.glob("*_Numbers3features.csv"))
    if not cands:
        print(f"[ERR] *_Numbers3features.csv が見つかりません: {DATA_RAW}")
        return None
    latest = max(cands, key=lambda x: (x.stat().st_mtime, x.name))
    print(f"[INFO] history: {latest}")
    return latest

def _make_date_key(df: pd.DataFrame, col: str = "抽せん日") -> pd.DataFrame:
    if col not in df.columns:
        df[col] = pd.NaT
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df["date_key"] = df[col].dt.date
    return df

def _next_weekday(d: date) -> date:
    """土日をスキップして次の平日に進める"""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d

def compute_target_draw_date(hist_last_date_str: str) -> str:
    """
    history 最終日の翌平日をベースに、
    「今日より過去にならないように」調整した抽せん日を ISO で返す。
    """
    last_d = datetime.strptime(hist_last_date_str, "%Y-%m-%d").date()
    base = _next_weekday(last_d + timedelta(days=1))
    today = datetime.now(JST).date()
    target = base
    if target < today:
        target = _next_weekday(today)
    while target <= last_d:
        target = _next_weekday(target + timedelta(days=1))
    return target.isoformat()

def _stable_merge_history(new_hist: pd.DataFrame) -> pd.DataFrame:
    """
    prediction_history.csv への安定マージ（first-write-wins）。
    すでに同じ date_key の行がある場合は、既存を優先。
    """
    base = read_csv_safe(PRED_HISTORY)
    if base is None or base.empty:
        if "抽せん日" in new_hist.columns:
            new_hist = _make_date_key(new_hist, "抽せん日")
        return new_hist.copy()

    if "抽せん日" in base.columns:
        base = _make_date_key(base, "抽せん日")
    if "抽せん日" in new_hist.columns:
        new_hist = _make_date_key(new_hist, "抽せん日")

    all_cols = list(dict.fromkeys(list(base.columns) + list(new_hist.columns)))
    base2 = base.reindex(columns=all_cols)
    new2  = new_hist.reindex(columns=all_cols)

    exist_keys = set(base2["date_key"].dropna().unique())
    add_rows   = new2[~new2["date_key"].isin(exist_keys)].copy()

    merged = pd.concat([base2, add_rows], ignore_index=True)
    if "抽せん日" in merged.columns:
        merged = merged.sort_values("抽せん日", ascending=False)

    return merged

def _write_stable_history_from_tmp(tmp_path: Path) -> None:
    tmp_df = read_csv_safe(tmp_path)
    if tmp_df is None or tmp_df.empty:
        print("[WARN] 一時履歴CSVが空のため、prediction_history は更新しません。")
        return
    merged = _stable_merge_history(tmp_df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PRED_HISTORY, index=False, encoding="utf-8-sig")
    print(f"[OK] prediction_history を更新: {PRED_HISTORY}")

def _ensure_cand3(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "候補_3桁" not in d.columns or d["候補_3桁"].isna().all():
        if "予測番号" in d.columns:
            d["候補_3桁"] = d["予測番号"]
        elif "番号" in d.columns:
            d["候補_3桁"] = d["番号"]
        elif all(c in d.columns for c in ["百", "十", "一"]):
            d["候補_3桁"] = (
                pd.to_numeric(d["百"], errors="coerce").fillna(0).astype(int).astype(str) +
                pd.to_numeric(d["十"], errors="coerce").fillna(0).astype(int).astype(str) +
                pd.to_numeric(d["一"], errors="coerce").fillna(0).astype(int).astype(str)
            )
        else:
            d["候補_3桁"] = ""
    d["候補_3桁"] = d["候補_3桁"].map(fmt3)
    return d

def find_update_script() -> Path | None:
    """
    モジュール n3.scrape_update が無い場合のフォールバック。
    それっぽい .py を探す。
    """
    for p in [
        SRC / "n3" / "scrape_update.py",
        ROOT / "data" / "scrape_update.py",
        ROOT / "scrape_update.py",
        SRC / "n3" / "scrape_all.py",
        ROOT / "data" / "scrape_all.py",
        ROOT / "scrape_all.py",
    ]:
        if p.exists():
            return p
    return None


# ========= メイン処理 ========= #

def main(price: int = 200, payout: int = 90000, topn: int = 1000, do_scrape: bool = True) -> int:
    print("==== ローカル最新化（scrape → joint予測 → EV生成）開始 ====")

    # --- 0) スクレイピング（任意） ---
    if do_scrape:
        print("---- スクレイピング開始 ----")
        rc_scr = 1

        if module_available(SCRAPE_MOD):
            rc_scr, out_scr = run_py_module(SCRAPE_MOD, [])
        else:
            script = find_update_script()
            if script is not None:
                rc_scr, out_scr = run_py_script(script, [])
            else:
                print("[WARN] スクレイピング用スクリプトが見つかりません。既存の history で続行します。")
                rc_scr = 0  # 単にスキップ扱いにする

        if rc_scr != 0:
            print("[WARN] スクレイピングでエラーが発生しましたが、既存の history を使って続行します。")
        else:
            print("---- スクレイピング完了 ----")

    # --- 1) history 検出 ---
    hist = find_latest_history()
    if hist is None:
        print("[FATAL] history が見つからないため終了します。")
        return 1

    # 抽せん日ターゲット計算
    try:
        df_hist = pd.read_csv(hist, encoding="utf-8-sig")
        if "抽せん日" not in df_hist.columns:
            raise RuntimeError("history に '抽せん日' 列がありません。")
        dmax = pd.to_datetime(df_hist["抽せん日"], errors="coerce").max()
        if pd.isna(dmax):
            raise RuntimeError("history の抽せん日が読み取れません。")
        hist_last_iso = dmax.date().isoformat()
        target_str = compute_target_draw_date(hist_last_iso)
        print(f"[INFO] history last date : {hist_last_iso}")
        print(f"[INFO] target draw date : {target_str}")
    except Exception as e:
        print(f"[ERR] 抽せん日ターゲット決定に失敗: {e}")
        return 1

    # 既存ファイルのクリア（念のため）
    for pth in [NEXT_CSV, PRED_HISTORY_TMP]:
        if pth.exists():
            try:
                pth.unlink()
                print(f"[INFO] 既存ファイル削除: {pth}")
            except Exception as e:
                print(f"[WARN] ファイル削除に失敗: {pth} ({e})")

    # --- 2) joint予測の実行 ---
    if not module_available(PREDICT_MOD):
        print(f"[FATAL] モジュールが見つかりません: {PREDICT_MOD}")
        print("PYTHONPATH や src/n3 配下の構造を確認してください。")
        return 1

    args = [
        "--models_dir", str(MODELS_V5_JOINT),
        "--history",    str(hist),
        "--out",        str(NEXT_CSV),
        "--hist_out",   str(PRED_HISTORY_TMP),
        "--price",      str(int(price)),
        "--payout",     str(int(payout)),
        "--topn",       str(int(topn)),
    ]
    rc1, out1 = run_py_module(PREDICT_MOD, args)
    if rc1 != 0:
        print("[FATAL] joint予測に失敗しました。上のログを確認してください。")
        return 1

    if not NEXT_CSV.exists():
        print(f"[FATAL] NEXT_CSV が見つかりません: {NEXT_CSV}")
        return 1

    print(f"[OK] next_prediction を生成: {NEXT_CSV}")

    # --- 3) prediction_history.tmp の抽せん日補正＆安定マージ ---
    if PRED_HISTORY_TMP.exists():
        try:
            tmp = read_csv_safe(PRED_HISTORY_TMP)
            if tmp is not None and not tmp.empty:
                # すべて target_str で上書き（jointは「次回」の予測なので）
                tmp["抽せん日"] = pd.to_datetime(target_str)
                tmp.to_csv(PRED_HISTORY_TMP, index=False, encoding="utf-8-sig")
                _write_stable_history_from_tmp(PRED_HISTORY_TMP)
            else:
                print("[WARN] prediction_history.tmp.csv が空でした。履歴は更新されません。")
        except Exception as e:
            print(f"[WARN] 履歴マージ中に例外: {e}")
    else:
        print("[INFO] prediction_history.tmp.csv が見つからなかったため、履歴マージをスキップします。")

    # --- 4) EV のローカル生成（ev_report.csv） ---
    try:
        df_next = read_csv_safe(NEXT_CSV)
        if df_next is None or df_next.empty:
            raise RuntimeError("NEXT_CSV が空です。joint予測で失敗の可能性。")

        df_next["抽せん日"] = target_str
        df_next = _ensure_cand3(df_next)
        df_next = ensure_joint_prob(df_next)

        jp = pd.to_numeric(df_next["joint_prob"], errors="coerce").fillna(0.0).clip(0, 1)
        df_next["EV_gross"] = jp * float(payout)
        df_next["EV_net"]   = df_next["EV_gross"] - float(price)

        sort_cols = [c for c in ["EV_net", "EV_gross", "joint_prob"] if c in df_next.columns]
        if sort_cols:
            df_next = df_next.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df_next.to_csv(EV_CSV, index=False, encoding="utf-8-sig")

        top = df_next.iloc[0]
        print(f"[OK] ev_report.csv を生成: {EV_CSV}")
        print(f"[TOP1] 抽せん日={target_str}, 候補_3桁={top.get('候補_3桁')}, EV_net={top.get('EV_net')}, joint_prob={top.get('joint_prob')}")
    except Exception as e:
        print(f"[FATAL] EV のローカル生成に失敗: {e}")
        return 1

    print("==== ローカル最新化 完了 ✅ ====")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Numbers3 ローカル最新化（scrape → joint予測→EV生成）")
    parser.add_argument("--price",   type=int, default=200,   help="1口あたり購入金額")
    parser.add_argument("--payout",  type=int, default=90000, help="1口あたり払戻金額")
    parser.add_argument("--topn",    type=int, default=1000,  help="joint予測の候補数")
    parser.add_argument("--no-scrape", action="store_true",  help="スクレイピングをスキップする場合に指定")
    args = parser.parse_args()

    sys.exit(
        main(
            price=args.price,
            payout=args.payout,
            topn=args.topn,
            do_scrape=not args.no_scrape,
        )
    )
