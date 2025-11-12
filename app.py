# app.py — Numbers3 EV Dashboard（ミニマル＋候補_3桁過去補完＋EV/回号フォールバック＋抽せん日補正）
from __future__ import annotations
import os, sys, subprocess, importlib.util
from pathlib import Path
from datetime import date, timedelta, datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import altair as alt


# ============ パス/定数 ============
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_RAW = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "artifacts" / "outputs"

EV_CSV        = OUT_DIR / "ev_report.csv"
NEXT_CSV      = OUT_DIR / "next_prediction.csv"
EV_BACKFILL   = OUT_DIR / "ev_backfill.csv"
PRED_HISTORY  = OUT_DIR / "prediction_history.csv"
PRED_HISTORY_TMP = OUT_DIR / "prediction_history.tmp.csv"  # 安定マージ用一時

MODELS_V4       = ROOT / "artifacts" / "models_V4_XGB"
MODELS_V5_JOINT = ROOT / "artifacts" / "models_V5_joint"

# === Cloud/Secrets ベースの環境値（任意・あれば使う） ===
DEFAULT_PRICE  = int(st.secrets.get("N3_PRICE",  200))
DEFAULT_PAYOUT = int(st.secrets.get("N3_PAYOUT", 90000))

PREDICT_MOD = "n3.prediction.predict_next_joint"

JST = timezone(timedelta(hours=9))
IS_CLOUD = bool(os.environ.get("STREAMLIT_RUNTIME", ""))

st.set_page_config(page_title="Numbers3 EV Dashboard（ミニマル）", layout="wide")


# ============ ユーティリティ ============
def _tail(txt: str, max_chars: int = 60_000) -> str:
    """長大ログでUIが重くなるのを緩和（末尾のみ表示）"""
    if txt is None:
        return ""
    if len(txt) <= max_chars:
        return txt
    head = "[...log truncated...]\n"
    return head + txt[-max_chars:]


def fmt3(v: object) -> str:
    s = str(v).strip()
    if s in ("", "None", "nan", "<NA>"):
        return ""
    try:
        return f"{int(float(s))%1000:03d}"
    except Exception:
        return ""


def ensure_joint_prob(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if all(c in df.columns for c in ["p_hundred","p_ten","p_one"]):
        p = (
            pd.to_numeric(df["p_hundred"], errors="coerce").clip(0,1) *
            pd.to_numeric(df["p_ten"],     errors="coerce").clip(0,1) *
            pd.to_numeric(df["p_one"],     errors="coerce").clip(0,1)
        )
    elif "joint_prob" in df.columns:
        p = pd.to_numeric(df["joint_prob"], errors="coerce")
    elif "score" in df.columns:
        p = pd.to_numeric(df["score"], errors="coerce")
    else:
        p = pd.Series([0.0]*len(df), index=df.index)
    df["joint_prob"] = p.fillna(0.0).clip(0,1)
    return df


def _env_with_src() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH","")
    return env


def module_available(modname: str) -> bool:
    try:
        return importlib.util.find_spec(modname) is not None
    except Exception:
        return False


def run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=str(cwd), text=True, capture_output=True, shell=False, env=_env_with_src()
            # , timeout=300  # 必要ならタイムアウトも設定可
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return 1, f"[runner-error] {e}"


def run_py_module(module: str, args: list[str]) -> tuple[int, str]:
    return run([sys.executable, "-m", module, *args])


def run_py_script(path: Path, args: list[str]) -> tuple[int, str]:
    return run([sys.executable, str(path), *args])


@st.cache_data(ttl=1800)  # 30分キャッシュ
def read_csv_safe(p: Path) -> pd.DataFrame | None:
    if not p or not p.exists():
        return None
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        return None


@st.cache_data(ttl=1800)
def find_latest_history() -> Path | None:
    if not DATA_RAW.exists():
        return None
    cands = list(DATA_RAW.glob("*_Numbers3features.csv"))
    if not cands:
        return None
    # mtime が同一の場合もあるのでファイル名降順も併用
    return max(cands, key=lambda x: (x.stat().st_mtime, x.name))


def _make_date_key(df: pd.DataFrame, col: str = "抽せん日") -> pd.DataFrame:
    if col not in df.columns:
        df[col] = pd.NaT
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df["date_key"] = df[col].dt.date
    return df


def weekday_ja(d: date) -> str:
    JA = ["月曜日","火曜日","水曜日","木曜日","金曜日","土曜日","日曜日"]
    return JA[d.weekday()]


def winner3_from_raw() -> pd.DataFrame | None:
    p = find_latest_history()
    if p is None: return None
    try:
        raw = pd.read_csv(
            p, encoding="utf-8-sig",
            usecols=lambda c: c in ["抽せん日","当せん番号","当選番号","百の位","十の位","一の位"]
        )
        raw["抽せん日"] = pd.to_datetime(raw["抽せん日"], errors="coerce")
        raw = raw[raw["抽せん日"].notna()].copy()
        base = None
        if "当選番号" in raw.columns:
            base = pd.to_numeric(raw["当選番号"], errors="coerce")
        elif "当せん番号" in raw.columns:
            base = pd.to_numeric(raw["当せん番号"], errors="coerce")
        if base is not None:
            raw["当選番号3"] = base.apply(fmt3)
        else:
            h = pd.to_numeric(raw.get("百の位"), errors="coerce")
            t = pd.to_numeric(raw.get("十の位"), errors="coerce")
            o = pd.to_numeric(raw.get("一の位"), errors="coerce")
            raw["当選番号3"] = (
                h.fillna(-1).astype(int).astype(str) +
                t.fillna(-1).astype(int).astype(str) +
                o.fillna(-1).astype(int).astype(str)
            ).apply(fmt3)
        return raw[["抽せん日","当選番号3"]].dropna(subset=["当選番号3"]).copy()
    except Exception:
        return None


# ========== 実績払戻（1口あたり）マップ ==========
def payouts_map_from_raw(kind: str = "ストレート_金額") -> pd.DataFrame | None:
    """
    history から 1口あたりの払戻（実績）を日付単位で返す。
    方針：`ストレート_金額` は 1口あたり固定としてそのまま採用。
    - 口数による割戻しは一切しない
    - 10,000〜300,000 の範囲に正規化（異常値は NaN として落とす）
    - 同日重複は最後のレコードを優先
    返す列: date_key, 回号, 払戻_実績
    """
    hist_path = find_latest_history()
    if hist_path is None:
        return None
    try:
        raw = pd.read_csv(hist_path, encoding="utf-8-sig")
        if "抽せん日" not in raw.columns:
            return None

        raw["抽せん日"] = pd.to_datetime(raw["抽せん日"], errors="coerce")
        raw = raw[raw["抽せん日"].notna()].copy()
        raw["date_key"] = raw["抽せん日"].dt.date

        # カラム存在チェック
        if kind not in raw.columns:
            alt_names = ["ストレート", "ストレート_1口", "ストレート(1口)", "ストレート_1口あたり"]
            use_col = next((c for c in alt_names if c in raw.columns), None)
            if use_col is None:
                st.info(f"payouts_map_from_raw: '{kind}' 列が見つかりません。")
                return None
        else:
            use_col = kind

        # 数値化＆範囲ガード（1万〜30万）
        per_unit = pd.to_numeric(raw[use_col], errors="coerce")
        valid = (per_unit >= 10000) & (per_unit <= 300000)
        per_unit = per_unit.where(valid, np.nan)

        df = raw[["date_key", "回号"]].copy()
        df["払戻_実績"] = per_unit

        # 同日重複は最新を採用
        df = df.sort_values("date_key").drop_duplicates("date_key", keep="last")

        st.caption("📄 使用しているhistoryファイル: " + str(hist_path))
        st.info("payouts_map_from_raw: モード='1口あたり固定（列をそのまま使用）', 列='" + use_col + f"', 行数={len(df.dropna(subset=['払戻_実績']))}")

        if df["払戻_実績"].notna().any():
            return df[["date_key", "回号", "払戻_実績"]].copy()
        else:
            return None
    except Exception as e:
        st.warning(f"payouts_map_from_raw で例外: {e}")
        return None


def persist_today_pick(pick_date: date, pick_num3: str,
                       ev_adj: float | None = None,
                       prob: float | None = None) -> None:
    df = read_csv_safe(PRED_HISTORY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        df = pd.DataFrame(columns=["抽せん日","候補_3桁_pick","EV_net_adj_pick","joint_prob_pick"])
        row = {"抽せん日": pd.to_datetime(pick_date),
               "候補_3桁_pick": fmt3(pick_num3),
               "EV_net_adj_pick": ev_adj,
               "joint_prob_pick": prob}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(PRED_HISTORY, index=False, encoding="utf-8-sig"); return
    if "抽せん日" not in df.columns:
        df["抽せん日"] = pd.NaT
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    mask = df["抽せん日"].dt.date == pick_date
    if not mask.any():
        row = {"抽せん日": pd.to_datetime(pick_date),
               "候補_3桁_pick": fmt3(pick_num3),
               "EV_net_adj_pick": ev_adj,
               "joint_prob_pick": prob}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df.loc[mask, "候補_3桁_pick"] = fmt3(pick_num3)
        if ev_adj is not None: df.loc[mask, "EV_net_adj_pick"] = float(ev_adj)
        if prob is not None:   df.loc[mask, "joint_prob_pick"] = float(prob)
    df.to_csv(PRED_HISTORY, index=False, encoding="utf-8-sig")


def _stable_merge_history(new_hist: pd.DataFrame) -> pd.DataFrame:
    base = read_csv_safe(PRED_HISTORY)
    if base is None or base.empty:
        if "抽せん日" in new_hist.columns:
            new_hist = _make_date_key(new_hist, "抽せん日")
        return new_hist.copy()
    if "抽せん日" in base.columns:     base = _make_date_key(base, "抽せん日")
    if "抽せん日" in new_hist.columns: new_hist = _make_date_key(new_hist, "抽せん日")
    all_cols = list(dict.fromkeys(list(base.columns) + list(new_hist.columns)))
    base2 = base.reindex(columns=all_cols); new2 = new_hist.reindex(columns=all_cols)
    exist_keys = set(base2["date_key"].dropna().unique())
    add_rows = new2[~new2["date_key"].isin(exist_keys)].copy()
    merged = pd.concat([base2, add_rows], ignore_index=True)
    if "抽せん日" in merged.columns:
        merged = merged.sort_values("抽せん日", ascending=False)
    return merged


def _write_stable_history_from_tmp(tmp_path: Path) -> None:
    tmp_df = read_csv_safe(tmp_path)
    if tmp_df is None or tmp_df.empty:
        st.warning("（注意）一時履歴CSVが空でした。履歴は更新しませんでした。"); return
    merged = _stable_merge_history(tmp_df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PRED_HISTORY, index=False, encoding="utf-8-sig")


def safe_to3(x) -> str:
    s = pd.to_numeric(pd.Series([x]), errors="coerce")
    if s.isna().iloc[0]: return ""
    return f"{int(s.iloc[0]):03d}"


def digit_boxes_html(three_digits: str) -> str:
    d0, d1, d2 = (list(three_digits) + ["", "", ""])[:3] if three_digits else ("", "", "")
    return f"""
<div style="display:flex;gap:10px;margin-top:4px;">
  <div style="display:inline-flex;align-items:center;justify-content:center;
              width:64px;height:64px;margin-right:10px;border:2px solid #111;
              border-radius:12px;background:#fff;color:#111;font-size:28px;font-weight:800;">{d0}</div>
  <div style="display:inline-flex;align-items:center;justify-content:center;
              width:64px;height:64px;margin-right:10px;border:2px solid #111;
              border-radius:12px;background:#fff;color:#111;font-size:28px;font-weight:800;">{d1}</div>
  <div style="display:inline-flex;align-items:center;justify-content:center;
              width:64px;height:64px;margin-right:10px;border:2px solid #111;
              border-radius:12px;background:#fff;color:#111;font-size:28px;font-weight:800;">{d2}</div>
</div>
""".strip()


def badge_html(label: str, value: str) -> str:
    return f"""
<div style="display:inline-flex;align-items:center;justify-content:center;
            min-width:140px;height:54px;margin-right:12px;border:1px solid #ddd;
            border-radius:12px;background:#f7f7f8;color:#111;font-size:18px;
            font-weight:700;padding:0 12px;">
  <div style="font-size:12px;color:#666;margin-right:8px;font-weight:600;">{label}</div>
  <div>{value}</div>
</div>
""".strip()


# --- 抽せん日ターゲット（JST・土日スキップ）
def _next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def compute_target_draw_date(hist_last_date_str: str) -> str:
    last_d = datetime.strptime(hist_last_date_str, "%Y-%m-%d").date()
    base = _next_weekday(last_d + timedelta(days=1))
    today = datetime.now(JST).date()
    target = base
    if target < today:
        target = _next_weekday(today)
    while target <= last_d:
        target = _next_weekday(target + timedelta(days=1))
    return target.isoformat()


def next_draw_from_history() -> date | None:
    hist = find_latest_history()
    if hist is None: return None
    try:
        df = pd.read_csv(hist, encoding="utf-8-sig")
        if "抽せん日" not in df.columns: return None
        dmax = pd.to_datetime(df["抽せん日"], errors="coerce").max()
        if pd.isna(dmax): return None
        hist_last = dmax.date().isoformat()
        target_str = compute_target_draw_date(hist_last)
        return datetime.strptime(target_str, "%Y-%m-%d").date()
    except Exception:
        return None


def next_index_from_history() -> str:
    hist = find_latest_history()
    if hist is None: return "—"
    try:
        df = pd.read_csv(hist, encoding="utf-8-sig", usecols=lambda c: c in ["抽せん日","回号"])
        if "回号" not in df.columns:
            return "—"
        df["抽せん日"] = pd.to_datetime(df.get("抽せん日"), errors="coerce")
        df = df[df["抽せん日"].notna()].copy()
        if df.empty:
            return "—"
        dmax = df["抽せん日"].max()
        m = pd.to_numeric(df.loc[df["抽せん日"] == dmax, "回号"], errors="coerce").dropna()
        if m.empty:
            m = pd.to_numeric(df["回号"], errors="coerce").dropna()
        return "—" if m.empty else f"{int(m.max()) + 1}"
    except Exception:
        return "—"


# ---- 候補_3桁の強制補完
def _ensure_cand3_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "候補_3桁" not in d.columns or d["候補_3桁"].isna().all():
        if "予測番号" in d.columns:
            d["候補_3桁"] = d["予測番号"]
        elif "番号" in d.columns:
            d["候補_3桁"] = d["番号"]
        elif all(c in d.columns for c in ["百","十","一"]):
            d["候補_3桁"] = (
                pd.to_numeric(d["百"], errors="coerce").astype("Int64").astype(str) +
                pd.to_numeric(d["十"], errors="coerce").astype("Int64").astype(str) +
                pd.to_numeric(d["一"], errors="coerce").astype("Int64").astype(str)
            )
        else:
            d["候補_3桁"] = ""
    d["候補_3桁"] = d["候補_3桁"].apply(fmt3).astype(str)
    if "候補_3桁_pick" not in d.columns:
        d["候補_3桁_pick"] = ""
    d["候補_3桁_pick"] = d["候補_3桁_pick"].apply(fmt3).astype(str).replace("nan","")
    return d


def _build_daily_rep_from_history() -> pd.DataFrame | None:
    hist = read_csv_safe(PRED_HISTORY)
    if hist is None or hist.empty: return None
    d = hist.copy()
    if "抽せん日" not in d.columns: return None
    d = _make_date_key(d, "抽せん日")
    d = _ensure_cand3_cols(d)
    d = ensure_joint_prob(d)
    d["_score_ev"] = pd.to_numeric(d.get("EV_net", 0), errors="coerce").fillna(-1)
    d["_score_p"]  = pd.to_numeric(d.get("joint_prob", 0), errors="coerce").fillna(-1)
    rep = (
        d.sort_values(["date_key","_score_ev","_score_p"], ascending=[True, False, False])
         .drop_duplicates(subset=["date_key"], keep="first")
         .loc[:, ["date_key","候補_3桁"]]
         .rename(columns={"候補_3桁":"cand3_rep"})
         .copy()
    )
    rep["cand3_rep"] = rep["cand3_rep"].apply(fmt3)
    return rep


# ============ 外部モデルの任意取得（Secrets: AZ_BLOB_URL_MODELS） ============
@st.cache_resource(show_spinner="モデルを読み込んでいます…", ttl=24*3600)
def _download_joint_model_to_cache() -> Path | None:
    url = st.secrets.get("AZ_BLOB_URL_MODELS")
    if not url:
        return None  # Secrets 未設定 → ローカルのモデルを使う
    try:
        import requests
        cache_dir = ROOT / ".cache" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / "joint_model.joblib"
        if not local_path.exists() or local_path.stat().st_size < 10 * 1024:  # 10KB 未満なら壊れとみなす
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        return local_path
    except Exception as e:
        st.warning(f"外部モデル取得に失敗しました（Secrets AZ_BLOB_URL_MODELS）。ローカルを使用します。detail={e}")
        return None


def resolve_models_v5_joint_path() -> Path:
    # 1) 外部DLが成功していればそれを使う（単一joblib想定で predict 側が対応している場合）
    dl = _download_joint_model_to_cache()
    if dl and dl.exists():
        return dl
    # 2) 従来の models ディレクトリを使う
    return MODELS_V5_JOINT


# ============ サイドバー（シンプル） ============
st.sidebar.header("⚡ クイック操作")
do_update  = st.sidebar.button("データ更新（scrape_update）", use_container_width=True)
do_refresh = st.sidebar.button("最新化（予測→EV）", use_container_width=True)

with st.sidebar.expander("⚙ 設定（基本）", expanded=True):
    payout_mode = st.radio("払戻の基準", ["実績（historyの金額を使う）", "固定（下の金額）"], index=0)
    if "実績" in payout_mode:
        payout_kind = st.selectbox("実績で使う列",
            ["ストレート_金額","ボックス_金額","セットS_金額","セットB_金額","ミニ_金額"], index=0)
    else:
        payout_kind = "ストレート_金額"
    c1, c2 = st.columns(2)
    with c1: price  = st.number_input("購入金額（円/口）", 100, 1000, DEFAULT_PRICE, 50)
    with c2: payout = st.number_input("払戻（固定モード）", 10000, 200000, DEFAULT_PAYOUT, 5000)

with st.sidebar.expander("🧪 デバッグ", expanded=False):
    hist_path = find_latest_history()
    if hist_path:
        st.write("使用中 history:", str(hist_path))
    else:
        st.error("history が見つかりません。data/raw を確認してください。")


with st.sidebar.expander("🛠 高度な操作（学習/バックフィル）", expanded=False):
    do_train = st.button("学習（V4）", use_container_width=True, key="train")
    do_backfill_hist = st.button("バックフィル（予測履歴）", use_container_width=True, key="bf_hist")
    do_backfill_ev   = st.button("バックフィル（EV）", use_container_width=True, key="bf_ev")


# ============ アクション実装 ============
def find_update_script() -> Path | None:
    for p in [
        SRC / "n3" / "scrape_update.py",
        ROOT / "data" / "scrape_update.py",
        ROOT / "scrape_update.py",
        SRC / "n3" / "scrape_all.py",
        ROOT / "data" / "scrape_all.py",
        ROOT / "scrape_all.py",
    ]:
        if p.exists(): return p
    return None


def find_train_v4_script() -> Path | None:
    for p in [
        SRC / "n3" / "training" / "train_evaluate_v4.py",
        SRC / "n3" / "training" / "train_evaluate.py",
        ROOT / "train_evaluate_v4.py",
        ROOT / "train_evaluate.py",
    ]:
        if p.exists(): return p
    return None


def find_backfill_script() -> Path | None:
    for p in [
        SRC / "n3" / "backfill" / "backfill_history.py",
        SRC / "n3" / "backfill" / "backfill_v4.py",
        ROOT / "backfill_history.py",
        ROOT / "backfill_v4.py",
    ]:
        if p.exists(): return p
    return None


if do_update:
    with st.status("データ更新中...", expanded=True) as s:
        if module_available("n3.scrape_update"):
            rc, out = run_py_module("n3.scrape_update", [])
            st.code(_tail(out), language="bash")
            s.update(label=("データ更新 完了 ✅" if rc == 0 else "データ更新 失敗 ❌"),
                     state=("complete" if rc == 0 else "error"))
        else:
            script = find_update_script()
            if script is None:
                s.update(label="データ更新 失敗 ❌", state="error")
                st.error("データ更新スクリプトが見つかりません。")
            else:
                rc, out = run_py_script(script, [])
                st.code(_tail(f"[INFO] use: {script}\n\n{out}"), language="bash")
                s.update(label=("データ更新 完了 ✅" if rc == 0 else "データ更新 失敗 ❌"),
                         state=("complete" if rc == 0 else "error"))


if do_refresh:
    with st.status("最新化を実行中...", expanded=True) as s:
        for pth in [NEXT_CSV, EV_CSV, PRED_HISTORY_TMP]:
            if pth.exists():
                try: pth.unlink()
                except Exception: pass
        rc1 = rc2 = 1; out1 = out2 = ""
        hist = find_latest_history()
        if hist is None:
            s.update(label="最新化 失敗 ❌", state="error")
            st.error("[ERROR] data/raw に *_Numbers3features.csv が見つかりません。先に『データ更新』を実行してください。")
        else:
            # 抽せん日ターゲット算出
            try:
                df_hist = pd.read_csv(hist, encoding="utf-8-sig")
                dmax = pd.to_datetime(df_hist["抽せん日"], errors="coerce").max()
                if pd.isna(dmax):
                    raise RuntimeError("history の抽せん日が読み取れません。")
                hist_last_iso = dmax.date().isoformat()
                target_str = compute_target_draw_date(hist_last_iso)
            except Exception as e:
                s.update(label="最新化 失敗 ❌", state="error")
                st.error(f"[ERROR] 抽せん日ターゲット決定に失敗: {e}")
                target_str = None

            # 1) 予測
            models_arg = str(resolve_models_v5_joint_path())
            rc1, out1 = run_py_module(PREDICT_MOD, [
                "--models_dir", models_arg,
                "--history",    str(hist),
                "--out",        str(NEXT_CSV),
                "--hist_out",   str(PRED_HISTORY_TMP),
                "--price",      str(int(price)),
                "--payout",     str(int(payout)),
                "--topn",       "1000",
            ])
            st.code(_tail(out1) or "(no output)", language="bash")

            # TMP 履歴の抽せん日補正 → 安定マージ
            if rc1 == 0 and PRED_HISTORY_TMP.exists():
                try:
                    tmp = read_csv_safe(PRED_HISTORY_TMP)
                    if tmp is not None and not tmp.empty and target_str:
                        if "抽せん日" not in tmp.columns:
                            tmp["抽せん日"] = target_str
                        else:
                            tmp["抽せん日"] = pd.to_datetime(tmp["抽せん日"], errors="coerce")
                            tmp["抽せん日"] = pd.to_datetime(target_str)
                        tmp.to_csv(PRED_HISTORY_TMP, index=False, encoding="utf-8-sig")
                except Exception:
                    pass
                _write_stable_history_from_tmp(PRED_HISTORY_TMP)
            elif rc1 != 0:
                st.warning("予測段階でエラーが発生しました。ログをご確認ください。")

            # 2) EV 生成（ローカル）
            try:
                df_next = read_csv_safe(NEXT_CSV)
                if df_next is None or df_next.empty:
                    raise RuntimeError("NEXT_CSV が空です。予測で失敗の可能性。")

                if target_str:
                    df_next["抽せん日"] = target_str

                if "候補_3桁" not in df_next.columns:
                    if all(c in df_next.columns for c in ["百","十","一"]):
                        df_next["候補_3桁"] = (
                            pd.to_numeric(df_next["百"], errors="coerce").fillna(0).astype(int).astype(str) +
                            pd.to_numeric(df_next["十"], errors="coerce").fillna(0).astype(int).astype(str) +
                            pd.to_numeric(df_next["一"], errors="coerce").fillna(0).astype(int).astype(str)
                        )
                    elif "候補番号" in df_next.columns:
                        df_next["候補_3桁"] = pd.to_numeric(df_next["候補番号"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(3)
                    elif "番号" in df_next.columns:
                        df_next["候補_3桁"] = pd.to_numeric(df_next["番号"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(3)
                    else:
                        df_next["候補_3桁"] = ""
                df_next["候補_3桁"] = df_next["候補_3桁"].map(fmt3)

                df_next = ensure_joint_prob(df_next)
                jp = pd.to_numeric(df_next["joint_prob"], errors="coerce").fillna(0.0).clip(0, 1)
                df_next["EV_gross"] = jp * float(payout)
                df_next["EV_net"]   = df_next["EV_gross"] - float(price)
                df_next = df_next.sort_values(["EV_net","EV_gross","joint_prob"], ascending=[False,False,False]).reset_index(drop=True)
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                df_next.to_csv(EV_CSV, index=False, encoding="utf-8-sig")
                out2 = "[OK] EV をアプリ内で計算して ev_report.csv を生成しました。"; rc2 = 0

                try:
                    uniq_dates = pd.to_datetime(df_next.get("抽せん日"), errors="coerce").dt.date.dropna().unique().tolist()
                    print(f"[INFO] history last : {hist_last_iso}")
                    print(f"[INFO] target draw  : {target_str} (JST today={datetime.now(JST).date()})")
                    print(f"[INFO] EV_CSV 抽せん日: {uniq_dates}")
                except Exception:
                    pass

            except Exception as e:
                out2 = f"[ERR] EV のローカル生成に失敗: {e}"; rc2 = 1
            st.code(out2, language="bash")

        s.update(label=("最新化 完了 ✅" if rc1 == 0 and rc2 == 0 and EV_CSV.exists() else "最新化 失敗 ❌"),
                 state=("complete" if rc1 == 0 and rc2 == 0 and EV_CSV.exists() else "error"))


# 学習・バックフィル
if 'do_train' in locals() and do_train:
    with st.status("学習中...", expanded=True) as s:
        hist = find_latest_history()
        if hist is None:
            s.update(label="学習 失敗 ❌", state="error")
            st.error("[ERR] data/raw の *_Numbers3features.csv が見つかりません。")
        else:
            module_name = "n3.training.train_evaluate_v4"
            args = ["--history", str(hist), "--models_dir", str(MODELS_V4),
                    "--use_xgb","1","--calibrate","1","--calib_method","isotonic",
                    "--valid_ratio","0.10","--test_ratio","0.20"]
            if module_available(module_name):
                rc, out = run_py_module(module_name, args)
                st.code(_tail(f"[INFO] history: {hist}\n\n{out}"), language="bash")
                s.update(label=("学習 完了 ✅" if rc == 0 else "学習 失敗 ❌"),
                         state=("complete" if rc == 0 else "error"))
            else:
                script = find_train_v4_script()
                if script is None:
                    s.update(label="学習 失敗 ❌", state="error"); st.error("train_evaluate_v4 が見つかりません。")
                else:
                    rc, out = run_py_script(script, args)
                    st.code(_tail(f"[INFO] use script: {script}\n[INFO] history: {hist}\n\n{out}"), language="bash")
                    s.update(label=("学習 完了 ✅" if rc == 0 else "学習 失敗 ❌"),
                             state=("complete" if rc == 0 else "error"))


if 'do_backfill_hist' in locals() and do_backfill_hist:
    with st.status("予測履歴のバックフィル中...", expanded=True) as s:
        hist = find_latest_history()
        if hist is None:
            s.update(label="バックフィル 失敗 ❌", state="error")
            st.error("history CSV が見つかりません。先に『データ更新』を実行してください。")
        else:
            module_name = "n3.backfill.backfill_history"
            if module_available(module_name):
                if PRED_HISTORY_TMP.exists():
                    try: PRED_HISTORY_TMP.unlink()
                    except Exception: pass
                rc, out = run_py_module(module_name, [
                    "--history", str(hist),
                    "--models_dir", str(MODELS_V4),
                    "--hist_out", str(PRED_HISTORY_TMP),
                    "--price", str(int(price)),
                    "--payout", str(int(payout)),
                ])
                st.code(_tail(out), language="bash")
                if rc == 0: _write_stable_history_from_tmp(PRED_HISTORY_TMP)
                s.update(label=("バックフィル 完了 ✅" if rc == 0 else "バックフィル 失敗 ❌"),
                         state=("complete" if rc == 0 else "error"))
            else:
                script = find_backfill_script()
                if script is None:
                    s.update(label="バックフィル 失敗 ❌", state="error"); st.error("backfill_history が見つかりません。")
                else:
                    if PRED_HISTORY_TMP.exists():
                        try: PRED_HISTORY_TMP.unlink()
                        except Exception: pass
                    rc, out = run_py_script(script, [
                        "--history", str(hist),
                        "--models_dir", str(MODELS_V4),
                        "--hist_out", str(PRED_HISTORY_TMP),
                        "--price", str(int(price)),
                        "--payout", str(int(payout)),
                    ])
                    st.code(_tail(f"[INFO] use script: {script}\n\n{out}"), language="bash")
                    if rc == 0: _write_stable_history_from_tmp(PRED_HISTORY_TMP)
                    s.update(label=("バックフィル 完了 ✅" if rc == 0 else "バックフィル 失敗 ❌"),
                             state=("complete" if rc == 0 else "error"))


if 'do_backfill_ev' in locals() and do_backfill_ev:
    with st.status("EVバックフィル中...", expanded=True) as s:
        hist_df = read_csv_safe(PRED_HISTORY)
        if hist_df is None or hist_df.empty:
            s.update(label="EVバックフィル 失敗 ❌", state="error")
            st.error("prediction_history.csv がありません。先に『バックフィル（予測履歴）』を実行してください。")
        else:
            df = hist_df.copy()
            if "候補_3桁" not in df.columns:
                if "予測番号" in df.columns:
                    df["候補_3桁"] = df["予測番号"].map(fmt3)
                elif all(c in df.columns for c in ["百","十","一"]):
                    df["候補_3桁"] = (
                        pd.to_numeric(df["百"], errors="coerce").astype("Int64").astype(str) +
                        pd.to_numeric(df["十"], errors="coerce").astype("Int64").astype(str) +
                        pd.to_numeric(df["一"], errors="coerce").astype("Int64").astype(str)
                    ).str.zfill(3)
                else:
                    df["候補_3桁"] = ""
            else:
                df["候補_3桁"] = df["候補_3桁"].map(fmt3)
            df = ensure_joint_prob(df)
            df["EV_gross"] = df["joint_prob"].clip(0,1) * float(payout)
            df["EV_net"]   = df["EV_gross"] - float(price)
            if "抽せん日" in df.columns:
                df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
                wdf = winner3_from_raw()
                if wdf is not None:
                    df = df.merge(wdf, on="抽せん日", how="left")
                    df["当選番号3"] = df.get("当選番号3","").map(fmt3)
                    df["hit"] = (df["候補_3桁"] != "") & (df["候補_3桁"] == df["当選番号3"])
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(EV_BACKFILL, index=False, encoding="utf-8-sig")
            s.update(label="EVバックフィル 完了 ✅（履歴由来）", state="complete")


# ============ 画面ヘッダ ============
st.title("Numbers3 Dashboard")
st.caption("データ更新 → 予測（EV生成）に特化。確率は常にモデル由来（joint_prob）。")

# 簡易ステータス（任意）
with st.expander("🩺 現在の状態（サマリ）", expanded=False):
    hist_p = find_latest_history()
    st.write("history:", str(hist_p) if hist_p else "—")
    st.write("EV_CSV:", str(EV_CSV), " / exists=", EV_CSV.exists())
    st.write("PRED_HISTORY:", str(PRED_HISTORY), " / exists=", Path(PRED_HISTORY).exists())
    try:
        if EV_CSV.exists():
            ev_rows = sum(1 for _ in open(EV_CSV, "r", encoding="utf-8-sig")) - 1
            st.write("EV 行数:", max(ev_rows, 0))
    except Exception:
        pass

d = next_draw_from_history()
draw_str = d.strftime("%Y年%m月%d日") if d else "—"
wday_str = weekday_ja(d) if d else "—"
idx_str  = next_index_from_history()

c1, c2, c3 = st.columns(3)
with c1: components.html(badge_html("抽せん日", draw_str), height=70)
with c2: components.html(badge_html("曜日",  wday_str),  height=70)
with c3: components.html(badge_html("回号",  idx_str),   height=70)

st.markdown("---")


# ============ EVレポート読込 & 並び ============
df_ev = read_csv_safe(EV_CSV)
if df_ev is None: df_ev = pd.DataFrame()
if not df_ev.empty:
    df_ev = ensure_joint_prob(df_ev)
    if "候補_3桁" not in df_ev.columns:
        if all(c in df_ev.columns for c in ["百","十","一"]):
            df_ev["候補_3桁"] = (
                pd.to_numeric(df_ev["百"], errors="coerce").fillna(0).astype(int).astype(str) +
                pd.to_numeric(df_ev["十"], errors="coerce").fillna(0).astype(int).astype(str) +
                pd.to_numeric(df_ev["一"], errors="coerce").fillna(0).astype(int).astype(str)
            )
        elif "候補番号" in df_ev.columns:
            df_ev["候補_3桁"] = pd.to_numeric(df_ev["候補番号"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(3)
        elif "番号" in df_ev.columns:
            df_ev["候補_3桁"] = pd.to_numeric(df_ev["番号"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(3)
        else:
            df_ev["候補_3桁"] = ""
    df_ev["候補_3桁"] = df_ev["候補_3桁"].map(fmt3)
    jp = pd.to_numeric(df_ev["joint_prob"], errors="coerce").fillna(0.0).clip(0,1)
    df_ev["EV_gross"] = jp * float(payout)
    df_ev["EV_net"]   = df_ev["EV_gross"] - float(price)
    sort_cols = [c for c in ["EV_net","EV_gross","joint_prob"] if c in df_ev.columns]
    df_ev = df_ev.sort_values(sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)


# ============ 最新Top1 ============
st.subheader("最新の予測（Top1）")
if not df_ev.empty:
    top = df_ev.iloc[0]
    num3 = safe_to3(top.get("候補_3桁", top.get("番号","")))
    components.html(digit_boxes_html(num3), height=90)
    target_date = next_draw_from_history() or date.today()
    st.session_state["latest_pick_num3"] = fmt3(num3)
    st.session_state["latest_pick_date"] = target_date
    persist_today_pick(
        pick_date=target_date,
        pick_num3=fmt3(num3),
        ev_adj=float(top.get("EV_net", 0)),
        prob=float(top.get("joint_prob", 0))
    )
else:
    st.info("EVレポートが見つかりません。左の『最新化（予測→EV）』を実行してください。")

st.markdown("---")


# ============ おすすめ Top3 ============
st.subheader("おすすめ Top3（EV順）")
cols = st.columns(3)
def card_html(rank: int, row: pd.Series, price: float, payout: float) -> str:
    if "候補_3桁" in row.index and str(row["候補_3桁"]):
        num3 = safe_to3(row["候補_3桁"])
    elif "番号" in row.index:
        num3 = safe_to3(row["番号"])
    elif all(c in row.index for c in ["百","十","一"]):
        try:
            num3 = f"{int(row['百'])}{int(row['十'])}{int(row['一'])}"
        except Exception:
            num3 = ""
    else:
        num3 = ""

    p_eff  = float(pd.to_numeric(pd.Series([row.get("joint_prob", 0)]), errors="coerce").fillna(0).iloc[0])
    ev_net = float(pd.to_numeric(pd.Series([row.get("EV_net", 0)]),       errors="coerce").fillna(0).iloc[0])
    fs     = str(row.get("feature_set", "—"))
    mdl    = str(row.get("model_name", "—"))

    return f"""
<div style="display:flex;flex-direction:column;gap:10px;border:1px solid #ddd;
            border-radius:14px;background:#fff;padding:16px;min-width:280px;
            max-width:360px;flex:1 1 0;box-shadow:0 1px 2px rgba(0,0,0,0.06);">
  <div style="display:flex;align-items:center;gap:8px;color:#888;font-size:12px;">
    <b style="color:#333">#{rank}</b>
    <span style="border:1px solid #eee;border-radius:999px;padding:2px 8px;background:#fafafa;">期待値で選定</span>
  </div>
  {digit_boxes_html(num3)}
  <div style="margin-top:6px;line-height:1.8;">
    <div>期待値: <b>{ev_net:,.0f} 円</b></div>
    <div>当選確率（モデル計算）: <b>{p_eff * 100:.2f}%</b></div>
  </div>
  <div style="margin-top:6px;font-size:12px;color:#666;">
    <span style="border:1px solid #eee;border-radius:999px;padding:2px 8px;background:#fafafa;">feature: {fs}</span>
    <span style="margin-left:6px;border:1px solid #eee;border-radius:999px;padding:2px 8px;background:#fafafa;">model: {mdl}</span>
  </div>
</div>
""".strip()

if df_ev.empty:
    for i in range(3):
        with cols[i]:
            st.info("データなし")
else:
    n = min(3, len(df_ev))
    for i in range(3):
        with cols[i]:
            if i < n:
                components.html(card_html(i+1, df_ev.iloc[i], price=float(price), payout=float(payout)), height=300)
            else:
                st.info("データなし")

if not df_ev.empty:
    st.download_button(
        "EVレポートをダウンロード（CSV）",
        df_ev.to_csv(index=False, encoding="utf-8-sig"),
        file_name="ev_report_view.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")


# ============ 検証（成績と信頼度） ============
st.subheader("検証（成績と信頼度）")
left, right = st.columns(2)
with left:
    days_window = st.selectbox("集計期間", ["30日","60日","90日","180日","365日","全期間"], index=2)
    days_map = {"30日":30,"60日":60,"90日":90,"180日":180,"365日":365,"全期間":None}
    K = days_map[days_window]
with right:
    if "実績" in payout_mode:
        st.info(f"払戻モード: 実績（{payout_kind}）")
    else:
        st.info(f"払戻モード: 固定（{payout:,} 円）")

def _load_for_eval() -> pd.DataFrame:
    df = read_csv_safe(EV_BACKFILL)
    if df is None or df.empty:
        df = read_csv_safe(PRED_HISTORY)
    if df is None: return pd.DataFrame()
    return df.copy()

# ==== PATCH A: 評価は 1日=1本 に正規化 ====
def _reduce_to_one_pick_for_eval(df: pd.DataFrame) -> pd.DataFrame:
    """評価用に 1日=1本 に正規化。
    優先順位: 候補_3桁_pick がある日→その行
              ない日      → EV_net 最大（なければ joint_prob 最大）
    """
    d = df.copy()
    # 日付キー
    date_col = next((c for c in ["抽せん日","date","draw_date"] if c in d.columns), None)
    d["date_key"] = pd.to_datetime(d[date_col], errors="coerce").dt.date

    # 文字列整形
    if "候補_3桁" not in d.columns: d["候補_3桁"] = ""
    d["候補_3桁"] = d["候補_3桁"].fillna("").astype(str).apply(fmt3)
    if "候補_3桁_pick" not in d.columns: d["候補_3桁_pick"] = ""
    d["候補_3桁_pick"] = d["候補_3桁_pick"].fillna("").astype(str).apply(fmt3)

    d["_has_pick"] = d["候補_3桁_pick"].ne("") & d["候補_3桁_pick"].ne("nan")
    d["__ev"] = pd.to_numeric(d.get("EV_net"), errors="coerce")
    d["__p"]  = pd.to_numeric(d.get("joint_prob"), errors="coerce").fillna(0.0)

    # pick一致を最優先、その次に EV, さらに prob
    d["_rank"] = np.where(d["_has_pick"] & (d["候補_3桁"] == d["候補_3桁_pick"]), 0, 1)
    d = d.sort_values(["date_key", "_rank", "__ev", "__p"], ascending=[True, True, False, False])
    d1 = d.drop_duplicates(subset=["date_key"], keep="first").copy()

    need = (d1["候補_3桁"] == "") | (d1["候補_3桁"] == "nan")
    d1.loc[need, "候補_3桁"] = d1.loc[need, "候補_3桁_pick"]
    d1["候補_3桁"] = d1["候補_3桁"].apply(fmt3)

    return d1.drop(columns=["_has_pick","_rank","__ev","__p"], errors="ignore")

df_eval = _load_for_eval()
if df_eval.empty:
    st.info("検証用データがありません。『最新化』や『バックフィル』を実行してください。")
else:
    date_col = None
    for c in ["抽せん日","date","draw_date"]:
        if c in df_eval.columns:
            date_col = c; break
    if date_col is None:
        st.warning("日付列が見つからないため検証をスキップします。")
    else:
        df_eval[date_col] = pd.to_datetime(df_eval[date_col], errors="coerce")
        df_eval = df_eval[df_eval[date_col].notna()].copy()
        df_eval["date_key"] = df_eval[date_col].dt.date

        if "候補_3桁" not in df_eval.columns: df_eval["候補_3桁"] = ""
        df_eval["候補_3桁"] = df_eval["候補_3桁"].fillna("").astype(str)
        if "候補_3桁_pick" not in df_eval.columns: df_eval["候補_3桁_pick"] = ""
        else: df_eval["候補_3桁_pick"] = df_eval["候補_3桁_pick"].fillna("").astype(str)

        mask_empty = (df_eval["候補_3桁"] == "") | (df_eval["候補_3桁"].str.lower()=="nan")
        df_eval.loc[mask_empty, "候補_3桁"] = df_eval.loc[mask_empty, "候補_3桁_pick"]
        df_eval["候補_3桁"] = df_eval["候補_3桁"].map(fmt3)

        if "当選番号3" not in df_eval.columns:
            wdf = winner3_from_raw()
            if wdf is not None:
                df_eval = df_eval.merge(wdf, left_on=date_col, right_on="抽せん日", how="left")
        if "当選番号3" in df_eval.columns:
            df_eval["当選番号3"] = df_eval["当選番号3"].map(fmt3)

        df_eval = ensure_joint_prob(df_eval)
        if ("hit" not in df_eval.columns) or df_eval["hit"].isna().all():
            if "当選番号3" in df_eval.columns:
                df_eval["hit"] = (df_eval["候補_3桁"] != "") & (df_eval["候補_3桁"] == df_eval["当選番号3"])
            else:
                df_eval["hit"] = False

        # ==== 1日=1本 に縮約 ====
        df_eval = _reduce_to_one_pick_for_eval(df_eval)

        if K is not None:
            dmax = df_eval[date_col].max()
            dmin = dmax - pd.Timedelta(days=K)
            df_win = df_eval[df_eval[date_col].between(dmin, dmax)].copy()
        else:
            df_win = df_eval.copy()

        # 払戻シリーズ（実績/固定）
        if "実績" in payout_mode:
            paymap = payouts_map_from_raw(payout_kind)  # date_key, 回号, 払戻_実績
            if paymap is not None and not paymap.empty:
                df_win["date_key"] = df_win[date_col].dt.date
                df_win = df_win.merge(paymap, on="date_key", how="left")
                payout_series = pd.to_numeric(df_win.get("払戻_実績"), errors="coerce")
                # ==== 1万〜30万でガード ====
                payout_series = payout_series.where(
                    (payout_series >= 10000) & (payout_series <= 300000),
                    np.nan
                ).fillna(float(payout))
            else:
                st.warning("payouts_map_from_raw の結果が空でした。列名やデータを確認してください。")
                payout_series = pd.Series(float(payout), index=df_win.index)
        else:
            payout_series = pd.Series(float(payout), index=df_win.index)

        # 念のため最終クリップ
        payout_series = pd.to_numeric(payout_series, errors="coerce").fillna(float(payout)).clip(10000, 300000)

        df_win["日付"] = df_win[date_col].dt.date
        df_win["spent"]  = float(price)
        df_win["return"] = df_win["hit"].map(lambda x: 1 if x else 0) * payout_series
        daily = df_win.groupby("日付", as_index=False).agg(
            picks=("候補_3桁","count"), hits=("hit","sum"),
            spent=("spent","sum"), ret=("return","sum"),
        )
        daily["profit"] = daily["ret"] - daily["spent"]
        daily = daily.sort_values("日付")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("期間内 Picks", int(daily["picks"].sum()) if not daily.empty else 0)
        with c2: st.metric("期間内 Hits",  int(daily["hits"].sum()) if not daily.empty else 0)
        with c3: st.metric("総消費", f"{daily['spent'].sum():,.0f} 円" if not daily.empty else "0 円")
        with c4: st.metric("総払戻", f"{daily['ret'].sum():,.0f} 円" if not daily.empty else "0 円")

        if not daily.empty:
            cum = daily.copy(); cum["cum_profit"] = cum["profit"].cumsum()
            st.markdown("**累積利益の推移（選択期間）**")
            st.line_chart(cum.set_index(pd.to_datetime(cum["日付"]))["cum_profit"])

        if "joint_prob" in df_win.columns:
            st.markdown("**予測確率のキャリブレーション（10ビン）**")
            svals = pd.to_numeric(df_win["joint_prob"], errors="coerce").fillna(0.0).clip(0, 1)

            # 等幅ビン
            bins = np.linspace(0.0, 1.0, 11)
            labels = [f"{int(a*100)}〜{int(b*100)}%" for a,b in zip(bins[:-1], bins[1:])]
            df_cal = pd.DataFrame({"p": svals, "hit": df_win["hit"].astype(bool)})
            df_cal["bin"] = pd.cut(df_cal["p"], bins=bins, labels=labels, include_lowest=True, right=True)

            cal = df_cal.groupby("bin", as_index=False, observed=True).agg(
                mean_p=("p", "mean"),
                acc=("hit", "mean"),
                n=("p", "count"),
            )
            cal["mean_p_pct"] = (cal["mean_p"] * 100).round(2)
            cal["acc_pct"]    = (cal["acc"] * 100).round(2)
            cal["range_label"] = cal["bin"].astype(str)
            cal["diff_pct"] = (cal["acc_pct"] - cal["mean_p_pct"]).round(2)
            def _note(d):
                if pd.isna(d): return ""
                if d >= 1.0:   return "控えめ（実測＞予測）"
                if d <= -1.0:  return "過信（予測＞実測）"
                return "概ね一致"
            cal["note"] = cal["diff_pct"].apply(_note)

            show = cal[["range_label", "n", "mean_p_pct", "acc_pct", "diff_pct", "note"]].copy()
            show.columns = ["予測確率の範囲", "件数", "平均予測確率（%）", "実際の当たり率（%）", "差（実測−予測）", "評価"]
            st.dataframe(
                show,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "予測確率の範囲": st.column_config.TextColumn(width="medium"),
                    "件数": st.column_config.NumberColumn(format="%d"),
                    "平均予測確率（%）": st.column_config.NumberColumn(format="%.2f"),
                    "実際の当たり率（%）": st.column_config.NumberColumn(format="%.2f"),
                    "差（実測−予測）": st.column_config.NumberColumn(format="%.2f"),
                    "評価": st.column_config.TextColumn(),
                },
            )

            ideal = pd.DataFrame({"x":[0,100], "y":[0,100]})
            points = alt.Chart(cal).mark_line(point=True).encode(
                x=alt.X("mean_p_pct", title="平均予測確率（%）",
                        scale=alt.Scale(domain=[0, max(100, float(cal["mean_p_pct"].max() or 0)+5)])),
                y=alt.Y("acc_pct",     title="実際の当たり率（%）",
                        scale=alt.Scale(domain=[0, max(100, float(cal["acc_pct"].max() or 0)+5)])),
                tooltip=["range_label","n","mean_p_pct","acc_pct","diff_pct","note"]
            )
            labels = points.mark_text(align="left", dx=6, dy=-6).encode(text="range_label")
            ideal_line = alt.Chart(ideal).mark_line(strokeDash=[6,4], color="gray").encode(x="x", y="y")
            chart = (ideal_line + points + labels).properties(
                width="container", height=360, title="予測確率の信頼度カーブ（y=x が理想）"
            ).configure_axis(grid=True)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("joint_prob が見つからないためキャリブレーションは省略しました。")

st.markdown("---")


# ============ 直近の予測履歴 ============
st.markdown("### 直近の予測履歴")
rows_option = st.selectbox("表示件数", ["直近30件", "直近60件", "直近120件", "全件"], index=0)
rows_map = {"直近30件": 30, "直近60件": 60, "直近120件": 120, "全件": None}
N = rows_map[rows_option]

hist = read_csv_safe(PRED_HISTORY)
if hist is None or hist.empty:
    st.info("予測履歴がまだありません。左の『最新化（予測→EV）』実行後にご確認ください。")
else:
    dfh = hist.copy()
    dfh = _make_date_key(dfh, "抽せん日")
    dfh = _ensure_cand3_cols(dfh)
    dfh = ensure_joint_prob(dfh)

    dfh["_score_ev"] = pd.to_numeric(dfh.get("EV_net", 0), errors="coerce").fillna(-1)
    dfh["_score_p"]  = pd.to_numeric(dfh.get("joint_prob", 0), errors="coerce").fillna(-1)

    dfh = (
        dfh.sort_values(["date_key", "_score_ev", "_score_p"], ascending=[False, False, False])
           .drop_duplicates(subset=["date_key"], keep="first")
           .copy()
           .drop(columns=["_score_ev","_score_p"], errors="ignore")
    )

    dfh["候補_3桁_view"] = dfh["候補_3桁_pick"].replace({"": None, "nan": None}).fillna(dfh["候補_3桁"])
    need_fill = dfh["候補_3桁_view"].isna() | (dfh["候補_3桁_view"] == "") | (dfh["候補_3桁_view"].str.lower() == "nan")
    if need_fill.any():
        rep = _build_daily_rep_from_history()
        if rep is not None and not rep.empty:
            dfh = dfh.merge(rep, on="date_key", how="left")
            dfh.loc[need_fill, "候補_3桁_view"] = dfh.loc[need_fill, "cand3_rep"].fillna(dfh.loc[need_fill, "候補_3桁_view"])
            dfh.drop(columns=["cand3_rep"], inplace=True, errors="ignore")
    dfh["候補_3桁_view"] = dfh["候補_3桁_view"].apply(fmt3).replace("", "—")

    # 当選番号＆回号
    def _load_answer_index_map() -> pd.DataFrame | None:
        p = find_latest_history()
        if p is None:
            return None
        try:
            raw = pd.read_csv(
                p, encoding="utf-8-sig",
                usecols=lambda c: c in ["抽せん日","当せん番号","当選番号","百の位","十の位","一の位","回号"]
            )
            raw["抽せん日"] = pd.to_datetime(raw["抽せん日"], errors="coerce")
            raw = raw[raw["抽せん日"].notna()].copy()
            raw["date_key"] = raw["抽せん日"].dt.date
            if "当選番号" in raw.columns:
                base = pd.to_numeric(raw["当選番号"], errors="coerce")
            else:
                base = pd.to_numeric(raw.get("当せん番号"), errors="coerce")
            if base is not None:
                raw["当選番号3"] = base.apply(fmt3)
            else:
                raw["当選番号3"] = (
                    pd.to_numeric(raw.get("百の位"), errors="coerce").astype("Int64").astype(str) +
                    pd.to_numeric(raw.get("十の位"), errors="coerce").astype("Int64").astype(str) +
                    pd.to_numeric(raw.get("一の位"), errors="coerce").astype("Int64").astype(str)
                ).str.zfill(3).apply(fmt3)
            raw["回号"] = pd.to_numeric(raw.get("回号"), errors="coerce").astype("Int64")
            raw = raw.sort_values("抽せん日").drop_duplicates("date_key", keep="last")
            return raw[["date_key","当選番号3","回号"]].copy()
        except Exception:
            return None

    ans = _load_answer_index_map()
    if ans is not None and not ans.empty:
        dfh = dfh.merge(ans, on="date_key", how="left")
    else:
        dfh["当選番号3"] = pd.NA
        dfh["回号"] = pd.NA

    if "回号_x" in dfh.columns or "回号_y" in dfh.columns:
        dfh["回号"] = dfh.get("回号_x").combine_first(dfh.get("回号_y"))
        dfh.drop(columns=[c for c in ["回号_x", "回号_y"] if c in dfh.columns], inplace=True)

    for c in ["当選番号3", "回号"]:
        if c not in dfh.columns:
            dfh[c] = pd.NA

    dfh["当選番号3"] = dfh["当選番号3"].fillna("").apply(fmt3)
    dfh["回号表示"] = (
        pd.to_numeric(dfh["回号"], errors="coerce")
          .astype("Int64")
          .astype(str)
          .replace("<NA>", "—")
    )

    # 未抽選日の行は除外
    dfh = dfh[dfh["当選番号3"].notna() & (dfh["当選番号3"] != "")]

    JA_WD = ["月曜日","火曜日","水曜日","木曜日","金曜日","土曜日","日曜日"]
    dfh["抽せん日"] = pd.to_datetime(dfh["抽せん日"], errors="coerce")
    dfh["抽せん日_表示"] = dfh["抽せん日"].dt.strftime("%Y年%m月%d日")
    dfh["曜日"] = dfh["抽せん日"].dt.weekday.map(lambda i: JA_WD[i] if pd.notna(i) else "")

    dfh["候補_3桁_view"] = dfh["候補_3桁_view"].fillna("").apply(fmt3)
    dfh["的中"] = (dfh["候補_3桁_view"] != "") & (dfh["候補_3桁_view"] == dfh["当選番号3"])

    # EV 表示フォールバック
    dfh["joint_prob"] = pd.to_numeric(dfh.get("joint_prob"), errors="coerce").fillna(0.0)
    dfh["EV_net"] = pd.to_numeric(dfh.get("EV_net"), errors="coerce")
    dfh["EV_net_view"] = dfh["EV_net"]

    need_ev = dfh["EV_net_view"].isna() | (dfh["EV_net_view"] == 0)
    if "EV_net_adj_pick" in dfh.columns:
        adj = pd.to_numeric(dfh["EV_net_adj_pick"], errors="coerce")
        dfh.loc[need_ev & adj.notna(), "EV_net_view"] = adj

    still = dfh["EV_net_view"].isna()
    if still.any():
        if "実績" in payout_mode:
            paymap = payouts_map_from_raw(payout_kind)
            if paymap is not None and not paymap.empty:
                if "date_key" not in dfh.columns:
                    dfh = _make_date_key(dfh, "抽せん日")
                dfh = dfh.merge(paymap[["date_key","払戻_実績"]], on="date_key", how="left")
                pays = pd.to_numeric(dfh.get("払戻_実績"), errors="coerce")
                # 1万〜30万でガード
                pays = pays.where((pays >= 10000) & (pays <= 300000), np.nan).fillna(float(payout))
            else:
                pays = pd.Series(float(payout), index=dfh.index)
        else:
            pays = pd.Series(float(payout), index=dfh.index)

        pays = pd.to_numeric(pays, errors="coerce").fillna(float(payout)).clip(10000, 300000)
        jp = pd.to_numeric(dfh.get("joint_prob"), errors="coerce").fillna(0.0).clip(0,1)
        dfh.loc[still, "EV_net_view"] = (jp * pays - float(price)).loc[still]

    view = pd.DataFrame({
        "抽選日": dfh["抽せん日_表示"].fillna("—"),
        "曜日": dfh["曜日"].fillna(""),
        "回号": dfh["回号表示"],
        "候補_3桁": dfh["候補_3桁_view"],
        "当選番号": dfh["当選番号3"].replace("", "—"),
        "当選確率（%）": (dfh["joint_prob"] * 100).round(2),
        "期待値（円）": dfh["EV_net_view"].round(0),
        "的中": dfh["的中"].astype(bool),
    })

    if N is not None:
        view = view.head(N)

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "抽選日": st.column_config.TextColumn(),
            "曜日": st.column_config.TextColumn(),
            "回号": st.column_config.TextColumn(),
            "候補_3桁": st.column_config.TextColumn(),
            "当選番号": st.column_config.TextColumn(),
            "当選確率（%）": st.column_config.NumberColumn(format="%.2f"),
            "期待値（円）": st.column_config.NumberColumn(format="%,.0f"),
            "的中": st.column_config.CheckboxColumn(),
        }
    )

    st.download_button(
        "この表示内容をダウンロード（CSV）",
        view.to_csv(index=False, encoding="utf-8-sig"),
        file_name="prediction_history_view.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============ 詳細テーブル（参考） ============
with st.expander("📊 詳細テーブル（上位200行）", expanded=False):
    if not df_ev.empty:
        st.dataframe(df_ev.head(200), use_container_width=True, hide_index=True)
    else:
        st.write("（なし）")
