# app.py — Numbers3 EV Dashboard（ビュー専用版：joint最新化ボタン削除）
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

EV_CSV           = OUT_DIR / "ev_report.csv"
NEXT_CSV         = OUT_DIR / "next_prediction.csv"
EV_BACKFILL      = OUT_DIR / "ev_backfill.csv"
PRED_HISTORY     = OUT_DIR / "prediction_history.csv"
PRED_HISTORY_TMP = OUT_DIR / "prediction_history.tmp.csv"  # 安定マージ用一時

MODELS_V4 = ROOT / "artifacts" / "models_V4_XGB"

JST = timezone(timedelta(hours=9))

# --- secrets 安全取得ヘルパ
def _secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

DEFAULT_PRICE  = int(_secret("N3_PRICE",  200))
DEFAULT_PAYOUT = int(_secret("N3_PAYOUT", 90000))

# ============ ページ設定 & グローバルCSS ============
st.set_page_config(
    page_title="Numbers3 EV Dashboard",
    page_icon="🎯",
    layout="wide",
)

def inject_global_css() -> None:
    """ダッシュボード全体の見た目（テーマ）を整える CSS."""
    st.markdown("""
        <style>
        /* メイン背景色 */
        .main {
            background-color: #F4F7FA;
        }

        /* コンテナ横幅調整 */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* サイドバー */
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
        }
        section[data-testid="stSidebar"] * {
            color: #E5E7EB !important;
        }

        /* 共通カードスタイル */
        .subaru-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 2px 6px rgba(15,23,42,0.08);
            border: 1px solid #E2E8F0;
        }
        .subaru-card-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #64748B;
        }
        .subaru-card-value {
            font-weight: 700;
            font-size: 1.6rem;
            letter-spacing: 0.08em;
            color: #0F172A;
        }
        .subaru-card-sub {
            font-size: 0.80rem;
            color: #6B7280;
            margin-top: 0.35rem;
            line-height: 1.6;
        }

        /* 小さなラベルピル */
        .subaru-pill {
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 500;
            background: #EFF6FF;
            color: #1D4ED8;
            margin-left: 0.25rem;
        }

        /* EV 高低で色を分けたい場合のクラス（必要に応じて使う） */
        .ev-positive {
            color: #16A34A;
            font-weight: 600;
        }
        .ev-negative {
            color: #DC2626;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

inject_global_css()


# ====== ライブ配信エリア ======
st.title("Numbers3 抽せんライブ中継")

st.components.v1.iframe(
    "https://takarakuji.webcdn.stream.ne.jp/www11/takarakuji/live/index.html",
    width=800,
    height=450
)


# ====== 最終更新日時の表示（ダークモード対応） ======
def get_last_update_time():
    """
    ビュー専用なので、
    1. ローカルで生成して Git に上げる ev_report.csv（EV_CSV）の mtime を最優先
    2. フォールバックとして prediction_history.csv（PRED_HISTORY）の mtime
    を最終更新日時として表示する。
    """
    # ① EVレポートの更新日時（ローカルでの最新化タイミングにほぼ一致）
    if EV_CSV.exists():
        ts = datetime.fromtimestamp(EV_CSV.stat().st_mtime, JST)
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    # ② 履歴ファイルの更新日時（EV_CSV がまだ無い場合）
    if PRED_HISTORY.exists():
        ts = datetime.fromtimestamp(PRED_HISTORY.stat().st_mtime, JST)
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    # ③ どちらも無い
    return "—（まだ生成されていません）"


last_update = get_last_update_time()

st.markdown(
    f"""
    <div style="
        padding:12px 18px;
        margin-bottom:18px;
        border-radius:10px;
        background:#2b2b2b;
        color:#ffffff;
        border:1px solid #555;
        font-size:16px;
        ">
        <b>🔄 最終更新:</b> {last_update}
    </div>
    """,
    unsafe_allow_html=True
)


# ============ ユーティリティ ============
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
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return 1, f"[runner-error] {e}"

def run_py_module(module: str, args: list[str]) -> tuple[int, str]:
    return run([sys.executable, "-m", module, *args])

def run_py_script(path: Path, args: list[str]) -> tuple[int, str]:
    return run([sys.executable, str(path), *args])

def read_csv_safe(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        return None

# --- 重要：キャッシュ無効化のため data/raw の mtime を引数に入れる
@st.cache_data(ttl=1800)
def find_latest_history(_dir_mtime: float | None = None) -> Path | None:
    if not DATA_RAW.exists():
        return None
    cands = list(DATA_RAW.glob("*_Numbers3features.csv"))
    if not cands:
        return None
    # mtime優先、同率なら名前で決め打ち
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
    p = find_latest_history(DATA_RAW.stat().st_mtime if DATA_RAW.exists() else None)
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
    方針：`ストレート_金額` は **1口あたりの固定金額** としてそのまま採用する。
    - 口数による割戻しは一切しない
    - 10,000〜300,000 の範囲に正規化（異常値は NaN として落とす）
    - 同日重複は最後のレコードを優先
    返す列: date_key, 回号, 払戻_実績
    """
    hist_path = find_latest_history(DATA_RAW.stat().st_mtime if DATA_RAW.exists() else None)
    if hist_path is None:
        return None
    try:
        raw = pd.read_csv(hist_path, encoding="utf-8-sig")
        if "抽せん日" not in raw.columns:
            return None

        raw["抽せん日"] = pd.to_datetime(raw["抽せん日"], errors="coerce")
        raw = raw[raw["抽せん日"].notna()].copy()
        raw["date_key"] = raw["抽せん日"].dt.date

        if kind not in raw.columns:
            alt_names = ["ストレート", "ストレート_1口", "ストレート(1口)", "ストレート_1口あたり"]
            use_col = next((c for c in alt_names if c in raw.columns), None)
            if use_col is None:
                st.info(f"payouts_map_from_raw: '{kind}' 列が見つかりません。")
                return None
        else:
            use_col = kind

        per_unit = pd.to_numeric(raw[use_col], errors="coerce")
        valid = (per_unit >= 10000) & (per_unit <= 300000)
        per_unit = per_unit.where(valid, np.nan)

        df = raw[["date_key", "回号"]].copy()
        df["払戻_実績"] = per_unit

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
    """当日のTop1を prediction_history.csv に first-write-wins で保存。
       - 列の dtype は string に統一して将来の pandas エラーを回避。
    """
    df = read_csv_safe(PRED_HISTORY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 既存がない場合の初期 DataFrame（すべて string / float 可）
    if df is None or df.empty:
        df = pd.DataFrame({
            "抽せん日": pd.Series([], dtype="datetime64[ns]"),
            "候補_3桁_pick": pd.Series([], dtype="string"),
            "EV_net_adj_pick": pd.Series([], dtype="float64"),
            "joint_prob_pick": pd.Series([], dtype="float64"),
        })

    # 必須列の dtype 整備
    if "抽せん日" not in df.columns:
        df["抽せん日"] = pd.Series([], dtype="datetime64[ns]")
    if "候補_3桁_pick" not in df.columns:
        df["候補_3桁_pick"] = pd.Series([], dtype="string")
    else:
        df["候補_3桁_pick"] = df["候補_3桁_pick"].astype("string")

    if "EV_net_adj_pick" not in df.columns:
        df["EV_net_adj_pick"] = pd.Series([], dtype="float64")
    if "joint_prob_pick" not in df.columns:
        df["joint_prob_pick"] = pd.Series([], dtype="float64")

    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    mask = df["抽せん日"].dt.date == pick_date

    if not mask.any():
        row = {
            "抽せん日": pd.to_datetime(pick_date),
            "候補_3桁_pick": str(fmt3(pick_num3)),
            "EV_net_adj_pick": float(ev_adj) if ev_adj is not None else np.nan,
            "joint_prob_pick": float(prob) if prob is not None else np.nan,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df.loc[mask, "候補_3桁_pick"] = str(fmt3(pick_num3))
        if ev_adj is not None: df.loc[mask, "EV_net_adj_pick"] = float(ev_adj)
        if prob  is not None: df.loc[mask, "joint_prob_pick"] = float(prob)

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


# === 今日の予測カード用ユーティリティ ===
@st.cache_data
def load_next_prediction() -> pd.DataFrame:
    if NEXT_CSV.exists():
        try:
            return pd.read_csv(NEXT_CSV, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(NEXT_CSV)
    return pd.DataFrame()

def render_today_cards(df_next: pd.DataFrame, topn: int = 3) -> None:
    """本日の TopN 予測をカード風に表示する."""
    if df_next is None or df_next.empty:
        st.info("まだ next_prediction.csv がありません。ローカルで最新化スクリプトを実行してください。")
        return

    # 抽せん日（列名は環境に合わせて調整）
    draw_date = str(df_next.iloc[0].get("抽せん日", "")).split(" ")[0]

    st.markdown(f"### 🎯 今日の予測候補 — {draw_date}")

    df_top = df_next.head(topn)

    cols = st.columns(len(df_top))
    for i, (_, row) in enumerate(df_top.iterrows()):
        with cols[i]:
            # 候補番号
            num = row.get("候補_3桁", row.get("予測番号", row.get("番号", "???")))
            try:
                num_text = f"{int(num):03d}"
            except Exception:
                num_text = fmt3(num)

            # 確率
            prob = row.get("joint_prob", np.nan)
            try:
                prob = float(prob)
            except Exception:
                prob = np.nan
            prob_text = f"{prob:.1%}" if isinstance(prob, (float, int)) and not np.isnan(prob) else "—"

            # EV（候補列をいくつか見る）
            ev = None
            for key in ["EV_net", "EV_net_adj", "EV", "ev"]:
                if key in row.index:
                    try:
                        ev = float(row.get(key))
                        break
                    except Exception:
                        continue
            ev_text = f"{ev:,.0f} 円" if isinstance(ev, (float, int)) and not np.isnan(ev) else "—"

            st.markdown(f"""
                <div class="subaru-card">
                  <div class="subaru-card-title">
                    第 {i+1} 候補
                    <span class="subaru-pill">Top {i+1}</span>
                  </div>
                  <div class="subaru-card-value">
                    {num_text}
                  </div>
                  <div class="subaru-card-sub">
                    予測確率: {prob_text}<br>
                    EV(期待値): {ev_text}
                  </div>
                </div>
            """, unsafe_allow_html=True)


# === 説明カード（UX強化） ===
def render_explanation_cards() -> None:
    """Numbers3 予測ロジックや使い方の説明カード群."""
    st.markdown("## ℹ 説明・ヘルプ")

    with st.expander("🔍 このダッシュボードの予測ロジック（ざっくり）", expanded=False):
        st.markdown("""
- このダッシュボードは、**機械学習モデルで 000〜999 の全パターンの確率を予測**しています。
- 過去の抽せんデータから特徴量（直近の出目傾向・出現頻度・曜日など）を作成し、
  **1000クラス分類モデル** で「次に出そうな 3 桁の番号」を計算しています。
- モデルは、すべての番号に対して「確率（joint_prob）」を出し、
  その中から **上位 TopN（例: 20件）だけをランキング** して `next_prediction.csv` に保存します。
- ダッシュボード上部のカードは、その中でも **特に上位の候補（例: Top3）** を抜き出したものです。
        """)

    with st.expander("🎲 なぜ前日と同じ番号が出ることがあるの？（pick が連続しすぎ問題）", expanded=False):
        st.markdown("""
**「前日とまったく同じ候補が出ているけど、バグじゃない？」**
というパターンについての説明です。

- モデルは「過去の出目パターン」から **確率の高い番号** を選んでいます。
- 直近の傾向が似ている場合、
  → **「昨日と今日で“条件”がほとんど同じ」** と判断されることがあります。
- その結果として、
  → **同じ番号が Top 候補として連続して選ばれる** ことがあります。

これは、

> 「同じ数字をゴリ押ししている」のではなく、
> **「似た状況では同じ答えを返す」という統計モデルの自然な動き**

です。

ただし、内部では次のようなチェックも行っています（運用方針）：

- 予測用データ（特徴量）が正しく更新されているか
- 同じ特徴量で毎回走っていないか（履歴読み込みのエラーなど）
- 予測結果の分布が極端に偏っていないか

これらを満たした上で **同じ番号が続く** 場合は、
「モデルがその番号をかなり有望と見ている状態」と解釈してください。
        """)

    with st.expander("💰 EV（期待値）の見方", expanded=False):
        st.markdown("""
このダッシュボードでは、**1口200円 / 当たり90,000円（ストレート）** を前提に
各番号の **EV（期待値）** を計算しています。

- ある番号の的中確率を `p` とします。
- この番号を 1口だけ買ったときの期待値は

> `EV = p × 90,000円 − 200円`

となります。

- `EV > 0` なら、**理論上は「買えば買うほど得」な番号**
- `EV < 0` なら、**理論上は「長期的にはマイナス」な番号**

として解釈できます。

ただし、Numbers3 はもともと **1/1000 の運ゲー** なので、

- EV がプラスでも「単発で当たる保証」はありません。
- あくまで「長期的に同じ条件で買い続けた場合の平均的な期待値」です。
        """)

    with st.expander("🧪 予測の更新タイミングと履歴の扱い", expanded=False):
        st.markdown("""
- 元データ（*_Numbers3features.csv）が更新されると、
  新しい履歴をもとに **最新1件分の予測** を行います。
- 予測結果は
  - `next_prediction.csv`（最新の TopN 候補）
  - `ev_report.csv`（EV順に並べたレポート）
  - `prediction_history.csv`（過去の予測履歴）
  として保存され、ダッシュボードで参照しています。

運用上のポイント：

- **最新日の1件だけをターゲットに予測** しているため、
  毎日スクリプトを回すだけで「今日の予測」が自動更新されます。
- 過去のバックテストや傾向を見るときは、
  `prediction_history.csv` を使って「どの番号をいつ推していたか」を検証できます。
        """)

    with st.expander("⚠ ご利用上の注意（免責）", expanded=False):
        st.markdown("""
- このツールは **当せんを保証するものではありません**。
- すべての予測は、過去データに基づく **統計的な推定** に過ぎません。
- Numbers3 は本質的にランダム性が強く、
  **短期的には「良い予測」でも外れることが普通にあります。**
- 購入は **余裕資金の範囲内で、自己責任で** お願いします。
        """)


# --- 抽せん日ターゲット（JST・土日スキップ） ---
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
    hist = find_latest_history(DATA_RAW.stat().st_mtime if DATA_RAW.exists() else None)
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
    hist = find_latest_history(DATA_RAW.stat().st_mtime if DATA_RAW.exists() else None)
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

# ---- 候補_3桁の強制補完 ----
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
        d["候補_3桁_pick"] = pd.Series([""]*len(d), dtype="string")
    else:
        d["候補_3桁_pick"] = d["候補_3桁_pick"].astype("string")
    d["候補_3桁_pick"] = d["候補_3桁_pick"].apply(fmt3).astype("string").replace("nan","")
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


# ============ サイドバー（シンプル） ============
st.sidebar.header("⚡ クイック操作（ビュー専用）")
st.sidebar.info(
    "このアプリは『ビュー専用』です。\n"
    "- 予測とEV計算はローカルPCで実行\n"
    "- ev_report.csv / prediction_history.csv などを GitHub に push\n"
    "- ここではその結果だけを表示します。"
)

# データ更新はローカル用に残しておく（Cloudでは基本使わない想定）
do_update = st.sidebar.button("データ更新（scrape_update）", use_container_width=True)

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
    try:
        st.write("DATA_RAW:", str(DATA_RAW))
        if DATA_RAW.exists():
            files = sorted(DATA_RAW.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for p in files[:20]:
                st.write(p.name, "-", f"{p.stat().st_size:,}", "bytes")
        else:
            st.write("(data/raw がありません)")
    except Exception as e:
        st.write(f"(list error: {e})")

with st.sidebar.expander("🧹 キャッシュ", expanded=False):
    if st.button("Cache クリア & 再実行", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

with st.sidebar.expander("🛠 高度な操作（学習/バックフィル）", expanded=False):
    do_train = st.button("学習（V4）", use_container_width=True, key="train")
    do_backfill_hist = st.button("バックフィル（予測履歴）", use_container_width=True, key="bf_hist")
    do_backfill_ev   = st.button("バックフィル（EV）", use_container_width=True, key="bf_ev")


# ============ データ更新（ローカル用） ============
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
    with st.status("データ更新中...(ローカル用)", expanded=True) as s:
        if module_available("n3.scrape_update"):
            rc, out = run_py_module("n3.scrape_update", [])
            st.code(out, language="bash")
            ok = (rc == 0)
            s.update(label=("データ更新 完了 ✅" if ok else "データ更新 失敗 ❌"),
                     state=("complete" if ok else "error"))
        else:
            script = find_update_script()
            if script is None:
                s.update(label="データ更新 失敗 ❌", state="error")
                st.error("データ更新スクリプトが見つかりません。")
                ok = False
            else:
                rc, out = run_py_script(script, [])
                st.code(f"[INFO] use: {script}\n\n{out}", language="bash")
                ok = (rc == 0)
                s.update(label=("データ更新 完了 ✅" if ok else "データ更新 失敗 ❌"),
                         state=("complete" if ok else "error"))

        if ok:
            st.cache_data.clear()
            st.success("データ更新のキャッシュをクリアしました。")
            st.rerun()


# ============ 学習・バックフィル（必要ならローカルで利用） ============
if 'do_train' in locals() and do_train:
    with st.status("学習中...(ローカル用)", expanded=True) as s:
        hist = find_latest_history(DATA_RAW.stat().st_mtime if DATA_RAW.exists() else None)
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
                st.code(f"[INFO] history: {hist}\n\n{out}", language="bash")
                s.update(label=("学習 完了 ✅" if rc == 0 else "学習 失敗 ❌"),
                         state=("complete" if rc == 0 else "error"))
            else:
                script = find_train_v4_script()
                if script is None:
                    s.update(label="学習 失敗 ❌", state="error"); st.error("train_evaluate_v4 が見つかりません。")
                else:
                    rc, out = run_py_script(script, args)
                    st.code(f"[INFO] use script: {script}\n[INFO] history: {hist}\n\n{out}", language="bash")
                    s.update(label=("学習 完了 ✅" if rc == 0 else "学習 失敗 ❌"),
                             state=("complete" if rc == 0 else "error"))

if 'do_backfill_hist' in locals() and do_backfill_hist:
    with st.status("予測履歴のバックフィル中...(ローカル用)", expanded=True) as s:
        hist = find_latest_history(DATA_RAW.stat().st_mtime if DATA_RAW.exists() else None)
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
                st.code(out, language="bash")
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
                    st.code(f"[INFO] use script: {script}\n\n{out}", language="bash")
                    if rc == 0: _write_stable_history_from_tmp(PRED_HISTORY_TMP)
                    s.update(label=("バックフィル 完了 ✅" if rc == 0 else "バックフィル 失敗 ❌"),
                             state=("complete" if rc == 0 else "error"))

if 'do_backfill_ev' in locals() and do_backfill_ev:
    with st.status("EVバックフィル中...(ローカル用)", expanded=True) as s:
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
st.title("Numbers3 Dashboard（ビュー専用）")
st.caption("ローカルで生成した予測結果・EVレポートを可視化します。")

# 🔍 デバッグ用：どの history を読んでいるか表示
try:
    hist_path = find_latest_history(DATA_RAW.stat().st_mtime if DATA_RAW.exists() else None)
    if hist_path is not None and hist_path.exists():
        df_dbg = pd.read_csv(hist_path, encoding="utf-8-sig", usecols=lambda c: c == "抽せん日")
        last_date_dbg = pd.to_datetime(df_dbg["抽せん日"], errors="coerce").max()
        st.caption(f"使用 history: {hist_path.name} / 最終 抽せん日: {last_date_dbg.date() if pd.notna(last_date_dbg) else '不明'}")
    else:
        st.caption("使用 history: なし")
except Exception as e:
    st.caption(f"history デバッグ中にエラー: {e}")

d = next_draw_from_history()
draw_str = d.strftime("%Y年%m月%d日") if d else "—"
wday_str = weekday_ja(d) if d else "—"
idx_str  = next_index_from_history()

c1, c2, c3 = st.columns(3)
with c1: components.html(badge_html("抽せん日（ターゲット推定）", draw_str), height=70)
with c2: components.html(badge_html("曜日",  wday_str),  height=70)
with c3: components.html(badge_html("次回 回号（推定）",  idx_str),   height=70)

st.markdown("---")

# 🎯 今日の予測カード（next_prediction.csv ベース）
df_next_for_cards = load_next_prediction()
render_today_cards(df_next_for_cards, topn=3)

st.markdown("---")


# ============ EVレポート読込 & 並び ============
df_ev = read_csv_safe(EV_CSV)
if df_ev is None:
    df_ev = pd.DataFrame()

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
            df_ev["候補_3桁"] = (
                pd.to_numeric(df_ev["候補番号"], errors="coerce")
                  .fillna(0).astype(int).astype(str).str.zfill(3)
            )
        elif "番号" in df_ev.columns:
            df_ev["候補_3桁"] = (
                pd.to_numeric(df_ev["番号"], errors="coerce")
                  .fillna(0).astype(int).astype(str).str.zfill(3)
            )
        else:
            df_ev["候補_3桁"] = ""

    df_ev["候補_3桁"] = df_ev["候補_3桁"].map(fmt3)

    jp = pd.to_numeric(df_ev["joint_prob"], errors="coerce").fillna(0.0).clip(0, 1)
    df_ev["EV_gross"] = jp * float(payout)
    df_ev["EV_net"]   = df_ev["EV_gross"] - float(price)

    sort_cols = [c for c in ["EV_net", "EV_gross", "joint_prob"] if c in df_ev.columns]
    df_ev = df_ev.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    # ★ Top1 の情報だけは履歴に保存（画面には出さない）
    top = df_ev.iloc[0]
    target_date = next_draw_from_history() or date.today()
    num3 = fmt3(top.get("候補_3桁", top.get("番号", "")))
    st.session_state["latest_pick_num3"] = num3
    st.session_state["latest_pick_date"] = target_date
    persist_today_pick(
        pick_date=target_date,
        pick_num3=num3,
        ev_adj=float(top.get("EV_net", 0)),
        prob=float(top.get("joint_prob", 0)),
    )


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
    date_col = next((c for c in ["抽せん日","date","draw_date"] if c in d.columns), None)
    d["date_key"] = pd.to_datetime(d[date_col], errors="coerce").dt.date

    if "候補_3桁" not in d.columns: d["候補_3桁"] = ""
    d["候補_3桁"] = d["候補_3桁"].fillna("").astype(str).apply(fmt3)
    if "候補_3桁_pick" not in d.columns: d["候補_3桁_pick"] = ""
    d["候補_3桁_pick"] = d["候補_3桁_pick"].fillna("").astype(str).apply(fmt3)

    d["_has_pick"] = d["候補_3桁_pick"].ne("") & d["候補_3桁_pick"].ne("nan")
    d["__ev"] = pd.to_numeric(d.get("EV_net"), errors="coerce")
    d["__p"]  = pd.to_numeric(d.get("joint_prob"), errors="coerce").fillna(0.0)

    d["_rank"] = np.where(d["_has_pick"] & (d["候補_3桁"] == d["候補_3桁_pick"]), 0, 1)
    d = d.sort_values(["date_key", "_rank", "__ev", "__p"], ascending=[True, True, False, False])
    d1 = d.drop_duplicates(subset=["date_key"], keep="first").copy()

    need = (d1["候補_3桁"] == "") | (d1["候補_3桁"] == "nan")
    d1.loc[need, "候補_3桁"] = d1.loc[need, "候補_3桁_pick"]
    d1["候補_3桁"] = d1["候補_3桁"].apply(fmt3)

    return d1.drop(columns=["_has_pick","_rank","__ev","__p"], errors="ignore")


def _build_hit_map_for_history() -> pd.DataFrame | None:
    """検証タブと同じロジックで『その日の pick が当たったか』を日付ごとに返す。"""
    df_eval = _load_for_eval()
    if df_eval is None or df_eval.empty:
        return None

    # 日付列
    date_col = None
    for c in ["抽せん日", "date", "draw_date"]:
        if c in df_eval.columns:
            date_col = c
            break
    if date_col is None:
        return None

    df_eval[date_col] = pd.to_datetime(df_eval[date_col], errors="coerce")
    df_eval = df_eval[df_eval[date_col].notna()].copy()
    df_eval["date_key"] = df_eval[date_col].dt.date

    # 候補番号
    if "候補_3桁" not in df_eval.columns:
        df_eval["候補_3桁"] = ""
    df_eval["候補_3桁"] = df_eval["候補_3桁"].fillna("").astype(str)
    if "候補_3桁_pick" not in df_eval.columns:
        df_eval["候補_3桁_pick"] = ""
    else:
        df_eval["候補_3桁_pick"] = df_eval["候補_3桁_pick"].fillna("").astype(str)

    # 当選番号
    if "当選番号3" not in df_eval.columns:
        wdf = winner3_from_raw()
        if wdf is not None:
            df_eval = df_eval.merge(wdf, left_on=date_col, right_on="抽せん日", how="left")
    if "当選番号3" in df_eval.columns:
        df_eval["当選番号3"] = df_eval["当選番号3"].map(fmt3)

    df_eval = ensure_joint_prob(df_eval)

    if "当選番号3" in df_eval.columns:
        df_eval["hit"] = (
            df_eval["候補_3桁"].map(fmt3).ne("") &
            (df_eval["候補_3桁"].map(fmt3) == df_eval["当選番号3"])
        )
    else:
        df_eval["hit"] = False

    df_one = _reduce_to_one_pick_for_eval(df_eval)
    hit_map = df_one[["date_key", "hit"]].copy().rename(columns={"hit": "hit_eval"})
    return hit_map


df_eval = _load_for_eval()
if df_eval.empty:
    st.info("検証用データがありません。ローカルでバックフィルを実行し、CSVを更新してください。")
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
                payout_series = payout_series.where(
                    (payout_series >= 10000) & (payout_series <= 300000),
                    np.nan
                ).fillna(float(payout))
            else:
                st.warning("payouts_map_from_raw の結果が空でした。列名やデータを確認してください。")
                payout_series = pd.Series(float(payout), index=df_win.index)
        else:
            payout_series = pd.Series(float(payout), index=df_win.index)

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
    st.info("予測履歴がまだありません。ローカルで予測＋履歴生成を行い、prediction_history.csv を更新してください。")
else:
    # --- 1) もとの履歴を前処理 ---
    dfh = hist.copy()

    # 日付キー作成
    dfh = _make_date_key(dfh, "抽せん日")
    dfh = _ensure_cand3_cols(dfh)
    dfh = ensure_joint_prob(dfh)

    # 🔥 ここが今回のポイント：1日=1本の pick に正規化
    dfh = _reduce_to_one_pick_for_eval(dfh)

    # date_key がなくなっている可能性があるので再生成
    dfh = _make_date_key(dfh, "抽せん日")

    # 表示用の候補番号：
    #  - 候補_3桁_pick があればそれを優先
    #  - なければ 候補_3桁
    dfh["候補_3桁"] = dfh["候補_3桁"].fillna("").astype(str).apply(fmt3)
    if "候補_3桁_pick" in dfh.columns:
        pick_col = dfh["候補_3桁_pick"].fillna("").astype(str).apply(fmt3)
        use_pick = pick_col.ne("") & pick_col.ne("nan")
        dfh.loc[use_pick, "候補_3桁"] = pick_col[use_pick]

    dfh["候補_3桁_view"] = dfh["候補_3桁"].apply(fmt3).replace("", "—")

    # joint_prob も pick 用があればそちらを優先
    jp = pd.to_numeric(dfh.get("joint_prob"), errors="coerce")
    if "joint_prob_pick" in dfh.columns:
        jp_pick = pd.to_numeric(dfh.get("joint_prob_pick"), errors="coerce")
        jp = jp_pick.where(jp_pick.notna(), jp)
    dfh["joint_prob"] = jp.fillna(0.0).clip(0, 1)

    # 当選番号＆回号を history から付与
    def _load_answer_index_map() -> pd.DataFrame | None:
        p = find_latest_history(DATA_RAW.stat().st_mtime if DATA_RAW.exists() else None)
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
    dfh = dfh[dfh["当選番号3"] != ""].copy()

    JA_WD = ["月曜日","火曜日","水曜日","木曜日","金曜日","土曜日","日曜日"]
    dfh["抽せん日"] = pd.to_datetime(dfh["抽せん日"], errors="coerce")
    dfh["抽せん日_表示"] = dfh["抽せん日"].dt.strftime("%Y年%m月%d日")
    dfh["曜日"] = dfh["抽せん日"].dt.weekday.map(lambda i: JA_WD[i] if pd.notna(i) else "")

    # --- EV の表示値を決定 ---
    dfh["EV_net"] = pd.to_numeric(dfh.get("EV_net"), errors="coerce")
    dfh["EV_net_view"] = dfh["EV_net"]

    need_ev = dfh["EV_net_view"].isna() | (dfh["EV_net_view"] == 0)
    if "EV_net_adj_pick" in dfh.columns:
        adj = pd.to_numeric(dfh.get("EV_net_adj_pick"), errors="coerce")
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
                pays = pays.where((pays >= 10000) & (pays <= 300000), np.nan).fillna(float(payout))
            else:
                pays = pd.Series(float(payout), index=dfh.index)
        else:
            pays = pd.Series(float(payout), index=dfh.index)

        pays = pd.to_numeric(pays, errors="coerce").fillna(float(payout)).clip(10000, 300000)
        jp2 = dfh["joint_prob"].fillna(0.0).clip(0, 1)
        dfh.loc[still, "EV_net_view"] = (jp2 * pays - float(price)).loc[still]

    # --- 検証タブと同じ hit 判定を利用 ---
    hit_map = _build_hit_map_for_history()
    if hit_map is not None and not hit_map.empty:
        dfh = dfh.merge(hit_map, on="date_key", how="left")
        dfh["的中"] = dfh["hit_eval"].fillna(False).astype(bool)
        dfh.drop(columns=["hit_eval"], inplace=True)
    else:
        dfh["的中"] = (dfh["候補_3桁_view"] != "") & (dfh["候補_3桁_view"] == dfh["当選番号3"])

    # 新しい方から並べ替え
    dfh = dfh.sort_values("date_key", ascending=False)

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
    )


# ============ 説明カード（UX向上セクション） ============
st.markdown("---")
render_explanation_cards()


# ============ 詳細テーブル（参考） ============
with st.expander("📊 詳細テーブル（上位200行）", expanded=False):
    if not df_ev.empty:
        st.dataframe(df_ev.head(200), use_container_width=True, hide_index=True)
    else:
        st.write("（なし）")
