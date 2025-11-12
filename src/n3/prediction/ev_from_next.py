# app.py — Numbers3 EV Dashboard（カードUI版）
# 目的：
# ・毎回 “最新の履歴CSV” を取り直してから predict_next → EV を実行
# ・古い中身が残らないよう出力CSVを一度削除してから上書き
# ・カードUIで Top3 を表示（最新予測＝Top1 を上段に）
# ・サイドバーから「スクレイプのみ」「予測→EV一括」「予測のみ」「EVのみ」を実行

from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, date
from typing import Tuple

import pandas as pd
import streamlit as st

# =========================
# パス設定
# =========================
ROOT = Path(__file__).resolve().parent
DATA_RAW    = ROOT / "data" / "raw"
OUTPUTS_DIR = ROOT / "artifacts" / "outputs"
MODELS_DIR  = ROOT / "artifacts" / "models_V4_XGB"  # あなたの環境に合わせて
NEXT_CSV    = OUTPUTS_DIR / "next_prediction.csv"
EV_CSV      = OUTPUTS_DIR / "ev_report.csv"

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# =========================
# ユーティリティ
# =========================
def run_py(args: list[str]) -> Tuple[int, str]:
    """仮想環境の python でモジュール/スクリプトを実行"""
    try:
        proc = subprocess.run(
            [sys.executable] + args,
            cwd=str(ROOT),
            text=True,
            capture_output=True
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out
    except Exception as e:
        return 1, f"[runner error] {e}"

def find_latest_raw_csv() -> Path | None:
    """data/raw の *_Numbers3features.csv から最新版を返す（名前/mtimeのハイブリッド）"""
    if not DATA_RAW.exists():
        return None
    files = list(DATA_RAW.glob("*_Numbers3features.csv"))
    if not files:
        return None
    # ファイル名末尾の yyyymmdd を優先しつつ、同点は mtime で
    def _key(p: Path):
        stem = p.stem
        # 例: 20201102-20251017_Numbers3features
        try:
            part = stem.split("_")[0]
            last = part.split("-")[-1]
            score = int(last)
        except Exception:
            score = 0
        return (score, p.stat().st_mtime)
    return sorted(files, key=_key)[-1]

@st.cache_data(show_spinner=False)
def _load_csv_sig(path_str: str, mtime_ns: int, size: int) -> pd.DataFrame | None:
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(p)
        except Exception:
            return None

def read_csv_cached(p: Path) -> pd.DataFrame | None:
    if not p or not p.exists():
        return None
    stt = p.stat()
    return _load_csv_sig(str(p), stt.st_mtime_ns, stt.st_size)

def _remove_if_exists(p: Path):
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass

def z3(n: int | str) -> str:
    try:
        return f"{int(n):03d}"
    except Exception:
        s = str(n)
        only = "".join(ch for ch in s if ch.isdigit())
        return only.zfill(3)[:3] if only else ""

def digit_badge(ch: str) -> str:
    # 枠付きの大きな一文字バッジ（ライト/ダーク両対応）
    return f"""
<div style="
  display:inline-flex;align-items:center;justify-content:center;
  width:56px;height:56px;margin:6px 8px 0 0;
  border:2px solid rgba(255,255,255,0.8);
  border-radius:12px;font-size:28px;font-weight:800;">
  {ch}
</div>"""

def three_digits_box(s3: str) -> str:
    s3 = (s3 or "").strip()
    if len(s3) != 3:
        s3 = z3(s3)
    return (
        "<div style='display:flex;flex-direction:row;'>"
        + digit_badge(s3[0]) + digit_badge(s3[1]) + digit_badge(s3[2])
        + "</div>"
    )

def yen(x: float | int) -> str:
    try:
        return f"{int(round(float(x))):,} 円"
    except Exception:
        return "-"

# =========================
# 予測→EV 実行（常に最新CSVで）
# =========================
def latest_master_csv() -> Path | None:
    return find_latest_raw_csv()

def run_predict_and_ev():
    st.cache_data.clear()  # 古い読み込みキャッシュを捨てる
    latest = latest_master_csv()
    if not latest:
        st.error("最新CSVが見つかりません。先に『データ更新』を実行してください。")
        return

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _remove_if_exists(NEXT_CSV)
    _remove_if_exists(EV_CSV)

    with st.status("predict_next 実行中...", expanded=True) as s1:
        rc1, out1 = run_py([
            "-m", "n3.predict_next",
            "--history", str(latest),
            "--models_dir", str(MODELS_DIR),
            "--out", str(NEXT_CSV),
        ])
        st.code(out1)
        s1.update(
            label=("predict_next 完了 ✅" if rc1 == 0 else "predict_next 失敗 ❌"),
            state=("complete" if rc1 == 0 else "error"),
        )
        if rc1 != 0:
            return

    with st.status("EV 作成中...", expanded=True) as s2:
        rc2, out2 = run_py([
            "-m", "n3.cli",
            "--make-ev",
            "--out", str(EV_CSV),
            "--price", "200", "--payout", "90000",
        ])
        st.code(out2)
        s2.update(
            label=("EV 作成 完了 ✅" if rc2 == 0 else "EV 作成 失敗 ❌"),
            state=("complete" if rc2 == 0 else "error"),
        )

# =========================
# UI
# =========================
st.set_page_config(page_title="Numbers3 EV Dashboard", layout="wide")
st.title("Numbers3 EV Dashboard（カードUI）")
st.caption("EV（期待値）上位のおすすめ候補を Top3 でカード表示。数字は枠内で強調。")

# ---- サイドバー：アクション ----
st.sidebar.header("🛠 アクション")

if st.sidebar.button("データ更新（scrape_update）", use_container_width=True):
    with st.status("scrape_update 実行中...", expanded=True) as s:
        rc, out = run_py(["data/scrape_update.py", "--force", "--months", "2"])
        st.code(out)
        s.update(label=("データ更新 完了 ✅" if rc == 0 else "データ更新 失敗 ❌"),
                 state=("complete" if rc == 0 else "error"))

if st.sidebar.button("最新更新（予測→EV一括）", use_container_width=True):
    run_predict_and_ev()

if st.sidebar.button("最新CSVで 予測だけ実行", use_container_width=True):
    st.cache_data.clear()
    latest = latest_master_csv()
    if not latest:
        st.error("最新CSVが見つかりません。")
    else:
        _remove_if_exists(NEXT_CSV)
        with st.status("predict_next 実行中...", expanded=True) as s:
            rc, out = run_py([
                "-m", "n3.predict_next",
                "--history", str(latest),
                "--models_dir", str(MODELS_DIR),
                "--out", str(NEXT_CSV),
            ])
            st.code(out)
            s.update(label=("予測 完了 ✅" if rc == 0 else "予測 失敗 ❌"),
                     state=("complete" if rc == 0 else "error"))

if st.sidebar.button("EVだけ再作成（make-ev）", use_container_width=True):
    st.cache_data.clear()
    _remove_if_exists(EV_CSV)
    with st.status("EV 作成中...", expanded=True) as s:
        rc, out = run_py([
            "-m", "n3.cli",
            "--make-ev",
            "--out", str(EV_CSV),
            "--price", "200", "--payout", "90000",
        ])
        st.code(out)
        s.update(label=("EV 作成 完了 ✅" if rc == 0 else "EV 作成 失敗 ❌"),
                 state=("complete" if rc == 0 else "error"))

st.markdown("---")

# =========================
# 画面上段：抽選日・曜日・回号 ＆ 最新予測（＝Top1）
# =========================
# 抽選日の決定ロジック：
# 1) next_prediction.csv に '抽せん日' があればそれを使用
# 2) なければ master の最終日 + 0（そのまま表示）/ または EV 側カラムがあれば流用
draw_date_str = "-"
weekday_str   = "-"
round_str     = "-"

df_next = read_csv_cached(NEXT_CSV)
if df_next is not None and not df_next.empty:
    # 最初の行の抽選日があるなら使う
    target_col = None
    for c in ["抽せん日", "対象日", "target_day"]:
        if c in df_next.columns:
            target_col = c
            break
    if target_col:
        try:
            dt = pd.to_datetime(df_next[target_col].iloc[0]).date()
            draw_date_str = dt.isoformat()
            weekday_str = ["月曜日","火曜日","水曜日","木曜日","金曜日","土曜日","日曜日"][dt.weekday()]
        except Exception:
            pass

st.markdown(
    f"""
    <div style="display:flex;gap:64px;align-items:flex-end;">
      <div style="font-size:28px;font-weight:900;">抽選日： {draw_date_str}</div>
      <div style="font-size:28px;font-weight:900;">曜日： {weekday_str}</div>
      <div style="font-size:28px;font-weight:900;">回号： {round_str}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 最新予測は「EV Top1」と同じものを見せたいので、後段の Top3 を先に算出
df_ev = read_csv_cached(EV_CSV)
if df_ev is None or df_ev.empty:
    st.warning("EVレポート（ev_report.csv）が見つかりません。『最新更新（予測→EV一括）』を押してください。")
    df_ev_top3 = pd.DataFrame()
else:
    # 期待値で降順
    sort_cols = [c for c in ["EV_net","EV_gross","joint_prob"] if c in df_ev.columns]
    if sort_cols:
        df_ev = df_ev.sort_values(sort_cols, ascending=[False]*len(sort_cols))
    # 正規化列名
    cand_col = None
    for c in ["候補_3桁","候補番号3","候補","候補_番号","number","num3","pred_3"]:
        if c in df_ev.columns:
            cand_col = c
            break
    if cand_col is None:
        # モデル別の百/十/一 から作るパターン
        if all(c in df_ev.columns for c in ["百","十","一"]):
            df_ev["候補_3桁"] = (df_ev["百"].astype(int)*100 + df_ev["十"].astype(int)*10 + df_ev["一"].astype(int)).astype(int)
            cand_col = "候補_3桁"
        else:
            # 最後の手段：最初の列を候補として扱う
            cand_col = df_ev.columns[0]
    df_ev_top3 = df_ev.head(3).copy()
    df_ev_top3["表示_候補3"] = df_ev_top3[cand_col].map(z3)

# 最新の予測＝EV Top1 と同じものを前段に表示
st.subheader("最新の予測（EV上位＝おすすめTop1 と同じ）")
if df_ev_top3 is None or df_ev_top3.empty:
    st.info("まだ EV がありません。『最新更新（予測→EV一括）』を実行してください。")
else:
    top1 = df_ev_top3.iloc[0]
    st.markdown(three_digits_box(top1["表示_候補3"]), unsafe_allow_html=True)

st.markdown("---")

# =========================
# おすすめ Top3（カード）
# =========================
st.subheader("おすすめ Top3（期待値に基づく候補）")
def card(idx: int, row: pd.Series) -> str:
    n3 = row.get("表示_候補3","")
    evn = row.get("EV_net", None)
    evg = row.get("EV_gross", None)
    pj  = row.get("joint_prob", None)

    p_evn = yen(evn).replace(" 円","")
    p_evg = yen(evg).replace(" 円","")
    p_pj  = f"{float(pj)*100:.2f}%" if pd.notna(pj) else "-"

    # モデル名/FS は小さく右上に
    fs = row.get("feature_set","")
    model = row.get("model_name","")
    meta = f"{fs} / {model}" if fs or model else ""

    return f"""
<div style="border:1px solid rgba(255,255,255,.25); border-radius:14px; padding:14px; margin-bottom:12px;">
  <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
    <div style="font-weight:700; opacity:.85;">おすすめ #{idx}</div>
    <div style="font-size:12px; opacity:.6;">{meta}</div>
  </div>
  {three_digits_box(n3)}
  <div style="margin-top:12px; border:1px dashed rgba(255,255,255,.25); border-radius:12px; padding:10px 12px;">
    <div><b>期待値（手取り）</b>：{p_evn}</div>
    <div><b>想定払戻（当たった場合）</b>：{p_evg}</div>
    <div><b>当選確率（推定）</b>：{p_pj}</div>
  </div>
</div>
"""

if df_ev_top3 is None or df_ev_top3.empty:
    st.info("候補がありません。")
else:
    col1, col2, col3 = st.columns(3)
    for i, (col, (_, row)) in enumerate(zip([col1,col2,col3], df_ev_top3.iterrows()), start=1):
        with col:
            st.markdown(card(i, row), unsafe_allow_html=True)

# 備考（共通の注意書き）
st.markdown(
    """
<div style="margin-top:6px; border:1px solid rgba(255,255,255,.15); border-radius:10px; padding:10px 12px;">
※ 期待値（手取り）は <b>当選確率 × 払戻額 − 購入金額（200円）</b> で試算。払戻額は単勝ち（90,000円）を想定しています。
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================
# 詳細テーブル（上位200）
# =========================
with st.expander("🔍 詳細テーブル（上位200行）", expanded=False):
    if df_ev is None or df_ev.empty:
        st.info("EVレポートがありません。")
    else:
        show_cols = []
        for c in ["表示_候補3","EV_net","EV_gross","joint_prob","feature_set","model_name"]:
            if c in df_ev.columns:
                show_cols.append(c)
        if "表示_候補3" not in df_ev.columns:
            df_ev["表示_候補3"] = df_ev_top3[cand_col].map(z3) if not df_ev_top3.empty else ""
            if "表示_候補3" not in show_cols:
                show_cols = ["表示_候補3"] + [c for c in show_cols if c != "表示_候補3"]
        st.dataframe(df_ev.head(200)[show_cols], use_container_width=True, hide_index=True)
