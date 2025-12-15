import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── プロジェクトルート（.../numbers3_clean） ──
ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "artifacts" / "outputs" / "prediction_history.csv"

RESULTS_CACHE = ROOT / "artifacts" / "inputs" / "numbers3_results_rakuten.csv"
RESULTS_CACHE.parent.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://takarakuji.rakuten.co.jp/backnumber/numbers3/{year:04d}{month:02d}/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}


def scrape_month(year: int, month: int) -> pd.DataFrame:
    """
    楽天×宝くじ ナンバーズ3 月別ページから
    回号・抽せん日・当せん番号(3桁) を取得する。
    """
    url = BASE_URL.format(year=year, month=month)
    print(f"[INFO] Fetching {url} ...")

    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        print(f"[WARN] HTTP {resp.status_code}: {url}")
        return pd.DataFrame(columns=["回号", "抽せん日", "winning_3桁"])

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator="\n")

    # デバッグ用: 見たいときはコメントアウトを外す
    # print(text[:1000])

    # 改善版パターン:
    # 「回号 第XXXX回」から「当せん番号 NNN」までをゆるめに拾う
    pattern = re.compile(
        r"回号\s*第\s*(\d+)\s*回.*?抽せん日\s*([0-9]{4}/[0-9]{2}/[0-9]{2}).*?当せん番号\s*([0-9]{3})",
        re.DOTALL
    )

    matches = pattern.findall(text)
    print(f"[INFO] {year}-{month:02d}: found {len(matches)} entries")

    rows = []
    for round_no, date_str, num_str in matches:
        rows.append(
            {
                "回号": int(round_no),
                "抽せん日": date_str,      # 後で datetime に変換
                "winning_3桁": num_str,    # 3桁（020などを維持）
            }
        )

    if not rows:
        # ここで一部テキストを出しておくと、今後デバッグしやすい
        snippet = "\n".join(text.splitlines()[20:60])
        print("[DEBUG] snippet around records:\n", snippet)

    return pd.DataFrame(rows)


def scrape_range_for_prediction_history(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    prediction_history の抽せん日レンジに対応する年月だけスクレイピング。
    """
    if "抽せん日" not in df_pred.columns:
        raise ValueError("prediction_history に '抽せん日' カラムがありません。")

    df_pred = df_pred.copy()
    df_pred["抽せん日"] = pd.to_datetime(df_pred["抽せん日"], errors="coerce")

    min_date = df_pred["抽せん日"].min()
    max_date = df_pred["抽せん日"].max()

    if pd.isna(min_date) or pd.isna(max_date):
        raise ValueError("prediction_history の '抽せん日' から有効な日付が取得できませんでした。")

    print(f"[INFO] prediction_history 抽せん日レンジ: {min_date.date()} ～ {max_date.date()}")

    months = pd.date_range(
        start=min_date.replace(day=1),
        end=max_date.replace(day=1),
        freq="MS",
    )

    all_results = []
    for dt in months:
        year = dt.year
        month = dt.month
        df_month = scrape_month(year, month)
        if not df_month.empty:
            all_results.append(df_month)

    if not all_results:
        raise RuntimeError("スクレイピング結果が空でした。正規表現 or HTML 構造を再確認してください。")

    df_res = pd.concat(all_results, ignore_index=True)
    df_res["抽せん日"] = pd.to_datetime(df_res["抽せん日"], format="%Y/%m/%d")

    df_res.to_csv(RESULTS_CACHE, index=False, encoding="utf-8-sig")
    print(f"[INFO] スクレイピング結果を保存しました: {RESULTS_CACHE}")

    return df_res


def main():
    print(f"[INFO] 読み込みファイル: {PRED_PATH}")
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"prediction_history.csv が見つかりません: {PRED_PATH}")

    df_pred = pd.read_csv(PRED_PATH)

    # 既にカラムがなければ追加
    if "winning_3桁" not in df_pred.columns:
        df_pred["winning_3桁"] = pd.NA

    # スクレイピングして当せんデータ取得
    df_res = scrape_range_for_prediction_history(df_pred)

    # 日付型そろえ
    df_pred["抽せん日"] = pd.to_datetime(df_pred["抽せん日"], errors="coerce")

    # 日付でマージして winning_3桁 を埋める
    df_merged = df_pred.merge(
        df_res[["抽せん日", "winning_3桁"]],
        on="抽せん日",
        how="left",
        suffixes=("", "_from_web"),
    )

    # 既存の winning_3桁 を優先しつつ、NaN だけ_WEBの値で埋める
    mask = df_merged["winning_3桁"].isna() & df_merged["winning_3桁_from_web"].notna()
    df_merged.loc[mask, "winning_3桁"] = df_merged.loc[mask, "winning_3桁_from_web"]

    df_merged = df_merged.drop(columns=["winning_3桁_from_web"])

    df_merged.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")
    print("✅ prediction_history の winning_3桁 更新が完了しました。")


if __name__ == "__main__":
    main()
