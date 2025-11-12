# data/scrape_all.py : Scrape full Numbers3 backnumber (~60 months) and save as CSV
# - ASCII-only logs
# - Output file named "{YYYYMMDD}-{YYYYMMDD}_Numbers3features.csv"

import os
import re
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ----------------------
# Settings
# ----------------------
BASE_URL = "https://takarakuji.rakuten.co.jp/backnumber/numbers3/"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = SCRIPT_DIR / "raw"
OUTPUT_DIR = Path(os.getenv("N3_RAW_DIR", str(DEFAULT_RAW_DIR)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ----------------------
# Helpers
# ----------------------
def clean_int(text: str) -> int:
    """Robust numeric cleaner: remove commas/units, normalize minus/percent/currency."""
    if text is None:
        return 0
    s = str(text).strip()
    if s in ("", "-", "—", "―", "該当なし", "なし", "NaN", "nan"):
        return 0
    # remove currency/percent/comma/space
    s = re.sub(r"[,\s¥￥円口]", "", s)
    s = s.replace("％", "").replace("%", "")
    s = s.replace("−", "-")  # U+2212 -> ASCII hyphen
    try:
        return int(float(s))
    except Exception:
        return 0

def classify_pattern(num3: str) -> str:
    s = str(num3).zfill(3)
    uniq = set(s)
    if len(uniq) == 1:
        return "ゾロ目"
    if len(uniq) == 2:
        return "2桁一致"
    if len(uniq) == 3:
        return "全て異なる"
    return "不明"

def pattern_code(a: int, b: int, c: int) -> int:
    if a == b == c:
        return 2  # ゾロ目
    if a == b or b == c or a == c:
        return 1  # ダブル
    return 0      # バラ

def month_range(months: int = 60) -> list[str]:
    """Return list of YYYYMM for past `months` months including this month."""
    outs = []
    today = datetime.today()
    base = today.replace(day=1)
    for i in range(months):
        ym = (base - timedelta(days=30 * i)).strftime("%Y%m")
        outs.append(ym)
    return outs

# ----------------------
# Main
# ----------------------
def main():
    months = month_range(60)
    urls = [f"{BASE_URL}{m}/" for m in months]

    all_rows = []

    for url in urls:
        print(f"[INFO] Accessing: {url}")
        try:
            res = requests.get(url, headers=HEADERS, timeout=20)
            if res.status_code != 200:
                print(f"[WARN] HTTP status not OK: {res.status_code}")
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            tables = soup.find_all("table", class_="tblType02 tblNumberGuid")

            for table in tables:
                rows = table.find_all("tr")
                # 初期値：数値系は 0 で開始（欠損を空文字にしない）
                result = {
                    "回号": "", "抽せん日": "", "当せん番号": "",
                    "ストレート_口数": 0, "ストレート_金額": 0,
                    "ボックス_口数": 0, "ボックス_金額": 0,
                    "セットS_口数": 0, "セットS_金額": 0,
                    "セットB_口数": 0, "セットB_金額": 0,
                    "ミニ_口数": 0, "ミニ_金額": 0,
                    "パターン": ""
                }
                skip = False

                for row in rows:
                    ths = row.find_all("th")
                    tds = row.find_all("td")
                    if not ths:
                        continue

                    title = ths[0].get_text(strip=True)

                    if title == "回号":
                        result["回号"] = ths[1].get_text(strip=True).replace("第", "").replace("回", "")
                    elif title == "抽せん日" and tds:
                        result["抽せん日"] = tds[0].get_text(strip=True)
                    elif title == "当せん番号" and tds:
                        number = tds[0].get_text(strip=True)
                        if number == "該当なし":
                            skip = True
                            break
                        num3 = str(number).zfill(3)
                        result["当せん番号"] = num3
                        result["パターン"] = classify_pattern(num3)

                    elif title.startswith("ストレート") and tds:
                        result["ストレート_口数"] = clean_int(tds[0].get_text())
                        result["ストレート_金額"] = clean_int(tds[1].get_text())
                    elif title.startswith("ボックス") and "セット" not in title and tds:
                        result["ボックス_口数"] = clean_int(tds[0].get_text())
                        result["ボックス_金額"] = clean_int(tds[1].get_text())
                    elif "セット（ストレート）" in title and tds:
                        result["セットS_口数"] = clean_int(tds[0].get_text())
                        result["セットS_金額"] = clean_int(tds[1].get_text())
                    elif "セット（ボックス）" in title and tds:
                        result["セットB_口数"] = clean_int(tds[0].get_text())
                        result["セットB_金額"] = clean_int(tds[1].get_text())
                    elif title == "ミニ" and tds:
                        result["ミニ_口数"] = clean_int(tds[0].get_text())
                        result["ミニ_金額"] = clean_int(tds[1].get_text())

                if not skip and result["回号"] and result["当せん番号"]:
                    all_rows.append(result)

        except Exception as e:
            print(f"[ERROR] Scrape error: {e}")

    if not all_rows:
        print("[INFO] No data scraped. Abort.")
        return

    df = pd.DataFrame(all_rows)

    # Format
    df["当せん番号"] = df["当せん番号"].astype(str).str.zfill(3)
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], format="mixed", errors="coerce")

    # Derived features
    df["百の位"] = df["当せん番号"].str[0].astype(int)
    df["十の位"] = df["当せん番号"].str[1].astype(int)
    df["一の位"] = df["当せん番号"].str[2].astype(int)
    df["合計"] = df["百の位"] + df["十の位"] + df["一の位"]
    df["最大"] = df[["百の位","十の位","一の位"]].max(axis=1)
    df["最小"] = df[["百の位","十の位","一の位"]].min(axis=1)
    df["最大-最小"] = df["最大"] - df["最小"]

    # 曜日（数値は確実、表示用の文字は任意）
    df["曜日_int"] = df["抽せん日"].dt.weekday
    try:
        df["曜日"] = df["抽せん日"].dt.day_name(locale="ja_JP")
    except Exception:
        df["曜日"] = df["抽せん日"].dt.day_name()

    # パターン_code
    df["パターン_code"] = [
        pattern_code(a,b,c) for a,b,c in df[["百の位","十の位","一の位"]].itertuples(index=False)
    ]

    # Save
    df = df.sort_values("抽せん日").drop_duplicates(subset=["抽せん日"], keep="last").reset_index(drop=True)
    min_date = pd.to_datetime(df["抽せん日"]).min().strftime("%Y%m%d")
    max_date = pd.to_datetime(df["抽せん日"]).max().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"{min_date}-{max_date}_Numbers3features.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved full scrape: {out_path}")

if __name__ == "__main__":
    main()
