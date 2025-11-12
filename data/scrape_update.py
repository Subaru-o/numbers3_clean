# data/scrape_update.py : Update/append latest Numbers3 results into data/raw
# - ASCII-only logs
# - Detect latest "*_Numbers3features.csv" under data/raw and append
# - Save merged file as "{YYYYMMDD}-{YYYYMMDD}_Numbers3features.csv"

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

# Output directory (raw CSV storage)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = SCRIPT_DIR / "raw"
OUTPUT_DIR = Path(os.getenv("N3_RAW_DIR", str(DEFAULT_RAW_DIR)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ----------------------
# Helpers
# ----------------------
def latest_master_csv(raw_dir: Path) -> Path | None:
    """Return latest '*_Numbers3features.csv' under raw_dir (by name), else None."""
    cands = sorted(raw_dir.glob("*_Numbers3features.csv"))
    return cands[-1] if cands else None

def clean_int(text: str) -> int:
    """Robust numeric cleaner: remove commas/units, normalize minus/percent/currency."""
    if text is None:
        return 0
    s = str(text).strip()
    if s in ("", "-", "—", "―", "該当なし", "なし", "NaN", "nan"):
        return 0
    s = re.sub(r"[,\s¥￥円口]", "", s)
    s = s.replace("％", "").replace("%", "")
    s = s.replace("−", "-")
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
        return 2
    if a == b or b == c or a == c:
        return 1
    return 0

def yyyymm_targets(today: datetime, months: int = 2) -> list[str]:
    """Return [YYYYMM, ...] for 'months' months starting from this month."""
    outs = []
    base = today.replace(day=1)
    for i in range(months):
        ym = (base - timedelta(days=30 * i)).strftime("%Y%m")
        outs.append(ym)
    return outs

# ----------------------
# Main
# ----------------------
def main():
    # 0) Load existing latest file (if any) to determine latest_date
    latest_csv = latest_master_csv(OUTPUT_DIR)
    if latest_csv and latest_csv.exists():
        print(f"[INFO] Found existing CSV: {latest_csv}")
        existing_df = pd.read_csv(latest_csv, encoding="utf-8-sig")
        if "抽せん日" in existing_df.columns:
            existing_df["抽せん日"] = pd.to_datetime(existing_df["抽せん日"], format="mixed", errors="coerce")
            latest_date = existing_df["抽せん日"].max()
        else:
            latest_date = datetime(2000, 1, 1)
    else:
        print("[INFO] No existing CSV detected. Starting fresh.")
        existing_df = pd.DataFrame()
        latest_date = datetime(2000, 1, 1)

    # 1) Target months (this month + last month)
    today = datetime.today()
    months = yyyymm_targets(today, months=2)
    urls = [f"{BASE_URL}{m}/" for m in months]

    new_rows = []

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
                # 初期値：数値は0、文字は空文字
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
                    new_rows.append(result)

        except Exception as e:
            print(f"[ERROR] Scrape error: {e}")

    # 2) Build df_new and filter rows newer than latest_date
    if new_rows:
        df_new = pd.DataFrame(new_rows)
    else:
        df_new = pd.DataFrame(columns=[
            "回号","抽せん日","当せん番号",
            "ストレート_口数","ストレート_金額",
            "ボックス_口数","ボックス_金額",
            "セットS_口数","セットS_金額",
            "セットB_口数","セットB_金額",
            "ミニ_口数","ミニ_金額",
            "パターン"
        ])

    if not df_new.empty:
        df_new["当せん番号"] = df_new["当せん番号"].astype(str).str.zfill(3)
        df_new["抽せん日"] = pd.to_datetime(df_new["抽せん日"], format="mixed", errors="coerce")
        df_new = df_new[df_new["抽せん日"] > latest_date]

    if df_new.empty:
        print("[INFO] No new data. Existing CSV is up to date.")
        if not existing_df.empty and "抽せん日" in existing_df.columns:
            existing_df = existing_df.sort_values("抽せん日").reset_index(drop=True)
            min_date = pd.to_datetime(existing_df["抽せん日"]).min().strftime("%Y%m%d")
            max_date = pd.to_datetime(existing_df["抽せん日"]).max().strftime("%Y%m%d")
            out_path = OUTPUT_DIR / f"{min_date}-{max_date}_Numbers3features.csv"
            existing_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[INFO] Re-saved existing data as: {out_path}")
        return

    # 3) Derive extra columns
    df_new["百の位"] = df_new["当せん番号"].str[0].astype(int)
    df_new["十の位"] = df_new["当せん番号"].str[1].astype(int)
    df_new["一の位"] = df_new["当せん番号"].str[2].astype(int)
    df_new["合計"] = df_new["百の位"] + df_new["十の位"] + df_new["一の位"]
    df_new["最大"] = df_new[["百の位","十の位","一の位"]].max(axis=1)
    df_new["最小"] = df_new[["百の位","十の位","一の位"]].min(axis=1)
    df_new["最大-最小"] = df_new["最大"] - df_new["最小"]

    # 曜日（数値＋表示用）
    df_new["曜日_int"] = df_new["抽せん日"].dt.weekday
    try:
        df_new["曜日"] = df_new["抽せん日"].dt.day_name(locale="ja_JP")
    except Exception:
        df_new["曜日"] = df_new["抽せん日"].dt.day_name()

    # パターン_code
    df_new["パターン_code"] = [
        pattern_code(a,b,c) for a,b,c in df_new[["百の位","十の位","一の位"]].itertuples(index=False)
    ]

    # 4) Merge with existing and save as date-ranged filename
    merged_df = pd.concat([existing_df, df_new], ignore_index=True)
    if "抽せん日" in merged_df.columns:
        merged_df["抽せん日"] = pd.to_datetime(merged_df["抽せん日"], format="mixed", errors="coerce")
        merged_df = merged_df.sort_values("抽せん日")
        merged_df = merged_df.drop_duplicates(subset=["抽せん日"], keep="last").reset_index(drop=True)

    min_date = pd.to_datetime(merged_df["抽せん日"]).min().strftime("%Y%m%d")
    max_date = pd.to_datetime(merged_df["抽せん日"]).max().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"{min_date}-{max_date}_Numbers3features.csv"
    merged_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Updated: {out_path}")

if __name__ == "__main__":
    main()
