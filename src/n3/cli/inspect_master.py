# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from .features import detect_schema, PRIZE_CANDIDATES

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="マスターCSVのパス")
    ap.add_argument("--head", type=int, default=5, help="先頭に表示する行数")
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.history, encoding="utf-8-sig")
    print("=== 列一覧 ===")
    for c in df.columns:
        print(" -", c)
    print()

    try:
        date_col, num_key, prize_map = detect_schema(df)
        print("=== 自動判定結果 ===")
        print("日付列:", date_col)
        print("数字列:", num_key, "(桁別なら '桁別')")
        if prize_map:
            print("金額列対応（見つかったもののみ）:")
            for std, actual in prize_map.items():
                print(f" - {std} -> {actual}")
        else:
            print("金額列は見つかりませんでした（任意）")
    except Exception as e:
        print("スキーマ自動判定エラー:", e)

    print("\n=== 先頭プレビュー ===")
    print(df.head(args.head))

if __name__ == "__main__":
    main()
