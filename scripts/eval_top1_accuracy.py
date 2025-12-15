import pandas as pd
from pathlib import Path
from datetime import timedelta

# ── プロジェクトルート（.../numbers3_clean） ──
ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "artifacts" / "outputs" / "prediction_history.csv"

# ループ抑制ペナルティのパラメータ
WINDOW_DAYS = 60   # 過去60日の Top1 回数を見る
ALPHA = 0.5        # ペナルティ強度（大きいほど同じ数字を出しづらい）


def compute_baseline_top1(df: pd.DataFrame) -> pd.DataFrame:
    """従来通り joint_prob 最大の候補を Top1 として日別に取得"""
    df = df.copy()
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")

    df_eval = df.dropna(subset=["winning_3桁", "joint_prob"])

    top1 = (
        df_eval.sort_values(["抽せん日", "joint_prob"], ascending=[True, False])
              .groupby("抽せん日")
              .head(1)
              .copy()
    )

    top1["候補_3桁_int"] = top1["候補_3桁"].astype(int)
    top1["winning_3桁_int"] = top1["winning_3桁"].astype(int)
    top1["correct"] = top1["候補_3桁_int"] == top1["winning_3桁_int"]

    return top1


def compute_penalized_top1(df: pd.DataFrame,
                           window_days: int = WINDOW_DAYS,
                           alpha: float = ALPHA) -> pd.DataFrame:
    """
    「過去 window_days 日間に Top1 に選ばれた回数」に応じて
    joint_prob にペナルティをかけたスコアで Top1 を決める。
    """
    df = df.copy()
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df.dropna(subset=["winning_3桁", "joint_prob"])

    # 対象の日付を昇順に
    all_dates = sorted(df["抽せん日"].dropna().unique())

    records = []
    history = []  # 過去の Top1: list of (抽せん日, 候補_3桁_int)

    for current_date in all_dates:
        day_df = df[df["抽せん日"] == current_date].copy()
        if day_df.empty:
            continue

        # winning_3桁（この日の正解）を取得
        winning_val = day_df["winning_3桁"].dropna().astype(int)
        if winning_val.empty:
            # 当選番号が不明ならスキップ
            continue
        winning_val = int(winning_val.iloc[0])

        # 過去 window_days 日以内の Top1 履歴からカウントを作成
        cutoff = current_date - timedelta(days=window_days)
        recent = [(d, n) for (d, n) in history if d >= cutoff]

        recent_counts = {}
        for _, num in recent:
            recent_counts[num] = recent_counts.get(num, 0) + 1

        # 各候補にペナルティ係数を付けてスコアを計算
        day_df["候補_3桁_int"] = day_df["候補_3桁"].astype(int)

        def calc_penalty(num: int) -> float:
            cnt = recent_counts.get(num, 0)
            return 1.0 + alpha * cnt  # 出ている回数に応じて 1+α*count

        day_df["penalty_factor"] = day_df["候補_3桁_int"].apply(calc_penalty)
        day_df["score_penalized"] = day_df["joint_prob"] / day_df["penalty_factor"]

        # ペナルティ後の Top1
        day_df = day_df.sort_values("score_penalized", ascending=False)
        top_row = day_df.iloc[0]

        chosen_num = int(top_row["候補_3桁_int"])
        correct = (chosen_num == winning_val)

        # 履歴にこの日の Top1 を追加
        history.append((current_date, chosen_num))

        records.append(
            {
                "抽せん日": current_date,
                "候補_3桁_int": chosen_num,
                "winning_3桁_int": winning_val,
                "joint_prob": float(top_row["joint_prob"]),
                "penalty_factor": float(top_row["penalty_factor"]),
                "score_penalized": float(top_row["score_penalized"]),
                "correct": bool(correct),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "抽せん日",
                "候補_3桁_int",
                "winning_3桁_int",
                "joint_prob",
                "penalty_factor",
                "score_penalized",
                "correct",
            ]
        )

    penalized_df = pd.DataFrame(records)
    penalized_df.sort_values("抽せん日", inplace=True)
    return penalized_df


def summarize_hit_rate(df: pd.DataFrame, label: str):
    """Top1命中率の集計を表示"""
    if df.empty:
        print(f"[{label}] データがありません。")
        return

    total_days = len(df)
    hit_days = int(df["correct"].sum())
    hit_rate = hit_days / total_days if total_days > 0 else 0.0

    print(f"--- {label} ---")
    print(f"評価対象日数        : {total_days} 日")
    print(f"Top1 的中日数       : {hit_days} 日")
    print(f"Top1 的中率         : {hit_rate * 100:.3f} %")
    print(f"ランダムの何倍か     : {hit_rate / 0.001:.1f} 倍")
    print("------------------------")

def compare_top1(baseline: pd.DataFrame, penalized: pd.DataFrame):
    """
    baseline と penalized の Top1 を日付で突合して差分を確認する
    """
    b = baseline[["抽せん日", "候補_3桁_int", "correct"]].copy()
    p = penalized[["抽せん日", "候補_3桁_int", "correct"]].copy()

    b = b.rename(columns={"候補_3桁_int": "top1_base", "correct": "correct_base"})
    p = p.rename(columns={"候補_3桁_int": "top1_pen", "correct": "correct_pen"})

    m = b.merge(p, on="抽せん日", how="inner")
    m["changed"] = m["top1_base"] != m["top1_pen"]

    changed_days = int(m["changed"].sum())
    total_days = len(m)
    changed_rate = changed_days / total_days if total_days else 0.0

    # 連続同一 Top1（ストリーク）カウント関数
    def count_streaks(series):
        streak = 0
        max_streak = 0
        prev = None
        for x in series:
            if prev is not None and x == prev:
                streak += 1
            else:
                streak = 1
            max_streak = max(max_streak, streak)
            prev = x
        return max_streak

    max_streak_base = count_streaks(m.sort_values("抽せん日")["top1_base"].tolist())
    max_streak_pen  = count_streaks(m.sort_values("抽せん日")["top1_pen"].tolist())

    # 過去60日内のTop1頻度（ざっくり全期間集計）: 同一番号が何回Top1になったか
    base_top_counts = m["top1_base"].value_counts().head(10)
    pen_top_counts  = m["top1_pen"].value_counts().head(10)

    print("===== Top1 差分分析 =====")
    print(f"Top1 が変わった日数     : {changed_days} / {total_days} 日（{changed_rate*100:.2f}%）")
    print(f"最大連続ストリーク（現行）: {max_streak_base} 日")
    print(f"最大連続ストリーク（抑制）: {max_streak_pen} 日")
    print("")
    print("Top1頻出（現行）TOP10:")
    print(base_top_counts.to_string())
    print("")
    print("Top1頻出（抑制）TOP10:")
    print(pen_top_counts.to_string())
    print("")
    print("Top1が変わった日（先頭20件）:")
    print(
        m[m["changed"]]
        .sort_values("抽せん日")
        .head(20)[["抽せん日", "top1_base", "top1_pen", "correct_base", "correct_pen"]]
        .to_string(index=False)
    )
    print("=========================")



def main():
    print(f"[INFO] 評価対象ファイル: {PRED_PATH}")
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"prediction_history.csv が見つかりません: {PRED_PATH}")

    df = pd.read_csv(PRED_PATH)

    if "winning_3桁" not in df.columns:
        raise ValueError("prediction_history に 'winning_3桁' カラムがありません。先に add_winning_column.py を実行してください。")

    if "joint_prob" not in df.columns:
        raise ValueError("prediction_history に 'joint_prob' カラムがありません。")

    # ベースライン（現行ロジック）の Top1
    baseline = compute_baseline_top1(df)

    # ペナルティ付き Top1
    penalized = compute_penalized_top1(df, window_days=WINDOW_DAYS, alpha=ALPHA)

    print("===== Top1 精度評価（ベースライン vs ループ抑制）=====")
    summarize_hit_rate(baseline, "現行: joint_prob 最大")
    summarize_hit_rate(penalized, f"ループ抑制: 過去{WINDOW_DAYS}日・α={ALPHA}")
    print("=================================================")

    main()
    compare_top1(baseline, penalized)

if __name__ == "__main__":
    main()