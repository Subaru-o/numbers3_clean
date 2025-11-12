from __future__ import annotations
import pandas as pd
from .config import CFG

def calc_ev(prob: float, payout: int | None = None, bet: int | None = None) -> float:
    P = payout if payout is not None else CFG.payout_straight
    B = bet if bet is not None else CFG.bet_amount
    return prob * P - (1 - prob) * B

def make_ev_report(pred_proba: dict) -> pd.DataFrame:
    # MVP: 各桁の最大確率の積を仮のストレート確率に
    row = {
        "候補": f"{pred_proba['百の位']['pred']}{pred_proba['十の位']['pred']}{pred_proba['一の位']['pred']}",
        "prob_仮定": pred_proba["百の位"]["pmax"]
                     * pred_proba["十の位"]["pmax"]
                     * pred_proba["一の位"]["pmax"],
    }
    row["期待値"] = calc_ev(row["prob_仮定"])
    return pd.DataFrame([row])
