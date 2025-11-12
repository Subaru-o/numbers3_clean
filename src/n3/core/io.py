from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_history(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    if "number" in cols:
        s = df[cols["number"]].astype(str).str.zfill(3)
        df["百の位"] = s.str[0].astype(int)
        df["十の位"] = s.str[1].astype(int)
        df["一の位"] = s.str[2].astype(int)
    else:
        rename = {}
        mapping = {"百":"百の位","十":"十の位","一":"一の位","n0":"百の位","n1":"十の位","n2":"一の位"}
        for k,v in mapping.items():
            if k in df.columns:
                rename[k] = v
        if rename:
            df = df.rename(columns=rename)
    if "抽せん日" in df.columns:
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], format="mixed", errors="coerce")
    return df

def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
