# -*- coding: utf-8 -*-
# src/n3/diff_features.py
# models_B と models_C の features_*.json を読み、列差分を表示

import json
from pathlib import Path

DIRS = {
    "B": Path("artifacts/models_B"),
    "C": Path("artifacts/models_C"),
}
TARGETS = ["百の位", "十の位", "一の位"]

def load_features(d: Path, tgt: str):
    p = d / f"features_{tgt}.json"
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    return set(j.get("features", []))

def main():
    for tgt in TARGETS:
        print("\n" + "="*50)
        print(f"### {tgt}")
        feat = {}
        for k, d in DIRS.items():
            cols = load_features(d, tgt)
            feat[k] = cols
            print(f"[{k}] 列数={len(cols)}")

        # 差分表示
        onlyB = sorted(list(feat["B"] - feat["C"]))
        onlyC = sorted(list(feat["C"] - feat["B"]))

        print(f"\n- B のみに存在（{len(onlyB)}）:")
        for s in onlyB[:30]:
            print("   ", s)
        if len(onlyB) > 30:
            print("   ... ({} more)".format(len(onlyB)-30))

        print(f"\n- C のみに存在（{len(onlyC)}）:")
        for s in onlyC[:30]:
            print("   ", s)
        if len(onlyC) > 30:
            print("   ... ({} more)".format(len(onlyC)-30))

if __name__ == "__main__":
    main()
