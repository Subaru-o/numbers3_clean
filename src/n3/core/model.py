# -*- coding: utf-8 -*-
# src/n3/model.py

from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier

def build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
