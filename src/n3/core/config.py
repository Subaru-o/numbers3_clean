from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

@dataclass
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    models_dir: Path = root / "artifacts" / "models"
    outputs_dir: Path = root / "artifacts" / "outputs"

@dataclass
class AppCfg:
    payout_straight: int = int(os.getenv("N3_PAYOUT_STRAIGHT", "100000"))
    bet_amount: int = int(os.getenv("N3_BET_AMOUNT", "200"))

PATHS = Paths()
CFG = AppCfg()

for p in [PATHS.models_dir, PATHS.outputs_dir]:
    p.mkdir(parents=True, exist_ok=True)
