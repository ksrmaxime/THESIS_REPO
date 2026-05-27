# src/run5_config.py
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return rows from run4 output with a valid (non-'No criticism') critic_answer."""
    has_text = df[text_col].notna() & (df[text_col].str.strip() != "")
    no_crit = df[text_col].str.strip().str.lower().str.startswith("no criticism").fillna(False)
    return has_text & ~no_crit
