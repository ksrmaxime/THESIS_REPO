# src/run4_config.py
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return rows from run3 output where keyword_answer == YES and have text."""
    has_text = df[text_col].notna() & (df[text_col].str.strip() != "")
    is_yes = df["keyword_answer"].astype(str).str.strip().str.upper() == "YES"
    return has_text & is_yes
