# src/run6_config.py
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return rows validated by run5eval (run5_valid=YES) with a non-'No criticism' critic_answer."""
    has_text = df[text_col].notna() & (df[text_col].str.strip() != "")
    no_crit = df[text_col].str.strip().str.lower().str.startswith("no criticism", na=False)
    mask = has_text & ~no_crit
    if "run5_valid" in df.columns:
        mask = mask & (df["run5_valid"].astype(str).str.strip().str.upper() == "YES")
    return mask
