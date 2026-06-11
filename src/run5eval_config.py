# src/run5eval_config.py
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Process rows that have critic_answer (text_col) and both source_category + source_reason.
    Skip rows already validated."""
    has_text     = df[text_col].notna() & (df[text_col].str.strip() != "")
    has_category = df["source_category"].notna() & (df["source_category"].str.strip() != "")
    has_reason   = df["source_reason"].notna()   & (df["source_reason"].str.strip()   != "")
    already      = (
        df["run5_valid"].notna()
        if "run5_valid" in df.columns
        else pd.Series(False, index=df.index)
    )
    return has_text & has_category & has_reason & ~already
