# src/run4arbitre_config.py
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Process only rows where run4_eval said NO and proposed a correction."""
    has_text         = df[text_col].notna() & (df[text_col].str.strip() != "")
    is_no            = df["run4_valid"].astype(str).str.strip().str.upper() == "NO"
    has_eval_answer  = df["run4_eval_answer"].notna() & (df["run4_eval_answer"].str.strip() != "")
    already          = (
        df["arbiter_choice"].notna()
        if "arbiter_choice" in df.columns
        else pd.Series(False, index=df.index)
    )
    return has_text & is_no & has_eval_answer & ~already
