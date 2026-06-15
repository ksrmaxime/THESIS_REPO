# src/run5_config.py
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return rows with a valid final answer to send to run5.

    A row is eligible if:
    - it has non-empty text in text_col (critic_answer_final, or critic_answer for legacy files)
    - it is not a 'No criticism' row
    - it was either validated directly (run4_valid=YES) OR passed through the arbitre (arbiter_choice set)
    """
    has_text = df[text_col].notna() & (df[text_col].str.strip() != "")
    no_crit  = df[text_col].str.strip().str.lower().str.startswith("no criticism", na=False)
    mask = has_text & ~no_crit

    if "run4_valid" in df.columns:
        is_yes        = df["run4_valid"].astype(str).str.strip().str.upper() == "YES"
        is_arbitrated = (
            df["arbiter_answer"].notna()
            if "arbiter_answer" in df.columns
            else pd.Series(False, index=df.index)
        )
        mask = mask & (is_yes | is_arbitrated)

    return mask
