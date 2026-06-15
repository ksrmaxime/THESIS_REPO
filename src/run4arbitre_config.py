# src/run4arbitre_config.py
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Process only NO rows that have an evaluator justification to synthesize from."""
    has_critic    = df["critic_answer"].notna() & (df["critic_answer"].str.strip() != "")
    is_no         = df["run4_valid"].astype(str).str.strip().str.upper() == "NO"
    has_eval_just = df["run4_eval_justification"].notna() & (df["run4_eval_justification"].str.strip() != "")
    already       = (
        df["arbiter_answer"].notna()
        if "arbiter_answer" in df.columns
        else pd.Series(False, index=df.index)
    )
    return has_critic & is_no & has_eval_just & ~already
