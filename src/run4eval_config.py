# src/run4eval_config.py
from __future__ import annotations
import pandas as pd


def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Process rows that have article text and a non-empty critic_answer (not 'No criticism').
    Skip rows already validated."""
    has_text   = df[text_col].notna() & (df[text_col].str.strip() != "")
    has_answer = df["critic_answer"].notna() & (df["critic_answer"].str.strip() != "")
    no_crit    = df["critic_answer"].str.strip().str.lower().str.startswith("no criticism", na=False)
    already    = (
        df["run4_valid"].notna()
        if "run4_valid" in df.columns
        else pd.Series(False, index=df.index)
    )
    return has_text & has_answer & ~no_crit & ~already
