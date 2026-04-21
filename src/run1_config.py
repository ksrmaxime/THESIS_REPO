# src/run1_config.py
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# Output columns — order matters: SWISS_CONTEXT is used as the resumability key
# ---------------------------------------------------------------------------
OUTPUT_COLS = [
    "SWISS_CONTEXT",       # string — YES / NO
    "CRITICISM",           # string — YES / NO
    "CRITICISM_SUMMARY",   # string — one paragraph or N/A
]

# ---------------------------------------------------------------------------
# Row selection mask
# ---------------------------------------------------------------------------
def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return a boolean Series selecting rows to send to the LLM."""
    return df[text_col].notna() & (df[text_col].str.strip() != "")
