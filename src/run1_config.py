# src/run1_config.py
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# Output columns
# SWISS_CONTEXT is used as the resumability key (skip rows already filled).
# CRITICISMS stores a JSON list of {target, source, topic} dicts, or pd.NA.
# ---------------------------------------------------------------------------
OUTPUT_COLS = [
    "SWISS_CONTEXT",   # string — YES / NO
    "CRITICISMS",      # string — JSON list of {target, source, topic} dicts, or N/A
]

# ---------------------------------------------------------------------------
# Row selection mask
# ---------------------------------------------------------------------------
def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return a boolean Series selecting rows to send to the LLM."""
    return df[text_col].notna() & (df[text_col].str.strip() != "")
