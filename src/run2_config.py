# src/run2_config.py
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# Output columns — SOURCE is used as the resumability key
# ---------------------------------------------------------------------------
OUTPUT_COLS = [
    "SOURCE",   # string — category — name/details, pipe-separated
    "TARGET",   # string — category — name/details, pipe-separated
    "WHAT",     # string — "Past Action" | "Future Proposal"
]

# ---------------------------------------------------------------------------
# Row selection mask — only rows with a non-empty CRITICISM_SUMMARY
# ---------------------------------------------------------------------------
def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return a boolean Series selecting rows to send to the LLM."""
    return df[text_col].notna() & (df[text_col].str.strip() != "")
