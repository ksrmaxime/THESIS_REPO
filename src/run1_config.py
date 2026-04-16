# src/run1_config.py
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# Output columns — order matters: SWISS_CONTEXT is used as the resumability key
# ---------------------------------------------------------------------------
OUTPUT_COLS = [
    "SWISS_CONTEXT",      # string — YES / NO
    "CRITICISM",          # string — YES / NO / N/A
    "TARGETED_ENTITY",    # string — name as in article or N/A
    "SOURCE_NAME",        # string — name + role/title as in article or N/A
    "SOURCE_ORIGIN",      # string — FEDERAL_EXECUTIVE / PARLIAMENT / EXTERNAL / N/A
    "CRITICISM_TOPIC",    # string — 1-2 sentences or N/A
]

# ---------------------------------------------------------------------------
# Row selection mask
# ---------------------------------------------------------------------------
def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return a boolean Series selecting rows to send to the LLM."""
    return df[text_col].notna() & (df[text_col].str.strip() != "")
