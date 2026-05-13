# src/run3_config.py
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# Output columns
# SWISS_CONTEXT is used as the resumability key (inherited from run1 output).
# keyword_stances holds a JSON dict: {"KEYWORD": "CRITICIZED|PRAISED|NEUTRAL", ...}
# ---------------------------------------------------------------------------
OUTPUT_COLS = [
    "keyword_stances",   # JSON string — resumability key
]

# ---------------------------------------------------------------------------
# Row selection mask
# ---------------------------------------------------------------------------
def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return rows that have both article text AND at least one matched keyword."""
    has_text = df[text_col].notna() & (df[text_col].str.strip() != "")
    has_keywords = (
        df["matched_keywords"].notna()
        & (df["matched_keywords"].astype(str).str.strip() != "")
    )
    return has_text & has_keywords
