# src/config_prompt.py
#
# Task-level configuration for the target-entity cleaning pipeline.
# Infrastructure (model path, I/O paths, batch params) lives in target.sbatch.
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------------
OUTPUT_COLS = [
    "CLEANED_TARGET",  # string — cleaned entity name or "other"
]

# ---------------------------------------------------------------------------
# Row selection mask — only rows where TARGETED_ENTITY_NAME is non-empty
# ---------------------------------------------------------------------------
def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return a boolean Series selecting rows to send to the LLM."""
    return df[text_col].notna() & (df[text_col].str.strip() != "")
