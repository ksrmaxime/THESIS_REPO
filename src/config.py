# src/config.py
#
# Task-level configuration only.
# Infrastructure (model path, I/O paths, batch params) lives in run.sbatch.
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# Output columns — order matters: SWISS_CONTEXT is used as the resumability key
# ---------------------------------------------------------------------------
OUTPUT_COLS = [
    "SWISS_CONTEXT",          # bool   — YES / NO
    "CRITICISM",              # string — YES / NO / N/A
    "TARGETED_ENTITY_TYPE",   # string — category or N/A
    "TARGETED_ENTITY_NAME",   # string — name as in text or N/A
    "SOURCE_TYPE",            # string — category or N/A
    "SOURCE_NAME",            # string — name as in text or N/A
    "CRITICISM_TOPIC",        # string — 1-2 sentences or N/A
    "POPULIST_RHETORIC",      # string — YES / NO / N/A
]

# ---------------------------------------------------------------------------
# Row selection mask — edit to filter which rows are sent to the LLM
# ---------------------------------------------------------------------------
def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return a boolean Series selecting rows to send to the LLM."""
    return df[text_col].notna() & (df[text_col].str.strip() != "")
