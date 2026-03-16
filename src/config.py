# src/config.py
#
# Task-level configuration only.
# Infrastructure (model path, I/O paths, batch params) lives in run.sbatch.
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# Output columns — order matters: CRITIC is used as the resumability key
# ---------------------------------------------------------------------------
OUTPUT_COLS = ["CRITIC", "TARGETED_ENTITY", "SOURCE_ENTITY", "JUSTIFICATION"]

# ---------------------------------------------------------------------------
# Row selection mask — edit to filter which rows are sent to the LLM
# ---------------------------------------------------------------------------
def build_mask(df: pd.DataFrame, *, text_col: str) -> pd.Series:
    """Return a boolean Series selecting rows to send to the LLM."""
    return df[text_col].notna() & (df[text_col].str.strip() != "")
