# src/run5eval_prompts.py
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a validation assistant for media analysis.
Your task is to verify whether a source classification is logically consistent with a description of criticism.\
"""

USER_TEMPLATE = """\
The following is a description of who criticises "{keyword}" in a Swiss newspaper article:

DESCRIPTION:
"{critic_answer}"

Based on this description, the source of the criticism was classified as:
SOURCE: {source_category}
REASON: {source_reason}

Is this classification and its justification consistent with the description?
Answer ONLY YES or NO.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    keyword         = str(row.get("keyword", "")).strip()
    critic_answer   = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    source_category = row.get("source_category", pd.NA)
    source_category = "" if pd.isna(source_category) else str(source_category).strip()
    source_reason   = row.get("source_reason", pd.NA)
    source_reason   = "" if pd.isna(source_reason) else str(source_reason).strip()
    return USER_TEMPLATE.format(
        keyword=keyword,
        critic_answer=critic_answer,
        source_category=source_category,
        source_reason=source_reason,
    )
