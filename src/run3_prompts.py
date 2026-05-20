# src/run3_prompts.py
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a media analyst. Your task is to determine whether a specific entity is being criticized in news articles. Criticism can be expressed even if the overall evaluation of the entity is positive. The criticism can be about who the entity is or what it did. Answer ONLY with YES or NO, nothing else.\
"""

USER_TEMPLATE = """\
Is "{keyword}" being criticized in the following article?

ARTICLE:
{article_text}\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    keyword = str(row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
