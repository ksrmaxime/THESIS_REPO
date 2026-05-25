# src/run4_prompts.py
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = ""

USER_TEMPLATE = """\
In the following article, "{keyword}" seems to be the target of criticism. If yes, tell me who expresses this criticism? If not, just say "No criticism".

ARTICLE:
{article_text}\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    keyword = str(row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
