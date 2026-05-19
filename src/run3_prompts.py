# src/run3_prompts.py
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = ""

USER_TEMPLATE = """\
You will receive an article to analyze. Your task is to tell if "{keyword}" is being criticized in this article, for who it is or what it did? Answer ONLY by YES or NO nothing else

ARTICLE:
{article_text}\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    keyword = str(row.get("keyword", "")).strip()
    article_text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, article_text=article_text)
