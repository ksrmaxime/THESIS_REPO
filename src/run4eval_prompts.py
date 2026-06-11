# src/run4eval_prompts.py
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a validation assistant for media analysis.
Your task is to verify whether a short description of criticism accurately reflects the content of a newspaper article.\
"""

USER_TEMPLATE = """\
The following newspaper article mentions "{keyword}".

ARTICLE:
{article_text}

A previous analysis produced this description of the criticism towards "{keyword}":
"{critic_answer}"

Does this description accurately capture who criticises "{keyword}" in this article?
Answer ONLY YES or NO.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    keyword       = str(row.get("keyword", "")).strip()
    article_text  = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    critic_answer = row.get("critic_answer", pd.NA)
    critic_answer = "" if pd.isna(critic_answer) else str(critic_answer).strip()
    return USER_TEMPLATE.format(
        keyword=keyword,
        article_text=article_text,
        critic_answer=critic_answer,
    )
