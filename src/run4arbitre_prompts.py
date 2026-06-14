# src/run4arbitre_prompts.py
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are a neutral arbitrator for media analysis.
Your task is to determine which of two descriptions of criticism more accurately reflects the content of a newspaper article.
You must always choose one — you cannot abstain.\
"""

USER_TEMPLATE = """\
The following newspaper article mentions "{keyword}".

ARTICLE:
{article_text}

Two descriptions have been proposed for who criticises "{keyword}" in this article:

DESCRIPTION A: "{critic_answer}"

DESCRIPTION B: "{run4_eval_answer}"

Which description more accurately captures who criticises "{keyword}" based solely on the article?
Answer ONLY with the letter A or B.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    keyword          = str(row.get("keyword", "")).strip()
    article_text     = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    critic_answer    = row.get("critic_answer", pd.NA)
    critic_answer    = "" if pd.isna(critic_answer) else str(critic_answer).strip()
    run4_eval_answer = row.get("run4_eval_answer", pd.NA)
    run4_eval_answer = "" if pd.isna(run4_eval_answer) else str(run4_eval_answer).strip()
    return USER_TEMPLATE.format(
        keyword=keyword,
        article_text=article_text,
        critic_answer=critic_answer,
        run4_eval_answer=run4_eval_answer,
    )
