# src/run4arbitre_prompts.py
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = """\
You are an editor for media analysis.
You are given a description of criticism that was found to be inaccurate, along with an explanation of what is wrong.
Your task is to rewrite the description to fix the identified inaccuracy.\
"""

USER_TEMPLATE = """\
ORIGINAL DESCRIPTION (found to be inaccurate): "{critic_answer}"

EVALUATOR FEEDBACK (what is wrong with it): "{run4_eval_justification}"

Using the evaluator's feedback, write the corrected one-sentence description of who criticises "{keyword}".
Respond with ONLY the corrected description, nothing else.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    keyword                 = str(row.get("keyword", "")).strip()
    critic_answer           = row.get("critic_answer", pd.NA)
    critic_answer           = "" if pd.isna(critic_answer) else str(critic_answer).strip()
    run4_eval_justification = row.get("run4_eval_justification", pd.NA)
    run4_eval_justification = "" if pd.isna(run4_eval_justification) else str(run4_eval_justification).strip()
    return USER_TEMPLATE.format(
        keyword=keyword,
        critic_answer=critic_answer,
        run4_eval_justification=run4_eval_justification,
    )
