# src/prompts.py
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a strict media analysis system specialised in Swiss politics.\n"
    "You analyse newspaper articles to detect criticism directed at Swiss public administration.\n"
    "You must respond with exactly FOUR lines, in this order, with no additional text:\n"
    "CRITIC: YES or NO\n"
    "TARGETED_ENTITY: the name of the Swiss public administration unit or entity being criticised, or NONE\n"
    "SOURCE_ENTITY: the actor (person, organisation, party, etc.) who expresses the criticism, or NONE\n"
    "JUSTIFICATION: a short explanation (1-2 sentences) in English of your answer\n"
    "Do not add anything else."
)

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------
USER_TEMPLATE = """Analyse the following newspaper article.
Determine whether it contains a criticism directed at a Swiss public administration entity (federal, cantonal, or local — any unit or body).
If yes, identify who is being criticised (TARGETED_ENTITY) and who is expressing the criticism (SOURCE_ENTITY).

Article:
{text}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(text=txt)
