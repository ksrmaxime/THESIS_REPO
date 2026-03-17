# src/prompts.py
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a strict media analysis system specialised in Swiss politics.\n"
    "You analyse newspaper articles to detect criticism directed at Swiss public administration.\n"
    "CRITIC must be YES only if ALL of the following conditions are met:\n"
    "  1. The article contains explicit criticism (not just neutral description).\n"
    "  2. The criticism is directed at a Swiss public administration entity (federal, cantonal, or local — e.g. a ministry, office, agency, municipality, canton, or public institution).\n"
    "  3. The target is NOT a foreign government or administration, NOT a private company, and NOT a non-governmental or political actor.\n"
    "If the criticism targets a foreign administration, a private company, a political party, an NGO, or any non-Swiss public body, answer CRITIC: NO.\n"
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
Determine whether it contains a criticism directed at a Swiss public administration entity (federal, cantonal, or local — e.g. a ministry, office, agency, municipality, canton, or public institution).

Important: answer CRITIC: NO if the criticism targets a foreign government, a private company, a political party, an NGO, or any actor that is not part of the Swiss public administration.

If yes, identify who is being criticised (TARGETED_ENTITY) and who is expressing the criticism (SOURCE_ENTITY).

Article:
{text}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(text=txt)
