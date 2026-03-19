# src/target_prompt.py
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------------
# System prompt — empty; the full instruction is in the user turn
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = ""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------
USER_TEMPLATE = """\
You clean entity mentions for later analysis.

Keep only:
- persons -> return only the person's name
- Swiss Federal Council -> return "Federal Council"
- Swiss federal departments -> return the main department name or abbreviation
- Swiss federal offices/agencies -> return the main office/agency name or abbreviation

For anything else, return:
other

Remove titles, articles, adjectives, and surrounding text.
Keep only the core entity.
If uncertain, return:
other

Return only the final cleaned value.
Examples:

Input: Bundesrat
Output: Federal Council

Input: Conseil fédéral
Output: Federal Council

Input: Département fédéral des affaires étrangères (DFAE)
Output: DFAE

Input: DDPS
Output: DDPS

Input: Staatssekretariat für Migration (SEM)
Output: SEM

Input: Das Bundesamt für Rüstung (Armasuisse)
Output: Armasuisse

Input: BAG-Chefin Anne Lévy
Output: Anne Lévy

Input: Ignazio Cassis
Output: Ignazio Cassis

Input: die Stadt Zürich
Output: other

Input: FDP
Output: other

Now process this input:

{entity}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    entity = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(entity=entity)
