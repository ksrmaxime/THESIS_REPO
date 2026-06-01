# src/run6_prompts.py
from __future__ import annotations
import pandas as pd

_SYSTEM_PROMPT_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a brief description of a criticism directed at "{keyword}" in a Swiss newspaper article.
Your task is to classify the NATURE of this criticism: is it directed at the person/entity ITSELF,
or at their political actions and proposals?

=== CRITICISM TYPOLOGY ===

  Person  → the criticism targets the person/entity ITSELF:
             their character, personality, morality, ethics, integrity, reputation,
             behaviour, management or leadership style, competence as an individual,
             specific statements or words they used, personal history, personal conduct.
             Use this when the attack is on who they ARE, not on a specific thing they did.

  Policy  → the criticism targets their ACTIONS, DECISIONS or PROPOSALS:
             a bill, a reform, a regulation, a budget, a strategy, a dossier or a project
             they are responsible for; a specific political decision, position or file.
             Use this when the attack is on a concrete thing they DID or PROPOSED.

  Both    → the description contains clear elements of BOTH personal and policy criticism
             (e.g. questions both the minister's integrity AND a specific reform they proposed).

  Unclear → the description does not provide enough information to determine
             whether the criticism is personal or policy-oriented.

=== EXAMPLES ===

Description: "Opposition parties question the minister's moral integrity and honesty"
TARGET: Person

Description: "The SVP criticises the proposed immigration reform"
TARGET: Policy

Description: "NGOs attack both his leadership style and the budget cuts he proposed"
TARGET: Both

Description: "Several critics expressed concerns"
TARGET: Unclear

=== OUTPUT FORMAT ===
Respond with EXACTLY this line and nothing else:

TARGET: [Person | Policy | Both | Unclear]

Do not add any explanation, preamble, or extra line.\
"""


def build_system_prompt(keyword: str) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(keyword=keyword)


USER_TEMPLATE = """\
Here is a description of the criticism directed at "{keyword}" in a Swiss newspaper article:

Description: "{critic_answer}"

Classify whether this criticism targets the person/entity itself or their political actions/proposals.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    keyword = str(row.get("keyword", "")).strip()
    critic_answer = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, critic_answer=critic_answer)
