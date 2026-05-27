# src/run5_prompts.py
from __future__ import annotations
import pandas as pd
from src.run2_prompts import get_composition_idx, get_council_for_date, _format_council_list

_SYSTEM_PROMPT_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a short description of who criticizes "{keyword}" in a Swiss newspaper article.
Your task is to classify the SOURCE of this criticism using ONLY the controlled vocabulary below.

=== SOURCE TYPOLOGY ===

Apply the categories in this strict order of priority:

  Federal Councillor  → a member of the Swiss federal government; append name and department
{council_list}
  Department          → use ONLY when a Federal Councillor is NOT named; append department abbreviation
  Parliamentary       → an elected member of parliament; append name and party abbreviation
                        Valid parties: SVP/UDC | SP/PS | FDP/PLR | Die Mitte/Le Centre | Die Grünen/Les Verts | GLP/PVL
  Civil Servant       → person whose role is to work FOR the state in a public administration
                        (fonctionnaire, public official, non-elected employee of any administration)
                        Even if only their name is given, use this if their state role is mentioned.
  Interest Group      → person or entity that represents an EXTERNAL actor vis-à-vis the state:
                        companies, trade unions, NGOs, lobbies, professional associations,
                        employers' federations, consumer groups, etc.
                        Also use when a named individual is identified by their role in such an organisation.
  Journalist          → use when NO individual name is given, OR when the criticism is formulated
                        directly in the text (e.g. "the article argues…", "the paper criticises…"),
                        OR when the critic is identified only as a journalist.
  General Public      → use ONLY when a name is given with absolutely no title, role, or affiliation.
                        This is the residual category — exhaust all others first.

If multiple distinct sources are mentioned, separate them with " | ".
Append " — [name/details]" after the category when a name or affiliation is given.

=== EXAMPLES ===

Description: "The green party"
SOURCE: Parliamentary

Description: "Members of parliament from the SVP"
SOURCE: Parliamentary

Description: "Environmental NGOs and consumer advocates"
SOURCE: Interest Group

Description: "Albert Rösti, Federal Councillor in charge of the environment"
SOURCE: Federal Councillor — Albert Rösti (DETEC/UVEK)

Description: "SVP member Hans Müller and the Swiss Banking Association"
SOURCE: Parliamentary — Hans Müller (SVP/UDC) | Interest Group — Swiss Banking Association

Description: "The journalist argues that the policy is flawed"
SOURCE: Journalist

=== OUTPUT FORMAT ===
Respond with EXACTLY this line and nothing else:

SOURCE: [Category — Name/Details, or Category alone if no name is given]

Do not add any explanation, preamble, or extra line.\
"""


def build_system_prompt(pubtime) -> str:
    composition = get_council_for_date(pubtime)
    return _SYSTEM_PROMPT_TEMPLATE.format(council_list=_format_council_list(composition))


USER_TEMPLATE = """\
Here is a description of who criticizes "{keyword}" in a Swiss newspaper article:

Description: "{critic_answer}"

Classify the source of this criticism according to the typology above.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    keyword = str(row.get("keyword", "")).strip()
    critic_answer = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, critic_answer=critic_answer)
