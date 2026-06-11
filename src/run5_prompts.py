# src/run5_prompts.py
from __future__ import annotations
import pandas as pd
from datetime import date, datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Federal Council compositions
# Each entry: (start: date, end: date, {dept_abbrev: councillor_name})
# ---------------------------------------------------------------------------
_COUNCIL_COMPOSITIONS: list[tuple[date, date, dict[str, str]]] = [
    (
        date(2000, 1, 1), date(2000, 12, 31),
        {
            "DFAE/EDA":   "Joseph Deiss",
            "DFI/EDI":    "Ruth Dreifuss",
            "DFJP/EJPD":  "Ruth Metzler-Arnold",
            "DDPS/VBS":   "Adolf Ogi",
            "DFF/EFD":    "Kaspar Villiger",
            "DFE/EVD":    "Pascal Couchepin",
            "DETEC/UVEK": "Moritz Leuenberger",
        },
    ),
    (
        date(2001, 1, 1), date(2002, 12, 31),
        {
            "DFAE/EDA":   "Joseph Deiss",
            "DFI/EDI":    "Ruth Dreifuss",
            "DFJP/EJPD":  "Ruth Metzler-Arnold",
            "DDPS/VBS":   "Samuel Schmid",
            "DFF/EFD":    "Kaspar Villiger",
            "DFE/EVD":    "Pascal Couchepin",
            "DETEC/UVEK": "Moritz Leuenberger",
        },
    ),
    (
        date(2003, 1, 1), date(2003, 12, 31),
        {
            "DFAE/EDA":   "Micheline Calmy-Rey",
            "DFI/EDI":    "Pascal Couchepin",
            "DFJP/EJPD":  "Ruth Metzler-Arnold",
            "DDPS/VBS":   "Samuel Schmid",
            "DFF/EFD":    "Kaspar Villiger",
            "DFE/EVD":    "Joseph Deiss",
            "DETEC/UVEK": "Moritz Leuenberger",
        },
    ),
    (
        date(2004, 1, 1), date(2006, 7, 31),
        {
            "DFAE/EDA":   "Micheline Calmy-Rey",
            "DFI/EDI":    "Pascal Couchepin",
            "DFJP/EJPD":  "Christoph Blocher",
            "DDPS/VBS":   "Samuel Schmid",
            "DFF/EFD":    "Hans-Rudolf Merz",
            "DFE/EVD":    "Joseph Deiss",
            "DETEC/UVEK": "Moritz Leuenberger",
        },
    ),
    (
        date(2006, 8, 1), date(2007, 12, 31),
        {
            "DFAE/EDA":   "Micheline Calmy-Rey",
            "DFI/EDI":    "Pascal Couchepin",
            "DFJP/EJPD":  "Christoph Blocher",
            "DDPS/VBS":   "Samuel Schmid",
            "DFF/EFD":    "Hans-Rudolf Merz",
            "DFE/EVD":    "Doris Leuthard",
            "DETEC/UVEK": "Moritz Leuenberger",
        },
    ),
    (
        date(2008, 1, 1), date(2008, 12, 31),
        {
            "DFAE/EDA":   "Micheline Calmy-Rey",
            "DFI/EDI":    "Pascal Couchepin",
            "DFJP/EJPD":  "Eveline Widmer-Schlumpf",
            "DDPS/VBS":   "Samuel Schmid",
            "DFF/EFD":    "Hans-Rudolf Merz",
            "DFE/EVD":    "Doris Leuthard",
            "DETEC/UVEK": "Moritz Leuenberger",
        },
    ),
    (
        date(2009, 1, 1), date(2009, 10, 31),
        {
            "DFAE/EDA":   "Micheline Calmy-Rey",
            "DFI/EDI":    "Pascal Couchepin",
            "DFJP/EJPD":  "Eveline Widmer-Schlumpf",
            "DDPS/VBS":   "Ueli Maurer",
            "DFF/EFD":    "Hans-Rudolf Merz",
            "DFE/EVD":    "Doris Leuthard",
            "DETEC/UVEK": "Moritz Leuenberger",
        },
    ),
    (
        date(2009, 11, 1), date(2010, 10, 31),
        {
            "DFAE/EDA":   "Micheline Calmy-Rey",
            "DFI/EDI":    "Didier Burkhalter",
            "DFJP/EJPD":  "Eveline Widmer-Schlumpf",
            "DDPS/VBS":   "Ueli Maurer",
            "DFF/EFD":    "Hans-Rudolf Merz",
            "DFE/EVD":    "Doris Leuthard",
            "DETEC/UVEK": "Moritz Leuenberger",
        },
    ),
    (
        date(2010, 11, 1), date(2011, 12, 31),
        {
            "DFAE/EDA":   "Micheline Calmy-Rey",
            "DFI/EDI":    "Didier Burkhalter",
            "DFJP/EJPD":  "Simonetta Sommaruga",
            "DDPS/VBS":   "Ueli Maurer",
            "DFF/EFD":    "Eveline Widmer-Schlumpf",
            "DEFR/WBF":   "Johann Schneider-Ammann",
            "DETEC/UVEK": "Doris Leuthard",
        },
    ),
    (
        date(2012, 1, 1), date(2015, 12, 31),
        {
            "DFAE/EDA":   "Didier Burkhalter",
            "DFI/EDI":    "Alain Berset",
            "DFJP/EJPD":  "Simonetta Sommaruga",
            "DDPS/VBS":   "Ueli Maurer",
            "DFF/EFD":    "Eveline Widmer-Schlumpf",
            "DEFR/WBF":   "Johann Schneider-Ammann",
            "DETEC/UVEK": "Doris Leuthard",
        },
    ),
    (
        date(2016, 1, 1), date(2017, 10, 31),
        {
            "DFAE/EDA":   "Didier Burkhalter",
            "DFI/EDI":    "Alain Berset",
            "DFJP/EJPD":  "Simonetta Sommaruga",
            "DDPS/VBS":   "Guy Parmelin",
            "DFF/EFD":    "Ueli Maurer",
            "DEFR/WBF":   "Johann Schneider-Ammann",
            "DETEC/UVEK": "Doris Leuthard",
        },
    ),
    (
        date(2017, 11, 1), date(2018, 12, 31),
        {
            "DFAE/EDA":   "Ignazio Cassis",
            "DFI/EDI":    "Alain Berset",
            "DFJP/EJPD":  "Simonetta Sommaruga",
            "DDPS/VBS":   "Guy Parmelin",
            "DFF/EFD":    "Ueli Maurer",
            "DEFR/WBF":   "Johann Schneider-Ammann",
            "DETEC/UVEK": "Doris Leuthard",
        },
    ),
    (
        date(2019, 1, 1), date(2022, 12, 31),
        {
            "DFAE/EDA":   "Ignazio Cassis",
            "DFI/EDI":    "Alain Berset",
            "DFJP/EJPD":  "Karin Keller-Sutter",
            "DDPS/VBS":   "Viola Amherd",
            "DFF/EFD":    "Ueli Maurer",
            "DEFR/WBF":   "Guy Parmelin",
            "DETEC/UVEK": "Simonetta Sommaruga",
        },
    ),
    (
        date(2023, 1, 1), date(2023, 12, 31),
        {
            "DFAE/EDA":   "Ignazio Cassis",
            "DFI/EDI":    "Alain Berset",
            "DFJP/EJPD":  "Elisabeth Baume-Schneider",
            "DDPS/VBS":   "Viola Amherd",
            "DFF/EFD":    "Karin Keller-Sutter",
            "DEFR/WBF":   "Guy Parmelin",
            "DETEC/UVEK": "Albert Rösti",
        },
    ),
    (
        date(2024, 1, 1), date(2025, 3, 31),
        {
            "DFAE/EDA":   "Ignazio Cassis",
            "DFI/EDI":    "Elisabeth Baume-Schneider",
            "DFJP/EJPD":  "Beat Jans",
            "DDPS/VBS":   "Viola Amherd",
            "DFF/EFD":    "Karin Keller-Sutter",
            "DEFR/WBF":   "Guy Parmelin",
            "DETEC/UVEK": "Albert Rösti",
        },
    ),
    (
        date(2025, 4, 1), date(9999, 12, 31),
        {
            "DFAE/EDA":   "Ignazio Cassis",
            "DFI/EDI":    "Elisabeth Baume-Schneider",
            "DFJP/EJPD":  "Beat Jans",
            "DDPS/VBS":   "Martin Pfister",
            "DFF/EFD":    "Karin Keller-Sutter",
            "DEFR/WBF":   "Guy Parmelin",
            "DETEC/UVEK": "Albert Rösti",
        },
    ),
]


def _coerce_date(val) -> Optional[date]:
    """Coerce any pubtime value (str, Timestamp, datetime, date, NaN) to date."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, pd.Timestamp):
        return val.date()
    try:
        return pd.to_datetime(str(val)).date()
    except Exception:
        return None


def get_composition_idx(pubtime) -> int:
    """Return the index into _COUNCIL_COMPOSITIONS for the given pubtime."""
    d = _coerce_date(pubtime)
    if d is None:
        return len(_COUNCIL_COMPOSITIONS) - 1
    for i, (start, end, _) in enumerate(_COUNCIL_COMPOSITIONS):
        if start <= d <= end:
            return i
    if d < _COUNCIL_COMPOSITIONS[0][0]:
        return 0
    return len(_COUNCIL_COMPOSITIONS) - 1


def get_council_for_date(pubtime) -> dict[str, str]:
    return _COUNCIL_COMPOSITIONS[get_composition_idx(pubtime)][2]


def _format_council_list(composition: dict[str, str]) -> str:
    lines = [
        f"                        {dept:<11} — {name}"
        for dept, name in composition.items()
    ]
    return "\n".join(lines)

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
REASON: The green party is a political party with seats in parliament.

Description: "Members of parliament from the SVP"
SOURCE: Parliamentary
REASON: SVP members of parliament are elected legislators.

Description: "Environmental NGOs and consumer advocates"
SOURCE: Interest Group
REASON: NGOs and consumer advocates are external organisations that lobby the state.

Description: "Albert Rösti, Federal Councillor in charge of the environment"
SOURCE: Federal Councillor — Albert Rösti (DETEC/UVEK)
REASON: Albert Rösti is a named member of the federal government heading DETEC/UVEK.

Description: "SVP member Hans Müller and the Swiss Banking Association"
SOURCE: Parliamentary — Hans Müller (SVP/UDC) | Interest Group — Swiss Banking Association
REASON: Hans Müller is a named SVP parliamentarian; the Banking Association is an interest group.

Description: "The journalist argues that the policy is flawed"
SOURCE: Journalist
REASON: The criticism is formulated directly in the article text, attributed to the journalist.

=== OUTPUT FORMAT ===
Respond with EXACTLY these two lines and nothing else:

SOURCE: [Category — Name/Details, or Category alone if no name is given]
REASON: [One sentence explaining why you chose this category]

Do not add any explanation, preamble, or extra line.\
"""


def build_system_prompt(pubtime, keyword: str) -> str:
    composition = get_council_for_date(pubtime)
    return _SYSTEM_PROMPT_TEMPLATE.format(
        keyword=keyword,
        council_list=_format_council_list(composition),
    )


USER_TEMPLATE = """\
Here is a description of who criticizes "{keyword}" in a Swiss newspaper article:

Description: "{critic_answer}"

Classify the source of this criticism according to the typology above.\
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    keyword = str(row.get("keyword", "")).strip()
    critic_answer = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(keyword=keyword, critic_answer=critic_answer)
