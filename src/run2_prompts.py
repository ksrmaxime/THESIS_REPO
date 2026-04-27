# src/run2_prompts.py
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
    """Return the index into _COUNCIL_COMPOSITIONS for the given pubtime.

    Accepts any pubtime value (date, datetime, pd.Timestamp, str, NaN).
    Falls back to the last composition for missing or out-of-range values.
    """
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


# ---------------------------------------------------------------------------
# System prompt template — {council_list} is filled per article date
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_TEMPLATE = """\
You are a media analysis assistant specialised in Swiss public affairs.
You are given a paragraph that summarises criticism found in a newspaper article.
Your task is to extract SOURCE, TARGET, WHAT using ONLY the controlled vocabularies below.

=== CONTROLLED VOCABULARIES ===

──────────────────────────────────────────────────────────────────────────────
SOURCE — who is doing the criticising?
WARNING: the SOURCE is NOT necessarily the main character of the paragraph.
If "X proposes something and Y criticises it", SOURCE = Y, not X.
Pick the SINGLE best-matching category per source entity.
Append " — [name/details]" after the category when a name or affiliation is given.
Separate multiple distinct sources with " | ".

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
  Journalist          → use when NO individual name is given, OR when the text itself formulates
                        the criticism directly (e.g. "the article argues…", "the paper criticises…",
                        "the journalist writes…"), OR when the critic is identified only as a journalist.
  General Public      → use ONLY when a name is given with absolutely no title, role, or affiliation.
                        This is the residual category — exhaust all others first.

──────────────────────────────────────────────────────────────────────────────
TARGET — who is being blamed?
Pick the SINGLE best-matching category per target entity.
Append " — [name/details]" after the category when relevant.
Separate multiple distinct targets with " | ".

  Federal Councillor  → (same name+department list as above)
  Federal Department  → append department abbreviation from list above
  Federal Council     → the Swiss federal government as a collective body
  Civil Servant       → a named fonctionnaire / public-administration employee (non-elected)
  System              → the system in itself is targeted
  City Entity         → a municipal government or city-level body; append city name
  Canton Entity       → a cantonal government or cantonal body; append canton name
  State-Owned Company → a state-owned enterprise (e.g. SBB, La Poste, RUAG); append company name

──────────────────────────────────────────────────────────────────────────────
WHAT — nature of the criticism (choose EXACTLY ONE)

  Past Action     → criticism targets something already done, said, or decided,
                    or the negative consequences of a past act or omission
  Future Proposal → criticism targets a proposal, bill, or plan that is currently
                    being pushed, discussed, or has not yet been enacted

──────────────────────────────────────────────────────────────────────────────

=== HOW TO EXTRACT ===
1. Locate the sentence(s) expressing a negative judgment.
2. Who is making that judgment? → SOURCE (pick category + add name/details)
3. Who is being judged negatively? → TARGET (pick category + add name/details)
4. Is the criticism about something already done, or something proposed? → WHAT

=== EXAMPLES (councillor names shown for format illustration only) ===

Summary: "Albert Rösti, Swiss Federal Councillor, proposes making economy class the standard for all flights. The Department of Economy, led by Guy Parmelin, has criticised the initiative, arguing that business-class travel is necessary for professional missions."
SOURCE: Federal Councillor — Guy Parmelin (DEFR/WBF)
TARGET: Federal Councillor — Albert Rösti (DETEC/UVEK)
WHAT: Future Proposal

Summary: "The EDA has kept a study on the recognition of Palestine as a state secret since June. Critics argue that this secrecy undermines transparency and public trust, especially given the ongoing conflict in the Middle East."
SOURCE: Journalist
TARGET: Federal Department — DFAE/EDA
WHAT: Past Action

Summary: "The article criticises the Swiss justice system for failing to hold Credit Suisse accountable for massive losses and bonuses. The journalist argues the system is biased towards protecting powerful institutions."
SOURCE: Journalist
TARGET: System
WHAT: Past Action

Summary: "SVP member Hans Müller and the Swiss Banking Association criticise Beat Jans's plan to tighten controls on crypto-asset transfers, arguing it would drive business abroad."
SOURCE: Parliamentary — Hans Müller (SVP/UDC) | Interest Group — Swiss Banking Association
TARGET: Federal Councillor — Beat Jans (DFJP/EJPD)
WHAT: Future Proposal

=== OUTPUT FORMAT ===
Respond with EXACTLY these 3 lines and nothing else:

SOURCE: [Category — Name/Details, or Category alone if no name is given]
TARGET: [Category — Name/Details, or Category alone if no name is given]
WHAT: [Past Action | Future Proposal]

Do not add any explanation, preamble, or extra line.\
"""


def build_system_prompt(pubtime) -> str:
    """Return the system prompt with the Federal Council composition for pubtime.

    Accepts any pubtime value (date, datetime, pd.Timestamp, str, NaN).
    """
    composition = get_council_for_date(pubtime)
    return _SYSTEM_PROMPT_TEMPLATE.format(council_list=_format_council_list(composition))


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------
USER_TEMPLATE = """\
Extract the SOURCE, TARGET, and WHAT from the following criticism summary.

Summary:
{text}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(text=txt)
