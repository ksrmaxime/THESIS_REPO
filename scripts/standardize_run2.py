"""
Standardise SOURCE and TARGET columns from a RUN2 output file.

Usage:
    python scripts/standardize_run2.py --input  data/output/run2_merged_jobXXX.parquet \
                                        --output data/output/run2_standardized.parquet

The script keeps only the analysis-relevant columns and appends SOURCE_STD / TARGET_STD.
Input can be .parquet or .csv ; output format matches the extension of --output.
"""

from __future__ import annotations
import sys
import re
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from src.run2_prompts import _COUNCIL_COMPOSITIONS, get_council_for_date

# ---------------------------------------------------------------------------
# Build reference data from run2_prompts
# ---------------------------------------------------------------------------

# All unique councillor names, longest first to avoid partial-name shadowing
ALL_COUNCILLORS: list[str] = sorted(
    {name for _, _, comp in _COUNCIL_COMPOSITIONS for name in comp.values()},
    key=lambda n: -len(n),
)


def _name_to_dept(pubtime) -> dict[str, str]:
    """Return {councillor_name: dept_abbrev} for the council active at pubtime."""
    composition = get_council_for_date(pubtime)  # {dept: name}
    return {name: dept for dept, name in composition.items()}


# Fallback: for each councillor, department from their most recent composition.
# Used when pubtime pre-dates their actual tenure (LLM anachronism).
_LAST_KNOWN_DEPT: dict[str, str] = {
    name: dept
    for _, _, comp in _COUNCIL_COMPOSITIONS
    for dept, name in comp.items()
}

# ---------------------------------------------------------------------------
# Controlled vocabulary constants
# ---------------------------------------------------------------------------

PARTIES: list[str] = [
    "Die Grünen/Les Verts",   # longest first so substring search works correctly
    "Die Mitte/Le Centre",
    "SVP/UDC",
    "FDP/PLR",
    "GLP/PVL",
    "SP/PS",
]

# Alias patterns for party names the LLM writes differently.
# Each entry: (compiled regex, canonical party string).
# Checked with re.IGNORECASE after the exact PARTIES list fails.
# Also fires when the LLM drops the "Parliamentary" keyword for later pipe entities.
PARTY_ALIAS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bSVP\b'),                   "SVP/UDC"),
    (re.compile(r'\bUDC\b'),                   "SVP/UDC"),
    (re.compile(r'\b(?:SP|PS)\b'),             "SP/PS"),
    (re.compile(r'\bFDP\b'),                   "FDP/PLR"),
    (re.compile(r'\bPLR\b'),                   "FDP/PLR"),
    (re.compile(r'\bGLP\b'),                   "GLP/PVL"),
    (re.compile(r'\bPVL\b'),                   "GLP/PVL"),
    (re.compile(r'\bGreen Liberals?\b', re.I), "GLP/PVL"),
    (re.compile(r'\bVerts libéraux\b',  re.I), "GLP/PVL"),
    (re.compile(r'\bMitte\b',           re.I), "Die Mitte/Le Centre"),
    (re.compile(r'\bLe Centre\b',       re.I), "Die Mitte/Le Centre"),
    (re.compile(r'\bCVP\b'),                   "Die Mitte/Le Centre"),
    (re.compile(r'\bPDC\b'),                   "Die Mitte/Le Centre"),
    (re.compile(r'\bGrünen?\b',         re.I), "Die Grünen/Les Verts"),
    (re.compile(r'\bLes Verts\b',       re.I), "Die Grünen/Les Verts"),
    (re.compile(r'\bGreen Party\b',     re.I), "Die Grünen/Les Verts"),
    (re.compile(r'\bGreens\b',          re.I), "Die Grünen/Les Verts"),
]

# Single-language department abbreviation → canonical bilingual form.
# Covers LLM outputs that omit one side (e.g. "EDI" instead of "DFI/EDI").
DEPT_SINGLE_MAP: dict[str, str] = {
    # French halves
    "DFAE":  "DFAE/EDA",
    "DFI":   "DFI/EDI",
    "DFJP":  "DFJP/EJPD",
    "DDPS":  "DDPS/VBS",
    "DFF":   "DFF/EFD",
    "DEFR":  "DEFR/WBF",
    "DETEC": "DETEC/UVEK",
    "ChF":   "ChF/BK",
    # German halves
    "EDA":   "DFAE/EDA",
    "EDI":   "DFI/EDI",
    "EJPD":  "DFJP/EJPD",
    "VBS":   "DDPS/VBS",
    "EFD":   "DFF/EFD",
    "WBF":   "DEFR/WBF",
    "UVEK":  "DETEC/UVEK",
    "BK":    "ChF/BK",
}

# Sub-office / secretariat abbreviation → parent department.
# Both FR and DE abbreviations map to the same bilingual dept form.
SUBOFFICE_MAP: dict[str, str] = {
    # ── DFI/EDI ──────────────────────────────────────────────────────────────
    "OFSP": "DFI/EDI",  "BAG":  "DFI/EDI",   # Santé publique / Gesundheit
    "OFAS": "DFI/EDI",  "BSV":  "DFI/EDI",   # Assurances sociales
    "OFS":  "DFI/EDI",  "BFS":  "DFI/EDI",   # Statistique
    "OFC":  "DFI/EDI",  "BAK":  "DFI/EDI",   # Culture
    "OSAV": "DFI/EDI",  "BLV":  "DFI/EDI",   # Sécurité alimentaire / vétérinaire
    "BFEG": "DFI/EDI",                         # Égalité femmes-hommes
    "AFS":  "DFI/EDI",  "BAR":  "DFI/EDI",   # Archives fédérales
    "MétéoSuisse": "DFI/EDI",
    # ── DFAE/EDA ─────────────────────────────────────────────────────────────
    "DSH":  "DFAE/EDA",                        # Droit international public
    "DAS":  "DFAE/EDA",                        # Aide humanitaire
    # ── DFJP/EJPD ────────────────────────────────────────────────────────────
    "SEM":    "DFJP/EJPD",                     # Secrétariat d'État aux migrations
    "fedpol": "DFJP/EJPD",                     # Police fédérale
    "OFJ":    "DFJP/EJPD", "BJ": "DFJP/EJPD", # Justice
    "ISDC":   "DFJP/EJPD",                     # Institut droit comparé
    # ── DDPS/VBS ─────────────────────────────────────────────────────────────
    "armasuisse": "DDPS/VBS",
    "swisstopo":  "DDPS/VBS",
    "OFPP": "DDPS/VBS", "BABS": "DDPS/VBS",   # Protection de la population
    # ── DFF/EFD ──────────────────────────────────────────────────────────────
    "AFC":   "DFF/EFD",  "ESTV":  "DFF/EFD",  # Contributions / impôts
    "AFD":   "DFF/EFD",  "BAZG":  "DFF/EFD",  # Douanes et sécurité des frontières
    "CDF":   "DFF/EFD",  "EFK":   "DFF/EFD",  # Contrôle des finances
    "AFF":   "DFF/EFD",  "FFA":   "DFF/EFD",  # Administration des finances
    "FOITT": "DFF/EFD",  "BIT":   "DFF/EFD",  # Informatique / télécommunications
    "FOBL":  "DFF/EFD",  "BBL":   "DFF/EFD",  # Construction et logistique
    # ── DEFR/WBF ─────────────────────────────────────────────────────────────
    "SECO":  "DEFR/WBF",                       # Secrétariat d'État à l'économie
    "SBFI":  "DEFR/WBF", "SEFRI": "DEFR/WBF", # Formation, recherche, innovation
    "SERI":  "DEFR/WBF",
    "OFAG":  "DEFR/WBF", "BLW":   "DEFR/WBF", # Agriculture
    # ── DETEC/UVEK ───────────────────────────────────────────────────────────
    "OFEV":  "DETEC/UVEK", "BAFU":  "DETEC/UVEK", # Environnement
    "OFT":   "DETEC/UVEK", "BAV":   "DETEC/UVEK", # Transports
    "OFEN":  "DETEC/UVEK", "BFE":   "DETEC/UVEK", # Énergie
    "ARE":   "DETEC/UVEK",                          # Développement territorial
    "OFROU": "DETEC/UVEK", "ASTRA": "DETEC/UVEK",  # Routes
    "OFAC":  "DETEC/UVEK", "BAZL":  "DETEC/UVEK",  # Aviation civile
    "OFCOM": "DETEC/UVEK", "BAKOM": "DETEC/UVEK",  # Communications
}

# Maps lower-case substring → canonical display name
SIMPLE_CATEGORIES: dict[str, str] = {
    "civil servant":      "Civil Servant",
    "interest group":     "Interest Group",
    "journalist":         "Journalist",
    "general public":     "General Public",
    "system":             "System",
    "city entity":        "City Entity",
    "cantonal entity":    "Cantonal Entity",
    "canton entity":      "Cantonal Entity",  # LLM may use either spelling
    "state-owned company":"State-Owned Company",
    "state owned company":"State-Owned Company",
}

# Department abbreviation pattern — full bilingual form (e.g. DFAE/EDA, DETEC/UVEK)
_DEPT_RE  = re.compile(r'\b(D[A-Z]+/[A-Z]+|ChF/BK)\b')
# Matches "Federal Council" but NOT "Federal Councillor" / "Federal Councilor"
_FC_RE    = re.compile(r'\bfederal council\b')
_FCLR_RE  = re.compile(r'\bfederal council(l)?or\b')  # councillor / councilor


# ---------------------------------------------------------------------------
# Core standardisation logic
# ---------------------------------------------------------------------------

def _std_single(token: str, pubtime=None) -> str:
    """Standardise one entity token (no pipe separators)."""
    token = token.strip()
    if not token:
        return token
    tl = token.lower()

    # 1. Federal Councillor by name (highest priority)
    #    Append the department from the authoritative composition list for pubtime.
    n2d = _name_to_dept(pubtime)
    for name in ALL_COUNCILLORS:
        if name.lower() in tl:
            dept = n2d.get(name) or _LAST_KNOWN_DEPT.get(name, "")
            return f"CF ({name} — {dept})" if dept else f"CF ({name})"

    # 2. Federal Department
    if "department" in tl:
        # a) Full bilingual pattern: DFAE/EDA, DETEC/UVEK, …
        m = _DEPT_RE.search(token)
        if m:
            return f"FD ({m.group(1)})"
        # b) Single-language half: EDI → DFI/EDI, VBS → DDPS/VBS, …
        for abbrev, full in DEPT_SINGLE_MAP.items():
            if re.search(rf'\b{re.escape(abbrev)}\b', token):
                return f"FD ({full})"
        # c) Sub-office / secretariat: OFSP → DFI/EDI, SECO → DEFR/WBF, …
        for suboffice, dept in SUBOFFICE_MAP.items():
            if re.search(rf'\b{re.escape(suboffice)}\b', token, re.IGNORECASE):
                return f"FD ({dept})"
        return "Other"  # "department" in text but abbreviation unrecognised

    # 3. Parliamentary
    # a) Exact bilingual abbreviations from controlled vocabulary
    for party in PARTIES:
        if party in token:
            return f"Parliamentary ({party})"
    # b) Alias patterns: catches variant spellings and continuation entities
    #    where the LLM drops the "Parliamentary" keyword after the first pipe
    for alias_re, canonical in PARTY_ALIAS_PATTERNS:
        if alias_re.search(token):
            return f"Parliamentary ({canonical})"
    # c) "parliamentary" keyword present but no recognisable party → incomplete
    if "parliamentary" in tl:
        return "Other"

    # 4. Federal Council (collective body) — never "Federal Councillor"
    if _FC_RE.search(tl) and not _FCLR_RE.search(tl):
        return "FC"

    # 5. Simple pass-through categories (strip surrounding details)
    for key, canonical in SIMPLE_CATEGORIES.items():
        if key in tl:
            return canonical

    # 6. Fallback
    return "Other"


def _resolve_source_target_overlap(source_std, target_std):
    """Remove from SOURCE any entity that also appears in TARGET.

    If nothing remains in SOURCE after removal → SOURCE = 'Journalist'
    (no independent critic could be identified).
    If SOURCE already has other entities → just drop the overlapping one(s).
    """
    if pd.isna(source_std) or pd.isna(target_std):
        return source_std

    src_parts = [p.strip() for p in str(source_std).split("|") if p.strip()]
    tgt_set   = {p.strip() for p in str(target_std).split("|") if p.strip()}

    remaining = [p for p in src_parts if p not in tgt_set]

    if not remaining:
        return "Journalist"
    return " | ".join(remaining)


def _postprocess(parts: list[str]) -> str:
    """Deduplicate and suppress Others when informative values exist."""
    # Deduplicate preserving order
    seen: list[str] = []
    for p in parts:
        if p not in seen:
            seen.append(p)

    informative = [p for p in seen if p != "Other"]
    if informative:
        return " | ".join(informative)   # drop Others when something useful exists
    return "Other"                        # all Others → keep one


def standardize_field(val, pubtime=None) -> str | float:
    """Standardise a full SOURCE or TARGET cell (may contain ' | ' separators)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return val
    s = str(val).strip()
    if not s or s.upper() in ("N/A", "NA", "NAN"):
        return val
    tokens = [t.strip() for t in s.split("|") if t.strip()]
    parts = [_std_single(t, pubtime) for t in tokens]
    return _postprocess(parts)


# ---------------------------------------------------------------------------
# Columns to keep in the slim output
# ---------------------------------------------------------------------------
KEEP_COLS = [
    "article_id",       # identifier – dropped later if absent
    "pubtime",
    "medium_name",
    "language",
    "SWISS_CONTEXT",
    "CRITICISM",
    "SOURCE",
    "TARGET",
    "WHAT",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Standardise RUN2 SOURCE / TARGET columns.")
    ap.add_argument("--input",      required=True, help="Path to RUN2 merged output (.parquet or .csv)")
    ap.add_argument("--output_dir", required=True, help="Output folder (created if absent); results.parquet + results.csv saved inside")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_dir  = Path(args.output_dir)

    # Load
    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    print(f"[load] {len(df):,} rows | columns: {df.columns.tolist()}")

    # Keep only relevant columns (ignore missing ones gracefully)
    present = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"[warn] columns not found and skipped: {missing}")
    df = df[present].copy()

    # Standardise — pass pubtime so CF entries include the correct department
    pubtime_col = "pubtime" if "pubtime" in df.columns else None

    def _std_row(col: str) -> pd.Series:
        if pubtime_col:
            return df.apply(lambda r: standardize_field(r[col], r[pubtime_col]), axis=1)
        return df[col].apply(standardize_field)

    df["SOURCE_STD"] = _std_row("SOURCE")
    df["TARGET_STD"] = _std_row("TARGET")

    # SOURCE entities that duplicate TARGET are removed (or replaced by Journalist)
    df["SOURCE_STD"] = df.apply(
        lambda r: _resolve_source_target_overlap(r["SOURCE_STD"], r["TARGET_STD"]),
        axis=1,
    )

    # Rows without criticism should have empty STD columns, not "Other"
    if "CRITICISM" in df.columns:
        no_crit = df["CRITICISM"].astype(str).str.strip().str.upper().isin(
            ["NO", "FALSE", "0", "NAN", "NONE", ""]
        )
        df.loc[no_crit, "SOURCE_STD"] = pd.NA
        df.loc[no_crit, "TARGET_STD"] = pd.NA
        print(f"[criticism] {no_crit.sum():,} rows without criticism → STD columns left empty")

    # Quick summary
    for col in ("SOURCE_STD", "TARGET_STD"):
        counts = df[col].value_counts(dropna=False)
        print(f"\n[{col}] top values:")
        print(counts.head(15).to_string())

    # Save both formats inside the output folder
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "results.parquet"
    csv_path     = out_dir / "results.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n[save] {len(df):,} rows → {parquet_path}")
    print(f"[save] {len(df):,} rows → {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
