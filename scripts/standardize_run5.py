"""
Standardise the source_category column from a RUN5 output file.

Usage:
    python scripts/standardize_run5.py --input  data/output/run5_merged_jobXXX/results.parquet \
                                        --output_dir data/output/run5_standardized_jobXXX

Applies the exact same controlled-vocabulary rules as standardize_run2.py (SOURCE column)
but only to the single source_category column produced by run5.
Input can be .parquet or .csv ; output folder receives results.parquet + results.csv.
"""

from __future__ import annotations
import sys
import re
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from src.run5_prompts import _COUNCIL_COMPOSITIONS, get_council_for_date

# ---------------------------------------------------------------------------
# Reference data (identical to standardize_run2)
# ---------------------------------------------------------------------------

ALL_COUNCILLORS: list[str] = sorted(
    {name for _, _, comp in _COUNCIL_COMPOSITIONS for name in comp.values()},
    key=lambda n: -len(n),
)


def _name_to_dept(pubtime) -> dict[str, str]:
    composition = get_council_for_date(pubtime)
    return {name: dept for dept, name in composition.items()}


_LAST_KNOWN_DEPT: dict[str, str] = {
    name: dept
    for _, _, comp in _COUNCIL_COMPOSITIONS
    for dept, name in comp.items()
}

# ---------------------------------------------------------------------------
# Controlled vocabulary (identical to standardize_run2)
# ---------------------------------------------------------------------------

PARTIES: list[str] = [
    "Die Grünen/Les Verts",
    "Die Mitte/Le Centre",
    "SVP/UDC",
    "FDP/PLR",
    "GLP/PVL",
    "SP/PS",
]

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

DEPT_SINGLE_MAP: dict[str, str] = {
    "DFAE":  "DFAE/EDA",
    "DFI":   "DFI/EDI",
    "DFJP":  "DFJP/EJPD",
    "DDPS":  "DDPS/VBS",
    "DFF":   "DFF/EFD",
    "DEFR":  "DEFR/WBF",
    "DETEC": "DETEC/UVEK",
    "ChF":   "ChF/BK",
    "EDA":   "DFAE/EDA",
    "EDI":   "DFI/EDI",
    "EJPD":  "DFJP/EJPD",
    "VBS":   "DDPS/VBS",
    "EFD":   "DFF/EFD",
    "WBF":   "DEFR/WBF",
    "UVEK":  "DETEC/UVEK",
    "BK":    "ChF/BK",
}

SUBOFFICE_MAP: dict[str, str] = {
    "OFSP": "DFI/EDI",  "BAG":  "DFI/EDI",
    "OFAS": "DFI/EDI",  "BSV":  "DFI/EDI",
    "OFS":  "DFI/EDI",  "BFS":  "DFI/EDI",
    "OFC":  "DFI/EDI",  "BAK":  "DFI/EDI",
    "OSAV": "DFI/EDI",  "BLV":  "DFI/EDI",
    "BFEG": "DFI/EDI",
    "AFS":  "DFI/EDI",  "BAR":  "DFI/EDI",
    "MétéoSuisse": "DFI/EDI",
    "DSH":  "DFAE/EDA",
    "DAS":  "DFAE/EDA",
    "SEM":    "DFJP/EJPD",
    "fedpol": "DFJP/EJPD",
    "OFJ":    "DFJP/EJPD", "BJ": "DFJP/EJPD",
    "ISDC":   "DFJP/EJPD",
    "armasuisse": "DDPS/VBS",
    "swisstopo":  "DDPS/VBS",
    "OFPP": "DDPS/VBS", "BABS": "DDPS/VBS",
    "AFC":   "DFF/EFD",  "ESTV":  "DFF/EFD",
    "AFD":   "DFF/EFD",  "BAZG":  "DFF/EFD",
    "CDF":   "DFF/EFD",  "EFK":   "DFF/EFD",
    "AFF":   "DFF/EFD",  "FFA":   "DFF/EFD",
    "FOITT": "DFF/EFD",  "BIT":   "DFF/EFD",
    "FOBL":  "DFF/EFD",  "BBL":   "DFF/EFD",
    "SECO":  "DEFR/WBF",
    "SBFI":  "DEFR/WBF", "SEFRI": "DEFR/WBF",
    "SERI":  "DEFR/WBF",
    "OFAG":  "DEFR/WBF", "BLW":   "DEFR/WBF",
    "OFEV":  "DETEC/UVEK", "BAFU":  "DETEC/UVEK",
    "OFT":   "DETEC/UVEK", "BAV":   "DETEC/UVEK",
    "OFEN":  "DETEC/UVEK", "BFE":   "DETEC/UVEK",
    "ARE":   "DETEC/UVEK",
    "OFROU": "DETEC/UVEK", "ASTRA": "DETEC/UVEK",
    "OFAC":  "DETEC/UVEK", "BAZL":  "DETEC/UVEK",
    "OFCOM": "DETEC/UVEK", "BAKOM": "DETEC/UVEK",
}

SIMPLE_CATEGORIES: dict[str, str] = {
    "civil servant":      "Civil Servant",
    "interest group":     "Interest Group",
    "journalist":         "Journalist",
    "general public":     "General Public",
    "system":             "System",
    "city entity":        "City Entity",
    "cantonal entity":    "Cantonal Entity",
    "canton entity":      "Cantonal Entity",
    "state-owned company":"State-Owned Company",
    "state owned company":"State-Owned Company",
}

_DEPT_RE = re.compile(r'\b(D[A-Z]+/[A-Z]+|ChF/BK)\b')
_FC_RE   = re.compile(r'\bfederal council\b')
_FCLR_RE = re.compile(r'\bfederal council(l)?or\b')

# ---------------------------------------------------------------------------
# Core standardisation logic (identical to standardize_run2)
# ---------------------------------------------------------------------------

def _std_single(token: str, pubtime=None) -> str:
    token = token.strip()
    if not token:
        return token
    tl = token.lower()

    n2d = _name_to_dept(pubtime)
    for name in ALL_COUNCILLORS:
        if name.lower() in tl:
            dept = n2d.get(name) or _LAST_KNOWN_DEPT.get(name, "")
            return f"CF ({name} — {dept})" if dept else f"CF ({name})"

    if "department" in tl:
        if re.search(r'\[', token):
            return "Federal Administration"
        m = _DEPT_RE.search(token)
        if m:
            return f"FD ({m.group(1)})"
        for abbrev, full in DEPT_SINGLE_MAP.items():
            if re.search(rf'\b{re.escape(abbrev)}\b', token):
                return f"FD ({full})"
        for suboffice, dept in SUBOFFICE_MAP.items():
            if re.search(rf'\b{re.escape(suboffice)}\b', token, re.IGNORECASE):
                return f"FD ({dept})"
        return "Other"

    if "parliamentary" in tl and re.search(r'\[', token):
        return "Parliamentarians"
    for party in PARTIES:
        if party in token:
            return f"Parliamentary ({party})"
    for alias_re, canonical in PARTY_ALIAS_PATTERNS:
        if alias_re.search(token):
            return f"Parliamentary ({canonical})"
    if "parliamentary" in tl:
        return "Other"

    if _FC_RE.search(tl) and not _FCLR_RE.search(tl):
        return "FC"

    for key, canonical in SIMPLE_CATEGORIES.items():
        if key in tl:
            return canonical

    return "Other"


def _postprocess(parts: list[str]) -> str:
    seen: list[str] = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    informative = [p for p in seen if p != "Other"]
    if informative:
        return " | ".join(informative)
    return "Other"


def standardize_field(val, pubtime=None) -> str | float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return val
    s = str(val).strip()
    if not s or s.upper() in ("N/A", "NA", "NAN"):
        return val
    tokens = [t.strip() for t in s.split("|") if t.strip()]
    parts = [_std_single(t, pubtime) for t in tokens]
    return _postprocess(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Standardise RUN5 source_category column.")
    ap.add_argument("--input",      required=True, help="Path to RUN5 merged output (.parquet or .csv)")
    ap.add_argument("--output_dir", required=True, help="Output folder; results.parquet + results.csv saved inside")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)

    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    print(f"[load] {len(df):,} rows | columns: {df.columns.tolist()}")

    if "source_category" not in df.columns:
        print("[ERROR] Column 'source_category' not found in input file.")
        return 1

    pubtime_col = "pubtime" if "pubtime" in df.columns else None

    if pubtime_col:
        df["SOURCE_CAT_STD"] = df.apply(
            lambda r: standardize_field(r["source_category"], r[pubtime_col]), axis=1
        )
    else:
        df["SOURCE_CAT_STD"] = df["source_category"].apply(standardize_field)

    counts = df["SOURCE_CAT_STD"].value_counts(dropna=False)
    print(f"\n[SOURCE_CAT_STD] top values:")
    print(counts.head(20).to_string())

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
