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

# Department abbreviation pattern (e.g. DFAE/EDA, DETEC/UVEK, DEFR/WBF …)
_DEPT_RE   = re.compile(r'\b(D[A-Z]+/[A-Z]+)\b')
# Matches "Federal Council" but NOT "Federal Councillor" / "Federal Councilor"
_FC_RE     = re.compile(r'\bfederal council\b')
_FCLR_RE   = re.compile(r'\bfederal council(l)?or\b')  # councillor / councilor


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
            dept = n2d.get(name, "")
            return f"CF ({name} — {dept})" if dept else f"CF ({name})"

    # 2. Federal Department
    if "department" in tl:
        m = _DEPT_RE.search(token)
        return f"FD ({m.group(1)})" if m else "FD"

    # 3. Parliamentary – identified by party abbreviation
    for party in PARTIES:
        if party in token:
            return f"Parliamentary ({party})"
    if "parliamentary" in tl:
        return "Parliamentary"

    # 4. Federal Council (collective body) — never "Federal Councillor"
    if _FC_RE.search(tl) and not _FCLR_RE.search(tl):
        return "FC"

    # 5. Simple pass-through categories (strip surrounding details)
    for key, canonical in SIMPLE_CATEGORIES.items():
        if key in tl:
            return canonical

    # 6. Fallback
    return "Other"


def standardize_field(val, pubtime=None) -> str | float:
    """Standardise a full SOURCE or TARGET cell (may contain ' | ' separators)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return val
    s = str(val).strip()
    if not s or s.upper() in ("N/A", "NA", "NAN"):
        return val
    tokens = [t.strip() for t in s.split("|") if t.strip()]
    return " | ".join(_std_single(t, pubtime) for t in tokens)


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
    ap.add_argument("--input",  required=True, help="Path to RUN2 merged output (.parquet or .csv)")
    ap.add_argument("--output", required=True, help="Destination file (.parquet or .csv)")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

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

    # Quick summary
    for col in ("SOURCE_STD", "TARGET_STD"):
        counts = df[col].value_counts(dropna=False)
        print(f"\n[{col}] top values:")
        print(counts.head(15).to_string())

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n[save] {len(df):,} rows → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
