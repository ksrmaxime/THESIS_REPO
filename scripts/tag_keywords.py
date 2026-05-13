"""
tag_keywords.py – Add a `matched_keywords` column to Swissdox output files.

Scans the `text` column of each article against every term in the original
Swissdox query (DE_TERMS, FR_TERMS, DEPARTMENTS, ADMIN_UNITS,
INDEPENDENT_AGENCIES, COUNCILLORS) and writes matched terms as a
pipe-separated string in a new `matched_keywords` column.

Keywords are matched with word boundaries (\b) and filtered by the article's
`language` column ("de" / "fr") so that language-specific abbreviations (e.g.
OFT = French for Office fédéral des transports) are never matched against
articles in the other language.

Usage
-----
  python scripts/tag_keywords.py --indir data/input [--outdir data/input]
  python scripts/tag_keywords.py --infile path/to/file.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.download_src import (
    ADMIN_UNITS,
    COUNCILLORS,
    DE_TERMS,
    DEPARTMENTS,
    FR_TERMS,
    INDEPENDENT_AGENCIES,
)

# ---------------------------------------------------------------------------
# Language detection for keywords
# ---------------------------------------------------------------------------

# Characters that only (or almost only) appear in French words
_FR_ACCENT_CHARS = frozenset("éèêëàâùûôœçîïú")
# Characters that only appear in German words
_DE_UMLAUT_CHARS = frozenset("äöüÄÖÜß")

# Explicit language tag for pure-ASCII terms (abbreviations, brand names, etc.)
# that cannot be detected by character heuristic alone.
# "de"   → only match against German-language articles
# "fr"   → only match against French-language articles
# "both" → match regardless of language (proper names, cross-lingual brands)
_TERM_LANG: dict[str, str] = {
    # ── Federal departments ──────────────────────────────────────────────────
    "VBS": "de",    "DDPS": "fr",
    "EDA": "de",    "DFAE": "fr",
    "UVEK": "de",   "DETEC": "fr",
    "EJPD": "de",   "DFJP": "fr",
    "EDI": "de",    "DFI": "fr",
    "EFD": "de",    "DFF": "fr",
    "WBF": "de",    "DEFR": "fr",
    # ── Admin units – DFAE / EDA ────────────────────────────────────────────
    "GS-EDA": "de",   "SG-DFAE": "fr",
    "DDIP": "fr",
    "DEZA": "de",     "DDC": "fr",
    # ── Admin units – EDI / DFI ─────────────────────────────────────────────
    "GS-EDI": "de",   "SG-DFI": "fr",
    "EBG": "de",      "BFEG": "fr",
    "BAK": "de",      "OFC": "fr",
    "AFS": "fr",
    "MeteoSchweiz": "de",               # MétéoSuisse has é → auto-detected as fr
    "BAG": "de",      "OFSP": "fr",
    "BLV": "de",      "OSAV": "fr",
    "BFS": "de",      "OFS": "fr",
    "BSV": "de",      "OFAS": "fr",
    # ── Admin units – EJPD / DFJP ───────────────────────────────────────────
    "GS-EJPD": "de",  "SG-DFJP": "fr",
    "SEM": "both",    # FR origin but used in both languages in Swiss press
    "OFJ": "fr",
    "fedpol": "both",
    "Service SCPT": "fr",
    # ── Admin units – VBS / DDPS ────────────────────────────────────────────
    "GS-VBS": "de",   "SG-DDPS": "fr",
    "BABS": "de",     "OFPP": "fr",
    "armasuisse": "both",
    "swisstopo": "both",
    "BASPO": "de",    "OFSPO": "fr",
    "BACS": "de",     "OFCS": "fr",
    "SEPOS": "fr",
    "NDB": "de",      "SRC": "fr",
    "OAC": "fr",
    "Schweizer Armee": "de",            # no umlaut → would be "both" by heuristic
    # ── Admin units – EFD / DFF ─────────────────────────────────────────────
    "GS-EFD": "de",   "SG-DFF": "fr",
    "SIF": "de",      "SFI": "fr",
    "EFV": "de",      "AFF": "fr",
    "EPA": "de",      "OFPER": "fr",
    "ESTV": "de",     "AFC": "fr",
    "BAZG": "de",     "OFDF": "fr",
    "OFIT": "fr",
    "BBL": "de",      "OFCL": "fr",
    # ── Admin units – WBF / DEFR ────────────────────────────────────────────
    "GS-WBF": "de",   "SG-DEFR": "fr",
    "SECO": "both",   # FR origin but standard across both languages in Swiss press
    "SBFI": "de",     "SEFRI": "fr",
    "BLW": "de",      "OFAG": "fr",
    "OFAE": "fr",
    "BWO": "de",      "OFL": "fr",
    "ZIVI": "de",     "CIVI": "fr",
    # ── Admin units – UVEK / DETEC ──────────────────────────────────────────
    "GS-UVEK": "de",  "SG-DETEC": "fr",
    "BAV": "de",      "OFT": "fr",     # OFT = "oft" (often) in German → must be fr-only
    "BAZL": "de",     "OFAC": "fr",
    "BFE": "de",      "OFEN": "fr",
    "ASTRA": "de",    "OFROU": "fr",
    "BAKOM": "de",    "OFCOM": "fr",
    "BAFU": "de",     "OFEV": "fr",
    "ARE": "both",    # same abbreviation used in both languages
    # ── Independent agencies ────────────────────────────────────────────────
    "ENSI": "de",     "IFSN": "fr",
    "ESTI": "de",
    "SUST": "de",     "SESE": "fr",
    "ElCom": "de",    "EICom": "fr",
    "ComCom": "both",
    "UBI": "de",      "AIEP": "fr",
    "PostCom": "both",
    "RailCom": "both",
    "PUE": "de",      "SPR": "fr",
    "WEKO": "de",     "COMCO": "fr",
    "ETH-Bereich": "de",  "domaine des EPF": "fr",
    "EHB": "de",      "HEFP": "fr",
    "Innosuisse": "both",
    "FINMA": "both",
    "EFK": "de",      "CDF": "fr",
    "PUBLICA": "both",
    "IGE": "de",      "IPI": "fr",
    "METAS": "both",
    "SIR": "de",      "ISDC": "fr",
    "RAB": "de",      "ASR": "fr",
    "ESBK": "de",     "CFMJ": "fr",
    "ESchK": "de",    "CAF": "fr",
    "NKVF": "de",     "CNPT": "fr",
    "EKM": "de",      "CFM": "fr",
    "Swissmedic": "both",
    "SNM": "de",      "MNS": "fr",
    "Pro Helvetia": "both",
}


def _kw_language(kw: str) -> str:
    """Return 'de', 'fr', or 'both' for a keyword from DEPARTMENTS/ADMIN_UNITS/INDEPENDENT_AGENCIES."""
    if kw in _TERM_LANG:
        return _TERM_LANG[kw]
    chars = set(kw)
    has_fr = bool(chars & _FR_ACCENT_CHARS)
    has_de = bool(chars & _DE_UMLAUT_CHARS)
    if has_fr and not has_de:
        return "fr"
    if has_de and not has_fr:
        return "de"
    return "both"


# ---------------------------------------------------------------------------
# Pre-compiled patterns: (keyword, pattern, language)
# ---------------------------------------------------------------------------
def _pat(kw: str) -> re.Pattern:
    return re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)


_ALL_PATTERNS: list[tuple[str, re.Pattern, str]] = []

for _kw in DE_TERMS:
    _ALL_PATTERNS.append((_kw, _pat(_kw), "de"))

for _kw in FR_TERMS:
    _ALL_PATTERNS.append((_kw, _pat(_kw), "fr"))

for _kw in COUNCILLORS:
    _ALL_PATTERNS.append((_kw, _pat(_kw), "both"))

for _kw in DEPARTMENTS + ADMIN_UNITS + INDEPENDENT_AGENCIES:
    _ALL_PATTERNS.append((_kw, _pat(_kw), _kw_language(_kw)))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def find_matched_keywords(text: str, lang: str) -> str:
    """Return pipe-separated list of keywords found in *text* for article language *lang*."""
    if not isinstance(text, str) or not text:
        return ""
    matched: list[str] = []
    for kw, pat, kw_lang in _ALL_PATTERNS:
        if kw_lang in ("both", lang) and pat.search(text):
            matched.append(kw)
    return "|".join(matched)


def tag_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add `matched_keywords` column using the `text` and `language` columns."""
    t0 = time.time()
    text_col = df["text"] if "text" in df.columns else df.get("content", pd.Series("", index=df.index))
    lang_col = df["language"] if "language" in df.columns else pd.Series("both", index=df.index)
    df = df.copy()
    df["matched_keywords"] = [
        find_matched_keywords(str(text) if pd.notna(text) else "", str(lang) if pd.notna(lang) else "both")
        for text, lang in zip(text_col, lang_col)
    ]
    print(
        f"  Tagged {len(df):,} rows in {time.time() - t0:.1f}s – "
        f"{(df['matched_keywords'] != '').sum():,} rows with at least one match"
    )
    return df


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def process_file(src: Path, dst: Path) -> None:
    suffix = src.suffix.lower()
    print(f"[tag_keywords] Reading {src} …")
    if suffix == ".parquet":
        df = pd.read_parquet(src)
    elif suffix == ".csv":
        df = pd.read_csv(src, low_memory=False)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    print(f"  Shape: {df.shape}")
    df = tag_dataframe(df)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() == ".parquet":
        df.to_parquet(dst, index=False)
    else:
        df_csv = df.copy()
        for col in ("text", "lead", "content"):
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].str.replace(r"\n+", " ", regex=True).str.strip()
        df_csv.to_csv(dst, index=False, lineterminator="\n")
    print(f"  Saved → {dst}")


def resolve_output_path(src: Path, outdir: Path | None) -> Path:
    if outdir is None:
        stem = src.stem + "_tagged"
        return src.with_name(stem + src.suffix)
    return outdir / (src.stem + "_tagged" + src.suffix)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tag Swissdox output files with matched keywords.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--indir", type=Path, help="Directory containing parquet/csv files to tag")
    group.add_argument("--infile", type=Path, help="Single parquet or CSV file to tag")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input, with _tagged suffix)",
    )
    args = ap.parse_args()

    if args.infile:
        files = [args.infile]
    else:
        indir: Path = args.indir
        files = sorted(indir.glob("*.parquet")) + sorted(indir.glob("*.csv"))
        files = [f for f in files if "_tagged" not in f.stem]
        if not files:
            print(f"[tag_keywords] No parquet/csv files found in {indir}")
            sys.exit(0)

    print(f"[tag_keywords] Processing {len(files)} file(s) …")
    for src in files:
        dst = resolve_output_path(src, args.outdir)
        process_file(src, dst)

    print("[tag_keywords] Done.")


if __name__ == "__main__":
    main()
