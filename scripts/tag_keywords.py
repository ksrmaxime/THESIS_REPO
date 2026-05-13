"""
tag_keywords.py – Add a `matched_keywords` column to Swissdox output files.

Scans the `text` column of each article against every term in the original
Swissdox query (DE_TERMS, FR_TERMS, DEPARTMENTS, ADMIN_UNITS,
INDEPENDENT_AGENCIES, COUNCILLORS) and writes matched terms as a
pipe-separated string in a new `matched_keywords` column.

Usage
-----
  python scripts/tag_keywords.py --indir data/input [--outdir data/input]
  python scripts/tag_keywords.py --infile path/to/file.parquet
"""

from __future__ import annotations

import argparse
import csv
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

# Flat list of (keyword, compiled_pattern) pairs – built once at module load.
_ALL_PATTERNS: list[tuple[str, re.Pattern]] = []
for _kw in (
    DE_TERMS + FR_TERMS + DEPARTMENTS + ADMIN_UNITS + INDEPENDENT_AGENCIES + COUNCILLORS
):
    _ALL_PATTERNS.append((_kw, re.compile(re.escape(_kw), re.IGNORECASE)))


def find_matched_keywords(text: str) -> str:
    """Return pipe-separated list of keywords found in *text*."""
    if not isinstance(text, str) or not text:
        return ""
    matched: list[str] = []
    for kw, pat in _ALL_PATTERNS:
        if pat.search(text):
            matched.append(kw)
    return "|".join(matched)


def tag_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add `matched_keywords` column based on the `text` column only."""
    t0 = time.time()
    text_col = df["text"] if "text" in df.columns else df.get("content", pd.Series("", index=df.index))
    df = df.copy()
    df["matched_keywords"] = text_col.fillna("").map(find_matched_keywords)
    print(
        f"  Tagged {len(df):,} rows in {time.time() - t0:.1f}s – "
        f"{(df['matched_keywords'] != '').sum():,} rows with at least one match"
    )
    return df


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
        df.to_csv(dst, index=False, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
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
        # Skip already-tagged files to avoid re-processing
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
