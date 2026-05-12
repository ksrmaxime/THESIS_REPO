from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge per-year Swissdox parquet files into one dataset."
    )
    ap.add_argument("--indir", default="data/input", help="Folder containing year_YYYY/ subdirs")
    ap.add_argument("--outdir", default="data/input", help="Destination for the merged file")
    ap.add_argument("--out-stem", default="swissdox_all", help="Output filename stem (no extension)")
    args = ap.parse_args()

    indir = Path(args.indir)
    parquet_files = sorted(indir.glob("year_*/*.parquet"))

    if not parquet_files:
        print(f"[merge] No parquet files found under {indir}/year_*/")
        sys.exit(1)

    dfs = []
    for p in parquet_files:
        df = pd.read_parquet(p)
        print(f"  {p.parent.name}/{p.name}: {len(df):,} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["article_id"])
    merged = merged.sort_values("pubtime").reset_index(drop=True)

    print(f"\n[merge] Total after dedup: {len(merged):,} rows")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = out_dir / f"{args.out_stem}.parquet"

    merged.to_parquet(out_parquet, index=False)

    print(f"[merge] Saved: {out_parquet}")


if __name__ == "__main__":
    main()
