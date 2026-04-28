"""
Comprehensive analysis and visualisation of RUN2 standardised output.

Usage:
    python scripts/analyze_run2.py \
        --input  data/output/run2_standardized_jobXXX/results.parquet \
        --output_dir data/output/run2_analysis_jobXXX

Produces:
    figures/  — PNG plots (numbered 01–12)
    tables/   — CSV tables for each analysis block
    report.txt — plain-text summary
"""

from __future__ import annotations
import sys
import re
import argparse
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── Palette ──────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
TEAL   = "#0891B2"
GRAY   = "#6B7280"
YELLOW = "#CA8A04"

CAT_COLORS = [BLUE, RED, GREEN, ORANGE, PURPLE, TEAL, GRAY, YELLOW,
              "#DB2777", "#059669", "#B45309", "#1D4ED8", "#374151"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, title: str = "") -> None:
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


def _flag(val) -> bool:
    """True if val is a 'yes' / truthy criticism/swiss flag."""
    if pd.isna(val):
        return False
    return str(val).strip().upper() in ("YES", "TRUE", "1", "OUI", "JA")


def _explode_std(series: pd.Series) -> pd.Series:
    """Split pipe-separated STD cells and explode into individual entities."""
    return (
        series.dropna()
        .astype(str)
        .str.split(r"\s*\|\s*")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )


def _first_std(series: pd.Series) -> pd.Series:
    """Keep only the first entity (before first pipe) for 1-to-1 analyses."""
    return series.astype(str).str.split("|").str[0].str.strip()


def _dept_from_std(val: str) -> str | None:
    """Extract dept abbreviation from FD or CF standardised values."""
    if pd.isna(val):
        return None
    m = re.search(r'\b([A-Z]+/[A-Z]+)\b', str(val))
    return m.group(1) if m else None


def _quarter_label(period) -> str:
    return str(period)


def _bar_h(ax, series: pd.Series, color=BLUE, pct_total: int | None = None) -> None:
    """Horizontal bar chart on ax from a value_counts Series."""
    vals = series.sort_values()
    colors = CAT_COLORS[:len(vals)][::-1]
    bars = ax.barh(vals.index, vals.values, color=colors)
    total = pct_total or vals.sum()
    for bar, v in zip(bars, vals.values):
        pct = 100 * v / total if total else 0
        ax.text(bar.get_width() + vals.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:,}  ({pct:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, vals.max() * 1.25)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────────────────

def fig_overview(df: pd.DataFrame, out: Path) -> dict:
    """Fig 01 — Dataset overview: Swiss / Criticism / Language / Time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    swiss_counts  = df["SWISS_CONTEXT"].map(lambda v: "Swiss" if _flag(v) else "Non-Swiss").value_counts()
    crit_counts   = df["CRITICISM"].map(lambda v: "YES" if _flag(v) else "NO").value_counts()
    lang_counts   = df["language"].fillna("unknown").value_counts().head(10)
    quarter_total = df.groupby("quarter").size()

    for ax, counts, title, color in zip(
        axes.flat[:3],
        [swiss_counts, crit_counts, lang_counts],
        ["Swiss Context", "Criticism", "Language (top 10)"],
        [TEAL, RED, PURPLE],
    ):
        _bar_h(ax, counts, color=color, pct_total=len(df))
        ax.set_title(title)

    ax_time = axes[1][1]
    ax_time.bar(range(len(quarter_total)), quarter_total.values, color=BLUE, alpha=0.85)
    ax_time.set_xticks(range(len(quarter_total)))
    ax_time.set_xticklabels(quarter_total.index.astype(str), rotation=45, ha="right", fontsize=7)
    ax_time.set_title("Articles per Quarter")
    ax_time.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save(fig, out / "fig01_overview.png", "Dataset Overview")
    return {
        "total_articles": len(df),
        "swiss_yes": int(swiss_counts.get("Swiss", 0)),
        "swiss_no":  int(swiss_counts.get("Non-Swiss", 0)),
        "criticism_yes": int(crit_counts.get("YES", 0)),
        "criticism_no":  int(crit_counts.get("NO", 0)),
    }


def fig_source_dist(df_crit: pd.DataFrame, out: Path) -> pd.Series:
    """Fig 02 — SOURCE_STD distribution (exploded)."""
    counts = _explode_std(df_crit["SOURCE_STD"]).value_counts()
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.45)))
    _bar_h(ax, counts, pct_total=len(df_crit))
    ax.set_title("Source Distribution  (criticism articles, entities exploded)")
    ax.set_xlabel("Count")
    _save(fig, out / "fig02_source_dist.png")
    return counts


def fig_target_dist(df_crit: pd.DataFrame, out: Path) -> pd.Series:
    """Fig 03 — TARGET_STD distribution (exploded)."""
    counts = _explode_std(df_crit["TARGET_STD"]).value_counts()
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.45)))
    _bar_h(ax, counts, pct_total=len(df_crit))
    ax.set_title("Target Distribution  (criticism articles, entities exploded)")
    ax.set_xlabel("Count")
    _save(fig, out / "fig03_target_dist.png")
    return counts


def fig_what_dist(df_crit: pd.DataFrame, out: Path) -> None:
    """Fig 04 — WHAT distribution overall and by source type."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall
    what = df_crit["WHAT"].dropna().value_counts()
    colors = [BLUE, ORANGE]
    ax1.bar(what.index, what.values, color=colors[:len(what)])
    for i, (label, v) in enumerate(what.items()):
        ax1.text(i, v + what.max() * 0.01, f"{v:,}\n({100*v/what.sum():.1f}%)",
                 ha="center", fontsize=9)
    ax1.set_title("WHAT — Overall")
    ax1.set_ylim(0, what.max() * 1.2)

    # Right: by first SOURCE_STD
    src_first = _first_std(df_crit["SOURCE_STD"])
    combo = pd.DataFrame({"source": src_first, "what": df_crit["WHAT"]}).dropna()
    top_sources = combo["source"].value_counts().head(8).index
    combo = combo[combo["source"].isin(top_sources)]
    pivot = combo.groupby(["source", "what"]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct = pivot_pct.loc[pivot_pct.sum(axis=1).sort_values(ascending=True).index]
    pivot_pct.plot(kind="barh", ax=ax2, color=colors[:pivot_pct.shape[1]], stacked=True)
    ax2.set_title("WHAT — by Source type (top 8, % stacked)")
    ax2.set_xlabel("% of criticisms")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    for label in ax2.get_yticklabels():
        label.set_fontsize(8)

    _save(fig, out / "fig04_what_dist.png", "Nature of Criticism (WHAT)")


def fig_time_evolution(df_crit: pd.DataFrame, col: str, label: str,
                       fignum: str, out: Path) -> None:
    """Fig 05/06 — Quarterly evolution of SOURCE or TARGET (stacked bars)."""
    entities = _explode_std(df_crit[[col, "quarter"]]
                            .rename(columns={col: "entity"})
                            .set_index("quarter")["entity"]
                            .reset_index(drop=False)
                            .set_index("quarter")
                            .squeeze())

    # Rebuild with quarter
    rows = []
    for _, row in df_crit[["quarter", col]].dropna().iterrows():
        for e in str(row[col]).split("|"):
            e = e.strip()
            if e:
                rows.append({"quarter": row["quarter"], "entity": e})
    tmp = pd.DataFrame(rows)
    if tmp.empty:
        return

    pivot = tmp.groupby(["quarter", "entity"]).size().unstack(fill_value=0)
    top_cats = pivot.sum().sort_values(ascending=False).head(10).index
    pivot = pivot[top_cats]

    fig, ax = plt.subplots(figsize=(16, 6))
    bottom = np.zeros(len(pivot))
    quarters = pivot.index.astype(str)
    x = np.arange(len(quarters))

    for i, cat in enumerate(top_cats):
        vals = pivot[cat].values
        ax.bar(x, vals, bottom=bottom, label=cat,
               color=CAT_COLORS[i % len(CAT_COLORS)], alpha=0.88)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=45, ha="right", fontsize=7)
    ax.legend(loc="upper left", fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1))
    ax.set_title(f"{label} — Quarterly Evolution (top 10 categories)")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    _save(fig, out / f"{fignum}_{label.lower().replace(' ', '_')}_time.png")


def fig_heatmap(df_crit: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Fig 07 — SOURCE × TARGET confusion matrix heatmap."""
    src = _first_std(df_crit["SOURCE_STD"])
    tgt = _first_std(df_crit["TARGET_STD"])
    combo = pd.DataFrame({"source": src, "target": tgt}).dropna()

    # Keep top categories for readability
    top_src = combo["source"].value_counts().head(10).index
    top_tgt = combo["target"].value_counts().head(10).index
    combo = combo[combo["source"].isin(top_src) & combo["target"].isin(top_tgt)]

    matrix = combo.groupby(["source", "target"]).size().unstack(fill_value=0)
    matrix_pct = matrix.div(matrix.sum(axis=1), axis=0) * 100

    cmap = LinearSegmentedColormap.from_list("blue_scale", ["#EFF6FF", "#1D4ED8"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, fmt, title in zip(
        axes,
        [matrix, matrix_pct],
        ["{:.0f}", "{:.1f}%"],
        ["SOURCE × TARGET  (raw counts)", "SOURCE × TARGET  (% of row)"],
    ):
        im = ax.imshow(data.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index, fontsize=7)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.03)
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                v = data.values[i, j]
                label = fmt.format(v)
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=6, color="white" if v > data.values.max() * 0.6 else "black")

    _save(fig, out / "fig07_heatmap.png", "Source × Target Heatmap")
    return matrix


def fig_dept_vs_person(df_crit: pd.DataFrame, out: Path) -> None:
    """Fig 08 — Department vs its head: FD(dept) vs CF(person) targeting."""
    tgt = _first_std(df_crit["TARGET_STD"])
    rows = []
    for val in tgt.dropna():
        if val.startswith("FD ("):
            dept = _dept_from_std(val)
            if dept:
                rows.append({"dept": dept, "type": "Department (FD)"})
        elif val.startswith("CF ("):
            dept = _dept_from_std(val)
            if dept:
                rows.append({"dept": dept, "type": "Person (CF)"})

    if not rows:
        print("  [skip] no FD/CF targets found")
        return

    tmp = pd.DataFrame(rows)
    pivot = tmp.groupby(["dept", "type"]).size().unstack(fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True).drop(columns="total")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(pivot) * 0.5)))

    # Raw counts
    pivot.plot(kind="barh", ax=ax1, color=[BLUE, RED], stacked=False)
    ax1.set_title("Department vs Person targeting\n(raw counts)")
    ax1.set_xlabel("Count")
    ax1.legend(fontsize=8)

    # Percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct.plot(kind="barh", ax=ax2, color=[BLUE, RED], stacked=True)
    ax2.set_title("Department vs Person targeting\n(% stacked)")
    ax2.set_xlabel("% of criticisms")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.legend(fontsize=8)

    _save(fig, out / "fig08_dept_vs_person.png", "Institutional vs Personal Targeting by Department")


def fig_top_combinations(df_crit: pd.DataFrame, out: Path) -> pd.Series:
    """Fig 09 — Top 20 SOURCE → TARGET pairs."""
    src = _first_std(df_crit["SOURCE_STD"])
    tgt = _first_std(df_crit["TARGET_STD"])
    combo = pd.Series(
        [f"{s}  →  {t}" for s, t in zip(src, tgt) if pd.notna(s) and pd.notna(t)]
    ).value_counts().head(20)

    fig, ax = plt.subplots(figsize=(12, max(5, len(combo) * 0.45)))
    _bar_h(ax, combo, pct_total=len(df_crit))
    ax.set_title("Top 20 Source → Target Combinations")
    ax.set_xlabel("Count")
    _save(fig, out / "fig09_top_combinations.png")
    return combo


def fig_language_breakdown(df_crit: pd.DataFrame, out: Path) -> None:
    """Fig 10 — Source/Target/WHAT proportions by language."""
    langs = df_crit["language"].fillna("unknown").value_counts().head(4).index
    fig, axes = plt.subplots(len(langs), 3, figsize=(16, len(langs) * 3.5))
    if len(langs) == 1:
        axes = [axes]

    for row_axes, lang in zip(axes, langs):
        sub = df_crit[df_crit["language"] == lang]
        for ax, col, title in zip(
            row_axes,
            ["SOURCE_STD", "TARGET_STD", "WHAT"],
            ["Source", "Target", "WHAT"],
        ):
            if col == "WHAT":
                counts = sub[col].dropna().value_counts()
            else:
                counts = _explode_std(sub[col]).value_counts().head(8)
            if counts.empty:
                ax.axis("off")
                continue
            _bar_h(ax, counts, pct_total=len(sub))
            ax.set_title(f"{lang} — {title}", fontsize=9)

    _save(fig, out / "fig10_language_breakdown.png", "Analysis by Language")


def fig_what_by_target(df_crit: pd.DataFrame, out: Path) -> None:
    """Fig 11 — WHAT breakdown by target type."""
    tgt_first = _first_std(df_crit["TARGET_STD"])
    combo = pd.DataFrame({"target": tgt_first, "what": df_crit["WHAT"]}).dropna()
    top_tgt = combo["target"].value_counts().head(10).index
    combo = combo[combo["target"].isin(top_tgt)]
    pivot = combo.groupby(["target", "what"]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct = pivot_pct.loc[pivot_pct.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot_pct) * 0.55)))
    colors = [BLUE, ORANGE]
    pivot_pct.plot(kind="barh", ax=ax, color=colors[:pivot_pct.shape[1]], stacked=True)
    ax.set_title("WHAT — by Target type (top 10, % stacked)")
    ax.set_xlabel("% of criticisms")
    ax.legend(loc="lower right", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    for label in ax.get_yticklabels():
        label.set_fontsize(8)
    _save(fig, out / "fig11_what_by_target.png")


def fig_medium_breakdown(df_crit: pd.DataFrame, out: Path) -> None:
    """Fig 12 — Top newspapers by criticism volume."""
    top_media = df_crit["medium_name"].fillna("Unknown").value_counts().head(15)

    fig, ax = plt.subplots(figsize=(10, max(4, len(top_media) * 0.45)))
    _bar_h(ax, top_media, pct_total=len(df_crit))
    ax.set_title("Top 15 Media Outlets by Criticism Volume")
    ax.set_xlabel("Count")
    _save(fig, out / "fig12_media_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(stats: dict, out: Path) -> None:
    lines = [
        "=" * 70,
        "  RUN2 ANALYSIS REPORT",
        "=" * 70,
        "",
        "── DATASET OVERVIEW ──────────────────────────────────────────────",
        f"  Total articles          : {stats['total_articles']:>8,}",
        f"  Swiss context (YES)     : {stats['swiss_yes']:>8,}  ({100*stats['swiss_yes']/stats['total_articles']:.1f}%)",
        f"  Swiss context (NO)      : {stats['swiss_no']:>8,}  ({100*stats['swiss_no']/stats['total_articles']:.1f}%)",
        f"  With criticism (YES)    : {stats['criticism_yes']:>8,}  ({100*stats['criticism_yes']/stats['total_articles']:.1f}%)",
        f"  Without criticism (NO)  : {stats['criticism_no']:>8,}  ({100*stats['criticism_no']/stats['total_articles']:.1f}%)",
        f"  Criticisms with full STD: {stats['full_std']:>8,}  ({100*stats['full_std']/max(stats['criticism_yes'],1):.1f}% of YES)",
        "",
        "── TOP SOURCE TYPES ──────────────────────────────────────────────",
    ]
    for cat, cnt in stats["source_counts"].head(10).items():
        pct = 100 * cnt / stats["source_counts"].sum()
        lines.append(f"  {cat:<35} {cnt:>6,}  ({pct:.1f}%)")
    lines += [
        "",
        "── TOP TARGET TYPES ──────────────────────────────────────────────",
    ]
    for cat, cnt in stats["target_counts"].head(10).items():
        pct = 100 * cnt / stats["target_counts"].sum()
        lines.append(f"  {cat:<35} {cnt:>6,}  ({pct:.1f}%)")
    lines += [
        "",
        "── WHAT ──────────────────────────────────────────────────────────",
    ]
    for cat, cnt in stats["what_counts"].items():
        pct = 100 * cnt / stats["what_counts"].sum()
        lines.append(f"  {cat:<35} {cnt:>6,}  ({pct:.1f}%)")
    lines += [
        "",
        "── TOP 10 SOURCE → TARGET COMBINATIONS ──────────────────────────",
    ]
    for combo, cnt in stats["top_combos"].head(10).items():
        pct = 100 * cnt / stats["criticism_yes"]
        lines.append(f"  {combo:<55} {cnt:>5,}  ({pct:.1f}%)")
    lines += ["", "=" * 70]

    report_path = out / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved → report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse RUN2 standardised output.")
    ap.add_argument("--input",      required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    in_path = Path(args.input)
    out     = Path(args.output_dir)
    fig_dir = out / "figures"
    tbl_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_parquet(in_path) if in_path.suffix == ".parquet" else pd.read_csv(in_path)
    print(f"[load] {len(df):,} rows")

    df["pubtime"] = pd.to_datetime(df["pubtime"], errors="coerce")
    df["quarter"] = df["pubtime"].dt.to_period("Q")

    # ── Criticism subset ──────────────────────────────────────────────────────
    df_crit = df[df["CRITICISM"].map(_flag)].copy()
    df_full = df_crit.dropna(subset=["SOURCE_STD", "TARGET_STD", "WHAT"])
    print(f"[filter] {len(df_crit):,} criticism rows  |  {len(df_full):,} fully labelled")

    stats: dict = {}

    # ── Fig 01 — Overview ──────────────────────────────────────────────────────
    print("\n[fig01] overview")
    stats.update(fig_overview(df, fig_dir))
    stats["full_std"] = len(df_full)

    # ── Fig 02/03 — Source / Target distributions ─────────────────────────────
    print("[fig02] source distribution")
    stats["source_counts"] = fig_source_dist(df_full, fig_dir)
    stats["source_counts"].to_csv(tbl_dir / "source_dist.csv", header=["count"])

    print("[fig03] target distribution")
    stats["target_counts"] = fig_target_dist(df_full, fig_dir)
    stats["target_counts"].to_csv(tbl_dir / "target_dist.csv", header=["count"])

    # ── Fig 04 — WHAT ─────────────────────────────────────────────────────────
    print("[fig04] what distribution")
    stats["what_counts"] = df_full["WHAT"].dropna().value_counts()
    stats["what_counts"].to_csv(tbl_dir / "what_dist.csv", header=["count"])
    fig_what_dist(df_full, fig_dir)

    # ── Fig 05/06 — Temporal evolution ────────────────────────────────────────
    print("[fig05] source quarterly evolution")
    fig_time_evolution(df_full, "SOURCE_STD", "Source", "fig05", fig_dir)

    print("[fig06] target quarterly evolution")
    fig_time_evolution(df_full, "TARGET_STD", "Target", "fig06", fig_dir)

    # ── Fig 07 — Heatmap ──────────────────────────────────────────────────────
    print("[fig07] source × target heatmap")
    matrix = fig_heatmap(df_full, fig_dir)
    matrix.to_csv(tbl_dir / "heatmap_counts.csv")

    # ── Fig 08 — Dept vs Person ───────────────────────────────────────────────
    print("[fig08] department vs person targeting")
    fig_dept_vs_person(df_full, fig_dir)

    # ── Fig 09 — Top combinations ─────────────────────────────────────────────
    print("[fig09] top source→target combinations")
    stats["top_combos"] = fig_top_combinations(df_full, fig_dir)
    stats["top_combos"].to_csv(tbl_dir / "top_combinations.csv", header=["count"])

    # ── Fig 10 — Language breakdown ───────────────────────────────────────────
    print("[fig10] language breakdown")
    fig_language_breakdown(df_full, fig_dir)

    # ── Fig 11 — WHAT by target ───────────────────────────────────────────────
    print("[fig11] what by target")
    fig_what_by_target(df_full, fig_dir)

    # ── Fig 12 — Media ────────────────────────────────────────────────────────
    print("[fig12] media breakdown")
    fig_medium_breakdown(df_crit, fig_dir)

    # ── Report ────────────────────────────────────────────────────────────────
    print("[report]")
    write_report(stats, out)

    print(f"\n[done] all outputs → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
