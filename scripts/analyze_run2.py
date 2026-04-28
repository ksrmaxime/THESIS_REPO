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

# ── Journal grouping ─────────────────────────────────────────────────────────
_JOURNAL_MAP: dict[str, str] = {
    # NZZ
    "nzz online":               "NZZ",
    "nzz am sonntag":           "NZZ",
    "neue zürcher zeitung":     "NZZ",
    "nzz.ch":                   "NZZ",
    # Le Temps
    "le temps":                 "Le Temps",
    "letemps.ch":               "Le Temps",
    # Tages-Anzeiger
    "tages-anzeiger":           "Tages-Anzeiger",
    "tagesanzeiger.ch":         "Tages-Anzeiger",
    "newsnet / tages-anzeiger": "Tages-Anzeiger",
    # 24 heures
    "newsnet / 24 heures":      "24 heures",
    "24 heures":                "24 heures",
    "24heures.ch":              "24 heures",
    # 20 Minuten
    "20 minuten online":        "20 Minuten",
    # 20 Minutes
    "20 minutes":               "20 Minutes",
    "20 minutes online":        "20 Minutes",
}


def _normalize_medium(val) -> str:
    """Return grouped journal name, falling back to the original value."""
    if pd.isna(val):
        return "Unknown"
    return _JOURNAL_MAP.get(str(val).strip().lower(), str(val).strip())


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


def _extract_parties(val) -> list[str]:
    """Return list of party names found in a SOURCE_STD cell."""
    if pd.isna(val):
        return []
    return re.findall(r'Parliamentary \(([^)]+)\)', str(val))


def _to_dept_label(val) -> str | None:
    """Normalise a FD or CF standardised value to its department abbreviation."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.startswith("FD (") or s.startswith("CF ("):
        return _dept_from_std(s)
    return None


def _fd_dept_label(val) -> str | None:
    """Extract dept abbreviation ONLY from FD entries (not CF)."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    return _dept_from_std(s) if s.startswith("FD (") else None


def _cf_name(val) -> str | None:
    """Extract councillor name from CF (Name — DEPT), stripping the dept part."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    m = re.match(r'CF \(([^—]+?)(?:\s*—.*?)?\)', s)
    return m.group(1).strip() if m else None


def _draw_heatmap(ax, matrix: pd.DataFrame, fmt: str, cmap) -> None:
    """Draw annotated heatmap on ax."""
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    vmax = matrix.values.max()
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            v = matrix.values[i, j]
            if v == 0:
                continue
            label = fmt.format(v)
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=6, color="white" if v > vmax * 0.6 else "black")


def fig_party_vs_target(df_full: pd.DataFrame, fig_dir: Path, tbl_dir: Path) -> None:
    """Fig 13 — Which targets (dept/CF) are criticised by each parliamentary party."""
    cmap = LinearSegmentedColormap.from_list("orange_scale", ["#FFF7ED", "#C2410C"])

    rows = []
    for _, row in df_full[["SOURCE_STD", "TARGET_STD"]].dropna().iterrows():
        parties = _extract_parties(row["SOURCE_STD"])
        if not parties:
            continue
        # Take the first TARGET entity for 1-to-1 pairing
        tgt_raw = str(row["TARGET_STD"]).split("|")[0].strip()
        for party in parties:
            rows.append({"party": party, "target_raw": tgt_raw})

    if not rows:
        print("  [skip] no parliamentary source rows found")
        return

    tmp = pd.DataFrame(rows)

    # ── Sub-fig A: party × target category ───────────────────────────────────
    top_tgt = tmp["target_raw"].value_counts().head(12).index
    sub_cat = tmp[tmp["target_raw"].isin(top_tgt)]
    matrix_cat = sub_cat.groupby(["party", "target_raw"]).size().unstack(fill_value=0)
    matrix_cat_pct = matrix_cat.div(matrix_cat.sum(axis=1), axis=0) * 100

    # ── Sub-fig B: party × department (normalise CF+FD → dept) ──────────────
    tmp["dept"] = tmp["target_raw"].map(_to_dept_label)
    sub_dept = tmp.dropna(subset=["dept"])
    matrix_dept = sub_dept.groupby(["party", "dept"]).size().unstack(fill_value=0)
    matrix_dept_pct = matrix_dept.div(matrix_dept.sum(axis=1), axis=0) * 100

    # ── Sub-fig C: party × FD-or-CF preference ───────────────────────────────
    def _fd_or_cf(val: str) -> str | None:
        if val.startswith("FD ("):
            return "Department (FD)"
        if val.startswith("CF ("):
            return "Person (CF)"
        return None

    tmp["fd_cf"] = tmp["target_raw"].map(_fd_or_cf)
    sub_fdcf = tmp.dropna(subset=["fd_cf"])
    if not sub_fdcf.empty:
        matrix_fdcf = (
            sub_fdcf.groupby(["party", "fd_cf"]).size()
            .unstack(fill_value=0)
            .reindex(columns=["Department (FD)", "Person (CF)"], fill_value=0)
        )
        matrix_fdcf_pct = matrix_fdcf.div(matrix_fdcf.sum(axis=1), axis=0) * 100
        # Sort parties by preference for CF over FD (descending % CF)
        if "Person (CF)" in matrix_fdcf_pct.columns:
            matrix_fdcf_pct = matrix_fdcf_pct.sort_values("Person (CF)")
            matrix_fdcf     = matrix_fdcf.loc[matrix_fdcf_pct.index]

    # ── Sub-fig D: dept × party, split FD vs CF attacks ─────────────────────
    # For each department, show how many attacks came from each party
    # and whether they targeted the department (FD) or its councillor (CF).
    tmp["dept"]  = tmp["target_raw"].map(_to_dept_label)
    tmp["fd_cf"] = tmp["target_raw"].map(_fd_or_cf)
    sub_split = tmp.dropna(subset=["dept", "fd_cf"])
    if not sub_split.empty:
        # FD attacks: source=party, target=dept (via FD entry)
        fd_rows = sub_split[sub_split["fd_cf"] == "Department (FD)"]
        matrix_dept_fd = (fd_rows.groupby(["dept", "party"]).size()
                          .unstack(fill_value=0) if not fd_rows.empty
                          else pd.DataFrame())
        # CF attacks: source=party, target=dept (via CF entry, dept extracted)
        cf_rows = sub_split[sub_split["fd_cf"] == "Person (CF)"]
        matrix_dept_cf = (cf_rows.groupby(["dept", "party"]).size()
                          .unstack(fill_value=0) if not cf_rows.empty
                          else pd.DataFrame())
        # Align columns across both matrices
        all_parties = sorted(
            set(matrix_dept_fd.columns.tolist() + matrix_dept_cf.columns.tolist()))
        all_depts = sorted(
            set(matrix_dept_fd.index.tolist() + matrix_dept_cf.index.tolist()))
        for m in [matrix_dept_fd, matrix_dept_cf]:
            for p in all_parties:
                if p not in m.columns:
                    m[p] = 0
            for d in all_depts:
                if d not in m.index:
                    m.loc[d] = 0
        matrix_dept_fd = matrix_dept_fd.reindex(index=all_depts, columns=all_parties, fill_value=0)
        matrix_dept_cf = matrix_dept_cf.reindex(index=all_depts, columns=all_parties, fill_value=0)

    # ── Sub-fig E: councillor × party ────────────────────────────────────────
    tmp["cf_name"] = tmp["target_raw"].apply(
        lambda v: _cf_name(v) if str(v).startswith("CF (") else None)
    sub_cf = tmp.dropna(subset=["cf_name"])
    matrix_cf_party = (sub_cf.groupby(["cf_name", "party"]).size()
                       .unstack(fill_value=0) if not sub_cf.empty
                       else pd.DataFrame())

    # ── Build figure (5 rows × 2 cols) ───────────────────────────────────────
    fig, axes = plt.subplots(5, 2, figsize=(22, max(20, len(matrix_cat) * 1.2 + 16)))

    # Row 0 — party × target category
    _draw_heatmap(axes[0][0], matrix_cat,     "{:.0f}",  cmap)
    axes[0][0].set_title("Party × Target category  (counts)")
    axes[0][0].set_xlabel("Target"); axes[0][0].set_ylabel("Party")
    _draw_heatmap(axes[0][1], matrix_cat_pct, "{:.0f}%", cmap)
    axes[0][1].set_title("Party × Target category  (% of party's criticisms)")
    axes[0][1].set_xlabel("Target")

    # Row 1 — party × department (all attacks merged)
    if not matrix_dept.empty:
        _draw_heatmap(axes[1][0], matrix_dept,     "{:.0f}",  cmap)
        axes[1][0].set_title("Party × Department targeted  (counts, FD+CF merged)")
        axes[1][0].set_xlabel("Department"); axes[1][0].set_ylabel("Party")
        _draw_heatmap(axes[1][1], matrix_dept_pct, "{:.0f}%", cmap)
        axes[1][1].set_title("Party × Department targeted  (% of party's criticisms)")
        axes[1][1].set_xlabel("Department")
    else:
        axes[1][0].axis("off"); axes[1][1].axis("off")

    # Row 2 — party FD/CF preference
    if not sub_fdcf.empty:
        colors = [BLUE, RED]
        matrix_fdcf.plot(kind="barh", ax=axes[2][0], color=colors, stacked=False)
        axes[2][0].set_title("Department vs Person — by party  (counts)")
        axes[2][0].set_xlabel("Count"); axes[2][0].set_ylabel("Party")
        axes[2][0].legend(fontsize=8)
        matrix_fdcf_pct.plot(kind="barh", ax=axes[2][1], color=colors, stacked=True)
        axes[2][1].set_title("Department vs Person — by party  (% stacked)")
        axes[2][1].set_xlabel("% targeting federal executive")
        axes[2][1].xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        axes[2][1].legend(fontsize=8)
        for lbl in axes[2][1].get_yticklabels():
            lbl.set_fontsize(8)
    else:
        axes[2][0].axis("off"); axes[2][1].axis("off")

    # Row 3 — dept × party split: FD attacks (left) and CF attacks (right)
    if not sub_split.empty and not matrix_dept_fd.empty:
        _draw_heatmap(axes[3][0], matrix_dept_fd, "{:.0f}", cmap)
        axes[3][0].set_title("Dept × Party — attacks on DEPARTMENT (FD)")
        axes[3][0].set_xlabel("Party"); axes[3][0].set_ylabel("Department")
    else:
        axes[3][0].axis("off")
    if not sub_split.empty and not matrix_dept_cf.empty:
        _draw_heatmap(axes[3][1], matrix_dept_cf, "{:.0f}", cmap)
        axes[3][1].set_title("Dept × Party — attacks on COUNCILLOR (CF)")
        axes[3][1].set_xlabel("Party"); axes[3][1].set_ylabel("Department")
    else:
        axes[3][1].axis("off")

    # Row 4 — councillor × party
    if not matrix_cf_party.empty:
        _draw_heatmap(axes[4][0], matrix_cf_party, "{:.0f}", cmap)
        axes[4][0].set_title("Councillor × Party — attacks on PERSON (CF)")
        axes[4][0].set_xlabel("Party"); axes[4][0].set_ylabel("Councillor")
        matrix_cf_pct = matrix_cf_party.div(matrix_cf_party.sum(axis=1), axis=0) * 100
        _draw_heatmap(axes[4][1], matrix_cf_pct, "{:.0f}%", cmap)
        axes[4][1].set_title("Councillor × Party  (% of attacks on councillor)")
        axes[4][1].set_xlabel("Party")
    else:
        axes[4][0].axis("off"); axes[4][1].axis("off")

    _save(fig, fig_dir / "fig13_party_vs_target.png",
          "Parliamentary Parties — Who Do They Criticise?")

    matrix_cat.to_csv(tbl_dir / "party_vs_target_counts.csv")
    matrix_cat_pct.round(1).to_csv(tbl_dir / "party_vs_target_pct.csv")
    if not matrix_dept.empty:
        matrix_dept.to_csv(tbl_dir / "party_vs_dept_counts.csv")
    if not sub_fdcf.empty:
        matrix_fdcf.to_csv(tbl_dir / "party_vs_fd_cf_counts.csv")
        matrix_fdcf_pct.round(1).to_csv(tbl_dir / "party_vs_fd_cf_pct.csv")
    if not sub_split.empty:
        if not matrix_dept_fd.empty:
            matrix_dept_fd.to_csv(tbl_dir / "dept_party_fd_attacks.csv")
        if not matrix_dept_cf.empty:
            matrix_dept_cf.to_csv(tbl_dir / "dept_party_cf_attacks.csv")
    if not matrix_cf_party.empty:
        matrix_cf_party.to_csv(tbl_dir / "cf_party_attacks.csv")


def fig_internal_federal(df_full: pd.DataFrame, fig_dir: Path, tbl_dir: Path) -> None:
    """Fig 14 — CF/FD criticising another CF/FD (intra-federal dynamics).

    Two levels:
      A) Department-level: normalise both SOURCE and TARGET to dept abbreviation
      B) Person-level: councillor name → councillor name (CF only)
    """
    cmap = LinearSegmentedColormap.from_list("red_scale", ["#FFF1F2", "#991B1B"])

    # ── Filter: SOURCE and TARGET both CF or FD ───────────────────────────────
    def _is_federal(val) -> bool:
        if pd.isna(val):
            return False
        first = str(val).split("|")[0].strip()
        return first.startswith("CF (") or first.startswith("FD (")

    mask = df_full["SOURCE_STD"].map(_is_federal) & df_full["TARGET_STD"].map(_is_federal)
    sub = df_full[mask].copy()

    if sub.empty:
        print("  [skip] no intra-federal rows found")
        return

    src_first = sub["SOURCE_STD"].str.split("|").str[0].str.strip()
    tgt_first = sub["TARGET_STD"].str.split("|").str[0].str.strip()

    def _build(src_series, tgt_series) -> pd.DataFrame:
        df_ = pd.DataFrame({"src": src_series, "tgt": tgt_series}).dropna()
        if df_.empty:
            return pd.DataFrame()
        return df_.groupby(["src", "tgt"]).size().unstack(fill_value=0)

    def _pct(m: pd.DataFrame) -> pd.DataFrame:
        return m.div(m.sum(axis=1), axis=0) * 100

    def _row(ax_counts, ax_pct, matrix, title_counts, title_pct,
             xlabel, ylabel) -> None:
        if matrix.empty:
            for ax, title in [(ax_counts, title_counts), (ax_pct, title_pct)]:
                ax.set_title(title)
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color=GRAY)
                ax.set_xticks([]); ax.set_yticks([])
            return
        _draw_heatmap(ax_counts, matrix,       "{:.0f}",  cmap)
        ax_counts.set_title(title_counts)
        ax_counts.set_xlabel(xlabel)
        ax_counts.set_ylabel(ylabel)
        _draw_heatmap(ax_pct,   _pct(matrix),  "{:.0f}%", cmap)
        ax_pct.set_title(title_pct)
        ax_pct.set_xlabel(xlabel)

    # ── A) FD → FD ────────────────────────────────────────────────────────────
    fd_fd = _build(
        src_first.map(_fd_dept_label),
        tgt_first.map(_fd_dept_label),
    )

    # ── B) CF → CF ────────────────────────────────────────────────────────────
    cf_mask = src_first.str.startswith("CF (") & tgt_first.str.startswith("CF (")
    cf_cf = _build(
        src_first[cf_mask].map(_cf_name),
        tgt_first[cf_mask].map(_cf_name),
    )

    # ── C) CF → FD  (councillor criticises a department) ─────────────────────
    cf_fd_mask = src_first.str.startswith("CF (") & tgt_first.str.startswith("FD (")
    cf_fd = _build(
        src_first[cf_fd_mask].map(_cf_name),
        tgt_first[cf_fd_mask].map(_fd_dept_label),
    )

    # ── D) FD → CF  (department criticises a councillor) ─────────────────────
    fd_cf_mask = src_first.str.startswith("FD (") & tgt_first.str.startswith("CF (")
    fd_cf = _build(
        src_first[fd_cf_mask].map(_fd_dept_label),
        tgt_first[fd_cf_mask].map(_cf_name),
    )

    fig, axes = plt.subplots(4, 2, figsize=(20, 32))

    _row(axes[0][0], axes[0][1], fd_fd,
         "FD → FD  (counts)",     "FD → FD  (% of source)",
         "Target Department",     "Source Department")

    _row(axes[1][0], axes[1][1], cf_cf,
         "CF → CF  (counts)",     "CF → CF  (% of source)",
         "Target Councillor",     "Source Councillor")

    _row(axes[2][0], axes[2][1], cf_fd,
         "CF → FD  (counts)",     "CF → FD  (% of source)",
         "Target Department",     "Source Councillor")

    _row(axes[3][0], axes[3][1], fd_cf,
         "FD → CF  (counts)",     "FD → CF  (% of source)",
         "Target Councillor",     "Source Department")

    _save(fig, fig_dir / "fig14_internal_federal.png",
          "Intra-Federal Criticism  (all CF/FD combinations)")

    for matrix, name in [
        (fd_fd, "fd_fd"), (cf_cf, "cf_cf"),
        (cf_fd, "cf_fd"), (fd_cf, "fd_cf"),
    ]:
        if not matrix.empty:
            matrix.to_csv(tbl_dir / f"internal_{name}_counts.csv")


def fig_medium_breakdown(df_crit: pd.DataFrame, out: Path) -> None:
    """Fig 12 — Top newspapers by criticism volume (grouped journals)."""
    top_media = df_crit["medium_group"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(10, max(4, len(top_media) * 0.45)))
    _bar_h(ax, top_media, pct_total=len(df_crit))
    ax.set_title("Top 15 Media Outlets by Criticism Volume  (grouped)")
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

    df["pubtime"]      = pd.to_datetime(df["pubtime"], errors="coerce")
    df["quarter"]      = df["pubtime"].dt.to_period("Q")
    df["medium_group"] = df["medium_name"].map(_normalize_medium)

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

    # ── Fig 13 — Parliamentary parties × targets ──────────────────────────────
    print("[fig13] parliamentary parties vs targets")
    fig_party_vs_target(df_full, fig_dir, tbl_dir)

    # ── Fig 14 — Intra-federal criticism ──────────────────────────────────────
    print("[fig14] intra-federal CF/FD → CF/FD")
    fig_internal_federal(df_full, fig_dir, tbl_dir)

    # ── Report ────────────────────────────────────────────────────────────────
    print("[report]")
    write_report(stats, out)

    print(f"\n[done] all outputs → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
