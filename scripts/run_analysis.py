"""
scripts/run_analysis.py

Final analysis stage of the pipeline. Consumes the run6-merged dataset
(one row per article x target keyword, enriched through run3-run6:
keyword_answer, critic_answer_final, SOURCE_CAT_STD, criticism_target) and
produces, under --output_dir:

  descriptive/      basic counts and target / source / content-type shares
  temporal/         monthly (or --granularity) evolution of the above
  crosstabs/        composition of target_type / content_type by source
                     category (parliamentary party, federal councillor,
                     interest group, civil servant, general public, and
                     federal-department-as-source), plus the E2 party x
                     content-bucket table
  partisan_alignment/   E3 — critic party vs. the targeted department's
                         minister's party at publication time
  crisis/           E4 — per-department(+minister) peak detection
                     (threshold = mean + crisis_k * std) and crisis-vs-
                     routine composition comparison
  analysis_report.md   narrative summary organised by E1-E4

Usage:
    python scripts/run_analysis.py --input <run6_merged/results.parquet> \\
        --output_dir <dir> [--granularity M] [--crisis_k 2.0]

Note on terminology: the run6 pipeline stage names its output column
"criticism_target" but its values (Person/Policy/Both/Unclear) correspond
to the thesis's CONTENT dimension (criticised for who it is vs. what it
does), not the TARGET dimension (which administrative entity is attacked).
This script renames it to `content_type` and reserves "target" for the
`keyword` column (department / admin unit / independent agency / minister).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.analysis_config import (
    COUNCILLOR_PARTY,
    DEPARTMENT_CODES,
    DEPT_LABELS,
    FAR_RIGHT_PARTIES,
    classify_keyword,
    minister_for_canonical_dept,
    parse_source_token,
)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


# =============================================================================
# Generic plotting helpers
# =============================================================================

def _clean_index_for_plot(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = [str(i) if pd.notna(i) else "(n/a)" for i in s.index]
    return s


def bar_chart(pct_series: pd.Series, title: str, path: Path, figsize=(8, 4)) -> None:
    s = _clean_index_for_plot(pct_series).sort_values(ascending=False)
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=figsize)
    s.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("%")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def line_chart(series: pd.Series, title: str, path: Path, ylabel: str = "") -> None:
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    series.plot(ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stacked_area_chart(df_pct: pd.DataFrame, title: str, path: Path) -> None:
    if df_pct.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    df_pct.plot.area(ax=ax, linewidth=0)
    ax.set_title(title)
    ax.set_ylabel("%")
    ax.set_xlabel("")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def multi_line_chart(df_counts: pd.DataFrame, title: str, path: Path) -> None:
    if df_counts.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    df_counts.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Critiques")
    ax.set_xlabel("")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_chart_grouped(df_pct: pd.DataFrame, title: str, path: Path) -> None:
    if df_pct.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(df_pct) + 2))
    df_pct.plot(kind="barh", stacked=True, ax=ax)
    ax.set_xlabel("%")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap(df_pct: pd.DataFrame, title: str, path: Path) -> None:
    if df_pct.empty or df_pct.shape[0] == 0 or df_pct.shape[1] == 0:
        return
    fig, ax = plt.subplots(figsize=(1.3 * df_pct.shape[1] + 2, 0.6 * df_pct.shape[0] + 2))
    vals = df_pct.values.astype(float)
    im = ax.imshow(vals, cmap="viridis", aspect="auto")
    ax.set_xticks(range(df_pct.shape[1]))
    ax.set_xticklabels(df_pct.columns, rotation=45, ha="right")
    ax.set_yticks(range(df_pct.shape[0]))
    ax.set_yticklabels(df_pct.index)
    vmax = np.nanmax(vals) if np.isfinite(vals).any() else 0
    for i in range(df_pct.shape[0]):
        for j in range(df_pct.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                         color="white" if vmax and v > vmax / 2 else "black", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def crosstab_with_heatmap(df: pd.DataFrame, row_col: str, col_col: str,
                            out_csv: Path, out_png: Path, title: str) -> pd.DataFrame:
    ct_pct = pd.crosstab(df[row_col], df[col_col], normalize="index") * 100
    ct_pct.to_csv(out_csv)
    heatmap(ct_pct, title, out_png)
    return ct_pct


def safe_chi2(table: pd.DataFrame) -> tuple[float, float]:
    try:
        if table.shape[0] < 2 or table.shape[1] < 2:
            return float("nan"), float("nan")
        if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
            return float("nan"), float("nan")
        chi2, p, _, _ = chi2_contingency(table)
        return float(chi2), float(p)
    except Exception:
        return float("nan"), float("nan")


def save_value_counts(series: pd.Series, path: Path, label: str) -> pd.DataFrame:
    vc = series.value_counts(dropna=False)
    pct = (100 * vc / vc.sum()).round(2) if vc.sum() else vc
    out = pd.DataFrame({"count": vc, "pct": pct})
    out.index.name = label
    out.to_csv(path)
    return out


def df_block(df, max_rows: int = 25) -> str:
    if df is None or len(df) == 0:
        return "_(no data)_"
    d = df.head(max_rows)
    if isinstance(d, pd.Series):
        d = d.to_frame()
    return "```\n" + d.round(2).to_string() + "\n```"


# =============================================================================
# Data loading & enrichment
# =============================================================================

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    required = ["article_id", "keyword", "keyword_answer", "pubtime"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")
    if "SOURCE_CAT_STD" not in df.columns:
        raise ValueError(
            "Column 'SOURCE_CAT_STD' not found — pass the run6_merged file produced "
            "downstream of scripts/standardize_run5.py, not a raw run5/run6 task output."
        )
    df["pubtime"] = pd.to_datetime(df["pubtime"], errors="coerce")
    return df


def enrich_targets(df: pd.DataFrame) -> pd.DataFrame:
    classified = [classify_keyword(kw, pt) for kw, pt in zip(df["keyword"], df["pubtime"])]
    df = df.assign(
        target_type=[c["target_type"] for c in classified],
        parent_dept=[c["parent_dept"] for c in classified],
        councillor_name=[c["councillor_name"] for c in classified],
    )
    df["content_type"] = df["criticism_target"] if "criticism_target" in df.columns else pd.NA
    return df


def add_period(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    df = df.copy()
    df["period"] = df["pubtime"].dt.to_period(granularity).dt.to_timestamp()
    return df


def explode_sources(df: pd.DataFrame, source_col: str = "SOURCE_CAT_STD") -> pd.DataFrame:
    work = df.copy()
    work["_source_tokens"] = work[source_col].fillna("").astype(str).apply(
        lambda s: [t.strip() for t in s.split("|") if t.strip()] if s.strip() else []
    )
    work = work.explode("_source_tokens")
    work = work[work["_source_tokens"].notna() & (work["_source_tokens"] != "")].copy()
    parsed = work["_source_tokens"].apply(parse_source_token)
    work["source_broad"] = parsed.apply(lambda d: d["broad_category"])
    work["party"] = parsed.apply(lambda d: d["party"])
    work["source_councillor"] = parsed.apply(lambda d: d["councillor_name"])
    work["source_dept"] = parsed.apply(lambda d: d["dept"])
    return work


# =============================================================================
# Section 1 — Descriptive statistics
# =============================================================================

def run_descriptive(df_crit: pd.DataFrame, df_source: pd.DataFrame,
                     n_articles_screened: int, n_keyword_article_pairs: int,
                     out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    overview = pd.Series({
        "articles_screened (mention >=1 tracked entity)": n_articles_screened,
        "article_keyword_pairs_total (YES+NO)": n_keyword_article_pairs,
        "critiques_total (keyword_answer == YES)": len(df_crit),
        "articles_with_at_least_one_critique": df_crit["article_id"].nunique(),
        "source_mentions_total (exploded)": len(df_source),
        "critiques_with_identified_source": df_source["critique_id"].nunique(),
    })
    overview.to_csv(out_dir / "overview.csv", header=["value"])

    target_type_dist = save_value_counts(df_crit["target_type"], out_dir / "target_type_distribution.csv", "target_type")
    dept_dist = save_value_counts(df_crit["parent_dept"], out_dir / "target_department_distribution.csv", "parent_dept")
    entity_dist = save_value_counts(df_crit["keyword"], out_dir / "target_entity_distribution.csv", "keyword")
    source_dist = save_value_counts(df_source["source_broad"], out_dir / "source_broad_distribution.csv", "source_broad")
    party_dist = save_value_counts(
        df_source.loc[df_source["source_broad"] == "Parliamentary", "party"],
        out_dir / "source_parliamentary_party_distribution.csv", "party",
    )
    content_dist = save_value_counts(df_crit["content_type"], out_dir / "content_type_distribution.csv", "content_type")

    bar_chart(target_type_dist["pct"], "Répartition des critiques par type de cible", out_dir / "target_type_distribution.png")
    bar_chart(dept_dist["pct"], "Répartition des critiques par département cible", out_dir / "target_department_distribution.png")
    bar_chart(source_dist["pct"], "Répartition des mentions de source par catégorie", out_dir / "source_broad_distribution.png")
    bar_chart(party_dist["pct"], "Sources parlementaires par parti", out_dir / "source_parliamentary_party_distribution.png")
    bar_chart(content_dist["pct"], "Répartition des critiques par type de contenu", out_dir / "content_type_distribution.png")

    print(f"[descriptive] {len(df_crit):,} critiques | {len(df_source):,} mentions de source")

    return {
        "overview": overview,
        "target_type_dist": target_type_dist,
        "dept_dist": dept_dist,
        "entity_dist": entity_dist,
        "source_dist": source_dist,
        "party_dist": party_dist,
        "content_dist": content_dist,
    }


# =============================================================================
# Section 2 — Temporal evolution
# =============================================================================

def run_temporal(df_crit: pd.DataFrame, df_source: pd.DataFrame, out_dir: Path, granularity: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = df_crit.groupby("period").size().rename("n_critiques").sort_index()
    counts.to_csv(out_dir / "critiques_per_period.csv")
    line_chart(counts, "Évolution du nombre de critiques", out_dir / "critiques_per_period.png", ylabel="Critiques")

    for col, fname, title in [
        ("target_type", "target_type", "Composition par type de cible"),
        ("content_type", "content_type", "Composition par type de contenu"),
    ]:
        ct = pd.crosstab(df_crit["period"], df_crit[col], normalize="index").sort_index() * 100
        ct.to_csv(out_dir / f"{fname}_share_per_period.csv")
        stacked_area_chart(ct, title, out_dir / f"{fname}_share_per_period.png")

    ct_source = pd.crosstab(df_source["period"], df_source["source_broad"], normalize="index").sort_index() * 100
    ct_source.to_csv(out_dir / "source_broad_share_per_period.csv")
    stacked_area_chart(ct_source, "Composition par catégorie de source", out_dir / "source_broad_share_per_period.png")

    dept_counts = (
        df_crit[df_crit["target_type"].isin(["Department", "Minister"]) & df_crit["parent_dept"].notna()]
        .groupby(["period", "parent_dept"]).size().unstack(fill_value=0).sort_index()
    )
    dept_counts.to_csv(out_dir / "critiques_per_period_per_department.csv")
    multi_line_chart(dept_counts, "Critiques par département (département+ministre combinés)",
                      out_dir / "critiques_per_period_per_department.png")

    print(f"[temporal] {len(counts):,} périodes ({granularity})")

    return {
        "n_periods": len(counts),
        "period_min": str(counts.index.min().date()) if len(counts) else None,
        "period_max": str(counts.index.max().date()) if len(counts) else None,
        "peak_period": str(counts.idxmax().date()) if len(counts) else None,
        "peak_value": int(counts.max()) if len(counts) else None,
    }


# =============================================================================
# Section 3 — Cross-tabs (target/content composition by source category)
# =============================================================================

def run_crosstabs(df_source: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    pivots = [
        ("Parliamentary", "party", "parliamentary_by_party"),
        ("Federal Councillor", "source_councillor", "federal_councillor_by_name"),
        ("Federal Councillor", "party", "federal_councillor_by_party"),
        ("Interest Group", None, "interest_group"),
        ("Civil Servant", None, "civil_servant"),
        ("General Public", None, "general_public"),
        ("Federal Department", "source_dept", "federal_department_source"),
    ]

    for broad_value, group_col, fname in pivots:
        subset = df_source[df_source["source_broad"] == broad_value].copy()
        if group_col is not None:
            subset = subset[subset[group_col].notna()]
        if subset.empty:
            summary[fname] = {"n": 0}
            continue
        eff_group_col = group_col
        if eff_group_col is None:
            subset["_group"] = broad_value
            eff_group_col = "_group"
        for target_col in ["target_type", "content_type"]:
            subset_valid = subset[subset[target_col].notna()]
            if subset_valid.empty:
                continue
            out_csv = out_dir / f"{fname}__{target_col}.csv"
            out_png = out_dir / f"{fname}__{target_col}.png"
            crosstab_with_heatmap(
                subset_valid, eff_group_col, target_col, out_csv, out_png,
                title=f"{broad_value} → composition par {target_col}",
            )
        summary[fname] = {"n": len(subset), "n_groups": subset[eff_group_col].nunique()}

    # E2 — ideological asymmetry: entity- vs policy-directed critique, by party,
    # restricted to Parliamentary sources with both party and content_type known.
    parl = df_source[
        (df_source["source_broad"] == "Parliamentary")
        & df_source["party"].notna()
        & df_source["content_type"].notna()
    ].copy()
    if not parl.empty:
        parl["content_bucket"] = np.where(parl["content_type"].isin(["Person", "Both"]),
                                           "Entity-directed", "Policy-directed")
        e2_ct = pd.crosstab(parl["party"], parl["content_bucket"], normalize="index") * 100
        e2_ct.to_csv(out_dir / "e2_party_content_bucket.csv")
        bar_chart_grouped(e2_ct, "E2 — critique entity- vs policy-directed, par parti",
                           out_dir / "e2_party_content_bucket.png")

        is_far_right = parl["party"].isin(FAR_RIGHT_PARTIES)
        far_right_pct = e2_ct.loc[e2_ct.index.isin(FAR_RIGHT_PARTIES), "Entity-directed"].mean() \
            if e2_ct.index.isin(FAR_RIGHT_PARTIES).any() else None
        other_pct = e2_ct.loc[~e2_ct.index.isin(FAR_RIGHT_PARTIES), "Entity-directed"].mean() \
            if (~e2_ct.index.isin(FAR_RIGHT_PARTIES)).any() else None
        chi2, p = safe_chi2(pd.crosstab(is_far_right, parl["content_bucket"]))
        summary["e2"] = {
            "table": e2_ct, "n": len(parl),
            "far_right_entity_pct": far_right_pct, "other_entity_pct": other_pct,
            "chi2": chi2, "p": p,
        }
    else:
        summary["e2"] = {"n": 0}

    print(f"[crosstabs] {len(pivots)} pivots de source écrits dans {out_dir}")
    return summary


# =============================================================================
# Section 4 — E3 partisan alignment
# =============================================================================

def run_partisan_alignment(df_source: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = df_source[
        (df_source["source_broad"] == "Parliamentary")
        & df_source["party"].notna()
        & df_source["target_type"].isin(["Department", "Minister"])
        & df_source["parent_dept"].notna()
    ].copy()

    if sub.empty:
        return {"n_parliamentary_critiques_with_known_minister": 0}

    sub["minister_name"] = [
        minister_for_canonical_dept(d, pt) for d, pt in zip(sub["parent_dept"], sub["pubtime"])
    ]
    sub["minister_party"] = sub["minister_name"].map(COUNCILLOR_PARTY)
    sub = sub[sub["minister_party"].notna()]
    if sub.empty:
        return {"n_parliamentary_critiques_with_known_minister": 0}

    sub["alignment"] = np.where(sub["party"] == sub["minister_party"], "aligned", "adversarial")

    overall = sub["alignment"].value_counts()
    overall_pct = (100 * overall / overall.sum()).round(2)
    pd.DataFrame({"count": overall, "pct": overall_pct}).to_csv(out_dir / "alignment_overall.csv")

    by_dept = pd.crosstab(sub["parent_dept"], sub["alignment"], normalize="index") * 100
    by_dept.to_csv(out_dir / "alignment_by_department.csv")
    bar_chart_grouped(by_dept, "Alignement partisan critique↔ministre, par département",
                       out_dir / "alignment_by_department.png")

    by_party = pd.crosstab(sub["party"], sub["alignment"], normalize="index") * 100
    by_party.to_csv(out_dir / "alignment_by_party.csv")

    chi2, p = safe_chi2(pd.crosstab(sub["parent_dept"], sub["alignment"]))

    print(f"[alignment] {len(sub):,} mentions parlementaires rattachées à un ministre connu")

    return {
        "n_parliamentary_critiques_with_known_minister": len(sub),
        "pct_adversarial": float(overall_pct.get("adversarial", 0.0)),
        "pct_aligned": float(overall_pct.get("aligned", 0.0)),
        "chi2_by_department": chi2,
        "chi2_p_by_department": p,
        "by_dept_table": by_dept,
    }


# =============================================================================
# Section 5 — E4 crisis / peak detection
# =============================================================================

def run_crisis_analysis(df_crit: pd.DataFrame, df_source: pd.DataFrame,
                         out_dir: Path, granularity: str, k: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    if df_crit["period"].isna().all():
        return {"n_crisis_periods_total": 0, "n_departments_with_crisis_months": 0, "k": k}

    all_periods = pd.period_range(df_crit["period"].min(), df_crit["period"].max(), freq=granularity).to_timestamp()

    combined = df_crit[df_crit["target_type"].isin(["Department", "Minister"]) & df_crit["parent_dept"].notna()].copy()
    counts = combined.groupby(["parent_dept", "period"]).size().rename("n").reset_index()

    full_index = pd.MultiIndex.from_product([DEPARTMENT_CODES, all_periods], names=["parent_dept", "period"])
    counts = counts.set_index(["parent_dept", "period"]).reindex(full_index, fill_value=0)["n"].reset_index()

    thresholds = counts.groupby("parent_dept")["n"].agg(mean="mean", std="std").reset_index()
    thresholds["threshold"] = thresholds["mean"] + k * thresholds["std"].fillna(0.0)
    counts = counts.merge(thresholds[["parent_dept", "threshold"]], on="parent_dept", how="left")
    counts["is_crisis"] = counts["n"] > counts["threshold"]
    counts.to_csv(out_dir / "crisis_monthly_counts.csv", index=False)

    crisis_periods = counts.loc[counts["is_crisis"], ["parent_dept", "period", "n", "threshold"]] \
        .sort_values(["parent_dept", "period"])
    crisis_periods.to_csv(out_dir / "crisis_periods.csv", index=False)

    for dept in DEPARTMENT_CODES:
        sub_d = counts[counts["parent_dept"] == dept].sort_values("period")
        if sub_d["n"].sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(sub_d["period"], sub_d["n"], label="Critiques", color="steelblue")
        ax.axhline(sub_d["threshold"].iloc[0], color="red", linestyle="--", label=f"Seuil (μ+{k}σ)")
        crisis_pts = sub_d[sub_d["is_crisis"]]
        if not crisis_pts.empty:
            ax.scatter(crisis_pts["period"], crisis_pts["n"], color="red", zorder=5, label="Pic de crise")
        ax.set_title(f"{DEPT_LABELS.get(dept, dept)} — critiques département+ministre")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"timeseries_{dept.replace('/', '-')}.png", dpi=150)
        plt.close(fig)

    combined = combined.merge(counts[["parent_dept", "period", "is_crisis"]], on=["parent_dept", "period"], how="left")
    crisis_critique_ids = set(combined.loc[combined["is_crisis"].fillna(False), "critique_id"])

    df_source_tagged = df_source[
        df_source["target_type"].isin(["Department", "Minister"]) & df_source["parent_dept"].notna()
    ].copy()
    df_source_tagged["regime"] = np.where(df_source_tagged["critique_id"].isin(crisis_critique_ids), "crisis", "routine")

    source_ct_counts = pd.crosstab(df_source_tagged["regime"], df_source_tagged["source_broad"])
    source_ct_pct = pd.crosstab(df_source_tagged["regime"], df_source_tagged["source_broad"], normalize="index") * 100
    source_ct_pct.to_csv(out_dir / "crisis_vs_routine_source_composition.csv")
    chi2_source, p_source = safe_chi2(source_ct_counts)

    content_ct_counts = pd.crosstab(
        pd.Series(np.where(combined["is_crisis"].fillna(False), "crisis", "routine"), name="regime"),
        combined["content_type"],
    )
    content_ct_pct = (100 * content_ct_counts.div(content_ct_counts.sum(axis=1), axis=0))
    content_ct_pct.to_csv(out_dir / "crisis_vs_routine_content_composition.csv")
    chi2_content, p_content = safe_chi2(content_ct_counts)

    dept_targets = df_crit[df_crit["parent_dept"].notna()].merge(
        counts[["parent_dept", "period", "is_crisis"]], on=["parent_dept", "period"], how="left"
    )
    dept_targets["regime"] = np.where(dept_targets["is_crisis"].fillna(False), "crisis", "routine")
    target_type_ct_pct = pd.crosstab(dept_targets["regime"], dept_targets["target_type"], normalize="index") * 100
    target_type_ct_pct.to_csv(out_dir / "crisis_vs_routine_target_type_composition.csv")

    global_crisis_periods = counts.groupby("period")["is_crisis"].any().rename("any_dept_in_crisis")
    df_crit_global = df_crit.merge(global_crisis_periods, on="period", how="left")
    df_crit_global["any_dept_in_crisis"] = df_crit_global["any_dept_in_crisis"].fillna(False)
    indep_ct_pct = pd.crosstab(
        pd.Series(np.where(df_crit_global["any_dept_in_crisis"], "crisis (any dept.)", "routine"), name="regime"),
        df_crit_global["target_type"], normalize="index",
    ) * 100
    indep_ct_pct.to_csv(out_dir / "crisis_vs_routine_global_target_type_composition.csv")

    diversity = df_source_tagged.groupby("regime")["source_broad"].nunique()

    print(f"[crisis] {int(counts['is_crisis'].sum())} période-département en crise "
          f"sur {counts['parent_dept'].nunique()} départements")

    return {
        "k": k,
        "n_departments_with_crisis_months": int(counts.loc[counts["is_crisis"], "parent_dept"].nunique()),
        "n_crisis_periods_total": int(counts["is_crisis"].sum()),
        "source_diversity_crisis": int(diversity.get("crisis", 0)),
        "source_diversity_routine": int(diversity.get("routine", 0)),
        "chi2_source": chi2_source, "chi2_source_p": p_source,
        "chi2_content": chi2_content, "chi2_content_p": p_content,
        "source_ct_pct": source_ct_pct,
        "content_ct_pct": content_ct_pct,
        "target_type_ct_pct": target_type_ct_pct,
        "indep_ct_pct": indep_ct_pct,
        "crisis_periods_table": crisis_periods,
    }


# =============================================================================
# Report writer
# =============================================================================

def write_report(path: Path, args: argparse.Namespace, sections: dict) -> None:
    desc = sections["descriptive"]
    temporal = sections["temporal"]
    crosstabs = sections["crosstabs"]
    alignment = sections["alignment"]
    crisis = sections["crisis"]

    lines: list[str] = []
    lines.append("# Analyse de la pression médiatique — rapport automatique\n")
    lines.append(f"- Fichier source : `{args.input}`")
    lines.append(f"- Granularité temporelle : `{args.granularity}`")
    lines.append(f"- Seuil de crise : moyenne + {args.crisis_k}×écart-type (calculé par département)")
    lines.append("")

    lines.append("## Statistiques descriptives\n")
    lines.append(df_block(desc["overview"].to_frame("value")))
    lines.append("\n**Répartition par type de cible :**\n")
    lines.append(df_block(desc["target_type_dist"]))
    lines.append("\n**Répartition par département cible (cumul département+ministre+unités) :**\n")
    lines.append(df_block(desc["dept_dist"]))
    lines.append("\n**Répartition par catégorie de source (mentions, peut dépasser 1 par critique) :**\n")
    lines.append(df_block(desc["source_dist"]))
    lines.append("\n**Répartition par type de contenu :**\n")
    lines.append(df_block(desc["content_dist"]))
    lines.append(
        f"\nPériode couverte : {temporal.get('period_min')} → {temporal.get('period_max')} "
        f"({temporal.get('n_periods')} périodes). Pic global : {temporal.get('peak_period')} "
        f"({temporal.get('peak_value')} critiques).\n"
    )

    lines.append("## E1 — Dominance parlementaire du sourcing\n")
    sd = desc["source_dist"]
    if "Parliamentary" in sd.index:
        parl_pct = sd.loc["Parliamentary", "pct"]
        top_cat = sd["pct"].idxmax()
        lines.append(f"- Les sources **Parliamentary** représentent **{parl_pct:.1f}%** des "
                      f"{int(sd['count'].sum()):,} mentions de source identifiées.")
        lines.append(f"- Catégorie la plus fréquente : **{top_cat}** "
                      f"({sd.loc[top_cat, 'pct']:.1f}%) — "
                      f"{'confirme' if top_cat == 'Parliamentary' else 'ne confirme pas'} E1 telle que formulée.")
    else:
        lines.append("_Aucune mention 'Parliamentary' trouvée dans les données._")
    lines.append("")

    lines.append("## E2 — Asymétrie idéologique du type de critique\n")
    e2 = crosstabs.get("e2", {})
    if e2.get("n", 0) > 0:
        lines.append(f"- {e2['n']:,} mentions parlementaires avec parti et type de contenu connus.")
        if e2.get("far_right_entity_pct") is not None:
            lines.append(f"- Part *entity-directed* pour les partis classés extrême-droite "
                          f"({', '.join(sorted(FAR_RIGHT_PARTIES))}) : **{e2['far_right_entity_pct']:.1f}%**")
        if e2.get("other_entity_pct") is not None:
            lines.append(f"- Part *entity-directed* pour les autres partis : **{e2['other_entity_pct']:.1f}%**")
        if not np.isnan(e2.get("p", np.nan)):
            lines.append(f"- Test du χ² (extrême-droite vs autres × entity/policy) : "
                          f"χ²={e2['chi2']:.2f}, p={e2['p']:.4f}")
        lines.append("\n" + df_block(e2["table"]))
    else:
        lines.append("_Pas assez de données parlementaires avec parti + type de contenu connus._")
    lines.append("")

    lines.append("## E3 — Alignement partisan critique↔ministre\n")
    if alignment.get("n_parliamentary_critiques_with_known_minister", 0) > 0:
        lines.append(f"- {alignment['n_parliamentary_critiques_with_known_minister']:,} mentions "
                      f"parlementaires rattachées à un département dont le parti du ministre en poste est connu.")
        lines.append(f"- Part adversariale (parti du critique ≠ parti du ministre en charge) : "
                      f"**{alignment['pct_adversarial']:.1f}%**")
        lines.append(f"- Part alignée (même parti) : **{alignment['pct_aligned']:.1f}%**")
        if not np.isnan(alignment.get("chi2_p_by_department", np.nan)):
            lines.append(f"- Test du χ² (département × alignement) : "
                          f"χ²={alignment['chi2_by_department']:.2f}, p={alignment['chi2_p_by_department']:.4f}")
        lines.append("\n**Part adversariale par département (%) :**\n")
        lines.append(df_block(alignment["by_dept_table"]))
    else:
        lines.append("_Pas assez de données pour tester l'alignement partisan._")
    lines.append("")

    lines.append("## E4 — Composition aux pics de pression\n")
    if crisis.get("n_crisis_periods_total", 0) > 0:
        lines.append(f"- {crisis['n_crisis_periods_total']} période(s)-département en régime de crise "
                      f"(seuil μ+{crisis['k']}σ), sur {crisis['n_departments_with_crisis_months']} département(s).")
        lines.append(f"- Diversité des catégories de source observées — crise : "
                      f"{crisis['source_diversity_crisis']}, routine : {crisis['source_diversity_routine']}.")
        if not np.isnan(crisis.get("chi2_source_p", np.nan)):
            lines.append(f"- χ² composition des sources (crise vs routine) : "
                          f"χ²={crisis['chi2_source']:.2f}, p={crisis['chi2_source_p']:.4f}")
        if not np.isnan(crisis.get("chi2_content_p", np.nan)):
            lines.append(f"- χ² composition du contenu (crise vs routine) : "
                          f"χ²={crisis['chi2_content']:.2f}, p={crisis['chi2_content_p']:.4f}")
        lines.append("\n**Composition des sources, crise vs routine (%) :**\n")
        lines.append(df_block(crisis["source_ct_pct"]))
        lines.append("\n**Composition du contenu, crise vs routine (%) :**\n")
        lines.append(df_block(crisis["content_ct_pct"]))
        lines.append("\n**Composition du type de cible (incl. unités admin.), crise vs routine (%) :**\n")
        lines.append(df_block(crisis["target_type_ct_pct"]))
        lines.append("\n**Cibles, mois où au moins un département est en crise vs routine globale (%) :**\n")
        lines.append(df_block(crisis["indep_ct_pct"]))
        lines.append("\n**Périodes de crise détectées (par département) :**\n")
        lines.append(df_block(crisis["crisis_periods_table"], max_rows=60))
    else:
        lines.append("_Aucune période de crise détectée avec ce seuil._")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Final analysis stage: descriptive stats, temporal evolution, "
                                              "cross-tabs, partisan alignment, and crisis detection.")
    ap.add_argument("--input", required=True,
                     help="run6_merged results file (.parquet/.csv); must contain SOURCE_CAT_STD and criticism_target")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--granularity", default="M", help="pandas period frequency for temporal aggregation (M=month, W=week)")
    ap.add_argument("--crisis_k", type=float, default=2.0, help="crisis threshold = mean + k*std, per department")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_data(in_path)
    print(f"[load] {len(df_all):,} rows from {in_path}")

    n_articles_screened = df_all["article_id"].nunique()
    n_keyword_article_pairs = len(df_all)

    df_crit = df_all[df_all["keyword_answer"].astype(str).str.upper() == "YES"].copy()
    df_crit = df_crit.reset_index(drop=True)
    df_crit["critique_id"] = df_crit.index
    print(f"[filter] {len(df_crit):,} critiques (keyword_answer == YES)")

    df_crit = enrich_targets(df_crit)
    df_crit = add_period(df_crit, args.granularity)
    df_source = explode_sources(df_crit)
    print(f"[explode] {len(df_source):,} source mentions across {df_crit['critique_id'].nunique():,} critiques")

    sections: dict = {}
    sections["descriptive"] = run_descriptive(
        df_crit, df_source, n_articles_screened, n_keyword_article_pairs, out_dir / "descriptive"
    )
    sections["temporal"] = run_temporal(df_crit, df_source, out_dir / "temporal", args.granularity)
    sections["crosstabs"] = run_crosstabs(df_source, out_dir / "crosstabs")
    sections["alignment"] = run_partisan_alignment(df_source, out_dir / "partisan_alignment")
    sections["crisis"] = run_crisis_analysis(df_crit, df_source, out_dir / "crisis", args.granularity, args.crisis_k)

    report_path = out_dir / "analysis_report.md"
    write_report(report_path, args, sections)
    print(f"[done] Report written to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
