#!/usr/bin/env python3
"""
dg_compare_two_systems.py

This is a paired bar plot intended to be used for compairing MMPBSA/MMGBSA decomposition data
of two systems where the protein is the same but the ligand may vary. Since it also computes
a statistical analysis (using Welch's t-test), the input dataframes must have data from multiple (3 or more)
replica simulations of each system. It groups residues together, so one can see the contributions to
the binding free energy change per residue, under the influence of different ligands. 

Features:
- Residues from two systems are aligned; if a residue is missing in one system,
  we still include it and draw an "x" at y=0 for that system's bar instead of
  pretending it's truly 0 kcal/mol.
- Bars are colored per system with error bars (SEM).
- Residue renaming via JSON (flexible matching of residue names like R:A:ASN:387 -> Asn387).
- Residue order is taken from one CSV (with --order-by A or B). Residues unique
  to the other CSV are appended in that system’s original order.
- Numeric value labels beyond error bars with optional |value| threshold.
- Welch-style significance test between systems A and B using across-replica
  SEMs from the combined tables, with an asterisk if p <= --p-threshold.
- Optional: write a stats summary CSV per residue including p-values.

Assumptions:
- The CSVs you provide are the "top_residue_stats.csv" across replicas (produced by the included "dg_barplot.py"
  script), where for each residue:
    *_mean  = mean across replicas (each replica contributes one number)
    *_sem   = stdev_across_replicas / sqrt(n_replicas)
  So *_sem represents replica-to-replica variation, not per-frame noise.
- Replicas are independent (not paired), and n_a, n_b are provided (default 3).
- Welch's t-test is computed from those means, reconstructed SDs, and n's.

Example usage:
  python compare_two_systems.py \
    --csv-a systemA/top_residue_stats.csv --label-a "System A" --color-a "#1f77b4" \
    --csv-b systemB/top_residue_stats.csv --label-b "System B" --color-b "#ff7f0e" \
    --metric TOTAL --order-by A \
    --n-a 3 --n-b 3 \
    --label-threshold 1.0 \
    --p-threshold 0.05 \
    --rename-json rename.json \
    --outfile paired_TOTAL.png \
    --merged-outcsv merged_union_TOTAL.csv \
    --stats-outcsv residue_stats_TOTAL.csv
"""

import argparse
import json
import os
from typing import Dict, Set, List
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COMPONENTS = ["TOTAL", "VDW", "EEL", "PBSOL", "NPSOL", "Internal"]


# ---------------- Renaming helpers ---------------- #

def simplify_label_from_original(residue_str: str) -> str:
    """
    Convert things like:
      'R:A:ASN:387' or 'L:B:ASN:387' -> 'Asn387'
      'ASN387' -> 'Asn387'
    Leave anything unrecognized as-is.
    """
    if not isinstance(residue_str, str):
        return residue_str
    s = residue_str.replace("R:A:", "").replace("L:B:", "")
    parts = s.split(":")
    if len(parts) == 2:
        resname, resnum = parts
        return f"{resname.capitalize()}{resnum}"
    if len(s) >= 4 and s[:3].isalpha() and s[3:].isdigit():
        # e.g. ASN387 -> Asn387
        return s[:1].upper() + s[1:3].lower() + s[3:]
    return s


def all_keys_for_lookup(label: str) -> Set[str]:
    """
    Generate multiple lookup keys for flexible residue renaming, e.g.:
    original, simplified, uppercase, no-colon versions.
    """
    keys = set()
    if isinstance(label, str):
        keys.add(label)
        simp = simplify_label_from_original(label)
        keys.add(simp)
        keys.add(simp.upper())
        keys.add(simp.replace(":", "").upper())
        keys.add(label.replace(":", "").upper())
    return keys


def apply_rename(label: str, rename_map: Dict[str, str]) -> str:
    """
    Apply user-provided rename mapping. Falls back to a simplified label
    for readability if no match is found.
    """
    for k in all_keys_for_lookup(label):
        if k in rename_map:
            return rename_map[k]
    return simplify_label_from_original(label)


def load_rename_map(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


# ---------------- Data loading / shaping ---------------- #

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_for_metric(df: pd.DataFrame, metric: str, suffix: str) -> pd.DataFrame:
    """
    Extract a per-residue table with columns:
      Residue,
      {metric}_mean_{suffix},
      {metric}_sem_{suffix}

    If a needed column doesn't exist, it's filled with NaN.
    """
    mmean = f"{metric}_mean"
    msem  = f"{metric}_sem"
    if mmean not in df.columns:
        df[mmean] = np.nan
    if msem not in df.columns:
        df[msem] = np.nan
    out = df[["Residue", mmean, msem]].copy()
    out = out.rename(columns={
        mmean: f"{metric}_mean_{suffix}",
        msem:  f"{metric}_sem_{suffix}",
    })
    return out


def order_residues_by_source(dfA: pd.DataFrame, dfB: pd.DataFrame, source: str) -> List[str]:
    """
    Create the residue order:
    - Take the row order from the chosen source CSV (A or B).
    - Then append residues that only appear in the other CSV,
      preserving that other CSV's row order.
    """
    listA = dfA["Residue"].tolist()
    listB = dfB["Residue"].tolist()
    if source.upper() == "A":
        base = listA
        extras = [r for r in listB if r not in base]
    else:
        base = listB
        extras = [r for r in listA if r not in base]
    return base + extras


def reindex_union_by_order(unioned: pd.DataFrame, ordered_residues: List[str]) -> pd.DataFrame:
    """
    Apply the given residue order to the merged dataframe (outer join of A and B).
    Anything not in the ordered list will fall to the end.
    """
    cat = pd.Categorical(unioned["Residue"], categories=ordered_residues, ordered=True)
    unioned = unioned.copy()
    unioned["_order"] = cat
    unioned = unioned.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return unioned


# ---------------- Welch t-test helpers ---------------- #

def student_t_two_sided_pvalue(t_stat: float, df: float) -> float:
    """
    Two-sided p-value for a given t-statistic and Welch-Satterthwaite df.

    We compute p = 2 * (1 - CDF_t(|t|, df))

    CDF_t(|t|; df) is computed via the regularized incomplete beta function
    representation of the Student t CDF. We implement a continued-fraction
    approximation here so we don't need SciPy.
    """
    if math.isnan(t_stat) or math.isnan(df):
        return float("nan")
    if df <= 0:
        return float("nan")

    x = abs(t_stat)

    if x == 0.0:
        # t = 0 -> p = 1.0
        return 1.0

    # We need regularized incomplete beta I_z(a,b)
    def betainc_reg(a, b, z):
        if z <= 0.0:
            return 0.0
        if z >= 1.0:
            return 1.0

        def betacf(a, b, x, max_iter=200, eps=3e-14):
            # Lentz-type continued fraction for incomplete beta
            am = 1.0
            bm = 1.0
            az = 1.0
            qab = a + b
            qap = a + 1.0
            qam = a - 1.0
            bz = 1.0 - (qab * x / qap)
            if abs(bz) < 1e-30:
                bz = 1e-30

            em = 0.0
            tem = 0.0
            d = 0.0
            ap = 0.0
            bp = 0.0
            app = 0.0
            bpp = 0.0

            az = 1.0
            bm = bz
            am = 1.0
            if abs(bm) < 1e-30:
                bm = 1e-30
            d = 1.0 / bm
            ap = 1.0
            bp = bm
            az = ap * d
            old_az = az

            for m in range(1, max_iter + 1):
                em = float(m)
                tem = em + em

                # even step
                d = em * (b - em) * x / ((qam + tem) * (a + tem))
                ap = az + d * am
                bp = bm + d * bm
                if abs(bp) < 1e-30:
                    bp = 1e-30
                d2 = 1.0 / bp
                am = ap * d2
                bm = bp * d2

                # odd step
                d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
                ap = am + d * az
                bp = bm + d * bm
                if abs(bp) < 1e-30:
                    bp = 1e-30
                d2 = 1.0 / bp
                az = ap * d2
                bm = bm * d2
                am = am * d2

                if abs(az - old_az) < eps * abs(az):
                    return az
                old_az = az

            return az  # fallback

        lnB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

        # use symmetry for stability
        if z < (a + 1.0) / (a + b + 2.0):
            cf = betacf(a, b, z)
            front = math.exp(
                a * math.log(z)
                + b * math.log(1.0 - z)
                - lnB
            ) / a
            return front * cf
        else:
            cf = betacf(b, a, 1.0 - z)
            front = math.exp(
                b * math.log(1.0 - z)
                + a * math.log(z)
                - lnB
            ) / b
            return 1.0 - front * cf

    nu = df
    z = nu / (nu + x * x)
    Ix = betainc_reg(nu / 2.0, 0.5, z)
    cdf_pos = 1.0 - 0.5 * Ix  # CDF at +|t|
    p_two_sided = 2.0 * (1.0 - cdf_pos)

    # clamp numerically
    if p_two_sided < 0.0:
        p_two_sided = 0.0
    if p_two_sided > 1.0:
        p_two_sided = 1.0
    return p_two_sided


def welch_t_and_p(mean_a: float, sem_a: float, n_a: float,
                  mean_b: float, sem_b: float, n_b: float) -> float:
    """
    Compute Welch's t-test p-value between system A and system B.

    We assume:
      sem_a = sd_a / sqrt(n_a)
      sem_b = sd_b / sqrt(n_b)

    So:
      sd_a = sem_a * sqrt(n_a)
      sd_b = sem_b * sqrt(n_b)

    Welch's t:
      t = (mean_a - mean_b) / sqrt( sd_a^2/n_a + sd_b^2/n_b )

    Welch-Satterthwaite df:
      df = (sd_a^2/n_a + sd_b^2/n_b)^2 /
           [ (sd_a^2/n_a)^2/(n_a-1) + (sd_b^2/n_b)^2/(n_b-1) ]

    Returns:
      p (two-sided). NaN if can't compute.
    """
    if any([
        pd.isna(mean_a), pd.isna(mean_b),
        pd.isna(sem_a), pd.isna(sem_b),
        n_a is None, n_b is None,
        n_a < 2, n_b < 2
    ]):
        return float("nan")

    sd_a = sem_a * math.sqrt(n_a)
    sd_b = sem_b * math.sqrt(n_b)

    var_a_over_na = (sd_a ** 2) / n_a
    var_b_over_nb = (sd_b ** 2) / n_b
    denom = var_a_over_na + var_b_over_nb
    if denom <= 0:
        return float("nan")

    t_stat = (mean_a - mean_b) / math.sqrt(denom)

    denom_df = (
        (var_a_over_na ** 2) / (n_a - 1.0) +
        (var_b_over_nb ** 2) / (n_b - 1.0)
    )
    if denom_df == 0:
        return float("nan")

    df = (denom ** 2) / denom_df
    if df <= 0:
        return float("nan")

    return student_t_two_sided_pvalue(t_stat, df)


# ---------------- Plotting ---------------- #

def paired_barplot(unioned: pd.DataFrame,
                   metric: str,
                   color_a: str,
                   color_b: str,
                   label_a: str,
                   label_b: str,
                   out_png: str,
                   title: str = None,
                   rename_map: Dict[str, str] = None,
                   legend_loc: str = "upper right",
                   label_threshold: float = 0.0,
                   p_threshold: float = 0.05,
                   n_a: int = 3,
                   n_b: int = 3):
    """
    Render the paired bar chart with:
    - bars for A and B
    - missing residues marked by 'x' at y=0
    - numeric labels for bars above |threshold|
    - '*' marking Welch-significant differences (p <= p_threshold)
    """
    rename_map = rename_map or {}
    mA = f"{metric}_mean_A"
    sA = f"{metric}_sem_A"
    mB = f"{metric}_mean_B"
    sB = f"{metric}_sem_B"

    residues_raw = unioned["Residue"].tolist()
    xlabels = [apply_rename(res, rename_map) for res in residues_raw]

    means_A = unioned[mA].astype(float)
    sems_A  = unioned[sA].astype(float)
    means_B = unioned[mB].astype(float)
    sems_B  = unioned[sB].astype(float)

    n = len(unioned)
    x = np.arange(n, dtype=float)
    width = 0.38  # bar width

    fig = plt.figure(figsize=(max(12, n * 0.45), 6), dpi=300)

    # Values to draw (replace NaN with 0 for plotting), but keep original NaN info
    drawA = means_A.fillna(0.0).values
    drawB = means_B.fillna(0.0).values
    errA  = sems_A.fillna(0.0).values
    errB  = sems_B.fillna(0.0).values

    barsA = plt.bar(
        x - width / 2,
        drawA,
        width,
        yerr=errA,
        capsize=6,
        color=color_a,
        edgecolor="black",
        ecolor="black",
        label=label_a,
    )

    barsB = plt.bar(
        x + width / 2,
        drawB,
        width,
        yerr=errB,
        capsize=6,
        color=color_b,
        edgecolor="black",
        ecolor="black",
        label=label_b,
    )

    plt.axhline(0, linewidth=0.8)
    plt.xticks(x, xlabels, rotation=45, ha="right", fontsize=11)
    plt.ylabel(r"$\Delta G$ Contribution (kcal/mol)", fontsize=13)
    if title:
        plt.title(title, fontsize=14)

    # y-limit logic based on |bar|+err
    abs_extents = np.concatenate([np.abs(drawA) + errA, np.abs(drawB) + errB])
    finite = abs_extents[np.isfinite(abs_extents)]
    ymax_core = max(1.0, float(np.max(finite))) if finite.size else 1.0
    label_pad = 0.06 * ymax_core
    ymax_total = ymax_core + 2 * label_pad
    plt.ylim(-1.35 * ymax_total, 1.35 * ymax_total)

    # Helper: numeric value label beyond err bar, only if |orig| >= label_threshold
    def place_value_label(bar, val, err, orig):
        if pd.isna(orig):
            return
        if abs(orig) < label_threshold:
            return
        if val >= 0:
            y_text = val + err + label_pad
            va = "bottom"
        else:
            y_text = val - err - label_pad
            va = "top"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            f"{val:.2f}",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )

    for bar, val, err, orig in zip(barsA, drawA, errA, means_A):
        place_value_label(bar, val, err, orig)
    for bar, val, err, orig in zip(barsB, drawB, errB, means_B):
        place_value_label(bar, val, err, orig)

    # Mark missing data explicitly with 'x' at y=0
    for bar, orig in zip(barsA, means_A):
        if pd.isna(orig):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                0.0,
                "x",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
            )
    for bar, orig in zip(barsB, means_B):
        if pd.isna(orig):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                0.0,
                "x",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
            )

    # Welch significance markers ('*' if p <= p_threshold)
    for i in range(n):
        mean_a = means_A.iat[i]
        sem_a  = sems_A.iat[i]
        mean_b = means_B.iat[i]
        sem_b  = sems_B.iat[i]

        # Skip if either side missing
        if pd.isna(mean_a) or pd.isna(mean_b):
            continue

        pval = welch_t_and_p(mean_a, sem_a, n_a, mean_b, sem_b, n_b)
        if pd.isna(pval) or pval > p_threshold:
            continue

        # pick bar with bigger span for star placement
        spanA = abs(drawA[i]) + errA[i]
        spanB = abs(drawB[i]) + errB[i]
        if spanA >= spanB:
            ref_bar = barsA[i]
            ref_val = drawA[i]
            ref_err = errA[i]
        else:
            ref_bar = barsB[i]
            ref_val = drawB[i]
            ref_err = errB[i]

        if ref_val >= 0:
            star_y = ref_val + ref_err + (label_pad * 1.6)
            va = "bottom"
        else:
            star_y = ref_val - ref_err - (label_pad * 1.6)
            va = "top"

        plt.text(
            ref_bar.get_x() + ref_bar.get_width() / 2,
            star_y,
            "*",
            ha="center",
            va=va,
            fontsize=12,
            fontweight="bold",
            color="black",
        )

    # Legend
    plt.legend(
        loc=legend_loc,
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------- Stats table builder ---------------- #

def build_stats_table(unioned: pd.DataFrame,
                      metric: str,
                      n_a: int,
                      n_b: int,
                      p_threshold: float) -> pd.DataFrame:
    """
    Create a table with per-residue stats and significance calls:
      Residue, mean_A, sem_A, n_A, mean_B, sem_B, n_B, diff(A-B),
      p_value (Welch), significant (p<=p_threshold)
    """
    mA = f"{metric}_mean_A"
    sA = f"{metric}_sem_A"
    mB = f"{metric}_mean_B"
    sB = f"{metric}_sem_B"

    rows = []
    for _, row in unioned.iterrows():
        residue = row["Residue"]
        mean_a = row.get(mA, np.nan)
        sem_a  = row.get(sA, np.nan)
        mean_b = row.get(mB, np.nan)
        sem_b  = row.get(sB, np.nan)

        if pd.isna(mean_a) or pd.isna(mean_b):
            pval = np.nan
        else:
            pval = welch_t_and_p(mean_a, sem_a, n_a, mean_b, sem_b, n_b)

        diff = np.nan
        if not pd.isna(mean_a) and not pd.isna(mean_b):
            diff = mean_a - mean_b

        rows.append({
            "Residue": residue,
            f"{metric}_mean_A": mean_a,
            f"{metric}_sem_A":  sem_a,
            "n_A": n_a,
            f"{metric}_mean_B": mean_b,
            f"{metric}_sem_B":  sem_b,
            "n_B": n_b,
            f"{metric}_diff_AminusB": diff,
            "p_value": pval,
            "significant": (not pd.isna(pval) and pval <= p_threshold),
        })

    return pd.DataFrame(rows)


# ---------------- Main ---------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Paired residue energy plot with missing-data X markers, Welch significance stars, label thresholding, and optional per-residue stats CSV."
    )
    ap.add_argument("--csv-a", required=True,
                    help="Path to Average_Across_Replicas/top_residue_stats.csv for dataset A")
    ap.add_argument("--csv-b", required=True,
                    help="Path to Average_Across_Replicas/top_residue_stats.csv for dataset B")
    ap.add_argument("--label-a", default="Set A",
                    help="Legend label for dataset A")
    ap.add_argument("--label-b", default="Set B",
                    help="Legend label for dataset B")
    ap.add_argument("--color-a", default="#1f77b4",
                    help="Bar color for dataset A (hex/name)")
    ap.add_argument("--color-b", default="#ff7f0e",
                    help="Bar color for dataset B (hex/name)")
    ap.add_argument("--metric", default="TOTAL", choices=COMPONENTS,
                    help="Energy component to plot (TOTAL, VDW, etc.)")
    ap.add_argument("--rename-json", default=None,
                    help="Optional JSON file for residue renaming")
    ap.add_argument("--outfile", default="paired_plot.png",
                    help="Output PNG filename")
    ap.add_argument("--merged-outcsv", default=None,
                    help="Optional path to save the merged union CSV (debug/inspection)")
    ap.add_argument("--stats-outcsv", default=None,
                    help="Optional path to save a per-residue stats table with p-values")
    ap.add_argument("--order-by", choices=["A", "B"], default="A",
                    help="Choose which CSV defines residue order; residues unique to the other CSV are appended.")
    ap.add_argument("--label-threshold", type=float, default=0.0,
                    help="Only show numeric value labels if |energy| >= this threshold.")
    ap.add_argument("--p-threshold", type=float, default=0.05,
                    help="Draw '*' or mark 'significant' if Welch test p <= this threshold.")
    ap.add_argument("--n-a", type=int, default=3,
                    help="Number of independent replicas in dataset A (used for Welch test).")
    ap.add_argument("--n-b", type=int, default=3,
                    help="Number of independent replicas in dataset B (used for Welch test).")
    args = ap.parse_args()

    # Load CSVs
    dfA = load_csv(args.csv_a)
    dfB = load_csv(args.csv_b)

    # Build per-metric tables
    A = prepare_for_metric(dfA, args.metric, "A")
    B = prepare_for_metric(dfB, args.metric, "B")

    # Outer join on 'Residue' so we include union of residues
    unioned = pd.merge(A, B, on="Residue", how="outer")

    # Apply ordering logic (preserve chosen CSV row order, append extras)
    ordered_residues = order_residues_by_source(dfA, dfB, args.order_by)
    unioned = reindex_union_by_order(unioned, ordered_residues)

    # Optionally save the merged table
    if args.merged_outcsv:
        unioned.to_csv(args.merged_outcsv, index=False)

    # Create per-residue stats summary (with Welch p-values)
    if args.stats_outcsv:
        stats_df = build_stats_table(
            unioned=unioned,
            metric=args.metric,
            n_a=args.n_a,
            n_b=args.n_b,
            p_threshold=args.p_threshold,
        )
        stats_df.to_csv(args.stats_outcsv, index=False)

    # Load optional rename map
    rename_map = load_rename_map(args.rename_json) if args.rename_json else {}

    # Title for the figure
    title = (
        f"{args.metric} Energy Contributions by Residue\n"
        f"{args.label_a} vs {args.label_b}"
    )

    # Plot
    paired_barplot(
        unioned=unioned,
        metric=args.metric,
        color_a=args.color_a,
        color_b=args.color_b,
        label_a=args.label_a,
        label_b=args.label_b,
        out_png=args.outfile,
        title=title,
        rename_map=rename_map,
        legend_loc="upper right",
        label_threshold=args.label_threshold,
        p_threshold=args.p_threshold,
        n_a=args.n_a,
        n_b=args.n_b,
    )

    print(f"Saved figure: {args.outfile}")
    if args.merged_outcsv:
        print(f"Saved merged CSV: {args.merged_outcsv}")
    if args.stats_outcsv:
        print(f"Saved stats table: {args.stats_outcsv}")


if __name__ == "__main__":
    main()
