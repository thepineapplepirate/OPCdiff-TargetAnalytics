#!/usr/bin/env python3
"""
rmsf_barplot.py

Plot per-residue RMSF from multiple .xvg files in four modes:

  1) mean                 Mean ± SEM across replicas (error bars), value labels above error bars
  2) grouped              Grouped bars per residue (one bar per replica), labels above bars (no error bars)
  3) mean_compare         Compare sets side-by-side per residue:
                          Complex 1, optional Complex 2, and Apo (each with mean ± SEM), labels above error bars.
                          Welch's t-tests between each pair of groups (Complex/Apo); significant residues
                          (p < threshold) are marked with a '*' above the tallest bar for that residue.
  4) mean_compare_overall Single summary bars: overall mean ± SEM (across residues) for Complex 1,
                          optional Complex 2, and Apo; labels above error bars.
                          Welch's t-tests on per-residue means between each pair of groups; significant
                          comparisons are marked with a '*' between the corresponding bars.

Features:
  - RMSF is converted from nm to Å (×10).
  - Residue renaming via --rename (inline JSON, JSON file, or CSV with columns: residue,label).
  - Tick thinning (--tick-step) and axis tick sizes (--xtick-size, --ytick-size).
  - Custom labels/colors for each compare group.
  - p-value threshold (--pvalue-threshold, default 0.05) for significance markers.
  - CSV/DataFrame export (--stats-out) with RMSF stats and p-values in compare modes, plus a
    'significance' column ("True"/"False").
"""

import argparse
import csv
import glob
import json
import os
import re
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_ind
except ImportError:
    ttest_ind = None


# ------------------------
# Utilities
# ------------------------
def load_xvg_series(path):
    """Load a two-column XVG (index, value) into a pandas Series keyed by residue index."""
    residues, rmsf = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(("@", "#")) or not line.strip():
                continue
            parts = line.strip().split()
            try:
                residues.append(int(parts[0]))
                rmsf.append(float(parts[1]))
            except (IndexError, ValueError):
                raise ValueError(f"Invalid data format in file {path}: {line}")
    return pd.Series(rmsf, index=residues, name=os.path.basename(path))


def parse_rename_mapping(arg: str) -> Dict[int, str]:
    """
    Parse residue renaming mapping.
      - Inline JSON: '{"55":"Ile55","56":"Val56"}'
      - JSON file with same structure
      - CSV file with columns: residue,label
    Returns dict[int, str], stripping non-numeric chars from keys (e.g. "Ile55" → 55).
    """

    def clean_key(k):
        num = re.sub(r"\D", "", str(k))
        return int(num) if num else None

    if not arg:
        return {}
    if os.path.exists(arg):
        if arg.lower().endswith(".json"):
            with open(arg, "r") as f:
                raw = json.load(f)
            return {clean_key(k): str(v) for k, v in raw.items() if clean_key(k) is not None}
        if arg.lower().endswith(".csv"):
            mapping = {}
            with open(arg, newline="") as f:
                reader = csv.DictReader(f)
                if "residue" not in reader.fieldnames or "label" not in reader.fieldnames:
                    raise ValueError("CSV must have columns: residue,label")
                for row in reader:
                    key = clean_key(row["residue"])
                    if key is not None:
                        mapping[key] = str(row["label"])
            return mapping
        raise ValueError("Unknown file type for --rename. Use .json or .csv")
    # inline JSON
    try:
        raw = json.loads(arg)
        return {clean_key(k): str(v) for k, v in raw.items() if clean_key(k) is not None}
    except Exception as e:
        raise ValueError(f"Failed to parse --rename as JSON: {e}")


def expand_paths(patterns):
    """Expand globs and keep only existing files, preserving order."""
    out = []
    for pat in patterns:
        expanded = glob.glob(pat)
        if expanded:
            out.extend(expanded)
        else:
            out.append(pat)
    out = [p for p in out if os.path.exists(p)]
    return out


def scale_pad(ymin, ymax, frac=0.01, fallback=0.05):
    """Return a padding value based on y-range (scale-aware)."""
    return frac * (ymax - ymin) if ymax > ymin else fallback


# ------------------------
# Main
# ------------------------
def main():
    p = argparse.ArgumentParser(description="Bar plots for RMSF from multiple XVG files.")

    # Base inputs (used by mean & grouped)
    p.add_argument(
        "--xvg",
        nargs="+",
        default=[
            "replica1/galaxy_analysis/rmsf_orthores.xvg",
            "replica2/galaxy_analysis/rmsf_orthores.xvg",
            "replica3/galaxy_analysis/rmsf_orthores.xvg",
        ],
        help="List of .xvg paths or globs (space-separated). Used by 'mean' and 'grouped' modes.",
    )

    # Compare inputs (two complexes + apo)
    p.add_argument(
        "--complex1-xvg",
        nargs="+",
        default=[
            "replica1/galaxy_analysis/rmsf_orthores.xvg",
            "replica2/galaxy_analysis/rmsf_orthores.xvg",
            "replica3/galaxy_analysis/rmsf_orthores.xvg",
        ],
        help="List of .xvg paths or globs for COMPLEX 1 simulations (compare modes).",
    )
    p.add_argument(
        "--complex2-xvg",
        nargs="+",
        default=[],
        help="Optional list of .xvg paths or globs for COMPLEX 2 simulations (compare modes).",
    )
    p.add_argument(
        "--apo-xvg",
        nargs="+",
        default=[
            "replica1/galaxy_analysis/rmsf_orthores_apo.xvg",
            "replica2/galaxy_analysis/rmsf_orthores_apo.xvg",
            "replica3/galaxy_analysis/rmsf_orthores_apo.xvg",
        ],
        help="List of .xvg paths or globs for APO/UNBOUND simulations (compare modes).",
    )

    p.add_argument(
        "--mode",
        choices=["mean", "grouped", "mean_compare", "mean_compare_overall"],
        default="mean",
        help="Plot mode.",
    )

    # Custom labels for compare modes
    p.add_argument("--complex1-label", default="Complex 1", help="Label for COMPLEX 1 in compare modes.")
    p.add_argument("--complex2-label", default="Complex 2", help="Label for COMPLEX 2 in compare modes.")
    p.add_argument("--apo-label", default="Apo", help="Label for APO in compare modes.")

    # Colors for compare modes
    p.add_argument("--complex1-color", default="blue", help="Bar color for COMPLEX 1.")
    p.add_argument("--complex2-color", default="green", help="Bar color for COMPLEX 2.")
    p.add_argument("--apo-color", default="orange", help="Bar color for APO.")

    # General formatting
    p.add_argument("--rename", default=None, help="Residue mapping (inline JSON or JSON/CSV file).")
    p.add_argument("--title", default="Per-Residue RMSF", help="Plot title.")
    p.add_argument("--xlabel", default=None, help="X-axis label (overridden in some modes).")
    p.add_argument("--ylabel", default="RMSF (Å)", help="Y-axis label. (RMSF converted from nm → Å)")
    p.add_argument("--figsize", default="10,6", help="Width,height in inches (e.g., 10,6).")
    p.add_argument("--dpi", type=int, default=600, help="Figure DPI.")
    p.add_argument("--out", default=None, help="Output image path (e.g., plot.png).")
    p.add_argument("--tick-step", type=int, default=None, help="Show every Nth x tick label.")
    p.add_argument("--ytick-size", type=int, default=14, help="Y-axis tick label font size.")
    p.add_argument("--xtick-size", type=int, default=16, help="X-axis tick label font size.")
    p.add_argument("--ymin", type=float, default=None, help="Lower limit for y-axis range.")
    p.add_argument("--ymax", type=float, default=None, help="Upper limit for y-axis range.")
    p.add_argument("--label-precision", type=int, default=2, help="Decimal places for value labels.")
    p.add_argument("--label-fontsize", type=int, default=12, help="Font size for value labels.")
    p.add_argument("--legend-fontsize", type=int, default=14, help="Font size for legend text (default: 10).")


    # Stats / p-values
    p.add_argument(
        "--pvalue-threshold",
        type=float,
        default=0.05,
        help="Significance threshold for Welch's t-test; significant comparisons get a '*' (default: 0.05).",
    )
    p.add_argument(
        "--stats-out",
        default=None,
        help="Optional CSV path for stats (RMSF + p-values) in compare modes. "
             "Defaults to 'rmsf_stats_<mode>.csv' if not set.",
    )

    args = p.parse_args()

    if args.mode.startswith("mean_compare") and ttest_ind is None:
        raise RuntimeError(
            "SciPy is required for Welch's t-tests in mean_compare modes. "
            "Install with 'pip install scipy' or 'conda install scipy'."
        )

    # Figure setup
    try:
        w, h = (float(x) for x in args.figsize.split(","))
    except Exception:
        raise ValueError("Invalid --figsize. Use 'width,height' like '10,6'.")
    plt.figure(figsize=(w, h))

    rename_map = parse_rename_mapping(args.rename) if args.rename else {}
    fmt = "{:." + str(args.label_precision) + "f}"

    # ------------------------
    # MEAN mode (Mean ± SEM per residue)
    # ------------------------
    if args.mode == "mean":
        xvg_files = expand_paths(args.xvg)
        if not xvg_files:
            raise FileNotFoundError("No input .xvg files found for --xvg.")

        series_list = [load_xvg_series(f) for f in xvg_files]
        df = pd.concat(series_list, axis=1)
        df.columns = [f"Replica_{i+1}" for i in range(len(xvg_files))]
        df = df * 10.0  # nm → Å

        mean = df.mean(axis=1)
        sem = df.std(axis=1) / np.sqrt(df.shape[1])

        x = range(len(df.index))
        plt.bar(x, mean, width=0.3, align="center")
        plt.errorbar(x, mean, yerr=sem, fmt="none", ecolor="black", elinewidth=1, capsize=3)

        ymin = float((mean - sem).min())
        ymax = float((mean + sem).max())
        pad = scale_pad(ymin, ymax)

        for xi, (val, err) in enumerate(zip(mean, sem)):
            plt.text(xi, val + err + pad, fmt.format(val), ha="center", va="bottom", fontsize=args.label_fontsize)

        x_labels = [rename_map.get(int(i), str(int(i))) for i in df.index]
        rotate = 90 if len(x_labels) > 40 else 0
        plt.xticks(x, x_labels, rotation=45, fontsize=args.xtick_size)
        plt.legend(["Mean ± SEM"])

    # ------------------------
    # MEAN_COMPARE mode (per-residue side-by-side)
    # ------------------------
    elif args.mode == "mean_compare":
        c1_files = expand_paths(args.complex1_xvg)
        c2_files = expand_paths(args.complex2_xvg) if args.complex2_xvg else []
        apo_files = expand_paths(args.apo_xvg)
        if not c1_files:
            raise FileNotFoundError("No input .xvg files found for --complex1-xvg.")
        if not apo_files:
            raise FileNotFoundError("No input .xvg files found for --apo-xvg.")

        df_c1 = pd.concat([load_xvg_series(f) for f in c1_files], axis=1) * 10.0
        df_apo = pd.concat([load_xvg_series(f) for f in apo_files], axis=1) * 10.0
        df_c2 = pd.concat([load_xvg_series(f) for f in c2_files], axis=1) * 10.0 if c2_files else None

        # Align residues across all present sets
        common_idx = df_c1.index.intersection(df_apo.index)
        if df_c2 is not None:
            common_idx = common_idx.intersection(df_c2.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping residue indices across provided sets.")
        df_c1 = df_c1.loc[common_idx]
        df_apo = df_apo.loc[common_idx]
        if df_c2 is not None:
            df_c2 = df_c2.loc[common_idx]

        residues = common_idx

        # Build groups (mean & SEM)
        groups = []
        group_dfs = []
        group_names = []

        c1_mean = df_c1.mean(axis=1)
        c1_sem = df_c1.std(axis=1) / np.sqrt(df_c1.shape[1])
        groups.append((args.complex1_label, c1_mean, c1_sem, args.complex1_color))
        group_dfs.append(df_c1)
        group_names.append(args.complex1_label)

        if df_c2 is not None:
            c2_mean = df_c2.mean(axis=1)
            c2_sem = df_c2.std(axis=1) / np.sqrt(df_c2.shape[1])
            groups.append((args.complex2_label, c2_mean, c2_sem, args.complex2_color))
            group_dfs.append(df_c2)
            group_names.append(args.complex2_label)

        apo_mean = df_apo.mean(axis=1)
        apo_sem = df_apo.std(axis=1) / np.sqrt(df_apo.shape[1])
        groups.append((args.apo_label, apo_mean, apo_sem, args.apo_color))
        group_dfs.append(df_apo)
        group_names.append(args.apo_label)

        apo_idx = len(group_dfs) - 1

        x_labels = [rename_map.get(int(i), str(int(i))) for i in residues]
        x = list(range(len(residues)))

        total_width = 0.85
        k = len(groups)
        bar_w = total_width / k
        offsets = [(-total_width / 2) + (i + 0.5) * bar_w for i in range(k)]

        ymin = float(min([(m - s).min() for _, m, s, _ in groups]))
        ymax = float(max([(m + s).max() for _, m, s, _ in groups]))
        pad = scale_pad(ymin, ymax)

        # Stats dict (per-residue)
        stats_dict = {
            "residue_index": list(residues),
            "residue_label": x_labels,
        }
        # Add group mean/SEM columns
        for (label, mean_s, sem_s, _) in groups:
            key_base = label.replace(" ", "_")
            stats_dict[f"mean_{key_base}"] = list(mean_s.values)
            stats_dict[f"sem_{key_base}"] = list(sem_s.values)

        significant_residues = set()

        # Complex vs Apo p-values
        for gi in range(apo_idx):  # complexes are indices 0..apo_idx-1
            label1 = group_names[gi]
            label2 = group_names[apo_idx]
            key1 = label1.replace(" ", "_")
            key2 = label2.replace(" ", "_")
            col_name = f"pvalue_{key1}_vs_{key2}"

            pvals = []
            for res in residues:
                a = group_dfs[gi].loc[res].values
                b = group_dfs[apo_idx].loc[res].values
                if len(a) < 2 or len(b) < 2:
                    pval = np.nan
                else:
                    _, pval = ttest_ind(a, b, equal_var=False)
                pvals.append(pval)
                if not np.isnan(pval) and pval < args.pvalue_threshold:
                    significant_residues.add(res)
            stats_dict[col_name] = pvals

        # Complex1 vs Complex2 p-values (if C2 present)
        if df_c2 is not None:
            key1 = args.complex1_label.replace(" ", "_")
            key2 = args.complex2_label.replace(" ", "_")
            col_name = f"pvalue_{key1}_vs_{key2}"
            pvals_c1_c2 = []
            for res in residues:
                a = df_c1.loc[res].values
                b = df_c2.loc[res].values
                if len(a) < 2 or len(b) < 2:
                    pval = np.nan
                else:
                    _, pval = ttest_ind(a, b, equal_var=False)
                pvals_c1_c2.append(pval)
                if not np.isnan(pval) and pval < args.pvalue_threshold:
                    significant_residues.add(res)
            stats_dict[col_name] = pvals_c1_c2

        # Significance column ("True" if any comparison at that residue is significant)
        stats_dict["significance"] = [
            "True" if res in significant_residues else "False" for res in residues
        ]

        # Plot each group
        for gi, (label, mean_s, sem_s, color) in enumerate(groups):
            x_pos = [xi + offsets[gi] for xi in x]
            plt.bar(x_pos, mean_s, width=bar_w, color=color, label=label)
            plt.errorbar(x_pos, mean_s, yerr=sem_s, fmt="none", ecolor="black", elinewidth=1, capsize=3)
            for xp, val, err in zip(x_pos, mean_s, sem_s):
                plt.text(xp, val + err + pad, fmt.format(val), ha="center", va="bottom", fontsize=args.label_fontsize)

        # Add significance stars above residues where any comparison is significant
        for idx, res in enumerate(residues):
            if res not in significant_residues:
                continue
            tops = [(m.iloc[idx] + s.iloc[idx]) for _, m, s, _ in groups]
            star_y = max(tops) + pad * 2.0
            plt.text(
                x[idx],
                star_y,
                "*",
                ha="center",
                va="bottom",
                fontsize=args.label_fontsize + 2,
                color="black",
            )

        rotate = 90 if len(x_labels) > 40 else 0
        plt.xticks(x, x_labels, rotation=45, fontsize=args.xtick_size)
        handles = [mpatches.Patch(color=c, label=lab) for lab, _, _, c in groups]
        plt.legend(handles=handles, fontsize=args.legend_fontsize)

        # Export stats DataFrame
        stats_df = pd.DataFrame(stats_dict)
        stats_path = args.stats_out or f"rmsf_stats_{args.mode}.csv"
        stats_df.to_csv(stats_path, index=False)
        print("Saved stats to:", stats_path)

    # ------------------------
    # MEAN_COMPARE_OVERALL mode (summary bars)
    # ------------------------
    elif args.mode == "mean_compare_overall":
        c1_files = expand_paths(args.complex1_xvg)
        c2_files = expand_paths(args.complex2_xvg) if args.complex2_xvg else []
        apo_files = expand_paths(args.apo_xvg)
        if not c1_files:
            raise FileNotFoundError("No input .xvg files found for --complex1-xvg.")
        if not apo_files:
            raise FileNotFoundError("No input .xvg files found for --apo-xvg.")

        df_c1 = pd.concat([load_xvg_series(f) for f in c1_files], axis=1) * 10.0
        df_apo = pd.concat([load_xvg_series(f) for f in apo_files], axis=1) * 10.0
        df_c2 = pd.concat([load_xvg_series(f) for f in c2_files], axis=1) * 10.0 if c2_files else None

        # Align residues across all present sets
        common_idx = df_c1.index.intersection(df_apo.index)
        if df_c2 is not None:
            common_idx = common_idx.intersection(df_c2.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping residue indices across provided sets.")
        df_c1 = df_c1.loc[common_idx]
        df_apo = df_apo.loc[common_idx]
        if df_c2 is not None:
            df_c2 = df_c2.loc[common_idx]

        # Per-residue means (across replicas)
        c1_prm = df_c1.mean(axis=1)
        apo_prm = df_apo.mean(axis=1)

        labels = []
        heights = []
        errors_sem = []
        n_res_list = []
        colors = []
        samples = []

        # Complex 1
        n_c1 = len(c1_prm)
        c1_mean_overall = float(c1_prm.mean())
        c1_sd = float(c1_prm.std())
        c1_sem = c1_sd / np.sqrt(n_c1)
        labels.append(args.complex1_label)
        heights.append(c1_mean_overall)
        errors_sem.append(c1_sem)
        n_res_list.append(n_c1)
        colors.append(args.complex1_color)
        samples.append(c1_prm)

        # Complex 2 (optional)
        if df_c2 is not None:
            c2_prm = df_c2.mean(axis=1)
            n_c2 = len(c2_prm)
            c2_mean_overall = float(c2_prm.mean())
            c2_sd = float(c2_prm.std())
            c2_sem = c2_sd / np.sqrt(n_c2)
            labels.append(args.complex2_label)
            heights.append(c2_mean_overall)
            errors_sem.append(c2_sem)
            n_res_list.append(n_c2)
            colors.append(args.complex2_color)
            samples.append(c2_prm)
        else:
            c2_prm = None

        # Apo
        n_apo = len(apo_prm)
        apo_mean_overall = float(apo_prm.mean())
        apo_sd = float(apo_prm.std())
        apo_sem = apo_sd / np.sqrt(n_apo)
        labels.append(args.apo_label)
        heights.append(apo_mean_overall)
        errors_sem.append(apo_sem)
        n_res_list.append(n_apo)
        colors.append(args.apo_color)
        samples.append(apo_prm)

        x_pos = list(range(len(labels)))
        plt.bar(x_pos, heights, yerr=errors_sem, capsize=4, color=colors, align="center")

        ymin = min([h - e for h, e in zip(heights, errors_sem)])
        ymax = max([h + e for h, e in zip(heights, errors_sem)])
        pad = scale_pad(ymin, ymax)

        # Value labels on top of bars
        for xp, val, err in zip(x_pos, heights, errors_sem):
            plt.text(xp, val + err + pad, fmt.format(val), ha="center", va="bottom", fontsize=args.label_fontsize)

        # Welch t-tests between each complex and Apo (on per-residue means)
        apo_idx = len(labels) - 1
        pvals_vs_apo = [np.nan] * len(labels)
        significance_pairs = []

        for i in range(apo_idx):
            a = samples[i].values
            b = samples[apo_idx].values
            if len(a) < 2 or len(b) < 2:
                pval = np.nan
            else:
                _, pval = ttest_ind(a, b, equal_var=False)
            pvals_vs_apo[i] = pval
            if not np.isnan(pval) and pval < args.pvalue_threshold:
                significance_pairs.append((i, apo_idx))

        # Complex 1 vs Complex 2 (if present)
        pvals_c1_c2 = [np.nan] * len(labels)
        if c2_prm is not None:
            a = samples[0].values   # Complex 1
            b = samples[1].values   # Complex 2
            if len(a) < 2 or len(b) < 2:
                pval_c1_c2 = np.nan
            else:
                _, pval_c1_c2 = ttest_ind(a, b, equal_var=False)
            pvals_c1_c2[0] = pval_c1_c2
            pvals_c1_c2[1] = pval_c1_c2
            if not np.isnan(pval_c1_c2) and pval_c1_c2 < args.pvalue_threshold:
                significance_pairs.append((0, 1))

        # Add significance stars between bars for all significant pairs
        for i, j in significance_pairs:
            x_mid = 0.5 * (x_pos[i] + x_pos[j])
            top = max(heights[i] + errors_sem[i], heights[j] + errors_sem[j])
            star_y = top + pad * 2.0
            plt.text(
                x_mid,
                star_y,
                "*",
                ha="center",
                va="bottom",
                fontsize=args.label_fontsize + 2,
                color="black",
            )

        plt.xticks(x_pos, labels, fontsize=args.xtick_size)
        plt.xlabel("")  # not residue-based
        handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        plt.legend(handles=handles, fontsize=args.legend_fontsize)
        if args.title == "Per-Residue RMSF":
            plt.title("Average RMSF: Comparison")

        # Export overall stats DataFrame
        stats_dict = {
            "group": labels,
            "mean": heights,
            "sem": errors_sem,
            "n_residues": n_res_list,
            "pvalue_vs_apo": pvals_vs_apo,
            "pvalue_complex1_vs_complex2": pvals_c1_c2,
        }

        # significance per group: True if this complex is significantly different
        # from Apo OR from the other complex (Apo row stays "False")
        significance_flags = []
        for i in range(len(labels)):
            if i == apo_idx:
                significance_flags.append("False")
                continue
            sig = False
            pval_apo = pvals_vs_apo[i]
            if not np.isnan(pval_apo) and pval_apo < args.pvalue_threshold:
                sig = True
            else:
                pval_cc = pvals_c1_c2[i]
                if not np.isnan(pval_cc) and pval_cc < args.pvalue_threshold:
                    sig = True
            significance_flags.append("True" if sig else "False")
        stats_dict["significance"] = significance_flags

        stats_df = pd.DataFrame(stats_dict)
        stats_path = args.stats_out or f"rmsf_stats_{args.mode}.csv"
        stats_df.to_csv(stats_path, index=False)
        print("Saved stats to:", stats_path)

    # ------------------------
    # GROUPED mode
    # ------------------------
    else:  # args.mode == "grouped"
        xvg_files = expand_paths(args.xvg)
        if not xvg_files:
            raise FileNotFoundError("No input .xvg files found for --xvg.")

        df = pd.concat([load_xvg_series(f) for f in xvg_files], axis=1)
        df.columns = [f"Replica_{i+1}" for i in range(len(xvg_files))]
        df = df * 10.0

        n_files = df.shape[1]
        x = range(len(df.index))
        total_width = 0.9
        bar_w = max(0.05, total_width / max(1, n_files))
        offsets = [(-total_width / 2) + (i + 0.5) * bar_w for i in range(n_files)]

        replica_colors = ["black", "blue", "red", "green", "purple", "orange"]

        ymin = float(df.min().min())
        ymax = float(df.max().max())
        pad = scale_pad(ymin, ymax)

        for i in range(n_files):
            x_pos_i = [xi + offsets[i] for xi in x]
            y = df.iloc[:, i].values
            color = replica_colors[i % len(replica_colors)]
            plt.bar(x_pos_i, y, width=bar_w, align="center", label=df.columns[i], color=color)
            for xp, val in zip(x_pos_i, y):
                plt.text(xp, val + pad, fmt.format(val), ha="center", va="bottom", fontsize=args.label_fontsize)

        x_labels = [rename_map.get(int(i), str(int(i))) for i in df.index]
        rotate = 90 if len(x_labels) > 40 else 0
        plt.xticks(range(len(x_labels)), x_labels, rotation=45, fontsize=args.xtick_size)
        plt.legend(fontsize=args.legend_fontsize, title="Dataset")

    # Common labels/grid/output
    plt.title(args.title, fontsize=18)
    plt.xlabel(plt.gca().get_xlabel() or (args.xlabel or ""), fontsize=18)
    plt.ylabel(args.ylabel, fontsize=18)

    if args.tick_step and args.tick_step > 1:
        ax = plt.gca()
        ticks = ax.get_xticks()
        labels = [t.get_text() for t in ax.get_xticklabels()]
        locs = list(range(0, len(ticks), args.tick_step))
        if locs:
            plt.xticks([ticks[i] for i in locs], [labels[i] for i in locs], fontsize=args.xtick_size)

    if args.ytick_size:
        plt.tick_params(axis="y", labelsize=args.ytick_size)

    if args.ymin is not None or args.ymax is not None:
        plt.ylim(args.ymin, args.ymax)

    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = args.out or f"rmsf_{args.mode}.png"
    plt.savefig(out_path, dpi=args.dpi)
    print("Saved plot to:", out_path)


if __name__ == "__main__":
    main()
