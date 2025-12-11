#!/usr/bin/env python3
"""
dg_barplot.py

Multi-replica MMPBSA residue decomposition plotting.

Features
--------
1) Parse multiple gmx_MMPBSA FINAL_DECOMP_MMPBSA.csv-like files (replicas).
   For each replica:
     - Extract the "Total Decomposition Contribution (TDC)" table
     - Compute per-residue mean ± SEM across frames
     - Generate per-replica energy component bar plots
     - Save per-replica top_residue_stats.csv

2) Compute across-replica "average of those averages":
     - For each residue, take per-replica means and compute:
         * Combined mean  (average of per-replica means, ignoring NaN)
         * Combined SEM   (std of per-replica means / sqrt(n), ignoring NaN)
     - Rank residues by |TOTAL_mean| and keep top_k
       (Set top_k = None to keep ALL residues.)
     - Save combined 'top_residue_stats.csv'
     - Generate combined plots (mean ± SEM)

3) Residue ordering for per-replica plots:
     - 'combined'    : use combined ordering by TOTAL_mean
     - 'per-replica' : each replica sorted by its own TOTAL_mean
     - 'manual'      : supply manual_sort_list (list or text file path)
                       Matching works against several label formats.

4) force_same_x_as_combined
     - When True and mode='combined', per-replica plots use the exact x-axis
       of the combined plot, keeping residues even if a replica is missing them.
       Missing values stay NaN in CSV; bars draw at 0 height for visibility,
       but labels are **omitted** for missing values to avoid “0.00” confusion.
"""

import os
import json
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ======================= CONFIG ======================= #

# 1) List your replica CSVs here (absolute or relative paths)
csv_paths = [
    "path/to/replica1/FINAL_DECOMP_MMPBSA.csv",
    "path/to/replica2/FINAL_DECOMP_MMPBSA.csv",
    "path/to/replica3/FINAL_DECOMP_MMPBSA.csv",
]

drug_name     = "Clemastine"
receptor_name = "M1"

# Tag for the across-replica (combined) outputs
combined_tag  = "Average_Across_Replicas"

# Optional tags per replica; if None or length mismatch, tags fall back to "Replica1", "Replica2", ...
replica_tags = ["Replica1", "Replica2", "Replica3"]

# Optional: path to JSON mapping of residue labels -> display labels; set to None to disable
rename_json_path = "../ortho_resname_dict.json"  # or None

# Exclude ligand residues that start with 'L:B:'
exclude_ligand = True

# How many residues to keep (by |TOTAL_mean|) for the combined summary and for plotting
# Set to None to keep ALL residues.
top_k = None  # e.g., 20, 50, or None for all

# Sorting mode for per-replica plots: 'combined' | 'per-replica' | 'manual'
replica_sort_mode = "combined"

# Manual sort list: either a Python list of strings, or a path to a text file with one residue per line.
# Each item can be 'R:A:ASN:387', 'Asn387', or 'ASN387' → script matches flexibly.
manual_sort_list = None  # e.g., ["Asn387", "Tyr104", "Asp86"] or "manual_sort_order.txt"

# Force per-replica plots to share the exact x-axis (residue list & order) with the combined plot
# Only applies when replica_sort_mode == "combined"
force_same_x_as_combined = True

# Output base directories (will be created if needed)
combined_output_dir_base = f"decomp/energy_component_plots_{combined_tag}"
per_replica_output_dir_base = "decomp/energy_component_plots_{tag}"  # {tag} placeholder is replaced

# ====================================================== #


# ---------------- Renaming / Label helpers ---------------- #

def load_rename_map(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

rename_map = load_rename_map(rename_json_path)

def simplify_label_from_original(residue_str):
    """
    From 'R:A:ASN:387' or 'L:B:ASN:387' -> 'Asn387'
    Leaves non-matching strings unchanged.
    """
    if not isinstance(residue_str, str):
        return residue_str
    clean = residue_str.replace("R:A:", "").replace("L:B:", "")
    parts = clean.split(":")
    if len(parts) == 2:
        resname, resnum = parts
        return f"{resname.capitalize()}{resnum}"
    return clean

def all_keys_for_lookup(label):
    """
    Return a set of possible lookup keys derived from an input label.
    Tries original, simplified, caps/no-colon.
    """
    keys = set()
    if isinstance(label, str):
        keys.add(label)
        simp = simplify_label_from_original(label)
        keys.add(simp)
        caps = simp.replace(":", "").upper() if ":" in simp else simp.upper()
        keys.add(caps)
    return keys

def apply_rename(label):
    """
    Use rename_map if available. Tries original, simplified, and caps/no-colon keys.
    Fallback: return the simplified label (for readability).
    """
    for k in all_keys_for_lookup(label):
        if k in rename_map:
            return rename_map[k]
    return simplify_label_from_original(label)


# ---------------- Parsing / Cleaning ---------------- #

COLUMN_NAMES = ["Frame", "Residue", "Internal", "VDW", "EEL", "PBSOL", "NPSOL", "TOTAL"]
COMPONENTS = [
    ("Internal", "Internal_mean", "Internal_sem"),
    ("VDW",      "VDW_mean",      "VDW_sem"),
    ("EEL",      "EEL_mean",      "EEL_sem"),
    ("PBSOL",    "PBSOL_mean",    "PBSOL_sem"),
    ("NPSOL",    "NPSOL_mean",    "NPSOL_sem"),
    ("TOTAL",    "TOTAL_mean",    "TOTAL_sem"),
]

def parse_tdc_block_to_df(csv_path):
    """
    Locate the TDC block under 'DELTAS:' followed by 'Total Decomposition Contribution (TDC)'.
    Return a cleaned DataFrame with the expected columns.
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()

    start = None
    for i in range(len(lines) - 1):
        if ("DELTAS:" in lines[i]) and ("Total Decomposition Contribution (TDC)" in lines[i + 1]):
            start = i + 4  # data typically starts here
            break

    if start is None:
        raise ValueError("Could not find TDC block in {}".format(csv_path))

    end = None
    for j in range(start, len(lines)):
        if "Sidechain Decomposition Contribution (SDC)" in lines[j]:
            end = j - 1
            break
    if end is None:
        end = len(lines)

    block_lines = lines[start:end]
    df = pd.read_csv(StringIO("".join(block_lines)), names=COLUMN_NAMES)

    # Clean up
    df = df[df["Residue"].notna()]
    df = df[df["Residue"].astype(str).str.strip().ne("")]
    df = df[df["Residue"] != "Residue"]
    df = df[df["Residue"].astype(str).str.strip().str.lower().ne("nan")]

    # numeric
    for col in COLUMN_NAMES[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if exclude_ligand:
        df = df[~df["Residue"].astype(str).str.startswith("L:B:")]

    return df

def per_replica_stats(df):
    """
    Group by Residue and compute mean & sem across frames for each component.
    Returns a DataFrame with columns like 'Internal_mean', 'Internal_sem', ..., 'TOTAL_mean', 'TOTAL_sem'.
    """
    stats = df.groupby("Residue").agg(["mean", "sem"])
    stats.columns = ["_".join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()
    # Drop frame columns if present
    for drop_col in ["Frame_mean", "Frame_sem"]:
        if drop_col in stats.columns:
            stats = stats.drop(columns=[drop_col])
    return stats


# ---------------- Across-replica aggregation ---------------- #

def combine_replica_means(replica_stats_list):
    """
    Given a list of per-replica stats (each has *_mean, *_sem per residue),
    compute the combined mean-of-means and SEM across replicas (on the means).
    We align by 'Residue' using outer join, and average ignoring NaN.
    """
    all_res = pd.DataFrame({"Residue": pd.unique(pd.concat([r[["Residue"]] for r in replica_stats_list], ignore_index=True)["Residue"])})
    combined = all_res.copy()

    for comp, mean_col, _sem_col in COMPONENTS:
        temp = all_res.copy()
        for idx, rstats in enumerate(replica_stats_list):
            if mean_col in rstats.columns:
                temp = temp.merge(
                    rstats[["Residue", mean_col]].rename(columns={mean_col: f"{comp}_mean_rep{idx+1}"}),
                    on="Residue", how="left"
                )
        cols = [c for c in temp.columns if c.startswith(f"{comp}_mean_rep")]
        combined[f"{comp}_mean"] = temp[cols].mean(axis=1, skipna=True)
        std = temp[cols].std(axis=1, ddof=1, skipna=True)
        n   = temp[cols].count(axis=1)
        combined[f"{comp}_sem"] = std / np.sqrt(n.replace(0, np.nan))

    return combined


# ---------------- Sorting helpers ---------------- #

def sort_residues_for_combined(combined_df, keep_top_k):
    """
    Sort by |TOTAL_mean| desc and keep top_k (or all if keep_top_k is None),
    then for plotting sort by signed TOTAL_mean (neg bottom → pos top).
    """
    tmp = combined_df.copy()
    tmp["Abs_TOTAL_mean"] = tmp["TOTAL_mean"].abs()
    tmp = tmp.sort_values("Abs_TOTAL_mean", ascending=False)
    if keep_top_k is not None:
        tmp = tmp.head(int(keep_top_k))
    tmp = tmp.drop(columns=["Abs_TOTAL_mean"])
    tmp = tmp.sort_values("TOTAL_mean").reset_index(drop=True)
    return tmp

def make_order_from_manual(list_or_path):
    if list_or_path is None:
        return None
    if isinstance(list_or_path, str) and os.path.exists(list_or_path):
        with open(list_or_path, "r") as f:
            items = [line.strip() for line in f if line.strip()]
    elif isinstance(list_or_path, (list, tuple)):
        items = list(list_or_path)
    else:
        return None
    buckets = []
    for it in items:
        buckets.append(all_keys_for_lookup(it))
    return buckets

def order_residues(df, mode, combined_sorted=None, manual_buckets=None):
    """
    Returns a reindexed df in the requested order.
    - mode='combined' : order matches 'combined_sorted' Residue order
    - mode='manual'   : order matches first match of each manual bucket
    - mode='per-replica' : sort by TOTAL_mean within df
    """
    if mode == "combined":
        if combined_sorted is None:
            return df.sort_values("TOTAL_mean").reset_index(drop=True)
        order = combined_sorted[["Residue"]].copy()
        out = order.merge(df, on="Residue", how="left")
        # keep even if NaN so x-axis stays identical when requested
        return out.reset_index(drop=True)

    if mode == "manual" and manual_buckets:
        key_map = {}
        for _, row in df.iterrows():
            res = row["Residue"]
            key_map[res] = all_keys_for_lookup(res)

        ordered_rows = []
        used = set()
        for bucket in manual_buckets:
            found = None
            for res, keys in key_map.items():
                if res in used:
                    continue
                if keys & bucket:
                    found = res
                    break
            if found is not None:
                ordered_rows.append(df[df["Residue"] == found])
                used.add(found)

        leftovers = df[~df["Residue"].isin(list(used))]
        if len(ordered_rows) > 0:
            out = pd.concat(ordered_rows + [leftovers], ignore_index=True)
        else:
            out = pd.concat([leftovers], ignore_index=True)
        return out.reset_index(drop=True)

    return df.sort_values("TOTAL_mean").reset_index(drop=True)


# ---------------- Plotting ---------------- #

def plot_components(sorted_df, outdir, drug_name, receptor_name, replica_tag, components=COMPONENTS):
    os.makedirs(outdir, exist_ok=True)

    disp = sorted_df["Residue"].apply(apply_rename).tolist()
    colors = cm.rainbow(np.linspace(0, 1, len(disp)))

    for comp, mean_col, sem_col in components:
        # Keep original values (with NaN) for label-skipping logic
        orig_vals = sorted_df[mean_col].astype(float).copy()
        # Fill for drawing bars and error bars only
        avg_vals = orig_vals.fillna(0.0)
        sem_vals = sorted_df[sem_col].astype(float).fillna(0.0)

        plt.figure(figsize=(12, 6), dpi=300)
        bars = plt.bar(disp, avg_vals, yerr=sem_vals, capsize=8,
                       ecolor="black", color=colors, edgecolor="black")
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.ylabel(r"$\Delta G$ Contribution (kcal/mol)", fontsize=14)
        plt.title(f"{comp} Energy Contributions by Residue ({drug_name} - {receptor_name})\n{replica_tag}", fontsize=14)
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.grid(axis="both", linestyle=":", linewidth=0.7, alpha=0.7)

        ymax = np.nanmax(np.abs(avg_vals + sem_vals)) if len(avg_vals) else 1.0
        if not np.isfinite(ymax) or ymax == 0:
            ymax = 1.0
        plt.ylim(-1.5 * ymax, 1.5 * ymax)

        rel_offset = 0.15 * (1.5 * ymax)
        for bar, value, orig in zip(bars, avg_vals, orig_vals):
            # Skip label if original value was NaN (missing in this replica)
            if np.isnan(orig):
                continue
            y = bar.get_height()
            offset = rel_offset if y >= 0 else -rel_offset
            va = "bottom" if y >= 0 else "top"
            plt.text(bar.get_x() + bar.get_width() / 2, y + offset, "{:.2f}".format(value),
                     ha="center", va=va, fontsize=10, fontweight="bold")

        plt.tight_layout()
        out_png = os.path.join(outdir, f"{comp}_Energy_Contribution.png")
        plt.savefig(out_png, dpi=300)
        plt.close()


# ---------------- Main ---------------- #

def main():
    tags = replica_tags if isinstance(replica_tags, list) and len(replica_tags) == len(csv_paths) \
           else ["Replica{}".format(i+1) for i in range(len(csv_paths))]

    # 1) Per-replica parsing + stats
    per_replica = []
    for path, tag in zip(csv_paths, tags):
        df = parse_tdc_block_to_df(path)
        stats = per_replica_stats(df)
        per_replica.append({"tag": tag, "stats": stats})

    # 2) Combined (mean-of-means across replicas)
    combined_stats = combine_replica_means([p["stats"] for p in per_replica])
    combined_sorted = sort_residues_for_combined(combined_stats, top_k)

    # Save combined table
    os.makedirs(combined_output_dir_base, exist_ok=True)
    combined_csv = os.path.join(combined_output_dir_base, "top_residue_stats.csv")
    combined_sorted.to_csv(combined_csv, index=False)

    # Combined plots
    plot_components(combined_sorted, combined_output_dir_base, drug_name, receptor_name, combined_tag)

    # 3) Per-replica plots + per-replica stats
    manual_buckets = make_order_from_manual(manual_sort_list) if replica_sort_mode == "manual" else None

    for p in per_replica:
        tag = p["tag"]
        stats = p["stats"].copy()

        if replica_sort_mode == "combined":
            ordered = order_residues(
                stats,
                mode="combined",
                combined_sorted=combined_sorted[["Residue"]],
                manual_buckets=None
            )
        else:
            ordered = order_residues(
                stats,
                mode=replica_sort_mode,
                combined_sorted=combined_sorted[["Residue"]],
                manual_buckets=manual_buckets
            )

        if replica_sort_mode == "combined" and force_same_x_as_combined:
            # Reindex to combined residues; keep NaNs so labels can be skipped
            template = combined_sorted[["Residue"]].copy()
            ordered = template.merge(ordered, on="Residue", how="left")
            # Ensure expected columns exist; DO NOT fill NaNs here
            for _, mean_col, sem_col in COMPONENTS:
                if mean_col not in ordered.columns:
                    ordered[mean_col] = np.nan
                if sem_col not in ordered.columns:
                    ordered[sem_col] = np.nan
            ordered_top = ordered.reset_index(drop=True)
        else:
            # Optional per-replica top_k trimming (if desired)
            if top_k is not None:
                tmp = ordered.copy()
                tmp["Abs_TOTAL_mean"] = tmp["TOTAL_mean"].abs()
                tmp = tmp.sort_values("Abs_TOTAL_mean", ascending=False).head(int(top_k))
                tmp = tmp.drop(columns=["Abs_TOTAL_mean"])
                ordered_top = tmp.sort_values("TOTAL_mean").reset_index(drop=True)
            else:
                # Keep all residues, then sort by signed TOTAL for readability
                ordered_top = ordered.sort_values("TOTAL_mean").reset_index(drop=True)

        outdir = per_replica_output_dir_base.format(tag=tag)
        os.makedirs(outdir, exist_ok=True)

        # Save replica's table (NaNs preserved when force_same_x_as_combined=True)
        out_csv = os.path.join(outdir, "top_residue_stats.csv")
        ordered_top.to_csv(out_csv, index=False)

        # Per-replica plots
        plot_components(ordered_top, outdir, drug_name, receptor_name, tag)

    # Helpful console summary
    total_sum = combined_sorted["TOTAL_mean"].sum(skipna=True)
    print("Combined TOTAL_mean sum over {} residues: {:.3f} kcal/mol".format(
        len(combined_sorted), total_sum))
    print("Saved combined stats to: {}".format(combined_csv))
    print("Combined plots in: {}".format(combined_output_dir_base))
    print("Per-replica plots in folders like: {}".format(per_replica_output_dir_base.format(tag="{ReplicaX}")))
    print("Replica sort mode: {}".format(replica_sort_mode))
    print("force_same_x_as_combined:", force_same_x_as_combined)
    print("top_k:", top_k)

if __name__ == "__main__":
    main()
