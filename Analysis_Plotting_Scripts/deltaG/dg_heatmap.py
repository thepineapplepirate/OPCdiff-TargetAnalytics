#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from pathlib import Path

'''
dg_heatmap.py

This script produces heatmaps for gmx_MMPBSA's free energy decomposition analysis (FINAL_DECOMP_MMPBSA.csv).
Two heatmaps are produced: 
    1) total energy contributions per-residue, per-frame, averaged across the replicas
    2) energy component contributions per-residue, averaged across all frames and replicas

The plots are intended to demonstrate how the total energy contributions from each residue are changing throughout 
the trajectories, as well as what the contributions to the component energies are once averaged across the trajectories. 
'''

# ================== CONFIG ==================
replica_csvs = [
    "path/to/replica1/FINAL_DECOMP_MMPBSA.csv",
    "path/to/replica2/FINAL_DECOMP_MMPBSA.csv",
    "path/to/replica3/FINAL_DECOMP_MMPBSA.csv",
]

# OPTIONAL: provide explicit labels (same length/order as replica_csvs)
replica_labels = ["Replica 1", "Replica 2", "Replica 3"]  # set to None to auto-label

drug_name = "Clemastine"
receptor_name = "M1"

output_dir = "across_replica_heatmap_filtered"
os.makedirs(output_dir, exist_ok=True)

# Filtering (applies to the Residue×Frame TOTAL heatmap)
remove_constant_residues = True

# --- Fixed threshold settings (used only if target_n_residues is None) ---
mean_abs_threshold = 0.04          # kcal/mol
use_absolute_for_threshold = True  # True => use |mean|; False => use signed mean

# --- Dynamic selection (Top-K) ---
# Keep exactly this many residues (after constant-row removal). Set to None to use fixed threshold logic.
target_n_residues = 20            # e.g., 40, 60, 80; or None to disable Top-K
use_absolute_for_topk = True      # True => rank by |mean| ; False => rank by signed mean
min_floor_threshold = 0.0         # optional pre-filter floor: drop residues with |mean| < this before Top-K

# Alignment behavior for residues/frames across replicas:
#   "union"        -> include any residue/frame that appears in at least one replica
#   "intersection" -> include only residues/frames present in ALL replicas
residue_set_mode = "union"         # "union" or "intersection"
frame_set_mode   = "union"         # "union" or "intersection"

# Plot settings
cmap_center = 0.0
dpi = 600

# Renaming for display only (prevents collisions during computation)
simplify_labels_for_display = True
rename_json_path = "../ortho_resname_dict.json"   # set None to disable renaming

# Component heatmap: optionally drop some components from the plot
# Valid component keys: "Internal", "VDW", "EEL", "PBSOL", "NPSOL", "TOTAL"
drop_components_in_component_heatmap = ["Internal", "NPSOL"]  # drop zeros if desired
# ============================================


# ----------------- Helpers ------------------
def assign_replica_labels(csv_paths, replica_labels=None, prefix="Run"):
    if replica_labels is not None:
        if len(replica_labels) != len(csv_paths):
            raise ValueError("replica_labels must have same length as replica_csvs.")
        if len(set(replica_labels)) != len(replica_labels):
            raise ValueError("replica_labels must be unique.")
        return list(replica_labels)
    stems = [Path(p).stem.strip() for p in csv_paths]
    stems_ok = all(stems) and (len(set(stems)) == len(stems))
    if stems_ok:
        return stems
    return [f"{prefix}{i+1}" for i in range(len(csv_paths))]


def read_tdc_block(csv_path: str) -> pd.DataFrame:
    with open(csv_path, "r") as f:
        lines = f.readlines()
    start = None
    for i in range(len(lines) - 1):
        if "DELTAS:" in lines[i] and "Total Decomposition Contribution (TDC)" in lines[i + 1]:
            start = i + 4
            break
    if start is None:
        raise ValueError(f"Could not find DELTAS/TDC block in {csv_path}")
    end = None
    for j in range(start, len(lines)):
        if "Sidechain Decomposition Contribution (SDC)" in lines[j]:
            end = j - 1
            break
    if end is None:
        end = len(lines)
    block = StringIO("".join(lines[start:end]))
    cols = ["Frame", "Residue", "Internal", "VDW", "EEL", "PBSOL", "NPSOL", "TOTAL"]
    df = pd.read_csv(block, names=cols)
    # Clean
    df = df[df["Residue"].notna()]
    df = df[df["Residue"].astype(str).str.strip().ne("")]
    df = df[df["Residue"] != "Residue"]
    df = df[~df["Residue"].astype(str).str.startswith("L:B:")]   # exclude ligand
    for c in ["Internal", "VDW", "EEL", "PBSOL", "NPSOL", "TOTAL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    return df


def simplify_label(residue_str: str) -> str:
    s = str(residue_str).replace("R:A:", "").replace("L:B:", "")
    parts = s.split(":")
    if len(parts) == 2:
        return f"{parts[0].capitalize()}{parts[1]}"
    return s


def load_rename_map(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def apply_display_rename(label: str, rename_map: dict) -> str:
    if not rename_map:
        return label
    if label in rename_map:
        return rename_map[label]
    uc = label.replace(":", "").upper()
    return rename_map.get(uc, label)


def display_name_from_original(orig_id: str, rename_map: dict) -> str:
    lbl = simplify_label(orig_id) if simplify_labels_for_display else orig_id
    return apply_display_rename(lbl, rename_map)


def three_col_df(res_list, rename_map, base_row_means: pd.Series) -> pd.DataFrame:
    means = [base_row_means.get(r, np.nan) for r in res_list]
    return pd.DataFrame({
        "Residue": res_list,
        "New Name": [display_name_from_original(r, rename_map) for r in res_list],
        "MeanAcrossFrames": means
    })


def per_replica_residue_means(df: pd.DataFrame) -> pd.DataFrame:
    means = df.groupby("Residue")[["Internal", "VDW", "EEL", "PBSOL", "NPSOL", "TOTAL"]].mean(numeric_only=True)
    means = means.rename(columns={c: f"{c}_mean" for c in means.columns}).reset_index()
    return means


def add_new_name_column(df: pd.DataFrame, rename_map: dict, residue_col: str = "Residue") -> pd.DataFrame:
    """Return a copy of df with a 'New Name' column added next to Residue."""
    out = df.copy()
    out.insert(out.columns.get_loc(residue_col) + 1, "New Name",
               [display_name_from_original(r, rename_map) for r in out[residue_col]])
    return out
# --------------------------------------------


# --------- Assign replica names (robust) ---------
replica_names = assign_replica_labels(replica_csvs, replica_labels)

# Save a mapping so you can audit later
pd.DataFrame({"Replica": replica_names, "Path": replica_csvs}).to_csv(
    os.path.join(output_dir, "replica_label_map.csv"), index=False
)

# --------- Load replicas (no renaming yet for computation) ---------
replica_frames = []   # raw per-frame per-residue for each replica
replica_pivots = []   # Residue×Frame pivot of TOTAL per replica

for csv_path, rep_name in zip(replica_csvs, replica_names):
    df = read_tdc_block(csv_path)
    df["Replica"] = rep_name
    replica_frames.append(df)
    piv = df.pivot(index="Residue", columns="Frame", values="TOTAL")
    replica_pivots.append(piv)

# --------- Align residue and frame sets explicitly ---------
if residue_set_mode == "intersection":
    residues_all = set(replica_pivots[0].index)
    for p in replica_pivots[1:]:
        residues_all &= set(p.index)
else:
    residues_all = set().union(*[set(p.index) for p in replica_pivots])

if frame_set_mode == "intersection":
    frames_all = set(replica_pivots[0].columns)
    for p in replica_pivots[1:]:
        frames_all &= set(p.columns)
else:
    frames_all = set().union(*[set(p.columns) for p in replica_pivots])

residues_all = sorted(residues_all)
frames_all = sorted(frames_all)

aligned = [p.reindex(index=residues_all, columns=frames_all) for p in replica_pivots]

# --------- Mean across replicas per (Residue, Frame) (TOTAL) ---------
stacked = np.stack([a.values for a in aligned], axis=0)  # (replicas, residues, frames)
mean_over_reps = np.nanmean(stacked, axis=0)             # (residues, frames)

# Base matrix BEFORE filtering (for consistent means)
base_df = pd.DataFrame(mean_over_reps, index=residues_all, columns=frames_all)
base_row_means = base_df.mean(axis=1, skipna=True)  # used for selection, printouts & CSVs

# Start from base_df for filtering; renumber frames 1..N for display
heatmap_df = base_df.copy()
heatmap_df.columns = range(1, len(heatmap_df.columns) + 1)

# Also save a base matrix CSV with New Name added
rename_map = load_rename_map(rename_json_path)
base_with_names = add_new_name_column(base_df.reset_index().rename(columns={"index": "Residue"}), rename_map)
base_with_names.to_csv(os.path.join(output_dir, "heatmap_base.csv"), index=False)

# --------- Filtering / Selection (TOTAL only) ---------
constant_residues = []
if remove_constant_residues:
    for res in heatmap_df.index:
        row = heatmap_df.loc[res].values
        if np.all(np.isnan(row)) or (np.nanmin(row) == np.nanmax(row)):
            constant_residues.append(res)
    heatmap_df = heatmap_df.drop(index=constant_residues, errors="ignore")

# Build candidate set after constant removal
candidates = pd.Index(heatmap_df.index)

selection_mode = "fixed_threshold"
dyn_thresh_score = None
dyn_threshold_expr = None
dropped_floor = pd.Index([])

if target_n_residues is not None:
    selection_mode = "top_k"
    # Use base_row_means so selection is stable and comparable
    vals = base_row_means.loc[candidates]

    # Optional floor: drop tiny contributors before Top-K (by |mean|)
    if min_floor_threshold > 0:
        keep_floor = vals.abs() >= min_floor_threshold
        dropped_floor = candidates[~keep_floor]
        candidates = candidates[keep_floor]
        heatmap_df = heatmap_df.loc[candidates]
        vals = vals.loc[candidates]

    if candidates.size == 0:
        print("\nWARNING: No residues remain after constant-removal and floor filter.")
        dropped_by_threshold = list(dropped_floor)
    else:
        # Score for ranking
        scores = vals.abs() if use_absolute_for_topk else vals
        order = scores.sort_values(ascending=False)

        if order.size <= target_n_residues:
            top_idx = order.index
            print(f"\nNOTE: Only {order.size} residues available; keeping all (requested {target_n_residues}).")
        else:
            top_idx = order.index[:target_n_residues]

        # K-th score (effective cutoff)
        dyn_thresh_score = float(order.iloc[min(target_n_residues-1, order.size-1)])
        dyn_threshold_expr = (f"|mean| ≥ {dyn_thresh_score:.6f} kcal/mol"
                              if use_absolute_for_topk else
                              f"mean ≥ {dyn_thresh_score:.6f} kcal/mol")

        dropped_by_threshold = [r for r in candidates if r not in set(top_idx)]
        heatmap_df = heatmap_df.loc[top_idx]

        # Explicit printout of the dynamic threshold used
        mode_label = "|mean|" if use_absolute_for_topk else "mean"
        print(f"\n[Top-K] Keeping K={len(top_idx)} residues by {mode_label}.")
        print(f"[Top-K] Dynamic threshold chosen: {dyn_threshold_expr}")
        if min_floor_threshold > 0:
            print(f"[Top-K] Applied floor: |mean| ≥ {min_floor_threshold:.6f} kcal/mol "
                  f"(dropped {len(dropped_floor)})")

else:
    # Fixed threshold path (original behavior)
    if use_absolute_for_threshold:
        keep_mask = base_row_means.abs() >= mean_abs_threshold
        threshold_str = f"|mean| ≥ {mean_abs_threshold} kcal/mol"
    else:
        keep_mask = base_row_means >= mean_abs_threshold
        threshold_str = f"mean ≥ {mean_abs_threshold} kcal/mol"

    dropped_by_threshold = [r for r in heatmap_df.index if not keep_mask.get(r, False)]
    heatmap_df = heatmap_df.loc[[r for r in heatmap_df.index if keep_mask.get(r, False)]]
    print(f"\n[Fixed threshold] Used threshold: {threshold_str}")
    dyn_threshold_expr = threshold_str  # for audit file

# Final sort (descending by across-frame mean) for plotting order
heatmap_df = heatmap_df.loc[heatmap_df.mean(axis=1).sort_values(ascending=False).index]

# --------- DISPLAY labels for plots ---------
orig_ids = heatmap_df.index.tolist()
display_labels = [display_name_from_original(r, rename_map) for r in orig_ids]

# --------- Screen printout summary ---------
print("\n=== Filtering / Selection summary ===")
print(f"Residue set mode: {residue_set_mode}, Frame set mode: {frame_set_mode}")
print(f"Constant-value removal: {remove_constant_residues} (removed {len(constant_residues)})")

if selection_mode == "top_k":
    mode_label = "|mean|" if use_absolute_for_topk else "mean"
    print(f"[Top-K] Requested K={target_n_residues}, kept {len(orig_ids)} residues by {mode_label}.")
    if dyn_thresh_score is not None:
        print(f"[Top-K] K-th score = {dyn_thresh_score:.6f} kcal/mol")
    if min_floor_threshold > 0:
        print(f"[Top-K] Applied floor: |mean| >= {min_floor_threshold:.6f} kcal/mol "
              f"(dropped {len(dropped_floor)})")
else:
    used = "|mean| >= " + str(mean_abs_threshold) if use_absolute_for_threshold \
           else "mean >= " + str(mean_abs_threshold)
    print(f"[Fixed threshold] Used threshold: {used}")

print("\nDropped constant-value residues:")
print("   None" if not constant_residues else "".join([f"\n   {r}" for r in constant_residues]))

print("\nDropped by selection rule:")
print("   None" if not dropped_by_threshold else "".join([f"\n   {r}" for r in dropped_by_threshold]))

# --------- Save dropped/kept CSVs (Residue, New Name, MeanAcrossFrames) ---------
three_col_df(constant_residues, rename_map, base_row_means).to_csv(
    os.path.join(output_dir, "removed_constant_residues.csv"), index=False
)
three_col_df(dropped_by_threshold, rename_map, base_row_means).to_csv(
    os.path.join(output_dir, "removed_by_selection.csv"), index=False
)
three_col_df(list(heatmap_df.index), rename_map, base_row_means).to_csv(
    os.path.join(output_dir, "kept_residues.csv"), index=False
)

# --------- Write a small audit file about selection ---------
audit_lines = []
audit_lines.append("=== Selection Audit ===")
audit_lines.append(f"Mode: {selection_mode}")
if selection_mode == "top_k":
    audit_lines.append(f"Requested K: {target_n_residues}")
    audit_lines.append(f"Kept: {len(orig_ids)}")
    audit_lines.append(f"Use absolute for Top-K: {use_absolute_for_topk}")
    if min_floor_threshold > 0:
        audit_lines.append(f"Floor: |mean| >= {min_floor_threshold:.6f} kcal/mol (dropped {len(dropped_floor)})")
    if dyn_thresh_score is not None:
        audit_lines.append(f"K-th score: {dyn_thresh_score:.6f} kcal/mol")
        audit_lines.append(f"Dynamic threshold chosen: {dyn_threshold_expr}")

else:
    audit_lines.append(f"Fixed threshold: {'|mean|' if use_absolute_for_threshold else 'mean'} >= {mean_abs_threshold}")

audit_lines.append("\nKept residues (in plot order):")
for r in orig_ids:
    audit_lines.append(f"  {r}")

audit_lines.append("\nDropped by selection rule:")
for r in dropped_by_threshold:
    audit_lines.append(f"  {r}")

with open(os.path.join(output_dir, "selection_audit.txt"), "w") as f:
    f.write("\n".join(audit_lines))

# --------- Heatmap 1: Residue × Frame (across-replica mean TOTAL) ---------
plt.figure(figsize=(15.5, max(6, 0.35 * len(heatmap_df))), dpi=dpi)
ax = sns.heatmap(
    heatmap_df,
    cmap="coolwarm",
    center=cmap_center,
    cbar_kws={"label": r"$\Delta G_{\mathrm{TOTAL}}$ (kcal/mol)"},
    yticklabels=display_labels,
)
# Adjust legend (colorbar) fonts
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_size(22)   # label font size
cbar.ax.tick_params(labelsize=22)  # tick font size
ax.set_title(
    f"Mean Per-Frame Total Energy Contributions Across Replicas \n({receptor_name} - {drug_name})",
    fontsize=22
)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18) 
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xlabel("Frame", fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmap_total_by_frame_mean_filtered.png"), dpi=dpi)
plt.close()

# --------- Per-replica per-residue means (for other two heatmaps & CSVs) ---------
per_rep_means = []
for (df, rep_name) in zip(replica_frames, replica_names):
    means = per_replica_residue_means(df)            # Residue + *_mean
    means["Replica"] = rep_name
    per_rep_means.append(means)
long_df = pd.concat(per_rep_means, axis=0, ignore_index=True)

# Add New Name to per_replica_means.csv
long_df_named = add_new_name_column(long_df, rename_map, residue_col="Residue")
long_df_named.to_csv(os.path.join(output_dir, "summary_by_replica.csv"), index=False)

# Across-replica summary of component means per residue
group_cols = ["Internal_mean", "VDW_mean", "EEL_mean", "PBSOL_mean", "NPSOL_mean", "TOTAL_mean"]
summary = long_df.groupby("Residue")[group_cols].agg(["mean", "sem"])
summary.columns = ["_".join(c) for c in summary.columns]  # e.g., 'VDW_mean_mean'
summary = summary.reset_index()

# Reuse filtered/sorted residue order
summary = summary[summary["Residue"].isin(orig_ids)]
summary = summary.set_index("Residue").loc[orig_ids].reset_index()

# Add New Name to summary_across_replicas.csv
summary_named = add_new_name_column(summary, rename_map, residue_col="Residue")
summary_named.to_csv(os.path.join(output_dir, "summary_across_replicas.csv"), index=False)

# --------- Heatmap 2: Residue × Replica (per-replica TOTAL means) ---------
total_by_rep = long_df.pivot_table(index="Residue", columns="Replica", values="TOTAL_mean", aggfunc="mean")
total_by_rep = total_by_rep.reindex(index=orig_ids)

# Save total_by_replica.csv with New Name included
total_by_rep_named = total_by_rep.reset_index().rename(columns={"index": "Residue"})
total_by_rep_named = add_new_name_column(total_by_rep_named, rename_map, residue_col="Residue")
total_by_rep_named.to_csv(os.path.join(output_dir, "total_by_replica.csv"), index=False)

plt.figure(figsize=(15.5, max(6, 0.35 * len(orig_ids))), dpi=dpi)
ax = sns.heatmap(
    total_by_rep.values,
    cmap="coolwarm",
    center=0.0,
    cbar_kws={"label": r"$\Delta G_{\mathrm{TOTAL}}$ (kcal/mol)"},
    yticklabels=[display_name_from_original(r, rename_map) for r in orig_ids],
    xticklabels=total_by_rep.columns.tolist(),  # uses explicit replica labels
)
# Adjust legend (colorbar) fonts
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_size(22)   # label font size
cbar.ax.tick_params(labelsize=22)  # tick font size
ax.set_title(f"Mean Per-Residue Total Energy Contributions Across Replicas \n({receptor_name} - {drug_name})", fontsize=22)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmap_total_by_replica.png"), dpi=dpi)
plt.close()

# --------- Heatmap 3: Residue × Component (across-replica mean of means) ---------
component_order = ["Internal", "VDW", "EEL", "PBSOL", "NPSOL", "TOTAL"]
component_order = [c for c in component_order if c not in drop_components_in_component_heatmap]
colmap = {c: f"{c}_mean_mean" for c in ["Internal", "VDW", "EEL", "PBSOL", "NPSOL", "TOTAL"]}
# (Better: rebuild comp_cols from summary columns directly)
comp_cols = [colmap[c] for c in component_order if colmap[c] in summary.columns]


comp_mat = summary.set_index("Residue").loc[orig_ids, comp_cols].values

plt.figure(figsize=(15.5, max(6, 0.35 * len(orig_ids))), dpi=dpi)
ax = sns.heatmap(
    comp_mat,
    cmap="coolwarm",
    center=0.0,
    cbar_kws={"label": "Energy (kcal/mol)"},
    yticklabels=[display_name_from_original(r, rename_map) for r in orig_ids],
    xticklabels=[f"$\\Delta G_{{{c}}}$" if c != "TOTAL" else "$\\Delta G_{\\mathrm{TOTAL}}$" for c in component_order],
)
# Adjust legend (colorbar) fonts
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_size(22)   # label font size
cbar.ax.tick_params(labelsize=22)  # tick font size
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_title(f"Mean Per-Residue Energy Components Across Replicas \n({receptor_name} - {drug_name})", fontsize=22)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmap_components_across_replicas_mean.png"), dpi=dpi)
plt.close()

print("\nSaved figures:")
print(" - heatmap_total_by_frame_mean_filtered.png")
print(" - heatmap_total_by_replica.png")
print(" - heatmap_components_across_replicas_mean.png")
print("\nSaved tables:")
print(" - replica_label_map.csv")
print(" - heatmap_base.csv")
print(" - removed_constant_residues.csv")
print(" - removed_by_selection.csv")
print(" - kept_residues.csv")
print(" - summary_by_replica.csv")
print(" - summary_across_replicas.csv")
print(" - total_by_replica.csv")
print(" - selection_audit.txt")

