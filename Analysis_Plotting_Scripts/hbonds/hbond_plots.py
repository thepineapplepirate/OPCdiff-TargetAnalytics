#!/usr/bin/env python3
"""
hbond_plots.py 

Plot data from Galaxy's "Hydrogen Bond Analysis using VMD" tool.
The script plots several ways of measuring the bonds formed between two 
components of your system (for example between a ligand and nearby protein residues),
averaged across replica simulations. 

Features
- Supports 1+ replicas: averages time series and occupancies across replicas.
- User-selectable error bars:
    * OCC_ERR: "SD" or "SEM" (occupancy bar plot)
    * TS_ERR:  "SD" or "SEM" (time-series shaded band)
- Occupancy error bars: lower side clipped to 0; upper side NOT clipped.
- Canonicalization step helps merge donor/acceptor names across replicas.
- Replica colors: R1=black, R2=blue, R3=red (cycle thereafter); Mean=solid green.
- Global font sizes & optional global y-limits (applied to all plots except Top-20 occupancy).
- Moving-average plot shows an annotation with the final MA mean value.
- Plot-only residue renames for occupancy labels (e.g., UNL445 -> Ligand).

Inputs
  Each replica dict:
    num -> "Number of H-bonds.txt" (frame vs count)
    occ -> "Percentage occupancy of the H-bond.txt" (donor, acceptor, occupancy %)
    vmd -> (optional) VMD log

Outputs
  hbond_timeseries_all_replicas.png
  hbond_timeseries_mean.png
  hbond_timeseries_mean_mavg.png
  hbond_occupancy_top20_mean_SD.png / _SEM.png
  hbond_occupancy_aggregated.csv
  hbond_summary.json
"""

from pathlib import Path
import re
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Global styling (font sizes) ----------
TITLE_FONTSIZE  = 18
LABEL_FONTSIZE  = 16
TICK_FONTSIZE   = 14
LEGEND_FONTSIZE = 14
DPI = 600

plt.rcParams.update({
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})

# =================== CONFIG (edit these) ===================

# the inputs below are a list of dictionaries pointing to the paths of each output file of the tool
# from each replica simulation.
INPUTS = [
    {"num": "eplica1/galaxy_analysis/Number of H-bonds.txt",
    "occ": "replica1/galaxy_analysis/Percentage occupancy of the H-bond.txt",
    "vmd": "replica1/galaxy_analysis/Hydrogen Bond Analysis using VMD on data 68 and data 63.txt"},
    {"num": "replica2/galaxy_analysis/Number of H-bonds.txt",
     "occ": "replica2/galaxy_analysis/Percentage occupancy of the H-bond.txt",
     "vmd": "replica2/galaxy_analysis/Hydrogen Bond Analysis using VMD on data 26 and data 21.txt"},
    {"num": "replica3/galaxy_analysis/Number of H-bonds.txt",
     "occ": "replica3/galaxy_analysis/Percentage occupancy of the H-bond.txt",
     "vmd": "replica3/galaxy_analysis/Hydrogen Bond Analysis using VMD on data 25 and data 20.txt"},
]

drug_name     = "Clemastine"
receptor_name = "M1"

# Error bar modes: "SD" or "SEM"
OCC_ERR = "SEM"   # occupancy bar plot error bars
TS_ERR  = "SD"    # time-series shaded band

# Optional JSON file with residue renames mapping (keys like "Phe166", dataset uses "PHE166").
RENAMES_JSON = "ortho_resname_dict.json"  # e.g., "/path/to/renames.json"

# Plot-only label replacements for occupancy figure (does NOT affect CSV/JSON values)
# e.g., {"UNL445": "Ligand", "TYR87": "Tyr87"}
RESIDUE_RENAMES = {
     "UNL445-Side": drug_name[:5],
}

# Global Y-axis limits for ALL plots EXCEPT the Top-20 occupancy bar plot.
# Set to None (auto) or (ymin, ymax), e.g. (0, 20)
Y_AXIS_LIMITS = None

# Annotate the moving-average plot with the final MA mean
ANNOTATE_MA_MEAN = True
ANNOTATE_MA_FMT  = "{:.2f}"      # number format on plot
ANNOTATE_MA_XY   = (0.02, 0.95)  # axes fraction (left, top)
# ===========================================================

# ---------- Helpers ----------
def load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delim_whitespace=True, header=None, names=["frame", "hbonds"])
    df = df[pd.to_numeric(df["frame"], errors="coerce").notna() & pd.to_numeric(df["hbonds"], errors="coerce").notna()]
    df["frame"] = df["frame"].astype(int)
    df["hbonds"] = df["hbonds"].astype(float)
    return df

def load_occupancy(path: Path) -> pd.DataFrame:
    raw = Path(path).read_text(errors="ignore").strip().splitlines()
    rows = []
    for line in raw:
        if not line.strip():
            continue
        if line.lower().startswith("found ") or line.lower().startswith("donor"):
            continue
        parts = re.split(r"\s{2,}|\t", line.strip())
        if len(parts) < 3:
            parts = re.split(r"\t", line.strip())
        if len(parts) >= 3:
            donor = parts[0].strip()
            acceptor = parts[1].strip()
            occ_str = parts[2].strip().replace("%","")
            try:
                occ = float(occ_str)
            except ValueError:
                continue
            rows.append((donor, acceptor, occ))
    df = pd.DataFrame(rows, columns=["donor","acceptor","occupancy_pct"])
    df["pair"] = df["donor"] + " \u2192 " + df["acceptor"]
    return df

def canonicalize_pair_strings(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s: str) -> str:
        s = " ".join(str(s).split())
        s = s.replace(":", "")
        return s
    out = df.copy()
    out["donor"] = out["donor"].map(norm)
    out["acceptor"] = out["acceptor"].map(norm)
    out["pair"] = out["donor"] + " \u2192 " + out["acceptor"]
    return out

def parse_vmd_meta(path: Path) -> dict:
    meta = {}
    if not path or not Path(path).exists():
        return meta
    txt = Path(path).read_text(errors="ignore")
    m = re.search(r"Final frame:\s*(\d+)", txt)
    if m: meta["final_frame"] = int(m.group(1))
    m = re.search(r"Initial frame:\s*(\d+)", txt)
    if m: meta["initial_frame"] = int(m.group(1))
    m = re.search(r"Donor-Acceptor distance:\s*([0-9.]+)", txt)
    if m: meta["distance_cutoff"] = float(m.group(1))
    m = re.search(r"Angle cutoff:\s*([0-9.]+)", txt)
    if m: meta["angle_cutoff"] = float(m.group(1))
    sel1 = re.search(r"Atomselection 1:\s*(.+)", txt)
    sel2 = re.search(r"Atomselection 2:\s*(.+)", txt)
    if sel1: meta["selection1"] = sel1.group(1).strip()
    if sel2: meta["selection2"] = sel2.group(1).strip()
    return meta

def moving_average(series: pd.Series, frac=0.01, min_window=5):
    window = max(min_window, int(len(series) * frac))
    return series.rolling(window=window, center=True, min_periods=max(1, window//2)).mean(), window

def ensure_list_inputs(INPUTS):
    if isinstance(INPUTS, dict):
        return [INPUTS]
    elif isinstance(INPUTS, list):
        return INPUTS
    else:
        raise ValueError("INPUTS must be a dict or a list of dicts")

# --- Renaming helpers (JSON keys like "Phe166" -> dataset tokens "PHE166") ---
def build_rename_map(renames_dict: dict):
    patterns = []
    for src, dst in renames_dict.items():
        m = re.match(r"([A-Za-z]{3})(\d+)$", str(src))
        if not m:
            continue
        dataset_form = f"{m.group(1)[:3].upper()}{m.group(2)}"  # e.g., PHE166
        pat = re.compile(rf"\b{re.escape(dataset_form)}\b")
        patterns.append((pat, str(dst)))
    return patterns

def apply_renames(text: str, patterns):
    if not patterns or not isinstance(text, str):
        return text
    out = text
    for pat, repl in patterns:
        out = pat.sub(repl, out)
    return out

# ---------- Main ----------
def main():
    replicas = ensure_list_inputs(INPUTS)

    times_list = []
    occ_list = []
    metas = []

    # Load renames mapping if provided
    rename_patterns = []
    if RENAMES_JSON:
        try:
            data = json.loads(Path(RENAMES_JSON).read_text())
            if isinstance(data, dict):
                rename_patterns = build_rename_map(data)
        except Exception as e:
            print(f"Warning: failed to read RENAMES_JSON '{RENAMES_JSON}': {e}")

    for i, cfg in enumerate(replicas, start=1):
        num_path = Path(cfg["num"])
        occ_path = Path(cfg["occ"])
        vmd_path = cfg.get("vmd", "")

        if not num_path.exists():
            raise FileNotFoundError(f"Replica {i}: timeseries file not found: {num_path}")
        if not Path(occ_path).exists():
            raise FileNotFoundError(f"Replica {i}: occupancy file not found: {occ_path}")

        tdf = load_timeseries(num_path).copy()
        tdf = tdf.rename(columns={"hbonds": f"hbonds_rep{i}"})
        times_list.append(tdf)

        odf = load_occupancy(occ_path).copy()
        odf = canonicalize_pair_strings(odf)
        if rename_patterns:
            odf["donor_renamed"] = odf["donor"].apply(lambda s: apply_renames(s, rename_patterns))
            odf["acceptor_renamed"] = odf["acceptor"].apply(lambda s: apply_renames(s, rename_patterns))
            odf["pair_renamed"] = odf["donor_renamed"] + " \u2192 " + odf["acceptor_renamed"]
        else:
            odf["donor_renamed"] = odf["donor"]
            odf["acceptor_renamed"] = odf["acceptor"]
            odf["pair_renamed"] = odf["pair"]
        odf["replica"] = i
        occ_list.append(odf)

        metas.append(parse_vmd_meta(vmd_path))

    # ===== Timeseries aggregation =====
    if times_list:
        merged = times_list[0]
        for t in times_list[1:]:
            merged = merged.merge(t, on="frame", how="outer")
        merged = merged.sort_values("frame").reset_index(drop=True)

        rep_cols = [c for c in merged.columns if c.startswith("hbonds_rep")]
        merged["hbonds_mean"] = merged[rep_cols].mean(axis=1, skipna=True)
        merged["hbonds_sd"]   = merged[rep_cols].std(axis=1, ddof=1, skipna=True)
        n_per_frame = merged[rep_cols].notna().sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            merged["hbonds_sem"] = merged["hbonds_sd"] / np.sqrt(n_per_frame.replace(0, np.nan))
        merged.loc[n_per_frame <= 1, ["hbonds_sd", "hbonds_sem"]] = 0.0

        # Plot all replicas + mean (no moving average)
        plt.figure(figsize=(10,5))
        replica_colors = ["black", "blue", "red"]
        for i, c in enumerate(rep_cols):
            color = replica_colors[i % len(replica_colors)]
            plt.plot(merged["frame"], merged[c], color=color, linewidth=0.8, alpha=0.9, label=f"Replica {i+1}")
        plt.plot(merged["frame"], merged["hbonds_mean"], color="green", linewidth=2.0, label="Mean", zorder=3)
        plt.title(f"Hydrogen bonds per frame \n {receptor_name} - {drug_name} Complex")
        plt.xlabel("Frame"); plt.ylabel("H-bonds")
        if Y_AXIS_LIMITS: plt.ylim(*Y_AXIS_LIMITS)
        if len(rep_cols) <= 8: plt.legend(ncol=2, fontsize=8)
        plt.tight_layout(); plt.savefig("hbond_timeseries_all_replicas.png", dpi=DPI); plt.close()

        # SD/SEM shaded band + moving average + annotation
        err_col  = "hbonds_sd" if TS_ERR.upper() == "SD" else "hbonds_sem"
        label_err = "SD" if TS_ERR.upper() == "SD" else "SEM"

        plt.figure(figsize=(10,5))
        x = merged["frame"].to_numpy()
        y = merged["hbonds_mean"].to_numpy(dtype=float)
        e = merged[err_col].to_numpy(dtype=float)
        mask = ~(np.isnan(y) | np.isnan(e))
        plt.plot(x[mask], y[mask], linewidth=1.8, label="mean")
        plt.fill_between(x[mask], (y - e)[mask], (y + e)[mask], alpha=0.25, label=f"±1 {label_err}")
        ma, win = moving_average(merged["hbonds_mean"])
        plt.plot(merged["frame"], ma, linewidth=1.8, label=f"moving avg (window={win})")

        # Final means
        final_mean_raw = float(np.nanmean(merged["hbonds_mean"].to_numpy()))
        final_mean_mavg = float(np.nanmean(ma.to_numpy()))
        final_mean_reported = final_mean_mavg

        # Annotate on plot
        if ANNOTATE_MA_MEAN:
            text_str = f"Final MA mean: {ANNOTATE_MA_FMT.format(final_mean_reported)}"
            ax = plt.gca()
            ax.text(ANNOTATE_MA_XY[0], ANNOTATE_MA_XY[1], text_str,
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        cutoffs = {(m.get("distance_cutoff"), m.get("angle_cutoff")) for m in metas if m}
        title = f"Hydrogen bonds per frame ± {label_err} \n {receptor_name} - {drug_name} Complex"
        if len(cutoffs) == 1:
            dc, ang = list(cutoffs)[0]
            if dc is not None and ang is not None:
                title += f" \n (cutoffs: {dc} Å, {ang}°)"
        plt.title(title); plt.xlabel("Frame"); plt.ylabel("H-bonds")
        if Y_AXIS_LIMITS: plt.ylim(*Y_AXIS_LIMITS)
        plt.legend(); plt.tight_layout(); plt.savefig("hbond_timeseries_mean_mavg.png", dpi=DPI); plt.close()

        # Print
        print(f"Final mean H-bonds across replicas & frames (moving-average based): {final_mean_reported:.3f}")
        print(f"(Raw mean of hbonds_mean across frames: {final_mean_raw:.3f})")

        # Mean-only plot
        plt.figure(figsize=(10,4))
        plt.plot(merged["frame"], merged["hbonds_mean"], linewidth=1.5, color="green")
        plt.title(f"Hydrogen bonds per frame — mean across replicas \n {receptor_name} - {drug_name} Complex")
        plt.xlabel("Frame"); plt.ylabel("H-bonds (mean)")
        if Y_AXIS_LIMITS: plt.ylim(*Y_AXIS_LIMITS)
        plt.tight_layout(); plt.savefig("hbond_timeseries_mean.png", dpi=DPI); plt.close()

    # ===== Occupancy aggregation =====
    if occ_list:
        occ_all = pd.concat(occ_list, ignore_index=True)
        pivot = occ_all.pivot_table(index=["donor","acceptor","pair"],
                                    columns="replica",
                                    values="occupancy_pct",
                                    aggfunc="mean")
        values   = pivot.values.astype(float)
        mean_vals = np.nanmean(values, axis=1)
        sd_vals   = np.nanstd(values, axis=1, ddof=1)
        n_vals    = np.sum(~np.isnan(values), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sem_vals = sd_vals / np.sqrt(n_vals)

        sd_vals  = np.where(n_vals <= 1, 0.0, sd_vals)
        sem_vals = np.where(n_vals <= 1, 0.0, sem_vals)

        agg = pivot.reset_index().copy()
        label_cols = (pd.concat(occ_list, ignore_index=True)
                      .drop_duplicates(subset=["donor","acceptor","pair"])
                      [["donor","acceptor","pair","donor_renamed","acceptor_renamed","pair_renamed"]])
        agg = agg.merge(label_cols, on=["donor","acceptor","pair"], how="left")

        agg["mean_pct"] = mean_vals
        agg["sd_pct"]   = sd_vals
        agg["sem_pct"]  = sem_vals
        agg["n"]        = n_vals

        agg_sorted = agg.sort_values("mean_pct", ascending=False).reset_index(drop=True)
        agg_sorted.to_csv("hbond_occupancy_aggregated.csv", index=False)

        # Error metric for Top-20
        use_sem = OCC_ERR.upper() == "SEM"
        err_label = "SEM" if use_sem else "SD"
        topn = agg_sorted.head(20).copy()

        y = np.arange(len(topn))
        mean_vals_top = topn["mean_pct"].to_numpy(dtype=float)
        err_vals_top  = (topn["sem_pct"] if use_sem else topn["sd_pct"]).to_numpy(dtype=float)
        n_vals_top    = topn["n"].to_numpy(dtype=int)

        mean_vals_top = np.nan_to_num(mean_vals_top, nan=0.0)
        err_vals_top  = np.nan_to_num(err_vals_top,  nan=0.0)

        lower_cap = np.maximum(mean_vals_top, 0.0)
        lower_err = np.minimum(err_vals_top, lower_cap)
        upper_err = np.clip(err_vals_top, 0.0, None)
        xerr = np.vstack([lower_err, upper_err])

        # Prefer renamed labels when available
        label_series = topn["pair_renamed"].where(
            topn["pair_renamed"].notna() & (topn["pair_renamed"] != ""),
            topn["pair"]
        )

        # --- Plot-only residue renames (does not alter CSV/JSON) ---
        if RESIDUE_RENAMES:
            def _replace_residues(label: str) -> str:
                if not isinstance(label, str):
                    return label
                for old, new in RESIDUE_RENAMES.items():
                    label = label.replace(old, new)
                return label
            label_series = label_series.apply(_replace_residues)

        plt.figure(figsize=(11,6))
        pair_labels = [f"{p}  (n={k})" for p, k in zip(label_series.values, n_vals_top)]
        plt.barh(y, mean_vals_top, xerr=xerr, capsize=3)
        plt.yticks(y, pair_labels)
        plt.gca().invert_yaxis()
        plt.xlabel(f"Occupancy (%) — mean ± {err_label} (upper side not clipped)")
        plt.title(f"Top-20 hydrogen-bond occupancies (across replicas) \n {receptor_name} - {drug_name} Complex")
        plt.tight_layout()
        plt.savefig(f"hbond_occupancy_top20_mean_{err_label}.png", dpi=DPI)
        plt.close()

        unique_n, counts = np.unique(n_vals_top, return_counts=True)
        print("Occupancy top-20 – n distribution:", dict(zip(unique_n, counts)))

    # ===== Summary JSON =====
    summary = {
        "replicas": len(replicas),
        "metas": metas,
        "config": {
            "OCC_ERR": OCC_ERR,
            "TS_ERR": TS_ERR,
            "Y_AXIS_LIMITS": Y_AXIS_LIMITS,
            "RENAMES_JSON": RENAMES_JSON,
            "ANNOTATE_MA_MEAN": ANNOTATE_MA_MEAN,
            "ANNOTATE_MA_FMT": ANNOTATE_MA_FMT,
            "RESIDUE_RENAMES": RESIDUE_RENAMES,
        },
    }
    if times_list:
        summary["timeseries_mean_stats"] = {
            "n_frames": int(merged["frame"].nunique()),
            "hbonds_mean_overall": float(merged["hbonds_mean"].mean()),
            "hbonds_final_mean_ma": float(final_mean_mavg),
            "hbonds_final_mean_raw": float(final_mean_raw),
        }
    if occ_list and not agg_sorted.empty:
        summary["occupancy_top_pair"] = {
            "pair": str(agg_sorted.iloc[0]["pair"]),
            "pair_renamed": str(agg_sorted.iloc[0]["pair_renamed"]),
            "mean_pct": float(agg_sorted.iloc[0]["mean_pct"]),
            "sd_pct": float(agg_sorted.iloc[0]["sd_pct"]),
            "sem_pct": float(agg_sorted.iloc[0]["sem_pct"]),
            "n": int(agg_sorted.iloc[0]["n"]),
        }

    Path("hbond_summary.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
