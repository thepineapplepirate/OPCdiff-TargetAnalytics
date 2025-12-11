#!/usr/bin/env python3
"""
dg_finalresults.py

Simple  script for plotting the total binding free energy values and components gmx_MMPBSA:
- TOTAL (ΔG_bind): bars with SD(Prop.) + data labels
- Components ("all"): grouped bars with SD(Prop.), NO data labels (legend only)

This script is intended to work with gmx_MMPBSA's "FINAL_RESULTS_MMPBSA.dat" outputfiles, 
which provide the overall energy values across the protein-ligand complex rather than the 
decomposed values on a per-residue basis. 

User must edit `files` and `labels` below.
"""

from pathlib import Path
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------- CONFIG ---------------- #

files = [
    "path/to/replica1/FINAL_RESULTS_MMPBSA.dat",
    "path/to/replica2/FINAL_RESULTS_MMPBSA.dat",
    "path/to/replica3/FINAL_RESULTS_MMPBSA.dat",
]

labels = [
    "Replica 1",
    "Replica 2",
    "Replica 3",
]

receptor_name = "M1"
drug_name = "Clemastine"

plot_mode = "component"     # "TOTAL" or "component"
outfile   = f"final/mmpbsa_plot_{plot_mode}.png"
dpi       = 600

# Filtering / tolerances
keep_zeros = False      # keep components whose Average == 0.0
ZERO_TOL   = 0.0        # treat |Average| <= ZERO_TOL as zero

# Sizes / aesthetics
TITLE_FONTSIZE = 20
AXISLABEL_FONTSIZE = 18
TICKLABEL_FONTSIZE = 16
DATALABEL_FONTSIZE_TOTAL = 16
GROUP_LABEL_ROTATION = 90  # for grouped bars
# axis padding (fractions of data span)
LABEL_PAD_FRACTION = 0.02         # distance of TOTAL labels from cap
BOTTOM_PAD_FRACTION = 0.06        # bottom padding so labels don’t touch spine
TOP_PAD_FRACTION_FOR_LEGEND = 0.2  # top white-space inside axes for legend (components plot)

# Optional fixed y-axis limits; use None to auto-compute.
FORCE_YLIMS_TOTAL = (-25, 0)      # applies to plot_total()
FORCE_YLIMS_COMP  = (-60, 80)      # applies to plot_components()

# Figure size (inches); set fixed if you prefer, e.g. (10, 6)
FIGSIZE_TOTAL = (12, 6)   # if None, auto: (max(6, 0.8*n), 5)
FIGSIZE_COMP  = (12, 6)   # if None, auto: (max(8, 0.9*n), max(5, 3+0.25*m))

# Optional per-component colors (leave empty to use defaults)
component_colors = {
    # "VDWAALS": "steelblue",
    # "EEL": "orange",
    # "EDISPER": "green",
    # "TOTAL": "red",
}

# ----------------------------------------- #

def clean_err(e):
    try:
        if e is None or math.isnan(e) or math.isinf(e):
            return 0.0
        return abs(float(e))
    except Exception:
        return 0.0

DELTA_HEADER = re.compile(r"^\s*Delta\s*\(.*Complex\s*-\s*Receptor\s*-\s*Ligand.*\):\s*$", re.IGNORECASE)
TABLE_HEADER = re.compile(r"^\s*Energy\s+Component\s+Average\s+SD\(Prop\.\)", re.IGNORECASE)
SEPARATOR = re.compile(r"^\s*-{5,}\s*$")

def parse_delta_table(text: str):
    """Parse 'Delta (Complex - Receptor - Ligand):' block and pick SD(Prop.) by header position."""
    lines = text.splitlines()
    n = len(lines)

    # 1) Find the Delta block header
    i = 0
    while i < n and not DELTA_HEADER.match(lines[i]):
        i += 1
    if i >= n:
        return {}

    # 2) Advance to the table header
    i += 1
    while i < n and not lines[i].strip():
        i += 1
    if i >= n:
        return {}

    # Helper to normalize component names
    def norm_name(raw: str):
        name = raw.strip().rstrip(":")
        if name.startswith("Δ"):
            name = name[1:].strip()
        name = name.upper()
        return name if name else None

    header_line = lines[i].rstrip()
    header_tokens = re.split(r"[ \t]+", header_line.strip())

    # Identify positions of key numeric columns
    try:
        avg_idx = next(j for j, t in enumerate(header_tokens) if t.lower().startswith("average"))
    except StopIteration:
        avg_idx = None

    def is_sdprop_token(tok: str) -> bool:
        t = tok.lower().replace(" ", "")
        return t.startswith("sd(prop") or t == "sd(prop.)" or "sd(prop" in t

    sdprop_idx = None
    for j, t in enumerate(header_tokens):
        if is_sdprop_token(t):
            sdprop_idx = j
            break

    has_structured_header = ("energy" in header_tokens[0].lower() and avg_idx is not None)

    # Move past header & optional separator line
    i += 1
    if i < n and SEPARATOR.match(lines[i]):
        i += 1

    comps = {}

    # 3) Read the main table rows until blank or next separator
    while i < n:
        line = lines[i].rstrip()
        if not line or SEPARATOR.match(line):
            break

        mnum = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
        if not mnum:
            i += 1
            continue

        comp_raw = line[:mnum.start()]
        # require at least one alphabetic character in the name
        if not re.search(r"[A-Za-z]", comp_raw):
            i += 1
            continue

        comp = norm_name(comp_raw)
        if not comp:
            i += 1
            continue

        nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
        if not nums:
            i += 1
            continue

        # First numeric after name is Average
        try:
            avg_val = float(nums[0])
        except Exception:
            i += 1
            continue

        # Choose SD(Prop.) using header position if available
        sdprop_val = None
        if has_structured_header and sdprop_idx is not None and avg_idx is not None:
            offset = sdprop_idx - avg_idx
            if 0 <= offset < len(nums):
                try:
                    sdprop_val = float(nums[offset])
                except Exception:
                    sdprop_val = None

        if sdprop_val is None:
            if len(nums) >= 2:
                try:
                    sdprop_val = float(nums[1])
                except Exception:
                    sdprop_val = 0.0
            else:
                sdprop_val = 0.0

        comps[comp] = {"avg": float(avg_val), "sdprop": clean_err(sdprop_val)}
        i += 1

    # 4) Summary lines below (e.g., GGAS / GSOLV / TOTAL)
    while i < n:
        line = lines[i].rstrip()
        if SEPARATOR.match(line):
            break
        if line.strip():
            mnum = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
            if mnum:
                comp_raw = line[:mnum.start()]
                if not re.search(r"[A-Za-z]", comp_raw):
                    i += 1
                    continue
                comp = norm_name(comp_raw)
                if not comp:
                    i += 1
                    continue
                nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                if len(nums) >= 2:
                    try:
                        avg_val = float(nums[0])
                        sdprop_val = float(nums[1])
                        comps[comp] = {"avg": avg_val, "sdprop": clean_err(sdprop_val)}
                    except Exception:
                        pass
        i += 1

    return comps

def read_final_results(path: Path):
    return parse_delta_table(path.read_text(errors="ignore"))

CANONICAL_ORDER = [
    "BOND", "ANGLE", "DIHED",
    "VDWAALS", "EEL", "1-4 VDW", "1-4 EEL",
    "EPB", "ENPOLAR", "EDISPER",
    "GGAS", "GSOLV", "TOTAL",
]

def order_components(keys):
    ordered = [k for k in CANONICAL_ORDER if k in keys]
    for k in keys:
        if k not in ordered:
            ordered.append(k)
    return ordered

def is_zero(val: float) -> bool:
    return abs(val) <= ZERO_TOL

# ---------------- Plotting (TOTAL) ---------------- #

def plot_total(dataset):
    labels_, heights, errs = [], [], []
    for label, comps in dataset:
        if "TOTAL" in comps:
            labels_.append(label)
            heights.append(comps["TOTAL"]["avg"])
            errs.append(clean_err(comps["TOTAL"]["sdprop"]))

    if not labels_:
        raise SystemExit("No ΔTOTAL rows found to plot.")

    x = np.arange(len(labels_))
    figsize = FIGSIZE_TOTAL or (max(6, 0.8*len(labels_)), 5)
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["black", "blue", "red"]
    ax.bar(x, heights, yerr=errs, capsize=4, color=colors)

    # Axis labels & fonts
    ax.set_xticks(x)
    ax.set_xticklabels(labels_, rotation=45, ha="right")
    ax.set_ylabel("ΔG$_{bind}$ (kcal/mol)", fontsize=AXISLABEL_FONTSIZE)
    ax.set_title(f"Final Binding Free Energy ΔG \n({receptor_name} - {drug_name})", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICKLABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICKLABEL_FONTSIZE)
    ax.axhline(0, linewidth=1)

    # Simple labels just below lower cap
    ymin0, ymax0 = ax.get_ylim()
    span0 = max(1e-9, ymax0 - ymin0)
    base_margin = LABEL_PAD_FRACTION * span0
    for xi, yi, err in zip(x, heights, errs):
        y_text = (yi - err) - base_margin if err > 0 else yi - base_margin
        ax.text(
            xi, y_text,
            f"{yi:.2f} ± {err:.2f}",
            ha="center", va="top", fontsize=DATALABEL_FONTSIZE_TOTAL
        )

    # Axis limits: use padding, but allow override if needed
    ymin, ymax = ax.get_ylim()
    pad = BOTTOM_PAD_FRACTION * (ymax - ymin)
    ymin_dyn, ymax_dyn = ymin - pad, ymax
    # override if FORCE_YLIMS_TOTAL defined
    ymin_fixed, ymax_fixed = FORCE_YLIMS_TOTAL
    if ymin_fixed is not None:
        ymin_dyn = ymin_fixed
    if ymax_fixed is not None:
        ymax_dyn = ymax_fixed
    ax.set_ylim(ymin_dyn, ymax_dyn)


    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)

# ---------------- Plotting (components, no labels) ---------------- #

def plot_components(dataset):
    # Gather which components to plot (respect zero filtering)
    union = set()
    for _, comps in dataset:
        for k, v in comps.items():
            if keep_zeros or not is_zero(v["avg"]):
                union.add(k)
    if not union:
        raise SystemExit("No non-zero components found across inputs.")

    # Order and drop any empty/None component names
    comps_to_plot = [c for c in order_components(list(union)) if c and c.strip()]

    labels_ = [label for label, _ in dataset]
    x = np.arange(len(labels_))
    n_comps = len(comps_to_plot)
    width = max(0.08, min(0.8 / max(1, n_comps), 0.22))

    figsize = FIGSIZE_COMP or (max(8, 0.9*len(labels_)), max(5, 3 + 0.25*n_comps))
    fig, ax = plt.subplots(figsize=figsize)

    # ---------- First pass: compute bars and global y-range from caps ----------
    data_per_comp = []  # store to draw in second pass: (comp, x_offsets, heights, errs)
    top_caps, bot_caps = [], []

    for i, comp_name in enumerate(comps_to_plot):
        if not comp_name:  # safety
            continue

        heights, errs = [], []
        for _, comps in dataset:
            if comp_name in comps and (keep_zeros or not is_zero(comps[comp_name]["avg"])):
                h = comps[comp_name]["avg"]
                e = clean_err(comps[comp_name]["sdprop"])
            else:
                h, e = 0.0, 0.0
            heights.append(h)
            errs.append(e)

        x_offsets = x + (i - (n_comps - 1) / 2) * width
        data_per_comp.append((comp_name, x_offsets, heights, errs))

        for h, e in zip(heights, errs):
            top_caps.append(h + e)
            bot_caps.append(h - e)

    # Span from data (including error bars)
    cap_min = min(bot_caps) if bot_caps else 0.0
    cap_max = max(top_caps) if top_caps else 1.0
    span = max(1e-9, cap_max - cap_min)

    bottom_pad = BOTTOM_PAD_FRACTION * span
    top_pad = TOP_PAD_FRACTION_FOR_LEGEND * span

    ymin_dyn = cap_min - bottom_pad
    ymax_dyn = cap_max + top_pad

    # Override if FORCE_YLIMS_COMP is defined
    ymin_fixed, ymax_fixed = FORCE_YLIMS_COMP
    if ymin_fixed is not None:
        ymin_dyn = ymin_fixed
    if ymax_fixed is not None:
        ymax_dyn = ymax_fixed

    # Apply limits BEFORE drawing so legend space is truly empty
    ax.set_ylim(ymin_dyn, ymax_dyn)


    # ---------- Second pass: draw all groups ----------
    for comp_name, x_offsets, heights, errs in data_per_comp:
        if not comp_name:  # safety
            continue
        color = component_colors.get(comp_name, None)
        ax.bar(
            x_offsets, heights, width=width,
            yerr=errs, capsize=3, label=comp_name, color=color
        )

    # Axes styling
    ax.set_xticks(x)
    ax.set_xticklabels(labels_, rotation=45, ha="right")
    ax.set_ylabel("Energy (kcal/mol)", fontsize=AXISLABEL_FONTSIZE)
    ax.set_title(f"Energy Components of ΔG \n({receptor_name} - {drug_name})", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICKLABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICKLABEL_FONTSIZE)
    ax.axhline(0, linewidth=1)

    # Minor y-ticks
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(axis="y", which="minor", length=3)

    # Legend INSIDE plot, top-right, on the reserved white space
    ax.legend(
        ncol=min(4, n_comps),
        fontsize=11,
        loc="upper right",
        frameon=True,
        borderaxespad=0.8
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)

# ---------------- RUN ---------------- #

def main():
    assert len(files) == len(labels), "files and labels must be the same length, in the same order."
    dataset = []
    for label, f in zip(labels, files):
        p = Path(f)
        if not p.is_file():
            raise FileNotFoundError(f"Not a file: {p}")
        comps = read_final_results(p)
        if not comps:
            raise RuntimeError(f"No 'Delta (Complex - Receptor - Ligand):' block parsed in: {p}")
        dataset.append((label, comps))

    if plot_mode.upper() == "TOTAL":
        plot_total(dataset)
    else:
        plot_components(dataset)

    print("Saved plot to: {}".format(Path(outfile).resolve()))

if __name__ == "__main__":
    main()
