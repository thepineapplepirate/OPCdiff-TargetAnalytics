#!/usr/bin/env python3
"""
compare_weighted_vs_unweighted.py

Rebuild the weighted vs unweighted MM-PBSA comparison figure directly from
gmx_MMPBSA FINAL_RESULTS_MMPBSA.dat files.

What it does
------------
For each system:
- parses the 'Delta (Complex - Receptor - Ligand)' block
- extracts the ΔTOTAL Average and SD(Prop.) values from each replica file

Then it computes:
1) Unweighted analysis:
   - nonparametric Monte Carlo by sampling replica means with replacement
2) Weighted analysis:
   - inverse-variance weighting using SD(Prop.) directly as the uncertainty term
   - parametric Monte Carlo by sampling from N(weighted_mean, weighted_uncertainty)

Important note
--------------
This reproduces the original exploratory analysis you recovered. It uses SD(Prop.)
directly for weighting, not a frame-corrected SE.

Usage
-----
Edit the CONFIG section, then run:

    python compare_weighted_vs_unweighted.py

Outputs
-------
- unweighted_vs_weighted_probabilities.png
- console printout of extracted means / SD(Prop.) / probabilities
"""

from pathlib import Path
import re
import math
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

systems = {
    "M1-Clemastine": {
        "files": [
            "../../../Data/M1_muscarinic_receptor/complexes/clemastine/analysis/replica1/mmpbsa/FINAL_RESULTS_MMPBSA.dat",
            "../../../Data/M1_muscarinic_receptor/complexes/clemastine/analysis/replica2/mmpbsa/FINAL_RESULTS_MMPBSA.dat",
            "../../../Data/M1_muscarinic_receptor/complexes/clemastine/analysis/replica3/mmpbsa/FINAL_RESULTS_MMPBSA.dat",
        ],
        "color": "orange",
    },
    "M1-CN045": {
        "files": [
            "../../../Data/M1_muscarinic_receptor/complexes/cn045/analysis/replica1/mmpbsa/FINAL_RESULTS_MMPBSA.dat",
            "../../../Data/M1_muscarinic_receptor/complexes/cn045/analysis/replica2/mmpbsa/FINAL_RESULTS_MMPBSA.dat",
            "../../../Data/M1_muscarinic_receptor/complexes/cn045/analysis/replica3/mmpbsa/FINAL_RESULTS_MMPBSA.dat",
        ],
        "color": "navy",
    },
    "H3-CN045": {
        "files": [
            "../../../Data/H3_histamine_receptor/complexes/cn045/analysis/replica1/mmpbsa/FINAL_RESULTS_MMPBSA.dat",
            "../../../Data/H3_histamine_receptor/complexes/cn045/analysis/replica2/mmpbsa/FINAL_RESULTS_MMPBSA.dat",
            "../../../Data/H3_histamine_receptor/complexes/cn045/analysis/replica3/mmpbsa/FINAL_RESULTS_MMPBSA.dat",
        ],
        "color": "limegreen",
    },
}

# Compare the second system against the first and third
# This mirrors the recovered original plot:
#   M1-CN045 vs M1-Clemastine
#   M1-CN045 vs H3-CN045
target_name = "M1-CN045"
compare_to = ["M1-Clemastine", "H3-CN045"]

n_draws = 200000
rng_seed = 42
bins = 50
outfile = "unweighted_vs_weighted_probabilities.png"

# =========================
# PARSING
# =========================

DELTA_HEADER = re.compile(
    r"^\s*Delta\s*\(.*Complex\s*-\s*Receptor\s*-\s*Ligand.*\):\s*$",
    re.IGNORECASE,
)

TOTAL_ROW = re.compile(r"^\s*(?:Δ)?TOTAL\b", re.IGNORECASE)


def clean_err(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return 0.0
        return abs(x)
    except Exception:
        return 0.0


def parse_total_avg_and_sdprop(path):
    """
    Return (TOTAL_average, TOTAL_SDProp) from the Delta block of a
    FINAL_RESULTS_MMPBSA.dat file.
    """
    path = Path(path)
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    # Find Delta block header
    i = 0
    while i < len(lines) and not DELTA_HEADER.match(lines[i]):
        i += 1
    if i >= len(lines):
        raise ValueError(f"Could not find Delta block in {path}")

    # Find TOTAL row after that
    while i < len(lines):
        line = lines[i].rstrip()
        if TOTAL_ROW.match(line):
            nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
            if len(nums) < 2:
                raise ValueError(f"Could not parse TOTAL row in {path}: {line}")
            avg = float(nums[0])
            sdprop = clean_err(nums[1])
            return avg, sdprop
        i += 1

    raise ValueError(f"Could not find TOTAL row in Delta block of {path}")


def load_system_data(file_list):
    means = []
    sds = []
    for f in file_list:
        avg, sdprop = parse_total_avg_and_sdprop(f)
        means.append(avg)
        sds.append(sdprop)
    return np.asarray(means, dtype=float), np.asarray(sds, dtype=float)


# =========================
# ANALYSIS
# =========================

def weighted_mean_and_se(means, sds):
    """
    Original recovered implementation:
    use SD(Prop.) directly as the uncertainty term for inverse-variance weighting.
    """
    ses = np.asarray(sds, dtype=float)
    weights = 1.0 / (ses ** 2)
    mean_w = np.sum(weights * means) / np.sum(weights)
    se_w = np.sqrt(1.0 / np.sum(weights))
    return mean_w, se_w


def unweighted_probability(means_a, means_b, rng, n_draws):
    draws_a = rng.choice(means_a, size=n_draws, replace=True)
    draws_b = rng.choice(means_b, size=n_draws, replace=True)
    prob = np.mean(draws_a < draws_b)
    return draws_a, draws_b, prob


def weighted_probability(means_a, sds_a, means_b, sds_b, rng, n_draws):
    mean_a_w, se_a_w = weighted_mean_and_se(means_a, sds_a)
    mean_b_w, se_b_w = weighted_mean_and_se(means_b, sds_b)
    draws_a = rng.normal(mean_a_w, se_a_w, n_draws)
    draws_b = rng.normal(mean_b_w, se_b_w, n_draws)
    prob = np.mean(draws_a < draws_b)
    return draws_a, draws_b, prob, mean_a_w, se_a_w, mean_b_w, se_b_w


# =========================
# MAIN
# =========================

def main():
    rng = np.random.default_rng(rng_seed)

    # Load all systems from files
    parsed = {}
    for name, meta in systems.items():
        means, sds = load_system_data(meta["files"])
        parsed[name] = {
            "means": means,
            "sds": sds,
            "color": meta["color"],
        }

    # Plot layout mirrors the recovered original figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    row_pairs = [
        ("unweighted", 0),
        ("weighted", 1),
    ]

    # Build the two comparisons
    for col, other_name in enumerate(compare_to):
        A = parsed[target_name]
        B = parsed[other_name]

        # Unweighted
        draws_A_unw, draws_B_unw, p_unw = unweighted_probability(
            A["means"], B["means"], rng, n_draws
        )
        ax = axes[0, col]
        ax.hist(draws_A_unw, bins=bins, alpha=0.6, density=True,
                label=target_name, color=A["color"])
        ax.hist(draws_B_unw, bins=bins, alpha=0.6, density=True,
                label=other_name, color=B["color"])
        ax.set_title(
            f"Unweighted posterior: {target_name} vs {other_name}\n"
            f"P({target_name}<{other_name})={p_unw * 100:.1f}%"
        )
        ax.set_xlabel("ΔG (kcal/mol)")
        ax.set_ylabel("Density")
        ax.legend()

        # Weighted
        draws_A_w, draws_B_w, p_w, A_mean_w, A_se_w, B_mean_w, B_se_w = weighted_probability(
            A["means"], A["sds"], B["means"], B["sds"], rng, n_draws
        )
        ax = axes[1, col]
        ax.hist(draws_A_w, bins=bins, alpha=0.6, density=True,
                label=target_name, color=A["color"])
        ax.hist(draws_B_w, bins=bins, alpha=0.6, density=True,
                label=other_name, color=B["color"])
        ax.set_title(
            f"Weighted posterior: {target_name} vs {other_name}\n"
            f"P({target_name}<{other_name})={p_w * 100:.1f}%"
        )
        ax.set_xlabel("ΔG (kcal/mol)")
        ax.set_ylabel("Density")
        ax.legend()

        # Console summary
        print(f"\n=== {target_name} vs {other_name} ===")
        print(f"{target_name} means: {A['means'].tolist()}")
        print(f"{target_name} SD(Prop.): {A['sds'].tolist()}")
        print(f"{other_name} means: {B['means'].tolist()}")
        print(f"{other_name} SD(Prop.): {B['sds'].tolist()}")
        print(f"Unweighted P({target_name}<{other_name}) = {p_unw * 100:.3f}%")
        print(f"Weighted P({target_name}<{other_name}) = {p_w * 100:.3f}%")
        print(f"{target_name} weighted mean ± uncertainty = {A_mean_w:.3f} ± {A_se_w:.3f}")
        print(f"{other_name} weighted mean ± uncertainty = {B_mean_w:.3f} ± {B_se_w:.3f}")

    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nSaved figure to: {outfile}")


if __name__ == "__main__":
    main()
