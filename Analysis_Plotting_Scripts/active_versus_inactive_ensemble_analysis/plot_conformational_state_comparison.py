#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# User settings
# =========================
SYSTEM_DIRS = {
    "M1-CN045": Path("../../Data/M1_muscarinic_receptor/complexes/cn045/analysis/"),
    "M1-Clemastine": Path("../../Data/M1_muscarinic_receptor/complexes/clemastine/analysis/"),
}
REPS = ["replica1", "replica2", "replica3"]
OUT_SUBDIR = "analysis_active_vs_inactive"

# Scalar metrics (time series -> reduced to replica mean over time)
FILES = {
    "Drug - Asp105 minimum distance": "mindist_UNL_r86.xvg",
    "Drug - Tyr106 minimum distance": "mindist_UNL_r87.xvg",
    "Drug - Tyr381 minimum distance": "mindist_UNL_r386.xvg",
    "Drug - Tyr404 minimum distance": "mindist_UNL_r409.xvg",
    "Drug - Tyr418 minimum distance": "mindist_UNL_r423.xvg",
    #"TM3_TM6_IC_dist":  "tm3dic_tm6ic_dist.xvg",
    #"TM3DRY_TM6IC_dist": "tm3dry_tm6ic_dist.xvg",
    "Arg³·⁵⁰(123)_Glu⁶·³⁰(360)_mean_dist": "tm3r104_tm6r365_dist.xvg",  # NEW (Arg104–Glu365)
    # "rmsd_to_inactive": "rmsd_to_inactive.xvg",
    # "rmsd_to_active":   "rmsd_to_active.xvg",
}

# Chi1 time series from gmx angle (-ov ... -all)s
CHI_METRIC_NAME = "χ1_Tyr⁷·⁵³(418)"
CHI_FILE = "chi1_Tyr423.xvg"   # set to your actual filename

# Discard equilibration (ns)
DISCARD_FIRST_NS = 0.0

# Chi histogram settings (match GROMACS -180..180 convention)
CHI_RANGE = (-180.0, 180.0)
CHI_BINS = 72  # 5° bins across 360°

# Unit conversion
NM_TO_ANGSTROM = 10.0

# Which metrics are distances/RMSDs in nm (convert to Å)
NM_METRICS = {
    "Drug - Asp105 minimum distance",
    "Drug - Tyr106 minimum distance",
    "Drug - Tyr381 minimum distance",
    "Drug - Tyr404 minimum distance",
    "Drug - Tyr418 minimum distance",
    #"TM3_TM6_IC_dist",
    "Arg³·⁵⁰(123)_Glu⁶·³⁰(360)_mean_dist",  # NEW
    "rmsd_to_inactive",
    "rmsd_to_active",
}

# =========================
# Plot styling controls
# =========================
SYSTEM_COLORS = {
    "M1-CN045": "#c42083",
    "M1-Clemastine": "tab:orange",
}


STATE_COLORS = {
    "active_like": "#1b229e",    # green/teal → activation
    "inactive_like": "#df3d0c",  # orange/red → inactive
    "other": "#11b106",          # neutral gray
}


FONT_SIZES = {
    "title": 14,
    "axis": 14,
    "ticks": 14,
    "legend": 12,
}

# resolution
DPI = 600

plt.rcParams.update({
    "axes.titlesize": FONT_SIZES["title"],
    "axes.labelsize": FONT_SIZES["axis"],
    "xtick.labelsize": FONT_SIZES["ticks"],
    "ytick.labelsize": FONT_SIZES["ticks"],
    "legend.fontsize": FONT_SIZES["legend"],
})

LEGEND_LOC = "best"

# =========================
# New analysis settings
# =========================
# Reference-anchored windows (center ± 25°)
# Active ref:  -47.595 deg  => [-72.595, -22.595]
# Inactive ref: +47.540 deg => [+22.540, +72.540]
CHI_ACTIVE_WINDOWS = [(-72.6, -22.6)]
CHI_INACTIVE_WINDOWS = [(22.5, 72.5)]
CHI_OCC_SMOOTH_NS = 0.5  # rolling mean window width (ns)

# Correlation: run against both a coarse TM3–TM6 IC distance AND a refined DRY/ionic-lock proxy distance
CORR_DISTANCE_METRICS = [
    #"TM3_TM6_IC_dist",
    "Arg³·⁵⁰(123)_Glu⁶·³⁰(360)_mean_dist", 
]
CORR_BIN_WIDTH_A = 0.5  # Å bins

# =========================
# Helpers
# =========================
def read_xvg_numeric(path: Path) -> np.ndarray:
    """Return numeric matrix for an XVG, skipping @/# lines."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if not rows:
        return np.empty((0, 0))
    return np.array(rows, dtype=float)

def to_ns(time_col: np.ndarray) -> np.ndarray:
    """Convert ps->ns if it looks like ps."""
    if time_col.size == 0:
        return time_col
    tmax = float(np.nanmax(time_col))
    return time_col / 1000.0 if tmax > 1e4 else time_col

def wrap_deg_pm180(x: np.ndarray) -> np.ndarray:
    """Wrap angles into [-180, 180)."""
    x = np.asarray(x, dtype=float)
    return ((x + 180.0) % 360.0) - 180.0

def maybe_convert_nm_to_a(metric: str, values: np.ndarray) -> np.ndarray:
    """Convert nm->Å for known distance/RMS metrics."""
    if metric in NM_METRICS:
        return values * NM_TO_ANGSTROM
    return values

def add_window_occupancy(
    df_long: pd.DataFrame,
    base_metric: str,
    windows,
    out_metric: str,
    smooth_ns: float,
) -> pd.DataFrame:
    """
    Adds two derived metrics as new rows to df_long:
      - out_metric: binary occupancy (0/1)
      - out_metric+"_smooth": rolling mean occupancy

    df_long must contain columns: system, replica, time, metric, value
    """
    base = df_long[df_long["metric"] == base_metric].copy()
    if base.empty:
        return df_long

    base = base.sort_values(["system", "replica", "time"])
    base["value"] = wrap_deg_pm180(base["value"].to_numpy())

    v = base["value"].to_numpy()
    occ_bool = np.zeros(len(base), dtype=bool)
    for lo, hi in windows:
        occ_bool |= (v >= lo) & (v <= hi)

    occ = base[["system", "replica", "time"]].copy()
    occ["metric"] = out_metric
    occ["value"] = occ_bool.astype(float)

    def _smooth(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("time").copy()
        t = group["time"].to_numpy()
        if len(t) < 3:
            group["value_smooth"] = group["value"].to_numpy()
            return group
        dt = float(np.median(np.diff(t)))
        win = max(1, int(round(smooth_ns / dt))) if dt > 0 else 1
        group["value_smooth"] = group["value"].rolling(
            win, center=True, min_periods=1
        ).mean().to_numpy()
        return group

    occ_parts = []
    for (_, _), group in occ.groupby(["system", "replica"], sort=False):
        occ_parts.append(_smooth(group))

    occ = pd.concat(occ_parts, ignore_index=True)

    occ_s = occ[["system", "replica", "time"]].copy()
    occ_s["metric"] = out_metric + "_smooth"
    occ_s["value"] = occ["value_smooth"].to_numpy()

    return pd.concat([df_long, occ, occ_s], ignore_index=True)

def system_mean_sem_over_reps(ts_long: pd.DataFrame) -> pd.DataFrame:
    out = (
        ts_long.groupby(["system", "time"])["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    out["sem"] = out.apply(lambda r: (r["std"] / np.sqrt(r["n"])) if r["n"] > 1 else np.nan, axis=1)
    return out

def align_by_time_interp(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """
    Align two time series by interpolating df_b onto df_a's time grid.
    df_a/df_b columns: time, value
    Returns columns: time, a_value, b_value
    """
    a = df_a.sort_values("time")
    b = df_b.sort_values("time")
    t = a["time"].to_numpy()
    av = a["value"].to_numpy()
    bv = np.interp(t, b["time"].to_numpy(), b["value"].to_numpy())
    return pd.DataFrame({"time": t, "a_value": av, "b_value": bv})

def compute_state_occupancy_from_chi(
    chi_ts: pd.DataFrame,
    active_windows,
    inactive_windows,
    systems_order,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes replica-level and system-level state occupancies from raw chi1 time series.

    Returns:
      rep_occ: columns [system, replica, active_like, inactive_like, other]
      sys_occ: index=system, columns [active_like, inactive_like, other]
    """
    if chi_ts.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = chi_ts[chi_ts["metric"] == CHI_METRIC_NAME].copy()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = base.sort_values(["system", "replica", "time"])
    ang = wrap_deg_pm180(base["value"].to_numpy())

    active = np.zeros(len(base), dtype=bool)
    for lo, hi in active_windows:
        active |= (ang >= lo) & (ang <= hi)

    inactive = np.zeros(len(base), dtype=bool)
    for lo, hi in inactive_windows:
        inactive |= (ang >= lo) & (ang <= hi)

    base["active_like"] = active.astype(float)
    base["inactive_like"] = inactive.astype(float)
    base["other"] = 1.0 - base["active_like"] - base["inactive_like"]

    rep_occ = (
        base.groupby(["system", "replica"])[["active_like", "inactive_like", "other"]]
        .mean()
        .reset_index()
    )

    sys_occ = (
        rep_occ.groupby("system")[["active_like", "inactive_like", "other"]]
        .mean()
        .reindex(systems_order)
    )

    return rep_occ, sys_occ

# =========================
# Load scalar time series
# =========================
rows = []
missing = []

for system, base in SYSTEM_DIRS.items():
    for rep in REPS:
        rep_dir = base / rep / OUT_SUBDIR
        if not rep_dir.exists():
            missing.append(f"{system}/{rep}: missing {rep_dir}")
            continue

        for metric, fname in FILES.items():
            f = rep_dir / fname
            if not f.exists():
                missing.append(f"{system}/{rep}: missing {fname}")
                continue

            mat = read_xvg_numeric(f)
            if mat.shape[1] < 2:
                missing.append(f"{system}/{rep}: {fname} has <2 numeric columns")
                continue

            t = to_ns(mat[:, 0])
            v = mat[:, 1]
            if DISCARD_FIRST_NS > 0:
                mask = t >= DISCARD_FIRST_NS
                t, v = t[mask], v[mask]

            v = maybe_convert_nm_to_a(metric, v)

            df = pd.DataFrame({"time": t, "value": v})
            df["system"] = system
            df["replica"] = rep
            df["metric"] = metric
            rows.append(df)

if not rows:
    raise SystemExit("No scalar time series found. Check paths and filenames.")

all_ts = pd.concat(rows, ignore_index=True)
all_ts.to_csv("all_timeseries.csv", index=False)

rep_summary = (
    all_ts.groupby(["system", "metric", "replica"])["value"]
    .agg(replica_time_mean="mean", replica_time_std="std", n_frames="count")
    .reset_index()
)
rep_summary.to_csv("replica_summary.csv", index=False)

sys_summary = (
    rep_summary.groupby(["system", "metric"])["replica_time_mean"]
    .agg(system_mean="mean", system_std="std", n_reps="count")
    .reset_index()
)
sys_summary["system_sem"] = sys_summary.apply(
    lambda r: (r["system_std"] / np.sqrt(r["n_reps"])) if r["n_reps"] > 1 else np.nan,
    axis=1
)
sys_summary.to_csv("system_summary.csv", index=False)

systems_order = list(SYSTEM_DIRS.keys())

# Barplots: LigA vs LigB for scalar metrics (color-coded)
for metric in sorted(sys_summary["metric"].unique()):
    sub = sys_summary[sys_summary["metric"] == metric].set_index("system")
    means = [sub.loc[s, "system_mean"] if s in sub.index else np.nan for s in systems_order]
    sems  = [sub.loc[s, "system_sem"]  if s in sub.index else np.nan for s in systems_order]
    colors = [SYSTEM_COLORS.get(s, None) for s in systems_order]

    plt.figure()
    plt.bar(systems_order, means, yerr=sems, color=colors)
    if metric.startswith("rmsd"):
        ylabel = "RMSD (Å)"
    elif metric.startswith("Arg"):
        ylabel = "Mean distance (Å)"
    else:
        ylabel = "Minimum distance (Å)" 
    plt.ylabel(ylabel)
    if metric == "Arg³·⁵⁰(123)_Glu⁶·³⁰(360)_mean_dist":
        plt.title("Arg³·⁵⁰-Glu⁶·³⁰ distance")
    else:
        plt.title(f"{metric}") # (mean ± SEM across replicas)
    plt.tight_layout()
    plt.savefig(f"compare_{metric}.png", dpi=DPI)
    plt.close()

# =========================
# Orthosteric engagement summary (grouped bar plot)
# =========================
ORTHO_METRICS = [
    ("Drug - Asp105 minimum distance",  "Asp³·³² (105)"),
    ("Drug - Tyr106 minimum distance",  "Tyr³·³³ (106)"),
    ("Drug - Tyr381 minimum distance", "Tyr⁶·⁵¹ (381)"),
    ("Drug - Tyr404 minimum distance", "Tyr⁷·³⁹ (404)"),
]

labels = [lbl for _, lbl in ORTHO_METRICS]
x = np.arange(len(labels))
width = 0.35

means = {s: [] for s in systems_order}
sems  = {s: [] for s in systems_order}

for metric, _ in ORTHO_METRICS:
    for s in systems_order:
        row = sys_summary[(sys_summary["system"] == s) & (sys_summary["metric"] == metric)]
        means[s].append(row["system_mean"].values[0])
        sems[s].append(row["system_sem"].values[0])

plt.figure(figsize=(8, 5))
for i, s in enumerate(systems_order):
    plt.bar(
        x + (i - 0.5) * width,
        means[s],
        width,
        yerr=sems[s],
        label=s,
        color=SYSTEM_COLORS.get(s, None),
        capsize=4,
    )

plt.xticks(x, labels, rotation=20)
plt.ylabel("Minimum distance (Å)")
plt.title("Orthosteric engagement summary (mean ± SEM across replicas)")
plt.legend()
plt.tight_layout()
plt.savefig("orthosteric_engagement_summary.png", dpi=DPI)
plt.close()

# =========================
# Load chi time series + histograms + occupancy + correlation
# =========================
chi_rows = []
chi_missing = []

for system, base in SYSTEM_DIRS.items():
    for rep in REPS:
        f = base / rep / OUT_SUBDIR / CHI_FILE
        if not f.exists():
            chi_missing.append(f"{system}/{rep}: missing {CHI_FILE}")
            continue

        mat = read_xvg_numeric(f)
        if mat.shape[1] < 2:
            chi_missing.append(f"{system}/{rep}: {CHI_FILE} has <2 numeric columns")
            continue

        t = to_ns(mat[:, 0])

        # With -all, file can be: time, avg, angle1, angle2...
        ang = mat[:, 1]

        if DISCARD_FIRST_NS > 0:
            mask = t >= DISCARD_FIRST_NS
            t, ang = t[mask], ang[mask]

        ang = wrap_deg_pm180(ang)

        df = pd.DataFrame({"time": t, "value": ang})
        df["system"] = system
        df["replica"] = rep
        df["metric"] = CHI_METRIC_NAME
        chi_rows.append(df)

if chi_rows:
    chi_ts = pd.concat(chi_rows, ignore_index=True)
    chi_ts.to_csv("chi_timeseries.csv", index=False)

    # Add active-like + inactive-like occupancy
    chi_aug = chi_ts.copy()
    chi_aug = add_window_occupancy(
        chi_aug, base_metric=CHI_METRIC_NAME,
        windows=CHI_ACTIVE_WINDOWS,
        out_metric="chi1_Tyr423_active_like",
        smooth_ns=CHI_OCC_SMOOTH_NS,
    )
    chi_aug = add_window_occupancy(
        chi_aug, base_metric=CHI_METRIC_NAME,
        windows=CHI_INACTIVE_WINDOWS,
        out_metric="chi1_Tyr423_inactive_like",
        smooth_ns=CHI_OCC_SMOOTH_NS,
    )
    chi_aug.to_csv("chi_timeseries_with_occupancy.csv", index=False)

    # ---- Histograms (system mean ± SEM)
    bin_edges = np.linspace(CHI_RANGE[0], CHI_RANGE[1], CHI_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hist_records = []
    for (system, rep), sub in chi_ts.groupby(["system", "replica"]):
        vals = sub["value"].to_numpy()
        counts, _ = np.histogram(vals, bins=bin_edges, density=False)
        prob = counts / counts.sum() if counts.sum() > 0 else np.full_like(counts, np.nan, dtype=float)
        for bc, p in zip(bin_centers, prob):
            hist_records.append({"system": system, "replica": rep, "angle_bin_center": bc, "prob": p})

    chi_hist = pd.DataFrame(hist_records)
    chi_hist.to_csv("chi_hist_replica_probs.csv", index=False)

    chi_hist_summary = (
        chi_hist.groupby(["system", "angle_bin_center"])["prob"]
        .agg(mean_prob="mean", std_prob="std", n_reps="count")
        .reset_index()
    )
    chi_hist_summary["sem_prob"] = chi_hist_summary.apply(
        lambda r: (r["std_prob"] / np.sqrt(r["n_reps"])) if r["n_reps"] > 1 else np.nan,
        axis=1
    )
    chi_hist_summary.to_csv("chi_hist_system_summary.csv", index=False)

    pred = (
        chi_hist_summary.sort_values(["system", "mean_prob"], ascending=[True, False])
        .groupby("system")
        .head(1)[["system", "angle_bin_center", "mean_prob"]]
        .rename(columns={"angle_bin_center": "predominant_angle_deg", "mean_prob": "peak_probability"})
    )
    pred.to_csv("chi_predominant_angle.csv", index=False)

    # ---- Plot: chi histogram (color-coded) + reference markers
    plt.figure()
    for system in systems_order:
        sub = chi_hist_summary[chi_hist_summary["system"] == system].sort_values("angle_bin_center")
        if sub.empty:
            continue
        x = sub["angle_bin_center"].to_numpy()
        y = sub["mean_prob"].to_numpy()
        ysem = sub["sem_prob"].to_numpy()
        c = SYSTEM_COLORS.get(system, None)
        plt.plot(x, y, label=system, color=c)
        if np.isfinite(ysem).any():
            plt.fill_between(x, y - ysem, y + ysem, alpha=0.2, color=c)

    # Reference χ1 values
    ACTIVE_REF_CHI1 = -47.595
    INACTIVE_REF_CHI1 = +47.540

    # Active reference: dashed
    plt.axvline(
        ACTIVE_REF_CHI1,
        color="k",
        linestyle="--",
        linewidth=1.6,
        alpha=0.9,
        label="Active ref χ₁ (−47.6°)",
    )

    # Inactive reference: dotted
    plt.axvline(
        INACTIVE_REF_CHI1,
        color="k",
        linestyle=":",
        linewidth=2.0,
        alpha=0.9,
        label="Inactive ref χ₁ (+47.5°)",
    )

    plt.xlabel("χ1 angle (deg, -180..180)")
    plt.ylabel("Probability")
    plt.title(f"{CHI_METRIC_NAME} angle distribution") # replica-averaged
    plt.xlim(CHI_RANGE[0], CHI_RANGE[1])
    plt.legend(loc=LEGEND_LOC)
    plt.tight_layout()
    plt.savefig(f"compare_{CHI_METRIC_NAME}_hist.png", dpi=DPI)
    plt.close()

   # ---- Plot: chi time series (replica-averaged mean ± SEM) with clear reference legend
    ACTIVE_REF_CHI1 = -47.595
    INACTIVE_REF_CHI1 = +47.540

    plt.figure()

    # compute system mean ± SEM at each time across replicas
    chi_summary = system_mean_sem_over_reps(chi_ts)   # columns: system,time,mean,std,n,sem

    for system in systems_order:
        sub = chi_summary[chi_summary["system"] == system].sort_values("time")
        if sub.empty:
            continue
        c = SYSTEM_COLORS.get(system, None)
        x = sub["time"].to_numpy()
        y = sub["mean"].to_numpy()
        ysem = sub["sem"].to_numpy()

        plt.plot(x, y, color=c, linewidth=1.8, label=system)
        if np.isfinite(ysem).any():
            plt.fill_between(x, y - ysem, y + ysem, color=c, alpha=0.20)

    # Reference lines (distinct dash styles)
    plt.axhline(ACTIVE_REF_CHI1,  color="k", linestyle="--", linewidth=1.3, alpha=0.8, label="Active ref χ₁")
    plt.axhline(INACTIVE_REF_CHI1, color="k", linestyle=":",  linewidth=1.6, alpha=0.8, label="Inactive ref χ₁")

    plt.xlabel("Time (ns)")
    plt.ylabel("Tyr⁷·⁵³(418) (deg)")
    plt.title(f"{CHI_METRIC_NAME} time series") # replica-average (mean ± SEM)
    plt.ylim(CHI_RANGE[0], CHI_RANGE[1])
    plt.legend(loc=LEGEND_LOC)
    plt.tight_layout()
    plt.savefig(f"timeseries_{CHI_METRIC_NAME}.png", dpi=DPI)
    plt.close()

    # ---- Occupancy vs time plots (active-like and inactive-like), system mean ± SEM
    for occ_label in ["active_like", "inactive_like"]:
        occ_metric = f"chi1_Tyr423_{occ_label}_smooth"
        occ_ts = chi_aug[chi_aug["metric"] == occ_metric].copy()
        if occ_ts.empty:
            continue

        occ_summary = system_mean_sem_over_reps(occ_ts)
        occ_summary.to_csv(f"chi_{occ_label}_occupancy_system_summary.csv", index=False)

        plt.figure()
        for system in systems_order:
            sub = occ_summary[occ_summary["system"] == system].sort_values("time")
            if sub.empty:
                continue
            c = SYSTEM_COLORS.get(system, None)
            x = sub["time"].to_numpy()
            y = sub["mean"].to_numpy()
            ysem = sub["sem"].to_numpy()
            plt.plot(x, y, label=system, color=c)
            if np.isfinite(ysem).any():
                plt.fill_between(x, y - ysem, y + ysem, alpha=0.2, color=c)

        plt.xlabel("Time (ns)")
        plt.ylabel(f"{occ_label.replace('_','-')} occupancy (0–1)")
        #plt.title(f"{CHI_METRIC_NAME}: {occ_label.replace('_','-')} occupancy vs time (mean ± SEM)")
        plt.title(f"{CHI_METRIC_NAME} state occupancy vs time (mean ± SEM)")
        plt.ylim(-0.05, 1.05)
        plt.legend(loc=LEGEND_LOC)
        plt.tight_layout()
        plt.savefig(f"timeseries_{CHI_METRIC_NAME}_{occ_label}_occupancy.png", dpi=DPI)
        plt.close()

    # ---- Correlation: binned occupancy vs distance metrics (both occupancies x both distances)
    for dist_metric in CORR_DISTANCE_METRICS:
        dist_ts = all_ts[all_ts["metric"] == dist_metric].copy().sort_values(["system", "replica", "time"])
        if dist_ts.empty:
            continue

        for occ_label in ["active_like", "inactive_like"]:
            occ_metric = f"chi1_Tyr423_{occ_label}_smooth"
            occ_ts_raw = chi_aug[chi_aug["metric"] == occ_metric].copy().sort_values(["system", "replica", "time"])
            if occ_ts_raw.empty:
                continue

            corr_records = []
            for system in systems_order:
                for rep in REPS:
                    d_rep = dist_ts[(dist_ts["system"] == system) & (dist_ts["replica"] == rep)]
                    o_rep = occ_ts_raw[(occ_ts_raw["system"] == system) & (occ_ts_raw["replica"] == rep)]
                    if d_rep.empty or o_rep.empty:
                        continue

                    aligned = align_by_time_interp(d_rep[["time", "value"]], o_rep[["time", "value"]])
                    aligned.rename(columns={"a_value": "dist_A", "b_value": "occ"}, inplace=True)

                    x = aligned["dist_A"].to_numpy()
                    y = aligned["occ"].to_numpy()

                    xmin, xmax = np.nanmin(x), np.nanmax(x)
                    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
                        continue

                    start = CORR_BIN_WIDTH_A * np.floor(xmin / CORR_BIN_WIDTH_A)
                    stop  = CORR_BIN_WIDTH_A * np.ceil(xmax / CORR_BIN_WIDTH_A)
                    bins = np.arange(start, stop + CORR_BIN_WIDTH_A, CORR_BIN_WIDTH_A)

                    inds = np.digitize(x, bins) - 1
                    for bi in range(len(bins) - 1):
                        m = inds == bi
                        if not np.any(m):
                            continue
                        x_center = 0.5 * (bins[bi] + bins[bi + 1])
                        corr_records.append({
                            "system": system,
                            "replica": rep,
                            "distance_metric": dist_metric,
                            "dist_bin_center_A": x_center,
                            "occ_mean_in_bin": float(np.nanmean(y[m])),
                            "n_frames": int(np.sum(m)),
                        })

            if not corr_records:
                continue

            corr_df = pd.DataFrame(corr_records)
            out_rep_csv = f"chi_{occ_label}_occ_vs_{dist_metric}_binned_replica.csv"
            corr_df.to_csv(out_rep_csv, index=False)

            corr_sum = (
                corr_df.groupby(["system", "dist_bin_center_A"])["occ_mean_in_bin"]
                .agg(mean="mean", std="std", n="count")
                .reset_index()
            )
            corr_sum["sem"] = corr_sum.apply(lambda r: (r["std"] / np.sqrt(r["n"])) if r["n"] > 1 else np.nan, axis=1)

            out_sys_csv = f"chi_{occ_label}_occ_vs_{dist_metric}_binned_system_summary.csv"
            corr_sum.to_csv(out_sys_csv, index=False)

            plt.figure()
            for system in systems_order:
                sub = corr_sum[corr_sum["system"] == system].sort_values("dist_bin_center_A")
                if sub.empty:
                    continue
                c = SYSTEM_COLORS.get(system, None)
                x = sub["dist_bin_center_A"].to_numpy()
                y = sub["mean"].to_numpy()
                ysem = sub["sem"].to_numpy()
                plt.plot(x, y, label=system, color=c)
                if np.isfinite(ysem).any():
                    plt.fill_between(x, y - ysem, y + ysem, alpha=0.2, color=c)

            plt.xlabel(f"{dist_metric} (Å)")
            plt.ylabel(f"{occ_label.replace('_','-')} occupancy (0–1)")
            #plt.title(f"{CHI_METRIC_NAME}: {occ_label.replace('_','-')} occupancy vs {dist_metric} (binned, mean ± SEM)")
            plt.title(f"{CHI_METRIC_NAME} state occupancy vs Arg³·⁵⁰-Glu⁶·³⁰ distance")
            plt.ylim(-0.05, 1.05)
            plt.legend(loc=LEGEND_LOC)
            plt.tight_layout()
            plt.savefig(f"compare_{CHI_METRIC_NAME}_{occ_label}_occ_vs_{dist_metric}.png", dpi=DPI)
            plt.close()
            
    # =========================
    # State occupancy summary (stacked bars) — robust
    # =========================
    rep_occ, sys_occ = compute_state_occupancy_from_chi(
        chi_ts=chi_ts,
        active_windows=CHI_ACTIVE_WINDOWS,
        inactive_windows=CHI_INACTIVE_WINDOWS,
        systems_order=systems_order,
    )

    if not rep_occ.empty:
        rep_occ.to_csv("chi1_Tyr423_state_occupancy_replica.csv", index=False)
        sys_occ.to_csv("chi1_Tyr423_state_occupancy_system.csv", index=True)

        plt.figure(figsize=(6, 4))
        bottom = np.zeros(len(sys_occ))

        for col, label in [
            ("active_like", "Active-like"),
            ("inactive_like", "Inactive-like"),
            ("other", "Other"),
        ]:
            vals = sys_occ[col].to_numpy()
            plt.bar(
                sys_occ.index.to_list(),
                vals,
                bottom=bottom,
                label=label,
                color=STATE_COLORS[col],
                edgecolor="black",
                linewidth=0.6,
            )
            bottom += vals

        plt.ylim(0, 1.0)
        plt.ylabel("Fraction of frames")
        plt.title("Tyr⁷·⁵³ (418) χ₁ state occupancy")
        plt.legend(loc=LEGEND_LOC)
        plt.tight_layout()
        plt.savefig("chi1_Tyr423_state_occupancy_summary.png", dpi=DPI)
        plt.close()

    else:
        print("[WARN] State occupancy summary skipped: chi_ts has no CHI_METRIC_NAME rows.")

else:
    chi_missing.append("No chi time series files found anywhere; skipping chi plots.")

# =========================
# Reporting
# =========================
if missing:
    print("\nMissing scalar items (skipped):")
    for m in missing:
        print("  -", m)

if chi_missing:
    print("\nChi items (skipped/missing):")
    for m in chi_missing:
        print("  -", m)

print("\nDone. Wrote CSVs and compare_*.png / timeseries_*.png.")
