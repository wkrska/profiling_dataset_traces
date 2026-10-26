#!/usr/bin/env python3
"""
plot_drift.py

Visualizes drift in Zipfian skew and cardinality over defined windows:
  1. Window Drift Trajectories (figures/drift/drift_trajectories_<dataset>.png):
     Skew s(w) and Cardinality C(w) plotted across window progression for key columns.
  2. Skew Drift Volatility (figures/drift/skew_drift_comparison.png):
     Comparing skew drift range (max(s) - min(s)) and volatility (std_skew) between
     Real-World datasets and Synthetic Benchmarks.
  3. Cumulative Cardinality Saturation Curves (figures/drift/cardinality_growth_curves.png):
     Normalized cardinality expansion (C_<=w / C_total) revealing closed vs open vocabulary.
  4. 2D Drift Phase Space (figures/drift/drift_phase_space.png):
     Cardinality CV (sigma_C / mu_C) vs Skew Drift Range (max(s) - min(s)) mapping
     columns into Static, Skew-Drifting, Cardinality-Drifting, and Dual-Drifting quadrants.
  5. Skew Regime Shift Breakdown (figures/drift/regime_transitions.png):
     Columns exhibiting transitions between Uniform, Gentle Zipf, and Hotspot regimes.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})

DRIFT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "drift")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "drift")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_drift_summaries(results_dir: str = DRIFT_RESULTS_DIR) -> pd.DataFrame:
    """Loads all *_drift_summary.csv files into a single standardized DataFrame."""
    summary_files = glob.glob(os.path.join(results_dir, "*_drift_summary.csv"))
    # Filter out all_datasets_drift_summary.csv if present
    summary_files = [f for f in summary_files if not f.endswith("all_datasets_drift_summary.csv")]

    dfs = []
    for f in summary_files:
        df = pd.read_csv(f)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No *_drift_summary.csv files found in {results_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    
    # Label source type
    def classify_source(ds_name):
        if "core" in ds_name or "bench" in ds_name or "classic" in ds_name or "geo" in ds_name:
            return "Synthetic Benchmark"
        return "Real-World Dataset"

    combined["source_type"] = combined["dataset"].apply(classify_source)
    return combined


def load_window_traces(results_dir: str = DRIFT_RESULTS_DIR) -> dict:
    """Loads all *_drift_windows.csv files mapped by dataset name."""
    trace_files = glob.glob(os.path.join(results_dir, "*_drift_windows.csv"))
    traces = {}
    for f in trace_files:
        base = os.path.basename(f).replace("_drift_windows.csv", "")
        traces[base] = pd.read_csv(f)
    return traces


def plot_dataset_drift_trajectories(traces: dict, dataset_name: str, key_columns: list = None):
    """
    Plots dual-panel trajectory for a dataset:
      Panel A: Zipfian Skew s(w) over window progression with Uniform/Hotspot boundaries.
      Panel B: Distinct Cardinality C(w) over window progression (log scale).
    """
    if dataset_name not in traces:
        print(f"[-] Dataset {dataset_name} not found in traces.")
        return

    df = traces[dataset_name]
    available_cols = df["column"].unique()

    if not key_columns:
        # Pick top 5-6 columns with highest skew drift range or distinct values
        summary_drift = df.groupby("column").agg(
            skew_range=("zipf_s_skew", lambda x: x.max() - x.min()),
            max_c=("cardinality", "max")
        ).sort_values(by="skew_range", ascending=False)
        key_columns = summary_drift.index[:6].tolist()
    else:
        key_columns = [c for c in key_columns if c in available_cols]

    if not key_columns:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    palette = sns.color_palette("tab10", len(key_columns))

    for idx, col in enumerate(key_columns):
        col_df = df[df["column"] == col].sort_values("window_idx")
        x = col_df["window_idx"]
        ax1.plot(x, col_df["zipf_s_skew"], marker="o", markersize=4, linewidth=2,
                 label=col, color=palette[idx])
        ax2.plot(x, col_df["cardinality"], marker="s", markersize=4, linewidth=2,
                 label=col, color=palette[idx])

    # Skew Panel styling
    ax1.axhline(0.01, color="gray", linestyle="--", alpha=0.7, label="Uniform (s=0.01)")
    ax1.axhline(2.0, color="purple", linestyle="--", alpha=0.7, label="Gentle Zipf Boundary (s=2.0)")
    ax1.set_ylabel("Zipf Skew Parameter ($s$)")
    ax1.set_title(f"Window Drift Trajectories: {dataset_name}\n(A) Zipfian Skew Parameter ($s$) Progression")
    ax1.legend(loc="upper right", framealpha=0.9, ncol=2)

    # Cardinality Panel styling
    ax2.set_yscale("log")
    ax2.set_xlabel("Window Index ($w$)")
    ax2.set_ylabel("Distinct Cardinality ($C_w$, Log Scale)")
    ax2.set_title("(B) Local Distinct Value Cardinality ($C_w$) Progression")
    ax2.legend(loc="upper right", framealpha=0.9, ncol=2)

    out_file = os.path.join(OUTPUT_DIR, f"drift_trajectories_{dataset_name}.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved {out_file}")


def plot_cardinality_growth_curves(traces: dict):
    """
    Plots cumulative cardinality growth (C_<=w / C_total) vs normalized window progression.
    Shows the contrast between rapidly saturating categorical columns vs continuous-growth keys.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # Collect a selection of diverse columns across datasets
    curve_data = []
    for ds_name, df in traces.items():
        for col in df["column"].unique():
            col_df = df[df["column"] == col].sort_values("window_idx")
            total_c = col_df["cumulative_cardinality"].max()
            if total_c <= 1:
                continue
            num_w = len(col_df)
            norm_w = col_df["window_idx"] / (num_w - 1) if num_w > 1 else np.array([0.0])
            norm_c = col_df["cumulative_cardinality"] / total_c
            
            # Select representative archetypes
            archetype = "General"
            if total_c < 20:
                archetype = "Low-Domain Flag (C < 20)"
            elif "datetime" in col or "time" in col or "date" in col or "timestamp" in col:
                archetype = "Temporal / Continuous Timestamp"
            elif "id" in col.lower() or "key" in col.lower():
                archetype = "High-Domain Entity ID"
            else:
                archetype = "Categorical / Numeric Attribute"

            for w_val, c_val in zip(norm_w, norm_c):
                curve_data.append({
                    "dataset": ds_name,
                    "column": f"{ds_name}.{col}",
                    "normalized_window": w_val,
                    "normalized_cardinality": c_val,
                    "archetype": archetype
                })

    if not curve_data:
        return

    plot_df = pd.DataFrame(curve_data)
    sns.lineplot(
        data=plot_df, x="normalized_window", y="normalized_cardinality",
        hue="archetype", style="archetype", linewidth=2.5, ax=ax
    )
    ax.set_title("Empirical Cumulative Cardinality Saturation ($C_{\\leq w} / C_{\\text{total}}$)")
    ax.set_xlabel("Normalized Window Progression ($w / W_{\\max}$)")
    ax.set_ylabel("Observed Distinct Fraction ($C_{\\leq w} / C_{\\text{total}}$)")
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5)

    out_file = os.path.join(OUTPUT_DIR, "cardinality_growth_curves.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved {out_file}")


def plot_skew_drift_comparison(summary_df: pd.DataFrame):
    """
    Compares Skew Drift Range (max(s) - min(s)) and Skew Volatility (std_skew)
    between Real-World Datasets and Synthetic Benchmarks.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Skew Drift Range Boxplot / Strip
    sns.boxplot(
        data=summary_df, x="source_type", y="skew_drift_range", hue="source_type",
        palette={"Real-World Dataset": "#2ca02c", "Synthetic Benchmark": "#e25438"},
        legend=False, ax=ax1, width=0.45, boxprops=dict(alpha=0.6)
    )
    sns.stripplot(
        data=summary_df, x="source_type", y="skew_drift_range",
        color="black", alpha=0.6, jitter=0.2, size=7, ax=ax1
    )
    ax1.set_title("(A) Skew Drift Range ($\Delta s = s_{\\max} - s_{\\min}$)")
    ax1.set_xlabel("Dataset Source")
    ax1.set_ylabel("Skew Drift Range ($\Delta s$)")

    # Panel 2: Step-by-Step Skew Volatility (max_step_skew_drift)
    sns.boxplot(
        data=summary_df, x="source_type", y="max_step_skew_drift", hue="source_type",
        palette={"Real-World Dataset": "#2ca02c", "Synthetic Benchmark": "#e25438"},
        legend=False, ax=ax2, width=0.45, boxprops=dict(alpha=0.6)
    )
    sns.stripplot(
        data=summary_df, x="source_type", y="max_step_skew_drift",
        color="black", alpha=0.6, jitter=0.2, size=7, ax=ax2
    )
    ax2.set_title("(B) Maximum Step Skew Delta ($\max |\Delta s|$)")
    ax2.set_xlabel("Dataset Source")
    ax2.set_ylabel("Max Window-to-Window Delta ($|\Delta s|$)")

    fig.suptitle("Drift Volatility: Real-World Datasets vs Synthetic Benchmarks", fontsize=16)
    out_file = os.path.join(OUTPUT_DIR, "skew_drift_comparison.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved {out_file}")


def plot_drift_phase_space(summary_df: pd.DataFrame):
    """
    2D Phase Space of Distributional Drift:
      X-axis: Cardinality Coefficient of Variation (sigma_C / mu_C)
      Y-axis: Skew Drift Range (max(s) - min(s))
      Categorizes columns into 4 stability quadrants.
    """
    fig, ax = plt.subplots(figsize=(10.5, 8))

    palette = {"Real-World Dataset": "#2ca02c", "Synthetic Benchmark": "#e25438"}

    sns.scatterplot(
        data=summary_df, x="cardinality_cv", y="skew_drift_range",
        hue="source_type", style="source_type", palette=palette, s=90, alpha=0.8, ax=ax
    )

    # Threshold guidelines
    cv_thresh = 0.15
    skew_thresh = 0.25

    ax.axvline(cv_thresh, color="gray", linestyle="--", alpha=0.6)
    ax.axhline(skew_thresh, color="gray", linestyle="--", alpha=0.6)

    # Quadrant labels
    max_x = max(summary_df["cardinality_cv"].max() * 1.05, 0.5)
    max_y = max(summary_df["skew_drift_range"].max() * 1.05, 1.0)
    ax.set_xlim(-0.02, max_x)
    ax.set_ylim(-0.05, max_y)

    ax.text(cv_thresh * 0.1, skew_thresh * 0.2, "I. Static\n(Stable C & s)",
            fontsize=12, fontweight="bold", color="darkblue", alpha=0.7)
    ax.text(cv_thresh * 0.1, max_y * 0.85, "II. Skew-Drifting\n(Stable C, Shifting s)",
            fontsize=12, fontweight="bold", color="darkred", alpha=0.7)
    ax.text(max_x * 0.65, skew_thresh * 0.2, "III. Cardinality-Drifting\n(Shifting C, Stable s)",
            fontsize=12, fontweight="bold", color="darkgreen", alpha=0.7)
    ax.text(max_x * 0.65, max_y * 0.85, "IV. Dual-Drifting\n(Highly Dynamic C & s)",
            fontsize=12, fontweight="bold", color="purple", alpha=0.7)

    # Annotate prominent drifting columns
    top_drifters = summary_df.nlargest(5, "skew_drift_range")
    for _, row in top_drifters.iterrows():
        label = f"{row['dataset']}.{row['column']}"
        ax.annotate(label, (row["cardinality_cv"], row["skew_drift_range"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9, alpha=0.85)

    ax.set_title("2D Drift Phase Space: Distinct Cardinality Drift vs Skew Drift")
    ax.set_xlabel("Cardinality Volatility (Coefficient of Variation, $\\sigma_C / \\mu_C$)")
    ax.set_ylabel("Skew Drift Range ($\Delta s = s_{\\max} - s_{\\min}$)")
    ax.legend(loc="upper right")

    out_file = os.path.join(OUTPUT_DIR, "drift_phase_space.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved {out_file}")


def plot_regime_transitions(summary_df: pd.DataFrame):
    """Visualizes columns that cross critical regime boundaries (Uniform <-> Gentle Zipf <-> Hotspot)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Categorize regime behavior
    def categorize_regimes(row):
        shifts = row["skew_regime_shifts"]
        visited = row["regimes_visited"]
        if shifts == 0:
            return f"Stable ({visited})"
        return f"Transitioning ({visited})"

    summary_df["regime_category"] = summary_df.apply(categorize_regimes, axis=1)
    counts = summary_df["regime_category"].value_counts()

    colors = sns.color_palette("viridis", len(counts))
    counts.plot(kind="barh", ax=ax, color=colors, edgecolor="black", alpha=0.8)

    ax.set_title("Distribution of Skew Regime Stability across Profiled Columns")
    ax.set_xlabel("Number of Columns")
    ax.set_ylabel("Regime Behavior")

    for i, v in enumerate(counts):
        ax.text(v + 0.2, i, str(v), va="center", fontweight="bold")

    out_file = os.path.join(OUTPUT_DIR, "regime_transitions.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved {out_file}")


if __name__ == "__main__":
    print(f"[+] Loading drift profiles from: {DRIFT_RESULTS_DIR}")
    summary_df = load_all_drift_summaries()
    traces = load_window_traces()

    print(f"    Loaded {len(summary_df)} profiled columns across {summary_df['dataset'].nunique()} datasets.")

    print("\n[+] Generating Drift Trajectory plots for datasets...")
    for ds_name in traces.keys():
        plot_dataset_drift_trajectories(traces, ds_name)

    print("\n[+] Generating Cross-Dataset Drift Comparative Visualizations...")
    plot_cardinality_growth_curves(traces)
    plot_skew_drift_comparison(summary_df)
    plot_drift_phase_space(summary_df)
    plot_regime_transitions(summary_df)

    print(f"\n[✓] All drift plots successfully saved to: {OUTPUT_DIR}/")

