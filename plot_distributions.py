#!/usr/bin/env python3
"""
plot_distributions.py

Generates detailed histograms and 2D comparative scatter plots for:
  - Total Records (N)
  - Cardinality (C)
  - NDV Ratio (f_cr = C / N)
  - Zipfian Skew Parameter s

Data Sources:
  - New Profiled Real-World & Multimodal Datasets (NYC Taxi, FineWeb-Edu, GitHub Events, OpenAQ, LAION-2B, ML BitcoinHeist, Menu Dish)
  - Profiled Benchmark Datasets (OpenFormat core, bi, classic, geo, log, ml)
  - Comprehensive Survey of 2,845 real-world columns (Public BI, ClickHouse, IMDb, Yelp, EDGAR, UCI-ML, Geonames)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Row counts for known profiled datasets to attach total records
RECORD_COUNTS = {
    "nyc_taxi_2023_01": 3066766,
    "fineweb_edu_sample": 726000,
    "gh_events": 161519,
    "openaq_2023": 3046,
    "laion_sample": 5329780,
    "BitcoinHeistData": 2916697,
    "Dish": 428146,
    "clickbench_hits": 99997497
}

def load_data():
    """Loads and standardizes all profile results."""
    dfs = []
    
    # 1. Real-world dataset survey (feature_olap.csv)
    real_path = "/mnt/hdd0/database_data/EvaluationOfColumnarFormats/OpenFormat/benchmark/feature/outputs/feature_olap.csv"
    if os.path.exists(real_path):
        df_real = pd.read_csv(real_path)
        standard_real = pd.DataFrame({
            "source": "Zeng et al. Survey (2,845 cols)",
            "dataset": df_real["dataset"],
            "total_records": df_real["nrows"],
            "cardinality": df_real["unique_num"],
            "ndv_ratio": df_real["cardinality"],
            "skew": df_real["zipf"]
        })
        dfs.append(standard_real)
        print(f"Loaded {len(standard_real)} columns from feature_olap.csv")

    # 2. Benchmarks profiled (benchmark_profiles.csv)
    bench_path = "benchmark_profiles.csv"
    if os.path.exists(bench_path):
        df_bench = pd.read_csv(bench_path)
        standard_bench = pd.DataFrame({
            "source": "OpenFormat Benchmarks",
            "dataset": df_bench["workload"],
            "total_records": df_bench["total_records"],
            "cardinality": df_bench["cardinality"],
            "ndv_ratio": df_bench["ndv_ratio"],
            "skew": df_bench["zipf_s_skew"]
        })
        dfs.append(standard_bench)
        print(f"Loaded {len(standard_bench)} benchmark columns from benchmark_profiles.csv")

    # 3. New individual dataset profile traces (*_profile.csv)
    new_profiles = sorted(glob.glob("*_profile.csv"))
    for p in new_profiles:
        base = os.path.basename(p).replace("_profile.csv", "")
        df_new = pd.read_csv(p)
        total_rows = RECORD_COUNTS.get(base, 1000000)
        standard_new = pd.DataFrame({
            "source": "New Profiled Datasets",
            "dataset": base,
            "total_records": total_rows,
            "cardinality": df_new["cardinality"],
            "ndv_ratio": df_new["ndv_ratio"],
            "skew": df_new["zipf_s_skew"]
        })
        dfs.append(standard_new)
        print(f"Loaded {len(standard_new)} columns from {p} ({base})")

    if not dfs:
        raise FileNotFoundError("No profile data found.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined

def plot_individual_histograms(df):
    """Generates detailed histograms for NDV, Cardinality, Skew, and Total Records."""
    palette = {
        "Zeng et al. Survey (2,845 cols)": "#2b5c8f", 
        "OpenFormat Benchmarks": "#e25438",
        "New Profiled Datasets": "#2ca02c"
    }

    # 1. NDV Ratio Histogram
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ndv_data = df["ndv_ratio"].clip(0.0, 1.0)
    sns.histplot(
        data=df, x="ndv_ratio", hue="source",
        bins=50, kde=True, element="step", stat="density", common_norm=False,
        palette=palette, alpha=0.55, ax=ax
    )
    median_val = ndv_data.median()
    mean_val = ndv_data.mean()
    ax.axvline(median_val, color="navy", linestyle=":", linewidth=2, label=f"Median ({median_val:.4f})")
    ax.axvline(mean_val, color="darkred", linestyle="--", linewidth=2, label=f"Mean ({mean_val:.4f})")
    ax.set_title("Distribution of NDV Ratio ($f_{cr} = \\text{NDV} / N$)")
    ax.set_xlabel("NDV Ratio (Distinct Values / Total Rows)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "histogram_ndv_ratio.png"), dpi=300)
    plt.close(fig)
    print("Saved histogram_ndv_ratio.png")

    # 2. Cardinality Histogram (Log Scale)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    card_data = df["cardinality"].replace(0, np.nan).dropna()
    bins = np.logspace(0, np.log10(max(card_data.max(), 10)), 50)
    sns.histplot(
        data=df[df["cardinality"] > 0], x="cardinality", hue="source",
        bins=bins, element="step", stat="density", common_norm=False,
        palette=palette, alpha=0.55, ax=ax
    )
    ax.set_xscale("log")
    ax.set_title("Distribution of Distinct Value Cardinality ($C$)")
    ax.set_xlabel("Cardinality (Unique Non-Null Count, Log Scale)")
    ax.set_ylabel("Density")
    fig.savefig(os.path.join(OUTPUT_DIR, "histogram_cardinality.png"), dpi=300)
    plt.close(fig)
    print("Saved histogram_cardinality.png")

    # 3. Zipfian Skew Parameter s Histogram
    fig, ax = plt.subplots(figsize=(8.5, 6))
    skew_data = df["skew"].clip(0.0, 7.0)
    sns.histplot(
        data=df[df["skew"] <= 7.0], x="skew", hue="source",
        bins=45, kde=True, element="step", stat="density", common_norm=False,
        palette=palette, alpha=0.55, ax=ax
    )
    ax.axvline(0.01, color="gray", linestyle="--", label="Uniform ($s \\leq 0.01$)")
    ax.axvline(2.0, color="purple", linestyle="--", label="Gentle Zipf ($0.01 < s \\leq 2.0$)")
    ax.text(0.05, ax.get_ylim()[1] * 0.85, "Uniform", fontsize=10, color="gray")
    ax.text(0.7, ax.get_ylim()[1] * 0.85, "Gentle Zipf\n(Long Tail)", fontsize=10, color="purple")
    ax.text(2.3, ax.get_ylim()[1] * 0.85, "Hotspot ($s > 2.0$)", fontsize=10, color="darkred")
    ax.set_title("Distribution of Zipfian Skew Exponent ($s$)")
    ax.set_xlabel("Zipf Skew Parameter $s$ (Fitted Exponent)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "histogram_skew.png"), dpi=300)
    plt.close(fig)
    print("Saved histogram_skew.png")

    # 4. Total Records Histogram (Log Scale)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    recs = df["total_records"].dropna()
    bins = np.logspace(np.log10(recs.min()), np.log10(recs.max()), 45)
    sns.histplot(
        data=df, x="total_records", hue="source",
        bins=bins, element="step", stat="density", common_norm=False,
        palette=palette, alpha=0.55, ax=ax
    )
    ax.set_xscale("log")
    ax.set_title("Distribution of Total Records ($N$)")
    ax.set_xlabel("Total Records ($N$, Log Scale)")
    ax.set_ylabel("Density")
    fig.savefig(os.path.join(OUTPUT_DIR, "histogram_total_records.png"), dpi=300)
    plt.close(fig)
    print("Saved histogram_total_records.png")

def plot_combined_detailed_histograms(df):
    """Creates a comprehensive 4-panel figure summarizing all 4 distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    palette = {
        "Zeng et al. Survey (2,845 cols)": "#2b5c8f", 
        "OpenFormat Benchmarks": "#e25438",
        "New Profiled Datasets": "#2ca02c"
    }

    # 1. NDV Ratio
    sns.histplot(
        data=df, x="ndv_ratio", hue="source",
        bins=40, kde=True, element="step", stat="density", common_norm=False,
        palette=palette, alpha=0.55, ax=axes[0, 0]
    )
    axes[0, 0].set_title("(A) NDV Ratio ($f_{cr} = \\text{NDV} / N$)")
    axes[0, 0].set_xlabel("NDV Ratio")
    axes[0, 0].set_ylabel("Density")

    # 2. Cardinality
    card_data = df[df["cardinality"] > 0]
    bins_c = np.logspace(0, np.log10(card_data["cardinality"].max()), 40)
    sns.histplot(
        data=card_data, x="cardinality", hue="source",
        bins=bins_c, element="step", stat="density", common_norm=False,
        palette=palette, alpha=0.55, ax=axes[0, 1]
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("(B) Distinct Value Cardinality ($C$)")
    axes[0, 1].set_xlabel("Cardinality ($C$, Log Scale)")
    axes[0, 1].set_ylabel("Density")

    # 3. Skew
    skew_data = df[df["skew"] <= 7.0]
    sns.histplot(
        data=skew_data, x="skew", hue="source",
        bins=40, kde=True, element="step", stat="density", common_norm=False,
        palette=palette, alpha=0.55, ax=axes[1, 0]
    )
    axes[1, 0].axvline(0.01, color="gray", linestyle="--", alpha=0.7)
    axes[1, 0].axvline(2.0, color="purple", linestyle="--", alpha=0.7)
    axes[1, 0].set_title("(C) Zipfian Skew Parameter ($s$)")
    axes[1, 0].set_xlabel("Zipfian Skew Exponent ($s$)")
    axes[1, 0].set_ylabel("Density")

    # 4. Total Records
    bins_r = np.logspace(np.log10(df["total_records"].min()), np.log10(df["total_records"].max()), 40)
    sns.histplot(
        data=df, x="total_records", hue="source",
        bins=bins_r, element="step", stat="density", common_norm=False,
        palette=palette, alpha=0.55, ax=axes[1, 1]
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("(D) Total Records ($N$)")
    axes[1, 1].set_xlabel("Total Records ($N$, Log Scale)")
    axes[1, 1].set_ylabel("Density")

    fig.suptitle("Empirical Column Properties across 3,000+ Surveyed & Profiled Datasets", fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "detailed_histograms.png"), dpi=300)
    plt.close(fig)
    print("Saved detailed_histograms.png")

def plot_2d_comparisons(df):
    """
    Generates 3 separate 2D scatter plots comparing:
      1. Total Records vs Cardinality
      2. Total Records vs Skew
      3. Cardinality vs Skew
    """
    palette = {
        "Zeng et al. Survey (2,845 cols)": "#2b5c8f", 
        "OpenFormat Benchmarks": "#e25438",
        "New Profiled Datasets": "#2ca02c"
    }

    # 1. Total Records vs Cardinality
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sns.scatterplot(
        data=df, x="total_records", y="cardinality", hue="source", style="source",
        palette=palette, alpha=0.75, s=65, ax=ax
    )
    min_val = min(df["total_records"].min(), df[df["cardinality"] > 0]["cardinality"].min())
    max_val = max(df["total_records"].max(), df["cardinality"].max())
    ref_line = np.logspace(np.log10(max(min_val, 1)), np.log10(max_val), 100)
    ax.plot(ref_line, ref_line, color="black", linestyle="--", linewidth=1.5, label="Upper Bound ($C = N$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Total Records ($N$) vs Distinct Cardinality ($C$)")
    ax.set_xlabel("Total Records ($N$, Log Scale)")
    ax.set_ylabel("Cardinality ($C$, Log Scale)")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(OUTPUT_DIR, "scatter_records_vs_cardinality.png"), dpi=300)
    plt.close(fig)
    print("Saved scatter_records_vs_cardinality.png")

    # 2. Total Records vs Zipfian Skew
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sns.scatterplot(
        data=df, x="total_records", y="skew", hue="source", style="source",
        palette=palette, alpha=0.75, s=65, ax=ax
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.2, min(df["skew"].max() * 1.05, 8.5))
    ax.axhline(0.01, color="gray", linestyle="--", alpha=0.7, label="Uniform ($s \\leq 0.01$)")
    ax.axhline(2.0, color="purple", linestyle="--", alpha=0.7, label="Gentle Zipf Boundary ($s = 2.0$)")
    ax.set_title("Total Records ($N$) vs Zipfian Skew ($s$)")
    ax.set_xlabel("Total Records ($N$, Log Scale)")
    ax.set_ylabel("Zipfian Skew Exponent ($s$)")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(OUTPUT_DIR, "scatter_records_vs_skew.png"), dpi=300)
    plt.close(fig)
    print("Saved scatter_records_vs_skew.png")

    # 3. Cardinality vs Zipfian Skew
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    card_filtered = df[df["cardinality"] > 0]
    sns.scatterplot(
        data=card_filtered, x="cardinality", y="skew", hue="source", style="source",
        palette=palette, alpha=0.75, s=65, ax=ax
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.2, min(df["skew"].max() * 1.05, 8.5))
    ax.axhline(0.01, color="gray", linestyle="--", alpha=0.7, label="Uniform ($s \\leq 0.01$)")
    ax.axhline(2.0, color="purple", linestyle="--", alpha=0.7, label="Gentle Zipf Boundary ($s = 2.0$)")
    ax.set_title("Distinct Cardinality ($C$) vs Zipfian Skew ($s$)")
    ax.set_xlabel("Distinct Cardinality ($C$, Log Scale)")
    ax.set_ylabel("Zipfian Skew Exponent ($s$)")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(OUTPUT_DIR, "scatter_cardinality_vs_skew.png"), dpi=300)
    plt.close(fig)
    print("Saved scatter_cardinality_vs_skew.png")

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Total datapoints (columns): {len(df)}")
    print(df.groupby("source")[["total_records", "cardinality", "ndv_ratio", "skew"]].describe())

    print("\nGenerating histograms...")
    plot_individual_histograms(df)
    plot_combined_detailed_histograms(df)

    print("\nGenerating 2D comparison scatter plots...")
    plot_2d_comparisons(df)

    print(f"\nAll plots saved to directory: {os.path.abspath(OUTPUT_DIR)}/")
