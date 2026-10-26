#!/usr/bin/env python3
"""
profile_drift.py

Profiles drift in Zipfian skew (s) and distinct cardinality (C) over defined
windows for tabular datasets (Parquet, CSV, TSV, JSON Lines).

Generates:
  1. Detailed window-by-window trace CSV: <dataset>_drift_windows.csv
  2. Column-level aggregated summary CSV:  <dataset>_drift_summary.csv
  3. Combined multi-dataset summary CSV:   all_datasets_drift_summary.csv (when batch profiling)

Metrics computed per window:
  - Cardinality (C_w) & NDV Ratio (f_cr,w)
  - Cumulative Cardinality (C_<=w) & Novelty Ratio (new unique values / C_w)
  - Jaccard similarity of active keys between consecutive windows
  - Zipfian Skew Parameter (s_w) via curve_fit / log-log regression
  - Step Skew Delta (Delta s = s_w - s_{w-1})
  - Step Cardinality Delta (Delta C = C_w - C_{w-1})
  - Top-1 and Top-5 value dominance ratios
  - Jensen-Shannon Divergence (JSD) between adjacent window distributions
  - Skew Regime: Uniform (s <= 0.01), Gentle Zipf (0.01 < s <= 2.0), Hotspot (s > 2.0)
"""

import sys
import os
import glob
import json
import gzip
import argparse
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "drift")
DEFAULT_DATA_DIR = "/mnt/hdd0/database_data"

# Known target datasets in the repository
KNOWN_TARGETS = [
    ("nyc_taxi_2023_01.parquet", {"window_size": 200000, "time_col": "tpep_pickup_datetime"}),
    ("openaq_2023.csv.gz", {"window_size": 300, "time_col": "datetime"}),
    ("gh_events.json.gz", {"window_size": 15000, "time_col": "created_at"}),
    ("fineweb_edu_sample.parquet", {"window_size": 50000, "time_col": None}),
    ("laion_sample.parquet", {"window_size": 250000, "time_col": None}),
    ("openformat_benchmarks/core_r1000000_c20.csv", {"window_size": 100000, "time_col": None}),
    ("FormatEvalData/ml_data/BitcoinHeistData.csv", {"window_size": 200000, "time_col": None}),
    ("FormatEvalData/menu/Dish.csv", {"window_size": 40000, "time_col": None}),
]


def compute_zipf_s(frequencies: np.ndarray, C: int) -> float:
    """
    Fits Zipfian parameter s given sorted frequency array (descending) and cardinality C.
    PMF: P(r; s, C) = (1 / r^s) / H(C, s)
    """
    if C <= 1:
        return 0.0

    ranks = np.arange(1, C + 1, dtype=np.float64)
    p_empirical = frequencies / np.sum(frequencies)

    def zipf_pmf(r, s):
        H_C_s = np.sum(1.0 / (ranks ** s))
        return (1.0 / (r ** s)) / H_C_s

    try:
        popt, _ = curve_fit(zipf_pmf, ranks, p_empirical, p0=[1.0], bounds=(0.0, 10.0), maxfev=500)
        return float(popt[0])
    except Exception:
        valid_idx = p_empirical > 0
        log_ranks = np.log(ranks[valid_idx])
        log_freqs = np.log(p_empirical[valid_idx])
        slope, _ = np.polyfit(log_ranks, log_freqs, 1)
        return float(max(0.0, -slope))


def compute_js_divergence(p_counts: pd.Series, q_counts: pd.Series) -> float:
    """
    Computes Jensen-Shannon Divergence between two value frequency distributions.
    Returns divergence in [0.0, 1.0] using base 2.
    """
    if p_counts is None or q_counts is None or len(p_counts) == 0 or len(q_counts) == 0:
        return 0.0

    all_vals = p_counts.index.union(q_counts.index)
    p = p_counts.reindex(all_vals, fill_value=0).values.astype(float)
    q = q_counts.reindex(all_vals, fill_value=0).values.astype(float)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return 0.0
    p /= p_sum
    q /= q_sum
    jsd = float(jensenshannon(p, q, base=2))
    return round(float(np.nan_to_num(jsd, nan=0.0)), 4)


def classify_skew_regime(s: float) -> str:
    if s <= 0.01:
        return "Uniform"
    elif s <= 2.0:
        return "Gentle Zipf"
    else:
        return "Hotspot"


def build_window_slices(total_rows: int, window_size: int, stride: int) -> list:
    """Computes (start, end) row indices for tumbling or sliding windows."""
    slices = []
    start = 0
    while start < total_rows:
        end = min(start + window_size, total_rows)
        slices.append((start, end))
        if end >= total_rows:
            break
        start += stride
    return slices


def profile_column_drift(
    col_series: pd.Series,
    window_slices: list,
    dataset_name: str,
    col_name: str,
    time_series: pd.Series = None
) -> tuple:
    """
    Profiles skew and cardinality drift across windows for a single column.
    Returns:
      (window_records_df, column_summary_dict)
    """
    N = len(col_series)
    non_null_all = col_series.dropna()
    if len(non_null_all) > 0 and isinstance(non_null_all.iloc[0], (dict, list)):
        col_series = col_series.astype(str)
        non_null_all = col_series.dropna()
    total_distinct = non_null_all.nunique()

    cum_unique_set = set()
    prev_counts = None
    prev_skew = None
    prev_cardinality = None
    prev_unique_set = None

    window_records = []

    for w_idx, (start, end) in enumerate(window_slices):
        w_raw = col_series.iloc[start:end]
        w_len = len(w_raw)
        non_nulls = w_raw.dropna()
        w_non_null = len(non_nulls)
        w_null_ratio = (w_len - w_non_null) / w_len if w_len > 0 else 0.0

        start_time_val = None
        end_time_val = None
        if time_series is not None and len(time_series) > start:
            start_time_val = str(time_series.iloc[start])
            end_time_val = str(time_series.iloc[min(end - 1, len(time_series) - 1)])

        if w_non_null == 0:
            rec = {
                "dataset": dataset_name,
                "column": col_name,
                "window_idx": w_idx,
                "start_row": start,
                "end_row": end,
                "window_size": w_len,
                "start_time": start_time_val,
                "end_time": end_time_val,
                "non_null_count": 0,
                "null_ratio": 1.0,
                "cardinality": 0,
                "cumulative_cardinality": len(cum_unique_set),
                "ndv_ratio": 0.0,
                "new_values_ratio": 0.0,
                "jaccard_overlap": 0.0 if prev_unique_set else 1.0,
                "zipf_s_skew": 0.0,
                "delta_skew": 0.0,
                "delta_cardinality": 0,
                "top1_share": 0.0,
                "top5_share": 0.0,
                "js_divergence": 0.0,
                "skew_regime": "Uniform"
            }
            window_records.append(rec)
            continue

        # Complex objects normalization
        if isinstance(non_nulls.iloc[0], (dict, list)):
            non_nulls = non_nulls.astype(str)

        val_counts = non_nulls.value_counts()
        w_cardinality = len(val_counts)
        w_ndv_ratio = w_cardinality / w_non_null
        w_unique_set = set(val_counts.index)

        # Cumulative and novelty tracking
        new_values_count = len(w_unique_set - cum_unique_set)
        cum_unique_set.update(w_unique_set)
        cum_cardinality = len(cum_unique_set)
        new_values_ratio = new_values_count / w_cardinality if w_cardinality > 0 else 0.0

        # Jaccard key overlap with previous window
        if prev_unique_set is not None:
            inter_size = len(w_unique_set & prev_unique_set)
            union_size = len(w_unique_set | prev_unique_set)
            jaccard_overlap = inter_size / union_size if union_size > 0 else 1.0
        else:
            jaccard_overlap = 1.0

        # Skew parameter
        zipf_s = compute_zipf_s(val_counts.values, w_cardinality)

        # Delta metrics
        delta_skew = (zipf_s - prev_skew) if prev_skew is not None else 0.0
        delta_card = (w_cardinality - prev_cardinality) if prev_cardinality is not None else 0

        # Dominance / concentration
        top1_share = val_counts.iloc[0] / w_non_null if w_cardinality > 0 else 0.0
        top5_share = val_counts.iloc[:5].sum() / w_non_null if w_cardinality > 0 else 0.0

        # Jensen-Shannon Divergence
        jsd = compute_js_divergence(prev_counts, val_counts) if prev_counts is not None else 0.0

        regime = classify_skew_regime(zipf_s)

        rec = {
            "dataset": dataset_name,
            "column": col_name,
            "window_idx": w_idx,
            "start_row": start,
            "end_row": end,
            "window_size": w_len,
            "start_time": start_time_val,
            "end_time": end_time_val,
            "non_null_count": w_non_null,
            "null_ratio": round(w_null_ratio, 4),
            "cardinality": int(w_cardinality),
            "cumulative_cardinality": int(cum_cardinality),
            "ndv_ratio": round(w_ndv_ratio, 4),
            "new_values_ratio": round(new_values_ratio, 4),
            "jaccard_overlap": round(jaccard_overlap, 4),
            "zipf_s_skew": round(zipf_s, 4),
            "delta_skew": round(delta_skew, 4),
            "delta_cardinality": int(delta_card),
            "top1_share": round(top1_share, 4),
            "top5_share": round(top5_share, 4),
            "js_divergence": round(jsd, 4),
            "skew_regime": regime
        }
        window_records.append(rec)

        prev_counts = val_counts
        prev_skew = zipf_s
        prev_cardinality = w_cardinality
        prev_unique_set = w_unique_set

    # Column-level summary calculation
    skews = [r["zipf_s_skew"] for r in window_records]
    cards = [r["cardinality"] for r in window_records]
    step_skews = [abs(r["delta_skew"]) for r in window_records[1:]]
    regimes = [r["skew_regime"] for r in window_records]

    regime_shifts = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
    regimes_visited = " -> ".join(dict.fromkeys(regimes))

    mean_c = float(np.mean(cards)) if cards else 0.0
    std_c = float(np.std(cards)) if cards else 0.0
    cv_c = (std_c / mean_c) if mean_c > 0 else 0.0

    mean_s = float(np.mean(skews)) if skews else 0.0
    std_s = float(np.std(skews)) if skews else 0.0
    min_s = float(min(skews)) if skews else 0.0
    max_s = float(max(skews)) if skews else 0.0
    skew_drift_range = max_s - min_s

    saturation_ratio = (cards[0] / total_distinct) if total_distinct > 0 and cards else 1.0

    col_summary = {
        "dataset": dataset_name,
        "column": col_name,
        "dtype": str(col_series.dtype),
        "total_records": N,
        "num_windows": len(window_records),
        "window_size": int(np.mean([r["window_size"] for r in window_records])) if window_records else 0,
        "total_cardinality": int(total_distinct),
        "min_cardinality": int(min(cards)) if cards else 0,
        "max_cardinality": int(max(cards)) if cards else 0,
        "mean_cardinality": round(mean_c, 2),
        "std_cardinality": round(std_c, 2),
        "cardinality_cv": round(cv_c, 4),
        "cardinality_drift_range": int(max(cards) - min(cards)) if cards else 0,
        "cardinality_saturation_ratio": round(saturation_ratio, 4),
        "initial_skew": round(skews[0], 4) if skews else 0.0,
        "final_skew": round(skews[-1], 4) if skews else 0.0,
        "min_skew": round(min_s, 4),
        "max_skew": round(max_s, 4),
        "mean_skew": round(mean_s, 4),
        "std_skew": round(std_s, 4),
        "skew_drift_range": round(skew_drift_range, 4),
        "max_step_skew_drift": round(max(step_skews), 4) if step_skews else 0.0,
        "mean_step_skew_drift": round(float(np.mean(step_skews)), 4) if step_skews else 0.0,
        "skew_regime_shifts": regime_shifts,
        "regimes_visited": regimes_visited,
        "mean_new_values_ratio": round(float(np.mean([r["new_values_ratio"] for r in window_records])), 4) if window_records else 0.0,
        "mean_jaccard_overlap": round(float(np.mean([r["jaccard_overlap"] for r in window_records[1:]])), 4) if len(window_records) > 1 else 1.0,
        "mean_js_divergence": round(float(np.mean([r["js_divergence"] for r in window_records[1:]])), 4) if len(window_records) > 1 else 0.0,
    }

    return pd.DataFrame(window_records), col_summary


def load_dataset_columns(file_path: str, columns_filter: list = None, max_rows: int = None):
    """
    Generator yielding (column_name, pd.Series, optional_time_series) efficiently.
    For Parquet: reads one column at a time to preserve RAM.
    For CSV/JSON: reads dataframe with pyarrow/pandas.
    """
    base_name = os.path.basename(file_path).split('.')[0]

    if file_path.endswith(".parquet"):
        pf = pq.ParquetFile(file_path)
        all_cols = pf.schema.names
        target_cols = [c for c in all_cols if (not columns_filter or c in columns_filter)]
        total_rows = pf.metadata.num_rows
        if max_rows:
            total_rows = min(total_rows, max_rows)

        # Detect potential timestamp column
        time_series = None
        for cand in ["tpep_pickup_datetime", "datetime", "created_at", "timestamp", "time"]:
            if cand in all_cols:
                try:
                    time_series = pf.read([cand]).column(0).to_pandas()
                    if max_rows:
                        time_series = time_series.iloc[:max_rows]
                    break
                except Exception:
                    pass

        for col in target_cols:
            try:
                col_series = pf.read([col]).column(0).to_pandas()
                if max_rows:
                    col_series = col_series.iloc[:max_rows]
                yield col, col_series, time_series
            except Exception as e:
                print(f"Error loading column {col} from {file_path}: {e}")

    elif file_path.endswith((".json", ".json.gz")):
        try:
            df = pd.read_json(file_path, lines=True, nrows=max_rows,
                              compression="gzip" if file_path.endswith(".gz") else None)
        except Exception:
            opener = gzip.open if file_path.endswith(".gz") else open
            with opener(file_path, "rt") as f:
                records = []
                for idx, line in enumerate(f):
                    if max_rows and idx >= max_rows:
                        break
                    records.append(json.loads(line))
            df = pd.json_normalize(records)

        time_series = None
        for cand in ["created_at", "timestamp", "datetime"]:
            if cand in df.columns:
                time_series = df[cand]
                break

        target_cols = [c for c in df.columns if (not columns_filter or c in columns_filter)]
        for col in target_cols:
            yield col, df[col], time_series

    elif file_path.endswith((".csv", ".csv.gz", ".tsv", ".csv.bz2")):
        try:
            dataset = ds.dataset(file_path, format="csv")
            if max_rows:
                scanner = dataset.scanner(columns=columns_filter, batch_size=max_rows)
                batches = []
                for b in scanner.to_batches():
                    batches.append(b)
                    if sum(len(x) for x in batches) >= max_rows:
                        break
                table = pyarrow.Table.from_batches(batches)[:max_rows]
                df = table.to_pandas()
            else:
                df = dataset.to_table(columns=columns_filter).to_pandas()
        except Exception:
            sep = '\t' if file_path.endswith('.tsv') else ','
            df = pd.read_csv(file_path, nrows=max_rows, sep=sep)

        time_series = None
        for cand in ["datetime", "timestamp", "date", "time"]:
            if cand in df.columns:
                time_series = df[cand]
                break

        target_cols = [c for c in df.columns if (not columns_filter or c in columns_filter)]
        for col in target_cols:
            yield col, df[col], time_series
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def profile_dataset_drift(
    file_path: str,
    output_dir: str = None,
    window_size: int = None,
    stride: int = None,
    num_windows: int = None,
    columns_filter: list = None,
    max_rows: int = None,
    time_col: str = None
) -> tuple:
    """
    Runs window drift analysis across columns of a dataset file.
    Saves:
      - <dataset_name>_drift_windows.csv
      - <dataset_name>_drift_summary.csv
    """
    if output_dir is None:
        output_dir = DEFAULT_RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(file_path).split('.')[0]
    print(f"\n================================================================================")
    print(f"[+] Profiling Drift for Dataset: {base_name}")
    print(f"    File: {file_path}")

    all_windows_dfs = []
    all_summary_records = []

    # Read first column or inspect length to determine total_rows
    col_gen = load_dataset_columns(file_path, columns_filter=columns_filter, max_rows=max_rows)
    try:
        first_col, first_series, detected_time_series = next(col_gen)
    except StopIteration:
        print(f"[-] No columns found or loaded for {file_path}")
        return None, None

    total_rows = len(first_series)

    # Determine window size and stride
    if num_windows is not None and num_windows > 0:
        w_size = max(10, total_rows // num_windows)
        w_stride = w_size
    elif window_size is not None and window_size > 0:
        w_size = window_size
        w_stride = stride or window_size
    else:
        # Default adaptive window sizing: aim for ~15 to 30 windows
        w_size = min(100000, max(200, total_rows // 20))
        w_stride = stride or w_size

    window_slices = build_window_slices(total_rows, w_size, w_stride)
    print(f"    Total Records: {total_rows:,} | Windows: {len(window_slices)} (Size: {w_size:,}, Stride: {w_stride:,})")

    # Time series override if user provided --time-col
    time_series = detected_time_series
    if time_col and time_col != first_col:
        try:
            for c_name, c_ser, _ in load_dataset_columns(file_path, columns_filter=[time_col], max_rows=max_rows):
                time_series = c_ser
                break
        except Exception:
            pass

    # Profile first column
    print(f"  -> Profiling column: {first_col} ({first_series.dtype})...")
    w_df, s_rec = profile_column_drift(first_series, window_slices, base_name, first_col, time_series)
    all_windows_dfs.append(w_df)
    all_summary_records.append(s_rec)

    # Profile remaining columns
    for col, col_series, _ in col_gen:
        print(f"  -> Profiling column: {col} ({col_series.dtype})...")
        w_df, s_rec = profile_column_drift(col_series, window_slices, base_name, col, time_series)
        all_windows_dfs.append(w_df)
        all_summary_records.append(s_rec)

    combined_windows_df = pd.concat(all_windows_dfs, ignore_index=True)
    summary_df = pd.DataFrame(all_summary_records)

    # Save to CSV
    windows_path = os.path.join(output_dir, f"{base_name}_drift_windows.csv")
    summary_path = os.path.join(output_dir, f"{base_name}_drift_summary.csv")

    combined_windows_df.to_csv(windows_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[✓] Window traces saved:  {windows_path}")
    print(f"[✓] Drift summary saved:  {summary_path}")

    # Display clean preview table
    disp_cols = ["column", "total_cardinality", "mean_cardinality", "cardinality_drift_range",
                 "mean_skew", "skew_drift_range", "max_step_skew_drift", "regimes_visited"]
    disp_cols = [c for c in disp_cols if c in summary_df.columns]
    print("\nDrift Summary Preview:")
    print(summary_df[disp_cols].to_string(index=False))

    return combined_windows_df, summary_df


def profile_targets_suite(data_dir: str, output_dir: str, max_rows: int = None):
    """Batch profiles the benchmark & real-world suite in data_dir."""
    print(f"[+] Running Drift Analysis Suite across Target Datasets in: {data_dir}")
    os.makedirs(output_dir, exist_ok=True)
    all_summaries = []

    for rel_path, config in KNOWN_TARGETS:
        full_path = os.path.join(data_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"[-] Skipping missing dataset: {full_path}")
            continue

        w_size = config.get("window_size", 100000)
        t_col = config.get("time_col", None)

        try:
            _, sum_df = profile_dataset_drift(
                file_path=full_path,
                output_dir=output_dir,
                window_size=w_size,
                max_rows=max_rows,
                time_col=t_col
            )
            if sum_df is not None:
                all_summaries.append(sum_df)
        except Exception as e:
            print(f"[!] Error profiling {rel_path}: {e}")

    if all_summaries:
        combined_all = pd.concat(all_summaries, ignore_index=True)
        all_sum_path = os.path.join(output_dir, "all_datasets_drift_summary.csv")
        combined_all.to_csv(all_sum_path, index=False)
        print(f"\n[✓] Combined multi-dataset summary saved to: {all_sum_path}")
        return combined_all
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile drift in Zipfian skew and cardinality over defined windows."
    )
    parser.add_argument("data_file", nargs="?", default=None,
                        help="Path to input dataset file (parquet, csv, json.gz, etc.)")
    parser.add_argument("--targets", action="store_true",
                        help="Run drift profiling across all target datasets in data_dir")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Base directory containing datasets (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("-w", "--window-size", type=int, default=None,
                        help="Window size in rows (default: adaptive, ~total_rows // 20)")
    parser.add_argument("-s", "--stride", type=int, default=None,
                        help="Stride / step size in rows (default: equal to window-size for tumbling windows)")
    parser.add_argument("-k", "--num-windows", type=int, default=None,
                        help="Number of uniform windows to partition dataset into")
    parser.add_argument("-c", "--columns", default=None,
                        help="Comma-separated list of column names to analyze")
    parser.add_argument("-o", "--out-dir", default=DEFAULT_RESULTS_DIR,
                        help=f"Directory to save drift CSV traces (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Maximum rows to load/analyze (useful for fast testing on massive datasets)")
    parser.add_argument("--time-col", default=None,
                        help="Optional datetime/timestamp column name to attach time bounds")

    args = parser.parse_args()

    filter_cols = [c.strip() for c in args.columns.split(",")] if args.columns else None

    if args.targets:
        profile_targets_suite(args.data_dir, args.out_dir, max_rows=args.max_rows)
    elif args.data_file:
        profile_dataset_drift(
            file_path=args.data_file,
            output_dir=args.out_dir,
            window_size=args.window_size,
            stride=args.stride,
            num_windows=args.num_windows,
            columns_filter=filter_cols,
            max_rows=args.max_rows,
            time_col=args.time_col
        )
    else:
        parser.print_help()
        sys.exit(1)

