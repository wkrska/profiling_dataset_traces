import sys
import os
import json
import gzip
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.optimize import curve_fit

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

def profile_column(col_series: pd.Series) -> dict:
    N = len(col_series)
    non_nulls = col_series.dropna()
    N_non_null = len(non_nulls)
    
    if N_non_null == 0:
        return {
            "null_ratio": 1.0, 
            "ndv_ratio": 0.0, 
            "cardinality": 0, 
            "bit_width": 0, 
            "sortedness": 1.0, 
            "zipf_s_skew": 0.0
        }

    # If elements are complex objects (e.g. dicts, lists in JSON), stringify for hashing
    if len(non_nulls) > 0 and isinstance(non_nulls.iloc[0], (dict, list)):
        non_nulls = non_nulls.astype(str)

    # 1. Null Ratio (f_nr)
    f_nr = (N - N_non_null) / N

    # 2. Cardinality (C) & NDV Ratio (f_cr)
    cardinality = non_nulls.nunique()
    f_cr = cardinality / N_non_null

    # 3. Value Bit-Width
    if pd.api.types.is_numeric_dtype(non_nulls):
        if pd.api.types.is_integer_dtype(non_nulls):
            val_range = int(non_nulls.max()) - int(non_nulls.min()) + 1
            bit_width = int(np.ceil(np.log2(max(val_range, 1))))
        else:
            bit_width = 32 if non_nulls.dtype == 'float32' else 64
    else:
        bit_width = int(np.ceil(np.log2(max(cardinality, 1))))

    # 4. Sortedness Score
    arr = non_nulls.to_numpy()
    if N_non_null > 1:
        try:
            diffs = np.diff(arr)
            asc_pairs = np.sum(diffs >= 0)
            desc_pairs = np.sum(diffs <= 0)
        except Exception:
            asc_pairs = np.sum(arr[1:] >= arr[:-1])
            desc_pairs = np.sum(arr[1:] <= arr[:-1])
        sortedness = max(asc_pairs, desc_pairs) / (N_non_null - 1)
    else:
        sortedness = 1.0

    # 5. Zipfian Skew Parameter s
    freq_counts = non_nulls.value_counts().values
    zipf_s = compute_zipf_s(freq_counts, cardinality)

    return {
        "null_ratio": round(f_nr, 4),
        "ndv_ratio": round(f_cr, 4),
        "cardinality": int(cardinality),
        "bit_width": bit_width,
        "sortedness": round(sortedness, 4),
        "zipf_s_skew": round(zipf_s, 4)
    }

def profile_dataset(file_path: str):
    # Support parquet, compressed json, csv, and tsv
    if file_path.endswith((".json", ".json.gz")):
        try:
            df = pd.read_json(file_path, lines=True, compression="gzip" if file_path.endswith(".gz") else None)
        except Exception:
            with (gzip.open(file_path, "rt") if file_path.endswith(".gz") else open(file_path, "r")) as f:
                records = [json.loads(line) for idx, line in enumerate(f)]
            df = pd.json_normalize(records)
        results = {}
        for col in df.columns:
            results[col] = profile_column(df[col])
    elif file_path.endswith((".csv", ".csv.gz", ".tsv", ".csv.bz2")):
        try:
            dataset = ds.dataset(file_path, format="csv")
            df = dataset.to_table().to_pandas()
        except Exception:
            df = pd.read_csv(file_path)
        results = {}
        for col in df.columns:
            results[col] = profile_column(df[col])
    else:
        # Parquet files: read column-by-column to preserve memory on large datasets (e.g. ClickBench 100M rows)
        pf = pq.ParquetFile(file_path)
        results = {}
        for col in pf.schema.names:
            try:
                col_series = pf.read([col]).column(0).to_pandas()
                results[col] = profile_column(col_series)
            except Exception as e:
                print(f"Error profiling {col}: {e}")

    out_df = pd.DataFrame(results).T
    
    # Save results directly to script directory
    script_dir = "/home/wkrska/Aggregation/profiling_dataset_traces"
    os.makedirs(script_dir, exist_ok=True)
    base_name = os.path.basename(file_path).split('.')[0]
    out_path = os.path.join(script_dir, f"{base_name}_profile.csv")
    out_df.to_csv(out_path)
    print(f"Results saved to: {out_path}")
    return out_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile_dataset.py <path_to_data_file>")
        sys.exit(1)
        
    data_file = sys.argv[1]
    profile_df = profile_dataset(data_file)
    print(profile_df.to_string())
