# Columnar Storage Dataset Profiling & Distribution Analysis

This repository contains tools to reproduce and analyze the empirical column property distributions and Zipfian skew metrics presented in:

> **An Empirical Evaluation of Columnar Storage Formats**  
> Xinyu Zeng, Yulong Hui, Jiahong Shen, Andrew Pavlo, Wes McKinney, Huanchen Zhang  
> *Proceedings of the VLDB Endowment (PVLDB)*, Vol. 17, No. 2, 2023.  
> [Paper Link](https://www.vldb.org/pvldb/vol17/p148-zeng.pdf) | [Upstream Repository](https://github.com/embryo-labs/EvaluationOfColumnarFormats)

---

## 1. Environment & Prerequisites

### System Requirements
* Python 3.9+
* Operating System: Linux (tested on Ubuntu 20.04/22.04 LTS)
* Persistent storage: ~2 GB for pre-generated synthetic benchmarks; ~20 GB compressed / ~100 GB uncompressed for full raw real-world datasets.

### Installation
Install the required analytical and visualization dependencies:

```bash
pip install pyarrow duckdb pandas numpy scipy matplotlib seaborn
```

---

## 2. Dataset Acquisition

We provide an automated script [`download_data.sh`](./download_data.sh) to retrieve the datasets directly to target storage (default `/mnt/hdd0/database_data`):

```bash
# Option A: Download all 7 target benchmark & real-world datasets (~22 GB compressed)
./download_data.sh /mnt/hdd0/database_data --targets

# Option B: Download the 6 pre-generated 1M-row x 20-col synthetic benchmarks (~1.1 GB)
./download_data.sh /mnt/hdd0/database_data --benchmarks

# Option C: Download the full real-world raw dataset archive (~19.4 GB compressed)
./download_data.sh /mnt/hdd0/database_data --raw

# Option D: Download benchmarks, raw archive, 7 targets, and clone the OpenFormat generator submodule
./download_data.sh /mnt/hdd0/database_data --all
```

### Profiled Benchmark & Real-World Datasets
| Dataset | Domain | Total Records ($N$) | Columns | Format | Key Column Properties |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ClickBench Hits** | Web Analytics / AdTech | 99,997,497 | 105 | Parquet | Heavy Zipfian skew ($s_{\text{median}}=2.24$), high sortedness (0.93), low NDV ratio ($f_{cr} < 0.001$) |
| **NYC Yellow Taxi** | Spatiotemporal / Trips | 3,066,766 | 19 | Parquet | Gentle Zipf ($s_{\text{median}}=1.50$), temporal ordering, clustered pick-up/drop-off locations |
| **FineWeb-Edu** | Unstructured Text / NLP | 726,000 | 10 | Parquet | High unique cardinality ($f_{cr}=0.24$), near-uniform text distributions ($s=0.26$) |
| **GitHub Event Logs** | Deeply Nested Event Logs | 161,519 | 8 | JSON Gzip | High distinct ratios ($f_{cr}=0.20$), power-law active repos/actors ($s=0.80$) |
| **OpenAQ** | IoT Sensor Time-Series | 3,046 | 9 | CSV Gzip | Strictly chronological ($f_{\text{sort}}=0.96$), uniform measurements ($s=0.04$) |
| **LAION-2B** | Multimodal / Embeddings | 5,329,780 | 10 | Parquet | Ultra-high NDV ratio ($f_{cr}=0.67$), uniform/shuffled layout ($f_{\text{sort}}=0.50$) |
| **FormatEvalData (ML / Menu)** | Relational / Tabular ML | 3.3M+ | 19 | CSV / bz2 | Long-tail relational properties ($s=0.92-1.19$), low NDV ratio ($f_{cr} \approx 0.001$) |
| **OpenFormat Benchmarks** | Synthetic Microbenchmarks | 1,000,000 × 6 | 120 | CSV | Synthetic control suites (`core`, `bi`, `classic`, `geo`, `log`, `ml`) matching Table 4 |
| **Zeng et al. Survey** | Comprehensive Survey | 206 tables | 2,845 | Various | Baseline survey from Zeng et al. (PVLDB 2023) across 2,845 columns |

---

## 3. Profiling Tool (`profile_dataset.py`)

[`profile_dataset.py`](./profile_dataset.py) profiles any tabular data file (Parquet, CSV, TSV) and calculates the core properties defined in Section 4 of Zeng et al.

### Usage
```bash
python3 profile_dataset.py <path_to_data_file>
```

Example:
```bash
python3 profile_dataset.py /mnt/hdd0/database_data/openformat_benchmarks/core_r1000000_c20.csv
```

### Metrics Computed
For every column $C = [a_1, a_2, \dots, a_N]$ with $N$ total records and non-null count $N_{\text{non\_null}}$:

1. **Total Records ($N$)**: Total row count of the column.
2. **Null Ratio ($f_{nr}$)**:
   $$f_{nr} = \frac{|\{i \mid a_i \text{ is null}\}|}{N}$$
3. **Cardinality ($C$) & NDV Ratio ($f_{cr}$)**:
   * Cardinality ($C$): Count of distinct non-null values.
   * NDV Ratio ($f_{cr}$): $f_{cr} = \frac{C}{N_{\text{non\_null}}}$.
4. **Value Bit-Width**:
   * Integers: $\lceil \log_2(\max(v) - \min(v) + 1) \rceil$ (minimum bits required for bitpacking).
   * Floating-point: 32 or 64 bits.
   * Categorical / Strings: $\lceil \log_2(C) \rceil$.
5. **Sortedness Score**:
   Local ascending or descending monotonicity evaluated in non-overlapping 512-record windows:
   $$asc = |\{i \mid a_i < a_{i+1}\}|, \quad desc = |\{i \mid a_i > a_{i+1}\}|, \quad eq = |\{i \mid a_i = a_{i+1}\}|$$
   $$f_{\text{sortedness}} = \frac{\max(asc, desc) + eq - \lfloor N_w / 2 \rfloor}{\lceil N_w / 2 \rceil - 1}$$
   Ranges from $0.5$ (random uniformly shuffled) to $1.0$ (fully monotonic).
6. **Zipfian Skew Parameter ($s$)**:
   Empirical value frequencies ordered by rank $r \in [1, C]$ are fitted against Zipf's Law:
   $$P(r; s, C) = \frac{r^{-s}}{\sum_{k=1}^C k^{-s}}$$
   * Fitting is performed using non-linear least squares optimization (`scipy.optimize.curve_fit`) bounded in $s \in [0.0, 10.0]$.
   * Falls back to log-log linear regression ($\ln(\text{Freq}_r) = \alpha - s \cdot \ln(r)$).
   * **Interpretation**:
     * $s \approx 0.0$: Uniform distribution (e.g., primary keys, synthetic IDs).
     * $0.01 < s \le 2.0$: Gentle Zipf / long-tail distribution (typical web logs, textual attributes).
     * $s > 2.0$: Hotspot / extreme skew (heavy-hitter categorical variables).

---

## 4. Visualizations & Empirical Distributions

Run the visualization pipeline to regenerate all distribution plots:

```bash
python3 plot_distributions.py
```

All plots are saved to the [`figures/`](./figures/) directory.

### Detailed Histograms
* [`figures/detailed_histograms.png`](./figures/detailed_histograms.png): 4-panel overview comparing real-world vs benchmark distributions across:
  * **NDV Ratio**: Over 80% of integer columns and 60% of string columns have $f_{cr} < 0.01$.
  * **Distinct Cardinality ($C$)**: Spans from low-cardinality flags ($C < 10$) to high-cardinality keys ($C > 10^7$).
  * **Zipfian Skew Exponent ($s$)**: Shows dominance of Gentle Zipf ($0.01 < s \le 2.0$) and heavy-hitter hotspots ($s > 2.0$).
  * **Total Records ($N$)**: Captures datasets scaling up to $10^8$ rows.
* Individual detailed figures:
  * [`figures/histogram_ndv_ratio.png`](./figures/histogram_ndv_ratio.png)
  * [`figures/histogram_cardinality.png`](./figures/histogram_cardinality.png)
  * [`figures/histogram_skew.png`](./figures/histogram_skew.png)
  * [`figures/histogram_total_records.png`](./figures/histogram_total_records.png)

### 2D Comparative Scatter Plots
Three separate 2D pairwise spaces comparing Total Records, Cardinality, and Zipfian Skew:
1. **Total Records vs Cardinality** ([`figures/scatter_records_vs_cardinality.png`](./figures/scatter_records_vs_cardinality.png)):
   * Plotted in log-log scale.
   * Compares empirical points against the theoretical upper bound identity line $C = N$.
2. **Total Records vs Skew** ([`figures/scatter_records_vs_skew.png`](./figures/scatter_records_vs_skew.png)):
   * Evaluates whether data scale influences value skewness, annotated with the Uniform ($s \le 0.01$) and Gentle Zipf ($s = 2.0$) boundaries.
3. **Cardinality vs Skew** ([`figures/scatter_cardinality_vs_skew.png`](./figures/scatter_cardinality_vs_skew.png)):
   * Highlights how extreme skew ($s > 2$) is concentrated in categorical columns, while high-cardinality columns settle toward moderate Zipfian long tails ($s \approx 0.8 - 1.5$).
