#!/usr/bin/env bash
set -e

# ==============================================================================
# Script to download datasets for Zeng et al. (PVLDB 2023) evaluation & profiling
# ==============================================================================

TARGET_DIR="${1:-/mnt/hdd0/database_data}"
BENCHMARK_DIR="${TARGET_DIR}/openformat_benchmarks"
RAW_DIR="${TARGET_DIR}/raw_datasets"
REPO_DIR="${TARGET_DIR}/EvaluationOfColumnarFormats"

echo "================================================================================"
echo "Dataset Downloader for Columnar Storage Profiling"
echo "Target Base Directory: ${TARGET_DIR}"
echo "================================================================================"

mkdir -p "${BENCHMARK_DIR}"
mkdir -p "${RAW_DIR}"

MODE="${2:---all-targets}"

download_7_targets() {
    echo ""
    echo "[+] Downloading 7 Target Benchmark & Real-World Datasets directly into ${TARGET_DIR}..."

    # 1. ClickBench Hits Sample (~14.8 GB Parquet - Web Analytics)
    echo "  [1/7] ClickBench Hits Sample..."
    if [ ! -f "${TARGET_DIR}/clickbench_hits.parquet" ]; then
        wget -c https://datasets.clickhouse.com/hits_compatible/hits.parquet -O "${TARGET_DIR}/clickbench_hits.parquet"
    else
        echo "        Already exists at ${TARGET_DIR}/clickbench_hits.parquet"
    fi

    # 2. NYC Yellow Taxi (Parquet - Spatiotemporal / Time-Series)
    echo "  [2/7] NYC Yellow Taxi Trip Data..."
    if [ ! -f "${TARGET_DIR}/nyc_taxi_2023_01.parquet" ]; then
        wget -c https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet -O "${TARGET_DIR}/nyc_taxi_2023_01.parquet"
    else
        echo "        Already exists at ${TARGET_DIR}/nyc_taxi_2023_01.parquet"
    fi

    # 3. FineWeb-Edu Sample (~2.1 GB Parquet - Unstructured Text / Tokens)
    echo "  [3/7] FineWeb-Edu Sample..."
    if [ ! -f "${TARGET_DIR}/fineweb_edu_sample.parquet" ]; then
        wget -c https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet -O "${TARGET_DIR}/fineweb_edu_sample.parquet"
    else
        echo "        Already exists at ${TARGET_DIR}/fineweb_edu_sample.parquet"
    fi

    # 4. GitHub Event Logs (~180 MB Gzip JSON - Deeply Nested Logs)
    echo "  [4/7] GitHub Event Logs..."
    if [ ! -f "${TARGET_DIR}/gh_events.json.gz" ]; then
        wget -c https://data.gharchive.org/2023-01-01-15.json.gz -O "${TARGET_DIR}/gh_events.json.gz"
    else
        echo "        Already exists at ${TARGET_DIR}/gh_events.json.gz"
    fi

    # 5. OpenAQ Air Quality Data (Gzip CSV - IoT Sensor Time-Series)
    echo "  [5/7] OpenAQ Air Quality Data..."
    if [ ! -f "${TARGET_DIR}/openaq_2023.csv.gz" ]; then
        wget -c "https://openaq-data-archive.s3.amazonaws.com/records/csv.gz/locationid=2178/year=2023/month=01/data.csv.gz" -O "${TARGET_DIR}/openaq_2023.csv.gz"
    else
        echo "        Already exists at ${TARGET_DIR}/openaq_2023.csv.gz"
    fi

    # 6. LAION-2B Embeddings Sample (Parquet - High-Dimensional Vectors)
    echo "  [6/7] LAION Sample..."
    if [ ! -f "${TARGET_DIR}/laion_sample.parquet" ]; then
        wget -c https://huggingface.co/datasets/jasonhuang23/laion2b_en_sd2.1base/resolve/main/part_1.parquet -O "${TARGET_DIR}/laion_sample.parquet"
    else
        echo "        Already exists at ${TARGET_DIR}/laion_sample.parquet"
    fi

    # 7. Paper Raw Artifacts Package (~19.4 GB zstd - Original Evaluation Suites)
    echo "  [7/7] Paper Raw Artifacts Package (FormatEvalData)..."
    ARCHIVE_PATH="${TARGET_DIR}/FormatEvalData.tar.zst"
    if [ ! -f "${ARCHIVE_PATH}" ]; then
        wget -c https://cfb-data.xinyuzeng.xyz/FormatEvalData.tar.zst -O "${ARCHIVE_PATH}"
    else
        echo "        Archive already exists at ${ARCHIVE_PATH}"
    fi
    if [ ! -d "${TARGET_DIR}/FormatEvalData" ]; then
        echo "        Extracting to ${TARGET_DIR}/..."
        tar --zstd -xf "${ARCHIVE_PATH}" -C "${TARGET_DIR}/"
    fi
}

download_benchmarks() {
    echo ""
    echo "[1/2] Downloading Pre-generated Synthetic Benchmarks (1M rows x 20 cols)..."
    PREFIX="https://cfb-data.xinyuzeng.xyz"
    WORKLOADS=("core" "bi" "classic" "geo" "log" "ml")

    for wl in "${WORKLOADS[@]}"; do
        OUT_FILE="${BENCHMARK_DIR}/${wl}_r1000000_c20.csv"
        if [ -f "${OUT_FILE}" ]; then
            echo "  - ${wl}: already exists at ${OUT_FILE}, skipping."
        else
            URL="${PREFIX}/${wl}_r1000000_c20/gen_data/${wl}_r1000000_c20.csv"
            echo "  - Fetching ${wl} from ${URL}..."
            curl -L -C - --retry 3 "${URL}" -o "${OUT_FILE}"
        fi
    done
    echo "Benchmark datasets download complete."
}

download_raw() {
    echo ""
    echo "[2/2] Downloading Full Real-World Raw Artifacts (FormatEvalData.tar.zst, ~19.4 GB)..."
    ARCHIVE_PATH="${TARGET_DIR}/FormatEvalData.tar.zst"
    URL="https://cfb-data.xinyuzeng.xyz/FormatEvalData.tar.zst"
    
    if [ ! -f "${ARCHIVE_PATH}" ]; then
        echo "  - Fetching ${URL} -> ${ARCHIVE_PATH}..."
        wget -c "${URL}" -O "${ARCHIVE_PATH}"
    else
        echo "  - Archive already exists at ${ARCHIVE_PATH}."
    fi

    echo "  - Extracting to ${RAW_DIR}..."
    tar --zstd -xvf "${ARCHIVE_PATH}" -C "${RAW_DIR}/"
    echo "Raw real-world datasets extraction complete."
}

clone_repo() {
    echo ""
    echo "[+] Cloning EvaluationOfColumnarFormats repository..."
    if [ ! -d "${REPO_DIR}" ]; then
        git clone https://github.com/embryo-labs/EvaluationOfColumnarFormats.git "${REPO_DIR}"
        cd "${REPO_DIR}"
        git submodule update --init OpenFormat
    else
        echo "  - Repository already exists at ${REPO_DIR}."
    fi
}

case "${MODE}" in
    --targets|--all-targets)
        download_7_targets
        ;;
    --benchmarks)
        download_benchmarks
        ;;
    --raw)
        download_raw
        ;;
    --all)
        download_benchmarks
        download_raw
        download_7_targets
        clone_repo
        ;;
    *)
        echo "Usage: $0 [TARGET_DIR] [--targets | --benchmarks | --raw | --all]"
        exit 1
        ;;
esac

echo ""
echo "All requested data operations completed successfully."
