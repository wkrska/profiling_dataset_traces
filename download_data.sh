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

MODE="${2:---benchmarks}"

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
    --benchmarks)
        download_benchmarks
        ;;
    --raw)
        download_raw
        ;;
    --all)
        download_benchmarks
        download_raw
        clone_repo
        ;;
    *)
        echo "Usage: $0 [TARGET_DIR] [--benchmarks | --raw | --all]"
        exit 1
        ;;
esac

echo ""
echo "All requested data operations completed successfully."
