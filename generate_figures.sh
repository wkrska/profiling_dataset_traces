#!/usr/bin/env bash
# ==============================================================================
# generate_figures.sh
#
# Generates all publication-ready figures for columnar profiling and windowed
# drift analysis into the figures/ directory.
#
# Usage:
#   ./generate_figures.sh           # Generates both distribution and drift figures
#   ./generate_figures.sh --all     # Same as default
#   ./generate_figures.sh --dist    # Generates static distribution plots only
#   ./generate_figures.sh --drift   # Generates windowed drift plots only
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

FIGURES_DIR="${SCRIPT_DIR}/figures"
DRIFT_FIGURES_DIR="${FIGURES_DIR}/drift"

mkdir -p "${FIGURES_DIR}"
mkdir -p "${DRIFT_FIGURES_DIR}"

# Determine Python interpreter (prefer project venv if present)
if [ -f "${SCRIPT_DIR}/.venv/bin/python3" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[-] Error: python3 not found." >&2
    exit 1
fi

export MPLCONFIGDIR="/tmp/matplotlib_$$"
mkdir -p "${MPLCONFIGDIR}"
trap 'rm -rf "${MPLCONFIGDIR}"' EXIT

MODE="${1:---all}"

echo "================================================================================"
echo "Generating Figures into: ${FIGURES_DIR}/"
echo "Using Python: ${PYTHON_BIN}"
echo "Mode: ${MODE}"
echo "================================================================================"

generate_distributions() {
    echo ""
    echo "[1/2] Generating Overall Column Property Distribution Figures..."
    "${PYTHON_BIN}" plot_distributions.py
    echo "[✓] Distribution figures generated in ${FIGURES_DIR}/"
}

generate_drift() {
    echo ""
    echo "[2/2] Generating Windowed Skew & Cardinality Drift Figures..."
    "${PYTHON_BIN}" plot_drift.py
    echo "[✓] Drift figures generated in ${DRIFT_FIGURES_DIR}/"
}

case "${MODE}" in
    --dist|--distributions)
        generate_distributions
        ;;
    --drift)
        generate_drift
        ;;
    --all|"")
        generate_distributions
        generate_drift
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Valid options: --all (default), --dist, --drift"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "Figure Generation Summary:"
echo "--------------------------------------------------------------------------------"
echo "Distribution plots (${FIGURES_DIR}):"
ls -lh "${FIGURES_DIR}"/*.png 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
echo ""
echo "Windowed Drift plots (${DRIFT_FIGURES_DIR}):"
ls -lh "${DRIFT_FIGURES_DIR}"/*.png 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
echo "================================================================================"

