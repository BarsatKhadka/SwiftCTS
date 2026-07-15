#!/usr/bin/env bash
# =============================================================================
# Local smoke test — runs 1 pipeline iteration for 1 design (aes) with Docker.
# Validates that all paths, scripts, and Docker calls are correct before HPC.
#
# Usage:
#   cd ~/SwiftCTS/CTS-Bench
#   bash hpc/test_local.sh
#
# Expected output:
#   - runs/aes_run_YYYYMMDD_HHMMSS/  placement directory created
#   - aes.saif                        SAIF generated
#   - CTS-experiments/CTS-1/         at least 1 CTS run completed
#   - test_shard.csv                  1 row written
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTS_BENCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CTS_BENCH_ROOT}"

echo "=== CTS-Bench local smoke test ==="
echo "    CTS_BENCH_ROOT: ${CTS_BENCH_ROOT}"
echo ""

# Check prereqs
echo "[CHECK] Docker..."
docker ps > /dev/null 2>&1 || { echo "ERROR: Docker not running. Start Docker first."; exit 1; }
echo "        OK"

echo "[CHECK] Python3..."
python3 --version || { echo "ERROR: python3 not found."; exit 1; }
echo "        OK"

echo "[CHECK] iverilog (via nix or system)..."
if ! command -v iverilog &>/dev/null; then
    # Try nix
    if command -v nix &>/dev/null; then
        NIX_SHELL="nix shell nixpkgs/nixos-24.05#verilog --command"
        echo "        Will use nix shell for iverilog"
    else
        echo "ERROR: iverilog not found. Install iverilog or Nix."
        exit 1
    fi
else
    NIX_SHELL=""
    echo "        OK ($(iverilog -V 2>&1 | head -1))"
fi

echo ""
echo "[1/4] Running placement for aes (1 iteration)..."
python3 scripts/1-gen-placement.py aes 10.0 clk
if [ ! -f latest_run.txt ]; then
    echo "ERROR: latest_run.txt not written — placement failed."
    exit 1
fi
TAG=$(cat latest_run.txt)
echo "      Placement tag: ${TAG}"

echo ""
echo "[2/4] Running SAIF simulation for ${TAG}..."
# Use nix shell to get iverilog on macOS (same as main-def-saif.py)
if command -v nix &>/dev/null; then
    nix shell nixpkgs/nixos-24.05#verilog --command python3 scripts/2-gen-saif.py "${TAG}"
elif command -v iverilog &>/dev/null; then
    python3 scripts/2-gen-saif.py "${TAG}"
else
    echo "ERROR: Neither nix nor iverilog found. Install one to run SAIF simulation."
    exit 1
fi
SAIF_FILE="runs/${TAG}/aes.saif"
if [ ! -f "${SAIF_FILE}" ]; then
    echo "ERROR: SAIF file not found at ${SAIF_FILE}"
    exit 1
fi
echo "      SAIF OK ($(wc -l < "${SAIF_FILE}") lines)"

echo ""
echo "[3/4] Running 1 CTS experiment for ${TAG}..."
MY_PDK_ROOT="${HOME}/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af"
docker run --rm \
    -v "${CTS_BENCH_ROOT}:${CTS_BENCH_ROOT}" \
    -v "${HOME}/.volare:${HOME}/.volare" \
    -w "${CTS_BENCH_ROOT}" \
    -e "PDK_ROOT=${MY_PDK_ROOT}" \
    ghcr.io/efabless/openlane2:2.3.10 \
    python3 scripts/5-run-cts.py "${TAG}" aes 10.0 clk
echo "      CTS done"

echo ""
echo "[4/4] Parsing reports..."
python3 scripts/6-parse-cts-reports.py "${TAG}"
echo "      Parse done"

echo ""
echo "=== Smoke test PASSED ==="
echo "    Run directory: runs/${TAG}/"
echo "    SAIF: runs/${TAG}/aes.saif"
echo ""
echo "Pipeline works from this location. Ready for HPC."
