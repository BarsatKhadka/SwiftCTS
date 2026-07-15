#!/usr/bin/env bash
# =============================================================================
# One-time HPC setup — run this once after cloning on the cluster.
# Usage: bash ~/SwiftCTS/CTS-Bench/hpc/setup.sh
#
# Host only needs: python3, singularity, pip
# openlane runs INSIDE the Singularity container — no host install needed.
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "=== CTS-Bench HPC Setup ==="
echo "    CTS_BENCH_ROOT : ${CTS_BENCH_ROOT}"
echo "    OPENLANE_SIF   : ${OPENLANE_SIF}"
echo "    SKY130_PDK     : ${SKY130_PDK}"
echo "    DATASET_ROOT   : ${DATASET_ROOT}"
echo ""

# ── 1. Singularity image ──────────────────────────────────────────────────────
echo "[1/3] Pulling OpenLane 2.3.10 Singularity image (~5GB)..."
mkdir -p "$(dirname "${OPENLANE_SIF}")"
if [ ! -f "${OPENLANE_SIF}" ]; then
    singularity pull "${OPENLANE_SIF}" docker://ghcr.io/efabless/openlane2:2.3.10
    echo "      Saved to ${OPENLANE_SIF}"
else
    echo "      Already exists, skipping."
fi

# ── 2. Sky130A PDK via volare ─────────────────────────────────────────────────
# volare is a lightweight pip package — works on any Python version (no C build)
echo "[2/3] Installing Sky130A PDK (hash ${PDK_HASH})..."
mkdir -p "${PDK_ROOT}"
pip install volare -q
volare enable --pdk sky130 --pdk-root "${PDK_ROOT}" "${PDK_HASH}"
echo "      PDK at ${SKY130_PDK}"

# ── 3. Dataset output dirs ────────────────────────────────────────────────────
echo "[3/3] Creating dataset directories..."
mkdir -p "${DATASET_ROOT}/placement_files"
mkdir -p "${SHARDS_DIR}"
mkdir -p "${CTS_BENCH_ROOT}/hpc/logs"

echo ""
echo "=== Setup complete ==="
echo "Submit array job:  sbatch ${CTS_BENCH_ROOT}/hpc/slurm/run_array.sbatch"
echo "After jobs finish: python3 ${CTS_BENCH_ROOT}/hpc/slurm/merge_csvs.py"
