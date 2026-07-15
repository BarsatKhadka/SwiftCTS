#!/usr/bin/env bash
# =============================================================================
# One-time HPC setup — run this once after cloning CTS-Bench on the cluster.
# Usage: bash ~/CTS-Bench/hpc/setup.sh
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "=== CTS-Bench HPC Setup ==="

# ── 1. Python venv + openlane package ─────────────────────────────────────────
echo "[1/4] Creating Python venv at ${VENV_DIR}..."
module load python 2>/dev/null || true   # adjust module name for your cluster
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q
pip install openlane==2.3.10 -q
echo "      openlane installed: $(python3 -c 'import openlane; print(openlane.__version__)')"

# ── 2. Singularity image ──────────────────────────────────────────────────────
echo "[2/4] Pulling OpenLane 2.3.10 Singularity image..."
mkdir -p "$(dirname "${OPENLANE_SIF}")"
if [ ! -f "${OPENLANE_SIF}" ]; then
    singularity pull "${OPENLANE_SIF}" docker://ghcr.io/efabless/openlane2:2.3.10
    echo "      Saved to ${OPENLANE_SIF}"
else
    echo "      Already exists, skipping."
fi

# ── 3. Sky130A PDK via volare ─────────────────────────────────────────────────
echo "[3/4] Installing Sky130A PDK (hash ${PDK_HASH})..."
mkdir -p "${PDK_ROOT}"
pip install volare -q
volare enable --pdk sky130 --pdk-root "${PDK_ROOT}" "${PDK_HASH}"
echo "      PDK at ${SKY130_PDK}"

# ── 4. Dataset output dirs ────────────────────────────────────────────────────
echo "[4/4] Creating dataset directories..."
mkdir -p "${DATASET_ROOT}/placement_files"
mkdir -p "${SHARDS_DIR}"
mkdir -p "${CTS_BENCH_ROOT}/hpc/logs"

echo ""
echo "=== Setup complete ==="
echo "Source env.sh in your jobs:  source ${CTS_BENCH_ROOT}/hpc/env.sh"
echo "Submit array job:             sbatch ${CTS_BENCH_ROOT}/hpc/slurm/run_array.sbatch"
