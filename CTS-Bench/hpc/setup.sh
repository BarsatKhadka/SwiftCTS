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
echo "[1/3] Pulling OpenLane 2.3.10 Singularity image (~5GB, may take 30-60 min)..."
mkdir -p "$(dirname "${OPENLANE_SIF}")"

# Use $HOME/tmp as TMPDIR — avoids /tmp quota issues on shared clusters
export APPTAINER_TMPDIR="${HOME}/tmp"
export SINGULARITY_TMPDIR="${HOME}/tmp"
export APPTAINER_CACHEDIR="${HOME}/apptainer_cache"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

# Disable HTTP/2 — apptainer is Go-based; GODEBUG=http2client=0 forces HTTP/1.1
# which avoids the "stream error: PROTOCOL_ERROR" from ghcr.io on HPC networks
export GODEBUG=http2client=0

# Use apptainer if available (it's Singularity's successor on this cluster)
PULL_CMD="singularity"
command -v apptainer &>/dev/null && PULL_CMD="apptainer"
echo "      Using: ${PULL_CMD} (HTTP/2 disabled via GODEBUG)"

if [ ! -f "${OPENLANE_SIF}" ]; then
    # Retry up to 5 times — large image, network can be flaky
    for attempt in 1 2 3 4 5; do
        echo "      Attempt ${attempt}/5..."
        if ${PULL_CMD} pull --force "${OPENLANE_SIF}" docker://ghcr.io/efabless/openlane2:2.3.10; then
            echo "      Saved to ${OPENLANE_SIF}"
            break
        fi
        [ "${attempt}" -lt 5 ] && echo "      Failed, retrying in 15s..." && sleep 15
    done
    [ ! -f "${OPENLANE_SIF}" ] && echo "ERROR: Failed to pull SIF after 5 attempts." && exit 1
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
