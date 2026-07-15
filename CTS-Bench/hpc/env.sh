#!/usr/bin/env bash
# =============================================================================
# CTS-Bench HPC environment — source this at the top of every job script
# and in your ~/.bashrc on the cluster.
#
#   source ~/SwiftCTS/CTS-Bench/hpc/env.sh   # if cloned as SwiftCTS
#   source ~/CTS-Bench/hpc/env.sh             # if cloned standalone
#
# CTS_BENCH_ROOT is auto-detected from this file's location — no edits needed.
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
# Auto-detect CTS-Bench root from env.sh location (works wherever repo is cloned)
export CTS_BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Where placement DEF / SAIF / timing CSVs are kept (shared across all jobs)
# Default: sibling of CTS-Bench; override with: export DATASET_ROOT=/my/path
export DATASET_ROOT="${DATASET_ROOT:-${HOME}/dataset_with_def}"

# Per-job CSV shards land here; merge_csvs.py consolidates them
export SHARDS_DIR="${DATASET_ROOT}/shards"

# Final merged CSV (created by merge_csvs.py)
export LOG_FILE="${DATASET_ROOT}/experiment_log.csv"

# ── PDK ───────────────────────────────────────────────────────────────────────
# Sky130A PDK — download once with setup.sh, then point here
export PDK_ROOT="${PDK_ROOT:-${HOME}/pdk/sky130}"
export PDK_HASH="0fe599b2afb6708d281543108caf8310912f54af"
export SKY130_PDK="${PDK_ROOT}/volare/sky130/versions/${PDK_HASH}"

# ── Singularity ───────────────────────────────────────────────────────────────
# Path to the OpenLane 2.3.10 singularity image (pull once with setup.sh)
export OPENLANE_SIF="${OPENLANE_SIF:-${HOME}/singularity/openlane2-2.3.10.sif}"

# ── Python ────────────────────────────────────────────────────────────────────
# Virtual-env with openlane python package installed (pip install openlane==2.3.10)
export VENV_DIR="${CTS_BENCH_ROOT}/hpc/venv"

# ── Convenience ───────────────────────────────────────────────────────────────
export HPC_SCRIPTS="${CTS_BENCH_ROOT}/hpc/scripts"
export HPC_SLURM="${CTS_BENCH_ROOT}/hpc/slurm"

# Activate venv if it exists (safe to call multiple times)
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi
