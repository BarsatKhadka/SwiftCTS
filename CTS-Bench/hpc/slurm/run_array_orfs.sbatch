#!/bin/bash
# =============================================================================
# CTS-Bench ORFS data generation — SLURM array job
#
# Generates CTS sweep rows for ASAP7 or NanGate45 using the
# OpenROAD-flow-scripts container (no OpenLane required).
#
# Each task = ITERS_PER_TASK complete pipeline runs
# (ORFS placement → CTS × 10 → parse → CSV shard)
#
# Select PDK before submitting:
#   export ACTIVE_PDK=asap7      sbatch run_array_orfs.sbatch
#   export ACTIVE_PDK=nangate45  sbatch run_array_orfs.sbatch
#
# After all tasks finish, merge shards:
#   python3 ~/SwiftCTS/CTS-Bench/hpc/slurm/merge_csvs.py
#
# Pull the ORFS container once (if not already done):
#   apptainer pull ~/singularity/orfs.sif docker://openroad/orfs:latest
# =============================================================================

#SBATCH --job-name=ctsbench-orfs
#SBATCH --partition=cpuq                 # ← adjust to your cluster partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00                  # ORFS synthesis can be slower
#SBATCH --array=0-199                    # 200 tasks × 5 iters × 10 CTS = 10,000 rows
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out

ITERS_PER_TASK=5

# ── Environment ───────────────────────────────────────────────────────────────
if   [ -f "${HOME}/SwiftCTS/CTS-Bench/hpc/env.sh" ]; then
    source "${HOME}/SwiftCTS/CTS-Bench/hpc/env.sh"
elif [ -f "${HOME}/CTS-Bench/hpc/env.sh" ]; then
    source "${HOME}/CTS-Bench/hpc/env.sh"
else
    echo "ERROR: env.sh not found." >&2
    exit 1
fi

module purge
module load python 2>/dev/null || true

echo "=== CTS-Bench ORFS task ${SLURM_ARRAY_TASK_ID} starting on $(hostname) ==="
echo "    CTS_BENCH_ROOT : ${CTS_BENCH_ROOT}"
echo "    ORFS_SIF       : ${ORFS_SIF}"
echo "    ACTIVE_PDK     : ${ACTIVE_PDK}"
echo "    SHARDS_DIR     : ${SHARDS_DIR}"
echo "    container      : $(which apptainer 2>/dev/null || which singularity 2>/dev/null || echo NOT FOUND)"

# ── Validate ──────────────────────────────────────────────────────────────────
if [ ! -f "${ORFS_SIF}" ]; then
    echo "ERROR: ORFS_SIF not found at ${ORFS_SIF}." >&2
    echo "Pull it with:" >&2
    echo "  apptainer pull ~/singularity/orfs.sif docker://openroadproject/openroad-flow-scripts:latest" >&2
    exit 1
fi

if ! command -v apptainer &>/dev/null && ! command -v singularity &>/dev/null; then
    echo "ERROR: neither apptainer nor singularity found." >&2
    exit 1
fi

if [[ "${ACTIVE_PDK}" != "asap7" && "${ACTIVE_PDK}" != "nangate45" ]]; then
    echo "ERROR: ACTIVE_PDK must be 'asap7' or 'nangate45', got '${ACTIVE_PDK}'." >&2
    exit 1
fi

mkdir -p "${SHARDS_DIR}"
cd "${CTS_BENCH_ROOT}"

python3 "${CTS_BENCH_ROOT}/hpc/main-hpc-orfs.py" \
    "${SLURM_ARRAY_TASK_ID}" \
    "${ITERS_PER_TASK}"

echo "=== Task ${SLURM_ARRAY_TASK_ID} done ==="
