#!/bin/bash
# =============================================================================
# CTS-Bench data generation — SLURM array job
#
# Each task = 1 SLURM job = ITERS_PER_TASK complete pipeline runs
# (placement → SAIF → CTS × 10 → parse → CSV shard)
#
# Total rows = N_TASKS × ITERS_PER_TASK × 10
# Example: 200 tasks × 5 iters × 10 CTS configs = 10,000 rows
#
# Submit (adjust path to wherever SwiftCTS is cloned on your HPC):
#   sbatch ~/SwiftCTS/CTS-Bench/hpc/slurm/run_array.sbatch
#
# After all tasks finish, merge shards:
#   python3 ~/SwiftCTS/CTS-Bench/hpc/slurm/merge_csvs.py
# =============================================================================

#SBATCH --job-name=ctsbench
#SBATCH --partition=cpuq                 # ← adjust to your cluster partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00                  # 6h per task (jpeg CTS is slowest ~1.5h)
#SBATCH --array=0-199                    # 200 tasks → 200×5×10 = 10,000 rows
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# ── One-time customisation ────────────────────────────────────────────────────
ITERS_PER_TASK=5    # pipeline iterations each task runs

# ── Environment ───────────────────────────────────────────────────────────────
# Auto-detect env.sh (works whether cloned as SwiftCTS or standalone CTS-Bench)
if   [ -f "${HOME}/SwiftCTS/CTS-Bench/hpc/env.sh" ]; then
    source "${HOME}/SwiftCTS/CTS-Bench/hpc/env.sh"
elif [ -f "${HOME}/CTS-Bench/hpc/env.sh" ]; then
    source "${HOME}/CTS-Bench/hpc/env.sh"
else
    echo "ERROR: env.sh not found. Clone SwiftCTS to ~/SwiftCTS or CTS-Bench to ~/CTS-Bench." >&2
    exit 1
fi

# Load iverilog (adjust module name to match your cluster)
module purge
module load python        2>/dev/null || true
module load iverilog      2>/dev/null || \
module load icarus-verilog 2>/dev/null || true   # name varies by cluster

echo "=== CTS-Bench task ${SLURM_ARRAY_TASK_ID} starting on $(hostname) ==="
echo "    CTS_BENCH_ROOT : ${CTS_BENCH_ROOT}"
echo "    OPENLANE_SIF   : ${OPENLANE_SIF}"
echo "    SKY130_PDK     : ${SKY130_PDK}"
echo "    SHARDS_DIR     : ${SHARDS_DIR}"
echo "    iverilog       : $(which iverilog 2>/dev/null || echo NOT FOUND)"

# Validate key files exist before running
if [ ! -f "${OPENLANE_SIF}" ]; then
    echo "ERROR: ${OPENLANE_SIF} not found. Run setup.sh first." >&2
    exit 1
fi
if [ ! -d "${SKY130_PDK}" ]; then
    echo "ERROR: PDK not found at ${SKY130_PDK}. Run setup.sh first." >&2
    exit 1
fi
if [ ! -f "$(which iverilog 2>/dev/null)" ]; then
    echo "ERROR: iverilog not in PATH. Check module load command." >&2
    exit 1
fi

mkdir -p "${SHARDS_DIR}"
cd "${CTS_BENCH_ROOT}"

python3 "${CTS_BENCH_ROOT}/hpc/main-hpc.py" \
    "${SLURM_ARRAY_TASK_ID}" \
    "${ITERS_PER_TASK}"

echo "=== Task ${SLURM_ARRAY_TASK_ID} done ==="
