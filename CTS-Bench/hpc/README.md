# CTS-Bench HPC

Data generation pipeline for SwiftCTS — produces the `experiment_log.csv` dataset
used to train and evaluate the SwiftCTS clock tree synthesis models.

**Repo:** https://github.com/BarsatKhadka/SwiftCTS  
Uses **Singularity** instead of Docker, **iverilog via module load** instead of nix.

---

## Quick Start on HPC

```bash
# 1. Clone the SwiftCTS repo to your HPC home
git clone https://github.com/BarsatKhadka/SwiftCTS.git ~/SwiftCTS

# 2. One-time setup: pull Singularity image, download sky130 PDK, create Python venv
bash ~/SwiftCTS/CTS-Bench/hpc/setup.sh

# 3. Adjust cluster-specific settings (see "Cluster Config" below), then submit
sbatch ~/SwiftCTS/CTS-Bench/hpc/slurm/run_array.sbatch

# 4. After all tasks finish, merge CSV shards into one file
source ~/SwiftCTS/CTS-Bench/hpc/env.sh
python3 ~/SwiftCTS/CTS-Bench/hpc/slurm/merge_csvs.py
```

---

## Cluster Config (edit before submitting)

**`hpc/slurm/run_array.sbatch`** — two lines to change:
```bash
#SBATCH --partition=cpu        # ← your cluster's CPU partition name
module load iverilog            # ← your cluster's iverilog module name
                                #   (might be icarus-verilog or similar)
```

**`hpc/env.sh`** — override these if you want non-default storage paths:
```bash
export DATASET_ROOT=/scratch/$USER/dataset_with_def   # default: ~/dataset_with_def
export PDK_ROOT=/shared/pdk/sky130                     # default: ~/pdk/sky130
export OPENLANE_SIF=/shared/singularity/openlane2.sif  # default: ~/singularity/...
```
All other paths auto-detect from the repo location — no edits needed.

---

## Scaling (rows = tasks × iters × 10)

Edit `run_array.sbatch`:
```bash
#SBATCH --array=0-199    # number of parallel tasks
ITERS_PER_TASK=5         # pipeline iterations per task
```

| Tasks | Iters/task | Total rows | Est. wall time |
|-------|-----------|------------|----------------|
| 100   | 5         | 5,000      | ~6h            |
| 200   | 5         | 10,000     | ~6h            |
| 200   | 10        | 20,000     | ~12h           |
| 500   | 10        | 50,000     | ~12h           |

---

## What Each File Does

| File | Purpose |
|------|---------|
| `env.sh` | All env vars — auto-detects CTS_BENCH_ROOT from its own path |
| `setup.sh` | One-time setup: Singularity pull, PDK via volare, Python venv |
| `main-hpc.py` | Single-task orchestrator (placement → SAIF → CTS×10 → shard) |
| `scripts/1-gen-placement.py` | OpenLane placement via `singularity exec` |
| `scripts/2-gen-saif.py` | Gate-level sim → SAIF (iverilog/vvp, no nix) |
| `scripts/5-run-cts.py` | CTS sweep (10 random configs per placement) |
| `slurm/run_array.sbatch` | SLURM array job — one task per CPU node |
| `slurm/merge_csvs.py` | Merge per-task shards → `experiment_log.csv` |
| `test_local.sh` | Local smoke-test (Docker) before going to HPC |

---

## Designs Covered (19 total)

**IWLS benchmarks (14):** aes, picorv32, sha256, zipdiv, i2c, spi, tv80, usb_phy,
mem_ctrl, jpeg, wb_dma, ac97_ctrl, pci, ethmac

**PlaceDreamer (5):** salsa20, xtea, y_huff, PPU, usb

---

## Output Structure

```
~/dataset_with_def/
├── experiment_log.csv          # merged master CSV (after merge_csvs.py)
├── shards/
│   ├── shard_00000.csv         # per-task shards (created during job run)
│   └── ...
└── placement_files/
    ├── aes_run_YYYYMMDD_HHMMSS/
    │   ├── aes.def             # required by SwiftCTS build_caches.py
    │   ├── aes.saif            # required by SwiftCTS build_caches.py
    │   └── timing_paths.csv    # required by SwiftCTS build_caches.py
    └── ...
```

## Using the Dataset with SwiftCTS

After data generation is done, point SwiftCTS at the placement files:

```bash
export SWIFTCTS_PLACEMENT_DIR=~/dataset_with_def/placement_files
python3 ~/SwiftCTS/SwiftCTS/build_caches.py
```
