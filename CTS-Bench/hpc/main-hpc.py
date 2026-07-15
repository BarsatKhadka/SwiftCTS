"""
CTS-Bench HPC single-iteration runner.

Each SLURM array task calls this script once:
    python3 main-hpc.py <task_id> <iterations_per_task>

task_id controls which design is picked (round-robin) and seeds randomness.
Output goes to $SHARDS_DIR/shard_<task_id>.csv (merged later by merge_csvs.py).
"""

import csv
import glob
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

# ── Env (set by env.sh) ───────────────────────────────────────────────────────
CTS_BENCH_ROOT = os.environ.get("CTS_BENCH_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATASET_ROOT   = os.environ.get("DATASET_ROOT",   os.path.join(os.path.dirname(CTS_BENCH_ROOT), "dataset_with_def"))
SHARDS_DIR     = os.environ.get("SHARDS_DIR",     os.path.join(DATASET_ROOT, "shards"))
OPENLANE_SIF   = os.environ.get("OPENLANE_SIF",   os.path.join(os.path.expanduser("~"), "singularity", "openlane2-2.3.10.sif"))
PDK_ROOT       = os.environ.get("PDK_ROOT",       os.path.join(os.path.expanduser("~"), "pdk", "sky130"))
PDK_HASH       = os.environ.get("PDK_HASH",       "0fe599b2afb6708d281543108caf8310912f54af")
SKY130_PDK     = os.environ.get("SKY130_PDK",     os.path.join(PDK_ROOT, "volare", "sky130", "versions", PDK_HASH))

LOG_FILE       = os.environ.get("LOG_FILE", os.path.join(DATASET_ROOT, "experiment_log.csv"))
KEPT_FILES_DIR = os.path.join(DATASET_ROOT, "placement_files")

DESIGN_CONFIG = {
    # IWLS designs
    "usb_phy":   {"clock_period": 10.0, "clock_port": "clk",       "top_module": "usb_phy"},
    "mem_ctrl":  {"clock_period": 10.0, "clock_port": "clk_i",     "top_module": "mc_top"},
    "jpeg":      {"clock_period": 10.0, "clock_port": "clk",       "top_module": "jpeg_top"},
    "wb_dma":    {"clock_period": 10.0, "clock_port": "clk_i",     "top_module": "wb_dma_top",  "max_core_util": 20},
    "ac97_ctrl": {"clock_period": 10.0, "clock_port": "clk_i",     "top_module": "ac97_top"},
    "pci":       {"clock_period": 10.0, "clock_port": "wb_clk_i",  "top_module": "pci_bridge32"},
    "i2c":       {"clock_period": 10.0, "clock_port": "wb_clk_i",  "top_module": "i2c_master_top"},
    "spi":       {"clock_period": 10.0, "clock_port": "clk_i",     "top_module": "simple_spi"},
    "tv80":      {"clock_period": 10.0, "clock_port": "clk",       "top_module": "tv80s"},
    "aes":       {"clock_period":  7.0, "clock_port": "clk",       "top_module": "aes"},
    "picorv32":  {"clock_period":  5.0, "clock_port": "clk",       "top_module": "picorv32"},
    "sha256":    {"clock_period":  9.0, "clock_port": "clk",       "top_module": "sha256"},
    "ethmac":    {"clock_period":  9.0, "clock_port": "wb_clk_i",  "top_module": "eth_top"},
    "zipdiv":    {"clock_period":  5.0, "clock_port": "i_clk",     "top_module": "zipdiv"},
    # PlaceDreamer designs
    "salsa20":   {"clock_period": 10.0, "clock_port": "clk",       "top_module": "salsa20"},
    "xtea":      {"clock_period": 10.0, "clock_port": "clock",     "top_module": "xtea"},
    "y_huff":    {"clock_period": 10.0, "clock_port": "clk",       "top_module": "y_huff"},
    "PPU":       {"clock_period": 10.0, "clock_port": "clk",       "top_module": "PPU"},
    "usb":       {"clock_period": 20.0, "clock_port": "clk_48",    "top_module": "usb"},
}

CSV_HEADER = [
    "run_id", "placement_id", "design_name",
    "aspect_ratio", "core_util", "density", "synth_strategy",
    "io_mode", "time_driven", "routability_driven",
    "cts_max_wire", "cts_buf_dist", "cts_cluster_size", "cts_cluster_dia",
    "skew_setup", "skew_hold",
    "setup_slack", "hold_slack",
    "setup_tns", "hold_tns",
    "setup_vio_count", "hold_vio_count",
    "power_total", "wirelength", "utilization",
    "clock_buffers", "clock_inverters", "timing_repair_buffers",
    "def_path", "saif_path", "timing_path_csv",
]


def setup_dirs():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(KEPT_FILES_DIR, exist_ok=True)
    os.makedirs(SHARDS_DIR, exist_ok=True)


def get_latest_run_tag():
    p = os.path.join(CTS_BENCH_ROOT, "latest_run.txt")
    return open(p).read().strip() if os.path.exists(p) else None


def save_essential_files(placement_id, design_name):
    run_dir  = os.path.join(CTS_BENCH_ROOT, "runs", placement_id)
    save_dir = os.path.join(KEPT_FILES_DIR, placement_id)
    os.makedirs(save_dir, exist_ok=True)

    dp_dirs  = glob.glob(os.path.join(run_dir, "*-openroad-detailedplacement"))
    dp_dir   = sorted(dp_dirs)[-1] if dp_dirs else os.path.join(run_dir, "33-openroad-detailedplacement")
    def_files = glob.glob(os.path.join(dp_dir, "*.def"))
    def_src   = sorted(def_files)[0] if def_files else os.path.join(dp_dir, f"{design_name}.def")
    saif_src  = os.path.join(run_dir, f"{design_name}.saif")
    timing_src = os.path.join(run_dir, "timing_paths.csv")

    saved = {"def_path": "", "saif_path": "", "timing_path": ""}
    for src, dst_name, key in [
        (def_src,    f"{design_name}.def",   "def_path"),
        (saif_src,   f"{design_name}.saif",  "saif_path"),
        (timing_src, "timing_paths.csv",     "timing_path"),
    ]:
        dst = os.path.join(save_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            saved[key] = dst
        else:
            print(f"  Warning: {src} not found")
    return saved


def log_to_shard(task_id, placement_id, saved_paths):
    shard_path = os.path.join(SHARDS_DIR, f"shard_{task_id:05d}.csv")
    write_header = not os.path.exists(shard_path)

    pl_stats = {}
    try:
        with open(os.path.join(CTS_BENCH_ROOT, "latest_stats.json")) as f:
            pl_stats = json.load(f)
    except FileNotFoundError:
        print("latest_stats.json not found"); return

    dataset_path = os.path.join(CTS_BENCH_ROOT, "runs", placement_id, "dataset.json")
    if not os.path.exists(dataset_path):
        print(f"dataset.json not found for {placement_id}"); return

    with open(dataset_path) as f:
        metric_map = {e["id"]: e["metrics"] for e in json.load(f)}

    base_cts_dir = os.path.join(CTS_BENCH_ROOT, "runs", placement_id, "CTS-experiments")

    with open(shard_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)

        for i in range(1, 11):
            cts_id = f"CTS-{i}"
            knobs  = {}
            knob_f = os.path.join(base_cts_dir, cts_id, "knobs.json")
            if os.path.exists(knob_f):
                with open(knob_f) as kf:
                    knobs = json.load(kf)
            m = metric_map.get(cts_id, {})
            writer.writerow([
                "",  # run_id filled by merge script
                placement_id,
                pl_stats.get("design_name", ""),
                pl_stats.get("aspect_ratio", ""),
                pl_stats.get("core_util", ""),
                pl_stats.get("density", ""),
                pl_stats.get("synth_strategy", ""),
                pl_stats.get("io_mode", ""),
                pl_stats.get("time_driven", ""),
                pl_stats.get("routability_driven", ""),
                knobs.get("CTS_CLK_MAX_WIRE_LENGTH", ""),
                knobs.get("CTS_DISTANCE_BETWEEN_BUFFERS", ""),
                knobs.get("CTS_SINK_CLUSTERING_SIZE", ""),
                knobs.get("CTS_SINK_CLUSTERING_MAX_DIAMETER", ""),
                m.get("skew_setup", ""),
                m.get("skew_hold", ""),
                m.get("setup_slack", ""),
                m.get("hold_slack", ""),
                m.get("setup_tns", ""),
                m.get("hold_tns", ""),
                m.get("setup_vio_count", ""),
                m.get("hold_vio_count", ""),
                m.get("power_total", ""),
                m.get("wirelength", ""),
                m.get("utilization", ""),
                m.get("clock_buffers", ""),
                m.get("clock_inverters", ""),
                m.get("timing_repair_buffers", ""),
                saved_paths["def_path"],
                saved_paths["saif_path"],
                saved_paths["timing_path"],
            ])
    print(f"  Logged 10 rows to {shard_path}")


def delete_run(placement_id):
    run_dir = os.path.join(CTS_BENCH_ROOT, "runs", placement_id)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)


def run_iteration(task_id, design_name):
    cfg          = DESIGN_CONFIG[design_name]
    clock_period = cfg["clock_period"]
    clock_port   = cfg["clock_port"]
    top_module   = cfg["top_module"]
    max_core_util = cfg.get("max_core_util", 70)

    placement_id = None
    try:
        # 1. Placement (Singularity-based OpenLane)
        print(f"[{design_name}] Placement...")
        r = subprocess.run([
            "python3",
            os.path.join(CTS_BENCH_ROOT, "hpc", "scripts", "1-gen-placement.py"),
            design_name, str(clock_period), clock_port, top_module, str(max_core_util),
        ], cwd=CTS_BENCH_ROOT)
        if r.returncode != 0:
            print(f"Placement failed. Skipping."); return
        placement_id = get_latest_run_tag()
        if not placement_id:
            print("No placement ID found. Skipping."); return

        dp = glob.glob(os.path.join(CTS_BENCH_ROOT, "runs", placement_id, "*-openroad-detailedplacement"))
        if not dp:
            print("No detailedplacement output. Skipping."); return

        # 2. SAIF (iverilog available via module load in job script)
        print(f"[{design_name}] SAIF simulation...")
        subprocess.run([
            "python3",
            os.path.join(CTS_BENCH_ROOT, "hpc", "scripts", "2-gen-saif.py"),
            placement_id,
        ], cwd=CTS_BENCH_ROOT, check=True)

        # 3. CTS sweep (Singularity)
        print(f"[{design_name}] CTS sweep (10 configs)...")
        subprocess.run([
            "singularity", "exec",
            "--bind", f"{CTS_BENCH_ROOT}:{CTS_BENCH_ROOT}",
            "--bind", f"{SKY130_PDK}:{SKY130_PDK}",
            "--pwd",  CTS_BENCH_ROOT,
            "--env",  f"PDK_ROOT={SKY130_PDK}",
            OPENLANE_SIF,
            "python3", "hpc/scripts/5-run-cts.py",
            placement_id, top_module, str(clock_period), clock_port,
        ], cwd=CTS_BENCH_ROOT, check=True)

        # 4. Parse reports
        print(f"[{design_name}] Parsing CTS reports...")
        subprocess.run([
            "python3",
            os.path.join(CTS_BENCH_ROOT, "scripts", "6-parse-cts-reports.py"),
            placement_id,
        ], cwd=CTS_BENCH_ROOT, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Step failed: {e}")
        if placement_id:
            delete_run(placement_id)
        return

    # 5. Save DEF/SAIF/timing
    design_name_logged = ""
    try:
        with open(os.path.join(CTS_BENCH_ROOT, "latest_stats.json")) as f:
            design_name_logged = json.load(f).get("design_name", "")
    except FileNotFoundError:
        pass

    saved = save_essential_files(placement_id, design_name_logged or top_module)

    # 6. Log to shard CSV
    log_to_shard(task_id, placement_id, saved)

    # 7. Cleanup
    delete_run(placement_id)
    print(f"[{design_name}] Done: {placement_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 main-hpc.py <task_id> [iterations_per_task]")
        sys.exit(1)

    task_id    = int(sys.argv[1])
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    random.seed(task_id * 1000 + 42)

    setup_dirs()
    os.chdir(CTS_BENCH_ROOT)

    design_names = list(DESIGN_CONFIG.keys())

    for i in range(iterations):
        design = design_names[(task_id * iterations + i) % len(design_names)]
        print(f"\n{'='*60}")
        print(f"Task {task_id} | Iteration {i+1}/{iterations} | Design: {design}")
        print(f"{'='*60}")
        run_iteration(task_id, design)
