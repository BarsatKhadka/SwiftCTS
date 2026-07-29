"""
CTS-Bench HPC orchestrator — ORFS variant (ASAP7 / NanGate45).

Mirrors main-hpc.py but replaces OpenLane with ORFS Makefile-based
placement and raw OpenROAD TCL-based CTS sweep.  No SAIF step (power
column in the CSV will be empty for ORFS rows).

Usage:
  python3 main-hpc-orfs.py <task_id> [iterations_per_task]

ACTIVE_PDK env var selects the PDK (asap7 | nangate45).
"""

import csv
import glob
import json
import os
import random
import shutil
import subprocess
import sys

# ── Env ───────────────────────────────────────────────────────────────────────
CTS_BENCH_ROOT = os.environ.get("CTS_BENCH_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATASET_ROOT   = os.environ.get("DATASET_ROOT",   os.path.join(os.path.dirname(CTS_BENCH_ROOT), "dataset_with_def"))
SHARDS_DIR     = os.environ.get("SHARDS_DIR",     os.path.join(DATASET_ROOT, "shards"))
ORFS_SIF       = os.environ.get("ORFS_SIF",       os.path.join(os.path.expanduser("~"), "singularity", "orfs.sif"))
CONTAINER_CMD  = os.environ.get("CONTAINER_CMD",  "apptainer" if shutil.which("apptainer") else "singularity")
ACTIVE_PDK     = os.environ.get("ACTIVE_PDK",     "asap7")

KEPT_FILES_DIR = os.path.join(DATASET_ROOT, "placement_files")

PDK_PROFILES = {
    "asap7": {
        "pdk":               "asap7",
        "scl":               "asap7sc7p5t_28_R",
        "clock_period_mult": 0.05,   # 10 ns baseline → 0.5 ns for asap7
    },
    "nangate45": {
        "pdk":               "nangate45",
        "scl":               "NangateOpenCellLibrary",
        "clock_period_mult": 0.25,   # 10 ns baseline → 2.5 ns for nangate45
    },
}

if ACTIVE_PDK not in PDK_PROFILES:
    sys.exit(f"Unknown ACTIVE_PDK='{ACTIVE_PDK}'. Choose from: {list(PDK_PROFILES)}")

DESIGN_CONFIG = {
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
    "salsa20":   {"clock_period": 10.0, "clock_port": "clk",       "top_module": "salsa20"},
    "xtea":      {"clock_period": 10.0, "clock_port": "clock",     "top_module": "xtea"},
    "y_huff":    {"clock_period": 10.0, "clock_port": "clk",       "top_module": "y_huff"},
    "PPU":       {"clock_period": 10.0, "clock_port": "clk",       "top_module": "PPU"},
    "usb":       {"clock_period": 20.0, "clock_port": "clk_48",    "top_module": "usb"},
}

# Shared CSV schema with main-hpc.py (power_total will be empty for ORFS rows)
CSV_HEADER = [
    "run_id", "placement_id", "design_name", "pdk_name",
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
    for d in (DATASET_ROOT, KEPT_FILES_DIR, SHARDS_DIR):
        os.makedirs(d, exist_ok=True)


def save_essential_files(placement_id, stats):
    """Copy placed DEF (if available) to the permanent dataset directory."""
    save_dir = os.path.join(KEPT_FILES_DIR, placement_id)
    os.makedirs(save_dir, exist_ok=True)

    design_name = stats.get("design_name", placement_id)
    saved = {"def_path": "", "saif_path": "", "timing_path": ""}

    placed_def = stats.get("placed_def", "")
    if placed_def and os.path.exists(placed_def):
        dst = os.path.join(save_dir, f"{design_name}.def")
        shutil.copy2(placed_def, dst)
        saved["def_path"] = dst

    return saved


def log_to_shard(task_id, placement_id, saved_paths, pdk_name):
    shard_path  = os.path.join(SHARDS_DIR, f"shard_orfs_{task_id:05d}.csv")
    write_header = not os.path.exists(shard_path)

    stats_path = os.path.join(CTS_BENCH_ROOT, f"latest_stats_{placement_id}.json")
    try:
        with open(stats_path) as f:
            pl_stats = json.load(f)
    except FileNotFoundError:
        print(f"latest_stats_{placement_id}.json not found"); return

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
                "",
                placement_id,
                pl_stats.get("design_name", ""),
                pdk_name,
                pl_stats.get("aspect_ratio", ""),
                pl_stats.get("core_util", ""),
                pl_stats.get("density", ""),
                f"ABC_AREA={pl_stats.get('abc_area', '')}",
                pl_stats.get("io_mode", ""),
                "",   # time_driven (not tracked for ORFS)
                "",   # routability_driven (not tracked for ORFS)
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
        try:
            shutil.rmtree(run_dir)
        except OSError as e:
            print(f"  Warning: could not delete {run_dir}: {e}")


def run_iteration(task_id, design_name):
    cfg          = DESIGN_CONFIG[design_name]
    pdk_prof     = PDK_PROFILES[ACTIVE_PDK]
    clock_period = round(cfg["clock_period"] * pdk_prof["clock_period_mult"], 4)
    clock_port   = cfg["clock_port"]
    top_module   = cfg["top_module"]
    max_core_util = cfg.get("max_core_util", 70)

    print(f"  PDK={ACTIVE_PDK}  clock_period={clock_period}ns")

    placement_id = None
    try:
        # 1. ORFS placement — retry up to 3x
        for attempt in range(1, 4):
            print(f"[{design_name}] ORFS placement (attempt {attempt}/3)...")
            r = subprocess.run([
                "python3",
                os.path.join(CTS_BENCH_ROOT, "hpc", "scripts", "1-gen-placement-orfs.py"),
                design_name, str(clock_period), clock_port, top_module, str(max_core_util),
                pdk_prof["pdk"], pdk_prof["scl"],
            ], cwd=CTS_BENCH_ROOT, capture_output=True, text=True)
            print(r.stdout)
            if r.stderr:
                print(r.stderr, file=sys.stderr)
            if r.returncode != 0:
                print(f"Placement attempt {attempt} failed (exit {r.returncode}).")
                continue
            for line in (r.stdout or "").splitlines():
                if line.startswith("PLACEMENT_TAG="):
                    placement_id = line.split("=", 1)[1].strip()
            if placement_id:
                break
            print(f"No PLACEMENT_TAG on attempt {attempt}.")
        if not placement_id:
            print("Placement failed after 3 attempts. Skipping."); return

        stats_path = os.path.join(CTS_BENCH_ROOT, f"latest_stats_{placement_id}.json")
        with open(stats_path) as f:
            pl_stats = json.load(f)

        if not os.path.exists(pl_stats.get("placed_odb", "")):
            print("Placed ODB missing after placement. Skipping."); return

        # 2. CTS sweep (10 configs via direct openroad TCL)
        print(f"[{design_name}] ORFS CTS sweep (10 configs)...")
        subprocess.run([
            "python3",
            os.path.join(CTS_BENCH_ROOT, "hpc", "scripts", "5-run-cts-orfs.py"),
            placement_id, pdk_prof["pdk"],
        ], cwd=CTS_BENCH_ROOT, check=True)

        # 3. Parse metrics
        print(f"[{design_name}] Parsing CTS reports...")
        subprocess.run([
            "python3",
            os.path.join(CTS_BENCH_ROOT, "hpc", "scripts", "6-parse-cts-reports-orfs.py"),
            placement_id,
        ], cwd=CTS_BENCH_ROOT, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Step failed: {e}")
        if placement_id:
            delete_run(placement_id)
        return

    # 4. Save essential files
    saved = save_essential_files(placement_id, pl_stats)

    # 5. Log to shard
    log_to_shard(task_id, placement_id, saved, ACTIVE_PDK)

    # 6. Cleanup
    delete_run(placement_id)
    print(f"[{design_name}] Done: {placement_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 main-hpc-orfs.py <task_id> [iterations_per_task]")
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
        print(f"Task {task_id} | Iter {i+1}/{iterations} | Design: {design} | PDK: {ACTIVE_PDK}")
        print(f"{'='*60}")
        run_iteration(task_id, design)
