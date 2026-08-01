"""
ORFS-based placement for ASAP7 and NanGate45.

Runs OpenROAD-flow-scripts Makefile through 3_place (synthesis + floorplan
+ placement) inside the ORFS Singularity container.  PDK libs are bundled
inside the container — no external PDK bind needed.

Usage (called by main-hpc-orfs.py):
  python3 1-gen-placement-orfs.py <design_name> <clock_period_ns>
          <clock_port> [top_module] [max_core_util] [pdk] [pdk_scl]

Prints "PLACEMENT_TAG=<tag>" as the last non-error line so the caller can
parse the tag.
"""

import glob
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

# ── Env ───────────────────────────────────────────────────────────────────────
CTS_BENCH_ROOT = os.environ.get(
    "CTS_BENCH_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)
ORFS_SIF      = os.environ.get("ORFS_SIF",     os.path.join(os.path.expanduser("~"), "singularity", "orfs.sif"))
CONTAINER_CMD = os.environ.get("CONTAINER_CMD", "apptainer" if shutil.which("apptainer") else "singularity")

DESIGNS_WITH_INCLUDES = {"ethmac", "i2c", "usb_phy", "mem_ctrl", "wb_dma", "ac97_ctrl", "pci"}

# PDK → (clock_layer, buf_list, root_buf, liberty_time_unit_is_ps)
PDK_CTS_META = {
    "asap7": {
        "clock_layer":      "M4",
        "buf_list":         "BUFx4_ASAP7_75t_R BUFx8_ASAP7_75t_R BUFx16f_ASAP7_75t_R",
        "root_buf":         "BUFx4_ASAP7_75t_R",
        "time_unit_ps":     True,   # ASAP7 liberty uses picoseconds
    },
    "nangate45": {
        "clock_layer":      "metal4",
        "buf_list":         "CLKBUF_X1 CLKBUF_X2 CLKBUF_X3",
        "root_buf":         "CLKBUF_X3",
        "time_unit_ps":     False,  # NanGate45 liberty uses nanoseconds
    },
}


def sdc_clock_period(clock_period_ns: float, pdk: str) -> str:
    """Return clock period in the unit expected by this PDK's liberty."""
    if PDK_CTS_META[pdk]["time_unit_ps"]:
        return str(int(round(clock_period_ns * 1000)))   # ns → ps
    return f"{clock_period_ns:.4f}"                       # stay in ns


def run_orfs_placement(
    design_name,
    clock_period_ns,
    clock_port,
    top_module=None,
    max_core_util=70,
    pdk="asap7",
    pdk_scl=None,   # unused for ORFS (scl is part of the PDK platform)
):
    verilog_files = sorted(glob.glob(
        os.path.join(CTS_BENCH_ROOT, "designs", design_name, "rtl", "*.v")
    ))
    if not verilog_files:
        raise FileNotFoundError(f"No RTL files for {design_name}")

    top = top_module or design_name
    core_util   = random.randint(20, max(20, max_core_util))
    density     = min(round((core_util / 100.0) + 0.05 + random.uniform(0.0, 0.20), 2), 0.95)
    aspect_ratio = random.choice([0.7, 1.0, 1.4, 2.0])
    abc_area    = random.choice([0, 1])
    io_mode     = random.choice([0, 1])

    tag = f"{design_name}_orfs_{pdk[:4]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir     = os.path.join(CTS_BENCH_ROOT, "runs", tag)
    results_dir = os.path.join(run_dir, "orfs_results")
    objects_dir = os.path.join(run_dir, "orfs_objects")   # MUST be outside container
    logs_dir    = os.path.join(run_dir, "orfs_logs")
    reports_dir = os.path.join(run_dir, "orfs_reports")
    for d in (run_dir, results_dir, objects_dir, logs_dir, reports_dir):
        os.makedirs(d, exist_ok=True)

    # ── SDC (clock period in PDK-native units) ────────────────────────────
    period_str = sdc_clock_period(clock_period_ns, pdk)
    sdc_path = os.path.join(run_dir, "constraint.sdc")
    with open(sdc_path, "w") as f:
        f.write(f"""set clk_period {period_str}
create_clock -name {clock_port} -period $clk_period [get_ports {clock_port}]
set_clock_transition 0.15 [all_clocks]
set_clock_uncertainty 0.25 [all_clocks]
set_input_delay  -clock [get_clocks {clock_port}] -add_delay [expr $clk_period * 0.1] [all_inputs]
set_output_delay -clock [get_clocks {clock_port}] -add_delay [expr $clk_period * 0.1] [all_outputs]
""")

    # ── ORFS config.mk ───────────────────────────────────────────────────
    verilog_list = " ".join(verilog_files)
    include_line = ""
    if design_name in DESIGNS_WITH_INCLUDES:
        rtl_dir = os.path.join(CTS_BENCH_ROOT, "designs", design_name, "rtl")
        include_line = f"export VERILOG_INCLUDE_DIRS = {rtl_dir}"

    config_mk = os.path.join(run_dir, "config.mk")
    with open(config_mk, "w") as f:
        f.write(f"""export PLATFORM          = {pdk}
export DESIGN_NAME       = {top}
export VERILOG_FILES     = {verilog_list}
{include_line}
export CLOCK_PORT        = {clock_port}
export CLOCK_PERIOD      = {period_str}
export SDC_FILE          = {sdc_path}
export CORE_UTILIZATION  = {core_util}
export CORE_ASPECT_RATIO = {aspect_ratio}
export PLACE_DENSITY     = {density}
export CORE_MARGIN       = 2
export ABC_AREA          = {abc_area}
export TNS_END_PERCENT   = 100
export RESYNTH_TIMING_RECOVER = 0
""")

    # Source ORFS env.sh so Yosys/OpenROAD are in PATH.
    # OBJECTS_DIR must point outside the read-only container SIF.
    make_cmd = (
        "source /OpenROAD-flow-scripts/env.sh && "
        f"make -C /OpenROAD-flow-scripts/flow -j8 "
        f"DESIGN_CONFIG={config_mk} "
        f"RESULTS_DIR={results_dir} "
        f"OBJECTS_DIR={objects_dir} "
        f"LOGS_DIR={logs_dir} "
        f"REPORTS_DIR={reports_dir} "
        "place"
    )
    print(f"Running ORFS placement ({CONTAINER_CMD}) for {design_name} pdk={pdk} tag={tag}")
    result = subprocess.run([
        CONTAINER_CMD, "exec",
        "--bind", f"{CTS_BENCH_ROOT}:{CTS_BENCH_ROOT}",
        ORFS_SIF,
        "bash", "-c", make_cmd,
    ], cwd=run_dir)

    if result.returncode != 0:
        print(f"ORFS placement failed for {tag} (exit {result.returncode})")
        return None

    # Find placed ODB/DEF
    odb_candidates = sorted(glob.glob(os.path.join(results_dir, "3*place*.odb")))
    def_candidates = sorted(glob.glob(os.path.join(results_dir, "3*place*.def")))
    if not odb_candidates:
        print(f"No placed ODB found in {results_dir}")
        return None

    placed_odb = odb_candidates[-1]
    placed_def = def_candidates[-1] if def_candidates else ""
    print(f"  Placed ODB: {placed_odb}")

    # ── Write metadata ────────────────────────────────────────────────────
    stats = {
        "design_name":       design_name,
        "top_module":        top,
        "pdk":               pdk,
        "aspect_ratio":      aspect_ratio,
        "core_util":         core_util,
        "density":           density,
        "abc_area":          abc_area,
        "io_mode":           io_mode,
        "placed_odb":        placed_odb,
        "placed_def":        placed_def,
        "sdc_path":          sdc_path,
        "clock_period_ns":   clock_period_ns,
        "clock_period_str":  period_str,
        "clock_port":        clock_port,
        "results_dir":       results_dir,
    }
    with open(os.path.join(run_dir, "placement_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    with open(os.path.join(CTS_BENCH_ROOT, f"latest_stats_{tag}.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"PLACEMENT_TAG={tag}")
    return tag


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 1-gen-placement-orfs.py <design_name> <clock_period_ns> "
              "<clock_port> [top_module] [max_core_util] [pdk] [pdk_scl]")
        sys.exit(1)
    design_name    = sys.argv[1]
    clock_period   = float(sys.argv[2])
    clock_port     = sys.argv[3]
    top_module     = sys.argv[4] if len(sys.argv) > 4 else None
    max_core_util  = int(sys.argv[5])   if len(sys.argv) > 5 else 70
    pdk            = sys.argv[6]        if len(sys.argv) > 6 else "asap7"
    pdk_scl        = sys.argv[7]        if len(sys.argv) > 7 else None
    run_orfs_placement(design_name, clock_period, clock_port, top_module, max_core_util, pdk, pdk_scl)
