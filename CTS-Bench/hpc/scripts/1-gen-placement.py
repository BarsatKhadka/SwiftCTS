"""
HPC version of 1-gen-placement.py
Uses apptainer/singularity exec instead of --dockerized Docker mode.
All paths driven by env vars set in env.sh.
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
CTS_BENCH_ROOT = os.environ.get("CTS_BENCH_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
PDK_ROOT       = os.environ.get("PDK_ROOT",       os.path.join(os.path.expanduser("~"), "pdk", "sky130"))
PDK_HASH       = os.environ.get("PDK_HASH",       "0fe599b2afb6708d281543108caf8310912f54af")
SKY130_PDK     = os.environ.get("SKY130_PDK",     os.path.join(PDK_ROOT, "volare", "sky130", "versions", PDK_HASH))
OPENLANE_SIF   = os.environ.get("OPENLANE_SIF",   os.path.join(os.path.expanduser("~"), "singularity", "openlane2-2.3.10.sif"))
CONTAINER_CMD  = os.environ.get("CONTAINER_CMD",  "apptainer" if shutil.which("apptainer") else "singularity")

LIB_PATH = f"{SKY130_PDK}/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
DESIGNS_WITH_INCLUDES = {"ethmac", "i2c", "usb_phy", "mem_ctrl", "wb_dma", "ac97_ctrl", "pci"}


def extract_timing_paths(run_dir, design_name):
    final_dir = os.path.join(run_dir, "final")

    def find_file(subdir, ext):
        d = os.path.join(final_dir, subdir)
        if not os.path.exists(d):
            return None
        for f in sorted(os.listdir(d)):
            if f.endswith(ext) and os.path.isfile(os.path.join(d, f)):
                return os.path.join(d, f)
        return None

    netlist  = find_file("nl", ".v") or find_file("pnl", ".v")
    sdc_file = find_file("sdc", ".sdc")
    spef_file = find_file("spef", ".spef")

    if not netlist or not sdc_file:
        print(f"Missing netlist or SDC in {final_dir}")
        return None

    tech_lef   = f"{SKY130_PDK}/sky130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef"
    cell_lef   = f"{SKY130_PDK}/sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef"
    output_csv = os.path.join(run_dir, "timing_paths.csv")
    tcl_path   = os.path.join(run_dir, "extract_paths.tcl")
    spef_cmd   = f"read_spef {spef_file}" if spef_file else ""

    tcl = """
read_lef {tech_lef}
read_lef {cell_lef}
read_liberty {lib}
read_verilog {netlist}
link_design {design}
read_sdc {sdc}
{spef}

set fp [open "{output}" w]
puts $fp "launch_flop,capture_flop,slack"

set paths [find_timing_paths -path_delay max -group_count 50000 -unique_paths_to_endpoint]
puts "Found [llength $paths] setup paths"
foreach path $paths {{
    set sp [get_property $path startpoint]
    set ep [get_property $path endpoint]
    if {{[get_property $sp is_port] || [get_property $ep is_port]}} continue
    set sp_name [join [lrange [split [get_property $sp full_name] "/"] 0 end-1] "/"]
    set ep_name [join [lrange [split [get_property $ep full_name] "/"] 0 end-1] "/"]
    puts $fp "$sp_name,$ep_name,[get_property $path slack]"
}}

set hold_paths [find_timing_paths -path_delay min -group_count 50000 -unique_paths_to_endpoint]
puts "Found [llength $hold_paths] hold paths"
foreach path $hold_paths {{
    set sp [get_property $path startpoint]
    set ep [get_property $path endpoint]
    if {{[get_property $sp is_port] || [get_property $ep is_port]}} continue
    set sp_name [join [lrange [split [get_property $sp full_name] "/"] 0 end-1] "/"]
    set ep_name [join [lrange [split [get_property $ep full_name] "/"] 0 end-1] "/"]
    puts $fp "$sp_name,$ep_name,[get_property $path slack]"
}}
close $fp
puts "Done writing timing paths"
exit
""".format(
        tech_lef=tech_lef, cell_lef=cell_lef, lib=LIB_PATH,
        netlist=netlist, design=design_name, sdc=sdc_file,
        spef=spef_cmd, output=output_csv,
    )
    with open(tcl_path, "w") as f:
        f.write(tcl)

    result = subprocess.run([
        CONTAINER_CMD, "exec",
        "--bind", f"{CTS_BENCH_ROOT}:{CTS_BENCH_ROOT}",
        "--bind", f"{PDK_ROOT}:{PDK_ROOT}",
        OPENLANE_SIF,
        "openroad", "-exit", tcl_path,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  STA extraction failed: {result.stderr[-1000:]}")
        return None
    if os.path.exists(output_csv):
        n = sum(1 for _ in open(output_csv)) - 1
        print(f"  Extracted {n} flop pairs -> {output_csv}")
    return output_csv


def run_single_experiment(design_name, clock_period, clock_port, top_module=None, max_core_util=70):
    verilog_files = sorted(glob.glob(os.path.join(CTS_BENCH_ROOT, "designs", design_name, "rtl", "*.v")))
    if not verilog_files:
        raise FileNotFoundError(f"No RTL files for {design_name}")

    io_mode            = random.choice([0, 1])
    core_util          = random.randint(20, max(20, max_core_util))
    pl_density         = min(round((core_util / 100.0) + 0.05 + random.uniform(0.0, 0.20), 2), 0.99)
    fp_ratio           = random.choice([0.7, 1.0, 1.4, 2.0])
    synth_strategy     = random.choice(["AREA 0","AREA 1","AREA 2","DELAY 0","DELAY 1","DELAY 2","DELAY 3","DELAY 4"])
    time_driven        = random.choice([True, False])
    routability_driven = random.choice([True, False])

    tag = f"{design_name}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = {
        "DESIGN_NAME": top_module or design_name,
        "VERILOG_FILES": verilog_files,
        "CLOCK_PERIOD": clock_period,
        "CLOCK_PORT": clock_port,
        "SDC_FILE": os.path.join(CTS_BENCH_ROOT, "designs", "base.sdc"),
        "SYNTH_STRATEGY": synth_strategy,
        "FP_ASPECT_RATIO": fp_ratio,
        "FP_CORE_UTIL": core_util,
        "FP_IO_MODE": io_mode,
        "PL_TARGET_DENSITY": pl_density,
        "PL_TIME_DRIVEN": time_driven,
        "PL_ROUTABILITY_DRIVEN": routability_driven,
        "PDK": "sky130A",
        "STD_CELL_LIBRARY": "sky130_fd_sc_hd",
        "ERROR_ON_SYNTH_CHECKS": False,
    }
    if design_name in DESIGNS_WITH_INCLUDES:
        config["VERILOG_INCLUDE_DIRS"] = [os.path.join(CTS_BENCH_ROOT, "designs", design_name, "rtl")]

    config_path = os.path.join(CTS_BENCH_ROOT, f"tmp_ol_config_{tag}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Running OpenLane ({CONTAINER_CMD}) for {design_name} tag={tag}")
    result = subprocess.run([
        CONTAINER_CMD, "exec",
        "--bind", f"{CTS_BENCH_ROOT}:{CTS_BENCH_ROOT}",
        "--bind", f"{PDK_ROOT}:{PDK_ROOT}",
        "--pwd",  CTS_BENCH_ROOT,
        "--env",  f"PDK_ROOT={PDK_ROOT}",
        OPENLANE_SIF,
        "python3", "-m", "openlane",
        "-T", "OpenROAD.DetailedPlacement",
        "-S", "Verilator.Lint",
        "-S", "Checker.LintTimingConstructs",
        "-S", "Checker.LintErrors",
        "-S", "Checker.LintWarnings",
        "-S", "Checker.YosysSynthChecks",
        "--pdk-root", SKY130_PDK,
        "--run-tag", tag,
        config_path,
    ], cwd=CTS_BENCH_ROOT)

    try:
        os.remove(config_path)
    except FileNotFoundError:
        pass

    if result.returncode != 0:
        print(f"OpenLane failed for {tag} (exit {result.returncode})")
        return None

    run_dir = os.path.join(CTS_BENCH_ROOT, "runs", tag)
    if not os.path.exists(run_dir):
        print(f"Run dir not found: {run_dir}")
        return None

    with open(os.path.join(CTS_BENCH_ROOT, f"latest_run_{tag}.txt"), "w") as f:
        f.write(tag + "\n")

    stats = {
        "design_name": design_name, "aspect_ratio": fp_ratio,
        "core_util": core_util, "density": pl_density,
        "synth_strategy": synth_strategy, "io_mode": io_mode,
        "time_driven": int(time_driven), "routability_driven": int(routability_driven),
    }
    with open(os.path.join(run_dir, "placement_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    with open(os.path.join(CTS_BENCH_ROOT, f"latest_stats_{tag}.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print("\n--- Extracting timing paths ---")
    extract_timing_paths(run_dir, top_module or design_name)
    print(f"PLACEMENT_TAG={tag}")  # must be last line — parsed by main-hpc.py
    return tag


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 1-gen-placement.py <design_name> <clock_period> <clock_port> [top_module] [max_core_util]")
        sys.exit(1)
    design_name   = sys.argv[1]
    clock_period  = float(sys.argv[2])
    clock_port    = sys.argv[3]
    top_module    = sys.argv[4] if len(sys.argv) > 4 else None
    max_core_util = int(sys.argv[5]) if len(sys.argv) > 5 else 70
    run_single_experiment(design_name, clock_period, clock_port, top_module, max_core_util)
