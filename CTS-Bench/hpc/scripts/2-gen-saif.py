"""
HPC version of 2-gen-saif.py
Runs iverilog/vvp inside the OpenLane Singularity container — no host install needed.
The OpenLane 2.3.10 SIF includes Icarus Verilog (iverilog/vvp).
"""
import glob
import os
import shutil
import subprocess
import sys

if len(sys.argv) < 2:
    print("Usage: python3 2-gen-saif.py <RUN_TAG>")
    sys.exit(1)

CTS_BENCH_ROOT = os.environ.get("CTS_BENCH_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
OPENLANE_SIF   = os.environ.get("OPENLANE_SIF",  os.path.expanduser("~/singularity/openlane2-2.3.10.sif"))
CONTAINER_CMD  = os.environ.get("CONTAINER_CMD", "apptainer" if shutil.which("apptainer") else "singularity")
FILENAME       = sys.argv[1]

DESIGN_CONFIG = {
    "picorv32":  {"tb_file": "testbench.v",       "vcd_file": "testbench.vcd",      "needs_firmware": True},
    "aes":       {"tb_file": "tb_aes.v",             "vcd_file": "tb_aes.vcd",         "needs_firmware": False},
    "ethmac":    {"tb_file": "tb_eth_top.v",       "vcd_file": "testbench.vcd",      "needs_firmware": False},
    "sha256":    {"tb_file": "tb_sha256.v",        "vcd_file": "tb_sha256.vcd",      "needs_firmware": False},
    "zipdiv":    {"tb_file": "tb_zipdiv.v",        "vcd_file": "tb_zipdiv.vcd",      "needs_firmware": False},
    "i2c":       {"tb_file": "tb_i2c.v",           "vcd_file": "tb_i2c.vcd",         "needs_firmware": False},
    "spi":       {"tb_file": "tb_spi.v",           "vcd_file": "tb_spi.vcd",         "needs_firmware": False},
    "tv80":      {"tb_file": "tb_tv80.v",          "vcd_file": "tb_tv80.vcd",        "needs_firmware": False},
    "usb_phy":   {"tb_file": "tb_usb_phy.v",      "vcd_file": "tb_usb_phy.vcd",     "needs_firmware": False},
    "mem_ctrl":  {"tb_file": "tb_mem_ctrl.v",      "vcd_file": "tb_mem_ctrl.vcd",    "needs_firmware": False},
    "jpeg":      {"tb_file": "tb_jpeg.v",          "vcd_file": "tb_jpeg.vcd",        "needs_firmware": False},
    "wb_dma":    {"tb_file": "tb_wb_dma.v",        "vcd_file": "tb_wb_dma.vcd",      "needs_firmware": False},
    "ac97_ctrl": {"tb_file": "tb_ac97_ctrl.v",     "vcd_file": "tb_ac97_ctrl.vcd",   "needs_firmware": False},
    "pci":       {"tb_file": "tb_pci.v",           "vcd_file": "tb_pci.vcd",         "needs_firmware": False},
    "salsa20":   {"tb_file": "tb_salsa20.v",       "vcd_file": "tb_salsa20.vcd",     "needs_firmware": False},
    "xtea":      {"tb_file": "tb_xtea.v",          "vcd_file": "tb_xtea.vcd",        "needs_firmware": False},
    "y_huff":    {"tb_file": "tb_y_huff.v",        "vcd_file": "tb_y_huff.vcd",      "needs_firmware": False},
    "PPU":       {"tb_file": "tb_PPU.v",           "vcd_file": "tb_PPU.vcd",         "needs_firmware": False},
    "usb":       {"tb_file": "tb_usb.v",           "vcd_file": "tb_usb.vcd",         "needs_firmware": False},
}

RUN_DIR        = os.path.join(CTS_BENCH_ROOT, "runs", FILENAME)
DESIGN_NAME    = FILENAME.split("_run_")[0]
DESIGN_SRC_DIR = os.path.join(CTS_BENCH_ROOT, "designs", DESIGN_NAME)

_dp_dirs = glob.glob(os.path.join(RUN_DIR, "*-openroad-detailedplacement"))
if not _dp_dirs:
    sys.exit(f"[ERROR] No detailedplacement dir in {RUN_DIR}")
PLACEMENT_DIR = sorted(_dp_dirs)[-1]

_nl_files = glob.glob(os.path.join(PLACEMENT_DIR, "*.nl.v"))
if not _nl_files:
    sys.exit(f"[ERROR] No .nl.v netlist in {PLACEMENT_DIR}")
NETLIST_PATH = sorted(_nl_files)[0]

cfg             = DESIGN_CONFIG[DESIGN_NAME]
TESTBENCH_PATH  = os.path.join(DESIGN_SRC_DIR, "tb", cfg["tb_file"])
PRIMITIVES_PATH = os.path.join(CTS_BENCH_ROOT, "designs", "primitives.v")
SKY130_PATH     = os.path.join(CTS_BENCH_ROOT, "designs", "sky130_fd_sc_hd.v")
WAVE2SAIF_PATH  = os.path.join(CTS_BENCH_ROOT, "vcd2saif.py")
FIRMWARE_PATH   = os.path.join(DESIGN_SRC_DIR, "firmware", "firmware.hex")

SIM_EXEC  = os.path.join(RUN_DIR, "sim_gate.out")
VCD_FILE  = os.path.join(RUN_DIR, cfg["vcd_file"])
SAIF_FILE = os.path.join(RUN_DIR, f"{DESIGN_NAME}.saif")

TB_INCLUDE  = os.path.join(DESIGN_SRC_DIR, "tb")
RTL_INCLUDE = os.path.join(DESIGN_SRC_DIR, "rtl")

DESIGNS_WITH_INCLUDES = {"ethmac", "i2c", "usb_phy", "mem_ctrl", "wb_dma", "ac97_ctrl", "pci"}

# iverilog uses absolute paths — any --pwd works
SING_PREFIX_COMPILE = [
    CONTAINER_CMD, "exec",
    "--bind", f"{CTS_BENCH_ROOT}:{CTS_BENCH_ROOT}",
    "--pwd",  CTS_BENCH_ROOT,
    OPENLANE_SIF,
]

# vvp writes $dumpfile with a relative path — must set --pwd to RUN_DIR
SING_PREFIX_SIM = [
    CONTAINER_CMD, "exec",
    "--bind", f"{CTS_BENCH_ROOT}:{CTS_BENCH_ROOT}",
    "--pwd",  RUN_DIR,
    OPENLANE_SIF,
]


def run(cmd, cwd=None):
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


# 1. Compile with iverilog (inside Singularity — no host install required)
iverilog_cmd = SING_PREFIX_COMPILE + ["iverilog", "-o", SIM_EXEC, "-DFUNCTIONAL", "-DUNIT_DELAY=#1"]
if DESIGN_NAME in DESIGNS_WITH_INCLUDES:
    iverilog_cmd += ["-I", TB_INCLUDE, "-I", RTL_INCLUDE]
iverilog_cmd += [TESTBENCH_PATH, NETLIST_PATH, PRIMITIVES_PATH, SKY130_PATH]
run(iverilog_cmd)

if not os.path.exists(SIM_EXEC):
    sys.exit(f"[ERROR] iverilog returned 0 but {SIM_EXEC} was not created (silent compile failure)")

# 2. Simulate with vvp (inside Singularity — pwd=RUN_DIR so $dumpfile lands there)
vvp_cmd = SING_PREFIX_SIM + ["vvp", SIM_EXEC, "+vcd"]
if cfg["needs_firmware"]:
    if not os.path.exists(FIRMWARE_PATH):
        sys.exit(f"[ERROR] Firmware not found: {FIRMWARE_PATH}")
    vvp_cmd.append(f"+firmware={FIRMWARE_PATH}")
run(vvp_cmd)

# 3. VCD → SAIF (Python-native, runs on host — no container needed)
run(["python3", WAVE2SAIF_PATH, "-o", SAIF_FILE, VCD_FILE])
