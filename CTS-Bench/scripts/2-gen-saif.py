import glob
import os
import subprocess
import sys


if len(sys.argv) < 2:
    print("Error: You must provide the Run Tag as an argument!")
    print("Usage: python 2-gen-saif.py <RUN_TAG>")
    sys.exit(1)

FILENAME = sys.argv[1]  # Takes the argument passed from Bash

DESIGN_CONFIG = {
    "picorv32": {
        "tb_file": "testbench.v",
        "vcd_file": "testbench.vcd",
        "needs_firmware": True
    },
    "aes": {
        "tb_file": "tb_aes.v",
        "vcd_file": "tb_aes.vcd", 
        "needs_firmware": False 
    },
        "ethmac": {
        "tb_file": "tb_eth_top.v",    
        "vcd_file": "testbench.vcd",  # 
        "needs_firmware": False
    } ,
    "sha256": {
        "tb_file": "tb_sha256.v",
        "vcd_file": "tb_sha256.vcd",
        "needs_firmware": False
    },
    "zipdiv": {
        "tb_file": "tb_zipdiv.v",
        "vcd_file": "tb_zipdiv.vcd",
        "needs_firmware": False
    },
    "i2c": {
        "tb_file": "tb_i2c.v",
        "vcd_file": "tb_i2c.vcd",
        "needs_firmware": False
    },
    "spi": {
        "tb_file": "tb_spi.v",
        "vcd_file": "tb_spi.vcd",
        "needs_firmware": False
    },
    "tv80": {
        "tb_file": "tb_tv80.v",
        "vcd_file": "tb_tv80.vcd",
        "needs_firmware": False
    },
    "usb_phy": {
        "tb_file": "tb_usb_phy.v",
        "vcd_file": "tb_usb_phy.vcd",
        "needs_firmware": False
    },
    "mem_ctrl": {
        "tb_file": "tb_mem_ctrl.v",
        "vcd_file": "tb_mem_ctrl.vcd",
        "needs_firmware": False
    },
    "jpeg": {
        "tb_file": "tb_jpeg.v",
        "vcd_file": "tb_jpeg.vcd",
        "needs_firmware": False
    },
    "wb_dma": {
        "tb_file": "tb_wb_dma.v",
        "vcd_file": "tb_wb_dma.vcd",
        "needs_firmware": False
    },
    "ac97_ctrl": {
        "tb_file": "tb_ac97_ctrl.v",
        "vcd_file": "tb_ac97_ctrl.vcd",
        "needs_firmware": False
    },
    "pci": {
        "tb_file": "tb_pci.v",
        "vcd_file": "tb_pci.vcd",
        "needs_firmware": False
    },
    "salsa20": {
        "tb_file": "tb_salsa20.v",
        "vcd_file": "tb_salsa20.vcd",
        "needs_firmware": False
    },
    "xtea": {
        "tb_file": "tb_xtea.v",
        "vcd_file": "tb_xtea.vcd",
        "needs_firmware": False
    },
    "y_huff": {
        "tb_file": "tb_y_huff.v",
        "vcd_file": "tb_y_huff.vcd",
        "needs_firmware": False
    },
    "PPU": {
        "tb_file": "tb_PPU.v",
        "vcd_file": "tb_PPU.vcd",
        "needs_firmware": False
    },
    "usb": {
        "tb_file": "tb_usb.v",
        "vcd_file": "tb_usb.vcd",
        "needs_firmware": False
    },
}



# paths
PROJECT_ROOT = os.getcwd()
RUN_DIR = os.path.join(PROJECT_ROOT, "runs", FILENAME)
_dp_dirs = glob.glob(os.path.join(RUN_DIR, "*-openroad-detailedplacement"))
if not _dp_dirs:
    sys.exit(f"[ERROR] No detailedplacement dir found in {RUN_DIR}")
PLACEMENT_DIR = sorted(_dp_dirs)[-1]

DESIGN_NAME = FILENAME.split("_run_")[0]
DESIGN_SRC_DIR = os.path.join(PROJECT_ROOT, "designs", DESIGN_NAME)


# (Required for full testbench)
FIRMWARE_PATH = os.path.join(DESIGN_SRC_DIR, "firmware", "firmware.hex")

current_config = DESIGN_CONFIG[DESIGN_NAME]
# Netlist is named after the top module (may differ from dir/design name)
_nl_files = glob.glob(os.path.join(PLACEMENT_DIR, "*.nl.v"))
if not _nl_files:
    sys.exit(f"[ERROR] No .nl.v netlist found in {PLACEMENT_DIR}")
NETLIST_PATH = sorted(_nl_files)[0]
TESTBENCH_PATH = os.path.join(DESIGN_SRC_DIR, "tb", current_config["tb_file"])
PRIMITIVES_PATH = os.path.join(PROJECT_ROOT , "designs" , "primitives.v")
SKY130_PATH = os.path.join(PROJECT_ROOT,  "designs", "sky130_fd_sc_hd.v")
WAVE2SAIF_PATH = os.path.join(PROJECT_ROOT, "vcd2saif.py")

# # Output files (Saved inside the placement folder for organization)
SIM_EXEC = os.path.join(RUN_DIR, "sim_gate.out")
VCD_FILE = os.path.join(RUN_DIR, current_config["vcd_file"])
SAIF_FILE = os.path.join(RUN_DIR, f"{DESIGN_NAME}.saif")



def run_command(cmd, cwd=None):
    """Helper to run shell commands and print output"""
    print(f"[RUNNING]: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
        print(result.stdout) # Uncomment for verbose logs
    except subprocess.CalledProcessError as e:
        print(f"[ERROR]: Command failed!\n{e.stderr}")
        sys.exit(1)



# Define include directories based on your file structure
TB_INCLUDE = os.path.join(DESIGN_SRC_DIR, "tb")
RTL_INCLUDE = os.path.join(DESIGN_SRC_DIR, "rtl") 
TB_COP_PATH = os.path.join(DESIGN_SRC_DIR, "tb", "tb_cop.v")

# Missing module definitions
ETH_COP_PATH = os.path.join(DESIGN_SRC_DIR, "rtl", "eth_cop.v") # Defines eth_cop


# Start with the base command components
iverilog_cmd = [
    "iverilog",
    "-o", SIM_EXEC,
    "-DFUNCTIONAL",
    "-DUNIT_DELAY=#1"
]

# Add include paths for designs that use `include directives in RTL/TB
DESIGNS_WITH_INCLUDES = {"ethmac", "i2c", "usb_phy", "mem_ctrl", "wb_dma", "ac97_ctrl", "pci"}
if DESIGN_NAME in DESIGNS_WITH_INCLUDES:
    iverilog_cmd.extend([
        "-I", TB_INCLUDE,
        "-I", RTL_INCLUDE
    ])

# append if ethmac
iverilog_cmd.extend([
    TESTBENCH_PATH,   
    NETLIST_PATH,     
    PRIMITIVES_PATH,  
    SKY130_PATH       
])


# 1. Compile with Iverilog
run_command(iverilog_cmd)


if not os.path.exists(WAVE2SAIF_PATH):
    print(f"[ERROR] vcd2saif.py not found at {WAVE2SAIF_PATH}")
    sys.exit(1)

# Run vvp to get vcd
vvp_cmd = ["vvp", SIM_EXEC, "+vcd"]

if current_config["needs_firmware"]:
    if not os.path.exists(FIRMWARE_PATH):
        print(f"[ERROR] Firmware not found at {FIRMWARE_PATH}")
        sys.exit(1)
    vvp_cmd.append(f"+firmware={FIRMWARE_PATH}")

run_command(vvp_cmd, cwd=RUN_DIR)

# VCD to SAIF (Python-native, replaces Linux-only wave2saif binary)
wave2saif_cmd = ["python3", WAVE2SAIF_PATH, "-o", SAIF_FILE, VCD_FILE]
run_command(wave2saif_cmd)


