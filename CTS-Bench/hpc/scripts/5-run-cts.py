"""
HPC version of 5-run-cts.py
PDK path driven by PDK_ROOT / SKY130_PDK env vars (set in env.sh).
Runs inside the Singularity container so openlane is importable.
"""
import json
import os
import random
import glob
from pathlib import Path as SysPath
import sys
from openlane.common import Path
from openlane.flows import SequentialFlow
from openlane.steps import OpenROAD
from openlane.state import State, DesignFormat

if len(sys.argv) > 1:
    FILENAME = sys.argv[1]
elif os.path.exists("latest_run.txt"):
    FILENAME = open("latest_run.txt").read().strip()
else:
    sys.exit("Error: No Run Tag provided and latest_run.txt not found.")

# PDK from env — set by singularity --env flags in main-hpc.py
SKY130_PDK_ENV = os.environ.get("SKY130_PDK", "")
PDK_ROOT_ENV   = os.environ.get("PDK_ROOT", os.path.expanduser("~/pdk/sky130"))
MY_PDK_ROOT    = SKY130_PDK_ENV if SKY130_PDK_ENV else PDK_ROOT_ENV
os.environ["PDK_ROOT"] = MY_PDK_ROOT


class CTSOnlyFlow(SequentialFlow):
    Steps = [OpenROAD.CTS, OpenROAD.STAMidPNR]


def load_snapshot(base_tag):
    final_dir = SysPath(f"./runs/{base_tag}/final")
    state_data = {}
    folder_map = {
        "odb":    DesignFormat.ODB,
        "def":    DesignFormat.DEF,
        "nl":     DesignFormat.NETLIST,
        "pnl":    DesignFormat.POWERED_NETLIST,
        "json_h": DesignFormat.JSON_HEADER,
        "sdc":    DesignFormat.SDC,
        "sdf":    DesignFormat.SDF,
        "spef":   DesignFormat.SPEF,
    }
    for folder, fmt in folder_map.items():
        target = final_dir / folder
        if target.exists():
            files = [f for f in target.glob("*") if f.is_file()]
            if files:
                state_data[fmt] = Path(str(files[0].resolve()))
                print(f"   [FOUND] {folder.upper()}: {files[0].name}")
    if not state_data:
        raise ValueError("Snapshot empty!")
    return State(state_data)


def run_cts_from_placement(DESIGN, clock_period, clock_port):
    base_state = load_snapshot(FILENAME)
    for i in range(10):
        knobs = {
            "CTS_SINK_CLUSTERING_MAX_DIAMETER": random.randint(35, 70),
            "CTS_SINK_CLUSTERING_SIZE":         random.randint(12, 30),
            "CTS_DISTANCE_BETWEEN_BUFFERS":     random.randint(70, 150),
            "CTS_CLK_MAX_WIRE_LENGTH":          random.randint(130, 280),
        }
        print(f"  CTS-{i+1} knobs: {knobs}")

        # Put both knobs.json and OpenLane output in the same directory so
        # 6-parse-cts-reports.py can find state_out.json via recursive glob.
        target_dir = os.path.abspath(
            os.path.join("runs", FILENAME, "CTS-experiments", f"CTS-{i+1}")
        )
        os.makedirs(target_dir, exist_ok=True)

        with open(os.path.join(target_dir, "knobs.json"), "w") as f:
            json.dump(knobs, f, indent=4)

        config = {
            "DESIGN_NAME": DESIGN,
            "PDK": "sky130A",
            "STD_CELL_LIBRARY": "sky130_fd_sc_hd",
            "CLOCK_PORT": clock_port,
            "CLOCK_PERIOD": clock_period,
            **knobs,
        }

        flow = CTSOnlyFlow(
            config=config,
            design_dir=".",
            pdk_root=MY_PDK_ROOT,
            pdk="sky130A",
            scl="sky130_fd_sc_hd",
        )
        try:
            # _force_run_dir puts OpenLane output directly in target_dir so
            # state_out.json lands at target_dir/{step}/state_out.json —
            # exactly where 6-parse-cts-reports.py's recursive glob looks.
            flow.start(
                with_initial_state=base_state,
                _force_run_dir=target_dir,
            )
            print(f"  CTS-{i+1} complete")
        except Exception as e:
            print(f"  CTS-{i+1} failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 5-run-cts.py <run_tag> <design> <clock_period> <clock_port>")
        sys.exit(1)
    _, _, DESIGN, clock_period, clock_port = sys.argv[:5]
    run_cts_from_placement(DESIGN, float(clock_period), clock_port)
