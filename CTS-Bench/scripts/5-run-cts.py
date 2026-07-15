import json
import os
import random
import glob
from pathlib import Path as SysPath
import sys
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


MY_PDK_ROOT = "/Users/barsat/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af"
os.environ["PDK_ROOT"] = MY_PDK_ROOT


class CTSOnlyFlow(SequentialFlow):
    Steps = [
        OpenROAD.CTS,           
        OpenROAD.STAMidPNR    
    ]

#loading the base snapshot of placement as State to continue cts from there 
def load_snapshot(base_tag):
    final_dir = SysPath(f"./runs/{base_tag}/final")
    
    state_data = {}
    folder_map = {
        "odb": DesignFormat.ODB,
        "def": DesignFormat.DEF,
        "nl":  DesignFormat.NETLIST,           
        "pnl": DesignFormat.POWERED_NETLIST,   
        "json_h": DesignFormat.JSON_HEADER,    
        "sdc": DesignFormat.SDC,
        "sdf": DesignFormat.SDF,               
        "spef": DesignFormat.SPEF,             
    }

    #goes through each folder in the final directory of the base and maps the files to state data 
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
    BASE = FILENAME
    try:
        base_state = load_snapshot(BASE)
        for i in range(10):
            knobs = {
                   "CTS_SINK_CLUSTERING_MAX_DIAMETER": random.randint(35, 70),

                    "CTS_SINK_CLUSTERING_SIZE": random.randint(12, 30),

                
                    "CTS_DISTANCE_BETWEEN_BUFFERS": random.randint(70, 150),

                    
                    "CTS_CLK_MAX_WIRE_LENGTH": random.randint(130, 280),
                }
            
            print(f"  Knobs: {knobs}")
            
            config = {
                "DESIGN_NAME": DESIGN,
                "PDK": "sky130A",
                "STD_CELL_LIBRARY": "sky130_fd_sc_hd",
                "SDC_FILE": "./designs/base.sdc",
                "CLOCK_PERIOD": clock_period,
                "CLOCK_PORT": clock_port,
                "CTS_SINK_CLUSTERING_SIZE": knobs["CTS_SINK_CLUSTERING_SIZE"],
                "CTS_SINK_CLUSTERING_MAX_DIAMETER": knobs["CTS_SINK_CLUSTERING_MAX_DIAMETER"],
                "CTS_CLK_MAX_WIRE_LENGTH": knobs["CTS_CLK_MAX_WIRE_LENGTH"],
                "CTS_DISTANCE_BETWEEN_BUFFERS": knobs["CTS_DISTANCE_BETWEEN_BUFFERS"],
            }

            flow = CTSOnlyFlow(config=config, design_dir=".")

            target_dir = os.path.abspath(os.path.join("runs", FILENAME, "CTS-experiments" , f"CTS-{i+1}"))
            
            flow.start(
                with_initial_state=base_state ,  #this is where we load the base snapshot
                _force_run_dir=target_dir 
            )

            # F. Save Knobs 
            knob_file = os.path.join(target_dir, "knobs.json")
            with open(knob_file, "w") as f:
                json.dump(knobs, f, indent=4)
        
    except Exception as e:
        print(f"\nCRITICAL FAIL: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 5-run-cts.py <run_tag> <design_name> <clock_period> <clock_port>")
        sys.exit(1)
    FILENAME = sys.argv[1]
    design   = sys.argv[2]
    period   = float(sys.argv[3])
    port     = sys.argv[4]
    run_cts_from_placement(design, period, port)