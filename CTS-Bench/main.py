import os
import subprocess
import sys
import csv
import json

# --- CONFIGURATION ---
DATASET_ROOT = "dataset_root"
GRAPH_DIR = os.path.join(DATASET_ROOT, "graphs")
RAW_GRAPH_DIR = os.path.join(GRAPH_DIR, "raw_graphs")
CLUSTERED_GRAPH_DIR = os.path.join(GRAPH_DIR, "clustered_graphs")
LOG_FILE = os.path.join(DATASET_ROOT, "experiment_log.csv")

VENV_PYTHON = "./venv/bin/python3" 

# Number of runs
NUM_ITERATIONS = 1

def setup_directories():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(CLUSTERED_GRAPH_DIR, exist_ok=True)
    os.makedirs(RAW_GRAPH_DIR, exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                # --- 1. ID & Metadata ---
                "run_id", "placement_id", "design_name",
                
                # --- 2. Placement Inputs ---
                "aspect_ratio", "core_util", "density", "synth_strategy",
                "io_mode", "time_driven", "routability_driven",
                
                # --- 3. CTS Knobs (The "Treatment") ---
                "cts_max_wire", "cts_buf_dist", "cts_cluster_size", "cts_cluster_dia",
                
                # --- 4. Quality of Result (The "Outcome") ---
                # 4a. Timing
                "skew_setup", "skew_hold", 
                "setup_slack", "hold_slack",
                "setup_tns", "hold_tns",
                "setup_vio_count", "hold_vio_count",
                
                # 4b. Power & Area
                "power_total", "wirelength", "utilization",
                
                # 4c. CTS Structure
                "clock_buffers", "clock_inverters", "timing_repair_buffers",

                # --- 5. Graph Files ---
                "raw_graph_path", "cluster_graph_path"
            ]
            writer.writerow(header)

def get_next_run_id():
    if not os.path.exists(LOG_FILE):
        return 1
    
    max_id = 0
    try:
        with open(LOG_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if row and row[0].isdigit():
                    current_id = int(row[0])
                    if current_id > max_id:
                        max_id = current_id
    except Exception as e:
        print(f"Error reading CSV for ID: {e}")
        return 1
        
    return max_id + 1

def get_latest_run_tag():
    try:
        with open("latest_run.txt", "r") as f:
            return f.readline().strip()
    except FileNotFoundError:
        return None
    
def log_data_to_csv(run_id, placement_id, raw_graph_path, cluster_graph_path):
    # 1. Load Placement Stats
    pl_stats = {}
    try:
        with open("latest_stats.json", "r") as f:
            pl_stats = json.load(f)
    except FileNotFoundError:
        print("latest_stats.json not found.")
        return 

    # 2. Load Parsed Metrics from dataset.json
    metric_map = {}
    dataset_path = os.path.join("runs", placement_id, "dataset.json")
    
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r") as f:
                data_list = json.load(f)
                for entry in data_list:
                    metric_map[entry['id']] = entry['metrics']
        except Exception as e:
            print(f"Error reading dataset.json: {e}")
            return
    else:
        print(f"Warning: dataset.json not found in {placement_id}")

    existing_ids = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].isdigit():
                    existing_ids.add(int(row[0]))

    base_cts_dir = os.path.join("runs", placement_id, "CTS-experiments")
    current_db_id = run_id

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        
        for i in range(1, 11): 
            cts_id = f"CTS-{i}"
            cts_folder = os.path.join(base_cts_dir, cts_id)
            knob_file = os.path.join(cts_folder, "knobs.json")

            if current_db_id in existing_ids:
                print(f"⚠️ Skipping Run ID {current_db_id} (Already exists in CSV)")
                current_db_id += 1
                continue
            
            # Load Knobs
            cts_knobs = {}
            if os.path.exists(knob_file):
                with open(knob_file, "r") as kf:
                    cts_knobs = json.load(kf)

            # Load Metrics (from the map created in step 2)
            m = metric_map.get(cts_id, {})

            row = [
                # 1. ID & Metadata
                current_db_id,
                placement_id,
                pl_stats.get("design_name", ""),
                
                # 2. Placement Inputs
                pl_stats.get("aspect_ratio", ""),
                pl_stats.get("core_util", ""),
                pl_stats.get("density", ""),
                pl_stats.get("synth_strategy", ""),
                pl_stats.get("io_mode", ""),
                pl_stats.get("time_driven", ""),
                pl_stats.get("routability_driven", ""),
                
                # 3. CTS Knobs
                cts_knobs.get("CTS_CLK_MAX_WIRE_LENGTH", ""),
                cts_knobs.get("CTS_DISTANCE_BETWEEN_BUFFERS", ""),
                cts_knobs.get("CTS_SINK_CLUSTERING_SIZE", ""),
                cts_knobs.get("CTS_SINK_CLUSTERING_MAX_DIAMETER", ""),
                
                # 4. CTS Metrics
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
                
                # 5. Graphs
                raw_graph_path,
                cluster_graph_path
            ]
            
            writer.writerow(row)
            current_db_id += 1

    print(f"Logged {current_db_id - run_id} entries for {placement_id}")

def run_pipeline():
    setup_directories()
    
    if not os.path.exists(VENV_PYTHON):
        print(f"Error. Create a VENV at {VENV_PYTHON}")
        return

    for i in range(NUM_ITERATIONS):
        print(f"Iteration: {i+1}")

        print("1. Generating Placement...")
        subprocess.run(["python3", "scripts/1-gen-placement.py"], check=True)
        placement_id = get_latest_run_tag()
        if not placement_id:
            print("Could not get placement ID.")
            break

        print("2. Generating SAIF...")
        subprocess.run(["python3", "scripts/2-gen-saif.py", placement_id], check=True)

        print("3. Building Graphs...")
        raw_path = os.path.join(RAW_GRAPH_DIR, f"{placement_id}_raw.pt")
        subprocess.run([VENV_PYTHON, "scripts/three_def_dict_to_raw_graph.py", placement_id], check=True)

        cluster_path = os.path.join(CLUSTERED_GRAPH_DIR, f"{placement_id}_clustered.pt")
        subprocess.run([VENV_PYTHON, "scripts/4-graph_to_cluster.py", placement_id], check=True)

        if not os.path.exists(raw_path) or not os.path.exists(cluster_path):
            print("Graph files not found.")
            break

        print("4. Running CTS Swarm...")
        subprocess.run(["python3", "scripts/5-run-cts.py", placement_id], check=True)

        print("5. Parsing Reports...")
        subprocess.run(["python3", "scripts/6-parse-cts-reports.py", placement_id], check=True)

        print("6. Logging to Database...")
        next_id = get_next_run_id()
        log_data_to_csv(next_id, placement_id, raw_path, cluster_path)

        print("Run Complete.")

if __name__ == "__main__":
    run_pipeline()