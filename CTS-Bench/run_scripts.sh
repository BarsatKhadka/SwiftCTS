#!/bin/bash
set -e  # Stop immediately if any script fails

echo "========================================"
echo "Starting Chip-to-Graph Pipeline"
echo "========================================"

# 1. Run Placement (Generates the run folder)
echo "Step 1: Generating Placement..."
python3 scripts/1-gen-placement.py

# 2. Read the Run Tag 
if [ ! -f "latest_run.txt" ]; then
    echo "Error: latest_run.txt was not found. Did script 1 fail?"
    exit 1
fi
RUN_TAG=$(cat latest_run.txt)
echo "Detected New Run: $RUN_TAG"

# 3. Run SAIF Generation (Passing the tag)
echo "Step 2: Running Simulation & SAIF Gen..."
python3 scripts/2-gen-saif.py "$RUN_TAG"

# 4. Run Graph Extraction (Inside Venv)
echo "Step 3: Extracting Graph..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    python3 scripts/three_def_dict_to_raw_graph.py "$RUN_TAG"
    python3 scripts/4-graph_to_cluster.py "$RUN_TAG"
    deactivate
else
    echo "Warning: Venv not found"
    python3 scripts/three_def_dict_to_raw_graph.py "$RUN_TAG"
fi

python3 scripts/5-run-cts.py "$RUN_TAG"
python3 scripts/6-parse-cts-reports.py "$RUN_TAG"

echo "========================================"
echo " Done! Graph saved: $RUN_TAG.pt"
echo "========================================"