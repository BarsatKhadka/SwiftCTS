"""
Merge all per-task CSV shards into a single experiment_log.csv.

Usage (after all SLURM tasks finish):
    python3 ~/CTS-Bench/hpc/slurm/merge_csvs.py

Reads:  $SHARDS_DIR/shard_*.csv
Writes: $LOG_FILE (experiment_log.csv)
        $LOG_FILE.stats.txt  (coverage summary)
"""
import csv
import glob
import os
import sys
from collections import Counter

SHARDS_DIR = os.environ.get("SHARDS_DIR", os.path.join(
    os.path.expanduser("~"), "dataset_with_def", "shards"))
LOG_FILE   = os.environ.get("LOG_FILE", os.path.join(
    os.path.dirname(SHARDS_DIR), "experiment_log.csv"))

HEADER = [
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

shards = sorted(glob.glob(os.path.join(SHARDS_DIR, "shard_*.csv")))
if not shards:
    print(f"No shards found in {SHARDS_DIR}")
    sys.exit(1)

print(f"Found {len(shards)} shards")

rows = []
design_counter = Counter()

for shard in shards:
    with open(shard, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row and row[0] != "run_id":  # skip any embedded headers
                rows.append(row)
                if len(row) > 2:
                    design_counter[row[2]] += 1

print(f"Total rows collected: {len(rows)}")

# Assign sequential run_ids
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    for run_id, row in enumerate(rows):
        row[0] = str(run_id)
        writer.writerow(row)

print(f"Written to: {LOG_FILE}")

# Stats
stats_path = LOG_FILE + ".stats.txt"
with open(stats_path, "w") as f:
    f.write(f"Total rows: {len(rows)}\n")
    f.write(f"Shards merged: {len(shards)}\n\n")
    f.write("Rows per design:\n")
    for design, count in sorted(design_counter.items(), key=lambda x: -x[1]):
        f.write(f"  {design:20s}: {count:6d}\n")

print(f"Stats: {stats_path}")
for design, count in sorted(design_counter.items(), key=lambda x: -x[1]):
    print(f"  {design:20s}: {count}")
