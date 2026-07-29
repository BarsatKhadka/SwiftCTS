"""
Parse ORFS CTS sweep output into dataset.json matching the schema produced
by the sky130 6-parse-cts-reports.py.

Reads from:
  runs/<placement_id>/CTS-experiments/CTS-{1..10}/
    knobs.json           — CTS knobs
    metrics.json         — timing metrics written by cts_sweep.tcl
    cts_stats.rpt        — report_cts output (buffer / subnet counts)

Writes:
  runs/<placement_id>/dataset.json

Usage:
  python3 6-parse-cts-reports-orfs.py <placement_id>
"""

import json
import os
import re
import sys

if len(sys.argv) > 1:
    RUN_TAG = sys.argv[1]
else:
    sys.exit("Error: placement_id required")

CTS_BENCH_ROOT = os.environ.get(
    "CTS_BENCH_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)

run_dir     = os.path.join(CTS_BENCH_ROOT, "runs", RUN_TAG)
cts_exp_dir = os.path.join(run_dir, "CTS-experiments")

if not os.path.exists(cts_exp_dir):
    sys.exit(f"CTS-experiments not found: {cts_exp_dir}")


def parse_cts_stats(rpt_path):
    """Extract buffer / subnet / sink counts from report_cts output."""
    counts = {"clock_buffers": "", "clock_subnets": "", "clock_sinks": ""}
    if not os.path.exists(rpt_path):
        return counts
    text = open(rpt_path).read()
    m = re.search(r"CTS inserted (\d+) buffers", text)
    if m:
        counts["clock_buffers"] = int(m.group(1))
    m = re.search(r"CTS created (\d+) clock subnets", text)
    if m:
        counts["clock_subnets"] = int(m.group(1))
    m = re.search(r"CTS found (\d+) sinks", text)
    if m:
        counts["clock_sinks"] = int(m.group(1))
    return counts


results = []
print(f"Parsing ORFS CTS metrics for: {RUN_TAG}")

for i in range(1, 11):
    cts_id  = f"CTS-{i}"
    cts_dir = os.path.join(cts_exp_dir, cts_id)

    knobs_path   = os.path.join(cts_dir, "knobs.json")
    metrics_path = os.path.join(cts_dir, "metrics.json")
    cts_rpt_path = os.path.join(cts_dir, "cts_stats.rpt")

    if not os.path.exists(metrics_path):
        print(f"  Skipping {cts_id}: metrics.json not found")
        continue

    knobs = {}
    if os.path.exists(knobs_path):
        with open(knobs_path) as f:
            knobs = json.load(f)

    with open(metrics_path) as f:
        raw = json.load(f)

    cts_counts = parse_cts_stats(cts_rpt_path)

    def maybe_float(v):
        try:
            return float(v) if v not in ("", None) else ""
        except (ValueError, TypeError):
            return ""

    # Prefer cts_stats.rpt buffer count (from report_cts) over TCL OpenDB count
    # because report_cts only counts CTS-inserted buffers; OpenDB counts all
    buf_count = cts_counts["clock_buffers"] if cts_counts["clock_buffers"] != "" \
                else maybe_float(raw.get("clock_buffers", ""))
    inv_count = maybe_float(raw.get("clock_inverters", ""))

    parsed = {
        # Timing (all values normalised to ns by the TCL script)
        "skew_setup":  maybe_float(raw.get("skew_setup",  "")),
        "skew_hold":   maybe_float(raw.get("skew_hold",   "")),
        "setup_slack": maybe_float(raw.get("setup_wns",   "")),
        "hold_slack":  maybe_float(raw.get("hold_wns",    "")),
        "setup_tns":   maybe_float(raw.get("setup_tns",   "")),
        "hold_tns":    maybe_float(raw.get("hold_tns",    "")),
        # ORFS CTS sweep doesn't compute violation count separately
        "setup_vio_count": "",
        "hold_vio_count":  "",
        # Structure
        "clock_buffers":          buf_count,
        "clock_inverters":        inv_count,
        "timing_repair_buffers":  "",
        # Power not available without SAIF simulation
        "power_total": "",
        # Estimated HPWL (µm) and area (µm²)
        "wirelength":  maybe_float(raw.get("wirelength",   "")),
        "utilization": maybe_float(raw.get("design_area",  "")),
    }

    results.append({"id": cts_id, "knobs": knobs, "metrics": parsed})
    print(f"  {cts_id}: setup_wns={parsed['setup_slack']}  skew={parsed['skew_setup']}"
          f"  buffers={parsed['clock_buffers']}")

output_path = os.path.join(run_dir, "dataset.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Wrote {len(results)} entries to {output_path}")
