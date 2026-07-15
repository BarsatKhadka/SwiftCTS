#!/usr/bin/env bash
# Run the CTS-Bench data generation pipeline on macOS.
# Sets up nix-provided iverilog/vvp and launches main-def-saif.py.
#
# Usage:
#   ./run_pipeline.sh                    # run all designs in DESIGN_CONFIG
#   DESIGN=i2c ./run_pipeline.sh         # run a single design (set env var)
#   NUM_ITERATIONS=10 ./run_pipeline.sh  # override iteration count

set -e
cd "$(dirname "$0")"

# Bring iverilog/vvp into PATH via nix
export PATH="$(nix shell nixpkgs/nixos-24.05#verilog --command sh -c 'echo $PATH' 2>/dev/null | head -1):$PATH"

# Confirm tools
echo "iverilog: $(which iverilog)"
echo "python3:  $(which python3)"
python3 -c "import openlane; print('openlane', openlane.__version__)"

# Optional single-design override (patch main-def-saif.py isn't needed;
# just set DESIGN env var and we filter in the loop below)
export DESIGN="${DESIGN:-}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-122}"

exec python3 main-def-saif.py
