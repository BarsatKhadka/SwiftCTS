# SwiftCTS

**Fast Cross-Design Prediction and Pareto Optimization of Clock Tree Metrics via Few-Shot Calibration**

SwiftCTS is a physics-informed surrogate framework for Clock Tree Synthesis (CTS) that trains from scratch on a standard CPU in under 5 seconds, predicts power, wirelength, and timing skew at sub-millisecond latency, and adapts to entirely unseen circuit designs using only 1–5 physical reference runs.

---

## Repository Structure

```
SWIFT-CTS/
├── SwiftCTS/          # Surrogate model, evaluation scripts, Pareto optimizer
└── CTS-Bench/         # Dataset generation benchmark (OpenROAD + Sky130)
```

---

## Overview

CTS is a computationally expensive physical design stage. Every candidate configuration requires a full EDA tool run, making exhaustive design space exploration prohibitive. SwiftCTS replaces EDA evaluations with a surrogate model that:

- **Predicts** clock power (mW), wirelength (mm), and setup skew (ns) for any CTS knob configuration in < 1 ms
- **Generalizes** to unseen designs out-of-the-box via physics-grounded, scale-invariant features
- **Calibrates** to a new design using K=1 actual EDA run, reducing power MAPE from 24.5% to 3.3% and wirelength MAPE from 10.3% to 0.6% (LODO)
- **Optimizes** 100,000 CTS configurations via NSGA-II in ~9 seconds on CPU, returning Pareto-optimal knob settings validated against actual OpenROAD routing

---

## Key Results

**LODO cross-validation (train on 3 designs, test on 4th):**

| Unseen Design | Power (K=1) | WL (K=1) | Skew (K=1) |
|---------------|-------------|----------|------------|
| AES           | 2.2%        | 0.7%     | 0.153 ns   |
| ETHmac        | 5.0%        | 0.8%     | 0.120 ns   |
| PicoRV32      | 2.3%        | 0.4%     | 0.095 ns   |
| SHA-256       | 3.9%        | 0.4%     | 0.098 ns   |
| **Mean**      | **3.3%**    | **0.6%** | **0.116 ns** |

**OOD benchmarks (train on all 4 base designs, test on structurally distinct macros):**

| Design | Scale vs Training | Power (K=1) | WL (K=1) | Skew (K=1) |
|--------|-------------------|-------------|----------|------------|
| JPEG Encoder | 5× larger | 4.4% | 1.4% | 0.249 ns |
| ZipDiv Core  | 21× smaller | 4.7% | 0.7% | 0.003 ns |

---

## Installation

```bash
cd SwiftCTS
pip install -r requirements.txt
```

**Requirements:** numpy, pandas, scikit-learn, xgboost, lightgbm

For Pareto search with NSGA-II:
```bash
pip install pymoo
```

No GPU required.

---

## Quick Start

### Predict on a New Design

```python
from swiftcts import SwiftCTS

model = SwiftCTS.load('SwiftCTS/saved_models/model.pkl')

model.add_design(
    'mydesign_run_001',
    def_path    = 'placement_files/mydesign_run_001/mydesign.def',
    saif_path   = 'placement_files/mydesign_run_001/mydesign.saif',
    timing_path = 'placement_files/mydesign_run_001/timing_paths.csv',
    t_clk       = 7.0   # clock period in ns
)

# Zero-shot prediction
pred = model.predict('mydesign_run_001', cd=55, cs=20, mw=220, bd=100)
print(f"Power : {pred.power_mW:.2f} mW")
print(f"WL    : {pred.wl_mm:.2f} mm")
print(f"Skew  : {pred.skew_ns:.3f} ns")
```

### K-Shot Calibration

After running K=1 actual EDA configurations on the new placement:

```python
model.calibrate_power('mydesign_run_001',
                      true_pw=[0.182],      # W from OpenROAD
                      pred_pw=[0.155])      # W from surrogate

model.calibrate_wl('mydesign_run_001',
                   true_wl=[6.74e6],        # µm from OpenROAD
                   pred_wl=[5.20e6])        # µm from surrogate

# All subsequent predictions use the multiplicative calibration scalar:
#   k_cal = exp( mean( log(y_true / y_pred) ) )
pred = model.predict('mydesign_run_001', cd=55, cs=20, mw=220, bd=100)
```

### Pareto Optimization

```python
pareto = model.optimize('mydesign_run_001', n=100_000,
                        cd_range=(35, 70), cs_range=(12, 30),
                        mw_range=(130, 280), bd_range=(70, 150))
# Returns DataFrame of non-dominated CTS configurations
print(pareto[['cd', 'cs', 'mw', 'bd', 'power_mW', 'wl_mm', 'skew_ns']].head(10))
```

---

## Input File Format

Each placement directory must contain three files:

| File | Description |
|------|-------------|
| `{design}.def` | Physical placement in DEF format (FF positions, die area, cell types) |
| `{design}.saif` | Switching activity file (toggle counts, signal probabilities per net) |
| `timing_paths.csv` | Pre-CTS timing path report (launch/capture FF pairs, slack values) |

Expected directory structure:
```
placement_files/
  {design}_{run_id}/
    {design}.def
    {design}.saif
    timing_paths.csv
```

---

## CTS Knob Search Space

| Knob | Symbol | Range | Description |
|------|--------|-------|-------------|
| Cluster Diameter | `cd` | 35–70 µm | Maximum spatial diameter of a clock sink cluster |
| Max Wire Length | `mw` | 130–280 µm | Maximum wire length in the clock tree |
| Cluster Size | `cs` | 12–30 sinks | Maximum flip-flops per cluster |
| Buffer Distance | `bd` | 70–150 µm | Minimum buffer insertion spacing |

Discretizing these four knobs yields ~8.36 million unique configurations.

---

## Reproducing the Paper from Scratch

### Step 1: Generate Dataset (CTS-Bench)

See `CTS-Bench/` for dataset generation instructions. The training corpus contains 5,400 CTS runs (4 designs × ~135 placements × 10 knob configurations) generated with OpenROAD and Sky130 PDK.

### Step 2: Build Feature Caches

Parses all DEF/SAIF/timing files once. Takes 10–30 minutes on a 14-core CPU.

```bash
cd SwiftCTS
export SWIFTCTS_PLACEMENT_DIR=/path/to/placement_files

python build_caches.py           # DEF, SAIF, timing, gravity caches
python build_fast_path_cache.py  # hold head cache
python build_skew_cache.py       # skew head spatial cache
```

Outputs in `SwiftCTS/caches/`:
- `def_cache.pkl` — FF positions, cell counts, die geometry per placement
- `saif_cache.pkl` — switching activity features per placement
- `timing_cache.pkl` — slack distribution statistics per placement
- `gravity_cache.pkl` — logic-pull gravity vectors per placement
- `skew_spatial_cache.pkl` — critical-path spatial features for skew head
- `fast_path_cache.pkl` — timing path degree features for hold head

### Step 3: Train Power and Wirelength Heads

```bash
python build_models.py
```

Runs LODO cross-validation (aborts if mean power MAPE > 28%), then trains on all data and saves to `saved_models/model.pkl`. Takes under 5 seconds.

### Step 4: Train Skew and Hold Heads

```bash
python update_skew_models.py
```

### Step 5: Reproduce Paper Tables

```bash
# Table II: LODO + K-shot calibration (power + WL)
python eval_lodo_kshot.py

# Table II: Skew LODO
python eval_skew.py

# OOD evaluation: JPEG Encoder
python eval_jpeg.py

# OOD evaluation: ZipDiv Core
python eval_zipdiv.py

# OOD skew evaluation
python eval_skew_ood.py

# Table III ablation: architecture-level vs placement-level calibration
python eval_cp_lodo.py

# Pareto search comparison (Random / Sobol / NSGA-II)
python pareto_search.py
```

---

## Model Architecture

```
Input: DEF + SAIF + timing_paths.csv
         │
         ▼ parse once per placement
         │
    ┌────┴─────────────────────────────────────────┐
    │            Feature Engine (110 dims)          │
    └────┬──────────────┬──────────────┬────────────┘
         │              │              │
    Power (20)     WL (75)       Skew (15)
    XGBoost        LGB+Ridge*    LightGBM
         │              │              │
    log(P/norm)    log(WL/norm)  per-placement z-score
         │              │              │
    ×n_ff·f·ds    ×√(n_ff·A)    ×σ + μ
         ▼              ▼              ▼
    power_mW        wl_mm          skew_ns
```

**Power normalizer:** `n_ff × f_GHz × avg_drive_strength`
(from the dynamic power equation P = α·C·V²·f)

**WL normalizer:** `sqrt(n_ff × die_area)`
(from BHH theorem: minimum Steiner tree scales as √(N·A))

**Skew target:** per-placement z-score `(y_ns - μ) / σ`, where μ and σ are the mean and std of skew across all 10 knob runs for that specific placement. Requires K≥1 calibration to recover absolute nanoseconds.

**WL boundary safeguard:** if the Ridge component predicts a log-penalty outside the range observed during training, it is disabled for that sample and LightGBM is used alone. Prevents extrapolation on designs far outside the training distribution (e.g., JPEG Encoder, 5× larger than largest training design).

---

## K-Shot Calibration Details

The calibration scalar is computed as a geometric mean of log-ratios to avoid positive arithmetic-mean bias:

```
k_cal = exp( (1/K) · Σ log(y_true_i / y_pred_i) )
y_calibrated = k_cal × y_pred
```

This scalar corrects the constant multiplicative offset introduced by a new design's unknown absolute scale and switching characteristics, while the underlying model retains its learned sensitivity to CTS knob variations.

**Calibration is per-placement**, not per-design-family. Architecture-level calibration (one scalar broadcast across all placements of the same design) fails because clock tree metrics are highly governed by the physical floorplan — congested vs. dispersed layouts introduce substantial baseline errors even within the same logical netlist.

---

## Dataset

| Split | Designs | Placements | CTS Runs |
|-------|---------|-----------|---------|
| Training | AES, ETHmac, PicoRV32, SHA-256 | 540 | 5,400 |
| OOD Test | JPEG Encoder, ZipDiv Core | 12 | 120 |

Each placement was routed under 10 distinct CTS configurations sampled uniformly from the knob space. The OOD designs were deliberately chosen to stress-test scale extrapolation: ZipDiv is 21× smaller than the smallest training design; JPEG Encoder is 5× larger than the largest.

Dataset generated with [CTS-Bench](CTS-Bench/) using OpenROAD and Sky130 PDK.

---

## Feature Space Summary

| Head | Dims | Feature Groups |
|------|------|----------------|
| Power | 20 | Switching activity, gate composition, timing slack, clock frequency |
| Wirelength | 75 | Layout geometry, cell composition, switching activity, timing/knobs, knob×circuit interactions, gravity/topology, timing-path degree |
| Skew | 15 | CTS knobs, critical-path HPWL, spatial asymmetry, launch-capture geometry interactions |

All features are dimensionless ratios or log-compressed to ensure scale invariance across designs of vastly different sizes.

---

## Known Limitations

- **SHA-256 power (zero-shot):** 54.4% MAPE at K=0. SHA-256's SAIF simulation ran for 5284 seconds (vs. microseconds for AES/ETHmac), producing a relative activity factor 2× outside the training range. K=1 calibration corrects this to 3.9%.
- **Skew at K=0:** Returns relative z-scores only. Absolute nanosecond values require K≥1 to anchor the per-placement distribution.
- **Very large designs (n_ff > ~14,000):** WL switches to LightGBM-only mode (Ridge disabled). Zero-shot WL will carry a systematic offset; K≥1 corrects it.
- **Unsupported DEF attributes:** `dont-touch` nets, hold-only optimizations, and microscopic placement blockages are not currently represented in the feature space.
