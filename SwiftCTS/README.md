# SwiftCTS — CTS Outcome Surrogate

Predicts clock tree synthesis (CTS) outcomes for **unseen VLSI designs** without running the full EDA flow. Trained on 4 designs (aes, ethmac, picorv32, sha256); generalizes to new circuits via Leave-One-Design-Out (LODO) evaluation.

## Results

| Evaluation | Power MAPE | WL MAPE |
|---|---|---|
| LODO mean (4 designs) | **24.2%** | **11.0%** |
| zipdiv OOD, K=2 | **4.1%** | **5.4%** |
| jpeg OOD, K=0/5 | **15.7%** | **18.2%** |
| vs old baseline | −7.8% | 0.0% |

---

## Quick Start

```python
from swiftcts import SwiftCTS

# Load the pre-trained model
model = SwiftCTS.load('saved_models/model.pkl')

# Register a new design (parses DEF + SAIF + timing paths on the fly)
model.add_design(
    'mydesign_run_001',
    def_path    = 'placement_files/mydesign_run_001/mydesign.def',
    saif_path   = 'placement_files/mydesign_run_001/mydesign.saif',
    timing_path = 'placement_files/mydesign_run_001/timing_paths.csv',
    t_clk       = 7.0          # clock period in nanoseconds
)

# Predict for a single knob configuration
pred = model.predict('mydesign_run_001', cd=55, cs=20, mw=220, bd=100)
print(f"Power : {pred.power_mW:.2f} mW")
print(f"WL    : {pred.wl_mm:.2f} mm")
print(f"Skew  : {pred.skew_ns:.3f} ns")     # requires sk_mu / sk_sig
print(f"Hold  : {pred.hold_vio:.0f} violations")

# K-shot calibration — collect K labeled runs from the SAME placement, then calibrate
true_pw  = [0.182, 0.191, 0.179]            # W (from OpenROAD)
true_wl  = [6.74e6, 6.71e6, 6.78e6]        # µm
pred_pw  = [model.predict('mydesign_run_001', cd=c, cs=s, mw=m, bd=b).power_mW / 1000
            for c,s,m,b in [(55,20,220,100),(50,16,200,110),(60,22,240,90)]]
pred_wl  = [model.predict('mydesign_run_001', cd=c, cs=s, mw=m, bd=b).wl_mm * 1000
            for c,s,m,b in [(55,20,220,100),(50,16,200,110),(60,22,240,90)]]
model.calibrate_power('mydesign_run_001', true_pw, pred_pw)
model.calibrate_wl('mydesign_run_001', true_wl, pred_wl)

# Pareto knob search: returns non-dominated configs sorted by power_mW
pareto = model.optimize('mydesign_run_001', n=5000,
                        cd_range=(35, 70), cs_range=(12, 30),
                        mw_range=(130, 280), bd_range=(70, 150))
print(pareto[['cd','cs','mw','bd','power_mW','wl_mm']].head(10))
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: numpy, pandas, scikit-learn, xgboost, lightgbm

---

## CTS Knobs

| Knob | Parameter | Meaning |
|------|-----------|---------|
| `cd` | `cts_cluster_dia` | Maximum diameter of a clock sink cluster (µm) |
| `cs` | `cts_cluster_size` | Maximum number of sinks per cluster |
| `mw` | `cts_max_wire` | Maximum wire length in the clock tree (µm) |
| `bd` | `cts_buf_dist` | Minimum buffer insertion spacing (µm) |

---

## Reproducing the Model from Scratch

```bash
pip install -r requirements.txt

# Step 1 — Build all feature caches from placement files (~15 min)
python build_caches.py           # def/saif/timing/gravity/skew_spatial caches
python build_fast_path_cache.py  # fast-path cache (hold head)

# Step 2 — Train power + WL heads (~30 sec)
python build_models.py           # saves saved_models/model.pkl

# Step 3 — Add pruned skew + hold heads (~2 min)
python update_skew_models.py     # updates saved_models/model.pkl in-place

# Step 4 — Verify
python eval_lodo.py
python eval_zipdiv.py
python eval_jpeg.py
python eval_skew.py
```

---

## File Structure

```
swiftCTS/
├── swiftcts.py              # Main class: SwiftCTS (parsers + feature engine + all heads)
├── build_caches.py          # Build all caches from raw placement files (run first)
├── build_fast_path_cache.py # Build fast-path cache (hold head input)
├── build_skew_cache.py      # Skew spatial feature builder (imported by swiftcts.py)
├── build_models.py          # Train power + WL heads → saved_models/model.pkl
├── update_skew_models.py    # Add pruned skew + hold heads to existing pkl
│
├── eval_lodo.py             # Leave-One-Design-Out accuracy (4 training designs)
├── eval_ood.py              # K-shot OOD evaluation (zipdiv + jpeg)
├── eval_zipdiv.py           # Standalone zipdiv K-shot table
├── eval_jpeg.py             # Standalone jpeg K-shot table
├── eval_skew.py             # Skew LODO evaluation
├── eval_skew_ood.py         # Skew OOD evaluation (zipdiv + jpeg)
├── eval_hold_negation.py    # Hold negation vs separate-model comparison
├── eval_importance.py       # Feature importance analysis
├── plot_importance.py       # Feature importance plots
│
├── prune_skew_eval.py       # Skew feature pruning sweep
├── prune_wl_eval.py         # WL feature pruning analysis
├── wl_blend_ablation.py     # WL Ridge blend alpha sweep
├── train_hold_delta.py      # Hold delta model experiments
├── update_skew_models.py    # Retrains skew heads and updates pkl
├── requirements.txt
│
├── saved_models/
│   └── .gitkeep             # Populated by build_models.py + update_skew_models.py
│
├── caches/
│   └── .gitkeep             # Populated by build_caches.py + build_fast_path_cache.py
│
├── data/
│   ├── unified_manifest.csv      # Training CSV (1400 rows)
│   ├── zipdiv.csv                # OOD eval CSV (20 rows)
│   └── oc_jpegencode.csv         # OOD eval CSV (20 rows)
│
├── dataset_with_def -> ../dataset_with_def   # symlink to placement files
├── sky130_fd_sc_hd.lef                       # Standard cell library (LEF)
├── sky130_fd_sc_hd_tt_025C_1v80.lib          # Liberty timing library
│
├── figures/
│   ├── feature_importance_all.png
│   ├── feature_importance_power.png
│   ├── feature_importance_skew.png
│   └── feature_importance_wl.png
│
├── README.md                # This file
├── ARCHITECTURE.md          # Full technical reference
├── PRUNING.md               # Feature selection analysis (power + skew)
├── ALL_RESULTS.md           # Complete results reference
├── hold.md                  # Hold time analysis
└── wl_blend_ablation_results.txt  # WL blend alpha sweep raw results
```

---

## Running Evaluations

```bash
# LODO accuracy on the 4 training designs
python eval_lodo.py

# Skew LODO (setup + hold)
python eval_skew.py

# K-shot OOD evaluation
python eval_zipdiv.py
python eval_jpeg.py
python eval_skew_ood.py

# Feature importance (power + WL heads)
python eval_importance.py

# Rebuild model from scratch (takes ~30 sec, requires caches/)
python build_models.py
```

---

## K-Shot Calibration Guide

| Design type | Recommended K | Power | WL |
|---|---|---|---|
| In-distribution (n_ff ≤ 13,710) | K=2 | ~4–6% | ~5–9% |
| Extreme OOD power | K=0 | best zero-shot | — |
| Extreme OOD WL | K=5 | — | ~18–20% |

K-shot calibration uses **log-space multiplicative scaling**:
```
scale = exp( mean( log(true / pred) ) )   for K labeled observations
```
Applied as `pred_calibrated = pred × scale`. This is robust to predictions being off by orders of magnitude (unlike linear clipping).

---

## Known Limitations

- **SHA256 power** (LODO): 51.5% — SHA256's SAIF simulation ran 5284 seconds (AES/ETH ran microseconds), making `rel_act = 0.104`, which is 2× outside the training maximum of 0.052. K-shot calibration fixes it for actual OOD use.
- **AES WL** (LODO): 24.9% — AES has low FF density (2994 FFs in a large die), making gravity features less discriminative.
- **Large designs (n_ff > 13,710)**: WL uses LGB-only mode (no Ridge blend). Zero-shot WL will be systematically off; K≥1 corrects it.
- **Gravity features**: Computed on-the-fly for new designs via `add_design`. This takes 5–30 seconds depending on FF count. All other features parse instantly.

---

## Architecture Summary

```
Input: DEF + SAIF + timing_paths.csv  →  parsers  →  4 feature vectors

Power  (20 dim) → XGBRegressor → exp(pred) × n_ff·f_ghz·avg_ds       → power_mW
WL     (75 dim) → LGB + Ridge* → exp(pred) × sqrt(n_ff·die_area)      → wl_mm
Skew   (63 dim) → LGBRegressor → z_score × sk_sig + sk_mu             → skew_ns
Hold   (66 dim) → LGBRegressor → expm1(pred)                          → hold_vio

* Ridge blend (0.7 LGB + 0.3 Ridge) for n_ff ≤ 13,710
  LGB-only for n_ff > 13,710 (prevents Ridge extrapolation on large OOD designs)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical reference.
