# SwiftCTS — Full Technical Architecture

## Overview

SwiftCTS is a physics-informed machine learning surrogate for Clock Tree Synthesis (CTS) outcome prediction. It predicts four targets for unseen VLSI designs without running the full EDA flow:

| Target | Unit | Head | Algorithm | Features |
|--------|------|------|-----------|----------|
| `power_mW` | milliwatts | Power | XGBRegressor | 20 (pruned from 76) |
| `wl_mm` | millimetres | WL | LGBMRegressor + Ridge blend | 75 |
| `skew_ns` | nanoseconds (setup) | Skew Setup | LGBMRegressor | 15 (pruned from 63) |
| `skew_hold_ns` | nanoseconds (hold) | Skew Hold | LGBMRegressor | 18 (pruned from 63) |
| `hold_vio` | violation count | Hold Violations | LGBMRegressor | 66 |

---

## Input Pipeline

```
DEF file          → _parse_def()         → geometry features (FF positions, cell counts, die area)
SAIF file         → _parse_saif()        → activity features (toggle rates, signal probability)
timing_paths.csv  → _parse_timing()      → timing features (slack distribution)
DEF + timing      → _compute_gravity()   → gravity features (logic-pull spatial vectors)
DEF + timing      → compute_skew_feats() → skew spatial features (critical-path geometry)
```

All parsers are self-contained in `swiftcts.py`. No external EDA tools required at inference.

---

## Feature Vectors

### Power (20 features)

Selected by LODO-consistent XGBoost gain importance. Top-4 carry 87.6% of importance.

| # | Feature | Gain% | Physics |
|---|---------|-------|---------|
| 1 | `log_act_scale` | 50.1% | log(n_active × rel_act × f_ghz) — P = α·C·V²·f |
| 2 | `slack_std` | 19.3% | timing heterogeneity → current transient spread |
| 3 | `f_ghz` | 12.6% | clock frequency — scales all dynamic power |
| 4 | `log_mux_active` | 5.6% | mux switching load (clock gating + datapath) |
| 5–20 | minor features | 12.4% | activity fractions, slack percentiles, area, cell mix |

The 56 features dropped from the original 76 collectively carried only 3.8% of importance but introduced spurious correlations that degraded LODO generalization by 7.8 percentage points.

### WL (75 features)

Not pruned — all 75 features are retained. WL is a global aggregate (not worst-case like skew), so the full feature set generalizes without overfitting.

| # | Feature group | Dims | Content |
|---|--------------|------|---------|
| 1–12 | Circuit geometry | 12 | n_ff, die_area, ff_hpwl, ff_spacing, ff_cx/cy, ff_std, cell fractions |
| 13–21 | Drive strength | 9 | avg_ds, std_ds, p90_ds, frac_ds4plus, cap_proxy |
| 22–30 | Activity | 9 | rel_act, mean_sig_prob, tc_std_norm, frac_zero, frac_high_act, n_nets |
| 31–40 | Design params | 10 | f_ghz, t_clk, core_util, density, log(cd,cs,mw,bd), cd,cs,mw,bd |
| 41–48 | Interactions | 8 | xor_comb, act_xor, act_comb, log_cd_dens, log_cs_sp, log_mw_hpwl, log_nff_cs, util_dens |
| 49–51 | Activity scale | 3 | log_act_scale, log_xor_active, log_mux_active |
| 52–63 | Gravity (9) + interactions (3) | 12 | grav_abs_mean/std/p75/p90/cv/gini, grav_norm_mean/cv, grav_anisotropy, grav×cd/mw/sp |
| 64–70 | Timing-path degree | 7 | tp_degree_mean/std/cv/p90/gini, tp_frac_involved, tp_paths_per_ff, tp_frac_hub |
| 71–75 | Scale | 5 | log_area_per_ff, log_n_comb, comb_scale, log_n_comb, comb_per_ff×log_n_ff |

**Top-4 WL features** (85.8% of importance):

| Rank | Feature | Gain% | Physics |
|------|---------|-------|---------|
| 1 | `comb_per_ff` | 46.6% | combinational depth → tree routing complexity |
| 2 | `grav_norm_mean` | 23.7% | FF spatial imbalance → routing detour distance |
| 3 | `frac_ff_active` | 12.0% | fraction active FFs → effective clock network size |
| 4 | `frac_ds4plus` | 3.4% | high-drive cells → buffer insertion density |

### Skew (63-dim base, pruned to 15/18)

Built from 4 feature groups:

| Group | Dims | Content |
|-------|------|---------|
| `sk_ctx` (context) | 22 | Circuit geometry: n_ff, die_area, ff_hpwl, ff_spacing, die_aspect, ff_cx/cy/std, frac_xor, comb_per_ff, activity, timing stats |
| `kn` (knobs) | 8 | log(cd,cs,mw,bd) + raw cd,cs,mw,bd |
| `ck` (critical-path geometry) | 16 | crit_max_dist, crit_mean_dist, crit_p90_dist, crit_ff_hpwl, centroid offsets, spread, boundary fraction, star/chain topology, asymmetry, eccentricity, density ratio |
| `ski` (knob × critical interactions) | 17 | cd/ff_spacing, bd/crit_max, mw/crit_max, star×cd, asymm×mw, dens×cs, cmax×cd, log(cmax/cd), log(cmax/bd), log(cmax/mw), cx×cd, cy×mw, log(nff/cs)×chpwl |

**Top-5 skew_setup features** (knob-dominated — key for generalization):

| Rank | Feature | Physics |
|------|---------|---------|
| 1 | `log_mw` | Maximum wire budget — dominant tree reach constraint |
| 2 | `cd/ff_spacing` | Cluster diameter vs FF spacing — grouping feasibility |
| 3 | `log_cs` | Cluster size — FF aggregation |
| 4 | `log_cd` | Cluster diameter |
| 5 | `log_bd` | Buffer distance — equalization stages |

Skew is knob-dominated: top-5 are all CTS parameters, making pruned features highly generalizable.

---

## Model Heads

### Power Head

```
X_pw (20-dim) → StandardScaler → XGBRegressor
Output:  log(power / norm)
Denorm:  power = exp(pred) × norm
Norm:    norm = n_ff × f_ghz × avg_ds    [scales by active switching load]
```

XGBoost hyperparameters: `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0`

### WL Head (LGB + Ridge blend)

```
X_wl (75-dim) → StandardScaler → LGBMRegressor   (alpha=0.3)
                               → Ridge             (alpha=0.7)
Blend rule: per-sample
  If Ridge raw pred ∈ [wl_y_min - 1, wl_y_max + 1]:
      pred = 0.3 × LGB + 0.7 × Ridge
  Else (Ridge extrapolating):
      pred = LGB only
Output:  log(wl / norm)
Denorm:  wl = exp(pred) × norm
Norm:    norm = sqrt(n_ff × die_area)    [scales by √(circuit × area)]
```

**Why blend?** LGB overpredicts large designs in LODO (ETH MAC: +38.8% bias); Ridge underpredicts. The 0.7 Ridge weight partially cancels opposing biases → ETH MAC residual bias -7.2% (vs ±38%). The per-sample adaptive rule prevents Ridge extrapolation for OOD designs outside the training target range.

LGB hyperparameters: `n_estimators=500, num_leaves=63, learning_rate=0.03, min_child_samples=10`
Ridge: `alpha=10.0`

### Skew Head (Setup + Hold)

```
X_sk (63-dim) → StandardScaler → subset[15 features] → LGBMRegressor → z_score
Denorm:  skew_ns = z × sk_sig + sk_mu   (per-placement normalization)
```

Hold is the negation of setup in the current model: `hold_z = -setup_z`. This works because skew_hold ≈ -skew_setup (correlation r = -0.94 globally, -0.90 within-placement). A separate hold model showed identical LODO accuracy (0.0747 ns vs 0.0746 ns). The negation is simpler and has no overfitting risk.

LGB hyperparameters: `n_estimators=300, num_leaves=31, learning_rate=0.03, min_child_samples=10`

### Hold Violations Head

```
X_hv (66-dim) → StandardScaler → LGBMRegressor → log1p(hold_count)
Denorm:  hold_vio = expm1(pred)
```

---

## Normalization

### Targets

Per-placement z-score (applied before training, inverted after prediction):

```python
# For each placement's 10 CTS runs:
mu = mean(target_values)
sig = max(std(target_values), max(abs(mu) * 0.01, 1e-4))
z = (value - mu) / sig
```

This makes the task: "how does this outcome compare to the average for this specific placement?" Global z-scores were rejected because they conflate design-family effects with CTS-parameter effects.

For power and WL, MAPE is computed in raw (un-normalized) space by inverting the log normalization. Skew is reported in nanoseconds (absolute).

---

## K-Shot Calibration

Log-space multiplicative correction applied per-design after observing K labeled samples:

```python
log_ratios = [log(true_i / pred_i) for i in range(K)]
scale = exp(mean(log_ratios))
pred_calibrated = pred × scale
```

Clipping: power `[e^{-4.6}, e^{4.6}]` (0.01× to 100×), WL `[e^{-15}, e^{15}]` (wide for extreme OOD).

**When it helps**: systematic bias due to OOD feature values (e.g., gravity=0 for uncached WL → 240% error reduced to 5.4% at K=2).
**When it hurts**: power at K≥1 for JPEG (two placements in different routing regimes; calibration offset from placement 1 does not transfer to placement 2).

General guidance: power use K=0 if zero-shot < 20%; WL always use K≥1 for uncached designs.

---

## Build Pipeline

```
1. Build raw feature caches (one-time, ~15 min):
   python build_caches.py             → caches/{def,saif,timing,gravity,skew_spatial}_cache.pkl
   python build_fast_path_cache.py    → caches/fast_path_cache.pkl

2. Train power + WL heads (~30 sec):
   python build_models.py             → saved_models/model.pkl (power + WL only)

3. Add skew + hold heads (~2 min):
   python update_skew_models.py       → updates saved_models/model.pkl (adds skew + hold)

4. Verify:
   python eval_lodo.py                → LODO accuracy on 4 training designs
   python eval_zipdiv.py              → OOD K-shot table for zipdiv
   python eval_jpeg.py                → OOD K-shot table for oc_jpegencode
```

---

## Model File Format (`saved_models/model.pkl`)

Pickle dict with keys:

| Key | Type | Content |
|-----|------|---------|
| `model_power` | XGBRegressor | Trained power head |
| `scaler_power` | StandardScaler | Power feature normalizer |
| `model_wl_lgb` | LGBMRegressor | WL LGB component |
| `model_wl_ridge` | Ridge | WL Ridge component |
| `scaler_wl` | StandardScaler | WL feature normalizer |
| `wl_blend_alpha` | float | LGB blend weight (0.3) |
| `model_skew` | LGBMRegressor | Skew setup head |
| `scaler_skew` | StandardScaler | Skew setup normalizer |
| `sk_feat_idx` | int32 array | 15 selected feature indices |
| `model_skew_hold` | LGBMRegressor | Skew hold head |
| `scaler_skew_hold` | StandardScaler | Skew hold normalizer |
| `skh_feat_idx` | int32 array | 18 selected feature indices |
| `hold_uses_negation` | bool | If True: hold_z = -setup_z |
| `model_hold_vio` | LGBMRegressor | Hold violation head |
| `scaler_hold_vio` | StandardScaler | Hold violation normalizer |
| `lodo` | dict | LODO results + metadata |
| `pw_feat_names` | list[str] | Power feature names (20) |
| `wl_feat_names` | list[str] | WL feature names (75) |

---

## File Dependency Graph

```
swiftcts.py
├── build_skew_cache.py   (imported for parse_def_ff_positions, compute_skew_features)
├── caches/def_cache.pkl
├── caches/saif_cache.pkl
├── caches/timing_cache.pkl
├── caches/gravity_cache.pkl
├── caches/skew_spatial_cache.pkl
├── caches/fast_path_cache.pkl
├── saved_models/model.pkl
└── data/unified_manifest.csv (+ zipdiv.csv, oc_jpegencode.csv for OOD eval)

build_caches.py
└── dataset_with_def/placement_files/{pid}/{design}.def + .saif + timing_paths.csv

build_models.py
└── caches/{def,saif,timing,gravity}_cache.pkl  →  saved_models/model.pkl

update_skew_models.py
├── caches/{def,saif,timing,skew_spatial}_cache.pkl
├── eval_skew.py     (build_features, per_placement_z)
├── prune_skew_eval.py  (get_gain_ranking, lodo_subset, LGB_PARAMS)
└── saved_models/model.pkl  (updated in-place)
```

---

## Known Limitations

| Scenario | Affected Head | Root Cause | Mitigation |
|----------|--------------|-----------|-----------|
| SHA256 power (LODO) | Power | `rel_act=0.104` is 2× training max (0.052). `log_act_scale` extrapolates. | K≥1 calibration |
| AES WL (LODO) | WL | Sparse FF placement (2994 FFs, large die). Gravity features less discriminative. | Accepted floor |
| Uncached WL (K=0) | WL | `grav_norm_mean` (23.7% importance) defaults to 0. Systematic 2–3× underestimate. | Always use K≥1 |
| JPEG power K-shot | Power | Calibration placement WL ~6.74M µm vs test ~3.37M µm (2× routing mismatch). | Use K=0 for power |
| Skew K=0 OOD | Skew | Model outputs z-score. Absolute ns requires per-placement mean/sigma. | Use K≥1 |
