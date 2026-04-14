# Hold Skew: Architecture, Analysis, and Results

---

## 1. What is skew_hold?

`skew_hold` is reported by OpenSTA/OpenROAD CTS as the worst-case clock arrival imbalance
under the **early (hold) timing corner**: `min_early_arrival - max_early_arrival` across all FFs.

Always negative. `-0.85 ns` means the earliest-arriving clock is 0.85 ns ahead of the latest,
creating hold violation risk on fast paths.

`skew_setup` is the same clock tree under the **late (setup) corner**:
`max_late_arrival - min_late_arrival`. Always positive.

**They are not different physical phenomena — same clock tree, two timing corners.**

---

## 2. Data Analysis: skew_hold ≈ -skew_setup

| Metric | Value |
|--------|-------|
| Global Pearson correlation | **-0.936** |
| `setup + hold` mean | 0.009 ns |
| `setup + hold` std | 0.057 ns |
| `\|residual\| / setup magnitude` | **4.2%** |
| Within-placement z-score correlation | -0.90 ± 0.17 |

Value ranges per design (training set):

| Design | skew_setup | skew_hold |
|--------|-----------|-----------|
| aes | [0.536, 1.397] ns | [-1.447, -0.534] ns |
| ethmac | [0.530, 1.526] ns | [-1.497, -0.531] ns |
| picorv32 | [0.515, 1.223] ns | [-1.247, -0.515] ns |
| sha256 | [0.518, 1.305] ns | [-1.296, -0.522] ns |

---

## 3. Architecture Evolution

### v1 — Separate Model (original)

```
X_sk (63-dim) → top-18 features → StandardScaler → LGB → pred_hold_z
pred_hold_ns = pred_hold_z * sig_hold + mu_hold
```

Problem: re-learns the 95.8% shared component with setup using setup-critical path features
(`crit_*` features from slow paths) — wrong features for hold physics.

---

### v2 — Negation

```
pred_hold_z = -pred_setup_z                    (one line)
pred_hold_ns = pred_hold_z * sig_hold + mu_hold
```

Physical basis: same clock tree, opposite sign convention.
Empirically confirmed: `setup + hold = 0` with only 4.2% residual.

---

### v3 — Negation + Delta (attempted, rejected)

```
pred_hold_z = -pred_setup_z + delta_lgb(fast_path_features + knobs)
delta_target = z_hold + z_setup    (residual to model)
```

**Motivation**: The within-placement z-score correlation is -0.90 ± 0.17. Some placements
have correlation up to +0.525 — a real physical signal. Hold violations are driven by
**fast/short paths** (high slack) while setup is driven by slow/long paths (negative slack).
We have zero fast-path features in the current set. Building them and modeling the delta
should capture the divergence.

**Implementation**:
- `build_fast_path_cache.py`: extract spatial features from top-50 highest-slack timing paths
- 22 fast-path features: distances, HPWL, self-loop fraction, slack distribution, spatial spread
- 8 knob features + 8 interaction terms = **38-dim delta feature vector**
- Delta LGB: 200 trees, 15 leaves, lr=0.02, more regularization (target noise std=0.45)

**Result: rejected**. See Section 6 for analysis.

---

## 4. Fast-Path Features (build_fast_path_cache.py)

Extracted from `timing_paths.csv` using `nlargest(50, 'slack')` — the 50 fastest paths.
These are the hold-critical paths: fast combinational logic between adjacent FFs.

Key observations from training data:
- `frac_self_loop ≈ 0.54` for aes (54% of top-50 fast paths are self-loops, launch==capture)
- `fast_min_dist = 0.0` always (self-loops have zero spatial distance)
- `slack_max ≈ 6.0 ns` — huge slack, pure hold risk
- jpeg: 11,801 / 29,210 paths are self-loops (40%)

All 544 placements cached (training + zipdiv + jpeg).

---

## 5. Comprehensive Results

### 5.1 LODO (4-design Leave-One-Design-Out)

| Design | v1 Separate | v2 Negation | v3 Neg+Delta |
|--------|------------|------------|--------------|
| aes | 0.0845 ns ✓ | 0.0827 ns ✓ | 0.0835 ns ✓ |
| ethmac | 0.0813 ns ✓ | 0.0832 ns ✓ | 0.0838 ns ✓ |
| picorv32 | 0.0678 ns ✓ | 0.0680 ns ✓ | 0.0678 ns ✓ |
| sha256 | 0.0651 ns ✓ | 0.0647 ns ✓ | 0.0649 ns ✓ |
| **Mean** | **0.0747 ns ✓** | **0.0746 ns ✓** | **0.0750 ns ✓** |

All three pass. v2 (negation) and v1 (separate) are within 0.002 ns of each other.
v3 (neg+delta) is marginally worse than negation-only — delta model introduces overfitting.

### 5.2 OOD — zipdiv (1,642 FFs)

Calibration on placement 1, test on placement 2.

| K | v1 Separate | v2 Negation | v3 Neg+Delta |
|---|------------|------------|--------------|
| 0 | 1.1979 ns ✗ | **0.4916 ns ✗** | **0.4813 ns ✗** |
| 1 | **0.0372 ns ✓** | **0.0394 ns ✓** | 0.0443 ns ✓ |
| 2 | 0.0429 ns ✓ | **0.0405 ns ✓** | 0.0443 ns ✓ |
| 5 | **0.0222 ns ✓** | **0.0235 ns ✓** | 0.0298 ns ✓ |

K=0: both negation variants dramatically better than separate (0.49 vs 1.20 ns).
K≥1: all pass; v3 slightly worse than v2 at every K.

### 5.3 OOD — oc_jpegencode (14,606 FFs, 3× OOD)

| K | v1 Separate | v2 Negation | v3 Neg+Delta |
|---|------------|------------|--------------|
| 0 | 1.7723 ns ✗ | **1.0409 ns ✗** | 1.0458 ns ✗ |
| 1 | **0.0818 ns ✓** | 0.0873 ns ✓ | **0.0869 ns ✓** |
| 2 | **0.0864 ns ✓** | 0.0894 ns ✓ | 0.0921 ns ✓ |
| 5 | **0.0747 ns ✓** | 0.0887 ns ✓ | **0.0866 ns ✓** |

K=0: negation variants better (1.04 vs 1.77 ns).
K≥1: all pass; v3 ≈ v2 for jpeg; v1 pulls ahead at K=5.

---

## 6. Why the Delta Model Fails

### Feature correlation analysis

```
Max r (original 63 sk features):     0.038  (mw)
Max r (new fast-path features):       0.000  (!)
Max r (fast-path interactions):       0.023  (log(slack_max × mw))
```

Fast-path spatial features have **exactly zero linear correlation** with `delta_z`.

### Root cause: feature-target mismatch

`timing_paths.csv` is **pre-CTS placement timing** — the same file for all 10 CTS configs of a placement. Fast-path spatial features are **constant per placement**, computed once from DEF geometry.

But `delta_z = z_hold + z_setup` varies **per CTS config** (different knobs → different arrival times → different hold-setup divergence). A constant feature cannot explain within-placement variation.

The delta model's top features by gain are CTS knobs (`log_mw` 9.9%, `log_bd` 7.4%) — not fast-path features. These knob effects are already modelled by the setup LGB. Adding them again via the delta path introduces redundancy and overfitting.

### Delta model feature importance

| Feature | Gain | % |
|---------|------|---|
| log_mw | 277 | 9.9% |
| log_bd | 206 | 7.4% |
| fdens×cs | 204 | 7.3% |
| fsl×cs | 202 | 7.2% |
| fast_hpwl/(cs+1) | 169 | 6.0% |
| log_cd | 168 | 6.0% |
| fsl×log(nff/cs) | 152 | 5.4% |
| fast_cx_offset | 149 | 5.3% |

Top 8 features: 6 are knob-derived, 2 are spatial. The spatial features (`fast_cx_offset`,
`fast_hpwl/(cs+1)`) contribute because they encode design-level between-placement differences —
but these are absorbed by per-placement z-scoring, giving no within-placement signal.

### What would be needed for a working delta model

The delta would require **post-CTS timing paths** — fast paths from timing analysis run *after*
the clock tree is inserted. These would vary per CTS config and capture the actual hold-critical
paths created by the specific insertion delay profile. We have pre-CTS timing only.

---

## 7. Final Architecture: v2 (Negation)

```python
pred_hold_z  = -pred_setup_z
pred_hold_ns = pred_hold_z * sig_hold + mu_hold
```

### Why this is correct (not just simple)

- `delta_z` residual std = 0.45 z-units ≈ 0.057 ns
- None of the 38 fast-path + interaction features predict it (max r = 0.023)
- Adding features makes things marginally worse (overfitting noise)
- The negation prior is supported by physics: both targets measure the same insertion delay distribution under different timing corners
- The 4.2% divergence is real but comes from post-CTS OCV derating details not in any available feature

### pkl implementation

```
hold_uses_negation: True   ← flag set in model.pkl
```

`swiftcts.py` checks this flag:
```python
if self.hold_uses_negation:
    skh_z = -sk_z          # one negation, zero cost
```

Backward compatible: old pkl without flag falls back to `model_skew_hold`.

---

## 8. Complete Comparison Table

| Metric | v1 Separate | v2 Negation | v3 Neg+Delta | Winner |
|--------|------------|------------|--------------|--------|
| LODO mean | 0.0747 ns ✓ | 0.0746 ns ✓ | 0.0750 ns ✓ | v2 ≈ v1 |
| LODO max | 0.0845 ns ✓ | 0.0832 ns ✓ | 0.0838 ns ✓ | v2 |
| zipdiv K=0 | 1.198 ns ✗ | 0.492 ns ✗ | 0.481 ns ✗ | v3 |
| zipdiv K=1 | 0.037 ns ✓ | 0.039 ns ✓ | 0.044 ns ✓ | v1 |
| zipdiv K=5 | 0.022 ns ✓ | 0.024 ns ✓ | 0.030 ns ✓ | v1 |
| jpeg K=0 | 1.772 ns ✗ | 1.041 ns ✗ | 1.046 ns ✗ | v2 |
| jpeg K=1 | 0.082 ns ✓ | 0.087 ns ✓ | 0.087 ns ✓ | v1 |
| jpeg K=5 | 0.075 ns ✓ | 0.089 ns ✓ | 0.087 ns ✓ | v1 |
| Extra model params | 18-feat LGB | none | 38-feat LGB | v2 |
| K=0 calibration | blind prior | placement-aware | placement-aware | v2/v3 |

**Verdict**: v2 (negation) is best overall. It matches v1 on LODO, significantly beats v1 at K=0
OOD, and needs zero additional parameters. v3 provides no improvement over v2 — the fast-path
features are constant per placement and cannot model within-placement hold-setup divergence.
