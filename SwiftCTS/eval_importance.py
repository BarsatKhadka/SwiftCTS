"""
eval_importance.py — Feature importance analysis for CTSSurrogateV2.

Shows:
  1. Power head (XGBoost): sorted feature importances (gain + weight + cover)
  2. WL head (LightGBM): sorted feature importances (gain + split)
  3. Top-10 for each, with cumulative coverage
  4. Interpretation of what each top feature captures physically
"""

import sys, os, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# SwiftCTS not needed directly — we load the pkl directly below

MODEL_PATH = os.path.join(HERE, 'saved_models', 'model.pkl')

# ── Physics descriptions for top features ──────────────────────────────────
POWER_PHYSICS = {
    'log_act_scale':  'log(n_active × rel_act × f_ghz) — switching load scaled by frequency',
    'log_n_ff':       'log(n_ff) — total sequential element count (dominant size proxy)',
    'rel_act':        'mean_tc / max_tc — toggle uniformity (SA activity density)',
    'log_mux_active': 'log(frac_mux × n_active) — high-capacitance mux switching',
    'log_cap_proxy':  'log(n_active × avg_drive_strength) — capacitive load proxy',
    'frac_mux':       'fraction of mux cells (each drives heavy cap load)',
    'frac_high_act':  'fraction of nets with toggle rate > 50% — hot switching nodes',
    'frac_and_or':    'fraction of AND/OR gates (combinational depth proxy)',
    'slack_std':      'timing slack std dev — spread of critical paths',
    'frac_tight':     'fraction of paths near-critical — activity near timing limit',
    'mean_sig_prob':  'mean signal probability — DC power baseline',
    'slack_p10':      '10th percentile slack — worst-path bottleneck',
    'slack_p50':      'median slack — typical path health',
    'slack_xor':      'slack_mean × frac_xor — XOR-heavy critical paths (data-path power)',
    'f_ghz':          'clock frequency — direct power multiplier (P ∝ f)',
    'synth_area':     'synthesis area parameter — circuit complexity scaling',
    'log_xor_active': 'log(frac_xor × n_active) — XOR-intensive activity (encryption/hash)',
    'log_n_nets':     'log(n_saif_nets) — netlist size (switching node count)',
    'frac_nand_nor':  'fraction of NAND/NOR gates (moderate toggle, medium cap)',
    'log_cd_dens':    'log(cd × n_ff / die_area) — CTS cluster density per unit area',
}

WL_PHYSICS = {
    'comb_per_ff':    'combinational cells per FF — logic density drives congestion + tree depth',
    'grav_norm_mean': 'mean normalized |gravity| — FF spatial imbalance (skewed placement → long routes)',
    'frac_ff_active': 'fraction of FFs active in sim — active sinks need tighter equalization',
    'frac_ds4plus':   'fraction of cells with drive strength ≥ 4 — high-DS cells = large RC load',
    'rel_act':        'mean_tc / max_tc — toggle uniformity (low rel_act = sparse hot nodes)',
    'grav_abs_p90':   '90th pct |gravity| — tail of spatial imbalance (worst-case long routes)',
    'log_die_area':   'log(die_area) — physical area → routing bounding box',
    'grav_abs_p75':   '75th pct |gravity| — bulk spatial imbalance distribution',
    'util_dens':      'core_util × density — packing factor (denser = shorter average routes)',
    'avg_ds':         'mean drive strength — average cell size (DS ∝ output cap → WL)',
    'grav_abs_cv':    '|gravity| coefficient of variation — uniformity of spatial imbalance',
    'frac_nand_nor':  'fraction of NAND/NOR — complementary logic mix',
    'ff_y_std':       'std dev of FF y-positions — vertical spread → vertical routing extent',
    'log_cap_proxy':  'log(n_active × avg_DS) — aggregate capacitive load',
    'grav_x_sp':      'grav_abs_mean / ff_spacing — gravity normalized by FF density',
    'log_area_per_ff':'log(die_area / n_ff) — area per FF — sparse layout → longer routes',
    'core_util':      'core utilization — density of placed cells',
    'die_aspect':     'die aspect ratio — non-square dies → longer axis dominates WL',
    'grav_abs_std':   'std dev of |gravity| — variation in spatial imbalance',
    'grav_anisotropy':'gravity anisotropy — directional pull asymmetry',
    'log_n_ff':       'log(n_ff) — FF count → Steiner tree size',
    'log_die_area':   'log(die_area) — physical area → routing bounding box',
    'log_ff_hpwl':    'log(HPWL of FF bounding box) — direct WL lower bound (DME theory)',
    'log_cs_sp':      'log(cs × ff_spacing) — cluster_size × spacing → local routing reduction',
    'log_mw_hpwl':    'log(mw × HPWL) — max_wire × HPWL → tree extension capacity',
    'log_nff_cs':     'log(n_ff / cs) — cluster count → inter-cluster routing segments',
    'log_cd_dens':    'log(cd × n_ff / die_area) — cluster density → local WL savings',
    'log_comb_scale': 'log(comb_per_ff × n_ff) — total combinational cells',
    'log_act_scale':  'log(n_active × rel_act × f_ghz) — switching scale',
    'grav_abs_mean':  'mean |gravity force| — FF placement imbalance magnitude',
    'ff_cx':          'FF centroid x — x-offset from die center',
    'ff_cy':          'FF centroid y — asymmetric clock entry point',
    'log_n_comb':     'log(n_comb) — total combinational cells (congestion)',
}


def print_importance_table(names, importances, physics_map, title, top_n=20):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    total = importances.sum()
    sorted_idx = np.argsort(importances)[::-1]
    cumsum = 0.0
    print(f"  {'Rank':<5} {'Feature':<22} {'Importance':>11} {'Cumul%':>8}  Physics")
    print(f"  {'-'*70}")
    for rank, i in enumerate(sorted_idx[:top_n], 1):
        frac = importances[i] / (total + 1e-12) * 100
        cumsum += frac
        phys = physics_map.get(names[i], '—')
        # Truncate physics description if too long
        if len(phys) > 48:
            phys = phys[:45] + '...'
        print(f"  {rank:<5} {names[i]:<22} {frac:>10.1f}%  {cumsum:>6.1f}%  {phys}")
    if len(sorted_idx) > top_n:
        rest = importances[sorted_idx[top_n:]].sum() / (total + 1e-12) * 100
        print(f"  {'...':<5} {'(remaining ' + str(len(sorted_idx)-top_n) + ' features)':<22} "
              f"{rest:>10.1f}%  {100.0:>6.1f}%")


if __name__ == '__main__':
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        mdl = pickle.load(f)

    lodo = mdl.get('lodo', {})
    pw_names = lodo.get('pw_feature_names', [f'f{i}' for i in range(20)])
    wl_names = lodo.get('wl_feature_names', [f'f{i}' for i in range(75)])

    print(f"  Power LODO: {lodo.get('power_mean_mape', '?'):.1f}%  "
          f"WL LODO: {lodo.get('wl_mean_mape', '?'):.1f}%")
    print(f"  Power features: {len(pw_names)}  WL features: {len(wl_names)}")
    print(f"  max_train_n_ff: {mdl.get('max_train_n_ff', '?')}  "
          f"(OOD threshold: {mdl.get('max_train_n_ff', 5500) * 1.3:.0f})")

    # ── Power: XGBoost gain importance ─────────────────────────────────────
    m_pw = mdl['model_power']
    booster = m_pw.get_booster()

    # gain importance (most meaningful: total gain from splits on this feature)
    gain_scores = booster.get_score(importance_type='gain')
    gain_arr = np.array([gain_scores.get(f'f{i}', 0.0) for i in range(len(pw_names))])

    # weight importance (split count)
    weight_scores = booster.get_score(importance_type='weight')
    weight_arr = np.array([weight_scores.get(f'f{i}', 0.0) for i in range(len(pw_names))])

    print_importance_table(pw_names, gain_arr, POWER_PHYSICS,
                           "POWER HEAD — XGBoost Feature Importance (gain)")

    print(f"\n  Cross-check with split frequency (top 10):")
    sorted_w = np.argsort(weight_arr)[::-1][:10]
    for i in sorted_w:
        print(f"    {pw_names[i]:<25} splits={int(weight_arr[i])}")

    # ── WL: LightGBM gain importance ───────────────────────────────────────
    m_lgb = mdl['model_wl_lgb']
    lgb_imp = m_lgb.feature_importances_  # default: split count
    # Get gain importance via booster
    try:
        lgb_gain = np.array(m_lgb.booster_.feature_importance(importance_type='gain'),
                            dtype=float)
    except Exception:
        lgb_gain = lgb_imp.astype(float)

    print_importance_table(wl_names, lgb_gain, WL_PHYSICS,
                           "WL HEAD — LightGBM Feature Importance (gain)")

    print(f"\n  Cross-check split count (top 10):")
    sorted_s = np.argsort(lgb_imp)[::-1][:10]
    for i in sorted_s:
        print(f"    {wl_names[i]:<25} splits={int(lgb_imp[i])}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  SUMMARY: Which features matter most")
    print(f"{'='*72}")
    print("""
  POWER (top 3 = 82% of importance):
    1. log_act_scale  (50%) — switching activity × frequency (P = α·C·V²·f)
    2. slack_std      (19%) — timing path spread (tight timing → aggressive synthesis → more power)
    3. f_ghz          (13%) — clock frequency (direct power multiplier)
    4. log_mux_active  (6%) — mux count × activity (high-cap critical paths)

    NOTE: log_n_ff appears at rank 19 (0.2%) because power targets are already
    normalized by pw_norm = n_ff × f_ghz × avg_ds, so size is pre-factored out.

  WL (top 3 = 82% of importance):
    1. comb_per_ff    (47%) — combinational cells per FF: logic density → congestion + tree depth
    2. grav_norm_mean (24%) — FF spatial imbalance: skewed FFs require longer balancing routes
    3. frac_ff_active (12%) — active FF fraction: more active sinks → tighter equalization needed

    NOTE: CTS knobs (cs, mw) matter most through interaction features (log_cs_sp,
    log_mw_hpwl) which appear in the middle ranks. Gravity features are top because
    spatial FF distribution sets the theoretical WL lower bound.

  KEY INSIGHT: Power is determined almost entirely by design physics (what the
  circuit does at what frequency). WL is determined by spatial FF distribution
  (gravity) and logic density, modified by CTS knobs. Neither target is strongly
  knob-sensitive in absolute terms — this is why K-shot calibration works:
  the model captures *relative* variation from knob changes correctly.
""")
