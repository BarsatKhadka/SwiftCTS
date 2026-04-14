"""
prune_skew_eval.py — Feature pruning sweep for skew_setup and skew_hold heads.

Ranks 63 features by LGB gain importance, then sweeps top-N subsets.
Checks if a smaller feature set matches or beats the full 63-feature LODO MAE.
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_skew import build_features, per_placement_z

DEF_CACHE   = os.path.join(HERE, 'caches', 'def_cache.pkl')
SAIF_CACHE  = os.path.join(HERE, 'caches', 'saif_cache.pkl')
TIMING_CACHE= os.path.join(HERE, 'caches', 'timing_cache.pkl')
SKEW_CACHE  = os.path.join(HERE, 'caches', 'skew_spatial_cache.pkl')
MANIFEST    = os.path.join(HERE, 'data', 'unified_manifest.csv')

LGB_PARAMS = dict(n_estimators=300, num_leaves=31, learning_rate=0.03,
                  min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)

SKEW_FEAT_NAMES = [
    'log_n_ff','log_die_area','log_ff_hpwl','log_ff_spacing','die_aspect',
    'ff_cx','ff_cy','ff_x_std','ff_y_std','frac_xor','comb_per_ff','avg_ds',
    'rel_act','mean_sig_prob','slack_mean','slack_std','slack_min','slack_p10',
    'frac_neg','frac_tight','frac_critical','log_paths_per_ff',
    'log_cd','log_cs','log_mw','log_bd','cd','cs','mw','bd',
    'crit_max_dist','crit_mean_dist','crit_p90_dist','crit_ff_hpwl',
    'crit_cx_offset','crit_cy_offset','crit_x_std','crit_y_std',
    'crit_frac_bndry','crit_star_deg','crit_chain_frac',
    'crit_asymmetry','crit_eccentric','crit_dens_ratio',
    'log_crit_max_um','log_crit_mean_um',
    'cd/ff_spacing','bd/crit_max','mw/crit_max',
    'star_deg*cd','asymm*mw','dens*cs',
    'cmax_dist*cd','asymm*cmax',
    'frac_neg*star','frac_tgt*chain',
    'crit_hpwl/(cs+1)',
    'log(cmax/cd)','log(cmax/bd)','log(cmax/mw)',
    'cx*cd','cy*mw','log(nff/cs)*chpwl',
]


def get_gain_ranking(X, y_z, seed=42):
    sc = StandardScaler()
    lgb = LGBMRegressor(**{**LGB_PARAMS, 'random_state': seed})
    lgb.fit(sc.fit_transform(X), y_z)
    gain = np.array(lgb.booster_.feature_importance(importance_type='gain'), dtype=float)
    return np.argsort(gain)[::-1], gain


def lodo_subset(X, y_raw, meta_df, feat_idx):
    Xs = X[:, feat_idx]
    designs = sorted(meta_df['design_name'].unique())
    y_z, mu_arr, sig_arr = per_placement_z(y_raw, meta_df)
    per_design = {}
    for held in designs:
        tr = (meta_df['design_name'] != held).values
        te = (meta_df['design_name'] == held).values
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xs[tr]); Xte = sc.transform(Xs[te])
        lgb = LGBMRegressor(**LGB_PARAMS)
        lgb.fit(Xtr, y_z[tr])
        pred_z = lgb.predict(Xte)
        pred_ns = pred_z * sig_arr[te] + mu_arr[te]
        per_design[held] = float(np.mean(np.abs(pred_ns - y_raw[te])))
    return per_design


if __name__ == '__main__':
    print("=" * 62)
    print("  Skew Feature Pruning Sweep (setup + hold)")
    print("=" * 62)

    with open(DEF_CACHE,   'rb') as f: dc = pickle.load(f)
    with open(SAIF_CACHE,  'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE,'rb') as f: tc_cache = pickle.load(f)
    with open(SKEW_CACHE,  'rb') as f: skc = pickle.load(f)
    df = pd.read_csv(MANIFEST)
    df = df[df['design_name'].isin(['aes','ethmac','picorv32','sha256'])]

    print("Building features...")
    X, y_setup, y_hold, meta_df = build_features(df, dc, sc_cache, tc_cache, skc)
    designs = sorted(meta_df['design_name'].unique())
    print(f"  Shape: {X.shape}")

    # Gain ranking on full data (seed=42)
    y_z_setup, _, _ = per_placement_z(y_setup, meta_df)
    ranked_idx, gain_arr = get_gain_ranking(X, y_z_setup)

    total = gain_arr.sum() + 1e-12
    print("\n  Top-20 features by gain (skew_setup):")
    cumsum = 0.0
    for r, i in enumerate(ranked_idx[:20], 1):
        pct = gain_arr[i] / total * 100; cumsum += pct
        print(f"    #{r:2d}  {SKEW_FEAT_NAMES[i]:<22}  {pct:5.1f}%  cumul={cumsum:.1f}%")

    N_VALUES = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 63]

    # ── skew_setup sweep ──────────────────────────────────────────────────
    print(f"\n  LODO sweep — skew_setup (ns MAE):")
    print(f"  {'N':>4}  {'aes':>8}  {'ethmac':>8}  {'picorv32':>10}  {'sha256':>8}  {'MEAN':>8}  vs_full")
    print(f"  {'-'*65}")
    setup_full = None
    setup_results = {}
    for N in N_VALUES:
        feat_idx = ranked_idx[:N]
        per_d = lodo_subset(X, y_setup, meta_df, feat_idx)
        mean_mae = np.mean(list(per_d.values()))
        setup_results[N] = {'per_design': per_d, 'mean': mean_mae}
        if N == 63:
            setup_full = mean_mae
        delta = (mean_mae - (setup_full or mean_mae))
        sign = '+' if delta > 0 else ''
        print(f"  {N:>4}  {per_d['aes']:>7.4f}  {per_d['ethmac']:>7.4f}  "
              f"{per_d['picorv32']:>9.4f}  {per_d['sha256']:>7.4f}  "
              f"{mean_mae:>7.4f}  ({sign}{delta:.4f})")

    best_N_setup = min(setup_results, key=lambda k: setup_results[k]['mean'])
    print(f"\n  Best N (setup): {best_N_setup}  MAE={setup_results[best_N_setup]['mean']:.4f}ns"
          f"  (full-63={setup_full:.4f}ns)")

    # ── skew_hold sweep ───────────────────────────────────────────────────
    # Re-rank by hold gain
    y_z_hold, _, _ = per_placement_z(y_hold, meta_df)
    ranked_hold, gain_hold = get_gain_ranking(X, y_z_hold)

    print(f"\n  LODO sweep — skew_hold (ns MAE, re-ranked by hold gain):")
    print(f"  {'N':>4}  {'aes':>8}  {'ethmac':>8}  {'picorv32':>10}  {'sha256':>8}  {'MEAN':>8}  vs_full")
    print(f"  {'-'*65}")
    hold_full = None
    hold_results = {}
    for N in N_VALUES:
        feat_idx = ranked_hold[:N]
        per_d = lodo_subset(X, y_hold, meta_df, feat_idx)
        mean_mae = np.mean(list(per_d.values()))
        hold_results[N] = {'per_design': per_d, 'mean': mean_mae}
        if N == 63:
            hold_full = mean_mae
        delta = (mean_mae - (hold_full or mean_mae))
        sign = '+' if delta > 0 else ''
        print(f"  {N:>4}  {per_d['aes']:>7.4f}  {per_d['ethmac']:>7.4f}  "
              f"{per_d['picorv32']:>9.4f}  {per_d['sha256']:>7.4f}  "
              f"{mean_mae:>7.4f}  ({sign}{delta:.4f})")

    best_N_hold = min(hold_results, key=lambda k: hold_results[k]['mean'])
    print(f"\n  Best N (hold): {best_N_hold}  MAE={hold_results[best_N_hold]['mean']:.4f}ns"
          f"  (full-63={hold_full:.4f}ns)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  PRUNING VERDICT")
    print(f"{'='*62}")
    for tag, results, full_mae, best_N in [
        ('skew_setup', setup_results, setup_full, best_N_setup),
        ('skew_hold',  hold_results,  hold_full,  best_N_hold),
    ]:
        best_mae = results[best_N]['mean']
        improvement = full_mae - best_mae
        if best_N < 63 and improvement > 0.001:
            print(f"  {tag}: CAN prune → top-{best_N} features  "
                  f"MAE {full_mae:.4f} → {best_mae:.4f}ns  (−{improvement:.4f}ns)")
            dropped = [SKEW_FEAT_NAMES[i] for i in (
                (ranked_idx if tag=='skew_setup' else ranked_hold)[best_N:])]
            print(f"    Drop {63-best_N} features: {', '.join(dropped[:8])}"
                  f"{'...' if len(dropped)>8 else ''}")
        else:
            print(f"  {tag}: cannot improve by pruning  "
                  f"(best N={best_N}, Δ={best_mae-full_mae:+.4f}ns)")
