"""
prune_wl_eval.py — Find optimal WL feature count and compare vs full model.

Steps:
  1. Load caches + CSV (same as build_models.py)
  2. Run LODO once (seed=42) to get gain importances → feature ranking
  3. Sweep top-N in [10, 15, 20, 25, 30, 40, 50, 60, 75] using 3 seeds
  4. Print table; find best N
  5. If pruned model beats full (11.0%), retrain + run OOD comparison
"""

import os, sys, time, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

HERE    = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, 'data')
SAVE_DIR = os.path.join(HERE, 'saved_models')

DEF_CACHE     = os.path.join(HERE, 'caches', 'def_cache.pkl')
SAIF_CACHE    = os.path.join(HERE, 'caches', 'saif_cache.pkl')
TIMING_CACHE  = os.path.join(HERE, 'caches', 'timing_cache.pkl')
GRAVITY_CACHE = os.path.join(HERE, 'caches', 'gravity_cache.pkl')
EXT_CACHE     = os.path.join(HERE, 'caches', 'ext_cache.pkl')

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0,
            'zipdiv': 5.0, 'oc_jpegencode': 7.0}

WL_BLEND_ALPHA = 0.3   # Ridge weight; LGB weight = 0.7
MAX_TRAIN_N_FF = None  # set after loading


def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)) * 100


def encode_synth(s):
    if pd.isna(s): return 0.5, 2.0, 0.5
    s = str(s).upper()
    sd = 1.0 if 'DELAY' in s else 0.0
    try: level = float(s.split()[-1])
    except: level = 2.0
    return sd, level, sd * level / 4.0


def build_all_features(df_in, dc, sc_cache, tc_cache, gc, ec):
    wl_rows, y_wl, meta = [], [], []

    for _, row in df_in.iterrows():
        pid    = row['placement_id']
        design = row['design_name']
        df_f   = dc.get(pid)
        sf     = sc_cache.get(pid)
        tf     = tc_cache.get(pid)
        gf     = gc.get(pid, {})
        ec.get(pid, {})

        if not df_f or not sf or not tf:
            continue

        wl = row['wirelength']
        if not (np.isfinite(wl) and wl > 0):
            continue

        t_clk = T_CLK_NS.get(design, 7.0)
        f_ghz = 1.0 / t_clk
        _, _, sa = encode_synth(row.get('synth_strategy', 'AREA 2'))
        core_util = float(row.get('core_util', 55.0)) / 100.0
        density   = float(row.get('density', 0.5))

        n_ff  = df_f['n_ff']; n_active = df_f['n_active']
        die_area = df_f['die_area']; ff_hpwl = df_f['ff_hpwl']
        ff_spacing = df_f['ff_spacing']; avg_ds = df_f['avg_ds']
        frac_xor = df_f['frac_xor']; frac_mux = df_f['frac_mux']
        comb_per_ff = df_f['comb_per_ff']; n_comb = df_f['n_comb']
        n_nets = sf['n_nets']; rel_act = sf['rel_act']
        cd = row['cts_cluster_dia']; cs = row['cts_cluster_size']
        mw = row['cts_max_wire'];    bd = row['cts_buf_dist']

        wl_norm = max(np.sqrt(n_ff * die_area), 1e-3)

        wl_feat = {
            'log_n_ff':        np.log1p(n_ff),
            'log_die_area':    np.log1p(die_area),
            'log_ff_hpwl':     np.log1p(ff_hpwl),
            'log_ff_spacing':  np.log1p(ff_spacing),
            'die_aspect':      df_f['die_aspect'],
            'aspect_ratio':    float(row.get('aspect_ratio', 1.0)),
            'ff_cx':           df_f['ff_cx'],
            'ff_cy':           df_f['ff_cy'],
            'ff_x_std':        df_f['ff_x_std'],
            'ff_y_std':        df_f['ff_y_std'],
            'frac_xor':        frac_xor,
            'frac_mux':        frac_mux,
            'frac_and_or':     df_f['frac_and_or'],
            'frac_nand_nor':   df_f['frac_nand_nor'],
            'frac_ff_active':  df_f['frac_ff_active'],
            'frac_buf_inv':    df_f['frac_buf_inv'],
            'comb_per_ff':     comb_per_ff,
            'avg_ds':          avg_ds,
            'std_ds':          df_f['std_ds'],
            'p90_ds':          df_f['p90_ds'],
            'frac_ds4plus':    df_f['frac_ds4plus'],
            'log_cap_proxy':   np.log1p(df_f['cap_proxy']),
            'rel_act':         rel_act,
            'mean_sig_prob':   sf['mean_sig_prob'],
            'tc_std_norm':     sf['tc_std_norm'],
            'frac_zero':       sf['frac_zero'],
            'frac_high_act':   sf['frac_high_act'],
            'log_n_nets':      sf['log_n_nets'],
            'nets_per_ff':     n_nets / (n_ff + 1),
            'f_ghz':           f_ghz,
            't_clk':           t_clk,
            'core_util':       core_util,
            'density':         density,
            'log_cd':          np.log1p(cd),
            'log_cs':          np.log1p(cs),
            'log_mw':          np.log1p(mw),
            'log_bd':          np.log1p(bd),
            'cd':              cd,
            'cs':              cs,
            'mw':              mw,
            'bd':              bd,
            'xor_comb':        frac_xor * comb_per_ff,
            'act_xor':         rel_act * frac_xor,
            'act_comb':        rel_act * (1 - df_f['frac_ff_active']),
            'log_cd_dens':     np.log1p(cd * n_ff / die_area),
            'log_cs_sp':       np.log1p(cs * ff_spacing),
            'log_mw_hpwl':     np.log1p(mw * ff_hpwl),
            'log_nff_cs':      np.log1p(n_ff / cs),
            'util_dens':       core_util * density,
            'log_act_scale':   np.log1p(n_active * rel_act * f_ghz),
            'log_xor_active':  np.log1p(frac_xor * n_active),
            'log_mux_active':  np.log1p(frac_mux * n_active),
            'log_comb_scale':  np.log1p(comb_per_ff * n_ff),
            'grav_abs_mean':   gf.get('grav_abs_mean', 0.0),
            'grav_abs_std':    gf.get('grav_abs_std', 0.0),
            'grav_abs_p75':    gf.get('grav_abs_p75', 0.0),
            'grav_abs_p90':    gf.get('grav_abs_p90', 0.0),
            'grav_abs_cv':     gf.get('grav_abs_cv', 0.0),
            'grav_abs_gini':   gf.get('grav_abs_gini', 0.0),
            'grav_norm_mean':  gf.get('grav_norm_mean', 0.0),
            'grav_norm_cv':    gf.get('grav_norm_cv', 0.0),
            'grav_anisotropy': gf.get('grav_anisotropy', 0.0),
            'grav_x_cd':       gf.get('grav_abs_mean', 0.0) * cd,
            'grav_x_mw':       gf.get('grav_abs_mean', 0.0) * mw,
            'grav_x_sp':       gf.get('grav_abs_mean', 0.0) / (ff_spacing + 1),
            'tp_deg_mean':     gf.get('tp_degree_mean', 0.0),
            'tp_deg_cv':       gf.get('tp_degree_cv', 0.0),
            'tp_deg_gini':     gf.get('tp_degree_gini', 0.0),
            'tp_deg_p90':      gf.get('tp_degree_p90', 0.0),
            'tp_frac_inv':     gf.get('tp_frac_involved', 0.0),
            'tp_paths_ff':     gf.get('tp_paths_per_ff', 0.0),
            'tp_frac_hub':     gf.get('tp_frac_hub', 0.0),
            'log_area_per_ff': np.log1p(die_area / (n_ff + 1)),
            'log_n_comb':      np.log1p(n_comb),
            'comb_scale':      comb_per_ff * np.log1p(n_ff),
        }

        wl_rows.append(list(wl_feat.values()))
        y_wl.append(np.log(wl / wl_norm))
        meta.append({'placement_id': pid, 'design_name': design,
                     'wirelength': wl, 'wl_norm': wl_norm, 'n_ff': n_ff})

    wl_names = list(wl_feat.keys())
    X_wl = np.array(wl_rows, dtype=np.float64)
    y_wl_a = np.array(y_wl)
    meta_df = pd.DataFrame(meta)

    for c in range(X_wl.shape[1]):
        bad = ~np.isfinite(X_wl[:, c])
        if bad.any():
            X_wl[bad, c] = np.nanmedian(X_wl[~bad, c]) if (~bad).any() else 0.0

    return X_wl, y_wl_a, meta_df, wl_names


def lodo_wl_subset(X, y, meta_df, feat_idx, seeds=(42, 7, 13)):
    """Run LODO on a feature subset given by feat_idx."""
    Xs = X[:, feat_idx]
    designs = sorted(meta_df['design_name'].unique())
    per_design = {d: [] for d in designs}
    for seed in seeds:
        for held in designs:
            tr = (meta_df['design_name'] != held).values
            te = (meta_df['design_name'] == held).values
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xs[tr]); Xte = sc.transform(Xs[te])
            lgb = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                                min_child_samples=10, verbose=-1, n_jobs=1, random_state=seed)
            lgb.fit(Xtr, y[tr])
            rdg = Ridge(alpha=1000., max_iter=10000); rdg.fit(Xtr, y[tr])
            p_lgb = lgb.predict(Xte)
            p_rdg = np.clip(rdg.predict(Xte), -5.0, 8.0)
            pred = np.exp(WL_BLEND_ALPHA * p_lgb + (1 - WL_BLEND_ALPHA) * p_rdg)
            pred *= meta_df[te]['wl_norm'].values
            per_design[held].append(mape(meta_df[te]['wirelength'].values, pred))
    return {d: np.mean(v) for d, v in per_design.items()}


def get_gain_ranking(X, y, meta_df, wl_names, seed=42):
    """Train on all data once, return features sorted by gain importance (descending)."""
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    lgb = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                        min_child_samples=10, verbose=-1, n_jobs=1, random_state=seed)
    lgb.fit(Xs, y)
    gain = np.array(lgb.booster_.feature_importance(importance_type='gain'), dtype=float)
    ranked = np.argsort(gain)[::-1]  # descending
    return ranked, gain


if __name__ == '__main__':
    print("=" * 60)
    print("WL Feature Pruning Analysis")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\n{T()} Loading caches...")
    with open(DEF_CACHE,     'rb') as f: dc = pickle.load(f)
    with open(SAIF_CACHE,    'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE,  'rb') as f: tc_cache = pickle.load(f)
    with open(GRAVITY_CACHE, 'rb') as f: gc = pickle.load(f)
    with open(EXT_CACHE,     'rb') as f: ec = pickle.load(f)

    df = pd.read_csv(os.path.join(DATASET, 'unified_manifest.csv'))
    df = df[df['design_name'].isin(['aes', 'ethmac', 'picorv32', 'sha256'])]
    print(f"  Rows: {len(df)}")

    X_wl, y_wl, meta_df, wl_names = build_all_features(df, dc, sc_cache, tc_cache, gc, ec)
    print(f"  Samples: {X_wl.shape[0]}  Features: {X_wl.shape[1]}")

    MAX_TRAIN_N_FF = int(meta_df['n_ff'].max())
    print(f"  max_train_n_ff: {MAX_TRAIN_N_FF}  OOD threshold: {MAX_TRAIN_N_FF*1.3:.0f}")

    # ── Step 1: get gain ranking on all data ──────────────────────────────
    print(f"\n{T()} Getting gain importance ranking (full-data LGB)...")
    ranked_idx, gain_arr = get_gain_ranking(X_wl, y_wl, meta_df, wl_names)

    total_gain = gain_arr.sum() + 1e-12
    print("  Top-15 by gain:")
    cumsum = 0.0
    for r, i in enumerate(ranked_idx[:15], 1):
        pct = gain_arr[i] / total_gain * 100
        cumsum += pct
        print(f"    #{r:2d}  {wl_names[i]:<22}  {pct:5.1f}%  cumul={cumsum:.1f}%")

    # ── Step 2: LODO sweep over top-N subsets ─────────────────────────────
    N_VALUES = [10, 15, 20, 25, 30, 40, 50, 60, 75]
    DESIGNS = sorted(meta_df['design_name'].unique())

    print(f"\n{T()} LODO sweep (3 seeds) for each top-N subset...")
    print(f"\n  {'N':>4}  {'aes':>7}  {'ethmac':>7}  {'picorv32':>9}  {'sha256':>7}  {'MEAN':>7}  vs_full")
    print(f"  {'-'*60}")

    results = {}
    for N in N_VALUES:
        feat_idx = ranked_idx[:N]
        per_d = lodo_wl_subset(X_wl, y_wl, meta_df, feat_idx)
        mean_mape = np.mean(list(per_d.values()))
        results[N] = {'per_design': per_d, 'mean': mean_mape}
        delta = mean_mape - 11.0
        sign = '+' if delta > 0 else ''
        print(f"  {N:>4}  {per_d['aes']:>6.1f}%  {per_d['ethmac']:>6.1f}%  "
              f"{per_d['picorv32']:>8.1f}%  {per_d['sha256']:>6.1f}%  "
              f"{mean_mape:>6.1f}%  ({sign}{delta:.1f}%)")

    # ── Step 3: Find best N ───────────────────────────────────────────────
    best_N = min(results, key=lambda k: results[k]['mean'])
    best_mean = results[best_N]['mean']
    print(f"\n  Best: N={best_N}  mean MAPE={best_mean:.1f}%  (full=11.0%)")

    if best_N < 75:
        feat_names_pruned = [wl_names[i] for i in ranked_idx[:best_N]]
        print(f"\n  Pruned feature set ({best_N} features):")
        for i, fn in enumerate(feat_names_pruned, 1):
            print(f"    {i:2d}. {fn}")
        dropped = [wl_names[i] for i in ranked_idx[best_N:]]
        print(f"\n  Dropped ({len(dropped)} features):")
        for fn in dropped:
            print(f"    - {fn}")
    else:
        print("  → No improvement found; keeping all 75 features.")
