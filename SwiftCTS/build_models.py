"""
build_models.py — Train and save SwiftCTS models.

Steps:
  1. Load caches + CSV
  2. Build features (20 power, 75 WL)
  3. Run LODO to verify performance matches expected
     (abort if power mean MAPE > 28%)
  4. Train final models on all data
  5. Save to saved_models/model.pkl
     (skew/hold heads are not included; set EXISTING_MODEL path if needed)

Expected LODO (multi-seed):
  Power: ~25.6% mean MAPE  (vs 32.0% with 76 features)
  WL:    ~11.0% mean MAPE  (same as baseline)
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

# Paths — all relative to this file's directory
HERE    = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, 'data')
SAVE_DIR = os.path.join(HERE, 'saved_models')
os.makedirs(SAVE_DIR, exist_ok=True)

DEF_CACHE     = os.path.join(HERE, 'caches', 'def_cache.pkl')
SAIF_CACHE    = os.path.join(HERE, 'caches', 'saif_cache.pkl')
TIMING_CACHE  = os.path.join(HERE, 'caches', 'timing_cache.pkl')
GRAVITY_CACHE = os.path.join(HERE, 'caches', 'gravity_cache.pkl')
# EXT_CACHE removed — ext features not used in any head
EXISTING_MODEL = None  # skew/hold heads not needed for standalone rebuild; set to None

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0,
            'zipdiv': 5.0, 'oc_jpegencode': 7.0}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)) * 100


def encode_synth(s):
    if pd.isna(s): return 0.5, 2.0, 0.5
    s = str(s).upper()
    sd = 1.0 if 'DELAY' in s else 0.0
    try: level = float(s.split()[-1])
    except: level = 2.0
    return sd, level, sd * level / 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder (identical to minimal_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_all_features(df_in, dc, sc_cache, tc_cache, gc, ec):
    """Build named feature matrices for power (20 feats) and WL (75 feats)."""
    pw_rows, wl_rows, y_pw, y_wl, meta = [], [], [], [], []

    for _, row in df_in.iterrows():
        pid    = row['placement_id']
        design = row['design_name']
        df_f   = dc.get(pid)
        sf     = sc_cache.get(pid)
        tf     = tc_cache.get(pid)
        gf     = gc.get(pid, {})
        ef     = ec.get(pid, {})

        if not df_f or not sf or not tf:
            continue

        pw = row['power_total']
        wl = row['wirelength']
        if not (np.isfinite(pw) and np.isfinite(wl) and pw > 0 and wl > 0):
            continue

        t_clk = T_CLK_NS.get(design, 7.0)
        f_ghz = 1.0 / t_clk
        sd, sl, sa = encode_synth(row.get('synth_strategy', 'AREA 2'))
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

        pw_norm = max(n_ff * f_ghz * avg_ds, 1e-10)
        wl_norm = max(np.sqrt(n_ff * die_area), 1e-3)

        sm = tf['slack_mean']; fn = tf['frac_neg']; ft = tf['frac_tight']

        # ── Power: top-20 minimal features ───────────────────────────────
        pw_feat = {
            'log_act_scale':   np.log1p(n_active * rel_act * f_ghz),
            'log_n_ff':        np.log1p(n_ff),
            'rel_act':         rel_act,
            'log_mux_active':  np.log1p(frac_mux * n_active),
            'log_cap_proxy':   np.log1p(df_f['cap_proxy']),
            'frac_mux':        frac_mux,
            'frac_high_act':   sf['frac_high_act'],
            'frac_and_or':     df_f['frac_and_or'],
            'slack_std':       tf['slack_std'],
            'frac_tight':      ft,
            'mean_sig_prob':   sf['mean_sig_prob'],
            'slack_p10':       tf['slack_p10'],
            'slack_p50':       tf['slack_p50'],
            'slack_xor':       sm * frac_xor,
            'f_ghz':           f_ghz,
            'synth_area':      sa,
            'log_xor_active':  np.log1p(frac_xor * n_active),
            'log_n_nets':      sf['log_n_nets'],
            'frac_nand_nor':   df_f['frac_nand_nor'],
            'log_cd_dens':     np.log1p(cd * n_ff / die_area),
        }  # 20 features

        # ── WL: full 75-feature set ───────────────────────────────────────
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
        }  # 75 features

        pw_rows.append(list(pw_feat.values()))
        wl_rows.append(list(wl_feat.values()))
        y_pw.append(np.log(pw / pw_norm))
        y_wl.append(np.log(wl / wl_norm))
        meta.append({'placement_id': pid, 'design_name': design,
                     'power_total': pw, 'wirelength': wl,
                     'pw_norm': pw_norm, 'wl_norm': wl_norm})

    pw_names = list(pw_feat.keys())
    wl_names = list(wl_feat.keys())
    X_pw = np.array(pw_rows, dtype=np.float64)
    X_wl = np.array(wl_rows, dtype=np.float64)
    y_pw_a = np.array(y_pw)
    y_wl_a = np.array(y_wl)
    meta_df = pd.DataFrame(meta)

    for X in [X_pw, X_wl]:
        if X.ndim < 2 or X.shape[0] == 0: continue
        for c in range(X.shape[1]):
            bad = ~np.isfinite(X[:, c])
            if bad.any():
                X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0

    return X_pw, X_wl, y_pw_a, y_wl_a, meta_df, pw_names, wl_names


# ─────────────────────────────────────────────────────────────────────────────
# LODO evaluation
# ─────────────────────────────────────────────────────────────────────────────

def lodo_power(X, y, meta_df, seeds=(42, 7, 13, 99, 2024)):
    designs = sorted(meta_df['design_name'].unique())
    per_design = {d: [] for d in designs}
    for seed in seeds:
        for held in designs:
            tr = meta_df['design_name'] != held
            te = meta_df['design_name'] == held
            sc = StandardScaler()
            m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, verbosity=0, n_jobs=1)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            pred = np.exp(m.predict(sc.transform(X[te]))) * meta_df[te]['pw_norm'].values
            per_design[held].append(mape(meta_df[te]['power_total'].values, pred))
    return {d: np.mean(v) for d, v in per_design.items()}


def lodo_wl(X, y, meta_df, wl_alpha=0.3, seeds=(42, 7, 13)):
    designs = sorted(meta_df['design_name'].unique())
    per_design = {d: [] for d in designs}
    for seed in seeds:
        for held in designs:
            tr = meta_df['design_name'] != held
            te = meta_df['design_name'] == held
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
            lgb = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                                min_child_samples=10, verbose=-1, n_jobs=1, random_state=seed)
            lgb.fit(Xtr, y[tr])
            rdg = Ridge(alpha=1000., max_iter=10000); rdg.fit(Xtr, y[tr])
            pred = np.exp(wl_alpha * lgb.predict(Xte) + (1 - wl_alpha) * rdg.predict(Xte))
            pred *= meta_df[te]['wl_norm'].values
            per_design[held].append(mape(meta_df[te]['wirelength'].values, pred))
    return {d: np.mean(v) for d, v in per_design.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Final model training (all data)
# ─────────────────────────────────────────────────────────────────────────────

def train_final(X_pw, X_wl, y_pw, y_wl, wl_alpha=0.3, seed=42):
    sc_pw = StandardScaler()
    m_pw = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, verbosity=0, n_jobs=1)
    m_pw.fit(sc_pw.fit_transform(X_pw), y_pw)

    sc_wl = StandardScaler()
    Xtr_wl = sc_wl.fit_transform(X_wl)
    lgb_wl = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                           min_child_samples=10, verbose=-1, n_jobs=1, random_state=seed)
    lgb_wl.fit(Xtr_wl, y_wl)
    ridge_wl = Ridge(alpha=1000., max_iter=10000)
    ridge_wl.fit(Xtr_wl, y_wl)

    return m_pw, sc_pw, lgb_wl, ridge_wl, sc_wl


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Building SwiftCTS models")
    print("=" * 60)

    # ── 1. Load caches ────────────────────────────────────────────────────
    print(f"\n{T()} Loading caches...")
    with open(DEF_CACHE,     'rb') as f: dc = pickle.load(f)
    with open(SAIF_CACHE,    'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE,  'rb') as f: tc_cache = pickle.load(f)
    with open(GRAVITY_CACHE, 'rb') as f: gc = pickle.load(f)
    ec = {}  # ext_cache unused in feature builders
    print(f"  DEF:{len(dc)}  SAIF:{len(sc_cache)}  Timing:{len(tc_cache)}  "
          f"Gravity:{len(gc)}")

    # ── 2. Load CSV ───────────────────────────────────────────────────────
    print(f"{T()} Loading CSV...")
    df = pd.read_csv(os.path.join(DATASET, 'unified_manifest.csv'))
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    designs = sorted(df['design_name'].unique())
    print(f"  Rows: {len(df)}  Designs: {designs}")

    # ── 3. Build features ─────────────────────────────────────────────────
    print(f"{T()} Building features...")
    X_pw, X_wl, y_pw, y_wl, meta_df, pw_names, wl_names = build_all_features(
        df, dc, sc_cache, tc_cache, gc, ec)
    print(f"  Power: {X_pw.shape}  ({len(pw_names)} features)")
    print(f"  WL:    {X_wl.shape}  ({len(wl_names)} features)")
    print(f"  Samples: {len(meta_df)}")
    sys.stdout.flush()

    # ── 4. LODO verification ──────────────────────────────────────────────
    print(f"\n{T()} Running LODO verification (5 seeds power, 3 seeds WL)...")
    print("  This confirms performance matches expected before saving.\n")
    sys.stdout.flush()

    pw_res = lodo_power(X_pw, y_pw, meta_df)
    wl_res = lodo_wl(X_wl, y_wl, meta_df)

    pw_mean = np.mean(list(pw_res.values()))
    wl_mean = np.mean(list(wl_res.values()))

    print(f"\n  {'Design':12} {'Power MAPE':>12} {'WL MAPE':>10}")
    print(f"  {'-'*36}")
    for d in sorted(pw_res.keys()):
        print(f"  {d:12} {pw_res[d]:11.1f}%  {wl_res.get(d, float('nan')):9.1f}%")
    print(f"  {'Mean':12} {pw_mean:11.1f}%  {wl_mean:9.1f}%")
    print(f"\n  Baseline v16_final: power=32.0%  WL=11.0%")
    print(f"  This model:         power={pw_mean:.1f}%  WL={wl_mean:.1f}%")
    sys.stdout.flush()

    # ── Abort if performance is worse than expected ───────────────────────
    POWER_ABORT_THRESHOLD = 28.0
    if pw_mean > POWER_ABORT_THRESHOLD:
        print(f"\n  ABORT: Power mean MAPE {pw_mean:.1f}% > threshold {POWER_ABORT_THRESHOLD}%")
        print(f"  Expected ~25.6%. Something is wrong with features or data.")
        print(f"  NOT saving model. Investigate discrepancy.")
        sys.exit(1)

    print(f"\n  Verification PASSED (power={pw_mean:.1f}% < {POWER_ABORT_THRESHOLD}% threshold)")
    sys.stdout.flush()

    # ── 5. Train final models on ALL data ────────────────────────────────
    max_train_n_ff = int(meta_df.groupby('placement_id').first()
                         .reset_index()['placement_id']
                         .map(lambda p: dc[p]['n_ff'] if p in dc else 0).max())
    print(f"\n{T()} Training final models on all {len(meta_df)} samples...")
    print(f"  max_train_n_ff (for OOD detection): {max_train_n_ff}")
    m_pw, sc_pw, lgb_wl, ridge_wl, sc_wl = train_final(X_pw, X_wl, y_pw, y_wl)
    print(f"  Power XGB: trained")
    print(f"  WL LGB+Ridge: trained")
    sys.stdout.flush()

    # ── 6. Assemble model dict (skew/hold from EXISTING_MODEL if set) ─────
    model_dict = {
        'model_power':      m_pw,
        'scaler_power':     sc_pw,
        'model_wl_lgb':     lgb_wl,
        'model_wl_ridge':   ridge_wl,
        'scaler_wl':        sc_wl,
        'wl_blend_alpha':   0.3,   # Ridge weight; LGB weight = 0.7
        'max_train_n_ff':   max_train_n_ff,  # OOD threshold: n_ff > 1.5x → LGB-only
        'lodo': {
            'power': pw_res,
            'wl':    wl_res,
            'power_mean_mape': pw_mean,
            'wl_mean_mape':    wl_mean,
            'n_pw_features':   len(pw_names),
            'n_wl_features':   len(wl_names),
            'pw_feature_names': pw_names,
            'wl_feature_names': wl_names,
            # Training target range for Ridge clipping (used by _Heads)
            'wl_y_min': float(y_wl.min()),
            'wl_y_max': float(y_wl.max()),
        },
    }

    if EXISTING_MODEL is not None:
        print(f"\n{T()} Loading skew + hold heads from {EXISTING_MODEL}...")
        try:
            with open(EXISTING_MODEL, 'rb') as f:
                existing = pickle.load(f)
            model_dict['model_skew']       = existing['model_skew']
            model_dict['scaler_skew']      = existing['scaler_skew']
            model_dict['model_hold_vio']   = existing['model_hold_vio']
            model_dict['scaler_hold_vio']  = existing['scaler_hold_vio']
            print(f"  Loaded skew/hold heads: {[k for k in existing.keys() if 'skew' in k or 'hold' in k]}")
        except Exception as e:
            print(f"  WARNING: Could not load skew/hold heads from {EXISTING_MODEL}: {e}")
            print(f"  Saving model without skew/hold heads.")
    else:
        print(f"\n{T()} WARNING: EXISTING_MODEL is None — skew/hold heads will not be included in saved model.")

    # ── 7. Save ───────────────────────────────────────────────────────────
    save_path = os.path.join(SAVE_DIR, 'model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"\n{T()} Model saved to: {save_path}")
    print(f"  Power MAPE (LODO): {pw_mean:.1f}%")
    print(f"  WL MAPE (LODO):    {wl_mean:.1f}%")
    print(f"  Power features:    {len(pw_names)}")
    print(f"  WL features:       {len(wl_names)}")
    print(f"\n{T()} DONE")
