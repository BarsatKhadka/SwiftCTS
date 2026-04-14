"""
eval_lodo.py — LODO evaluation for SwiftCTS (power + wirelength).
Usage: python3 eval_lodo.py
"""

import os, sys, pickle, warnings, time
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

DEF_CACHE     = os.path.join(HERE, 'caches', 'def_cache.pkl')
SAIF_CACHE    = os.path.join(HERE, 'caches', 'saif_cache.pkl')
TIMING_CACHE  = os.path.join(HERE, 'caches', 'timing_cache.pkl')
GRAVITY_CACHE = os.path.join(HERE, 'caches', 'gravity_cache.pkl')
MODEL_PATH    = os.path.join(HERE, 'saved_models', 'model.pkl')

sys.path.insert(0, HERE)
from swiftcts import SwiftCTS

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0,
            'zipdiv': 5.0, 'oc_jpegencode': 7.0}


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
    pw_rows, wl_rows, y_pw, y_wl, meta = [], [], [], [], []

    for _, row in df_in.iterrows():
        pid    = row['placement_id']
        design = row['design_name']
        df_f   = dc.get(pid); sf = sc_cache.get(pid)
        tf     = tc_cache.get(pid); gf = gc.get(pid, {})

        if not df_f or not sf or not tf: continue
        pw = row['power_total']; wl = row['wirelength']
        if not (np.isfinite(pw) and np.isfinite(wl) and pw > 0 and wl > 0): continue

        t_clk = T_CLK_NS.get(design, 7.0); f_ghz = 1.0 / t_clk
        sd, sl, sa = encode_synth(row.get('synth_strategy', 'AREA 2'))
        core_util = float(row.get('core_util', 55.0)) / 100.0
        density   = float(row.get('density', 0.5))

        n_ff = df_f['n_ff']; n_active = df_f['n_active']
        die_area = df_f['die_area']; ff_hpwl = df_f['ff_hpwl']
        ff_spacing = df_f['ff_spacing']; avg_ds = df_f['avg_ds']
        frac_xor = df_f['frac_xor']; frac_mux = df_f['frac_mux']
        comb_per_ff = df_f['comb_per_ff']; n_comb = df_f['n_comb']
        n_nets = sf['n_nets']; rel_act = sf['rel_act']
        cd = row['cts_cluster_dia']; cs = row['cts_cluster_size']
        mw = row['cts_max_wire'];    bd = row['cts_buf_dist']

        pw_norm = max(n_ff * f_ghz * avg_ds, 1e-10)
        wl_norm = max(np.sqrt(n_ff * die_area), 1e-3)
        sm = tf['slack_mean']; ft = tf['frac_tight']

        pw_feat = [
            np.log1p(n_active * rel_act * f_ghz), np.log1p(n_ff), rel_act,
            np.log1p(frac_mux * n_active), np.log1p(df_f['cap_proxy']), frac_mux,
            sf['frac_high_act'], df_f['frac_and_or'], tf['slack_std'], ft,
            sf['mean_sig_prob'], tf['slack_p10'], tf['slack_p50'], sm * frac_xor,
            f_ghz, sa, np.log1p(frac_xor * n_active), sf['log_n_nets'],
            df_f['frac_nand_nor'], np.log1p(cd * n_ff / die_area),
        ]

        wl_feat = [
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            df_f['die_aspect'], float(row.get('aspect_ratio', 1.0)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']), rel_act,
            sf['mean_sig_prob'], sf['tc_std_norm'], sf['frac_zero'],
            sf['frac_high_act'], sf['log_n_nets'], n_nets / (n_ff + 1),
            f_ghz, t_clk, core_util, density,
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
            cd, cs, mw, bd,
            frac_xor * comb_per_ff, rel_act * frac_xor,
            rel_act * (1 - df_f['frac_ff_active']),
            np.log1p(cd * n_ff / die_area), np.log1p(cs * ff_spacing),
            np.log1p(mw * ff_hpwl), np.log1p(n_ff / cs),
            core_util * density, np.log1p(n_active * rel_act * f_ghz),
            np.log1p(frac_xor * n_active), np.log1p(frac_mux * n_active),
            np.log1p(comb_per_ff * n_ff),
            gf.get('grav_abs_mean', 0.0), gf.get('grav_abs_std', 0.0),
            gf.get('grav_abs_p75', 0.0), gf.get('grav_abs_p90', 0.0),
            gf.get('grav_abs_cv', 0.0), gf.get('grav_abs_gini', 0.0),
            gf.get('grav_norm_mean', 0.0), gf.get('grav_norm_cv', 0.0),
            gf.get('grav_anisotropy', 0.0),
            gf.get('grav_abs_mean', 0.0) * cd,
            gf.get('grav_abs_mean', 0.0) * mw,
            gf.get('grav_abs_mean', 0.0) / (ff_spacing + 1),
            gf.get('tp_degree_mean', 0.0), gf.get('tp_degree_cv', 0.0),
            gf.get('tp_degree_gini', 0.0), gf.get('tp_degree_p90', 0.0),
            gf.get('tp_frac_involved', 0.0), gf.get('tp_paths_per_ff', 0.0),
            gf.get('tp_frac_hub', 0.0),
            np.log1p(die_area / (n_ff + 1)), np.log1p(n_comb),
            comb_per_ff * np.log1p(n_ff),
        ]

        pw_rows.append(pw_feat); wl_rows.append(wl_feat)
        y_pw.append(np.log(pw / pw_norm)); y_wl.append(np.log(wl / wl_norm))
        meta.append({'placement_id': pid, 'design_name': design,
                     'power_total': pw, 'wirelength': wl,
                     'pw_norm': pw_norm, 'wl_norm': wl_norm})

    X_pw = np.array(pw_rows, dtype=np.float64)
    X_wl = np.array(wl_rows, dtype=np.float64)
    for X in [X_pw, X_wl]:
        if X.ndim < 2 or X.shape[0] == 0: continue
        for c in range(X.shape[1]):
            bad = ~np.isfinite(X[:, c])
            if bad.any():
                X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0
    return X_pw, X_wl, np.array(y_pw), np.array(y_wl), pd.DataFrame(meta)


def run_lodo(X_pw, X_wl, y_pw, y_wl, meta_df, wl_alpha=0.3, seed=42):
    designs = sorted(meta_df['design_name'].unique())
    results = {}
    for held in designs:
        tr = (meta_df['design_name'] != held).values
        te = (meta_df['design_name'] == held).values

        sc_pw = StandardScaler()
        m_pw = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=seed, verbosity=0, n_jobs=1)
        m_pw.fit(sc_pw.fit_transform(X_pw[tr]), y_pw[tr])
        pred_pw = np.exp(m_pw.predict(sc_pw.transform(X_pw[te]))) * meta_df[te]['pw_norm'].values

        sc_wl = StandardScaler()
        Xtr = sc_wl.fit_transform(X_wl[tr]); Xte = sc_wl.transform(X_wl[te])
        lgb = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                            min_child_samples=10, verbose=-1, n_jobs=1, random_state=seed)
        lgb.fit(Xtr, y_wl[tr])
        rdg = Ridge(alpha=1000., max_iter=10000); rdg.fit(Xtr, y_wl[tr])
        pred_wl = np.exp(wl_alpha * lgb.predict(Xte) + (1 - wl_alpha) * rdg.predict(Xte))
        pred_wl *= meta_df[te]['wl_norm'].values

        results[held] = {
            'pw_mape': mape(meta_df[te]['power_total'].values, pred_pw),
            'wl_mape': mape(meta_df[te]['wirelength'].values, pred_wl),
            'n':       int(te.sum()),
        }
    return results


if __name__ == '__main__':
    print(f"{T()} Loading caches...")
    with open(DEF_CACHE,     'rb') as f: dc = pickle.load(f)
    with open(SAIF_CACHE,    'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE,  'rb') as f: tc_cache = pickle.load(f)
    with open(GRAVITY_CACHE, 'rb') as f: gc = pickle.load(f)
    ec = {}

    print(f"{T()} Loading CSV...")
    df = pd.read_csv(os.path.join(DATASET, 'unified_manifest.csv'))
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)

    print(f"{T()} Building features ({len(df)} rows)...")
    X_pw, X_wl, y_pw, y_wl, meta_df = build_all_features(df, dc, sc_cache, tc_cache, gc, ec)

    print(f"{T()} Running LODO...")
    sys.stdout.flush()
    results = run_lodo(X_pw, X_wl, y_pw, y_wl, meta_df)

    pw_vals = [results[d]['pw_mape'] for d in sorted(results)]
    wl_vals = [results[d]['wl_mape'] for d in sorted(results)]

    print()
    print(f"{'Design':<12}  {'N':>4}  {'Power MAPE':>10}  {'WL MAPE':>8}")
    print("-" * 42)
    for d in sorted(results):
        r = results[d]
        print(f"{d:<12}  {r['n']:>4}  {r['pw_mape']:>9.1f}%  {r['wl_mape']:>7.1f}%")
    print("-" * 42)
    print(f"{'Mean':<12}  {'':>4}  {np.mean(pw_vals):>9.1f}%  {np.mean(wl_vals):>7.1f}%")
    print(f"{'Std':<12}  {'':>4}  {np.std(pw_vals):>9.1f}%  {np.std(wl_vals):>7.1f}%")
    print()
    print("Note: for skew results K-wise see eval_lodo_kshot.py")
    print(f"\n{T()} done")
