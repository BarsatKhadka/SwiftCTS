"""
eval_lodo_kshot.py — LODO + K-shot multiplicative calibration.

For each held-out design: train on 3, predict on 4th (zero-shot),
then apply per-placement K-shot scaling for K in {0,1,2,5,10}.

Usage: python3 eval_lodo_kshot.py
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

HERE          = os.path.dirname(os.path.abspath(__file__))
DATASET       = os.path.join(HERE, 'data')
DEF_CACHE     = os.path.join(HERE, 'caches', 'def_cache.pkl')
SAIF_CACHE    = os.path.join(HERE, 'caches', 'saif_cache.pkl')
TIMING_CACHE  = os.path.join(HERE, 'caches', 'timing_cache.pkl')
GRAVITY_CACHE = os.path.join(HERE, 'caches', 'gravity_cache.pkl')

sys.path.insert(0, HERE)
from eval_skew import build_features as build_skew_features, per_placement_z

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0}
K_VALUES = [0, 1, 2, 5]
WL_ALPHA = 0.3
SEED     = 42


def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)) * 100


def encode_synth(s):
    if pd.isna(s): return 0.5, 2.0, 0.5
    s = str(s).upper()
    sd = 1.0 if 'DELAY' in s else 0.0
    try: level = float(s.split()[-1])
    except: level = 2.0
    return sd, level, sd * level / 4.0


def build_all_features(df_in, dc, sc_cache, tc_cache, gc):
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


LGB_PARAMS = dict(n_estimators=300, num_leaves=31, learning_rate=0.03,
                  min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)


def kshot_skew_ns_mae(pred_z, true_ns, te_pids, k):
    """
    Skew MAE (ns) using per-placement K-shot mu/sig estimation.

    For each placement, the first K runs estimate mu/sig and evaluation is
    on the remaining runs only (cal runs excluded to avoid leakage):

        mu  = mean(true_ns[:K])
        sig = std(true_ns[:K])  with floor = max(|mu|*0.01, 1e-4)
        pred_ns = pred_z * sig + mu   (evaluated on runs K+1..N only)
    """
    if k == 0:
        return None, True   # no samples → nothing to show

    errors = []
    for pid in sorted(set(te_pids)):
        all_idx  = np.where(te_pids == pid)[0]
        cal_idx  = all_idx[:k]
        eval_idx = all_idx[k:]
        if len(eval_idx) == 0: continue
        cal_ns  = true_ns[cal_idx]
        mu      = float(cal_ns.mean())
        sig     = max(float(cal_ns.std()), max(abs(mu) * 0.01, 1e-4))
        pred_ns = pred_z[eval_idx] * sig + mu
        errors.extend(np.abs(pred_ns - true_ns[eval_idx]).tolist())
    if not errors:
        return None, False
    return float(np.mean(errors)), False


def log_space_scale(true_vals, pred_vals, clip=4.6):
    """
    Multiplicative calibration scale from K observations.

    Because power and wirelength errors are inherently multiplicative, we
    compute this scale using the geometric mean of the ratios via a log-space
    transformation.  This ensures that symmetric multiplicative errors correctly
    cancel, preventing the positive bias inherent in an arithmetic mean of
    ratios:

        k_cal = exp( (1/K) * sum_i log(y_i / y_hat_i) )

    Equivalently this is the geometric mean of (y_i / y_hat_i).
    """
    ratios = np.log(np.array(true_vals) / (np.array(pred_vals) + 1e-12))
    return float(np.exp(np.clip(np.mean(ratios), -clip, clip)))


def run_lodo_kshot(X_pw, X_wl, y_pw, y_wl, meta_df,
                   X_sk=None, y_sk_raw=None, sk_meta_df=None,
                   sk_feat_idx=None,
                   k_values=K_VALUES, seed=SEED):
    designs = sorted(meta_df['design_name'].unique())
    results = {d: {} for d in designs}

    # Pre-compute skew z-scores for all data (uses all runs — only for training
    # the model weights, not for the test-time ns conversion).
    if X_sk is not None:
        y_sk_z, _, _ = per_placement_z(y_sk_raw, sk_meta_df)

    for held in designs:
        tr = (meta_df['design_name'] != held).values
        te = (meta_df['design_name'] == held).values

        # ── Power (XGBoost) ──────────────────────────────────────────────
        sc_pw = StandardScaler()
        m_pw = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=seed, verbosity=0, n_jobs=1)
        m_pw.fit(sc_pw.fit_transform(X_pw[tr]), y_pw[tr])

        # ── WL (LGB + Ridge blend) ────────────────────────────────────────
        sc_wl = StandardScaler()
        Xtr_wl = sc_wl.fit_transform(X_wl[tr])
        Xte_wl = sc_wl.transform(X_wl[te])
        lgb = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                            min_child_samples=10, verbose=-1, n_jobs=1, random_state=seed)
        lgb.fit(Xtr_wl, y_wl[tr])
        rdg = Ridge(alpha=1000., max_iter=10000)
        rdg.fit(Xtr_wl, y_wl[tr])

        pred_pw_raw = (np.exp(m_pw.predict(sc_pw.transform(X_pw[te])))
                       * meta_df[te]['pw_norm'].values)
        pred_wl_raw = (np.exp(WL_ALPHA * lgb.predict(Xte_wl) +
                              (1 - WL_ALPHA) * rdg.predict(Xte_wl))
                       * meta_df[te]['wl_norm'].values)

        true_pw  = meta_df[te]['power_total'].values
        true_wl  = meta_df[te]['wirelength'].values
        te_pids  = meta_df[te]['placement_id'].values

        # ── Skew (LGB on pruned features, z-score target) ────────────────
        pred_sk_z  = None
        true_sk_ns = None
        sk_te_pids = None
        if X_sk is not None:
            sk_tr = (sk_meta_df['design_name'] != held).values
            sk_te = (sk_meta_df['design_name'] == held).values
            Xs = X_sk[:, sk_feat_idx] if sk_feat_idx is not None else X_sk
            sc_sk = StandardScaler()
            m_sk  = LGBMRegressor(**LGB_PARAMS)
            m_sk.fit(sc_sk.fit_transform(Xs[sk_tr]), y_sk_z[sk_tr])
            pred_sk_z  = m_sk.predict(sc_sk.transform(Xs[sk_te]))
            true_sk_ns = y_sk_raw[sk_te]
            sk_te_pids = sk_meta_df[sk_te]['pid'].values

        for k in k_values:
            # Per-placement K-shot: for each test placement independently,
            # the first K runs calibrate the model and evaluation is on the
            # remaining N−K runs only.  K=0 is pure zero-shot (no calibration).
            eval_mask = np.ones(len(te_pids), dtype=bool)
            pw_out    = pred_pw_raw.copy()
            wl_out    = pred_wl_raw.copy()

            if k > 0:
                for pid in sorted(set(te_pids)):
                    pid_mask = te_pids == pid
                    cal_idx  = np.where(pid_mask)[0][:k]
                    if len(cal_idx) == 0: continue
                    pw_out[pid_mask] = pred_pw_raw[pid_mask] * log_space_scale(
                        true_pw[cal_idx], pred_pw_raw[cal_idx])
                    wl_out[pid_mask] = pred_wl_raw[pid_mask] * log_space_scale(
                        true_wl[cal_idx], pred_wl_raw[cal_idx])
                    eval_mask[cal_idx] = False   # exclude the K cal runs from MAPE

            entry = {
                'pw_mape': mape(true_pw[eval_mask], pw_out[eval_mask]),
                'wl_mape': mape(true_wl[eval_mask], wl_out[eval_mask]),
                'sk_val':  None,
                'sk_is_z': False,
            }

            if pred_sk_z is not None:
                sk_val, is_z = kshot_skew_ns_mae(
                    pred_sk_z, true_sk_ns, sk_te_pids, k)
                entry['sk_val']  = sk_val
                entry['sk_is_z'] = is_z

            results[held][k] = entry

    return results


if __name__ == '__main__':
    SKEW_CACHE = os.path.join(HERE, 'caches', 'skew_spatial_cache.pkl')
    MODEL_PATH = os.path.join(HERE, 'saved_models', 'model.pkl')

    print(f"{T()} Loading caches...")
    with open(DEF_CACHE,     'rb') as f: dc = pickle.load(f)
    with open(SAIF_CACHE,    'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE,  'rb') as f: tc_cache = pickle.load(f)
    with open(GRAVITY_CACHE, 'rb') as f: gc = pickle.load(f)
    with open(SKEW_CACHE,    'rb') as f: skc = pickle.load(f)

    with open(MODEL_PATH, 'rb') as f: mdl = pickle.load(f)
    sk_feat_idx = mdl.get('sk_feat_idx', None)   # pruned 15-feature indices

    print(f"{T()} Loading CSV...")
    df = pd.read_csv(os.path.join(DATASET, 'unified_manifest.csv'))
    df = df[df['design_name'].isin(list(T_CLK_NS.keys()))].reset_index(drop=True)
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)

    print(f"{T()} Building features ({len(df)} rows)...")
    X_pw, X_wl, y_pw, y_wl, meta_df = build_all_features(df, dc, sc_cache, tc_cache, gc)
    X_sk, y_sk_setup, _, sk_meta_df  = build_skew_features(df, dc, sc_cache, tc_cache, skc)
    print(f"  Power:{X_pw.shape}  WL:{X_wl.shape}  Skew:{X_sk.shape}  "
          f"sk_feat_idx: {len(sk_feat_idx) if sk_feat_idx is not None else 'all'}")

    print(f"{T()} Running LODO + K-shot...")
    sys.stdout.flush()
    results = run_lodo_kshot(X_pw, X_wl, y_pw, y_wl, meta_df,
                             X_sk=X_sk, y_sk_raw=y_sk_setup,
                             sk_meta_df=sk_meta_df, sk_feat_idx=sk_feat_idx)

    designs   = sorted(results.keys())
    k_header  = "".join(f"   K={k:>2}" for k in K_VALUES)
    sep       = "-" * (12 + 8 * len(K_VALUES))

    print()
    print("=== Power MAPE (%)  [ K=0 = zero-shot ] ===")
    print(f"{'Design':<12}{k_header}")
    print(sep)
    for d in designs:
        print(f"{d:<12}" + "".join(f"  {results[d][k]['pw_mape']:>5.1f}%" for k in K_VALUES))
    means_pw = [np.mean([results[d][k]['pw_mape'] for d in designs]) for k in K_VALUES]
    print(sep)
    print(f"{'Mean':<12}" + "".join(f"  {m:>5.1f}%" for m in means_pw))

    print()
    print("=== WL MAPE (%) ===")
    print(f"{'Design':<12}{k_header}")
    print(sep)
    for d in designs:
        print(f"{d:<12}" + "".join(f"  {results[d][k]['wl_mape']:>5.1f}%" for k in K_VALUES))
    means_wl = [np.mean([results[d][k]['wl_mape'] for d in designs]) for k in K_VALUES]
    print(sep)
    print(f"{'Mean':<12}" + "".join(f"  {m:>5.1f}%" for m in means_wl))

    print()
    print("=== Skew Setup MAE (ns)  [ K>=1: ns using K-run mu/sig est. ] ===")
    print(f"{'Design':<12}{k_header}")
    print(sep)
    for d in designs:
        row = f"{d:<12}"
        for k in K_VALUES:
            v, isz = results[d][k]['sk_val'], results[d][k]['sk_is_z']
            row += f"  {'---':>6} " if (v is None or isz) else f"  {v:>5.4f} "
        print(row)
    print(sep)
    means_sk = []
    for k in K_VALUES:
        vals = [results[d][k]['sk_val'] for d in designs
                if results[d][k]['sk_val'] is not None and not results[d][k]['sk_is_z']]
        means_sk.append(np.nanmean(vals) if vals else None)
    mean_row = f"{'Mean':<12}"
    for m in means_sk:
        mean_row += f"  {'---':>6} " if m is None else f"  {m:>5.4f} "
    print(mean_row)

    print(f"\n{T()} done")
    print("K=0: zero-shot  |  K>=1: first K runs per placement used for calibration")
