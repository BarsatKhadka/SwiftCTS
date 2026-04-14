"""
train_hold_delta.py — Train and evaluate the hold skew delta architecture.

Architecture:
  pred_hold_z = -pred_setup_z + pred_delta_z
  pred_delta_z = delta_lgb(fast_path_features + knobs + interactions)

Where:
  delta_z = z_hold + z_setup   (residual between hold and negated setup)

Fast-path features (from fast_path_cache.pkl):
  Spatial distribution of top-50 highest-slack timing paths.
  These are hold-relevant: fast paths are closest to hold violations.
  Key features: frac_self_loop, slack_max, fast_min_dist, fast_density_ratio.

Runs:
  [0] Feature correlation check: fast-path features vs delta_z
  [1] LODO: negation-only vs negation+delta vs separate-model
  [2] OOD zipdiv K-shot
  [3] OOD jpeg K-shot
  [4] Feature importance of delta model
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # build_skew_cache.py is co-located

from eval_skew import build_features, per_placement_z, build_sk_features_row

DEF_CACHE   = os.path.join(HERE, 'caches', 'def_cache.pkl')
SAIF_CACHE  = os.path.join(HERE, 'caches', 'saif_cache.pkl')
TIMING_CACHE= os.path.join(HERE, 'caches', 'timing_cache.pkl')
SKEW_CACHE  = os.path.join(HERE, 'caches', 'skew_spatial_cache.pkl')
FAST_CACHE  = os.path.join(HERE, 'caches', 'fast_path_cache.pkl')
MANIFEST    = os.path.join(HERE, 'data', 'unified_manifest.csv')
MODEL_PATH  = os.path.join(HERE, 'saved_models', 'model.pkl')
PLACEMENT_DIR = os.path.join(HERE, '..', 'dataset_with_def', 'placement_files')

SIGMA_PRIOR = 0.123
K_VALUES    = [0, 1, 2, 5]

SETUP_LGB_PARAMS = dict(n_estimators=300, num_leaves=31, learning_rate=0.03,
                        min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)

# Delta model: more regularized — target is noisier (std≈0.45 vs ≈1.0)
DELTA_LGB_PARAMS = dict(n_estimators=200, num_leaves=15, learning_rate=0.02,
                        min_child_samples=20, reg_lambda=1.0, reg_alpha=0.5,
                        verbose=-1, n_jobs=1, random_state=42)

FAST_FEAT_NAMES = [
    'slack_max', 'slack_p90', 'slack_p75', 'slack_spread',
    'frac_self_loop', 'frac_fast_1ns',
    'fast_mean_dist', 'fast_min_dist', 'fast_p10_dist', 'fast_max_dist',
    'fast_mean_dist_um', 'fast_min_dist_um',
    'fast_hpwl', 'fast_cx_offset', 'fast_cy_offset',
    'fast_x_std', 'fast_y_std', 'fast_eccentricity', 'fast_asymmetry',
    'fast_star_degree', 'fast_chain_frac', 'fast_density_ratio',
]


def build_delta_features(meta_df, dc, fast_cache, df_rows):
    """
    Build delta model features for each row: fast-path features + knobs + interactions.
    meta_df rows align 1:1 with X (filtered by build_features).
    df_rows is the full manifest — we match by joining on pid + knob values.
    Returns X_delta (N, n_delta_feats).
    """
    # Create a fast lookup: pid → list of manifest rows (for knob retrieval)
    df_rows = df_rows.reset_index(drop=True)
    pid_to_rows = {}
    for _, row in df_rows.iterrows():
        pid_to_rows.setdefault(row['placement_id'], []).append(row)
    # Iterator per pid to consume rows in order
    pid_iter = {pid: iter(rows) for pid, rows in pid_to_rows.items()}

    # Support both meta_df format (has 'pid') and raw manifest format (has 'placement_id')
    pid_col = 'pid' if 'pid' in meta_df.columns else 'placement_id'

    rows_out = []
    for _, mrow in meta_df.iterrows():
        pid = mrow[pid_col]
        # Get next manifest row for this pid (preserves CTS knob ordering)
        try:
            row = next(pid_iter[pid])
        except (KeyError, StopIteration):
            rows_out.append([0.0] * (len(FAST_FEAT_NAMES) + 16))
            continue
        fpc = fast_cache.get(pid, {})
        d   = dc.get(pid, {})
        nf  = d.get('n_ff', 1000)
        sp  = d.get('ff_spacing', 1.0)
        da  = d.get('die_area', 1.0)

        cd = row['cts_cluster_dia']; cs = row['cts_cluster_size']
        mw = row['cts_max_wire'];    bd = row['cts_buf_dist']

        # 22 fast-path features
        fp = [fpc.get(k, 0.0) for k in FAST_FEAT_NAMES]

        # 8 knob features
        kn = [np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
              cd, cs, mw, bd]

        # Interaction features: knob × hold-relevant geometry
        fsl = fpc.get('frac_self_loop', 0.0)
        fmd = fpc.get('fast_min_dist_um', 0.0)
        fp10 = fpc.get('fast_p10_dist', 0.0)
        smax = fpc.get('slack_max', 0.0)
        fhpwl = fpc.get('fast_hpwl', 0.0)
        fdens = fpc.get('fast_density_ratio', 1.0)

        interactions = [
            fsl * cs,                          # self-loops per cluster → hold risk × grouping
            fmd / (cd + 1),                    # nearest fast pair vs cluster dia
            fp10 * cd,                         # cluster dia vs typical fast path
            np.log1p(smax * mw),               # max slack × max wire (budgets)
            fhpwl / (cs + 1),                  # fast path extent per cluster
            fsl * np.log1p(nf / (cs + 1)),     # self-loops × FF-per-cluster
            fdens * cs,                        # fast FF density × cluster size
            np.log1p(fmd * cd),                # log of nearest pair × cluster dia
        ]

        rows_out.append(fp + kn + interactions)

    return np.array(rows_out, dtype=np.float64)


# ── LODO ──────────────────────────────────────────────────────────────────────

def lodo_full(X_sk, y_setup, y_hold, meta_df, sk_feat_idx, dc, fast_cache, df_rows):
    """LODO with negation + delta. Also returns negation-only for comparison."""
    designs = sorted(meta_df['design_name'].unique())
    z_s, _,    _     = per_placement_z(y_setup, meta_df)
    z_h, mu_h, sig_h = per_placement_z(y_hold,  meta_df)
    delta_z           = z_h + z_s  # residual target

    X_delta = build_delta_features(meta_df, dc, fast_cache, df_rows)

    Xp = X_sk[:, sk_feat_idx]

    results_neg   = {}
    results_delta = {}

    for held in designs:
        tr = (meta_df['design_name'] != held).values
        te = (meta_df['design_name'] == held).values

        # Setup model
        sc_s = StandardScaler()
        lgb_s = LGBMRegressor(**SETUP_LGB_PARAMS)
        lgb_s.fit(sc_s.fit_transform(Xp[tr]), z_s[tr])
        pred_setup_z = lgb_s.predict(sc_s.transform(Xp[te]))

        # Negation-only
        pred_hold_neg = (-pred_setup_z) * sig_h[te] + mu_h[te]
        results_neg[held] = float(np.mean(np.abs(pred_hold_neg - y_hold[te])))

        # Delta model
        sc_d = StandardScaler()
        lgb_d = LGBMRegressor(**DELTA_LGB_PARAMS)
        # Clean NaN/inf in delta features
        Xd_tr = X_delta[tr].copy()
        Xd_te = X_delta[te].copy()
        for c in range(Xd_tr.shape[1]):
            bad = ~np.isfinite(Xd_tr[:, c])
            if bad.any():
                Xd_tr[bad, c] = np.nanmedian(Xd_tr[~bad, c]) if (~bad).any() else 0.0
            bad2 = ~np.isfinite(Xd_te[:, c])
            if bad2.any():
                Xd_te[bad2, c] = 0.0

        lgb_d.fit(sc_d.fit_transform(Xd_tr), delta_z[tr])
        pred_delta_z = lgb_d.predict(sc_d.transform(Xd_te))

        pred_hold_z  = -pred_setup_z + pred_delta_z
        pred_hold_ns = pred_hold_z * sig_h[te] + mu_h[te]
        results_delta[held] = float(np.mean(np.abs(pred_hold_ns - y_hold[te])))

    return results_neg, results_delta


# ── OOD ───────────────────────────────────────────────────────────────────────

def parse_ood(pid, design, df_rows, dc, sc_cache, tc_cache, skc):
    from swiftcts import _parse_def, _parse_saif, _parse_timing
    if pid not in dc:
        def_path  = os.path.join(PLACEMENT_DIR, pid, f'{design}.def')
        saif_path = os.path.join(PLACEMENT_DIR, pid, f'{design}.saif')
        tim_path  = os.path.join(PLACEMENT_DIR, pid, 'timing_paths.csv')
        try:
            dc[pid]       = _parse_def(def_path)
            sc_cache[pid] = _parse_saif(saif_path)
            tc_cache[pid] = _parse_timing(tim_path)
        except Exception as e:
            print(f"    Parse error {pid}: {e}"); return None, None, None
    if pid not in skc:
        try:
            from build_skew_cache import parse_def_ff_positions, compute_skew_features
            def_path = os.path.join(PLACEMENT_DIR, pid, f'{design}.def')
            tim_path = os.path.join(PLACEMENT_DIR, pid, 'timing_paths.csv')
            ff_pos, dw, dh, origin = parse_def_ff_positions(def_path)
            td = pd.read_csv(tim_path)
            skc[pid] = compute_skew_features(ff_pos, dw, dh, origin, td) or {}
        except Exception:
            skc[pid] = {}

    d = dc[pid]; s = sc_cache[pid]; t = tc_cache[pid]; sk = skc.get(pid, {})
    rows = []
    for _, row in df_rows.iterrows():
        rows.append(build_sk_features_row(d, s, t, sk,
            row['cts_cluster_dia'], row['cts_cluster_size'],
            row['cts_max_wire'], row['cts_buf_dist']))
    X = np.array(rows, dtype=np.float64)
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0
    return X, df_rows['skew_setup'].values, df_rows['skew_hold'].values


def kshot_hold(neg_cal, true_hold_cal, K):
    if K == 0:
        return SIGMA_PRIOR, 0.0
    pz = neg_cal[:K]; tn = true_hold_cal[:K]
    if K <= 2:
        a = SIGMA_PRIOR; b = tn.mean() - a * pz.mean(); return a, b
    pz_c = pz - pz.mean(); tn_c = tn - tn.mean()
    a_ols = (pz_c * tn_c).sum() / ((pz_c**2).sum() + 1e-9)
    a = (1 - 3.0/K) * a_ols + (3.0/K) * SIGMA_PRIOR
    a = max(a, 0.001)
    b = tn.mean() - a * pz.mean()
    return a, b


def eval_ood(design, ood_csv_path, dc, sc_cache, tc_cache, skc, fast_cache,
             setup_lgb, setup_sc, sk_feat_idx,
             delta_lgb, delta_sc):
    ood_csv = pd.read_csv(ood_csv_path)
    pids = ood_csv['placement_id'].unique()
    cal_pid, test_pid = pids[0], pids[1]

    print(f"    Parsing {cal_pid}...")
    X_cal, _, th_cal = parse_ood(cal_pid, design,
                                  ood_csv[ood_csv['placement_id']==cal_pid],
                                  dc, sc_cache, tc_cache, skc)
    print(f"    Parsing {test_pid}...")
    X_test, _, th_test = parse_ood(test_pid, design,
                                    ood_csv[ood_csv['placement_id']==test_pid],
                                    dc, sc_cache, tc_cache, skc)
    if X_cal is None or X_test is None:
        return {}, {}

    # Setup predictions → negated for hold
    pz_cal  = setup_lgb.predict(setup_sc.transform(X_cal[:,  sk_feat_idx]))
    pz_test = setup_lgb.predict(setup_sc.transform(X_test[:, sk_feat_idx]))
    neg_cal  = -pz_cal
    neg_test = -pz_test

    # Delta predictions
    df_cal  = ood_csv[ood_csv['placement_id']==cal_pid].reset_index(drop=True)
    df_test = ood_csv[ood_csv['placement_id']==test_pid].reset_index(drop=True)
    # Build delta features using the already-parsed dc
    Xd_cal  = build_delta_features(df_cal,  dc, fast_cache, df_cal)
    Xd_test = build_delta_features(df_test, dc, fast_cache, df_test)
    for X in [Xd_cal, Xd_test]:
        for c in range(X.shape[1]):
            bad = ~np.isfinite(X[:, c])
            if bad.any():
                X[bad, c] = 0.0
    delta_cal  = delta_lgb.predict(delta_sc.transform(Xd_cal))
    delta_test = delta_lgb.predict(delta_sc.transform(Xd_test))

    neg_only = {}
    neg_delta = {}

    for K in K_VALUES:
        # Negation-only
        a, b = kshot_hold(neg_cal, th_cal, K)
        mae_neg = float(np.mean(np.abs(neg_test * a + b - th_test)))
        neg_only[K] = mae_neg

        # Negation + delta
        # Calibrate: true_hold = a*(neg_z + delta_z) + b
        pred_combined_cal  = neg_cal  + delta_cal
        pred_combined_test = neg_test + delta_test
        a2, b2 = kshot_hold(pred_combined_cal, th_cal, K)
        mae_delta = float(np.mean(np.abs(pred_combined_test * a2 + b2 - th_test)))
        neg_delta[K] = mae_delta

    return neg_only, neg_delta


if __name__ == '__main__':
    print("=" * 68)
    print("  Hold Skew — Negation + Delta Architecture")
    print("=" * 68)

    # ── Load ──────────────────────────────────────────────────────────────────
    with open(DEF_CACHE,    'rb') as f: dc       = pickle.load(f)
    with open(SAIF_CACHE,   'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE, 'rb') as f: tc       = pickle.load(f)
    with open(SKEW_CACHE,   'rb') as f: skc      = pickle.load(f)
    with open(FAST_CACHE,   'rb') as f: fpc      = pickle.load(f)
    with open(MODEL_PATH,   'rb') as f: mdl      = pickle.load(f)

    df   = pd.read_csv(MANIFEST)
    df4  = df[df['design_name'].isin(['aes','ethmac','picorv32','sha256'])]
    print(f"  Training samples: {len(df4)}")

    X, y_setup, y_hold, meta = build_features(df4, dc, sc_cache, tc, skc)
    sk_feat_idx = mdl['sk_feat_idx']

    z_s, _, _     = per_placement_z(y_setup, meta)
    z_h, mu_h, sig_h = per_placement_z(y_hold,  meta)
    delta_z = z_h + z_s

    # ── [0] Feature correlation check ─────────────────────────────────────────
    print("\n[0] Feature correlation with delta_z = z_hold + z_setup")
    print("-" * 60)
    print(f"  delta_z stats: mean={delta_z.mean():.4f}  std={delta_z.std():.4f}")
    print()

    # Build delta features for all training rows
    X_delta = build_delta_features(meta, dc, fpc, df4.reset_index(drop=True))
    n_delta_feats = X_delta.shape[1]

    delta_feat_names = FAST_FEAT_NAMES + [
        'log_cd','log_cs','log_mw','log_bd','cd','cs','mw','bd',
        'fsl*cs','fmd/cd','fp10*cd','log(smax*mw)','fhpwl/(cs+1)',
        'fsl*log(nff/cs)','fdens*cs','log(fmd*cd)',
    ]

    corrs = []
    for i in range(n_delta_feats):
        col = X_delta[:, i]
        col = col[np.isfinite(col)]
        if len(col) < 100: corrs.append(0.0); continue
        dz_i = delta_z[np.isfinite(X_delta[:, i])]
        r = abs(np.corrcoef(col, dz_i)[0, 1])
        corrs.append(r if np.isfinite(r) else 0.0)

    top_idx = np.argsort(corrs)[::-1][:12]
    print("  Top features correlated with delta_z:")
    for i in top_idx:
        name = delta_feat_names[i] if i < len(delta_feat_names) else f'feat_{i}'
        print(f"    [{i:2d}] {name:<30}  r={corrs[i]:.4f}")
    print(f"\n  Max r (original 63 sk features, for reference): 0.038")
    print(f"  Max r (new fast-path features):                  {max(corrs[:22]):.4f}")
    print(f"  Max r (interactions):                            {max(corrs[30:]):.4f}")

    # ── [1] LODO ──────────────────────────────────────────────────────────────
    print("\n[1] LODO — Negation vs Negation+Delta vs Separate Model")
    print("-" * 60)
    results_neg, results_delta = lodo_full(
        X, y_setup, y_hold, meta, sk_feat_idx, dc, fpc, df4.reset_index(drop=True))

    # Separate model baseline
    from eval_hold_negation import lodo_separate
    sep_lodo = lodo_separate(X, y_hold, meta, mdl['skh_feat_idx'])

    print(f"  {'Design':<12}  {'Neg-only':>11}  {'Neg+Delta':>11}  {'Separate':>11}  {'Δ (delta-neg)':>14}")
    print(f"  {'-'*62}")
    for d in sorted(results_neg):
        n = results_neg[d]; nd = results_delta[d]; s = sep_lodo[d]
        ok_n  = '✓' if n  < 0.10 else '✗'
        ok_nd = '✓' if nd < 0.10 else '✗'
        ok_s  = '✓' if s  < 0.10 else '✗'
        print(f"  {d:<12}  {n:>9.4f}ns{ok_n}  {nd:>9.4f}ns{ok_nd}  "
              f"{s:>9.4f}ns{ok_s}  {nd-n:>+12.4f}ns")
    m_n  = np.mean(list(results_neg.values()))
    m_nd = np.mean(list(results_delta.values()))
    m_s  = np.mean(list(sep_lodo.values()))
    print(f"  {'-'*62}")
    ok_n  = '✓' if m_n  < 0.10 else '✗'
    ok_nd = '✓' if m_nd < 0.10 else '✗'
    ok_s  = '✓' if m_s  < 0.10 else '✗'
    print(f"  {'Mean':<12}  {m_n:>9.4f}ns{ok_n}  {m_nd:>9.4f}ns{ok_nd}  "
          f"{m_s:>9.4f}ns{ok_s}  {m_nd-m_n:>+12.4f}ns")

    # ── Train final models on all 4 designs for OOD ───────────────────────────
    print("\n  Training final models on all 4 designs for OOD...")
    sc_final_s = StandardScaler()
    lgb_final_s = LGBMRegressor(**SETUP_LGB_PARAMS)
    lgb_final_s.fit(sc_final_s.fit_transform(X[:, sk_feat_idx]), z_s)

    X_delta_clean = X_delta.copy()
    for c in range(X_delta_clean.shape[1]):
        bad = ~np.isfinite(X_delta_clean[:, c])
        if bad.any():
            X_delta_clean[bad, c] = np.nanmedian(X_delta_clean[~bad, c]) if (~bad).any() else 0.0
    sc_final_d = StandardScaler()
    lgb_final_d = LGBMRegressor(**DELTA_LGB_PARAMS)
    lgb_final_d.fit(sc_final_d.fit_transform(X_delta_clean), delta_z)
    print("  Done.")

    # ── [2] OOD zipdiv ────────────────────────────────────────────────────────
    print("\n[2] OOD — zipdiv (1,642 FFs)")
    print("-" * 60)
    zip_neg, zip_delta = eval_ood(
        'zipdiv', os.path.join(HERE, 'data', 'zipdiv.csv'),
        dc, sc_cache, tc, skc, fpc,
        lgb_final_s, sc_final_s, sk_feat_idx,
        lgb_final_d, sc_final_d)

    # Separate model OOD comparison
    from eval_hold_negation import parse_ood as parse_ood_h
    from eval_skew_ood import kshot_calibrate
    hold_lgb = mdl['model_skew_hold']; hold_sc = mdl['scaler_skew_hold']
    skh_feat_idx = mdl['skh_feat_idx']
    df_zip = pd.read_csv(os.path.join(HERE, 'data', 'zipdiv.csv'))
    pz_cal_zip, pz_test_zip, th_cal_zip, th_test_zip = None, None, None, None
    pids_zip = df_zip['placement_id'].unique()
    X_cz, _, th_cz = parse_ood_h(pids_zip[0], 'zipdiv',
        df_zip[df_zip['placement_id']==pids_zip[0]], dc, sc_cache, tc, skc)
    X_tz, _, th_tz = parse_ood_h(pids_zip[1], 'zipdiv',
        df_zip[df_zip['placement_id']==pids_zip[1]], dc, sc_cache, tc, skc)
    sep_zip = {}
    if X_cz is not None:
        pz_c = hold_lgb.predict(hold_sc.transform(X_cz[:, skh_feat_idx]))
        pz_t = hold_lgb.predict(hold_sc.transform(X_tz[:, skh_feat_idx]))
        for K in K_VALUES:
            a, b = kshot_calibrate(pz_c, th_cz, K)
            sep_zip[K] = float(np.mean(np.abs(a * pz_t + b - th_tz)))

    print(f"\n  {'K':>3}  {'Neg-only':>11}  {'Neg+Delta':>11}  {'Separate':>11}")
    print(f"  {'-'*42}")
    for K in K_VALUES:
        n = zip_neg.get(K, float('nan'))
        nd = zip_delta.get(K, float('nan'))
        s  = sep_zip.get(K, float('nan'))
        ok_n  = '✓' if n  < 0.10 else '✗'
        ok_nd = '✓' if nd < 0.10 else '✗'
        ok_s  = '✓' if s  < 0.10 else '✗'
        print(f"  K={K}  {n:>9.4f}ns{ok_n}  {nd:>9.4f}ns{ok_nd}  {s:>9.4f}ns{ok_s}")

    # ── [3] OOD jpeg ──────────────────────────────────────────────────────────
    print("\n[3] OOD — oc_jpegencode (14,606 FFs)")
    print("-" * 60)
    jpeg_neg, jpeg_delta = eval_ood(
        'oc_jpegencode', os.path.join(HERE, 'data', 'oc_jpegencode.csv'),
        dc, sc_cache, tc, skc, fpc,
        lgb_final_s, sc_final_s, sk_feat_idx,
        lgb_final_d, sc_final_d)

    df_jpeg = pd.read_csv(os.path.join(HERE, 'data', 'oc_jpegencode.csv'))
    pids_j = df_jpeg['placement_id'].unique()
    X_cj, _, th_cj = parse_ood_h(pids_j[0], 'oc_jpegencode',
        df_jpeg[df_jpeg['placement_id']==pids_j[0]], dc, sc_cache, tc, skc)
    X_tj, _, th_tj = parse_ood_h(pids_j[1], 'oc_jpegencode',
        df_jpeg[df_jpeg['placement_id']==pids_j[1]], dc, sc_cache, tc, skc)
    sep_jpeg = {}
    if X_cj is not None:
        pz_c = hold_lgb.predict(hold_sc.transform(X_cj[:, skh_feat_idx]))
        pz_t = hold_lgb.predict(hold_sc.transform(X_tj[:, skh_feat_idx]))
        for K in K_VALUES:
            a, b = kshot_calibrate(pz_c, th_cj, K)
            sep_jpeg[K] = float(np.mean(np.abs(a * pz_t + b - th_tj)))

    print(f"\n  {'K':>3}  {'Neg-only':>11}  {'Neg+Delta':>11}  {'Separate':>11}")
    print(f"  {'-'*42}")
    for K in K_VALUES:
        n = jpeg_neg.get(K, float('nan'))
        nd = jpeg_delta.get(K, float('nan'))
        s  = sep_jpeg.get(K, float('nan'))
        ok_n  = '✓' if n  < 0.10 else '✗'
        ok_nd = '✓' if nd < 0.10 else '✗'
        ok_s  = '✓' if s  < 0.10 else '✗'
        print(f"  K={K}  {n:>9.4f}ns{ok_n}  {nd:>9.4f}ns{ok_nd}  {s:>9.4f}ns{ok_s}")

    # ── [4] Feature importance ─────────────────────────────────────────────────
    print("\n[4] Delta model feature importance (gain)")
    print("-" * 60)
    imp = lgb_final_d.feature_importances_
    top_i = np.argsort(imp)[::-1][:12]
    total_imp = imp.sum()
    for i in top_i:
        name = delta_feat_names[i] if i < len(delta_feat_names) else f'feat_{i}'
        print(f"  [{i:2d}] {name:<30}  gain={imp[i]:6.0f}  ({100*imp[i]/total_imp:.1f}%)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    print(f"  LODO:  neg={m_n:.4f}  neg+delta={m_nd:.4f}  separate={m_s:.4f}")
    print(f"  zip K=1:  neg={zip_neg.get(1,0):.4f}  neg+delta={zip_delta.get(1,0):.4f}  "
          f"sep={sep_zip.get(1,0):.4f}")
    print(f"  jpeg K=2: neg={jpeg_neg.get(2,0):.4f}  neg+delta={jpeg_delta.get(2,0):.4f}  "
          f"sep={sep_jpeg.get(2,0):.4f}")
