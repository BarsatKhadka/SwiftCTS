"""
eval_hold_negation.py — Evaluate new hold skew architecture.

New architecture: pred_hold_z = -pred_setup_z
  skew_hold ≈ -skew_setup at 95.8% global correlation (-0.90 within-placement z-space).
  The residual (hold + setup) has max feature correlation 0.038 — unpredictable noise.

Three evaluations:
  1. LODO     — negation vs current separate-model (4-design LODO)
  2. OOD zipdiv  (1,642 FFs): K-shot calibration on negated setup z-scores
  3. OOD jpeg   (14,606 FFs): same

K-shot for hold:
  Calibrate (a, b) from K hold samples using pred_neg_z = -pred_setup_z as predictor.
  K=0      : a = SIGMA_PRIOR (~0.123 ns), b = 0 → pred_hold = -pred_setup_z * 0.123
  K=1,2    : offset-only, a=prior, b = mean(true_hold) - a*mean(pred_neg_z)
  K>=3     : regularized OLS on (pred_neg_z, 1) → (a, b)
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_skew import build_features, per_placement_z, build_sk_features_row
from prune_skew_eval import LGB_PARAMS

DEF_CACHE    = os.path.join(HERE, 'caches', 'def_cache.pkl')
SAIF_CACHE   = os.path.join(HERE, 'caches', 'saif_cache.pkl')
TIMING_CACHE = os.path.join(HERE, 'caches', 'timing_cache.pkl')
SKEW_CACHE   = os.path.join(HERE, 'caches', 'skew_spatial_cache.pkl')
MANIFEST     = os.path.join(HERE, 'data', 'unified_manifest.csv')
PLACEMENT_DIR = os.path.join(HERE, 'dataset_with_def', 'placement_files')
MODEL_PATH   = os.path.join(HERE, 'saved_models', 'model.pkl')

SIGMA_PRIOR = 0.123   # ns — prior for sig_hold (same as setup)
K_VALUES    = [0, 1, 2, 5]


# ── Calibration ───────────────────────────────────────────────────────────────

def kshot_hold(pred_neg_z_cal, true_hold_cal, K):
    """Fit (a, b): true_hold ≈ a * pred_neg_z + b from K samples."""
    if K == 0:
        return SIGMA_PRIOR, 0.0
    pz = pred_neg_z_cal[:K]; tn = true_hold_cal[:K]
    if K <= 2:
        a = SIGMA_PRIOR
        b = tn.mean() - a * pz.mean()
        return a, b
    pz_c = pz - pz.mean(); tn_c = tn - tn.mean()
    a_ols = (pz_c * tn_c).sum() / ((pz_c**2).sum() + 1e-9)
    blend = 3.0 / K
    a = (1 - blend) * a_ols + blend * SIGMA_PRIOR
    a = max(a, 0.001)
    b = tn.mean() - a * pz.mean()
    return a, b


# ── LODO ──────────────────────────────────────────────────────────────────────

def lodo_negation(X, y_setup, y_hold, meta_df, sk_feat_idx):
    """Retrain setup LGB per LODO fold. Negate pred_setup_z for hold."""
    designs = sorted(meta_df['design_name'].unique())
    z_s, _,    _     = per_placement_z(y_setup, meta_df)
    z_h, mu_h, sig_h = per_placement_z(y_hold,  meta_df)
    Xp = X[:, sk_feat_idx]
    out = {}
    for held in designs:
        tr = (meta_df['design_name'] != held).values
        te = (meta_df['design_name'] == held).values
        sc = StandardScaler()
        lgb = LGBMRegressor(**LGB_PARAMS)
        lgb.fit(sc.fit_transform(Xp[tr]), z_s[tr])
        pred_neg_z   = -lgb.predict(sc.transform(Xp[te]))
        pred_hold_ns = pred_neg_z * sig_h[te] + mu_h[te]
        out[held] = float(np.mean(np.abs(pred_hold_ns - y_hold[te])))
    return out


def lodo_separate(X, y_hold, meta_df, skh_feat_idx):
    """Current separate hold model (baseline)."""
    designs = sorted(meta_df['design_name'].unique())
    z_h, mu_h, sig_h = per_placement_z(y_hold, meta_df)
    Xp = X[:, skh_feat_idx]
    out = {}
    for held in designs:
        tr = (meta_df['design_name'] != held).values
        te = (meta_df['design_name'] == held).values
        sc = StandardScaler()
        lgb = LGBMRegressor(**LGB_PARAMS)
        lgb.fit(sc.fit_transform(Xp[tr]), z_h[tr])
        pred_z   = lgb.predict(sc.transform(Xp[te]))
        pred_ns  = pred_z * sig_h[te] + mu_h[te]
        out[held] = float(np.mean(np.abs(pred_ns - y_hold[te])))
    return out


# ── OOD parsing ───────────────────────────────────────────────────────────────

def parse_ood(pid, design, df_rows, dc, sc_cache, tc_cache, skc):
    """Build skew features for one OOD placement. Parse from disk if not cached."""
    if pid not in dc:
        from swiftcts import _parse_def, _parse_saif, _parse_timing
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
        except Exception as e:
            skc[pid] = {}

    d = dc[pid]; s = sc_cache[pid]; t = tc_cache[pid]; sk = skc.get(pid, {})
    rows = []
    for _, row in df_rows.iterrows():
        rows.append(build_sk_features_row(d, s, t, sk,
            row['cts_cluster_dia'], row['cts_cluster_size'],
            row['cts_max_wire'],    row['cts_buf_dist']))
    X = np.array(rows, dtype=np.float64)
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0
    return X, df_rows['skew_setup'].values, df_rows['skew_hold'].values


def eval_ood_negation(design, ood_csv, dc, sc_cache, tc_cache, skc,
                      setup_lgb, setup_sc, sk_feat_idx):
    """
    OOD evaluation using negation architecture.
    cal_pid: first placement — K samples for calibrating (a, b)
    test_pid: second placement — predict and measure MAE
    """
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
        return {}

    # Predict setup z-scores → negate for hold
    pz_cal  = setup_lgb.predict(setup_sc.transform(X_cal[:,  sk_feat_idx]))
    pz_test = setup_lgb.predict(setup_sc.transform(X_test[:, sk_feat_idx]))
    neg_cal  = -pz_cal
    neg_test = -pz_test

    results = {}
    for K in K_VALUES:
        a, b = kshot_hold(neg_cal, th_cal, K)
        pred_hold_ns = neg_test * a + b
        mae = float(np.mean(np.abs(pred_hold_ns - th_test)))
        results[K] = mae

    return results


if __name__ == '__main__':
    print("=" * 65)
    print("  Hold Skew — Negation Architecture: Full Evaluation")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────────
    with open(DEF_CACHE,    'rb') as f: dc       = pickle.load(f)
    with open(SAIF_CACHE,   'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE, 'rb') as f: tc       = pickle.load(f)
    with open(SKEW_CACHE,   'rb') as f: skc      = pickle.load(f)
    with open(MODEL_PATH,   'rb') as f: mdl      = pickle.load(f)

    df   = pd.read_csv(MANIFEST)
    df4  = df[df['design_name'].isin(['aes','ethmac','picorv32','sha256'])]
    print(f"  Training samples: {len(df4)}")

    X, y_setup, y_hold, meta = build_features(df4, dc, sc_cache, tc, skc)

    sk_feat_idx  = mdl['sk_feat_idx']
    skh_feat_idx = mdl['skh_feat_idx']

    # ── [1] LODO ──────────────────────────────────────────────────────────────
    print("\n[1] LODO — Negation vs Separate Model")
    print("-" * 55)
    neg_lodo = lodo_negation(X, y_setup, y_hold, meta, sk_feat_idx)
    sep_lodo = lodo_separate(X, y_hold, meta, skh_feat_idx)

    print(f"  {'Design':<12}  {'Negation':>12}  {'Separate':>12}  {'Δ (neg-sep)':>12}")
    print(f"  {'-'*52}")
    for d in sorted(neg_lodo):
        n = neg_lodo[d]; s = sep_lodo[d]
        ok_n = '✓' if n < 0.10 else '✗'
        ok_s = '✓' if s < 0.10 else '✗'
        print(f"  {d:<12}  {n:>10.4f}ns{ok_n}  {s:>10.4f}ns{ok_s}  {n-s:>+10.4f}ns")
    m_n = np.mean(list(neg_lodo.values()))
    m_s = np.mean(list(sep_lodo.values()))
    print(f"  {'-'*52}")
    ok_n = '✓' if m_n < 0.10 else '✗'
    ok_s = '✓' if m_s < 0.10 else '✗'
    print(f"  {'Mean':<12}  {m_n:>10.4f}ns{ok_n}  {m_s:>10.4f}ns{ok_s}  {m_n-m_s:>+10.4f}ns")

    # ── Train final setup model on all 4 designs ───────────────────────────────
    print("\n  Training final setup model on all 4 designs for OOD...")
    z_s, _, _ = per_placement_z(y_setup, meta)
    Xp = X[:, sk_feat_idx]
    sc_final = StandardScaler()
    lgb_final = LGBMRegressor(**LGB_PARAMS)
    lgb_final.fit(sc_final.fit_transform(Xp), z_s)

    # ── [2] OOD zipdiv ────────────────────────────────────────────────────────
    print("\n[2] OOD — zipdiv (1,642 FFs)")
    print("-" * 55)
    df_zip = pd.read_csv(os.path.join(HERE, 'data', 'zipdiv.csv'))
    print(f"  Samples: {len(df_zip)}")
    zip_results = eval_ood_negation(
        'zipdiv', df_zip, dc, sc_cache, tc, skc,
        lgb_final, sc_final, sk_feat_idx)

    # Also run current separate model for comparison
    hold_lgb = mdl['model_skew_hold']
    hold_sc  = mdl['scaler_skew_hold']
    pids_zip = df_zip['placement_id'].unique()
    cal_pid_z, test_pid_z = pids_zip[0], pids_zip[1]
    # Use cached parse from above
    X_cal_z, _, th_cal_z = parse_ood(cal_pid_z, 'zipdiv',
        df_zip[df_zip['placement_id']==cal_pid_z], dc, sc_cache, tc, skc)
    X_test_z, _, th_test_z = parse_ood(test_pid_z, 'zipdiv',
        df_zip[df_zip['placement_id']==test_pid_z], dc, sc_cache, tc, skc)
    sep_zip = {}
    if X_cal_z is not None:
        from eval_skew_ood import kshot_calibrate
        pz_cal_sep  = hold_lgb.predict(hold_sc.transform(X_cal_z[:, skh_feat_idx]))
        pz_test_sep = hold_lgb.predict(hold_sc.transform(X_test_z[:, skh_feat_idx]))
        for K in K_VALUES:
            a, b = kshot_calibrate(pz_cal_sep, th_cal_z, K)
            mae = float(np.mean(np.abs(a * pz_test_sep + b - th_test_z)))
            sep_zip[K] = mae

    print(f"\n  {'K':>3}  {'Negation':>12}  {'Separate':>12}  {'Δ':>10}")
    print(f"  {'-'*42}")
    for K in K_VALUES:
        n = zip_results.get(K, float('nan'))
        s = sep_zip.get(K, float('nan'))
        ok_n = '✓' if n < 0.10 else '✗'
        ok_s = '✓' if s < 0.10 else '✗'
        print(f"  K={K}  {n:>10.4f}ns{ok_n}  {s:>10.4f}ns{ok_s}  {n-s:>+8.4f}ns")

    # ── [3] OOD jpeg ──────────────────────────────────────────────────────────
    print("\n[3] OOD — oc_jpegencode (14,606 FFs)")
    print("-" * 55)
    df_jpeg = pd.read_csv(os.path.join(HERE, 'data', 'oc_jpegencode.csv'))
    print(f"  Samples: {len(df_jpeg)}")
    jpeg_results = eval_ood_negation(
        'oc_jpegencode', df_jpeg, dc, sc_cache, tc, skc,
        lgb_final, sc_final, sk_feat_idx)

    pids_jpeg = df_jpeg['placement_id'].unique()
    cal_pid_j, test_pid_j = pids_jpeg[0], pids_jpeg[1]
    X_cal_j, _, th_cal_j = parse_ood(cal_pid_j, 'oc_jpegencode',
        df_jpeg[df_jpeg['placement_id']==cal_pid_j], dc, sc_cache, tc, skc)
    X_test_j, _, th_test_j = parse_ood(test_pid_j, 'oc_jpegencode',
        df_jpeg[df_jpeg['placement_id']==test_pid_j], dc, sc_cache, tc, skc)
    sep_jpeg = {}
    if X_cal_j is not None:
        pz_cal_sep  = hold_lgb.predict(hold_sc.transform(X_cal_j[:, skh_feat_idx]))
        pz_test_sep = hold_lgb.predict(hold_sc.transform(X_test_j[:, skh_feat_idx]))
        for K in K_VALUES:
            a, b = kshot_calibrate(pz_cal_sep, th_cal_j, K)
            mae = float(np.mean(np.abs(a * pz_test_sep + b - th_test_j)))
            sep_jpeg[K] = mae

    print(f"\n  {'K':>3}  {'Negation':>12}  {'Separate':>12}  {'Δ':>10}")
    print(f"  {'-'*42}")
    for K in K_VALUES:
        n = jpeg_results.get(K, float('nan'))
        s = sep_jpeg.get(K, float('nan'))
        ok_n = '✓' if n < 0.10 else '✗'
        ok_s = '✓' if s < 0.10 else '✗'
        print(f"  K={K}  {n:>10.4f}ns{ok_n}  {s:>10.4f}ns{ok_s}  {n-s:>+8.4f}ns")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  LODO mean:  negation={m_n:.4f}ns  separate={m_s:.4f}ns")
    print(f"\n  OOD zipdiv (K=1):   negation={zip_results.get(1,0):.4f}ns  "
          f"separate={sep_zip.get(1,0):.4f}ns")
    print(f"  OOD jpeg   (K=2):   negation={jpeg_results.get(2,0):.4f}ns  "
          f"separate={sep_jpeg.get(2,0):.4f}ns")
    print(f"\n  Architecture change: pred_hold_z = -pred_setup_z")
    print(f"  Removed: model_skew_hold (18-feat LGB), scaler_skew_hold, skh_feat_idx")
    print(f"  Reuses:  sk_feat_idx (15-feat setup model) — zero extra cost")
