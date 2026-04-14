"""
eval_skew_ood.py — OOD evaluation of skew_setup and skew_hold on zipdiv + jpeg.

Compares three models:
  full-63  : full 63-dim skew feature vector (current deployed model)
  pruned-15: top-15 features by gain (skew_setup)
  pruned-18: top-18 features by gain (skew_hold)

K-shot calibration for skew:
  Predict z-scores with the model (trained on all 4 training designs).
  Convert to absolute ns using (sigma, mu) estimated from K labeled runs
  of the calibration placement via OLS: true_ns_i = a*pred_z_i + b.
    K=0 : a=training_sigma_prior (0.123ns), b=training_mu_prior (0.732ns)
    K=1 : b = true_ns_1 - a*pred_z_1  (offset only, a=prior)
    K>=2: OLS on (pred_z, 1) -> (a, b)

Test placement predictions: pred_ns = a*pred_z + b.
MAE on the 10 test-placement runs.

OOD designs:
  zipdiv        : 1,642 FFs — in-distribution size
  oc_jpegencode : 14,606 FFs — 3x larger than training max
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

from eval_skew import build_sk_features_row, per_placement_z

DEF_CACHE   = os.path.join(HERE, 'caches', 'def_cache.pkl')
SAIF_CACHE  = os.path.join(HERE, 'caches', 'saif_cache.pkl')
TIMING_CACHE= os.path.join(HERE, 'caches', 'timing_cache.pkl')
SKEW_CACHE  = os.path.join(HERE, 'caches', 'skew_spatial_cache.pkl')
MANIFEST    = os.path.join(HERE, 'data', 'unified_manifest.csv')
PLACEMENT_DIR = os.environ.get('SWIFTCTS_PLACEMENT_DIR',
    os.path.join(HERE, 'dataset_with_def', 'placement_files'))

LGB_PARAMS = dict(n_estimators=300, num_leaves=31, learning_rate=0.03,
                  min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)

# Training priors (K=0 fallback)
MU_PRIOR    = 0.732   # mean per-placement skew mean across training designs (ns)
SIGMA_PRIOR = 0.123   # mean per-placement skew std across training designs (ns)


# ── Feature names (for selecting subsets) ────────────────────────────────────
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

T_CLK_NS = {'aes':7.0,'picorv32':5.0,'sha256':9.0,'ethmac':9.0,
            'zipdiv':5.0,'oc_jpegencode':7.0}


def encode_synth(s):
    if pd.isna(s): return 0.5, 2.0, 0.5
    s = str(s).upper()
    sd = 1.0 if 'DELAY' in s else 0.0
    try: level = float(s.split()[-1])
    except: level = 2.0
    return sd, level, sd * level / 4.0


def build_train_features(df_in, dc, sc_cache, tc_cache, skc):
    rows, y_setup, y_hold, meta = [], [], [], []
    for _, row in df_in.iterrows():
        pid = row['placement_id']; design = row['design_name']
        d = dc.get(pid); s = sc_cache.get(pid); t = tc_cache.get(pid)
        sk = skc.get(pid, {})
        if not d or not s or not t: continue
        if not np.isfinite(row['skew_setup']) or not np.isfinite(row['skew_hold']): continue
        cd = row['cts_cluster_dia']; cs = row['cts_cluster_size']
        mw = row['cts_max_wire'];    bd = row['cts_buf_dist']
        rows.append(build_sk_features_row(d, s, t, sk, cd, cs, mw, bd))
        y_setup.append(row['skew_setup'])
        y_hold.append(row['skew_hold'])
        meta.append({'pid': pid, 'design_name': design})
    X = np.array(rows, dtype=np.float64)
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0
    return X, np.array(y_setup), np.array(y_hold), pd.DataFrame(meta)


def train_model(X, y_raw, meta_df, feat_idx):
    """Train LGB on z-scored target, return (model, scaler, ranked_feat_idx)."""
    Xs = X[:, feat_idx]
    y_z, _, _ = per_placement_z(y_raw, meta_df)
    sc = StandardScaler()
    lgb = LGBMRegressor(**LGB_PARAMS)
    lgb.fit(sc.fit_transform(Xs), y_z)
    return lgb, sc


def parse_ood_placement(pid, design, df_rows, dc, sc_cache, tc_cache, skc):
    """Build 63-dim skew features for one OOD placement's 10 runs."""
    # Parse DEF/SAIF/timing if not cached
    if pid not in dc:
        sample = df_rows.iloc[0]
        def_path = os.path.join(PLACEMENT_DIR, pid, f'{design}.def')
        saif_path = os.path.join(PLACEMENT_DIR, pid, f'{design}.saif')
        tim_path  = sample.get('timing_path_csv', '')
        if not os.path.exists(def_path):
            # Try direct path from CSV
            def_path = sample.get('def_path', def_path)
            saif_path = sample.get('saif_path', saif_path)

        from swiftcts import _parse_def, _parse_saif, _parse_timing
        try:
            dc[pid]       = _parse_def(def_path)
            sc_cache[pid] = _parse_saif(saif_path)
            tc_cache[pid] = _parse_timing(tim_path if tim_path else
                                           os.path.join(PLACEMENT_DIR, pid, 'timing_paths.csv'))
        except Exception as e:
            print(f"    Parse error for {pid}: {e}")
            return None, None, None

    # Parse skew spatial (on-the-fly if not in cache)
    if pid not in skc:
        sample = df_rows.iloc[0]
        def_path = os.path.join(PLACEMENT_DIR, pid, f'{design}.def')
        tim_path  = sample.get('timing_path_csv',
                    os.path.join(PLACEMENT_DIR, pid, 'timing_paths.csv'))
        if not os.path.exists(def_path):
            def_path = sample.get('def_path', def_path)
        try:
            from build_skew_cache import parse_def_ff_positions, compute_skew_features
            ff_pos, dw, dh, origin = parse_def_ff_positions(def_path)
            td = pd.read_csv(tim_path)
            skc[pid] = compute_skew_features(ff_pos, dw, dh, origin, td) or {}
            print(f"    Parsed skew spatial for {pid}: {len(skc[pid])} features")
        except Exception as e:
            print(f"    Skew spatial parse error for {pid}: {e}")
            skc[pid] = {}

    d = dc[pid]; s = sc_cache[pid]; t = tc_cache[pid]; sk = skc.get(pid, {})
    rows = []
    for _, row in df_rows.iterrows():
        cd = row['cts_cluster_dia']; cs = row['cts_cluster_size']
        mw = row['cts_max_wire'];    bd = row['cts_buf_dist']
        rows.append(build_sk_features_row(d, s, t, sk, cd, cs, mw, bd))
    X = np.array(rows, dtype=np.float64)
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0
    true_setup = df_rows['skew_setup'].values
    true_hold  = df_rows['skew_hold'].values
    return X, true_setup, true_hold


def kshot_calibrate(pred_z_cal, true_ns_cal, K):
    """Estimate (a=sigma, b=mu) from K labeled cal-placement runs.

    K=0      : training priors only
    K=1,2    : offset-only (b = mean(true) - a*mean(pred_z), a = prior)
               OLS is unstable with <=2 points (perfect fit → bad transfer)
    K>=3     : full OLS (a, b) with prior-anchored regularization
    """
    if K == 0:
        return SIGMA_PRIOR, MU_PRIOR
    pz = pred_z_cal[:K]; tn = true_ns_cal[:K]
    if K <= 2:
        # Offset-only: trust the prior scale, just shift to match mean
        a = SIGMA_PRIOR
        b = tn.mean() - a * pz.mean()
        return a, b
    # K>=3: OLS with soft regularization toward prior scale
    pz_c = pz - pz.mean(); tn_c = tn - tn.mean()
    var_pz = (pz_c ** 2).sum()
    a_ols = (pz_c * tn_c).sum() / (var_pz + 1e-9)
    # Blend toward prior with weight that decays as K grows
    blend = 3.0 / K          # K=3→1.0, K=5→0.6, K=10→0.3
    a = (1 - blend) * a_ols + blend * SIGMA_PRIOR
    a = max(a, 0.001)
    b = tn.mean() - a * pz.mean()
    return a, b


def eval_ood_design(design, ood_csv, dc, sc_cache, tc_cache, skc,
                    models_setup, models_hold, K_values=(0,1,2,5)):
    """Evaluate all models on one OOD design."""
    pids = ood_csv['placement_id'].unique()
    cal_pid, test_pid = pids[0], pids[1]

    print(f"  Parsing {cal_pid}...")
    X_cal, ts_cal, th_cal = parse_ood_placement(
        cal_pid, design, ood_csv[ood_csv['placement_id']==cal_pid],
        dc, sc_cache, tc_cache, skc)
    print(f"  Parsing {test_pid}...")
    X_test, ts_test, th_test = parse_ood_placement(
        test_pid, design, ood_csv[ood_csv['placement_id']==test_pid],
        dc, sc_cache, tc_cache, skc)

    if X_cal is None or X_test is None:
        print("  ERROR: could not parse placements")
        return

    results = {}
    for tag, models, true_cal, true_test in [
        ('setup', models_setup, ts_cal, ts_test),
        ('hold',  models_hold,  th_cal, th_test),
    ]:
        results[tag] = {}
        for mname, (lgb, sc_model, feat_idx) in models.items():
            row_results = {}
            Xc = X_cal[:,  feat_idx]
            Xt = X_test[:, feat_idx]
            pz_cal  = lgb.predict(sc_model.transform(Xc))
            pz_test = lgb.predict(sc_model.transform(Xt))
            for K in K_values:
                a, b = kshot_calibrate(pz_cal, true_cal, K)
                pred_ns = a * pz_test + b
                mae = float(np.mean(np.abs(pred_ns - true_test)))
                row_results[K] = mae
            results[tag][mname] = row_results

    return results


if __name__ == '__main__':
    print("=" * 65)
    print("  SwiftCTS — Skew OOD Evaluation: zipdiv + jpeg")
    print("=" * 65)

    # ── Load training caches ──────────────────────────────────────────────
    with open(DEF_CACHE,   'rb') as f: dc = pickle.load(f)
    with open(SAIF_CACHE,  'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE,'rb') as f: tc_cache = pickle.load(f)
    with open(SKEW_CACHE,  'rb') as f: skc = pickle.load(f)

    df_train = pd.read_csv(MANIFEST)
    df_train = df_train[df_train['design_name'].isin(['aes','ethmac','picorv32','sha256'])]

    print(f"\nBuilding training features ({len(df_train)} rows)...")
    X_tr, y_setup_tr, y_hold_tr, meta_tr = build_train_features(
        df_train, dc, sc_cache, tc_cache, skc)

    # ── Get pruned feature rankings ───────────────────────────────────────
    from prune_skew_eval import get_gain_ranking
    y_z_setup, _, _ = per_placement_z(y_setup_tr, meta_tr)
    y_z_hold,  _, _ = per_placement_z(y_hold_tr,  meta_tr)
    ranked_setup, _ = get_gain_ranking(X_tr, y_z_setup)
    ranked_hold,  _ = get_gain_ranking(X_tr, y_z_hold)

    all63 = np.arange(63)

    print("\nTraining models on all 4 designs...")
    models_setup = {
        'full-63':   (*train_model(X_tr, y_setup_tr, meta_tr, all63),          all63),
        'pruned-15': (*train_model(X_tr, y_setup_tr, meta_tr, ranked_setup[:15]), ranked_setup[:15]),
    }
    models_hold = {
        'full-63':   (*train_model(X_tr, y_hold_tr,  meta_tr, all63),          all63),
        'pruned-18': (*train_model(X_tr, y_hold_tr,  meta_tr, ranked_hold[:18]), ranked_hold[:18]),
    }
    print("  Done.")

    K_VALUES = (0, 1, 2, 5)

    # ── Evaluate on zipdiv and jpeg ───────────────────────────────────────
    for design, csv_path, note in [
        ('zipdiv',       os.path.join(HERE, 'data', 'zipdiv.csv'),
         '1,642 FFs — in-distribution'),
        ('oc_jpegencode', os.path.join(HERE, 'data', 'oc_jpegencode.csv'),
         '14,606 FFs — 3× OOD'),
    ]:
        ood_csv = pd.read_csv(csv_path)
        print(f"\n{'='*65}")
        print(f"  OOD: {design} ({note})")
        print(f"{'='*65}")

        res = eval_ood_design(design, ood_csv, dc, sc_cache, tc_cache, skc,
                              models_setup, models_hold, K_VALUES)
        if res is None:
            continue

        for target in ['setup', 'hold']:
            print(f"\n  skew_{target} (absolute ns MAE):")
            mnames = sorted(res[target].keys())
            header = f"  {'K':>4}  " + "  ".join(f"{m:>12}" for m in mnames)
            print(header)
            print(f"  {'-'*50}")
            for K in K_VALUES:
                row = f"  {K:>4}  "
                for m in mnames:
                    mae = res[target][m][K]
                    ok = ' ✓' if mae < 0.10 else ' ✗'
                    row += f"  {mae:.4f}ns{ok}"
                label = ' (zero-shot)' if K==0 else ''
                print(row + label)

    print(f"\n{'='*65}")
    print("  NOTES:")
    print("  K=0 : use training priors (mu=0.732ns, sigma=0.123ns)")
    print("  K>=1: OLS calibration from K runs of calibration placement")
    print("  Target: MAE < 0.10 ns")
    print("  zipdiv per-placement skew std ~0.005 ns → tiny absolute variation")
    print("  jpeg  per-placement skew std ~0.15-0.25 ns → larger variation")
