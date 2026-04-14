"""
eval_cp_lodo.py — Cross-Placement LODO evaluation.

K = number of calibration PLACEMENTS (each with all 10 runs).
Protocol:
  - Hold out one design (LODO).
  - Train on the other 3 designs.
  - Randomly sample K placements from the held design → compute one global
    multiplicative scale from all K×10 observations.
  - Evaluate on ALL runs of ALL remaining placements (not used for calibration).
  - Repeat N_SEEDS times; report mean ± std across seeds.

K=0 is pure zero-shot (no calibration placements).

Usage: python3 eval_cp_lodo.py
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
from eval_lodo_kshot import (build_all_features, log_space_scale, mape,
                              LGB_PARAMS, WL_ALPHA, SEED, T_CLK_NS)
from eval_skew import build_features as build_skew_features, per_placement_z

K_VALUES  = [0, 1, 2, 5, 10]
N_SEEDS   = 20    # random seeds for calibration placement sampling


def cp_skew_ns_mae(pred_z, true_ns, te_pids, cal_pids):
    """
    Skew MAE (ns) for cross-placement: cal_pids excluded entirely.
    Each remaining placement uses its own runs for mu/sig estimation
    (all runs, since there is no within-placement K contamination here).
    """
    errors = []
    cal_set = set(cal_pids)
    for pid in sorted(set(te_pids)):
        if pid in cal_set:
            continue
        idx    = np.where(te_pids == pid)[0]
        ns_pid = true_ns[idx]
        mu     = float(ns_pid.mean())
        sig    = max(float(ns_pid.std()), max(abs(mu) * 0.01, 1e-4))
        pred_ns = pred_z[idx] * sig + mu
        errors.extend(np.abs(pred_ns - ns_pid).tolist())
    if not errors:
        return None
    return float(np.mean(errors))


def run_cp_lodo(X_pw, X_wl, y_pw, y_wl, meta_df,
                X_sk=None, y_sk_raw=None, sk_meta_df=None, sk_feat_idx=None,
                k_values=K_VALUES, n_seeds=N_SEEDS, seed=SEED):
    rng     = np.random.default_rng(seed)
    designs = sorted(meta_df['design_name'].unique())
    # results[design][k] = {'pw': [mape per seed], 'wl': [...], 'sk': [...]}
    results = {d: {k: {'pw': [], 'wl': [], 'sk': []} for k in k_values}
               for d in designs}

    if X_sk is not None:
        y_sk_z, _, _ = per_placement_z(y_sk_raw, sk_meta_df)

    for held in designs:
        tr = (meta_df['design_name'] != held).values
        te = (meta_df['design_name'] == held).values

        # ── Train power (XGBoost) ─────────────────────────────────────
        sc_pw = StandardScaler()
        m_pw  = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, verbosity=0, n_jobs=1)
        m_pw.fit(sc_pw.fit_transform(X_pw[tr]), y_pw[tr])

        # ── Train WL (LGB + Ridge blend) ──────────────────────────────
        sc_wl  = StandardScaler()
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

        true_pw = meta_df[te]['power_total'].values
        true_wl = meta_df[te]['wirelength'].values
        te_pids = meta_df[te]['placement_id'].values
        unique_pids = np.array(sorted(set(te_pids)))

        # ── Train skew ────────────────────────────────────────────────
        pred_sk_z  = None
        true_sk_ns = None
        sk_te_pids = None
        if X_sk is not None:
            sk_tr = (sk_meta_df['design_name'] != held).values
            sk_te = (sk_meta_df['design_name'] == held).values
            Xs    = X_sk[:, sk_feat_idx] if sk_feat_idx is not None else X_sk
            sc_sk = StandardScaler()
            m_sk  = LGBMRegressor(**LGB_PARAMS)
            m_sk.fit(sc_sk.fit_transform(Xs[sk_tr]), y_sk_z[sk_tr])
            pred_sk_z  = m_sk.predict(sc_sk.transform(Xs[sk_te]))
            true_sk_ns = y_sk_raw[sk_te]
            sk_te_pids = sk_meta_df[sk_te]['pid'].values

        for k in k_values:
            if k == 0:
                # Zero-shot: evaluate on all placements, no calibration.
                eval_mask = np.ones(len(te_pids), dtype=bool)
                pw_mape   = mape(true_pw, pred_pw_raw)
                wl_mape   = mape(true_wl, pred_wl_raw)
                sk_mae    = (cp_skew_ns_mae(pred_sk_z, true_sk_ns, sk_te_pids, [])
                             if pred_sk_z is not None else None)
                results[held][k]['pw'].append(pw_mape)
                results[held][k]['wl'].append(wl_mape)
                results[held][k]['sk'].append(sk_mae)
                continue

            if k >= len(unique_pids):
                # Not enough placements to hold any back for evaluation.
                continue

            for _ in range(n_seeds):
                cal_pids  = rng.choice(unique_pids, size=k, replace=False)
                eval_mask = ~np.isin(te_pids, cal_pids)

                if eval_mask.sum() == 0:
                    continue

                # Compute one global scale from all runs of all cal placements.
                cal_mask_idx = np.where(np.isin(te_pids, cal_pids))[0]
                pw_sc = log_space_scale(true_pw[cal_mask_idx],
                                        pred_pw_raw[cal_mask_idx])
                wl_sc = log_space_scale(true_wl[cal_mask_idx],
                                        pred_wl_raw[cal_mask_idx])

                results[held][k]['pw'].append(
                    mape(true_pw[eval_mask], pred_pw_raw[eval_mask] * pw_sc))
                results[held][k]['wl'].append(
                    mape(true_wl[eval_mask], pred_wl_raw[eval_mask] * wl_sc))

                if pred_sk_z is not None:
                    sk_mae = cp_skew_ns_mae(pred_sk_z, true_sk_ns,
                                            sk_te_pids, cal_pids)
                    results[held][k]['sk'].append(sk_mae)

    return results


def _fmt(vals, is_sk=False):
    """Format mean ± std from a list of values."""
    clean = [v for v in vals if v is not None]
    if not clean:
        return f"{'---':>14}"
    mu, sd = np.mean(clean), np.std(clean)
    if is_sk:
        return f"{mu:6.4f}±{sd:.4f}"
    return f"{mu:5.1f}%±{sd:.1f}%"


if __name__ == '__main__':
    SKEW_CACHE = os.path.join(HERE, 'caches', 'skew_spatial_cache.pkl')
    MODEL_PATH = os.path.join(HERE, 'saved_models', 'model.pkl')

    print(f"{T()} Loading caches...")
    with open(DEF_CACHE,     'rb') as f: dc       = pickle.load(f)
    with open(SAIF_CACHE,    'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE,  'rb') as f: tc_cache = pickle.load(f)
    with open(GRAVITY_CACHE, 'rb') as f: gc       = pickle.load(f)
    with open(SKEW_CACHE,    'rb') as f: skc      = pickle.load(f)

    with open(MODEL_PATH, 'rb') as f: mdl = pickle.load(f)
    sk_feat_idx = mdl.get('sk_feat_idx', None)

    print(f"{T()} Loading CSV...")
    df = pd.read_csv(os.path.join(DATASET, 'unified_manifest.csv'))
    df = df[df['design_name'].isin(list(T_CLK_NS.keys()))].reset_index(drop=True)
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)

    print(f"{T()} Building features ({len(df)} rows)...")
    X_pw, X_wl, y_pw, y_wl, meta_df = build_all_features(df, dc, sc_cache, tc_cache, gc)
    X_sk, y_sk_setup, _, sk_meta_df  = build_skew_features(df, dc, sc_cache, tc_cache, skc)
    print(f"  Power:{X_pw.shape}  WL:{X_wl.shape}  Skew:{X_sk.shape}  "
          f"sk_feat_idx: {len(sk_feat_idx) if sk_feat_idx is not None else 'all'}")

    print(f"{T()} Running cross-placement LODO (K_VALUES={K_VALUES}, "
          f"N_SEEDS={N_SEEDS})...")
    sys.stdout.flush()
    results = run_cp_lodo(X_pw, X_wl, y_pw, y_wl, meta_df,
                          X_sk=X_sk, y_sk_raw=y_sk_setup,
                          sk_meta_df=sk_meta_df, sk_feat_idx=sk_feat_idx)

    designs  = sorted(results.keys())
    k_header = "".join(f"  {'K='+str(k):>15}" for k in K_VALUES)
    sep      = "-" * (12 + 16 * len(K_VALUES))

    print()
    print("=== Power MAPE  [ mean ± std over seeds ] ===")
    print(f"{'Design':<12}{k_header}")
    print(sep)
    for d in designs:
        print(f"{d:<12}" + "".join(f"  {_fmt(results[d][k]['pw']):>15}"
                                    for k in K_VALUES))
    print(sep)
    mean_row = f"{'Mean':<12}"
    for k in K_VALUES:
        all_v = [v for d in designs for v in results[d][k]['pw'] if v is not None]
        mean_row += f"  {_fmt(all_v):>15}"
    print(mean_row)

    print()
    print("=== WL MAPE  [ mean ± std over seeds ] ===")
    print(f"{'Design':<12}{k_header}")
    print(sep)
    for d in designs:
        print(f"{d:<12}" + "".join(f"  {_fmt(results[d][k]['wl']):>15}"
                                    for k in K_VALUES))
    print(sep)
    mean_row = f"{'Mean':<12}"
    for k in K_VALUES:
        all_v = [v for d in designs for v in results[d][k]['wl'] if v is not None]
        mean_row += f"  {_fmt(all_v):>15}"
    print(mean_row)

    print()
    print("=== Skew Setup MAE (ns)  [ K=0: z-score MAE not shown ] ===")
    print(f"{'Design':<12}{k_header}")
    print(sep)
    for d in designs:
        row = f"{d:<12}"
        for k in K_VALUES:
            vals = [v for v in results[d][k]['sk'] if v is not None]
            row += f"  {_fmt(vals, is_sk=True):>15}" if vals else f"  {'---':>15}"
        print(row)
    print(sep)
    mean_row = f"{'Mean':<12}"
    for k in K_VALUES:
        all_v = [v for d in designs for v in results[d][k]['sk'] if v is not None]
        mean_row += f"  {_fmt(all_v, is_sk=True):>15}" if all_v else f"  {'---':>15}"
    print(mean_row)

    print(f"\n{T()} done")
    print(f"K = number of calibration placements (each with all 10 runs)  |  "
          f"N_SEEDS={N_SEEDS} random draws per (design, K)")
