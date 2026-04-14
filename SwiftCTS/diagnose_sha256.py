"""
diagnose_sha256.py — Understand why sha256 power K-shot only reaches ~35%.

Checks:
  1. Per-placement true/pred ratio distribution (is the scale consistent?)
  2. Per-placement MAPE (is one placement an outlier, or all bad?)
  3. rel_act variation across sha256 placements
  4. Within-placement R² (is knob sensitivity being predicted correctly?)
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

HERE          = os.path.dirname(os.path.abspath(__file__))
DEF_CACHE     = os.path.join(HERE, 'caches', 'def_cache.pkl')
SAIF_CACHE    = os.path.join(HERE, 'caches', 'saif_cache.pkl')
TIMING_CACHE  = os.path.join(HERE, 'caches', 'timing_cache.pkl')
GRAVITY_CACHE = os.path.join(HERE, 'caches', 'gravity_cache.pkl')
DATASET       = os.path.join(HERE, 'data')

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0}

sys.path.insert(0, HERE)
from eval_lodo_kshot import build_all_features, mape


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def train_power_lodo(X_pw, y_pw, meta_df, held='sha256', seed=42):
    tr = (meta_df['design_name'] != held).values
    te = (meta_df['design_name'] == held).values
    sc = StandardScaler()
    m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=seed, verbosity=0, n_jobs=1)
    m.fit(sc.fit_transform(X_pw[tr]), y_pw[tr])
    pred_log = m.predict(sc.transform(X_pw[te]))
    pred_raw = np.exp(pred_log) * meta_df[te]['pw_norm'].values
    true_raw = meta_df[te]['power_total'].values
    return pred_raw, true_raw, meta_df[te].copy().reset_index(drop=True)


if __name__ == '__main__':
    print("Loading caches...")
    with open(DEF_CACHE,     'rb') as f: dc = pickle.load(f)
    with open(SAIF_CACHE,    'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE,  'rb') as f: tc_cache = pickle.load(f)
    with open(GRAVITY_CACHE, 'rb') as f: gc = pickle.load(f)

    df = pd.read_csv(os.path.join(DATASET, 'unified_manifest.csv'))
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)

    X_pw, X_wl, y_pw, y_wl, meta_df = build_all_features(df, dc, sc_cache, tc_cache, gc)

    print("\nTraining power head (LODO, hold sha256)...")
    pred, true, sha_meta = train_power_lodo(X_pw, y_pw, meta_df)

    # ── 1. rel_act distribution for sha256 placements ─────────────────────
    sha_pids = sha_meta['placement_id'].unique()
    print(f"\n=== sha256: {len(sha_pids)} placements, {len(sha_meta)} rows ===")
    print("\n[1] rel_act per placement:")
    for pid in sorted(sha_pids)[:10]:
        mask = sha_meta['placement_id'] == pid
        rel = sc_cache.get(pid, {}).get('rel_act', None)
        print(f"  {pid[-20:]}  rel_act={rel:.4f}" if rel else f"  {pid[-20:]}  rel_act=N/A")

    # ── 2. Per-placement true/pred ratio ──────────────────────────────────
    print("\n[2] Per-placement true/pred ratio (scale needed):")
    print(f"  {'Placement':<35} {'n':>4}  {'mean(true/pred)':>16}  {'MAPE':>8}  {'within-R²':>10}")
    print(f"  {'-'*80}")
    scales = []
    for pid in sorted(sha_pids):
        mask = (sha_meta['placement_id'] == pid).values
        t = true[mask]; p = pred[mask]
        ratio = np.mean(t / p)
        scales.append(ratio)
        within_r2 = r2(t, p)
        print(f"  {pid[-35:]:<35}  {mask.sum():>4}  {ratio:>16.3f}  {mape(t,p):>7.1f}%  {within_r2:>10.3f}")

    print(f"\n  Scale mean: {np.mean(scales):.3f}  std: {np.std(scales):.3f}  "
          f"min: {np.min(scales):.3f}  max: {np.max(scales):.3f}")
    print(f"  → If std is large, K-shot from 1 placement gives a noisy scale estimate")

    # ── 3. What K-shot from placement 1 gives ─────────────────────────────
    cal_pid = sorted(sha_pids)[0]
    cal_mask = (sha_meta['placement_id'] == cal_pid).values
    cal_scale = np.exp(np.mean(np.log(true[cal_mask] / pred[cal_mask])))
    pred_cal = pred * cal_scale

    print(f"\n[3] K-shot calibration using placement: {cal_pid[-35:]}")
    print(f"  Calibration scale: {cal_scale:.3f}")
    print(f"  Zero-shot MAPE:    {mape(true, pred):.1f}%")
    print(f"  K-shot MAPE:       {mape(true, pred_cal):.1f}%")

    # ── 4. Per-placement MAPE after calibration ───────────────────────────
    print(f"\n[4] Per-placement MAPE after calibration (scale={cal_scale:.3f}):")
    for pid in sorted(sha_pids):
        mask = (sha_meta['placement_id'] == pid).values
        t = true[mask]; p = pred_cal[mask]
        tag = " ← cal" if pid == cal_pid else ""
        print(f"  {pid[-35:]:<35}  {mape(t,p):>7.1f}%{tag}")

    # ── 5. Oracle: what if we knew the perfect scale per placement? ────────
    oracle_preds = pred.copy()
    for pid in sorted(sha_pids):
        mask = (sha_meta['placement_id'] == pid).values
        t = true[mask]; p = pred[mask]
        s = np.exp(np.mean(np.log(t / p)))
        oracle_preds[mask] = p * s
    print(f"\n[5] Oracle MAPE (perfect scale per placement): {mape(true, oracle_preds):.1f}%")
    print(f"    → This is the ceiling for multiplicative K-shot")
    print(f"    → Residual after oracle = irreducible prediction error on knob sensitivity")
