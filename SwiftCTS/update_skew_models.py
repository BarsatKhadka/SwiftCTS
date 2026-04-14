"""
update_skew_models.py — Retrain pruned skew heads and update saved_models/model.pkl.

Changes to pkl:
  model_skew      : retrained on top-15 features (was 63)
  scaler_skew     : re-fit on 15-dim subspace
  sk_feat_idx     : np.array of 15 feature indices (stored for inference)
  model_skew_hold : NEW — LGB for skew_hold (18 features)
  scaler_skew_hold: NEW — StandardScaler for 18-dim subspace
  skh_feat_idx    : np.array of 18 feature indices

LODO verification run before saving.
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
from prune_skew_eval import get_gain_ranking, lodo_subset, LGB_PARAMS

DEF_CACHE   = os.path.join(HERE, 'caches', 'def_cache.pkl')
SAIF_CACHE  = os.path.join(HERE, 'caches', 'saif_cache.pkl')
TIMING_CACHE= os.path.join(HERE, 'caches', 'timing_cache.pkl')
SKEW_CACHE  = os.path.join(HERE, 'caches', 'skew_spatial_cache.pkl')
MANIFEST    = os.path.join(HERE, 'data', 'unified_manifest.csv')
MODEL_PATH  = os.path.join(HERE, 'saved_models', 'model.pkl')

N_SETUP = 15
N_HOLD  = 18


def lodo_verify(X, y_raw, meta_df, feat_idx):
    """Quick LODO verification (1 seed)."""
    return lodo_subset(X, y_raw, meta_df, feat_idx)


if __name__ == '__main__':
    print("=" * 60)
    print("  Updating pkl with pruned skew heads")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    with open(DEF_CACHE,   'rb') as f: dc = pickle.load(f)
    with open(SAIF_CACHE,  'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE,'rb') as f: tc_cache = pickle.load(f)
    with open(SKEW_CACHE,  'rb') as f: skc = pickle.load(f)
    with open(MODEL_PATH,  'rb') as f: mdl = pickle.load(f)

    df = pd.read_csv(MANIFEST)
    df = df[df['design_name'].isin(['aes', 'ethmac', 'picorv32', 'sha256'])]
    print(f"  Samples: {len(df)}")

    # ── Build 63-dim features ──────────────────────────────────────────────
    print("\nBuilding 63-dim skew features...")
    X, y_setup, y_hold, meta_df = build_features(df, dc, sc_cache, tc_cache, skc)
    print(f"  Shape: {X.shape}")

    # ── Gain rankings ─────────────────────────────────────────────────────
    y_z_setup, _, _ = per_placement_z(y_setup, meta_df)
    y_z_hold,  _, _ = per_placement_z(y_hold,  meta_df)
    ranked_setup, g_setup = get_gain_ranking(X, y_z_setup)
    ranked_hold,  g_hold  = get_gain_ranking(X, y_z_hold)

    sk_feat_idx  = ranked_setup[:N_SETUP]
    skh_feat_idx = ranked_hold[:N_HOLD]

    from prune_skew_eval import SKEW_FEAT_NAMES
    print(f"\n  Top-{N_SETUP} features for skew_setup:")
    for r, i in enumerate(sk_feat_idx, 1):
        print(f"    {r:2d}. [{i:2d}] {SKEW_FEAT_NAMES[i]}")
    print(f"\n  Top-{N_HOLD} features for skew_hold:")
    for r, i in enumerate(skh_feat_idx, 1):
        print(f"    {r:2d}. [{i:2d}] {SKEW_FEAT_NAMES[i]}")

    # ── LODO verification ─────────────────────────────────────────────────
    print(f"\nLODO verification (1 seed, retraining per fold):")
    setup_lodo = lodo_verify(X, y_setup, meta_df, sk_feat_idx)
    hold_lodo  = lodo_verify(X, y_hold,  meta_df, skh_feat_idx)

    print(f"  {'Design':<12}  {'setup MAE':>10}  {'hold MAE':>10}")
    print(f"  {'-'*36}")
    for d in sorted(setup_lodo):
        ok_s = '✓' if setup_lodo[d] < 0.10 else '✗'
        ok_h = '✓' if hold_lodo[d]  < 0.10 else '✗'
        print(f"  {d:<12}  {setup_lodo[d]:>8.4f}ns{ok_s}  {hold_lodo[d]:>8.4f}ns{ok_h}")
    m_s = np.mean(list(setup_lodo.values()))
    m_h = np.mean(list(hold_lodo.values()))
    ok_s = '✓' if m_s < 0.10 else '✗'
    ok_h = '✓' if m_h < 0.10 else '✗'
    print(f"  {'Mean':<12}  {m_s:>8.4f}ns{ok_s}  {m_h:>8.4f}ns{ok_h}")

    if m_s >= 0.10 or m_h >= 0.10:
        print("\n  WARNING: mean MAE >= 0.10ns — check before saving")
        sys.exit(1)

    # ── Train final models on ALL data ────────────────────────────────────
    print(f"\nTraining final models on all {len(df)} samples...")

    # skew_setup (15 features)
    Xs_setup = X[:, sk_feat_idx]
    sc_setup = StandardScaler()
    lgb_setup = LGBMRegressor(**LGB_PARAMS)
    lgb_setup.fit(sc_setup.fit_transform(Xs_setup), y_z_setup)

    # skew_hold (18 features)
    Xs_hold = X[:, skh_feat_idx]
    sc_hold = StandardScaler()
    lgb_hold = LGBMRegressor(**LGB_PARAMS)
    lgb_hold.fit(sc_hold.fit_transform(Xs_hold), y_z_hold)

    print("  Done.")

    # ── Update pkl ────────────────────────────────────────────────────────
    print(f"\nUpdating {MODEL_PATH}...")
    old_sk_dim = mdl['model_skew'].n_features_in_ if 'model_skew' in mdl else 63
    mdl['model_skew']       = lgb_setup
    mdl['scaler_skew']      = sc_setup
    mdl['sk_feat_idx']      = sk_feat_idx.astype(np.int32)
    mdl['model_skew_hold']  = lgb_hold
    mdl['scaler_skew_hold'] = sc_hold
    mdl['skh_feat_idx']     = skh_feat_idx.astype(np.int32)

    # Update lodo dict with new results
    lodo = mdl.get('lodo', {})
    lodo['skew_setup']       = setup_lodo
    lodo['skew_hold']        = hold_lodo
    lodo['sk_n_features']    = N_SETUP
    lodo['skh_n_features']   = N_HOLD
    lodo['sk_feat_idx']      = sk_feat_idx.tolist()
    lodo['skh_feat_idx']     = skh_feat_idx.tolist()
    mdl['lodo'] = lodo

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(mdl, f, protocol=4)

    print(f"  Saved. skew_setup: {old_sk_dim}→{N_SETUP} features, "
          f"skew_hold: NEW {N_HOLD} features")
    print(f"  Keys now: {sorted(mdl.keys())}")
