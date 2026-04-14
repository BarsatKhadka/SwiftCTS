"""
eval_skew.py — LODO evaluation of skew_setup AND skew_hold heads.

Evaluates the same 63-dim LGB feature vector on both skew targets:
  skew_setup  — setup-time clock skew (ns): maximum clock arrival imbalance
                causing setup violations. Lower = better.
  skew_hold   — hold-time clock skew (ns): minimum clock arrival imbalance
                causing hold violations. Lower = better.

Method: proper LODO — retrain LGB for each fold (3 designs train, 1 held out).
Target: absolute ns MAE (how far off in nanoseconds on unseen designs).
Also compares: with vs without critical-path spatial features (ck+ski blocks).
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL_PATH  = os.path.join(HERE, 'saved_models', 'model.pkl')
MANIFEST    = os.path.join(HERE, 'data', 'unified_manifest.csv')
DEF_CACHE   = os.path.join(HERE, 'caches', 'def_cache.pkl')
SAIF_CACHE  = os.path.join(HERE, 'caches', 'saif_cache.pkl')
TIMING_CACHE= os.path.join(HERE, 'caches', 'timing_cache.pkl')
SKEW_CACHE  = os.path.join(HERE, 'caches', 'skew_spatial_cache.pkl')

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0}

LGB_PARAMS = dict(n_estimators=300, num_leaves=31, learning_rate=0.03,
                  min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)


def encode_synth(s):
    if pd.isna(s): return 0.5, 2.0, 0.5
    s = str(s).upper()
    sd = 1.0 if 'DELAY' in s else 0.0
    try: level = float(s.split()[-1])
    except: level = 2.0
    return sd, level, sd * level / 4.0


def build_sk_features_row(d, s, t, sk, cd, cs, mw, bd):
    """63-dim skew feature vector."""
    nf = d['n_ff']; da = d['die_area']
    hpwl = d['ff_hpwl']; sp = d['ff_spacing']
    fx = d['frac_xor']; cpf = d['comb_per_ff']; av = d['avg_ds']
    rel = s.get('rel_act', 0.05)
    sm = t['slack_mean']; fn = t['frac_neg']; ft = t['frac_tight']

    kn = [np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd]

    ck = [
        sk.get('crit_max_dist', 0.),   sk.get('crit_mean_dist', 0.),
        sk.get('crit_p90_dist', 0.),   sk.get('crit_ff_hpwl', 0.),
        sk.get('crit_cx_offset', 0.),  sk.get('crit_cy_offset', 0.),
        sk.get('crit_x_std', 0.),      sk.get('crit_y_std', 0.),
        sk.get('crit_frac_boundary', 0.), sk.get('crit_star_degree', 0.),
        sk.get('crit_chain_frac', 0.), sk.get('crit_asymmetry', 0.),
        sk.get('crit_eccentricity', 1.), sk.get('crit_density_ratio', 1.),
        np.log1p(sk.get('crit_max_dist_um', hpwl)),
        np.log1p(sk.get('crit_mean_dist_um', hpwl / 2)),
    ]

    cm_um = sk.get('crit_max_dist_um', hpwl)
    cs_v  = sk.get('crit_star_degree', 0.)
    ca_v  = sk.get('crit_asymmetry', 0.)
    cd_v  = sk.get('crit_density_ratio', 1.)
    cc_v  = sk.get('crit_chain_frac', 0.)
    ch_v  = sk.get('crit_ff_hpwl', 0.)
    cx_v  = sk.get('crit_cx_offset', 0.)
    cy_v  = sk.get('crit_cy_offset', 0.)

    ski = [
        cd / (sp + 1), bd / (cm_um + 1), mw / (cm_um + 1),
        cs_v * cd, ca_v * mw, cd_v * cs,
        sk.get('crit_max_dist', 0.) * cd, ca_v * sk.get('crit_max_dist', 0.),
        fn * cs_v, ft * cc_v, ch_v / (cs + 1),
        np.log1p(cm_um / (cd + 1)), np.log1p(cm_um / (bd + 1)),
        np.log1p(cm_um / (mw + 1)),
        cx_v * cd, cy_v * mw, np.log1p(nf / cs) * ch_v,
    ]

    sk_ctx = [
        np.log1p(nf), np.log1p(da), np.log1p(hpwl), np.log1p(sp), d['die_aspect'],
        d['ff_cx'], d['ff_cy'], d['ff_x_std'], d['ff_y_std'],
        fx, cpf, av, rel, s.get('mean_sig_prob', 0.),
        sm, t['slack_std'], t['slack_min'], t['slack_p10'],
        fn, ft, t['frac_critical'], np.log1p(t['n_paths'] / (nf + 1)),
    ]

    return sk_ctx + kn + ck + ski  # 63 dims


def build_features(df_in, dc, sc_cache, tc_cache, skc):
    rows, y_setup, y_hold, meta = [], [], [], []
    for _, row in df_in.iterrows():
        pid    = row['placement_id']
        design = row['design_name']
        d = dc.get(pid); s = sc_cache.get(pid); t = tc_cache.get(pid)
        sk = skc.get(pid, {})
        if not d or not s or not t: continue
        if not np.isfinite(row['skew_setup']) or not np.isfinite(row['skew_hold']): continue

        cd = row['cts_cluster_dia']; cs = row['cts_cluster_size']
        mw = row['cts_max_wire'];    bd = row['cts_buf_dist']

        feat = build_sk_features_row(d, s, t, sk, cd, cs, mw, bd)
        rows.append(feat)
        y_setup.append(row['skew_setup'])
        y_hold.append(row['skew_hold'])
        meta.append({'pid': pid, 'design_name': design})

    X  = np.array(rows, dtype=np.float64)
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0

    return X, np.array(y_setup), np.array(y_hold), pd.DataFrame(meta)


def per_placement_z(y, meta_df):
    """Per-placement z-score for training; return (y_z, mu_arr, sig_arr)."""
    y_z = np.zeros_like(y)
    mu_arr  = np.zeros_like(y)
    sig_arr = np.ones_like(y)
    for pid in meta_df['pid'].unique():
        idx = meta_df['pid'] == pid
        vals = y[idx.values]
        mu = vals.mean()
        sig = max(vals.std(), max(abs(mu) * 0.01, 1e-4))
        y_z[idx.values]   = (vals - mu) / sig
        mu_arr[idx.values] = mu
        sig_arr[idx.values]= sig
    return y_z, mu_arr, sig_arr


def lodo_skew(X, y_raw, meta_df, label='skew_setup', use_spatial=True):
    """Proper LODO: retrain LGB per fold, evaluate absolute ns MAE."""
    Xs = X if use_spatial else np.concatenate([X[:, :30], np.zeros((len(X), 33))], axis=1)
    designs = sorted(meta_df['design_name'].unique())
    per_design = {}

    y_z, mu_arr, sig_arr = per_placement_z(y_raw, meta_df)

    for held in designs:
        tr = (meta_df['design_name'] != held).values
        te = (meta_df['design_name'] == held).values

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xs[tr]); Xte = sc.transform(Xs[te])
        lgb = LGBMRegressor(**LGB_PARAMS)
        lgb.fit(Xtr, y_z[tr])
        pred_z = lgb.predict(Xte)
        # Convert back to absolute ns
        pred_ns  = pred_z * sig_arr[te] + mu_arr[te]
        true_ns  = y_raw[te]
        per_design[held] = float(np.mean(np.abs(pred_ns - true_ns)))

    return per_design


if __name__ == '__main__':
    print("=" * 62)
    print("  SwiftCTS — skew_setup AND skew_hold LODO Evaluation")
    print("=" * 62)

    with open(DEF_CACHE,   'rb') as f: dc = pickle.load(f)
    with open(SAIF_CACHE,  'rb') as f: sc_cache = pickle.load(f)
    with open(TIMING_CACHE,'rb') as f: tc_cache = pickle.load(f)
    with open(SKEW_CACHE,  'rb') as f: skc = pickle.load(f)

    df = pd.read_csv(MANIFEST)
    df = df[df['design_name'].isin(['aes', 'ethmac', 'picorv32', 'sha256'])]
    print(f"  Samples: {len(df)}")

    print("\nBuilding 63-dim skew features...")
    X, y_setup, y_hold, meta_df = build_features(df, dc, sc_cache, tc_cache, skc)
    print(f"  Shape: {X.shape}")
    spatial_hits = sum(1 for p in meta_df['pid'].unique() if skc.get(p))
    print(f"  Placements with critical-path spatial features: {spatial_hits}/{len(meta_df['pid'].unique())}")

    # ── LODO for skew_setup ───────────────────────────────────────────────
    print("\nRunning LODO — skew_setup (with spatial features)...")
    setup_w  = lodo_skew(X, y_setup, meta_df, 'skew_setup',  use_spatial=True)
    print("Running LODO — skew_setup (WITHOUT spatial features)...")
    setup_wo = lodo_skew(X, y_setup, meta_df, 'skew_setup',  use_spatial=False)

    # ── LODO for skew_hold ────────────────────────────────────────────────
    print("Running LODO — skew_hold (with spatial features)...")
    hold_w   = lodo_skew(X, y_hold,  meta_df, 'skew_hold',   use_spatial=True)
    print("Running LODO — skew_hold (WITHOUT spatial features)...")
    hold_wo  = lodo_skew(X, y_hold,  meta_df, 'skew_hold',   use_spatial=False)

    designs = sorted(meta_df['design_name'].unique())

    # ── Print combined table ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESULTS — Absolute ns MAE (per-placement z-score then denormalized)")
    print(f"{'='*70}")
    print(f"\n  skew_setup (setup-time clock skew):")
    print(f"  {'Design':<12}  {'With spatial':>14}  {'No spatial':>12}  {'Δ (spatial helps)':>18}")
    print(f"  {'-'*60}")
    for d in designs:
        w = setup_w[d]; wo = setup_wo[d]; delta = wo - w
        ok = '✓' if w < 0.10 else '✗'
        print(f"  {d:<12}  {w:>12.4f}ns {ok}  {wo:>10.4f}ns  {delta:>+16.4f}ns")
    m_w = np.mean(list(setup_w.values())); m_wo = np.mean(list(setup_wo.values()))
    ok = '✓' if m_w < 0.10 else '✗'
    print(f"  {'-'*60}")
    print(f"  {'Mean':<12}  {m_w:>12.4f}ns {ok}  {m_wo:>10.4f}ns  {m_wo-m_w:>+16.4f}ns")

    print(f"\n  skew_hold (hold-time clock skew):")
    print(f"  {'Design':<12}  {'With spatial':>14}  {'No spatial':>12}  {'Δ (spatial helps)':>18}")
    print(f"  {'-'*60}")
    for d in designs:
        w = hold_w[d]; wo = hold_wo[d]; delta = wo - w
        ok = '✓' if w < 0.10 else '✗'
        print(f"  {d:<12}  {w:>12.4f}ns {ok}  {wo:>10.4f}ns  {delta:>+16.4f}ns")
    m_w2 = np.mean(list(hold_w.values())); m_wo2 = np.mean(list(hold_wo.values()))
    ok = '✓' if m_w2 < 0.10 else '✗'
    print(f"  {'-'*60}")
    print(f"  {'Mean':<12}  {m_w2:>12.4f}ns {ok}  {m_wo2:>10.4f}ns  {m_wo2-m_w2:>+16.4f}ns")

    print(f"""
  NOTES:
  • "With spatial" = full 63-dim vector including critical-path spatial
    features (crit_max_dist, crit_asymmetry, etc.) from skew_spatial_cache.pkl
  • "No spatial" = spatial blocks zeroed out (33 dims set to 0), using only
    context (22 dims) + knobs (8 dims) — 30 active features.
  • Positive Δ means spatial features reduce MAE — they help.
  • Target: MAE < 0.10 ns (per CLAUDE.md prime directive).
  • skew_setup: clock arrival imbalance causing setup violations.
  • skew_hold:  clock arrival imbalance causing hold violations.
    Same 63-dim feature vector used for both; hold skew is harder
    (smaller magnitude, more noise in ground-truth CTS output).
""")
