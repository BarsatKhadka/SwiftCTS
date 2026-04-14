"""
pareto_search.py — Compare random / Sobol / NSGA-II search strategies.

All three methods use N = 1,000,000 function evaluations.

  random : uniform unique sampling without replacement from the 8.4M integer grid
  sobol  : Sobol low-discrepancy sequence (2^20 = 1,048,576 points)
  nsga2  : NSGA-II evolutionary (pop=500 × 2000 gen ≈ 1M evals), pymoo

For each method × design the script:
  1. Loads the model and does K=1 calibration.
  2. Runs the search.
  3. Saves pareto_<design>_<method>.csv
  4. Prints a timing + front-size summary table at the end.

Usage:
    python3 pareto_search.py                   # all methods, all designs
    python3 pareto_search.py nsga2             # one method, all designs
    python3 pareto_search.py sobol aes         # one method, one design
"""

import os, sys, time
HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'pareto_comparison')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, HERE)
from swiftcts import SwiftCTS

import pandas as pd
import numpy as np

MODEL_PATH  = os.path.join(HERE, 'saved_models', 'model.pkl')
DATA_DIR    = os.path.join(HERE, 'data')
DATASET_DIR = os.path.normpath(os.path.join(HERE, '..', 'dataset_with_def'))

# Evaluations per method.
# random/sobol: Pareto quality saturates ~200K; 1M adds little but 5× slower.
# nsga2: directed search, 50K directed >> 200K random; no point going to 1M.
N_PER_METHOD = {
    'random': 100_000,
    'sobol':  100_000,
    'nsga2':   50_000,
}
SEED    = 42
METHODS = ['random', 'sobol', 'nsga2']

DESIGNS = {
    'aes': {
        'csv':       os.path.join(DATA_DIR, 'unified_manifest.csv'),
        'filter':    'aes',
        'pf_dir':    os.path.join(DATASET_DIR, 'placement_files'),
        'def_name':  'aes.def',
        'saif_name': 'aes.saif',
        't_clk':     7.0,
    },
    'picorv32': {
        'csv':       os.path.join(DATA_DIR, 'unified_manifest.csv'),
        'filter':    'picorv32',
        'pf_dir':    os.path.join(DATASET_DIR, 'placement_files'),
        'def_name':  'picorv32.def',
        'saif_name': 'picorv32.saif',
        't_clk':     5.0,
    },
    'sha256': {
        'csv':       os.path.join(DATA_DIR, 'unified_manifest.csv'),
        'filter':    'sha256',
        'pf_dir':    os.path.join(DATASET_DIR, 'placement_files'),
        'def_name':  'sha256.def',
        'saif_name': 'sha256.saif',
        't_clk':     9.0,
    },
    'ethmac': {
        'csv':       os.path.join(DATA_DIR, 'unified_manifest.csv'),
        'filter':    'ethmac',
        'pf_dir':    os.path.join(DATASET_DIR, 'placement_files'),
        'def_name':  'ethmac.def',
        'saif_name': 'ethmac.saif',
        't_clk':     9.0,
    },
    'oc_jpegencode': {
        'csv':       os.path.join(DATA_DIR, 'oc_jpegencode.csv'),
        'filter':    None,
        'pf_dir':    os.path.join(DATASET_DIR, 'placement_files'),
        'def_name':  'oc_jpegencode.def',
        'saif_name': 'oc_jpegencode.saif',
        't_clk':     10.0,
    },
    'zipdiv': {
        'csv':       os.path.join(DATASET_DIR, 'zipdiv_test.csv'),
        'filter':    None,
        'pf_dir':    os.path.join(DATASET_DIR, 'zipdiv_placement_files'),
        'def_name':  'zipdiv.def',
        'saif_name': 'zipdiv.saif',
        't_clk':     5.0,
    },
}


def calibrate(model, pid, placement_df, cfg):
    """K=1 calibration — returns (sk_mu, sk_sig)."""
    cal = placement_df.iloc[0]
    p0  = model.predict(pid, cd=cal.cts_cluster_dia, cs=cal.cts_cluster_size,
                        mw=cal.cts_max_wire, bd=cal.cts_buf_dist)
    model.calibrate_power(pid, true_pw=[cal.power_total],
                               pred_pw=[p0.power_mW / 1000])
    model.calibrate_wl(   pid, true_wl=[cal.wirelength],
                               pred_wl=[p0.wl_mm * 1000])
    sk_mu  = float(cal.skew_setup)
    sk_sig = max(abs(sk_mu) * 0.01, 1e-4)
    return sk_mu, sk_sig


# ── Parse CLI args ────────────────────────────────────────────────────────────
args    = sys.argv[1:]
methods = [a for a in args if a in METHODS] or METHODS
designs = [a for a in args if a in DESIGNS] or list(DESIGNS.keys())

print(f"Methods : {methods}")
print(f"Designs : {designs}")
for m in methods:
    print(f"  N[{m}] = {N_PER_METHOD[m]:,}")
print()

print("Loading model...")
model = SwiftCTS.load(MODEL_PATH)

# Summary table rows: {design, method, n_pareto, elapsed}
summary = []

for design_name in designs:
    cfg = DESIGNS[design_name]

    df_csv = pd.read_csv(cfg['csv'])
    if cfg['filter']:
        df_csv = df_csv[df_csv['design_name'] == cfg['filter']].reset_index(drop=True)
    if len(df_csv) == 0:
        print(f"  [{design_name}] No data — skipping.")
        continue

    pid          = df_csv['placement_id'].iloc[0]
    placement_df = df_csv[df_csv['placement_id'] == pid].reset_index(drop=True)

    model.add_design(pid,
        def_path    = os.path.join(cfg['pf_dir'], pid, cfg['def_name']),
        saif_path   = os.path.join(cfg['pf_dir'], pid, cfg['saif_name']),
        timing_path = os.path.join(cfg['pf_dir'], pid, 'timing_paths.csv'),
        t_clk=cfg['t_clk'])

    sk_mu, sk_sig = calibrate(model, pid, placement_df, cfg)

    print(f"\n{'='*62}")
    print(f"  {design_name.upper()}  (pid={pid}  T_clk={cfg['t_clk']}ns)")
    print(f"  sk_mu={sk_mu:.4f}ns  sk_sig={sk_sig:.4f}ns")
    print(f"{'='*62}")

    for method in methods:
        n_evals = N_PER_METHOD[method]
        t0 = time.time()
        print(f"  [{method}] running {n_evals:,} evals...", end=' ', flush=True)

        n_evals = N_PER_METHOD[method]
        result = model.optimize(pid, n=n_evals, sk_mu=sk_mu, sk_sig=sk_sig,
                                seed=SEED, method=method)

        elapsed  = time.time() - t0
        n_pareto = len(result)
        print(f"{n_pareto} Pareto pts  ({elapsed:.1f}s)")

        # Save CSV
        csv_out = os.path.join(OUT_DIR, f'pareto_{design_name}_{method}.csv')
        result.to_csv(csv_out, index=False)

        # Print top 10
        print(f"\n  {'#':>3}  {'cd':>4} {'cs':>4} {'mw':>5} {'bd':>4}  "
              f"{'Power(mW)':>10}  {'WL(mm)':>8}  {'Skew(ns)':>9}")
        print(f"  {'-'*58}")
        for i, r in result.head(10).iterrows():
            print(f"  {i+1:>3}  {int(r.cd):>4} {int(r.cs):>4} {int(r.mw):>5} {int(r.bd):>4}  "
                  f"{r.power_mW:>10.4f}  {r.wl_mm:>8.4f}  {r.skew_ns:>9.4f}")
        if n_pareto > 10:
            print(f"  ... ({n_pareto - 10} more in {os.path.basename(csv_out)})")
        print()

        # Hypervolume indicator
        from pymoo.indicators.hv import HV as _HV
        F = result[['power_mW', 'wl_mm', 'skew_ns']].values
        ref = F.max(axis=0) * 1.10
        hv = float(_HV(ref_point=ref)(F))

        summary.append({
            'design':       design_name,
            'method':       method,
            'n_evals':      n_evals,
            'n_pareto':     n_pareto,
            'elapsed_s':    round(elapsed, 2),
            'hypervolume':  round(hv, 6),
            'min_power_mW': round(result.power_mW.min(), 4),
            'min_wl_mm':    round(result.wl_mm.min(), 4),
            'min_skew_ns':  round(result.skew_ns.min(), 4),
        })

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  SUMMARY")
print(f"{'='*62}")
print(f"  {'Design':<16} {'Method':<8} {'N_evals':>10} {'#Pareto':>8} {'Time(s)':>9}  "
      f"{'HV':>12}  {'MinPow(mW)':>12}  {'MinWL(mm)':>10}  {'MinSkew(ns)':>12}")
print(f"  {'-'*105}")
for s in summary:
    print(f"  {s['design']:<16} {s['method']:<8} {s['n_evals']:>10,} {s['n_pareto']:>8} {s['elapsed_s']:>9.1f}  "
          f"{s['hypervolume']:>12.4f}  {s['min_power_mW']:>12.4f}  {s['min_wl_mm']:>10.4f}  {s['min_skew_ns']:>12.4f}")

# Save summary CSV
sum_csv = os.path.join(OUT_DIR, 'pareto_search_summary.csv')
pd.DataFrame(summary).to_csv(sum_csv, index=False)
print(f"\n  Summary saved → {sum_csv}")
print(f"{'='*62}\n")
