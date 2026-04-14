"""
eval_jpeg.py — Per-placement K-shot evaluation on oc_jpegencode (OOD design).

All 6 placements evaluated independently.  For each placement:
  - First K runs calibrate the model (power/WL scale + skew mu/sig).
  - Remaining N−K runs are evaluated (cal runs excluded).
  - K=0 is pure zero-shot; skew shows '---' at K=0 (no mu/sig estimate).

Usage: python3 eval_jpeg.py
"""

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from swiftcts import SwiftCTS

import pandas as pd
import numpy as np

MODEL_PATH    = os.path.join(HERE, 'saved_models', 'model.pkl')
PLACEMENT_DIR = os.environ.get('SWIFTCTS_PLACEMENT_DIR',
                               os.path.join(HERE, 'data', 'placement_files'))
BASE_DATA     = os.environ.get('SWIFTCTS_DATA_DIR', os.path.join(HERE, 'data'))

DESIGN   = 'oc_jpegencode'
T_CLK    = 10.0
K_VALUES = [0, 1, 2, 5]

# ── Load model and data ──────────────────────────────────────────────────────
model = SwiftCTS.load(MODEL_PATH)

df = pd.read_csv(os.path.join(BASE_DATA, 'oc_jpegencode.csv'))
if len(df) == 0:
    raise ValueError(f"No rows found for {DESIGN}")

def _resolve(rel):
    return os.path.normpath(os.path.join(BASE_DATA, rel))

# Register all placements with the model.
for pid in df['placement_id'].unique():
    row0 = df[df['placement_id'] == pid].iloc[0]
    if 'def_path' in df.columns and pd.notna(row0.get('def_path')):
        model.add_design(pid,
            def_path    = _resolve(row0['def_path']),
            saif_path   = _resolve(row0['saif_path']),
            timing_path = _resolve(row0['timing_path_csv']),
            t_clk=T_CLK)
    else:
        model.add_design(pid,
            def_path    = os.path.join(PLACEMENT_DIR, pid, f'{DESIGN}.def'),
            saif_path   = os.path.join(PLACEMENT_DIR, pid, f'{DESIGN}.saif'),
            timing_path = os.path.join(PLACEMENT_DIR, pid, 'timing_paths.csv'),
            t_clk=T_CLK)

# ── Per-placement K-shot sweep ───────────────────────────────────────────────
results = {}   # results[pid][k] = {'pw_mape', 'wl_mape', 'sk_mae', 'n_eval'}

for pid in sorted(df['placement_id'].unique()):
    placement_df = df[df['placement_id'] == pid].reset_index(drop=True)
    n_total      = len(placement_df)
    results[pid] = {}

    for K in K_VALUES:
        if K >= n_total:
            continue

        cal_df  = placement_df.iloc[:K]
        test_df = placement_df.iloc[K:].reset_index(drop=True)

        # Calibration: geometric mean of ratios (log-space) for power and WL.
        if K == 0:
            pw_scale = wl_scale = 1.0
            sk_mu = sk_sig = None        # no skew conversion at K=0
        else:
            pw_log, wl_log = [], []
            for _, row in cal_df.iterrows():
                p = model.predict(pid, cd=row.cts_cluster_dia,
                                  cs=row.cts_cluster_size, mw=row.cts_max_wire,
                                  bd=row.cts_buf_dist)
                pw_log.append(np.log(row.power_total) - np.log(p.power_mW / 1000))
                wl_log.append(np.log(row.wirelength)  - np.log(p.wl_mm * 1000))
            pw_scale = np.exp(np.mean(pw_log))
            wl_scale = np.exp(np.mean(wl_log))

            cal_ns = cal_df['skew_setup'].values
            sk_mu  = float(cal_ns.mean())
            sk_sig = max(float(cal_ns.std()), max(abs(sk_mu) * 0.01, 1e-4))

        pw_errs, wl_errs, sk_errs = [], [], []
        for _, row in test_df.iterrows():
            pred = model.predict(pid, cd=row.cts_cluster_dia,
                                 cs=row.cts_cluster_size, mw=row.cts_max_wire,
                                 bd=row.cts_buf_dist,
                                 sk_mu=sk_mu, sk_sig=sk_sig)
            pred_pw = pred.power_mW / 1000 * pw_scale
            pred_wl = pred.wl_mm * 1000    * wl_scale
            pw_errs.append(abs(row.power_total - pred_pw) / row.power_total * 100)
            wl_errs.append(abs(row.wirelength  - pred_wl) / row.wirelength  * 100)
            if pred.skew_ns is not None:
                sk_errs.append(abs(row.skew_setup - pred.skew_ns))

        results[pid][K] = {
            'pw_mape': float(np.mean(pw_errs)),
            'wl_mape': float(np.mean(wl_errs)),
            'sk_mae':  float(np.mean(sk_errs)) if sk_errs else None,
            'n_eval':  len(test_df),
        }

# ── Print tables ─────────────────────────────────────────────────────────────
pids     = sorted(results.keys())
labels   = {pid: f"JPEG-{i+1}" for i, pid in enumerate(pids)}
k_header = "".join(f"   K={k:>2}" for k in K_VALUES)
sep      = "-" * (12 + 8 * len(K_VALUES))

print(f"\n{DESIGN}  N_placements={len(pids)}  T_clk={T_CLK}ns\n")

print("=== Power MAPE (%)  [ K=0 = zero-shot ] ===")
print(f"{'Placement':<10}{k_header}")
print(sep)
for pid in pids:
    print(f"{labels[pid]:<10}" + "".join(
        f"  {results[pid][k]['pw_mape']:>5.1f}%" if k in results[pid] else f"  {'---':>6} "
        for k in K_VALUES))
means = [np.mean([results[p][k]['pw_mape'] for p in pids if k in results[p]]) for k in K_VALUES]
print(sep)
print(f"{'Mean':<10}" + "".join(f"  {m:>5.1f}%" for m in means))

print()
print("=== WL MAPE (%) ===")
print(f"{'Placement':<10}{k_header}")
print(sep)
for pid in pids:
    print(f"{labels[pid]:<10}" + "".join(
        f"  {results[pid][k]['wl_mape']:>5.1f}%" if k in results[pid] else f"  {'---':>6} "
        for k in K_VALUES))
means = [np.mean([results[p][k]['wl_mape'] for p in pids if k in results[p]]) for k in K_VALUES]
print(sep)
print(f"{'Mean':<10}" + "".join(f"  {m:>5.1f}%" for m in means))

print()
print("=== Skew Setup MAE (ns)  [ K>=1: ns using K-run mu/sig est. ] ===")
print(f"{'Placement':<10}{k_header}")
print(sep)
for pid in pids:
    row_str = f"{labels[pid]:<10}"
    for k in K_VALUES:
        v = results[pid][k]['sk_mae'] if k in results[pid] else None
        row_str += f"  {'---':>6} " if v is None else f"  {v:>6.4f} "
    print(row_str)
print(sep)
mean_str = f"{'Mean':<10}"
for k in K_VALUES:
    vals = [results[p][k]['sk_mae'] for p in pids
            if k in results[p] and results[p][k]['sk_mae'] is not None]
    mean_str += f"  {'---':>6} " if not vals else f"  {np.mean(vals):>6.4f} "
print(mean_str)
