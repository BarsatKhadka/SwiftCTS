"""
pareto_demo.py — Pareto frontier for AES, JPEG, and Zipdiv.

Protocol per design:
  1. Pick one placement.
  2. K=1 calibration: 1 real run anchors power/WL scales and skew distribution.
  3. Sweep N_SWEEP random knob combos (vectorized batch, ms-scale).
  4. Build 3-objective cost matrix [power, WL, skew] normalised to [0,1].
  5. Remove dominated configs → Pareto front.
  6. Print table + save pareto_<design>.csv and pareto_<design>.txt.

Note: all Pareto configs are equally non-dominated.
      The output table is sorted by power for readability only,
      not because power has higher priority.

Usage: python3 pareto_demo.py
"""

import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from swiftcts import SwiftCTS

import pandas as pd
import numpy as np

MODEL_PATH  = os.path.join(HERE, 'saved_models', 'model.pkl')
DATA_DIR    = os.path.join(HERE, 'data')
DATASET_DIR = os.path.normpath(os.path.join(HERE, '..', 'dataset_with_def'))
N_SWEEP     = 1_000_000   # unique combos, no overlap (grid has ~8.4M total)
SEED        = 42
EXHAUSTIVE  = False

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


print("Loading model...")
model = SwiftCTS.load(MODEL_PATH)

for design_name, cfg in DESIGNS.items():
    t0 = time.time()
    print(f"\n{'='*62}")
    print(f"  {design_name.upper()}  (T_clk={cfg['t_clk']}ns)")
    print(f"{'='*62}")

    # ── Load CSV, pick first placement ───────────────────────────────────────
    df = pd.read_csv(cfg['csv'])
    if cfg['filter']:
        df = df[df['design_name'] == cfg['filter']].reset_index(drop=True)
    if len(df) == 0:
        print("  No data — skipping.")
        continue

    pid          = df['placement_id'].iloc[0]
    placement_df = df[df['placement_id'] == pid].reset_index(drop=True)
    pf_dir       = cfg['pf_dir']

    model.add_design(pid,
        def_path    = os.path.join(pf_dir, pid, cfg['def_name']),
        saif_path   = os.path.join(pf_dir, pid, cfg['saif_name']),
        timing_path = os.path.join(pf_dir, pid, 'timing_paths.csv'),
        t_clk=cfg['t_clk'])

    # ── K=1 calibration (run #0) ─────────────────────────────────────────────
    cal = placement_df.iloc[0]
    p0  = model.predict(pid, cd=cal.cts_cluster_dia, cs=cal.cts_cluster_size,
                        mw=cal.cts_max_wire, bd=cal.cts_buf_dist)

    model.calibrate_power(pid, true_pw=[cal.power_total],
                                pred_pw=[p0.power_mW / 1000])
    model.calibrate_wl(   pid, true_wl=[cal.wirelength],
                                pred_wl=[p0.wl_mm * 1000])

    # Skew anchor: K=1 gives mu only; sig uses floor (no std from 1 sample)
    sk_mu  = float(cal.skew_setup)
    sk_sig = max(abs(sk_mu) * 0.01, 1e-4)

    pw_scale = model._pw_cal.get(pid, 1.0)
    wl_scale = model._wl_cal.get(pid, 1.0)

    print(f"  pid      : {pid}")
    print(f"  cal knobs: cd={cal.cts_cluster_dia:.0f}  cs={cal.cts_cluster_size:.0f}  "
          f"mw={cal.cts_max_wire:.0f}  bd={cal.cts_buf_dist:.0f}")
    print(f"  pw_scale : {pw_scale:.4f}   wl_scale: {wl_scale:.4f}")
    print(f"  sk_mu    : {sk_mu:.4f} ns   sk_sig: {sk_sig:.4f} ns")

    # ── Vectorised batch sweep + 3-obj Pareto (via model.optimize) ───────────
    if EXHAUSTIVE:
        n_cd = 70 - 35 + 1; n_cs = 30 - 12 + 1
        n_mw = 280 - 130 + 1; n_bd = 150 - 70 + 1
        total_grid = n_cd * n_cs * n_mw * n_bd
        print(f"  Exhaustive grid: {n_cd}×{n_cs}×{n_mw}×{n_bd} = {total_grid:,} configs...")
        result = model.optimize(pid, sk_mu=sk_mu, sk_sig=sk_sig, seed=SEED,
                                exhaustive=True)
    else:
        print(f"  Sweeping {N_SWEEP:,} configs...")
        result = model.optimize(pid, n=N_SWEEP, sk_mu=sk_mu, sk_sig=sk_sig, seed=SEED)
    n_pareto = len(result)
    elapsed  = time.time() - t0
    print(f"  Pareto front: {n_pareto} configs  ({elapsed:.1f}s)\n")

    # ── Print table (top 15, sorted by power for display only) ───────────────
    print(f"  {'#':>3}  {'cd':>4} {'cs':>4} {'mw':>5} {'bd':>4}  "
          f"{'Power(mW)':>10}  {'WL(mm)':>8}  {'Skew(ns)':>9}")
    print(f"  {'-'*60}")
    for i, r in result.head(15).iterrows():
        print(f"  {i+1:>3}  {int(r.cd):>4} {int(r.cs):>4} {int(r.mw):>5} {int(r.bd):>4}  "
              f"{r.power_mW:>10.4f}  {r.wl_mm:>8.4f}  {r.skew_ns:>9.4f}")
    if len(result) > 15:
        print(f"  ... ({len(result) - 15} more configs in CSV)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_out = os.path.join(HERE, f'pareto_{design_name}.csv')
    result.to_csv(csv_out, index=False)

    # ── Save TXT report ───────────────────────────────────────────────────────
    txt_out = os.path.join(HERE, f'pareto_{design_name}.txt')

    PLACEMENT_COLS = ['aspect_ratio', 'core_util', 'density', 'synth_strategy',
                      'io_mode', 'time_driven', 'routability_driven']

    with open(txt_out, 'w') as f:
        f.write(f"SwiftCTS Pareto Report — {design_name}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"[Placement]\n")
        f.write(f"  placement_id : {pid}\n")
        f.write(f"  t_clk        : {cfg['t_clk']} ns\n")
        for col in PLACEMENT_COLS:
            if col in placement_df.columns:
                val = placement_df[col].iloc[0]
                f.write(f"  {col:<22}: {val}\n")

        f.write(f"\n[K=1 Calibration]\n")
        f.write(f"  cal knobs    : cd={cal.cts_cluster_dia:.0f}  cs={cal.cts_cluster_size:.0f}  "
                f"mw={cal.cts_max_wire:.0f}  bd={cal.cts_buf_dist:.0f}\n")
        f.write(f"  true power   : {cal.power_total:.6f} W\n")
        f.write(f"  true WL      : {cal.wirelength:.1f} um\n")
        f.write(f"  true skew    : {cal.skew_setup:.4f} ns\n")
        f.write(f"  pw_scale     : {pw_scale:.6f}\n")
        f.write(f"  wl_scale     : {wl_scale:.6f}\n")
        f.write(f"  sk_mu        : {sk_mu:.6f} ns\n")
        f.write(f"  sk_sig       : {sk_sig:.6f} ns\n")

        f.write(f"\n[Sweep]\n")
        if EXHAUSTIVE:
            n_cd = 70 - 35 + 1; n_cs = 30 - 12 + 1
            n_mw = 280 - 130 + 1; n_bd = 150 - 70 + 1
            f.write(f"  exhaustive   : True ({n_cd*n_cs*n_mw*n_bd:,} combos)\n")
        else:
            f.write(f"  n_sweep      : {N_SWEEP:,}\n")
        f.write(f"  cd_range     : [35, 70]\n")
        f.write(f"  cs_range     : [12, 30]\n")
        f.write(f"  mw_range     : [130, 280]\n")
        f.write(f"  bd_range     : [70, 150]\n")
        f.write(f"  elapsed      : {elapsed:.2f}s\n")

        f.write(f"\n[Pareto Front — {n_pareto} configs, sorted by power]\n")
        f.write(f"  {'#':>4}  {'cd':>4} {'cs':>4} {'mw':>5} {'bd':>4}  "
                f"{'Power(mW)':>10}  {'WL(mm)':>8}  {'Skew(ns)':>9}\n")
        f.write(f"  {'-'*60}\n")
        for i, r in result.iterrows():
            f.write(f"  {i+1:>4}  {int(r.cd):>4} {int(r.cs):>4} {int(r.mw):>5} {int(r.bd):>4}  "
                    f"{r.power_mW:>10.4f}  {r.wl_mm:>8.4f}  {r.skew_ns:>9.4f}\n")

        f.write(f"\n[Extremes]\n")
        f.write(f"  Min power : {result.iloc[result.power_mW.argmin()].to_dict()}\n")
        f.write(f"  Min WL    : {result.iloc[result.wl_mm.argmin()].to_dict()}\n")
        f.write(f"  Min skew  : {result.iloc[result.skew_ns.argmin()].to_dict()}\n")

    print(f"\n  Saved: {csv_out}")
    print(f"  Saved: {txt_out}")

print(f"\n{'='*62}")
print("  Done.")
print(f"{'='*62}\n")
