"""
eval_ood.py — OOD evaluation with K-shot calibration sweep.

For each OOD design with a CSV in data/:
  - Loads the saved SwiftCTS model
  - Parses DEF/SAIF/timing on-the-fly
  - Calibration placement: first unique placement_id in CSV
  - Test placement: second unique placement_id
  - K-shot sweep: K ∈ [0, 1, 2, 5, 10, 20]
  - Prints MAPE table per K value

Gravity features default to 0 for OOD designs (no precomputed cache).

OOD CSVs must be placed in the data/ folder by the user:
  swiftcts/data/zipdiv.csv
  swiftcts/data/oc_jpegencode.csv

Placement files directory can be overridden via the SWIFTCTS_PLACEMENT_DIR
environment variable. Defaults to swiftcts/data/placement_files/.

Usage:
  python3 eval_ood.py
"""

import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, 'saved_models', 'model.pkl')

# Placement files directory — override via environment variable if needed
PLACEMENT_DIR = os.environ.get('SWIFTCTS_PLACEMENT_DIR',
                               os.path.join(HERE, 'data', 'placement_files'))

# Import surrogate from same package
sys.path.insert(0, HERE)
from swiftcts import SwiftCTS

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0,
            'zipdiv': 5.0, 'oc_jpegencode': 7.0}

# OOD CSV paths — place these CSVs in the data/ folder before running
# (e.g. swiftcts/data/zipdiv.csv, swiftcts/data/oc_jpegencode.csv)
OOD_CSVS = {
    'zipdiv':        os.path.join(HERE, 'data', 'zipdiv.csv'),
    'oc_jpegencode': os.path.join(HERE, 'data', 'oc_jpegencode.csv'),
}


def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)) * 100


def eval_ood_design(model, df_ood, design_name, t_clk):
    """
    Evaluate one OOD design with K-shot calibration sweep.
    Uses first placement for calibration, second for test.
    """
    placements = df_ood['placement_id'].unique()
    if len(placements) < 2:
        print(f"  WARNING: only {len(placements)} placements, need at least 2. Skipping.")
        return

    cal_pid, test_pid = placements[0], placements[1]
    print(f"  Calibration placement: {cal_pid}")
    print(f"  Test placement:        {test_pid}")
    sys.stdout.flush()

    # Determine DEF/SAIF/timing paths — try CSV columns, then construct from pid
    def _get_paths(pid, design):
        rows = df_ood[df_ood['placement_id'] == pid].iloc[0]
        base_dir = os.path.join(PLACEMENT_DIR, pid)

        # Try CSV columns first
        def_path    = str(rows.get('def_path', ''))
        saif_path   = str(rows.get('saif_path', ''))
        timing_path = str(rows.get('timing_path_csv', str(rows.get('timing_path', ''))))

        # Resolve relative paths
        def _resolve(p, fallback):
            if os.path.isabs(p) and os.path.exists(p):
                return p
            # Try relative to data/
            candidate = os.path.join(HERE, 'data', p.lstrip('./'))
            if os.path.exists(candidate):
                return candidate
            # Try relative to PLACEMENT_DIR
            candidate2 = os.path.join(PLACEMENT_DIR, p.lstrip('./'))
            if os.path.exists(candidate2):
                return candidate2
            return fallback

        def_path    = _resolve(def_path,    os.path.join(base_dir, f'{design}.def'))
        saif_path   = _resolve(saif_path,   os.path.join(base_dir, f'{design}.saif'))
        timing_path = _resolve(timing_path, os.path.join(base_dir, 'timing_paths.csv'))
        return def_path, saif_path, timing_path

    # Load both placements
    for pid in [cal_pid, test_pid]:
        if model.features.has(pid):
            continue
        def_path, saif_path, timing_path = _get_paths(pid, design_name)
        if not os.path.exists(def_path):
            print(f"  ERROR: DEF not found: {def_path}")
            return
        print(f"  Parsing {pid}...")
        sys.stdout.flush()
        model.add_design(pid, def_path, saif_path, timing_path, t_clk=t_clk)

    # ── Get raw predictions for calibration placement ──────────────────
    cal_rows = df_ood[df_ood['placement_id'] == cal_pid].copy().reset_index(drop=True)
    cal_pred_pw, cal_pred_wl = [], []
    cal_true_pw, cal_true_wl = [], []

    for _, row in cal_rows.iterrows():
        p = model.predict(cal_pid,
                          cd=row['cts_cluster_dia'], cs=row['cts_cluster_size'],
                          mw=row['cts_max_wire'],    bd=row['cts_buf_dist'])
        cal_pred_pw.append(p.power_mW / 1000)   # convert back to W
        cal_pred_wl.append(p.wl_mm * 1000)       # convert back to µm
        cal_true_pw.append(row['power_total'])
        cal_true_wl.append(row['wirelength'])

    cal_pred_pw = np.array(cal_pred_pw)
    cal_pred_wl = np.array(cal_pred_wl)
    cal_true_pw = np.array(cal_true_pw)
    cal_true_wl = np.array(cal_true_wl)

    # ── Get raw predictions for test placement ─────────────────────────
    test_rows = df_ood[df_ood['placement_id'] == test_pid].copy().reset_index(drop=True)
    test_pred_pw, test_pred_wl = [], []
    test_true_pw, test_true_wl = [], []

    for _, row in test_rows.iterrows():
        p = model.predict(test_pid,
                          cd=row['cts_cluster_dia'], cs=row['cts_cluster_size'],
                          mw=row['cts_max_wire'],    bd=row['cts_buf_dist'])
        test_pred_pw.append(p.power_mW / 1000)
        test_pred_wl.append(p.wl_mm * 1000)
        test_true_pw.append(row['power_total'])
        test_true_wl.append(row['wirelength'])

    test_pred_pw = np.array(test_pred_pw)
    test_pred_wl = np.array(test_pred_wl)
    test_true_pw = np.array(test_true_pw)
    test_true_wl = np.array(test_true_wl)

    # ── K-shot calibration sweep ────────────────────────────────────────
    K_VALUES = [0, 1, 2, 5, 10, 20]
    n_cal = len(cal_rows)

    print(f"\n  K    Power MAPE    WL MAPE")
    print(f"  {'-'*32}")
    for K in K_VALUES:
        if K == 0:
            pw_m = mape(test_true_pw, test_pred_pw)
            wl_m = mape(test_true_wl, test_pred_wl)
            tag = "(zero-shot)"
        elif K > n_cal:
            continue
        else:
            # Use first K rows of calibration placement as support.
            # Log-space mean: robust to extreme OOD scale factors (e.g. jpeg WL).
            log_supp_pw = np.log(cal_true_pw[:K] + 1e-12) - np.log(cal_pred_pw[:K] + 1e-12)
            log_supp_wl = np.log(cal_true_wl[:K] + 1e-12) - np.log(cal_pred_wl[:K] + 1e-12)
            k_hat_pw = np.exp(np.clip(np.mean(log_supp_pw), -4.6, 4.6))   # ±4.6 → [0.01, 100x]
            k_hat_wl = np.exp(np.clip(np.mean(log_supp_wl), -15.0, 15.0)) # ±15 → full OOD range
            pw_m = mape(test_true_pw, test_pred_pw * k_hat_pw)
            wl_m = mape(test_true_wl, test_pred_wl * k_hat_wl)
            tag = ""
        print(f"  {K:<4} {pw_m:>9.1f}%    {wl_m:>6.1f}%   {tag}")
    sys.stdout.flush()


if __name__ == '__main__':
    # ── Load model ──────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run build_models.py first.")
        sys.exit(1)

    print(f"{T()} Loading model from {MODEL_PATH}...")
    model = SwiftCTS.load(MODEL_PATH)
    print(f"  Loaded. LODO results: {model.lodo_results.get('power_mean_mape', 'N/A')} power MAPE")
    sys.stdout.flush()

    # ── Evaluate each OOD design ────────────────────────────────────────
    found_any = False
    for design_name, csv_path in OOD_CSVS.items():
        if not os.path.exists(csv_path):
            print(f"\n[{design_name}] CSV not found at {csv_path} — skipping.")
            continue

        found_any = True
        t_clk = T_CLK_NS.get(design_name, 7.0)
        print(f"\n{'='*50}")
        print(f"=== OOD Evaluation: {design_name} (t_clk={t_clk}ns) ===")
        print(f"{'='*50}")

        df_ood = pd.read_csv(csv_path)
        df_ood = df_ood.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
        print(f"  CSV rows: {len(df_ood)}")

        eval_ood_design(model, df_ood, design_name, t_clk)

    if not found_any:
        print("\nNo OOD CSVs found. Checked:")
        for k, v in OOD_CSVS.items():
            print(f"  {k}: {v}")

    print(f"\n{T()} DONE")
