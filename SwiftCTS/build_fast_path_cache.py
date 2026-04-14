"""
build_fast_path_cache.py — Extract fast-path (hold-relevant) spatial features.

Critical vs fast paths:
  setup-critical: nsmallest(50, 'slack') → existing skew_spatial_cache
  hold-relevant:  nlargest(50, 'slack')  → this cache (fast_path_cache.pkl)

Hold violations occur on fast/short paths where the early clock arrives before
the data from a fast combinational path has settled. Features:
  - Spatial: distance distribution of fast launch-capture pairs
  - Topology: self-loops (launch==capture, zero delay), star/chain patterns
  - Slack tail: slack_max, p75, p90, spread (overall timing margin distribution)
  - Interactions: knob × fast-path geometry (computed at build time in _build_sk_features)
"""

import re, os, sys, pickle, time
import numpy as np
import pandas as pd
from collections import Counter

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

HERE    = os.path.dirname(os.path.abspath(__file__))
BASE    = os.path.join(HERE, '..')
PLACEMENT_DIR = os.path.join(BASE, 'dataset_with_def', 'placement_files')
OUT_CACHE     = os.path.join(HERE, 'caches', 'fast_path_cache.pkl')
MANIFEST      = os.path.join(HERE, 'data', 'unified_manifest.csv')

# OOD CSVs
OOD_CSVS = [
    os.path.join(HERE, 'data', 'zipdiv.csv'),
    os.path.join(HERE, 'data', 'oc_jpegencode.csv'),
]


def parse_def_ff_positions(def_path):
    """Return {ff_name: (x_um, y_um)}, die_w, die_h, (x0, y0)."""
    try:
        with open(def_path) as f:
            content = f.read()
    except Exception:
        return None, None, None, None

    units_m = re.search(r'UNITS DISTANCE MICRONS (\d+)', content)
    units = int(units_m.group(1)) if units_m else 1000

    die_m = re.search(r'DIEAREA\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)', content)
    if not die_m:
        return None, None, None, None
    x0, y0, x1, y1 = [float(v) / units for v in die_m.groups()]
    die_w, die_h = x1 - x0, y1 - y0

    ff_pattern = r'-\s+(\S+)\s+sky130_fd_sc_hd__df\w+\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)'
    ff_positions = {}
    for name, x, y in re.findall(ff_pattern, content):
        ff_positions[name] = (float(x) / units, float(y) / units)

    return ff_positions, die_w, die_h, (x0, y0)


def compute_fast_path_features(ff_positions, die_w, die_h, origin, timing_df, top_k=50):
    """
    Compute spatial features of fast (high-slack) timing paths.
    These are hold-relevant: fast paths are the most at-risk for hold violations.
    """
    if ff_positions is None or len(ff_positions) == 0:
        return _zeros()

    dw = max(die_w, 1.0)
    dh = max(die_h, 1.0)

    all_xs = np.array([p[0] for p in ff_positions.values()])
    all_ys = np.array([p[1] for p in ff_positions.values()])
    all_cx, all_cy = all_xs.mean(), all_ys.mean()

    slacks = timing_df['slack'].values
    n_total = len(slacks)

    # --- Slack distribution features (full set) ---
    slack_max  = float(slacks.max())
    slack_p90  = float(np.percentile(slacks, 90))
    slack_p75  = float(np.percentile(slacks, 75))
    slack_min  = float(slacks.min())
    slack_spread = slack_max - slack_min

    frac_self_loop = float((timing_df['launch_flop'] == timing_df['capture_flop']).mean())
    frac_fast_1ns  = float((slacks > 1.0).mean())

    # --- Spatial features of top-K fast paths ---
    fast_paths = timing_df.nlargest(top_k, 'slack')

    fast_launch_pos, fast_capture_pos, path_dists = [], [], []
    launch_counts = Counter()

    for _, row in fast_paths.iterrows():
        lf = str(row['launch_flop']).strip()
        cf = str(row['capture_flop']).strip()
        # Self-loops: distance = 0 (most hold-critical), include them
        if lf in ff_positions:
            lp = ff_positions[lf]
            if cf in ff_positions:
                cp = ff_positions[cf]
            else:
                cp = lp  # fallback for self-loop where cf not in DEF
            fast_launch_pos.append(lp)
            fast_capture_pos.append(cp)
            dist = np.sqrt((lp[0] - cp[0])**2 + (lp[1] - cp[1])**2)
            path_dists.append(dist)
            launch_counts[lf] += 1

    if not path_dists:
        return _zeros(slack_max, slack_p90, slack_p75, slack_spread,
                      frac_self_loop, frac_fast_1ns)

    path_dists = np.array(path_dists)
    all_fast = np.array(fast_launch_pos + fast_capture_pos)
    fast_xs, fast_ys = all_fast[:, 0], all_fast[:, 1]

    # Centroid offset of fast FFs vs all FFs
    fast_cx, fast_cy = fast_xs.mean(), fast_ys.mean()
    cx_offset = abs(fast_cx - all_cx) / dw
    cy_offset = abs(fast_cy - all_cy) / dh

    # HPWL and spread
    fast_hpwl = ((fast_xs.max() - fast_xs.min()) + (fast_ys.max() - fast_ys.min()))
    fast_hpwl_norm = fast_hpwl / (dw + dh)
    fast_x_std = fast_xs.std() / dw
    fast_y_std = fast_ys.std() / dh

    # Eccentricity
    fast_eccentricity = max(fast_xs.std(), 1.0) / max(fast_ys.std(), 1.0)
    if fast_eccentricity > 1.0:
        fast_eccentricity = 1.0 / fast_eccentricity

    # Spatial asymmetry
    left_frac = (fast_xs < (all_xs.min() + 0.5 * dw)).mean()
    top_frac  = (fast_ys > (all_ys.min() + 0.5 * dh)).mean()
    asymmetry = abs(left_frac - 0.5) + abs(top_frac - 0.5)

    # Star/chain pattern
    top_launch_count = max(launch_counts.values()) if launch_counts else 0
    star_degree = top_launch_count / max(len(path_dists), 1)
    n_unique_launch = len(launch_counts)
    chain_frac = n_unique_launch / max(len(path_dists), 1)

    # Density ratio: fast FF density vs all FF density
    try:
        fast_area = max(fast_hpwl ** 2, 1.0)
        all_area  = max((all_xs.max() - all_xs.min()) * (all_ys.max() - all_ys.min()), 1.0)
        density_ratio = min((len(all_fast) / fast_area) / (len(all_xs) / all_area), 10.0)
    except Exception:
        density_ratio = 1.0

    return {
        # Slack distribution
        'slack_max':       slack_max,
        'slack_p90':       slack_p90,
        'slack_p75':       slack_p75,
        'slack_spread':    slack_spread,
        'frac_self_loop':  frac_self_loop,
        'frac_fast_1ns':   frac_fast_1ns,
        # Fast path distances
        'fast_mean_dist':  float(path_dists.mean() / (dw + dh)),
        'fast_min_dist':   float(path_dists.min()  / (dw + dh)),
        'fast_p10_dist':   float(np.percentile(path_dists, 10) / (dw + dh)),
        'fast_max_dist':   float(path_dists.max()  / (dw + dh)),
        'fast_mean_dist_um': float(path_dists.mean()),
        'fast_min_dist_um':  float(path_dists.min()),
        # Spatial
        'fast_hpwl':       float(fast_hpwl_norm),
        'fast_cx_offset':  float(cx_offset),
        'fast_cy_offset':  float(cy_offset),
        'fast_x_std':      float(fast_x_std),
        'fast_y_std':      float(fast_y_std),
        'fast_eccentricity': float(fast_eccentricity),
        'fast_asymmetry':  float(asymmetry),
        # Topology
        'fast_star_degree': float(star_degree),
        'fast_chain_frac':  float(chain_frac),
        'fast_density_ratio': float(density_ratio),
    }


def _zeros(slack_max=0.0, slack_p90=0.0, slack_p75=0.0,
           slack_spread=0.0, frac_self_loop=0.0, frac_fast_1ns=0.0):
    return {
        'slack_max': slack_max, 'slack_p90': slack_p90, 'slack_p75': slack_p75,
        'slack_spread': slack_spread, 'frac_self_loop': frac_self_loop,
        'frac_fast_1ns': frac_fast_1ns,
        'fast_mean_dist': 0.0, 'fast_min_dist': 0.0, 'fast_p10_dist': 0.0,
        'fast_max_dist': 0.0, 'fast_mean_dist_um': 0.0, 'fast_min_dist_um': 0.0,
        'fast_hpwl': 0.0, 'fast_cx_offset': 0.0, 'fast_cy_offset': 0.0,
        'fast_x_std': 0.0, 'fast_y_std': 0.0, 'fast_eccentricity': 1.0,
        'fast_asymmetry': 0.0, 'fast_star_degree': 0.0, 'fast_chain_frac': 0.0,
        'fast_density_ratio': 1.0,
    }


def build_cache():
    # Collect all placement ids: training + OOD
    df = pd.read_csv(MANIFEST)
    pids = list(df['placement_id'].unique())
    for ood_csv in OOD_CSVS:
        if os.path.exists(ood_csv):
            ood_df = pd.read_csv(ood_csv)
            pids += list(ood_df['placement_id'].unique())
    pids = sorted(set(pids))
    print(f"{T()} Building fast-path cache for {len(pids)} placements...")

    cache = {}
    n_ok, n_fail = 0, 0

    for pid in pids:
        design = pid.split('_run_')[0]
        place_dir = os.path.join(PLACEMENT_DIR, pid)
        def_path  = os.path.join(place_dir, f'{design}.def')
        tim_path  = os.path.join(place_dir, 'timing_paths.csv')

        if not os.path.exists(def_path) or not os.path.exists(tim_path):
            n_fail += 1
            continue

        ff_positions, die_w, die_h, origin = parse_def_ff_positions(def_path)
        if ff_positions is None or len(ff_positions) == 0:
            n_fail += 1
            continue

        try:
            timing_df = pd.read_csv(tim_path)
            feats = compute_fast_path_features(ff_positions, die_w, die_h, origin, timing_df)
            cache[pid] = feats
            n_ok += 1
        except Exception as e:
            n_fail += 1

        if (n_ok + n_fail) % 100 == 0:
            print(f"  {T()} {n_ok+n_fail}/{len(pids)} ({n_ok} ok, {n_fail} fail)")

    print(f"{T()} Done: {n_ok} ok, {n_fail} fail → {OUT_CACHE}")
    with open(OUT_CACHE, 'wb') as f:
        pickle.dump(cache, f, protocol=4)

    # Show sample
    k = list(cache.keys())[0]
    print(f"\nSample ({k}):")
    for key, val in sorted(cache[k].items()):
        print(f"  {key}: {val:.4f}")

    return cache


if __name__ == '__main__':
    cache = build_cache()
    print(f"\n{T()} DONE — {len(cache)} placements cached")
