"""
build_caches.py — Build all feature caches from raw placement files.

Run this before build_models.py. Takes ~10-30 minutes depending on CPU.

Caches produced in caches/:
  def_cache.pkl       — DEF layout features per placement (FF positions, cell counts, geometry)
  saif_cache.pkl      — SAIF activity features per placement (toggle counts, signal probability)
  timing_cache.pkl    — Timing path statistics per placement (slack distribution)
  gravity_cache.pkl   — Logic-pull gravity features per placement (spatial imbalance)
  skew_spatial_cache.pkl — Critical-path spatial features per placement (skew head input)

fast_path_cache.pkl is built separately by build_fast_path_cache.py (hold head input).

Usage:
    python build_caches.py

Environment:
    SWIFTCTS_PLACEMENT_DIR — override path to placement_files directory
"""

import os, sys, re, time, pickle, warnings
import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings('ignore')

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

HERE          = os.path.dirname(os.path.abspath(__file__))
PLACEMENT_DIR = os.environ.get('SWIFTCTS_PLACEMENT_DIR',
                    os.path.join(HERE, 'dataset_with_def', 'placement_files'))
MANIFEST      = os.path.join(HERE, 'data', 'unified_manifest.csv')
CACHE_DIR     = os.path.join(HERE, 'caches')
os.makedirs(CACHE_DIR, exist_ok=True)

# Import parsers from swiftcts.py
sys.path.insert(0, HERE)
from swiftcts import _parse_def, _parse_saif, _parse_timing, _compute_gravity_features
from build_skew_cache import parse_def_ff_positions, compute_skew_features


def _build_def_saif_timing(df):
    """Parse DEF, SAIF, and timing_paths.csv for all placements."""
    def_cache, saif_cache, timing_cache = {}, {}, {}
    designs = df['design_name'].unique()
    all_pids = df['placement_id'].unique()
    n = len(all_pids)

    print(f"{T()} Building DEF / SAIF / timing caches for {n} placements...")
    ok = fail = 0

    for pid in sorted(all_pids):
        design = pid.split('_run_')[0]
        place_dir = os.path.join(PLACEMENT_DIR, pid)
        def_path    = os.path.join(place_dir, f'{design}.def')
        saif_path   = os.path.join(place_dir, f'{design}.saif')
        timing_path = os.path.join(place_dir, 'timing_paths.csv')

        missing = [p for p in [def_path, saif_path, timing_path] if not os.path.exists(p)]
        if missing:
            fail += 1
            continue

        try:
            def_cache[pid]    = _parse_def(def_path)
            saif_cache[pid]   = _parse_saif(saif_path)
            timing_cache[pid] = _parse_timing(timing_path)
            ok += 1
        except Exception as e:
            fail += 1

        if (ok + fail) % 50 == 0:
            print(f"  {T()} {ok+fail}/{n} done ({ok} ok, {fail} fail)")

    print(f"{T()} DEF/SAIF/timing: {ok} ok, {fail} fail")
    return def_cache, saif_cache, timing_cache


def _build_gravity(df, def_cache):
    """Compute gravity (logic-pull) features. Slow: ~5-30s per placement."""
    gravity_cache = {}
    all_pids = df['placement_id'].unique()
    n = len(all_pids)
    print(f"\n{T()} Building gravity cache for {n} placements (slow — ~10 min)...")
    ok = fail = 0

    for pid in sorted(all_pids):
        design = pid.split('_run_')[0]
        place_dir = os.path.join(PLACEMENT_DIR, pid)
        def_path    = os.path.join(place_dir, f'{design}.def')
        timing_path = os.path.join(place_dir, 'timing_paths.csv')

        if pid not in def_cache or not os.path.exists(def_path):
            fail += 1
            continue

        n_ff = def_cache[pid]['n_ff']
        try:
            gf = _compute_gravity_features(def_path, timing_path, n_ff)
            gravity_cache[pid] = gf
            ok += 1
        except Exception:
            fail += 1

        if (ok + fail) % 25 == 0:
            print(f"  {T()} {ok+fail}/{n} done ({ok} ok, {fail} fail)")

    print(f"{T()} Gravity: {ok} ok, {fail} fail")
    return gravity_cache


def _build_skew_spatial(df):
    """Compute critical-path spatial features for skew head."""
    skew_cache = {}
    all_pids = df['placement_id'].unique()
    n = len(all_pids)
    print(f"\n{T()} Building skew spatial cache for {n} placements...")
    ok = fail = 0

    for pid in sorted(all_pids):
        design = pid.split('_run_')[0]
        place_dir = os.path.join(PLACEMENT_DIR, pid)
        def_path    = os.path.join(place_dir, f'{design}.def')
        timing_path = os.path.join(place_dir, 'timing_paths.csv')

        if not os.path.exists(def_path) or not os.path.exists(timing_path):
            fail += 1
            continue

        try:
            ff_pos, dw, dh, origin = parse_def_ff_positions(def_path)
            if ff_pos is None or len(ff_pos) == 0:
                fail += 1
                continue
            td = pd.read_csv(timing_path)
            feats = compute_skew_features(ff_pos, dw, dh, origin, td)
            if feats is not None:
                skew_cache[pid] = feats
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1

        if (ok + fail) % 50 == 0:
            print(f"  {T()} {ok+fail}/{n} done ({ok} ok, {fail} fail)")

    print(f"{T()} Skew spatial: {ok} ok, {fail} fail")
    return skew_cache


def main():
    print("=" * 60)
    print("  SwiftCTS — Cache Builder")
    print("=" * 60)
    print(f"  Placement dir : {PLACEMENT_DIR}")
    print(f"  Cache output  : {CACHE_DIR}")

    # Load manifest (covers all 4 training designs)
    df = pd.read_csv(MANIFEST)
    df = df[df['design_name'].isin(['aes', 'ethmac', 'picorv32', 'sha256'])]
    # Deduplicate: one row per placement_id suffices for cache building
    df_pids = df.drop_duplicates('placement_id')
    print(f"\n  Placements: {len(df_pids)} across {df['design_name'].nunique()} designs")

    # ── DEF / SAIF / Timing ───────────────────────────────────────────────
    def_cache, saif_cache, timing_cache = _build_def_saif_timing(df_pids)

    for name, cache, path in [
        ('def_cache',    def_cache,    os.path.join(CACHE_DIR, 'def_cache.pkl')),
        ('saif_cache',   saif_cache,   os.path.join(CACHE_DIR, 'saif_cache.pkl')),
        ('timing_cache', timing_cache, os.path.join(CACHE_DIR, 'timing_cache.pkl')),
    ]:
        with open(path, 'wb') as f:
            pickle.dump(cache, f, protocol=4)
        print(f"  Saved {name}: {len(cache)} entries → {os.path.basename(path)}")

    # ── Gravity ───────────────────────────────────────────────────────────
    gravity_cache = _build_gravity(df_pids, def_cache)
    grav_path = os.path.join(CACHE_DIR, 'gravity_cache.pkl')
    with open(grav_path, 'wb') as f:
        pickle.dump(gravity_cache, f, protocol=4)
    print(f"  Saved gravity_cache: {len(gravity_cache)} entries → gravity_cache.pkl")

    # ── Skew spatial ──────────────────────────────────────────────────────
    skew_cache = _build_skew_spatial(df_pids)
    skew_path = os.path.join(CACHE_DIR, 'skew_spatial_cache.pkl')
    with open(skew_path, 'wb') as f:
        pickle.dump(skew_cache, f, protocol=4)
    print(f"  Saved skew_spatial_cache: {len(skew_cache)} entries → skew_spatial_cache.pkl")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{T()} All caches built.")
    print("\nNext steps:")
    print("  python build_fast_path_cache.py  # hold head cache")
    print("  python build_models.py           # train power + WL heads")
    print("  python update_skew_models.py     # add skew + hold heads to pkl")


if __name__ == '__main__':
    main()
