"""
swiftcts.py — SwiftCTS: Fast, Minimal-Feature CTS Outcome Surrogate
====================================================================

Predicts three CTS outcomes for unseen VLSI designs:
  - power_mW   : total clock power (milliwatts)
  - wl_mm      : total clock wirelength (millimetres)
  - skew_ns    : setup skew (nanoseconds, requires per-placement normalization)
  - hold_vio   : hold violation count

Quick Start
-----------
    from swiftcts import SwiftCTS

    model = SwiftCTS.load('saved_models/model.pkl')
    model.add_design('mydesign_run_001',
                     def_path    = 'placement_files/mydesign_run_001/mydesign.def',
                     saif_path   = 'placement_files/mydesign_run_001/mydesign.saif',
                     timing_path = 'placement_files/mydesign_run_001/timing_paths.csv',
                     t_clk       = 7.0)           # clock period in ns

    pred = model.predict('mydesign_run_001', cd=55, cs=20, mw=220, bd=100)
    print(pred.power_mW, pred.wl_mm, pred.skew_ns)

    # K-shot calibration (after seeing K labeled runs from the same placement)
    model.calibrate_power('mydesign_run_001', true_pw=[0.18, 0.19], pred_pw=[0.15, 0.16])
    model.calibrate_wl   ('mydesign_run_001', true_wl=[4.2e6, 4.3e6], pred_wl=[3.8e6, 3.9e6])

    # Pareto-optimal knob search
    pareto = model.optimize('mydesign_run_001', n=5000)

Architecture
------------
  Power  head : XGBRegressor,           20 features, normalizer = n_ff·f_ghz·avg_ds
  WL     head : LGBMRegressor+Ridge,    75 features, normalizer = sqrt(n_ff·die_area)
                (adaptive: LGB-only per-sample when Ridge extrapolates outside training target range)
  Skew   head : LGBMRegressor,          63 features, per-placement z-score
  Hold   head : LGBMRegressor,          66 features, log1p(hold_vio_count)

LODO Results (Leave-One-Design-Out, 4 training designs)
--------------------------------------------------------
  aes:      power 21.0%   WL 24.9%
  ethmac:   power  5.7%   WL  8.2%
  picorv32: power 18.5%   WL  5.7%
  sha256:   power 51.5%   WL  5.1%
  Mean:     power 24.2%   WL 11.0%   (vs 32.0% / 11.0% baseline)

OOD Results (K-shot calibration)
---------------------------------
  zipdiv      K=2: power 4.1%  WL 5.4%
  oc_jpegencode K=0: power 15.7% (zero-shot best for power)
              K=5: WL 18.2%
"""

import re, os, sys, warnings, pickle, time
import numpy as np
import pandas as pd
from collections import Counter, namedtuple
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

warnings.filterwarnings('ignore')

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

T_CLK_PRIOR = {
    'aes': 7.0, 'ethmac': 9.0, 'picorv32': 5.0,
    'sha256': 9.0, 'zipdiv': 5.0, 'oc_jpegencode': 7.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Output data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CTSPrediction:
    """All CTS outcome predictions for one (placement, knob) pair."""
    power_mW:       float
    wl_mm:          float
    skew_z:         float
    skew_ns:        Optional[float] = None
    skew_hold_z:    float = 0.0
    skew_hold_ns:   Optional[float] = None
    hold_vio:       float = 0.0
    pw_norm:        float = 1.0
    wl_norm:        float = 1.0

    def __repr__(self):
        sk = f"{self.skew_ns:.4f}ns" if self.skew_ns is not None else f"z={self.skew_z:.3f}"
        skh = f"{self.skew_hold_ns:.4f}ns" if self.skew_hold_ns is not None else f"z={self.skew_hold_z:.3f}"
        return (f"CTSPrediction(power={self.power_mW:.3f}mW  wl={self.wl_mm:.2f}mm  "
                f"skew_setup={sk}  skew_hold={skh}  hold_vio={self.hold_vio:.1f})")


# ─────────────────────────────────────────────────────────────────────────────
# Parsers (DEF, SAIF, timing) — copied from cts_surrogate_pkg
# ─────────────────────────────────────────────────────────────────────────────

def _gini(arr: np.ndarray) -> float:
    a = np.sort(np.abs(arr))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * a) / (n * a.sum())) - (n + 1) / n)


def _parse_def(path: str) -> dict:
    with open(path) as f:
        txt = f.read()
    u = int(re.search(r'UNITS DISTANCE MICRONS (\d+)', txt).group(1))
    x0, y0, x1, y1 = [float(v) / u for v in re.search(
        r'DIEAREA\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)',
        txt).groups()]
    dw, dh, da = x1 - x0, y1 - y0, (x1 - x0) * (y1 - y0)
    ct = Counter(re.findall(r'sky130_fd_sc_hd__(\w+)', txt))
    fk = ['tap', 'decap', 'fill', 'phy']
    nt  = sum(ct.values())
    ntp = sum(v for k, v in ct.items() if any(x in k for x in fk))
    na  = nt - ntp
    nff = sum(v for k, v in ct.items() if k.startswith('df') or k.startswith('ff'))
    nbf = sum(v for k, v in ct.items() if k.startswith('buf'))
    niv = sum(v for k, v in ct.items() if k.startswith('inv'))
    nxo = sum(v for k, v in ct.items() if k.startswith('xor') or k.startswith('xnor'))
    nmx = sum(v for k, v in ct.items() if k.startswith('mux'))
    nao = sum(v for k, v in ct.items() if k.startswith('and') or k.startswith('or'))
    nnn = sum(v for k, v in ct.items() if k.startswith('nand') or k.startswith('nor'))
    nc  = max(na - nff - nbf - niv, 0)
    ds  = []
    for k, v in ct.items():
        if not any(x in k for x in fk):
            m = re.search(r'_(\d+)$', k)
            if m:
                ds.extend([int(m.group(1))] * v)
    avg = np.mean(ds) if ds else 1.0
    fp  = r'-\s+\S+\s+(sky130_fd_sc_hd__df\w+)\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)'
    xy  = [(float(x) / u, float(y) / u) for _, x, y in re.findall(fp, txt)]
    xs, ys = np.array([p[0] for p in xy]), np.array([p[1] for p in xy])
    if len(xs) == 0:
        xs = ys = np.array([0.0])
    return dict(
        die_area=da, die_w=dw, die_h=dh, die_aspect=dw / (dh + 1e-6),
        ff_hpwl=(xs.max() - xs.min()) + (ys.max() - ys.min()),
        ff_spacing=np.sqrt(((xs.max() - xs.min()) * (ys.max() - ys.min()) + 1) / max(len(xy), 1)),
        ff_density=len(xy) / da, ff_cx=xs.mean() / dw, ff_cy=ys.mean() / dh,
        ff_x_std=xs.std() / dw, ff_y_std=ys.std() / dh,
        n_ff=len(xy), n_active=na, n_total=nt, n_tap=ntp,
        n_buf=nbf, n_inv=niv, n_comb=nc, n_xor_xnor=nxo, n_mux=nmx,
        n_and_or=nao, n_nand_nor=nnn,
        frac_xor=nxo / (na + 1), frac_mux=nmx / (na + 1),
        frac_and_or=nao / (na + 1), frac_nand_nor=nnn / (na + 1),
        frac_ff_active=nff / (na + 1), frac_buf_inv=(nbf + niv) / (na + 1),
        comb_per_ff=nc / (nff + 1), avg_ds=avg,
        std_ds=np.std(ds) if len(ds) > 1 else 0.0,
        p90_ds=np.percentile(ds, 90) if ds else 1.0,
        frac_ds4plus=sum(1 for d in ds if d >= 4) / (len(ds) + 1),
        cap_proxy=na * avg, ff_cap_proxy=len(xy) * avg,
    )


def _parse_saif(path: str) -> dict:
    tc_v = []; tt = tn = mk = 0; dur = None
    with open(path) as f:
        for ln in f:
            if '(DURATION' in ln:
                m = re.search(r'[\d.]+', ln)
                if m:
                    dur = float(m.group())
            m = re.search(r'\(TC\s+(\d+)\)', ln)
            if m:
                v = int(m.group(1)); tc_v.append(v); tn += 1; tt += v; mk = max(mk, v)
    if tn == 0 or mk == 0:
        return {}
    a = np.array(tc_v, float); mn = tt / tn
    return dict(n_nets=tn, rel_act=mn / mk, mean_sig_prob=0.0,
                tc_std_norm=a.std() / (mn + 1), frac_zero=(a == 0).mean(),
                frac_high_act=(a > mn * 2).mean(), log_n_nets=np.log1p(tn))


def _parse_timing(path: str) -> dict:
    sl = pd.read_csv(path)['slack'].values
    return dict(n_paths=len(sl), slack_mean=sl.mean(), slack_std=sl.std(),
                slack_min=sl.min(), slack_p10=np.percentile(sl, 10),
                slack_p50=np.percentile(sl, 50),
                frac_neg=(sl < 0).mean(), frac_tight=(sl < 0.5).mean(),
                frac_critical=(sl < 0.1).mean())


def _parse_skew_spatial(def_path: str, timing_path: str) -> dict:
    """Critical-path spatial features from DEF + timing_paths.csv."""
    try:
        sys.path.insert(0, PKG_DIR)  # build_skew_cache.py is co-located
        from build_skew_cache import parse_def_ff_positions, compute_skew_features
        ff_pos, dw, dh, origin = parse_def_ff_positions(def_path)
        td = pd.read_csv(timing_path)
        return compute_skew_features(ff_pos, dw, dh, origin, td) or {}
    except Exception:
        return {}


def _compute_gravity_features(def_path: str, timing_path: str, n_ff: int) -> dict:
    """Gravity (logic-pull) + timing-degree features from DEF + timing CSV."""
    try:
        with open(def_path) as f:
            txt = f.read()
    except Exception:
        return {}

    units_m = re.search(r'UNITS DISTANCE MICRONS (\d+)', txt)
    if not units_m:
        return {}
    u = int(units_m.group(1))
    die_m = re.search(
        r'DIEAREA\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)', txt)
    if not die_m:
        return {}
    x0, y0, x1, y1 = [float(v) / u for v in die_m.groups()]
    die_area = (x1 - x0) * (y1 - y0)
    die_scale = max(np.sqrt(die_area), 1e-6)

    cell_pos = {m.group(1): (float(m.group(2)) / u, float(m.group(3)) / u)
                for m in re.finditer(
                    r'-\s+(\S+)\s+sky130_fd_sc_hd__\S+\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d]+)\s+([\d]+)\s*\)',
                    txt)}
    if not cell_pos:
        return {}

    ff_names = set()
    clock_m = re.search(r'-\s+clk\s+\(.*?\);', txt, re.DOTALL)
    if clock_m:
        ff_names = set(re.findall(r'\(\s+((?!PIN)\S+)\s+CLK\s+\)', clock_m.group(0)))
    if not ff_names:
        ff_pat = re.compile(r'-\s+(\S+)\s+sky130_fd_sc_hd__df\w+\s+\+')
        ff_names = {m.group(1) for m in ff_pat.finditer(txt)}
    if not ff_names:
        return {}

    nets_m = re.search(r'\bNETS\s+\d+\s*;(.*?)\bEND NETS\b', txt, re.DOTALL)
    logic_pat = re.compile(r'^_\d+_$')
    gravity_vecs: dict = {ff: [] for ff in ff_names}

    if nets_m:
        conn_pat = re.compile(r'\(\s+(\S+)\s+\S+\s+\)')
        for block in nets_m.group(1).split(';'):
            conns = conn_pat.findall(block)
            net_ff = [c for c in conns if c in ff_names]
            net_lg = [c for c in conns if logic_pat.match(c) and c in cell_pos]
            if net_ff and net_lg:
                for ff in net_ff:
                    for lg in net_lg:
                        gravity_vecs[ff].append(cell_pos[lg])

    mags_abs, mags_norm, dx_vals, dy_vals = [], [], [], []
    for ff in ff_names:
        if ff not in cell_pos or not gravity_vecs.get(ff):
            continue
        fx, fy = cell_pos[ff]
        lx = [c[0] for c in gravity_vecs[ff]]
        ly = [c[1] for c in gravity_vecs[ff]]
        dx = np.mean(lx) - fx
        dy = np.mean(ly) - fy
        mag = np.sqrt(dx ** 2 + dy ** 2)
        mags_abs.append(mag)
        mags_norm.append(mag / die_scale)
        dx_vals.append(abs(dx))
        dy_vals.append(abs(dy))

    grav: dict = {}
    if mags_abs:
        ma = np.array(mags_abs)
        mn_arr = np.array(mags_norm)
        grav = {
            'grav_abs_mean':    float(ma.mean()),
            'grav_abs_std':     float(ma.std()),
            'grav_abs_p25':     float(np.percentile(ma, 25)),
            'grav_abs_p75':     float(np.percentile(ma, 75)),
            'grav_abs_p90':     float(np.percentile(ma, 90)),
            'grav_abs_max':     float(ma.max()),
            'grav_abs_cv':      float(ma.std() / (ma.mean() + 1e-9)),
            'grav_abs_gini':    _gini(ma),
            'grav_norm_mean':   float(mn_arr.mean()),
            'grav_norm_cv':     float(mn_arr.std() / (mn_arr.mean() + 1e-9)),
            'grav_dx_mean':     float(np.mean(dx_vals)),
            'grav_dy_mean':     float(np.mean(dy_vals)),
            'grav_anisotropy':  float(abs(np.mean(dx_vals) - np.mean(dy_vals)) /
                                       (np.mean(dx_vals) + np.mean(dy_vals) + 1e-9)),
            'grav_frac_local':  float((ma < np.percentile(ma, 50)).mean()),
            'grav_frac_longrange': float((ma > np.percentile(ma, 90)).mean()),
        }

    tp_feats: dict = {}
    try:
        tp = pd.read_csv(timing_path)
        all_ffs = list(tp['launch_flop']) + list(tp['capture_flop'])
        deg_counter = Counter(all_ffs)
        degs = np.array(list(deg_counter.values()), dtype=float)
        nf_total = max(n_ff, 1)
        tp_feats = {
            'tp_degree_mean':   float(degs.mean()),
            'tp_degree_std':    float(degs.std()),
            'tp_degree_max':    float(degs.max()),
            'tp_degree_p90':    float(np.percentile(degs, 90)),
            'tp_degree_cv':     float(degs.std() / (degs.mean() + 1e-9)),
            'tp_degree_gini':   _gini(degs),
            'tp_frac_involved': float(len(deg_counter) / nf_total),
            'tp_paths_per_ff':  float(len(tp) / nf_total),
            'tp_frac_hub':      float((degs > 2 * degs.mean()).mean()),
        }
    except Exception:
        pass

    return {**grav, **tp_feats}


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_synth(s) -> tuple:
    if pd.isna(s):
        return 0.5, 2.0, 0.5
    s = str(s).upper()
    sd = 1.0 if 'DELAY' in s else 0.0
    try:
        level = float(s.split()[-1])
    except Exception:
        level = 2.0
    return sd, level, sd * level / 4.0


def _build_pw_features(d, s, t, f_ghz, sa, cd, n_ff=None, die_area=None) -> list:
    """Build 20-dim minimal power feature vector."""
    nf = d['n_ff']; na = d['n_active']
    rel = s.get('rel_act', 0.05)
    fx  = d['frac_xor']; fm = d['frac_mux']
    sm  = t['slack_mean']; ft = t['frac_tight']

    return [
        np.log1p(na * rel * f_ghz),          # log_act_scale
        np.log1p(nf),                          # log_n_ff
        rel,                                   # rel_act
        np.log1p(fm * na),                     # log_mux_active
        np.log1p(d['cap_proxy']),              # log_cap_proxy
        fm,                                    # frac_mux
        s.get('frac_high_act', 0.0),           # frac_high_act
        d['frac_and_or'],                      # frac_and_or
        t['slack_std'],                        # slack_std
        ft,                                    # frac_tight
        s.get('mean_sig_prob', 0.0),           # mean_sig_prob
        t['slack_p10'],                        # slack_p10
        t['slack_p50'],                        # slack_p50
        sm * fx,                               # slack_xor
        f_ghz,                                 # f_ghz
        sa,                                    # synth_area
        np.log1p(fx * na),                     # log_xor_active
        s.get('log_n_nets', 0.0),              # log_n_nets
        d['frac_nand_nor'],                    # frac_nand_nor
        np.log1p(cd * nf / (d['die_area'] + 1e-9)),  # log_cd_dens
    ]  # 20 dims


def _build_wl_features(d, s, t, f_ghz, t_clk, core_util, density, cd, cs, mw, bd, gf) -> list:
    """Build 75-dim WL feature vector (matches minimal_model.py exactly)."""
    nf  = d['n_ff']; na = d['n_active']; da = d['die_area']
    hpwl = d['ff_hpwl']; sp = d['ff_spacing']
    fx   = d['frac_xor']; fm = d['frac_mux']
    cpf  = d['comb_per_ff']; nc = d['n_comb']
    rel  = s.get('rel_act', 0.05)
    n_nets = s.get('n_nets', 1)

    return [
        np.log1p(nf),
        np.log1p(da),
        np.log1p(hpwl),
        np.log1p(sp),
        d['die_aspect'],
        1.0,                                    # aspect_ratio placeholder
        d['ff_cx'], d['ff_cy'],
        d['ff_x_std'], d['ff_y_std'],
        fx, fm,
        d['frac_and_or'], d['frac_nand_nor'],
        d['frac_ff_active'], d['frac_buf_inv'],
        cpf,
        d['avg_ds'], d['std_ds'], d['p90_ds'], d['frac_ds4plus'],
        np.log1p(d['cap_proxy']),
        rel,
        s.get('mean_sig_prob', 0.0),
        s.get('tc_std_norm', 0.0),
        s.get('frac_zero', 0.0),
        s.get('frac_high_act', 0.0),
        s.get('log_n_nets', 0.0),
        n_nets / (nf + 1),
        f_ghz, t_clk,
        core_util, density,
        np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
        cd, cs, mw, bd,
        fx * cpf,
        rel * fx,
        rel * (1 - d['frac_ff_active']),
        np.log1p(cd * nf / (da + 1e-9)),
        np.log1p(cs * sp),
        np.log1p(mw * hpwl),
        np.log1p(nf / (cs + 1e-9)),
        core_util * density,
        np.log1p(na * rel * f_ghz),
        np.log1p(fx * na),
        np.log1p(fm * na),
        np.log1p(cpf * nf),
        # Gravity (9 features + 3 interactions)
        gf.get('grav_abs_mean', 0.0),
        gf.get('grav_abs_std', 0.0),
        gf.get('grav_abs_p75', 0.0),
        gf.get('grav_abs_p90', 0.0),
        gf.get('grav_abs_cv', 0.0),
        gf.get('grav_abs_gini', 0.0),
        gf.get('grav_norm_mean', 0.0),
        gf.get('grav_norm_cv', 0.0),
        gf.get('grav_anisotropy', 0.0),
        gf.get('grav_abs_mean', 0.0) * cd,
        gf.get('grav_abs_mean', 0.0) * mw,
        gf.get('grav_abs_mean', 0.0) / (sp + 1),
        # Timing-path degree (7 features)
        gf.get('tp_degree_mean', 0.0),
        gf.get('tp_degree_cv', 0.0),
        gf.get('tp_degree_gini', 0.0),
        gf.get('tp_degree_p90', 0.0),
        gf.get('tp_frac_involved', 0.0),
        gf.get('tp_paths_per_ff', 0.0),
        gf.get('tp_frac_hub', 0.0),
        # Scale features (3 features)
        np.log1p(da / (nf + 1)),
        np.log1p(nc),
        cpf * np.log1p(nf),
    ]  # 75 dims


def _build_sk_features(d, s, t, sk, cd, cs, mw, bd) -> list:
    """Build 63-dim skew feature vector (matches cts_surrogate_pkg exactly)."""
    nf   = d['n_ff']; da = d['die_area']
    hpwl = d['ff_hpwl']; sp = d['ff_spacing']
    fx   = d['frac_xor']; cpf = d['comb_per_ff']; av = d['avg_ds']
    rel  = s.get('rel_act', 0.05)
    sm   = t['slack_mean']; fn = t['frac_neg']; ft = t['frac_tight']

    kn = [np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd]  # 8

    ck = [
        sk.get('crit_max_dist', 0.),    sk.get('crit_mean_dist', 0.),
        sk.get('crit_p90_dist', 0.),    sk.get('crit_ff_hpwl', 0.),
        sk.get('crit_cx_offset', 0.),   sk.get('crit_cy_offset', 0.),
        sk.get('crit_x_std', 0.),       sk.get('crit_y_std', 0.),
        sk.get('crit_frac_boundary', 0.), sk.get('crit_star_degree', 0.),
        sk.get('crit_chain_frac', 0.),  sk.get('crit_asymmetry', 0.),
        sk.get('crit_eccentricity', 1.), sk.get('crit_density_ratio', 1.),
        np.log1p(sk.get('crit_max_dist_um', hpwl)),
        np.log1p(sk.get('crit_mean_dist_um', hpwl / 2)),
    ]  # 16

    cm_um  = sk.get('crit_max_dist_um', hpwl)
    cmn_um = sk.get('crit_mean_dist_um', hpwl / 2)
    cs_v   = sk.get('crit_star_degree', 0.)
    ca_v   = sk.get('crit_asymmetry', 0.)
    cd_v   = sk.get('crit_density_ratio', 1.)
    cc_v   = sk.get('crit_chain_frac', 0.)
    ch_v   = sk.get('crit_ff_hpwl', 0.)
    cx_v   = sk.get('crit_cx_offset', 0.)
    cy_v   = sk.get('crit_cy_offset', 0.)

    ski = [
        cd / (sp + 1), bd / (cm_um + 1), mw / (cm_um + 1),
        cs_v * cd, ca_v * mw, cd_v * cs,
        sk.get('crit_max_dist', 0.) * cd, ca_v * sk.get('crit_max_dist', 0.),
        fn * cs_v, ft * cc_v, ch_v / (cs + 1),
        np.log1p(cm_um / (cd + 1)), np.log1p(cm_um / (bd + 1)),
        np.log1p(cm_um / (mw + 1)),
        cx_v * cd, cy_v * mw, np.log1p(nf / cs) * ch_v,
    ]  # 17

    sk_ctx = [
        np.log1p(nf), np.log1p(da), np.log1p(hpwl), np.log1p(sp), d['die_aspect'],
        d['ff_cx'], d['ff_cy'], d['ff_x_std'], d['ff_y_std'],
        fx, cpf, av, rel, s.get('mean_sig_prob', 0.),
        sm, t['slack_std'], t['slack_min'], t['slack_p10'],
        fn, ft, t['frac_critical'], np.log1p(t['n_paths'] / (nf + 1)),
    ]  # 22

    return sk_ctx + kn + ck + ski  # 63 dims


def _build_hv_features(d, s, t, sk, cd, cs, mw, bd, f_ghz, t_clk, core_util, density) -> list:
    """Build 66-dim hold violation feature vector."""
    nf   = d['n_ff']; na = d['n_active']; da = d['die_area']
    hpwl = d['ff_hpwl']; sp = d['ff_spacing']
    fx   = d['frac_xor']; fm = d['frac_mux']
    cpf  = d['comb_per_ff']; nc = d['n_comb']
    rel  = s.get('rel_act', 0.05)
    n_nets = s.get('n_nets', 1)

    ctx = [
        np.log1p(nf), np.log1p(da), np.log1p(hpwl), np.log1p(sp),
        d['die_aspect'], 1.0,
        d['ff_cx'], d['ff_cy'], d['ff_x_std'], d['ff_y_std'],
        d['frac_xor'], d['frac_mux'], d['frac_and_or'], d['frac_nand_nor'],
        d['frac_ff_active'], d['frac_buf_inv'], d['comb_per_ff'],
        d['avg_ds'], d['std_ds'], d['p90_ds'], d['frac_ds4plus'],
        np.log1p(d['cap_proxy']),
        s.get('rel_act', 0.05), s.get('mean_sig_prob', 0.0),
        s.get('tc_std_norm', 0.0), s.get('frac_zero', 0.0),
        s.get('frac_high_act', 0.0), s.get('log_n_nets', 0.0),
        n_nets / (nf + 1),
    ]  # 29
    dw = [f_ghz, t_clk, core_util, density]  # 4
    kn = [np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd]  # 8
    iw = [
        fx * cpf, rel * fx, rel * (1 - d['frac_ff_active']),
        np.log1p(cd * nf / (da + 1e-9)), np.log1p(cs * sp),
        np.log1p(mw * hpwl), np.log1p(nf / (cs + 1e-9)),
        core_util * density, np.log1p(na * rel * f_ghz),
        np.log1p(fx * na), np.log1p(fm * na), np.log1p(cpf * nf),
    ]  # 12

    cm_um  = sk.get('crit_max_dist_um', hpwl)
    cs_v   = sk.get('crit_star_degree', 0.)
    ca_v   = sk.get('crit_asymmetry', 0.)
    cc_v   = sk.get('crit_chain_frac', 0.)

    hp = [
        np.log1p(nf / (cs + 1e-9)), np.log1p(cs * sp), np.log1p(cd / (sp + 1)),
        np.log1p(bd / (hpwl + 1)), bd / (cm_um + 1e-3),
        cs_v * cs, cc_v * bd, ca_v * cd,
        np.log1p(sk.get('crit_max_dist', 0.) * bd),
    ]  # 9

    nt4 = [0.0, 0.0, 0.0, 0.0]  # net features (rudy_mean, rudy_p90, frac_high_fanout, rsmt)

    return ctx + dw + kn + iw + hp + nt4  # 66 dims


def _fix_nan(X: np.ndarray) -> np.ndarray:
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            good = ~bad
            X[bad, c] = np.nanmedian(X[good, c]) if good.any() else 0.0
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engine
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngine:
    """Parses DEF/SAIF/timing once per placement; builds features on demand."""

    def __init__(self):
        self._dc:  Dict[str, dict] = {}
        self._sc:  Dict[str, dict] = {}
        self._tc:  Dict[str, dict] = {}
        self._skc: Dict[str, dict] = {}
        self._gc:  Dict[str, dict] = {}
        self._tclk: Dict[str, float] = {}

    def load_caches(self, def_cache, saif_cache, timing_cache,
                    skew_cache=None, gravity_cache=None):
        self._dc.update(def_cache)
        self._sc.update(saif_cache)
        self._tc.update(timing_cache)
        if skew_cache:
            self._skc.update(skew_cache)
        if gravity_cache:
            self._gc.update(gravity_cache)

    def add_placement(self, pid: str, def_path: str, saif_path: str,
                      timing_path: str, t_clk: float = 7.0):
        """Parse raw files for a new placement (unseen design OK)."""
        self._dc[pid]   = _parse_def(def_path)
        self._sc[pid]   = _parse_saif(saif_path)
        self._tc[pid]   = _parse_timing(timing_path)
        self._skc[pid]  = _parse_skew_spatial(def_path, timing_path)
        # Gravity: compute on-the-fly for new designs (may be slow)
        n_ff = self._dc[pid]['n_ff']
        self._gc[pid]   = _compute_gravity_features(def_path, timing_path, n_ff)
        self._tclk[pid] = t_clk

    def has(self, pid: str) -> bool:
        return pid in self._dc

    def build(self, pid: str, cd: float, cs: float, mw: float, bd: float,
              t_clk: float = 7.0, core_util: float = 0.55, density: float = 0.5,
              synth: tuple = (0., 0., 1.)
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Build all feature vectors. Returns X_pw(20), X_wl(75), X_sk(63), X_hv(66), pw_norm, wl_norm."""
        d   = self._dc[pid]
        s   = self._sc[pid]
        t   = self._tc[pid]
        sk  = self._skc.get(pid, {})
        gf  = self._gc.get(pid, {})
        _, _, sa = synth
        f_ghz = 1.0 / t_clk

        nf = d['n_ff']; da = d['die_area']; av = d['avg_ds']
        pw_norm = max(nf * f_ghz * av, 1e-10)
        wl_norm = max(np.sqrt(nf * da), 1e-3)

        X_pw = np.array(_build_pw_features(d, s, t, f_ghz, sa, cd), dtype=np.float64)
        X_wl = np.array(_build_wl_features(d, s, t, f_ghz, t_clk, core_util, density,
                                            cd, cs, mw, bd, gf), dtype=np.float64)
        X_sk = np.array(_build_sk_features(d, s, t, sk, cd, cs, mw, bd), dtype=np.float64)
        X_hv = np.array(_build_hv_features(d, s, t, sk, cd, cs, mw, bd,
                                            f_ghz, t_clk, core_util, density), dtype=np.float64)
        return X_pw, X_wl, X_sk, X_hv, pw_norm, wl_norm

    def batch_build(self, pid, cd_arr, cs_arr, mw_arr, bd_arr,
                    t_clk=7.0, core_util=0.55, density=0.5, synth=(0., 0., 1.)):
        """Build feature matrices for N knob configs simultaneously."""
        X0_pw, X0_wl, X0_sk, X0_hv, pw_norm, wl_norm = self.build(
            pid, float(np.median(cd_arr)), float(np.median(cs_arr)),
            float(np.median(mw_arr)), float(np.median(bd_arr)),
            t_clk, core_util, density, synth)

        d  = self._dc[pid]
        nf = d['n_ff']; da = d['die_area']; hpwl = d['ff_hpwl']; sp = d['ff_spacing']

        N = len(cd_arr)

        # Power knob positions in 20-dim vector: index 19 = log_cd_dens (only knob)
        # There are no other knobs in the 20-dim power vector
        # We need to rebuild all N rows properly for batch (knobs vary)
        # For power: only log_cd_dens (feature 19) varies with knobs
        Xpw = np.tile(X0_pw, (N, 1)).astype(np.float64)
        Xpw[:, 19] = np.log1p(cd_arr * nf / (da + 1e-9))  # log_cd_dens

        # WL knob positions (from _build_wl_features order):
        # indices 33-36: log(cd,cs,mw,bd), 37-40: cd,cs,mw,bd
        # indices 41-47: interactions
        Xwl = np.tile(X0_wl, (N, 1)).astype(np.float64)
        Xwl[:, 33] = np.log1p(cd_arr)
        Xwl[:, 34] = np.log1p(cs_arr)
        Xwl[:, 35] = np.log1p(mw_arr)
        Xwl[:, 36] = np.log1p(bd_arr)
        Xwl[:, 37] = cd_arr
        Xwl[:, 38] = cs_arr
        Xwl[:, 39] = mw_arr
        Xwl[:, 40] = bd_arr
        # interaction features
        Xwl[:, 41] = d['frac_xor'] * d['comb_per_ff']   # const
        Xwl[:, 42] = d.get('rel_act_placeholder', 0.)   # placeholder
        # Recalculate interactions properly
        rel = self._sc[pid].get('rel_act', 0.05)
        Xwl[:, 42] = rel * d['frac_xor']
        Xwl[:, 43] = rel * (1 - d['frac_ff_active'])
        Xwl[:, 44] = np.log1p(cd_arr * nf / (da + 1e-9))
        Xwl[:, 45] = np.log1p(cs_arr * sp)
        Xwl[:, 46] = np.log1p(mw_arr * hpwl)
        Xwl[:, 47] = np.log1p(nf / (cs_arr + 1e-9))
        # index 48: core_util * density — constant, already set
        # Gravity interaction features that depend on knobs (indices 62, 63)
        grav_mean = self._gc.get(pid, {}).get('grav_abs_mean', 0.0)
        Xwl[:, 62] = grav_mean * cd_arr   # grav_abs_mean * cd
        Xwl[:, 63] = grav_mean * mw_arr   # grav_abs_mean * mw

        # Skew knob positions (indices 22-29: log+raw knobs)
        Xsk = np.tile(X0_sk, (N, 1)).astype(np.float64)
        Xsk[:, 22] = np.log1p(cd_arr)
        Xsk[:, 23] = np.log1p(cs_arr)
        Xsk[:, 24] = np.log1p(mw_arr)
        Xsk[:, 25] = np.log1p(bd_arr)
        Xsk[:, 26] = cd_arr
        Xsk[:, 27] = cs_arr
        Xsk[:, 28] = mw_arr
        Xsk[:, 29] = bd_arr
        # Skew ski interaction block (indices 46-62): update all knob-dependent terms
        # Constants from skew spatial cache (geometry, not knob-dependent)
        sk_feat = self._skc.get(pid, {})
        cm_um    = sk_feat.get('crit_max_dist_um', hpwl)
        crit_md  = sk_feat.get('crit_max_dist', 0.)
        cs_v     = sk_feat.get('crit_star_degree', 0.)
        ca_v     = sk_feat.get('crit_asymmetry', 0.)
        cd_v     = sk_feat.get('crit_density_ratio', 1.)
        ch_v     = sk_feat.get('crit_ff_hpwl', 0.)
        cx_v     = sk_feat.get('crit_cx_offset', 0.)
        cy_v     = sk_feat.get('crit_cy_offset', 0.)
        Xsk[:, 46] = cd_arr / (sp + 1)                          # cd/ff_spacing
        Xsk[:, 47] = bd_arr / (cm_um + 1)                       # bd/crit_max
        Xsk[:, 48] = mw_arr / (cm_um + 1)                       # mw/crit_max
        Xsk[:, 49] = cs_v * cd_arr                              # star_deg*cd
        Xsk[:, 50] = ca_v * mw_arr                              # asymm*mw
        Xsk[:, 51] = cd_v * cs_arr                              # dens*cs
        Xsk[:, 52] = crit_md * cd_arr                           # cmax_dist*cd
        # indices 53,54,55: asymm*cmax, frac_neg*star, frac_tgt*chain — all constant
        Xsk[:, 56] = ch_v / (cs_arr + 1)                        # crit_hpwl/(cs+1)
        Xsk[:, 57] = np.log1p(cm_um / (cd_arr + 1))             # log(cmax/cd)
        Xsk[:, 58] = np.log1p(cm_um / (bd_arr + 1))             # log(cmax/bd)
        Xsk[:, 59] = np.log1p(cm_um / (mw_arr + 1))             # log(cmax/mw)
        Xsk[:, 60] = cx_v * cd_arr                              # cx*cd
        Xsk[:, 61] = cy_v * mw_arr                              # cy*mw
        Xsk[:, 62] = np.log1p(nf / (cs_arr + 1e-9)) * ch_v     # log(nff/cs)*chpwl

        # HV knob positions (indices 33-40: log+raw knobs)
        Xhv = np.tile(X0_hv, (N, 1)).astype(np.float64)
        Xhv[:, 33] = np.log1p(cd_arr)
        Xhv[:, 34] = np.log1p(cs_arr)
        Xhv[:, 35] = np.log1p(mw_arr)
        Xhv[:, 36] = np.log1p(bd_arr)
        Xhv[:, 37] = cd_arr
        Xhv[:, 38] = cs_arr
        Xhv[:, 39] = mw_arr
        Xhv[:, 40] = bd_arr
        # HV iw interaction block (indices 41-52): update knob-dependent terms
        Xhv[:, 44] = np.log1p(cd_arr * nf / (da + 1e-9))       # log_cd_dens
        Xhv[:, 45] = np.log1p(cs_arr * sp)                      # log_cs_sp
        Xhv[:, 46] = np.log1p(mw_arr * hpwl)                   # log_mw_hpwl
        Xhv[:, 47] = np.log1p(nf / (cs_arr + 1e-9))            # log_nff_cs
        # HV hp block (indices 53-61): all vary with knobs
        Xhv[:, 53] = np.log1p(nf / (cs_arr + 1e-9))            # log(nff/cs)
        Xhv[:, 54] = np.log1p(cs_arr * sp)                      # log(cs*sp)
        Xhv[:, 55] = np.log1p(cd_arr / (sp + 1))               # log(cd/sp)
        Xhv[:, 56] = np.log1p(bd_arr / (hpwl + 1))             # log(bd/hpwl)
        Xhv[:, 57] = bd_arr / (cm_um + 1e-3)                   # bd/crit_max
        Xhv[:, 58] = cs_v * cs_arr                              # star_deg*cs
        Xhv[:, 59] = sk_feat.get('crit_chain_frac', 0.) * bd_arr  # chain*bd
        Xhv[:, 60] = ca_v * cd_arr                              # asymm*cd
        Xhv[:, 61] = np.log1p(crit_md * bd_arr)                # log(crit_max*bd)

        return Xpw, Xwl, Xsk, Xhv, pw_norm, wl_norm


# ─────────────────────────────────────────────────────────────────────────────
# Prediction heads
# ─────────────────────────────────────────────────────────────────────────────

class _Heads:
    def __init__(self, mdl):
        self.m_pw  = mdl['model_power']
        self.sc_pw = mdl['scaler_power']
        self.m_lgb = mdl['model_wl_lgb']
        self.m_rdg = mdl['model_wl_ridge']
        self.sc_wl = mdl['scaler_wl']
        self.alpha = mdl.get('wl_blend_alpha', 0.3)
        # Ridge is used per-sample only when its raw prediction is within the training
        # target range [wl_y_min, wl_y_max]. Outside that range → LGB-only.
        # This threshold is derived from training data, not a hardcoded n_ff multiplier.
        lodo = mdl.get('lodo', {})
        self.wl_y_min = lodo.get('wl_y_min', 0.0) - 1.0
        self.wl_y_max = lodo.get('wl_y_max', 6.0) + 1.0
        self.m_sk   = mdl['model_skew']
        self.sc_sk  = mdl['scaler_skew']
        # Pruned feature indices for skew_setup (15 features)
        self.sk_feat_idx = mdl.get('sk_feat_idx', np.arange(63))
        # Hold architecture: pred_hold_z = -pred_setup_z
        # skew_hold ≈ -skew_setup (r=-0.94 global, -0.90 within-placement).
        # Residual has max feature correlation 0.038 — not learnable.
        # LODO accuracy: negation=0.0746ns vs separate=0.0747ns (identical).
        # Fallback: use stored model_skew_hold if flag absent (backward compat).
        self.hold_uses_negation = mdl.get('hold_uses_negation', False)
        self.m_skh  = mdl.get('model_skew_hold')
        self.sc_skh = mdl.get('scaler_skew_hold')
        self.skh_feat_idx = mdl.get('skh_feat_idx', np.arange(63))
        self.m_hv   = mdl.get('model_hold_vio')
        self.sc_hv  = mdl.get('scaler_hold_vio')
        self.lodo   = lodo

    def _predict_wl(self, Xs, use_ridge: bool) -> np.ndarray:
        """Compute log(wl/wl_norm).
        Blend LGB + Ridge where Ridge is interpolating (raw prediction within training
        target range). Fall back to LGB-only per-sample when Ridge extrapolates outside
        [wl_y_min, wl_y_max] — Ridge's linear model overshoots wildly beyond the
        training distribution boundary. Threshold derived from training data, not n_ff.
        """
        lgb_pred = self.m_lgb.predict(Xs)
        if use_ridge:
            rdg_raw = self.m_rdg.predict(Xs)
            in_range = (rdg_raw >= self.wl_y_min) & (rdg_raw <= self.wl_y_max)
            rdg_clipped = np.clip(rdg_raw, self.wl_y_min, self.wl_y_max)
            blended = self.alpha * lgb_pred + (1 - self.alpha) * rdg_clipped
            return np.where(in_range, blended, lgb_pred)
        return lgb_pred

    def predict_single(self, X_pw, X_wl, X_sk, X_hv, pw_norm, wl_norm,
                       sk_mu=None, sk_sig=None, use_ridge=True):
        pw    = float(np.exp(self.m_pw.predict(self.sc_pw.transform(X_pw.reshape(1, -1)))[0])) * pw_norm
        Xs    = self.sc_wl.transform(X_wl.reshape(1, -1))
        wl    = float(np.exp(self._predict_wl(Xs, use_ridge)[0])) * wl_norm
        sk_in = X_sk[self.sk_feat_idx].reshape(1, -1)
        sk_z  = float(self.m_sk.predict(self.sc_sk.transform(sk_in))[0])
        sk_ns = sk_z * sk_sig + sk_mu if sk_sig is not None else None
        # Hold: negate setup z-score (skew_hold ≈ -skew_setup, r=-0.94).
        # Falls back to separate model if hold_uses_negation not set in pkl.
        if self.hold_uses_negation:
            skh_z = -sk_z
            skh_ns = skh_z * sk_sig + sk_mu if sk_sig is not None else None
        elif self.m_skh is not None:
            skh_in = X_sk[self.skh_feat_idx].reshape(1, -1)
            skh_z  = float(self.m_skh.predict(self.sc_skh.transform(skh_in))[0])
            skh_ns = skh_z * sk_sig + sk_mu if sk_sig is not None else None
        else:
            skh_z = 0.0; skh_ns = None
        hv_z  = float(self.m_hv.predict(self.sc_hv.transform(X_hv.reshape(1, -1)))[0]) if self.m_hv is not None else 0.0
        return CTSPrediction(
            power_mW=pw * 1000, wl_mm=wl / 1000,
            skew_z=sk_z, skew_ns=sk_ns,
            skew_hold_z=skh_z, skew_hold_ns=skh_ns,
            hold_vio=float(np.expm1(np.clip(hv_z, 0, 20))),
            pw_norm=pw_norm, wl_norm=wl_norm)

    def predict_batch(self, Xpw, Xwl, Xsk, Xhv, pw_norm, wl_norm,
                      sk_mu=None, sk_sig=None, use_ridge=True):
        pw    = np.exp(self.m_pw.predict(self.sc_pw.transform(Xpw))) * pw_norm
        Xs    = self.sc_wl.transform(Xwl)
        wl    = np.exp(self._predict_wl(Xs, use_ridge)) * wl_norm
        sk_z  = self.m_sk.predict(self.sc_sk.transform(Xsk[:, self.sk_feat_idx]))
        sk_ns = sk_z * sk_sig + sk_mu if sk_sig is not None else sk_z
        if self.hold_uses_negation:
            skh_z = -sk_z
        elif self.m_skh is not None:
            skh_z = self.m_skh.predict(self.sc_skh.transform(Xsk[:, self.skh_feat_idx]))
        else:
            skh_z = np.zeros(len(sk_z))
        hv_z  = self.m_hv.predict(self.sc_hv.transform(Xhv)) if self.m_hv is not None else np.zeros(len(sk_z))
        hv    = np.expm1(np.clip(hv_z, 0, 20))
        return pw * 1000, wl / 1000, sk_ns, skh_z, hv


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class SwiftCTS:
    """
    SwiftCTS — physics-informed CTS outcome predictor.

    Uses minimal feature sets (20 for power, 75 for WL) selected by
    LODO-consistent importance ranking. Adaptive WL blend protects
    against Ridge extrapolation (per-sample: LGB-only when Ridge predicts outside training target range).

    See module docstring for quick start and full API.
    """

    def __init__(self):
        self.features = FeatureEngine()
        self._heads: Optional[_Heads] = None
        self._t_clk: Dict[str, float] = {}
        self._pw_cal: Dict[str, float] = {}  # multiplicative calibration scales
        self._wl_cal: Dict[str, float] = {}
        self.lodo_results: dict = {}

    @classmethod
    def load(cls, model_path: str) -> 'SwiftCTS':
        """Load a saved model from pickle."""
        obj = cls()
        with open(model_path, 'rb') as f:
            mdl = pickle.load(f)
        obj._heads = _Heads(mdl)
        obj.lodo_results = mdl.get('lodo', {})
        return obj

    @classmethod
    def load_with_caches(cls, model_path: str, def_cache, saif_cache, timing_cache,
                         skew_cache=None, gravity_cache=None) -> 'SwiftCTS':
        """Load model + caches in one call."""
        obj = cls.load(model_path)
        obj.features.load_caches(def_cache, saif_cache, timing_cache,
                                 skew_cache, gravity_cache)
        return obj

    def add_design(self, name_or_pid: str, def_path: str, saif_path: str,
                   timing_path: str, t_clk: float = 7.0):
        """Register a new placement from raw files. Supports unseen OOD designs."""
        self.features.add_placement(name_or_pid, def_path, saif_path, timing_path, t_clk)
        self._t_clk[name_or_pid] = t_clk

    def _get_t_clk(self, pid: str) -> float:
        design = pid.split('_run_')[0] if '_run_' in pid else pid
        return self._t_clk.get(pid, T_CLK_PRIOR.get(design, 7.0))

    def calibrate_power(self, pid: str, true_pw, pred_pw):
        """
        K-shot multiplicative calibration for power (log-space).
        After observing K labeled samples: call this with their true and predicted values.
        Applies scale = exp(mean(log(true/pred))) at predict time.
        """
        self._pw_cal[pid] = float(
            np.exp(np.mean(np.log(np.array(true_pw) / (np.array(pred_pw) + 1e-12)))))

    def calibrate_wl(self, pid: str, true_wl, pred_wl):
        """
        K-shot multiplicative calibration for WL (log-space).
        Same as calibrate_power but for wirelength.
        """
        self._wl_cal[pid] = float(
            np.exp(np.mean(np.log(np.array(true_wl) / (np.array(pred_wl) + 1e-12)))))

    def _use_ridge(self, pid: str) -> bool:
        """Always pass use_ridge=True. _predict_wl now decides per-sample whether to
        blend Ridge based on whether its raw prediction is within the training target
        range [wl_y_min, wl_y_max]. No n_ff threshold needed.
        """
        return True

    def predict(self, pid: str, cd: float, cs: float, mw: float, bd: float,
                sk_mu: float = None, sk_sig: float = None) -> CTSPrediction:
        """
        Predict all CTS outcomes for one (placement, knob) configuration.
        Calibration scales (if set via calibrate_*) are applied automatically.
        WL uses LGB+Ridge blend for in-distribution designs; LGB-only for OOD.
        """
        t = self._get_t_clk(pid)
        X_pw, X_wl, X_sk, X_hv, pw_n, wl_n = self.features.build(pid, cd, cs, mw, bd, t)
        pred = self._heads.predict_single(X_pw, X_wl, X_sk, X_hv, pw_n, wl_n, sk_mu, sk_sig,
                                          use_ridge=self._use_ridge(pid))
        # Apply calibration
        cal_pw = self._pw_cal.get(pid, 1.0)
        cal_wl = self._wl_cal.get(pid, 1.0)
        if cal_pw != 1.0 or cal_wl != 1.0:
            pred = CTSPrediction(
                power_mW=pred.power_mW * cal_pw,
                wl_mm=pred.wl_mm * cal_wl,
                skew_z=pred.skew_z,
                skew_ns=pred.skew_ns,
                hold_vio=pred.hold_vio,
                pw_norm=pred.pw_norm,
                wl_norm=pred.wl_norm,
            )
        return pred

    def optimize(self, pid: str, n: int = 5000,
                 cd_range=(35, 70), cs_range=(12, 30),
                 mw_range=(130, 280), bd_range=(70, 150),
                 sk_mu=None, sk_sig=None, seed: int = 42,
                 exhaustive: bool = False,
                 pred_chunk: int = 200_000,
                 method: str = 'random') -> pd.DataFrame:
        """
        Pareto-optimal knob search over (cd, cs, mw, bd).

        exhaustive=False (default): random sweep of n configs.
        exhaustive=True: enumerate ALL integer combinations in the grid.
            cd: cd_range[0]..cd_range[1]  (inclusive)
            cs: cs_range[0]..cs_range[1]
            mw: mw_range[0]..mw_range[1]
            bd: bd_range[0]..bd_range[1]
            Prediction is chunked (pred_chunk rows at a time) and uses a
            two-phase Pareto to stay memory-safe at 8M+ configs.

        Returns DataFrame of non-dominated solutions sorted by power_mW.
        """
        t = self._get_t_clk(pid)
        cal_pw = self._pw_cal.get(pid, 1.0)
        cal_wl = self._wl_cal.get(pid, 1.0)
        use_r  = self._use_ridge(pid)

        # ── Build knob arrays ──────────────────────────────────────────────────
        n_cd = cd_range[1] - cd_range[0] + 1
        n_cs = cs_range[1] - cs_range[0] + 1
        n_mw = mw_range[1] - mw_range[0] + 1
        n_bd = bd_range[1] - bd_range[0] + 1
        total_grid = n_cd * n_cs * n_mw * n_bd

        if exhaustive or method == 'exhaustive':
            cd_vals = np.arange(cd_range[0], cd_range[1] + 1, dtype=np.float64)
            cs_vals = np.arange(cs_range[0], cs_range[1] + 1, dtype=np.float64)
            mw_vals = np.arange(mw_range[0], mw_range[1] + 1, dtype=np.float64)
            bd_vals = np.arange(bd_range[0], bd_range[1] + 1, dtype=np.float64)
            g = np.meshgrid(cd_vals, cs_vals, mw_vals, bd_vals, indexing='ij')
            cd_a = g[0].ravel(); cs_a = g[1].ravel()
            mw_a = g[2].ravel(); bd_a = g[3].ravel()
            del g

        elif method == 'sobol':
            import math
            from scipy.stats.qmc import Sobol
            m = math.ceil(math.log2(max(n, 2)))     # 2^m >= n
            sampler = Sobol(d=4, scramble=True, seed=seed)
            pts = sampler.random_base2(m)[:n]        # [n, 4] in [0,1]
            cd_a = (pts[:, 0] * (n_cd - 1) + cd_range[0]).round().astype(int).astype(float)
            cs_a = (pts[:, 1] * (n_cs - 1) + cs_range[0]).round().astype(int).astype(float)
            mw_a = (pts[:, 2] * (n_mw - 1) + mw_range[0]).round().astype(int).astype(float)
            bd_a = (pts[:, 3] * (n_bd - 1) + bd_range[0]).round().astype(int).astype(float)

        elif method == 'nsga2':
            # NSGA-II via pymoo — returns Pareto-directed population.
            # n interpreted as total evaluations: pop_size * n_gen ≈ n.
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.core.problem import Problem
            from pymoo.optimize import minimize as _pymoo_min

            _self = self
            class _CTSProblem(Problem):
                def __init__(self):
                    xl = np.array([cd_range[0], cs_range[0], mw_range[0], bd_range[0]], float)
                    xu = np.array([cd_range[1], cs_range[1], mw_range[1], bd_range[1]], float)
                    super().__init__(n_var=4, n_obj=3, xl=xl, xu=xu)

                def _evaluate(self, X, out, *args, **kwargs):
                    Xi = X.round().astype(int)
                    cd_b, cs_b, mw_b, bd_b = Xi[:,0].astype(float), Xi[:,1].astype(float), \
                                              Xi[:,2].astype(float), Xi[:,3].astype(float)
                    Xpw, Xwl, Xsk, Xhv, pw_n, wl_n = _self.features.batch_build(
                        pid, cd_b, cs_b, mw_b, bd_b, t)
                    pw_b, wl_b, sk_b, _, _ = _self._heads.predict_batch(
                        Xpw, Xwl, Xsk, Xhv, pw_n, wl_n, sk_mu, sk_sig,
                        use_ridge=_self._use_ridge(pid))
                    pw_b = pw_b * cal_pw;  wl_b = wl_b * cal_wl
                    out['F'] = np.column_stack([pw_b, wl_b, sk_b])

            pop_size = 500
            n_gen    = max(1, n // pop_size)
            algo     = NSGA2(pop_size=pop_size)
            res      = _pymoo_min(_CTSProblem(), algo,
                                  ('n_gen', n_gen), seed=seed, verbose=False)
            X_res = res.X.round().astype(int)
            cd_a  = X_res[:, 0].astype(float); cs_a = X_res[:, 1].astype(float)
            mw_a  = X_res[:, 2].astype(float); bd_a = X_res[:, 3].astype(float)
            # pymoo already returns a Pareto-optimal set → skip internal Pareto filter
            F = res.F
            df = pd.DataFrame(dict(
                cd=cd_a.astype(int), cs=cs_a.astype(int),
                mw=mw_a.astype(int), bd=bd_a.astype(int),
                power_mW=F[:, 0], wl_mm=F[:, 1], skew_ns=F[:, 2]))
            return df.sort_values('power_mW').reset_index(drop=True)

        else:  # 'random' — unique, no overlap
            n_actual = min(n, total_grid)
            rng      = np.random.default_rng(seed)
            flat_idx = rng.choice(total_grid, size=n_actual, replace=False)
            bd_a = (flat_idx % n_bd + bd_range[0]).astype(float); flat_idx //= n_bd
            mw_a = (flat_idx % n_mw + mw_range[0]).astype(float); flat_idx //= n_mw
            cs_a = (flat_idx % n_cs + cs_range[0]).astype(float); flat_idx //= n_cs
            cd_a = (flat_idx       + cd_range[0]).astype(float)

        total = len(cd_a)

        # ── Chunked prediction ────────────────────────────────────────────────
        all_pw  = np.empty(total, dtype=np.float64)
        all_wl  = np.empty(total, dtype=np.float64)
        all_sk  = np.empty(total, dtype=np.float64)

        for start in range(0, total, pred_chunk):
            end = min(start + pred_chunk, total)
            Xpw, Xwl, Xsk, Xhv, pw_n, wl_n = self.features.batch_build(
                pid, cd_a[start:end], cs_a[start:end],
                mw_a[start:end], bd_a[start:end], t)
            pw_c, wl_c, sk_c, _, _ = self._heads.predict_batch(
                Xpw, Xwl, Xsk, Xhv, pw_n, wl_n, sk_mu, sk_sig, use_ridge=use_r)
            all_pw[start:end] = pw_c * cal_pw
            all_wl[start:end] = wl_c * cal_wl
            all_sk[start:end] = sk_c if sk_c is not None else 0.0

        # ── Two-phase Pareto (memory-safe for large N) ────────────────────────
        costs_all = np.stack([all_pw, all_wl, all_sk], axis=1)

        def _pareto_mask(costs):
            """Return boolean mask of non-dominated rows."""
            m   = len(costs)
            dom = np.zeros(m, bool)
            eps = 1e-9
            for i in range(0, m, 500):
                ci = costs[i:i + 500]
                d2 = (np.all(costs[:, None, :] <= ci[None, :, :] + eps, axis=2) &
                      np.any(costs[:, None, :] <  ci[None, :, :] - eps, axis=2))
                dom[i:i + 500] = d2.any(axis=0)
            return ~dom

        if total <= pred_chunk:
            # Small enough: exact Pareto directly
            keep = _pareto_mask(costs_all)
        else:
            # Phase 1: exact Pareto on a 200K sample → approximate front
            rng2   = np.random.default_rng(seed + 1)
            s_idx  = rng2.choice(total, size=min(pred_chunk, total), replace=False)
            approx = costs_all[s_idx]
            approx_keep = _pareto_mask(approx)
            front  = approx[approx_keep]          # ~100–500 pts

            # Phase 2: find candidates — any row not dominated by the approx front
            eps = 1e-9
            chunk2 = pred_chunk
            cand_mask = np.zeros(total, bool)
            for start in range(0, total, chunk2):
                end   = min(start + chunk2, total)
                block = costs_all[start:end]       # [B, 3]
                # dominated by front if ANY front pt dominates it
                # front[j] dominates block[i]: front[j] <= block[i]+eps ALL & < block[i]-eps SOME
                dominated = (
                    np.all(front[:, None, :] <= block[None, :, :] + eps, axis=2) &
                    np.any(front[:, None, :] <  block[None, :, :] - eps, axis=2)
                ).any(axis=0)
                cand_mask[start:end] = ~dominated

            # Phase 3: exact Pareto among candidates
            cand_idx  = np.where(cand_mask)[0]
            cand_costs = costs_all[cand_idx]
            cand_keep  = _pareto_mask(cand_costs)
            keep       = np.zeros(total, bool)
            keep[cand_idx[cand_keep]] = True

        df = pd.DataFrame(dict(
            cd=cd_a[keep].astype(int), cs=cs_a[keep].astype(int),
            mw=mw_a[keep].astype(int), bd=bd_a[keep].astype(int),
            power_mW=all_pw[keep], wl_mm=all_wl[keep], skew_ns=all_sk[keep]))
        return df.sort_values('power_mW').reset_index(drop=True)


# Backward-compatible alias
CTSSurrogateV2 = SwiftCTS
