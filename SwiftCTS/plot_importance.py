"""
plot_importance.py — Feature importance bar charts for all three model heads.

Generates:
  feature_importance_power.png   — XGBoost gain (top 20)
  feature_importance_wl.png      — LightGBM gain (top 20)
  feature_importance_skew.png    — LightGBM gain (top 20)
  feature_importance_all.png     — Combined 3-panel figure

Method: gain importance = total reduction in loss (MSE) attributable to each
feature across all splits in all trees. Gain is the most meaningful importance
type for boosted trees — it directly measures how much each feature improves
the model, unlike 'weight' (split count) which favors high-cardinality features.
"""

import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, 'saved_models', 'model.pkl')

# ── Skew feature names (63 dims, derived from _build_sk_features) ─────────────
# Order: sk_ctx(22) + kn(8) + ck(16) + ski(17)
SKEW_FEAT_NAMES = [
    # sk_ctx — 22 context features
    'log_n_ff', 'log_die_area', 'log_ff_hpwl', 'log_ff_spacing',
    'die_aspect', 'ff_cx', 'ff_cy', 'ff_x_std', 'ff_y_std',
    'frac_xor', 'comb_per_ff', 'avg_ds', 'rel_act', 'mean_sig_prob',
    'slack_mean', 'slack_std', 'slack_min', 'slack_p10',
    'frac_neg', 'frac_tight', 'frac_critical', 'log_paths_per_ff',
    # kn — 8 knob features (log_cd=22, log_cs=23, log_mw=24, log_bd=25)
    'log_cd', 'log_cs', 'log_mw', 'log_bd', 'cd', 'cs', 'mw', 'bd',
    # ck — 16 critical-path spatial features
    'crit_max_dist', 'crit_mean_dist', 'crit_p90_dist', 'crit_ff_hpwl',
    'crit_cx_offset', 'crit_cy_offset', 'crit_x_std', 'crit_y_std',
    'crit_frac_bndry', 'crit_star_deg', 'crit_chain_frac',
    'crit_asymmetry', 'crit_eccentric', 'crit_dens_ratio',
    'log_crit_max_um', 'log_crit_mean_um',
    # ski — 17 knob×spatial interaction features
    'cd/ff_spacing', 'bd/crit_max', 'mw/crit_max',
    'star_deg×cd', 'asymm×mw', 'dens×cs',
    'cmax_dist×cd', 'asymm×cmax',
    'frac_neg×star', 'frac_tgt×chain',
    'crit_hpwl/(cs+1)',
    'log(cmax/cd)', 'log(cmax/bd)', 'log(cmax/mw)',
    'cx×cd', 'cy×mw', 'log(nff/cs)×chpwl',
]

# Colour scheme — one per head
COLORS = {
    'power': '#e05c3a',   # warm red-orange
    'wl':    '#3a7ec0',   # steel blue
    'skew':  '#2faa62',   # emerald green
}


def get_xgb_gain(model, n_features):
    booster = model.get_booster()
    scores = booster.get_score(importance_type='gain')
    return np.array([scores.get(f'f{i}', 0.0) for i in range(n_features)])


def get_lgb_gain(model):
    try:
        return np.array(model.booster_.feature_importance(importance_type='gain'), dtype=float)
    except Exception:
        return model.feature_importances_.astype(float)


def make_bar_chart(ax, names, importances, color, title, top_n=20, show_pct=True):
    total = importances.sum() + 1e-12
    idx = np.argsort(importances)[::-1][:top_n]
    vals = importances[idx] / total * 100
    lbls = [names[i] for i in idx]

    # Reverse so highest is at top
    vals = vals[::-1]
    lbls = lbls[::-1]

    y_pos = np.arange(len(vals))
    bars = ax.barh(y_pos, vals, color=color, alpha=0.85, height=0.7, edgecolor='white', linewidth=0.4)

    # Cumulative line on secondary axis
    ax2 = ax.twiny()
    cumvals = np.cumsum(vals[::-1])[::-1]   # cumulative from top (largest first) in reversed order
    # Actually: we want cumulative from the largest (top of chart) going down
    # Recompute properly for display
    cum = np.cumsum(vals[::-1])[::-1]
    ax2.plot(cum + vals, y_pos, 'k--', alpha=0.3, linewidth=1.0, label='cumul %')
    ax2.set_xlim(0, 105)
    ax2.set_xlabel('Cumulative %', fontsize=8, color='grey')
    ax2.tick_params(labelsize=7, colors='grey')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(lbls, fontsize=8.5)
    ax.set_xlabel('Gain importance (%)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis='x', alpha=0.25, linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate bars
    for bar, v in zip(bars, vals):
        if v > 0.5:
            ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.1f}%', va='center', ha='left', fontsize=7.5, color='#333333')

    # Remaining count
    remaining = len(importances) - top_n
    if remaining > 0:
        rest_pct = (importances[np.argsort(importances)[:-top_n]].sum() / total) * 100
        ax.text(0.98, 0.02, f'+ {remaining} more features = {rest_pct:.1f}%',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color='grey', style='italic')

    return ax


def main():
    print(f'Loading model from {MODEL_PATH}')
    with open(MODEL_PATH, 'rb') as f:
        mdl = pickle.load(f)

    lodo = mdl.get('lodo', {})
    pw_names = lodo.get('pw_feature_names', [f'f{i}' for i in range(20)])
    wl_names = lodo.get('wl_feature_names', [f'f{i}' for i in range(75)])
    sk_names = SKEW_FEAT_NAMES   # 63 dims

    # ── Extract importances ───────────────────────────────────────────────────
    pw_gain = get_xgb_gain(mdl['model_power'], len(pw_names))
    wl_gain = get_lgb_gain(mdl['model_wl_lgb'])
    sk_gain = get_lgb_gain(mdl['model_skew'])

    print(f'Power: {len(pw_names)} features, top={pw_names[pw_gain.argmax()]} '
          f'({pw_gain.max()/pw_gain.sum()*100:.1f}%)')
    print(f'WL:    {len(wl_names)} features, top={wl_names[wl_gain.argmax()]} '
          f'({wl_gain.max()/wl_gain.sum()*100:.1f}%)')
    print(f'Skew:  {len(sk_names)} features, top={sk_names[sk_gain.argmax()]} '
          f'({sk_gain.max()/sk_gain.sum()*100:.1f}%)')

    # ── Individual plots ──────────────────────────────────────────────────────
    for tag, names, gain, color, title in [
        ('power', pw_names, pw_gain, COLORS['power'],
         'Power Head — XGBoost Gain Importance (20 features)'),
        ('wl',    wl_names, wl_gain, COLORS['wl'],
         'Wirelength Head — LightGBM Gain Importance (75 features)'),
        ('skew',  sk_names, sk_gain, COLORS['skew'],
         'Skew Head — LightGBM Gain Importance (63 features)'),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))
        make_bar_chart(ax, names, gain, color, title, top_n=20)
        fig.tight_layout()
        out = os.path.join(HERE, f'feature_importance_{tag}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    # ── Combined 3-panel figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('SwiftCTS — Feature Importance by Model Head\n'
                 '(Gain = total MSE reduction from splits on this feature)',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, (names, gain, color, title) in zip(axes, [
        (pw_names, pw_gain, COLORS['power'], 'Power (XGBoost, 20 feat)'),
        (wl_names, wl_gain, COLORS['wl'],    'Wirelength (LightGBM, 75 feat)'),
        (sk_names, sk_gain, COLORS['skew'],  'Skew (LightGBM, 63 feat)'),
    ]):
        make_bar_chart(ax, names, gain, color, title, top_n=15)

    fig.tight_layout()
    out_all = os.path.join(HERE, 'feature_importance_all.png')
    fig.savefig(out_all, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved combined: {out_all}')

    # ── Print summary table ───────────────────────────────────────────────────
    print('\n' + '='*65)
    print('  METHOD: Gain importance (total MSE reduction per feature)')
    print('  across all trees and splits. Best for comparing feature')
    print('  contributions in boosted ensembles.')
    print('='*65)
    for tag, names, gain, lbl in [
        ('POWER', pw_names, pw_gain, '20-feat XGBoost'),
        ('WL',    wl_names, wl_gain, '75-feat LightGBM'),
        ('SKEW',  sk_names, sk_gain, '63-feat LightGBM'),
    ]:
        total = gain.sum() + 1e-12
        top3 = np.argsort(gain)[::-1][:3]
        print(f'\n  {tag} ({lbl}):')
        for r, i in enumerate(top3, 1):
            print(f'    #{r}: {names[i]:<25}  {gain[i]/total*100:.1f}%')


if __name__ == '__main__':
    main()
