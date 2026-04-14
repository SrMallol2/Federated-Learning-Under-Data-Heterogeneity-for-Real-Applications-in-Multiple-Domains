"""
UC1PrintingUtils.py
─────────────────────────────────────────────────────────────────────────────
Shared plotting and printing utilities for UC1 analysis notebooks.

All functions accept a `label` argument ('filtered' or 'unfiltered') that is
used in figure titles and save-path suffixes, allowing the same function to be
called twice in 002_UC1_FederatedAnalysis with different partitions without
name collisions.

Exports
-------
Clinical analysis (001):
    plot_class_distribution
    plot_continuous_features
    plot_categorical_features
    plot_engineered_features
    plot_feature_correlations

Federated analysis — compute (002):
    compute_wasserstein_df
    compute_js_df
    compute_eff_df
    compute_heterogeneity_metrics

Federated analysis — plot (002):
    plot_wasserstein_bars
    plot_mean_w1_line
    plot_js_covariate_shift
    plot_n_pos_per_client
    print_client_summary
    plot_client_distribution
    plot_label_prior
    plot_heterogeneity_summary

Latent space (003):
    plot_latent_pca
    plot_latent_prototypes
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as scipy_entropy

PALETTE = {'Not readmitted (0)': '#4878CF', 'Readmitted <30d (1)': '#E24A33'}
_C0, _C1 = list(PALETTE.values())


# ═════════════════════════════════════════════════════════════════════════════
# CLINICAL ANALYSIS  (001_UC1_ClinicalAnalysis)
# ═════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(y, palette=None, save_path='figures/01_class_distribution.png'):
    colors = list((palette or PALETTE).values())
    counts = pd.Series(y).value_counts().sort_index()
    labels = list((palette or PALETTE).keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Global Class Distribution', fontweight='bold', fontsize=13)

    axes[0].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=0.8)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 300, f'{v:,}', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Absolute counts')
    axes[0].tick_params(axis='x', rotation=10)

    axes[1].pie(
        counts.values, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    axes[1].set_title('Class proportions')

    plt.tight_layout()
    _save(save_path)
    plt.show()
    print(f'Imbalance ratio: {counts[0]/counts[1]:.2f}:1  |  '
          f'Positive rate: {counts[1]/len(y)*100:.2f}%')


def plot_continuous_features(df, y, continuous_features,
                              palette=None, save_path='figures/02_continuous_features.png'):
    colors = list((palette or PALETTE).values())
    df0 = df[df['readmitted_binary'] == 0]
    df1 = df[df['readmitted_binary'] == 1]
    n_cols = 4
    n_rows = int(np.ceil(len(continuous_features) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    fig.suptitle('Class-Conditional Distributions: Continuous Features',
                 fontweight='bold', fontsize=14, y=1.01)
    axes = axes.flatten()

    for i, feat in enumerate(continuous_features):
        ax = axes[i]
        v0, v1 = df0[feat].dropna(), df1[feat].dropna()
        ks_stat, ks_p = ks_2samp(v0, v1)
        wass = wasserstein_distance(v0, v1)
        bins = min(40, int(v0.nunique()))
        ax.hist(v0, bins=bins, alpha=0.55, color=colors[0], density=True, label='Not readmitted')
        ax.hist(v1, bins=bins, alpha=0.55, color=colors[1], density=True, label='Readmitted <30d')
        ax.set_title(feat, fontsize=10, fontweight='bold')
        sign = '***' if ks_p < 0.001 else ('**' if ks_p < 0.01 else ('*' if ks_p < 0.05 else 'ns'))
        ax.annotate(f'KS={ks_stat:.3f} {sign}\nW={wass:.3f}',
                    xy=(0.97, 0.97), xycoords='axes fraction',
                    ha='right', va='top', fontsize=8)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(len(continuous_features), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(save_path)
    plt.show()


def plot_categorical_features(df, y, cat_features,
                               palette=None, save_path='figures/03_categorical_features.png'):
    colors = list((palette or PALETTE).values())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Class-Conditional Distributions: Categorical Features',
                 fontweight='bold', fontsize=14)
    axes = axes.flatten()

    for i, feat in enumerate(cat_features):
        ax = axes[i]
        ct = pd.crosstab(df[feat], df['readmitted_binary'], normalize='index') * 100
        ct.columns = ['Not readmitted', 'Readmitted <30d']
        ct.sort_values('Readmitted <30d', ascending=False, inplace=True)
        ct.plot(kind='barh', ax=ax, color=colors, edgecolor='white', linewidth=0.6)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xlabel('% within category')
        ax.set_ylabel('')
        ax.legend(fontsize=8, loc='lower right')
        ax.axvline(x=y.mean() * 100, color='gray', linestyle='--', alpha=0.6, linewidth=1)

    axes[-1].set_visible(False)
    plt.tight_layout()
    _save(save_path)
    plt.show()


def plot_engineered_features(df, y, palette=None,
                              save_path='figures/04_engineered_features.png'):
    colors = list((palette or PALETTE).values())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Engineered Feature Distributions by Class', fontweight='bold', fontsize=13)

    # service_utilization
    ax = axes[0]
    for label, color in zip([0, 1], colors):
        vals = df[df['readmitted_binary'] == label]['service_utilization']
        ax.hist(vals.clip(upper=20), bins=21, alpha=0.6, color=color,
                density=True, label=list((palette or PALETTE).keys())[label])
    ks_s, _ = ks_2samp(df[df['readmitted_binary']==0]['service_utilization'],
                        df[df['readmitted_binary']==1]['service_utilization'])
    ax.set_title(f'service_utilization\n(clipped at 20)\nKS={ks_s:.3f}', fontsize=10)
    ax.set_xlabel('Total visits (past 12 months)')
    ax.legend(fontsize=8)

    # medication_count
    ax = axes[1]
    mc_cross = pd.crosstab(df['medication_count'].clip(upper=10),
                            df['readmitted_binary'], normalize='index') * 100
    if 1 in mc_cross.columns:
        ax.bar(mc_cross.index, mc_cross[1], color=colors[1], alpha=0.8)
    ax.axhline(y.mean()*100, color='gray', linestyle='--', alpha=0.6, linewidth=1,
               label='Global rate')
    ax.set_title('Readmission rate\nby medication_count', fontsize=10)
    ax.set_xlabel('Medications changed (clipped at 10)')
    ax.set_ylabel('% readmitted within 30d')
    ax.legend(fontsize=8)

    # HbA1c_diabetes_interaction
    ax = axes[2]
    if 'HbA1c_diabetes_interaction' in df.columns:
        ct = pd.crosstab(df['HbA1c_diabetes_interaction'],
                         df['readmitted_binary'], normalize='index') * 100
        ct.index = ['No interaction', 'HbA1c + diabetes']
        if 1 in ct.columns:
            ax.bar(ct.index, ct[1], color=colors[1], alpha=0.8)
        ax.axhline(y.mean()*100, color='gray', linestyle='--', alpha=0.6, linewidth=1,
                   label='Global rate')
        ax.set_title('Readmission rate\nby HbA1c_diabetes_interaction', fontsize=10)
        ax.set_ylabel('% readmitted within 30d')
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(save_path)
    plt.show()


def plot_feature_correlations(df, continuous_features,
                               save_path='figures/05_feature_correlations.png'):
    import seaborn as sns
    cont_cols = continuous_features + ['HbA1c_diabetes_interaction']
    corr = df[[c for c in cont_cols + ['readmitted_binary'] if c in df.columns]].corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Correlations', fontweight='bold', fontsize=13)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=axes[0], mask=mask, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, annot=True, fmt='.2f',
                annot_kws={'size': 7}, linewidths=0.5, square=True)
    axes[0].set_title('Pearson correlation')
    axes[0].tick_params(axis='x', rotation=45)

    if 'readmitted_binary' in corr.columns:
        target_corr = corr['readmitted_binary'].drop('readmitted_binary').sort_values(
            key=abs, ascending=False)
        colors_bar = [_C1 if v > 0 else _C0 for v in target_corr.values]
        axes[1].barh(target_corr.index, target_corr.values,
                     color=colors_bar, alpha=0.8, edgecolor='white')
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].set_title('Correlation with target (ranked by |r|)')
        axes[1].set_xlabel('Pearson r')

    plt.tight_layout()
    _save(save_path)
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# FEDERATED ANALYSIS — COMPUTE  (002_UC1_FederatedAnalysis)
# ═════════════════════════════════════════════════════════════════════════════

WASS_FEATURES_DEFAULT = [
    'time_in_hospital', 'number_inpatient', 'number_emergency',
    'service_utilization', 'medication_count', 'num_medications', 'age'
]


def compute_wasserstein_df(partitions, df, wass_features=None):
    """
    Compute per-(alpha, client, feature) W₁ distances from global distribution.

    Parameters
    ----------
    partitions  : dict {alpha: {client_id: [patient_ids]}}
    df          : pre-OHE DataFrame with patient_nbr column
    wass_features : list of feature names (defaults to WASS_FEATURES_DEFAULT)

    Returns
    -------
    pd.DataFrame with columns [alpha, client, feature, wasserstein]
    """
    feats = wass_features or WASS_FEATURES_DEFAULT
    global_dists = {f: df[f].values for f in feats if f in df.columns}
    records = []

    for alpha, partition in partitions.items():
        for client_id, patient_ids in partition.items():
            pid_set = set(patient_ids)
            for feat, global_vals in global_dists.items():
                client_vals = df.loc[df['patient_nbr'].isin(pid_set), feat].values
                if len(client_vals) > 10:
                    records.append({
                        'alpha'      : alpha,
                        'client'     : f'C{client_id}',
                        'feature'    : feat,
                        'wasserstein': wasserstein_distance(global_vals, client_vals),
                    })

    return pd.DataFrame(records)


def compute_js_df(partitions, patient_labels_map, y, wass_df):
    """
    Compute per-(alpha, client) JS divergence (label shift) and mean W₁ (covariate shift).

    Returns
    -------
    pd.DataFrame with columns [alpha, client, js_divergence, mean_w1, pos_rate, n_patients]
    """
    global_label_dist = np.array([1 - y.mean(), y.mean()])
    records = []

    for alpha, partition in partitions.items():
        for client_id, patient_ids in partition.items():
            pid_set = set(patient_ids)
            client_labels = [patient_labels_map.get(p, 0) for p in pid_set]
            n_pos   = sum(client_labels)
            n_total = len(client_labels)
            client_label_dist = np.array([(n_total - n_pos) / n_total, n_pos / n_total])
            js = jensenshannon(global_label_dist, client_label_dist, base=2)
            mean_w1 = wass_df[
                (wass_df['alpha'] == alpha) & (wass_df['client'] == f'C{client_id}')
            ]['wasserstein'].mean()
            records.append({
                'alpha'       : alpha,
                'client'      : f'C{client_id}',
                'js_divergence': js,
                'mean_w1'     : mean_w1,
                'pos_rate'    : n_pos / n_total if n_total > 0 else 0,
                'n_patients'  : n_total,
            })

    return pd.DataFrame(records)


def compute_eff_df(partitions, patient_labels_map):
    """
    Compute per-(alpha, client) effective minority-class sample size.

    Returns
    -------
    pd.DataFrame with columns [alpha, client, n_total, n_pos, n_neg, pos_rate]
    """
    records = []
    for alpha, partition in partitions.items():
        for client_id, patient_ids in partition.items():
            pid_set = set(patient_ids)
            labels  = [patient_labels_map.get(p, 0) for p in pid_set]
            n_total = len(labels)
            n_pos   = sum(labels)
            records.append({
                'alpha'   : alpha,
                'client'  : f'C{client_id}',
                'n_total' : n_total,
                'n_pos'   : n_pos,
                'n_neg'   : n_total - n_pos,
                'pos_rate': n_pos / n_total if n_total > 0 else 0,
            })
    return pd.DataFrame(records)


def compute_heterogeneity_metrics(partitions, patient_labels_map, y, wass_df):
    """
    Compute composite heterogeneity metrics per alpha:
    Gini (size), Gini (label), CV, mean JS, max JS, mean TV, mean W₁, effective clients.

    Returns
    -------
    pd.DataFrame — one row per alpha
    """
    def _gini(values):
        v = np.sort(np.array(values, dtype=float))
        n = len(v)
        if v.sum() == 0:
            return 0.0
        return (2 * np.sum(np.arange(1, n + 1) * v) / (n * v.sum())) - (n + 1) / n

    def _tv(p, q):
        return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))

    global_label_dist = np.array([1 - y.mean(), y.mean()])
    records = []

    for alpha, partition in partitions.items():
        sizes     = [len(v) for v in partition.values()]
        pos_rates = [
            sum(patient_labels_map.get(p, 0) for p in v) / max(len(v), 1)
            for v in partition.values()
        ]
        js_vals = [jensenshannon(global_label_dist,
                                  np.array([1 - r, r]), base=2)
                   for r in pos_rates]
        tv_vals = [_tv(global_label_dist, [1 - r, r]) for r in pos_rates]
        s = np.array(sizes, dtype=float)
        eff = float(np.exp(scipy_entropy(s / s.sum()))) if s.sum() > 0 else 0.0
        mean_w1 = wass_df[wass_df['alpha'] == alpha]['wasserstein'].mean() \
                  if not wass_df.empty else np.nan

        records.append({
            'alpha'          : alpha,
            'gini_size'      : _gini(sizes),
            'gini_label'     : _gini(pos_rates),
            'cv_label'       : np.std(pos_rates) / max(np.mean(pos_rates), 1e-9),
            'mean_js'        : float(np.mean(js_vals)),
            'max_js'         : float(np.max(js_vals)),
            'mean_tv'        : float(np.mean(tv_vals)),
            'mean_w1'        : float(mean_w1),
            'effective_clients': eff,
        })

    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════════════════
# FEDERATED ANALYSIS — PLOT  (002_UC1_FederatedAnalysis)
# ═════════════════════════════════════════════════════════════════════════════

def plot_wasserstein_bars(wass_df, alpha_sweep, label='filtered', save_dir='figures'):
    """Bar chart of W₁ per feature per client, one panel per alpha."""
    if wass_df.empty:
        print(f'[{label}] W₁ DataFrame is empty, skipping.')
        return

    fig, axes = plt.subplots(1, len(alpha_sweep),
                             figsize=(5 * len(alpha_sweep), 5), sharey=True)
    if len(alpha_sweep) == 1:
        axes = [axes]
    fig.suptitle(f'W₁: Client vs Global Feature Distribution [{label}]',
                 fontweight='bold', fontsize=13)

    for ax, alpha in zip(axes, alpha_sweep):
        sub = wass_df[wass_df['alpha'] == alpha]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot(index='feature', columns='client', values='wasserstein')
        pivot.plot(kind='barh', ax=ax, colormap='Set2', alpha=0.85, edgecolor='white')
        ax.set_title(f'α = {alpha}', fontsize=12, fontweight='bold')
        ax.set_xlabel('W₁ distance from global')
        ax.set_ylabel('Feature' if ax == axes[0] else '')
        ax.legend(title='Client', fontsize=8, loc='lower right')

    plt.tight_layout()
    _save(os.path.join(save_dir, f'07_wasserstein_bars_{label}.png'))
    plt.show()


def plot_mean_w1_line(wass_df, label='filtered', save_dir='figures'):
    """Line plot of mean W₁ per alpha."""
    if wass_df.empty:
        return
    mean_wass = wass_df.groupby('alpha')['wasserstein'].mean()
    most_het  = mean_wass.idxmax()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mean_wass.index, mean_wass.values, 'o-', color='steelblue',
            linewidth=2, markersize=8)
    for alpha_val, w1_val in mean_wass.items():
        ax.annotate(f'W₁={w1_val:.3f}', xy=(alpha_val, w1_val),
                    xytext=(6, 6), textcoords='offset points', fontsize=8, color='steelblue')
    ax.annotate('Most heterogeneous\nrealized partition',
                xy=(most_het, mean_wass[most_het]),
                xytext=(-60, -30), textcoords='offset points', fontsize=8, color='#E24A33',
                arrowprops=dict(arrowstyle='->', color='#E24A33', lw=1.2))
    ax.set_xlabel('Dirichlet α  (generation parameter, not realized heterogeneity)')
    ax.set_ylabel('Mean W₁ (across clients and features)')
    ax.set_title(f'Realized W₁ heterogeneity [{label}]\nα is not a monotone proxy for W₁',
                 fontweight='bold')
    ax.invert_xaxis()
    plt.tight_layout()
    _save(os.path.join(save_dir, f'08_mean_w1_{label}.png'))
    plt.show()


def plot_js_covariate_shift(js_df, label='filtered', save_dir='figures'):
    """Scatter: JS divergence (label shift) vs mean W₁ (covariate shift) per client."""
    if js_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Label Shift vs Covariate Shift per Client [{label}]',
                 fontweight='bold', fontsize=13)

    for ax, (metric, ylabel, color) in zip(axes, [
        ('js_divergence', 'JS divergence (label shift)',  '#E24A33'),
        ('mean_w1',       'Mean W₁ (covariate shift)',    'steelblue'),
    ]):
        alphas = sorted(js_df['alpha'].unique())
        x = np.arange(len(alphas))
        width = 0.8 / js_df['client'].nunique()
        for j, client in enumerate(sorted(js_df['client'].unique())):
            vals = [js_df[(js_df['alpha']==a) & (js_df['client']==client)][metric].values
                    for a in alphas]
            vals = [v[0] if len(v) > 0 else np.nan for v in vals]
            ax.bar(x + j * width, vals, width=width * 0.9,
                   label=client, alpha=0.85)
        ax.set_xticks(x + width * (js_df['client'].nunique() - 1) / 2)
        ax.set_xticklabels([str(a) for a in alphas])
        ax.set_xlabel('Dirichlet α')
        ax.set_ylabel(ylabel)
        ax.legend(title='Client', fontsize=8)

    plt.tight_layout()
    _save(os.path.join(save_dir, f'09_label_covariate_shift_{label}.png'))
    plt.show()


def plot_n_pos_per_client(eff_df, alpha_sweep, min_viable=200,
                           label='filtered', save_dir='figures'):
    """Bar chart of n_pos per client, one panel per alpha."""
    if eff_df.empty:
        return
    fig, axes = plt.subplots(1, len(alpha_sweep),
                             figsize=(5 * len(alpha_sweep), 5), sharey=False)
    if len(alpha_sweep) == 1:
        axes = [axes]
    fig.suptitle(f'Effective Minority-Class Sample Size per Client [{label}]\n'
                 '(n_pos = patients ever readmitted within 30d)',
                 fontweight='bold', fontsize=13)

    for ax, alpha in zip(axes, alpha_sweep):
        sub = eff_df[eff_df['alpha'] == alpha].sort_values('client')
        if sub.empty:
            ax.set_visible(False)
            continue
        colors = ['#E24A33' if n < min_viable else '#4878CF' for n in sub['n_pos']]
        ax.bar(sub['client'], sub['n_pos'], color=colors, alpha=0.85, edgecolor='white')
        ax.axhline(min_viable, color='black', linestyle='--', linewidth=1.2,
                   label=f'min_viable={min_viable}')
        ax.set_title(f'α = {alpha}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Client')
        ax.set_ylabel('n_pos' if ax == axes[0] else '')
        ax.legend(fontsize=8)
        for _, row in sub.iterrows():
            ax.text(row['client'], row['n_pos'] + 5,
                    str(int(row['n_pos'])), ha='center', fontsize=9)

    plt.tight_layout()
    _save(os.path.join(save_dir, f'10_n_pos_per_client_{label}.png'))
    plt.show()


def print_client_summary(alpha_sweep, base_dir, n_clients):
    """
    Print train/val/test table and bar charts for each alpha from client_info.json.
    Only works for filtered partition (which has preprocessed arrays and client_info.json).
    """
    import json as _json
    label = os.path.basename(base_dir)

    for alpha in alpha_sweep:
        print(f'\n{"─" * 65}')
        print(f'  [{label}] α={alpha} client data summary:')
        print(f'  {"Client":<10} {"Train":>8} {"Val":>8} {"Test":>8} '
              f'{"Features":>10} {"Pos. rate":>10}')
        print(f'  {"─"*10} {"─"*8} {"─"*8} {"─"*8} {"─"*10} {"─"*10}')

        client_info = {}
        for i in range(n_clients):
            info_path = os.path.join(base_dir, f'alpha_{alpha}', f'client_{i}',
                                     'client_info.json')
            if not os.path.exists(info_path):
                print(f'  client_{i}: client_info.json not found, skipping.')
                continue
            with open(info_path) as f:
                client_info[f'client_{i}'] = _json.load(f)

        for k, v in client_info.items():
            print(f'  {k:<10} {v["n_train"]:>8,} {v["n_val"]:>8,} '
                  f'{v["n_test"]:>8,} {v["n_features"]:>10,} '
                  f'{v["positive_rate"]:>10.3f}')


def plot_client_distribution(alpha_sweep, base_dir, n_clients,
                              label='filtered', save_dir='figures'):
    """Bar charts of training set size and positive rate per client for each alpha."""
    import json as _json

    for alpha in alpha_sweep:
        client_info = {}
        for i in range(n_clients):
            info_path = os.path.join(base_dir, f'alpha_{alpha}', f'client_{i}',
                                     'client_info.json')
            if not os.path.exists(info_path):
                continue
            with open(info_path) as f:
                client_info[f'client_{i}'] = _json.load(f)

        if not client_info:
            continue

        keys       = list(client_info.keys())
        train_sizes = [client_info[c]['n_train'] for c in keys]
        pos_rates   = [client_info[c]['positive_rate'] for c in keys]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Client Data Distribution — [{label}] α={alpha}',
                     fontweight='bold')
        axes[0].bar(keys, train_sizes, color='steelblue')
        axes[0].set_title('Training set size per client')
        axes[0].set_ylabel('Samples')
        axes[0].tick_params(axis='x', rotation=15)

        axes[1].bar(keys, pos_rates, color='darkorange')
        axes[1].axhline(sum(pos_rates) / len(pos_rates), color='gray',
                        linestyle='--', label='Mean')
        axes[1].set_title('Positive rate per client')
        axes[1].set_ylabel('Positive rate')
        axes[1].set_ylim(0, 0.25)
        axes[1].tick_params(axis='x', rotation=15)
        axes[1].legend()

        plt.tight_layout()
        _save(os.path.join(save_dir, f'11_client_dist_{label}_alpha{alpha}.png'))
        plt.show()


def plot_label_prior(eff_df, y, n_clients, alpha_sweep,
                     label='filtered', save_dir='figures'):
    """
    Two-panel figure: estimated label prior per alpha vs true global rate,
    and per-client positive rate grouped by alpha.
    """
    if eff_df.empty:
        return
    available_alphas = sorted(eff_df['alpha'].unique())
    true_pos_rate    = y.mean()

    prior_df = (eff_df.groupby('alpha')['pos_rate'].mean()
                .reset_index().rename(columns={'pos_rate': 'estimated_phat'}))
    prior_df['true_rate']  = true_pos_rate
    prior_df['distortion'] = prior_df['estimated_phat'] - prior_df['true_rate']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'FedGen Label Prior Estimation [{label}]: '
                 'How Heterogeneity Distorts $\\hat{{p}}(y)$',
                 fontweight='bold', fontsize=13)

    # Left: estimated vs true prior
    ax = axes[0]
    plot_prior = prior_df[prior_df['alpha'].isin(available_alphas)].reset_index(drop=True)
    ax.bar(plot_prior['alpha'].astype(str), plot_prior['estimated_phat'],
           color='steelblue', alpha=0.8, label='Estimated $\\hat{p}(y=1)$', width=0.4)
    ax.axhline(true_pos_rate, color='#E24A33', linestyle='--', linewidth=1.5,
               label=f'True global rate ({true_pos_rate:.3f})')
    ax.set_xlabel('α')
    ax.set_ylabel('P(y=1)')
    ax.set_title('Estimated label prior $\\hat{p}(y=1)$\nvs true global rate')
    ax.legend(fontsize=9)
    for i, row in plot_prior.iterrows():
        ax.text(i, row['estimated_phat'] + 0.002,
                f"{row['estimated_phat']:.3f}", ha='center', fontsize=9)

    # Right: per-client positive rate grouped by alpha
    ax = axes[1]
    x     = np.arange(n_clients)
    n_a   = len(available_alphas)
    width = 0.8 / n_a
    colors_cycle = plt.cm.tab10(np.linspace(0, 1, n_a))
    for j, alpha in enumerate(available_alphas):
        sub = eff_df[eff_df['alpha'] == alpha].sort_values('client')
        if len(sub) == n_clients:
            ax.bar(x + j * width, sub['pos_rate'].values, width=width * 0.9,
                   color=colors_cycle[j], alpha=0.85, label=f'α={alpha}')
    ax.axhline(true_pos_rate, color='black', linestyle='--', linewidth=1.2,
               label=f'Global ({true_pos_rate:.3f})')
    ax.set_xticks(x + width * (n_a - 1) / 2)
    ax.set_xticklabels([f'C{i}' for i in range(n_clients)])
    ax.set_xlabel('Client')
    ax.set_ylabel('Positive rate')
    ax.set_title('Per-client positive rate by α')
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    _save(os.path.join(save_dir, f'12_label_prior_{label}.png'))
    plt.show()


def plot_heterogeneity_summary(het_df, label='filtered', save_dir='figures'):
    """
    Multi-metric heterogeneity summary table and bar charts.
    het_df is the output of compute_heterogeneity_metrics.
    """
    if het_df.empty:
        return

    metrics = ['gini_label', 'gini_size', 'cv_label', 'mean_js',
               'mean_tv', 'mean_w1', 'effective_clients']
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    fig.suptitle(f'Heterogeneity Metrics by α [{label}]',
                 fontweight='bold', fontsize=12)

    for ax, metric in zip(axes, metrics):
        ax.bar(het_df['alpha'].astype(str), het_df[metric],
               color='steelblue', alpha=0.85, edgecolor='white')
        ax.set_title(metric.replace('_', '\n'), fontsize=9, fontweight='bold')
        ax.set_xlabel('α')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    _save(os.path.join(save_dir, f'13_heterogeneity_summary_{label}.png'))
    plt.show()

    print(f'\nHeterogeneity summary [{label}]:')
    print(het_df.to_string(index=False))


# ═════════════════════════════════════════════════════════════════════════════
# LATENT SPACE  (003_UC1_LatentSpace)
# ═════════════════════════════════════════════════════════════════════════════

def plot_latent_pca(Z, y, pca, palette=None, seed=42,
                    save_dir='figures'):
    """PCA scatter (PC1 vs PC2) and explained variance bar chart."""
    colors = list((palette or PALETTE).values())
    Z_pca   = pca.transform(Z) if hasattr(pca, 'components_') else pca.fit_transform(Z)
    rng     = np.random.default_rng(seed)
    idx_sub = rng.choice(len(Z), size=min(8000, len(Z)), replace=False)
    Z_sub   = Z_pca[idx_sub]
    y_sub   = y[idx_sub]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Latent Space Geometry (penultimate layer)',
                 fontweight='bold', fontsize=13)

    for label_val, (name, color) in zip([0, 1], (palette or PALETTE).items()):
        mask = y_sub == label_val
        axes[0].scatter(Z_sub[mask, 0], Z_sub[mask, 1],
                        c=color, alpha=0.3, s=5, label=name, rasterized=True)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title('PCA: PC1 vs PC2')
    axes[0].legend(markerscale=3, fontsize=9)

    n_comp = len(pca.explained_variance_ratio_)
    axes[1].bar(range(1, n_comp + 1), pca.explained_variance_ratio_ * 100,
                color='steelblue', alpha=0.8, edgecolor='white')
    axes[1].plot(range(1, n_comp + 1),
                 np.cumsum(pca.explained_variance_ratio_) * 100,
                 'o-', color='darkorange', linewidth=2, markersize=6, label='Cumulative')
    axes[1].set_xlabel('Principal component')
    axes[1].set_ylabel('% variance explained')
    axes[1].set_title('Latent space: variance explained')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    _save(os.path.join(save_dir, '09_latent_pca.png'))
    plt.show()

    Z0, Z1 = Z[y == 0], Z[y == 1]
    print('\nLatent space class-conditional statistics:')
    print(f'  Class 0 — mean norm: {np.linalg.norm(Z0.mean(0)):.3f} | '
          f'mean intra-class dist (sample): '
          f'{np.mean(np.linalg.norm(Z0[:500] - Z0[:500].mean(0), axis=1)):.3f}')
    print(f'  Class 1 — mean norm: {np.linalg.norm(Z1.mean(0)):.3f} | '
          f'mean intra-class dist (sample): '
          f'{np.mean(np.linalg.norm(Z1[:500] - Z1[:500].mean(0), axis=1)):.3f}')
    print(f'  Inter-class centroid distance: '
          f'{np.linalg.norm(Z0.mean(0) - Z1.mean(0)):.3f}')
    print(f'  Latent dimension: {Z.shape[1]}')

    return Z_pca, idx_sub  # return for use in plot_latent_prototypes


def plot_latent_prototypes(Z, y, pca, prototypes, palette=None,
                            Z_pca=None, idx_sub=None, seed=42,
                            save_dir='figures'):
    """Scatter of latent points with prototype centroids overlaid."""
    colors = list((palette or PALETTE).values())
    if Z_pca is None:
        Z_pca = pca.transform(Z)
    if idx_sub is None:
        idx_sub = np.random.default_rng(seed).choice(len(Z), size=min(8000, len(Z)),
                                                       replace=False)
    Z_sub = Z_pca[idx_sub]
    y_sub = y[idx_sub]

    fig, ax = plt.subplots(figsize=(7, 6))
    for label_val, (name, color) in zip([0, 1], (palette or PALETTE).items()):
        mask = y_sub == label_val
        ax.scatter(Z_sub[mask, 0], Z_sub[mask, 1],
                   c=color, alpha=0.2, s=5, rasterized=True)
        proto_pca = pca.transform(prototypes[label_val].reshape(1, -1))[0]
        ax.scatter(*proto_pca[:2], c=color, s=120, marker='*',
                   edgecolors='black', linewidth=1, zorder=5,
                   label=f'{name} (prototype z̄_y)')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Latent space: class prototypes z̄_y\n'
                 '(generator constraint anchors)', fontweight='bold')
    ax.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    _save(os.path.join(save_dir, '10_latent_prototypes.png'))
    plt.show()

    # Print prototype stats
    print('Prototype centroids (z̄_y) computed:')
    for cls in [0, 1]:
        p = prototypes[cls]
        stds = Z[y == cls].std(axis=0)
        print(f'  z̄_{{y={cls}}}: shape={p.shape}, norm={np.linalg.norm(p):.3f}, '
              f'mean std={stds.mean():.3f}')
    intra_var_0 = np.mean(np.var(Z[y == 0], axis=0))
    intra_var_1 = np.mean(np.var(Z[y == 1], axis=0))
    print(f'\nMean intra-class variance: class0={intra_var_0:.3f}, '
          f'class1={intra_var_1:.3f}')
    print(f'Suggested λ starting point: ~0.1 (ablate over [0.0, 0.01, 0.1, 1.0])')


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPER
# ═════════════════════════════════════════════════════════════════════════════

def _save(path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    plt.savefig(path, bbox_inches='tight')