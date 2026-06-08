"""
reframe_results.py  —  Corrected re-analysis of saved UC1 result JSONs.
NO model re-training. Reads the same files the Results notebook reads.

Per (variant, alpha, data_case) it derives:
  saved_pooled    : r['test_auc']                  (OLD headline; buggy for partial)
  valsel_pooled   : history[argmax(val)]['test']   (checkpoint-FIXED pooled)
  perclient_mean  : mean over r['per_client']       (HONEST, decision-relevant)
  perclient_worst : min  over r['per_client']        (equity / worst hospital)
  final_mb        : r['cumul_mb'][-1]

Public API (importable from the notebook):
  build_dataframe(data_cases=('filtered','unfiltered'))  -> pandas.DataFrame
  write_csv(path=None, df=None)                          -> path written
  plot_auc_vs_alpha(df, data_case, metric='perclient_mean')
  plot_pooled_vs_perclient(df, data_case)
  plot_pareto(df, data_case, metric='perclient_mean')
  plot_worst_client(df, data_case)
  plot_checkpoint_correction(data_case='filtered')

CAVEATS (carried into the thesis text):
  * valsel_pooled fully fixes the checkpoint bug for the POOLED metric.
  * per_client is saved at the final encoder state; for PARTIAL variants that is
    the (best-predictor + final-encoder) pairing (per-round per-client AUC was
    never logged). The partial-vs-partial ranking is robust to this.
  * pooled AUC is inflated by between-client separation (grows as alpha falls);
    perclient_mean / perclient_worst are the honest metrics.
"""
import json, glob, os, statistics as st
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
ROOTS = {'fedavg': os.path.join(BASE, '02_FedAvg', 'results'),
         'fedgen': os.path.join(BASE, '03_FedGen', 'results')}
FEDDATA = os.path.join(BASE, 'federated_data')   # per-client test sizes live here

_SIZE_CACHE = {}


def _client_test_sizes(data_case, alpha):
    """{client_idx_str: n_test} from federated_data/<case>/alpha_<a>/client_<i>/client_info.json.
    Used to sample-weight per-client AUC so UC1 matches UC2's client-local aggregation."""
    key = (data_case, alpha)
    if key in _SIZE_CACHE:
        return _SIZE_CACHE[key]
    sizes = {}
    adir = os.path.join(FEDDATA, data_case, f'alpha_{alpha}')
    for ci in glob.glob(os.path.join(adir, 'client_*')):
        idx = os.path.basename(ci).split('_')[1]
        info = os.path.join(ci, 'client_info.json')
        if os.path.exists(info):
            sizes[idx] = json.load(open(info)).get('n_test')
    _SIZE_CACHE[key] = sizes
    return sizes
ARTIFACT_FLAG = {'fedgen_gmm', 'fedgen_gmm_full'}   # pooled win is a separation artifact

# curated variants for the headline figures (others still go to the CSV)
HEADLINE = ['fedavg_full', 'fedavg_partial', 'fedgen_full', 'fedgen_partial',
            'fedgen_zhu', 'fedgen_zhu_full', 'fedgen_paper_partial', 'fedgen_paper_full',
            'fedgen_zhu_code_partial', 'fedgen_zhu_code_full', 'fedgen_gmm', 'fedgen_gmm_full']


def kind(variant):
    if 'partial' in variant:
        return 'partial'
    if 'full' in variant:
        return 'full'
    return 'partial'   # bare fedgen_zhu / fedgen_gmm etc. are partial-sharing


def _val_selected_pooled(d):
    hv = [h for h in d.get('history', []) if 'val' in h and 'test' in h]
    if not hv:
        return None
    bi = max(range(len(hv)), key=lambda i: hv[i]['val'])
    return hv[bi]['test']


def _per_client_stats(d):
    pc = list(d.get('per_client', {}).values())
    if not pc:
        return None, None, 0
    return st.mean(pc), min(pc), len(pc)


def collect():
    """rows[(data_case, variant, alpha)] = list of per-seed dicts."""
    rows = defaultdict(list)
    for root in ROOTS.values():
        for path in glob.glob(f'{root}/**/*.json', recursive=True):
            parts = path.replace(os.sep, '/').split('/')
            data_case, alpha = parts[-4], float(parts[-3].split('_')[1])
            variant = os.path.splitext(parts[-1])[0]
            d = json.load(open(path))
            pcm, pcw, npc = _per_client_stats(d)
            rows[(data_case, variant, alpha)].append(dict(
                saved_pooled=d.get('test_auc'), valsel_pooled=_val_selected_pooled(d),
                perclient_mean=pcm, perclient_worst=pcw, n_pc=npc,
                pc_dict=dict(d.get('per_client', {})),     # {client_idx: auc}
                pc_values=list(d.get('per_client', {}).values()),
                n_rounds=len(d.get('history', [])),
                final_mb=(d.get('cumul_mb') or [None])[-1]))
    return rows


def _weighted_perclient(pc_dict, sizes):
    """Sample-weighted mean of per-client AUCs (UC2-equivalent client-local aggregation)."""
    num = den = 0.0
    for idx, auc in pc_dict.items():
        n = sizes.get(idx)
        if n:
            num += auc * n
            den += n
    return (num / den) if den else float('nan')


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return float('nan'), float('nan')
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def build_dataframe(data_cases=('filtered', 'unfiltered')):
    import pandas as pd, numpy as np
    rows = collect()
    out = []
    keys = sorted(rows, key=lambda k: (k[0], k[1], k[2]))
    for (dc, v, a) in keys:
        if dc not in data_cases:
            continue
        rs = rows[(dc, v, a)]
        sp_m, sp_s = _agg([r['saved_pooled'] for r in rs])      # == your test_auc Mean/Std
        vp_m, vp_s = _agg([r['valsel_pooled'] for r in rs])
        pc_m, pc_s = _agg([r['perclient_mean'] for r in rs])         # equal-weight (macro)
        pw_m, pw_s = _agg([r['perclient_worst'] for r in rs])
        sizes = _client_test_sizes(dc, a)
        pcw_m, _ = _agg([_weighted_perclient(r['pc_dict'], sizes) for r in rs])  # size-weighted
        mb_m, _ = _agg([r['final_mb'] for r in rs])
        rounds = np.mean([r['n_rounds'] for r in rs]) if rs else float('nan')
        all_pc = [x for r in rs for x in r['pc_values']]
        client_sigma = float(np.std(all_pc)) if all_pc else float('nan')
        out.append(dict(
            # ── keys ──
            data_case=dc, variant=v, sharing=kind(v), alpha=a, n_seeds=len(rs),
            # ── PRIMARY: your original headline metric (mean/std of test_auc) ──
            test_auc_mean=round(sp_m, 4), test_auc_std=round(sp_s, 4),
            client_sigma=round(client_sigma, 4),
            rounds=round(rounds, 1), final_mb=round(mb_m, 3),
            # ── CLIENT-LOCAL metrics (the UC2-equivalent view) ──
            perclient_wmean=round(pcw_m, 4),     # size-weighted  (matches UC2 aggregation)
            perclient_mean=round(pc_m, 4), perclient_std=round(pc_s, 4),   # equal-weight macro
            perclient_worst_mean=round(pw_m, 4),
            # ── other diagnostics ──
            valsel_pooled_mean=round(vp_m, 4),
            artifact=(v in ARTIFACT_FLAG)))
    import pandas as pd
    return pd.DataFrame(out)


def write_csv(path=None, df=None):
    df = build_dataframe() if df is None else df
    path = path or os.path.join(BASE, 'reframed_results.csv')
    df.to_csv(path, index=False)
    return path


# ───────────────────── statistics (Wilcoxon signed-rank) ─────────────────────
# Mirrors the original notebook's cell 21: each FedGen variant vs its MATCHED
# FedAvg baseline (fedavg_partial / fedavg_full), paired by alpha x seed.
# Exact test via DP over sign-flips (no scipy dependency; matches scipy for n<=~20).
ALPHAS_DEFAULT = [0.1, 0.5, 1.0, 5.0, 10.0]


def _avg_rank(a):
    import numpy as np
    order = np.argsort(a, kind='mergesort'); ranks = np.empty(len(a), float); sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def wilcoxon_signed_rank(b, x):
    """Two-sided exact Wilcoxon signed-rank. Returns (W, p, n_nonzero) or None."""
    import numpy as np
    d = np.asarray(x, float) - np.asarray(b, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return None
    ranks = _avg_rank(np.abs(d))
    Wplus = ranks[d > 0].sum(); S = ranks.sum()
    r2 = np.rint(ranks * 2).astype(int); tot = int(r2.sum())
    counts = np.zeros(tot + 1); counts[0] = 1.0
    for r in r2:
        counts = counts + np.concatenate([np.zeros(r), counts[:len(counts) - r]])
    mu2 = tot / 2.0; dev = abs(round(Wplus * 2) - mu2)
    ks = np.arange(len(counts))
    p = counts[np.abs(ks - mu2) >= dev - 1e-9].sum() / (2.0 ** n)
    return min(Wplus, S - Wplus), min(p, 1.0), n


def _paired_series(rows, dc, variant, field, alphas):
    out = []
    for a in alphas:
        for r in rows.get((dc, variant, a), []):
            out.append(r[field])
    return out


def significance_table(metric='saved_pooled', data_cases=('filtered', 'unfiltered'),
                       alphas=None):
    """Paired Wilcoxon of every FedGen variant vs its matched FedAvg baseline.
    metric in {'saved_pooled','valsel_pooled','perclient_mean','perclient_worst'}.
    Returns a tidy pandas.DataFrame."""
    import pandas as pd, numpy as np
    alphas = alphas or ALPHAS_DEFAULT
    rows = collect()
    present = sorted({v for (_, v, _) in rows})
    out = []
    for dc in data_cases:
        for family, base in [('partial', 'fedavg_partial'), ('full', 'fedavg_full')]:
            bser = _paired_series(rows, dc, base, metric, alphas)
            variants = [v for v in present
                        if v.startswith('fedgen') and kind(v) == family]
            for v in sorted(variants):
                xser = _paired_series(rows, dc, v, metric, alphas)
                n = min(len(bser), len(xser))
                if n < 5:
                    continue
                b, x = np.array(bser[:n]), np.array(xser[:n])
                res = wilcoxon_signed_rank(b, x)
                delta = float((x - b).mean())
                if res is None:
                    w, p, sig = float('nan'), 1.0, 'tied'
                else:
                    w, p, _ = res
                    sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
                out.append(dict(data_case=dc, family=family, variant=v, baseline=base,
                                n_pairs=n, delta=round(delta, 4),
                                W=(None if res is None else round(w, 1)),
                                p=round(p, 4), sig=sig,
                                better=('FedGen' if delta > 0 else 'FedAvg')))
    return pd.DataFrame(out)


# ───────────────────────── plotting ─────────────────────────
def _style(variant):
    c = 'tab:blue' if kind(variant) == 'full' else 'tab:orange'
    ls = '--' if 'gmm' in variant else ('-.' if 'zhu_code' in variant else
          (':' if 'paper' in variant else '-'))
    return c, ls


def plot_auc_vs_alpha(df, data_case='filtered', metric='perclient_mean', ax=None):
    import matplotlib.pyplot as plt
    d = df[(df.data_case == data_case) & (df.variant.isin(HEADLINE))]
    shares = ['partial', 'full']
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    else:
        axes = ax
    for sh, a in zip(shares, axes):
        for v in sorted(d[d.sharing == sh].variant.unique()):
            sub = d[d.variant == v].sort_values('alpha')
            c, ls = _style(v)
            a.errorbar(sub.alpha, sub[metric], yerr=sub.get(metric.replace('mean', 'std')),
                       marker='o', ms=5, lw=1.8, ls=ls, capsize=3,
                       label=v + (' (artifact)' if v in ARTIFACT_FLAG else ''))
        a.set_xscale('log'); a.grid(alpha=.3)   # ascending: small α left, large α right
        a.axhline(.5, color='grey', ls=':', lw=1)
        a.set_title(f'{sh}-sharing'); a.set_xlabel('Dirichlet α (log, ←more heterogeneous)')
        a.legend(fontsize=7, ncol=1)
    axes[0].set_ylabel(metric)
    plt.suptitle(f'UC1 — {metric} vs heterogeneity  [{data_case}]', fontweight='bold')
    plt.tight_layout()
    return plt.gcf()


def plot_pooled_vs_perclient(df, data_case='filtered', ax=None):
    import matplotlib.pyplot as plt
    d = df[(df.data_case == data_case) & (df.variant.isin(HEADLINE))]
    g = d.groupby('alpha').apply(
        lambda x: (x.valsel_pooled_mean - x.perclient_mean).mean()).sort_index()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(a) for a in g.index], g.values, color='tab:red', alpha=.7)
    ax.set_xlabel('Dirichlet α'); ax.set_ylabel('pooled − per-client AUC')
    ax.set_title(f'UC1 — pooled-AUC inflation vs heterogeneity  [{data_case}]', fontweight='bold')
    ax.grid(alpha=.3, axis='y')
    for i, val in enumerate(g.values):
        ax.text(i, val, f'+{val:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    return plt.gcf()


def plot_pareto(df, data_case='filtered', metric='perclient_mean', ax=None):
    import matplotlib.pyplot as plt
    d = df[(df.data_case == data_case) & (df.variant.isin(HEADLINE))]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5))
    for sh, col in [('partial', 'tab:orange'), ('full', 'tab:blue')]:
        sub = d[d.sharing == sh]
        ax.scatter(sub.final_mb, sub[metric], c=col, s=42, alpha=.75, label=f'{sh}-sharing',
                   edgecolor='k', lw=.4)
    art = d[d.artifact]
    ax.scatter(art.final_mb, art[metric], facecolors='none', edgecolors='red', s=120,
               lw=1.5, label='GMM (pooled artifact)')
    ax.set_xscale('log'); ax.set_xlabel('cumulative MB (log)'); ax.set_ylabel(metric)
    ax.set_title(f'UC1 — accuracy vs communication frontier  [{data_case}]', fontweight='bold')
    ax.grid(alpha=.3); ax.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_worst_client(df, data_case='filtered', ax=None):
    import matplotlib.pyplot as plt
    d = df[(df.data_case == data_case) & (df.variant.isin(HEADLINE))]
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.6))
    for v in sorted(d.variant.unique()):
        sub = d[d.variant == v].sort_values('alpha')
        c, ls = _style(v)
        ax.plot(sub.alpha, sub.perclient_worst_mean, marker='s', ls=ls, lw=1.6, label=v)
    ax.set_xscale('log'); ax.axhline(.5, color='red', ls=':', lw=1.2,
                                     label='random (0.5)')   # ascending α left→right
    ax.set_xlabel('Dirichlet α (log)'); ax.set_ylabel('worst-client AUC')
    ax.set_title(f'UC1 — worst-served client (equity)  [{data_case}]', fontweight='bold')
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    return plt.gcf()


def plot_checkpoint_correction(data_case='filtered', ax=None):
    import matplotlib.pyplot as plt
    rows = collect()
    diffs = defaultdict(list)
    for (dc, v, a), rs in rows.items():
        if dc != data_case:
            continue
        for r in rs:
            if r['saved_pooled'] is not None and r['valsel_pooled'] is not None:
                diffs[kind(v)].append(r['saved_pooled'] - r['valsel_pooled'])
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist([diffs['full'], diffs['partial']], bins=25, label=['full', 'partial'],
            color=['tab:blue', 'tab:orange'])
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('saved_pooled − valsel_pooled  (checkpoint error)')
    ax.set_ylabel('runs')
    ax.set_title(f'UC1 — checkpoint inconsistency (partial only)  [{data_case}]', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return plt.gcf()


def main():
    df = build_dataframe()
    p = write_csv(df=df)
    print('Wrote', p, '|', len(df), 'rows')
    # quick console head-to-head on the honest metric
    for dc in ['filtered', 'unfiltered']:
        sub = df[df.data_case == dc]
        if sub.empty:
            continue
        print(f'\n[{dc}] per-client AUC, best FedAvg vs best FedGen by alpha:')
        for a in sorted(sub.alpha.unique()):
            s = sub[sub.alpha == a]
            fa = s[s.variant.str.startswith('fedavg')].perclient_mean.max()
            fg = s[s.variant.str.startswith('fedgen')].perclient_mean.max()
            who = 'FedGen' if fg > fa else 'FedAvg'
            print(f'  α={a:<5} FedAvg={fa:.4f}  FedGen={fg:.4f}  -> {who} by {abs(fg-fa):.4f}')


if __name__ == '__main__':
    main()
