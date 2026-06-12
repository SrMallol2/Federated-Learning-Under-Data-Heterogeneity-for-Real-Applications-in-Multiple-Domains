"""
uc2_honest_table.py  — corrected re-analysis of UC2 saved metrics. NO retraining.

FINAL experiments live under results/newpart (client_local test) and
results/newpart_global (global test). The bare results/ tree is superseded.

Reports the *scaled* mse/mae (computed directly on output vs y in
userbase.test — correct), NOT unscaled_mae/mape (10**(...) space, dominated by
the largest-load samples, ~1e8 MAPE). Selection: best round by scaled mse.

METRIC CAVEAT (important): clients train on L1 (scaled MAE). For FedAvg the
scaled MSE never improves on the round-0 init (best-by-MSE selects the
UNTRAINED model) while MAE decreases monotonically — that is a stall under
MSE, NOT divergence. Statements about learning dynamics must use MAE;
best_mse is kept for cross-method comparison only.

Column semantics:
  mse_improves : training beat the round-0 init on scaled MSE (argmin > 0)
  diverges     : true divergence — non-finite metrics or final MAE worse
                 than the round-0 init (does not fire on any current run)

Public API (importable from the notebook):
  PROTOCOLS                                  -> {'client_local': 'results/newpart', 'global': ...}
  build_dataframe()                          -> pandas.DataFrame (both protocols)
  write_csv(path=None, df=None)              -> path written
  trajectory(protocol, method, alpha, rep=0) -> list[mse_per_round]
  le_sweep_dataframe()                       -> local-steps sweep (le = batch steps/round, K=1)
  plot_le_sweep(df)                          -> MAE-based sweep figure (FedAvg vs FedGen)
  plot_mse_vs_alpha(df, protocol='client_local')
  plot_convergence(protocol='client_local', alpha='1.0')
  plot_local_vs_global(df)
"""
import pickle, glob, os, statistics as st
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
PROTOCOLS = {'client_local': os.path.join(BASE, 'results', 'newpart'),
             'global':       os.path.join(BASE, 'results', 'newpart_global')}
ALPHAS  = ['0.01', '0.1', '0.5', '1.0', '5.0', '10.0']
METHODS = ['centralized', 'fedavg', 'fedavg-partial', 'fedgen', 'fedgen-partial']
COLORS  = {'centralized': 'k', 'fedavg': 'tab:red', 'fedavg-partial': 'tab:pink',
           'fedgen': 'tab:green', 'fedgen-partial': 'tab:olive'}


def _gm(repdir):
    fr = os.path.join(repdir, 'full_results.pkl')
    if os.path.exists(fr):
        return pickle.load(open(fr, 'rb'))['metrics']['glob_test_metric']
    return None


def _repdirs(protocol, method, alpha):
    return sorted(glob.glob(os.path.join(PROTOCOLS[protocol], method,
                                         f'alpha_{alpha}', 'lstm', 'rep_*')))


def trajectory(protocol, method, alpha, rep=0):
    gm = _gm(os.path.join(PROTOCOLS[protocol], method, f'alpha_{alpha}', 'lstm', f'rep_{rep}'))
    return [x['mse'] for x in gm] if gm else None


def build_dataframe():
    import math
    import pandas as pd
    out = []
    for proto in PROTOCOLS:
        for m in METHODS:
            for a in ALPHAS:
                bms, lms, bmae, lmae, improves, diverged = [], [], [], [], [], []
                for rep in _repdirs(proto, m, a):
                    gm = _gm(rep)
                    if not gm:
                        continue
                    mse = [x['mse'] for x in gm]
                    mae = [x['mae'] for x in gm]
                    bms.append(min(mse)); lms.append(mse[-1])
                    bmae.append(min(mae)); lmae.append(mae[-1])
                    # did training ever beat the round-0 init on MSE?
                    improves.append(min(range(len(mse)), key=lambda i: mse[i]) > 0)
                    # true divergence: non-finite metrics or final MAE worse than init
                    finite = all(math.isfinite(v) for v in mse + mae)
                    diverged.append((not finite) or mae[-1] > mae[0])
                if not bms:
                    continue
                out.append(dict(
                    protocol=proto, method=m, alpha=float(a), n_reps=len(bms),
                    best_mse=round(st.mean(bms), 4),
                    best_mse_std=round(st.pstdev(bms) if len(bms) > 1 else 0.0, 4),
                    last_mse=round(st.mean(lms), 4),
                    best_mae=round(st.mean(bmae), 4),
                    last_mae=round(st.mean(lmae), 4),
                    mse_improves=bool(all(improves)),
                    diverges=bool(any(diverged))))
    return pd.DataFrame(out)


def write_csv(path=None, df=None):
    df = build_dataframe() if df is None else df
    path = path or os.path.join(BASE, 'uc2_results.csv')
    df.to_csv(path, index=False)
    return path


# ───────────────────── local-steps sweep (le = batch steps per round) ─────────
# `local_epochs` in this framework is the number of mini-batch steps per round
# (K=1, batch 32), NOT full passes over the client data. le ∈ {1,5,10} live in
# {method}-le{N}/ dirs; le=20 is the existing baseline ({method}/), seed 0 only.
#
# Read the sweep on MAE (the L1 objective the clients actually train). best_mse
# is uninformative here: for FedAvg argmin(mse) is the untrained round-0 model,
# so its best-MSE row is flat across le by construction.

def le_sweep_dataframe(protocol='client_local', methods=('fedavg', 'fedgen'),
                       les=(1, 5, 10, 20), rep=0):
    import pandas as pd
    out = []
    for m in methods:
        for le in les:
            d = m if le == 20 else f'{m}-le{le}'
            for a in ALPHAS:
                gm = _gm(os.path.join(PROTOCOLS[protocol], d,
                                      f'alpha_{a}', 'lstm', f'rep_{rep}'))
                if not gm:
                    continue
                mse = [x['mse'] for x in gm]
                mae = [x['mae'] for x in gm]
                out.append(dict(
                    method=m, le=le, alpha=float(a), rounds=len(gm),
                    first_mae=round(mae[0], 4),
                    best_mae=round(min(mae), 4),
                    last_mae=round(mae[-1], 4),
                    best_mse=round(min(mse), 4),
                    last_mse=round(mse[-1], 4),
                    mse_improves=bool(min(range(len(mse)), key=lambda i: mse[i]) > 0)))
    return pd.DataFrame(out)


def plot_le_sweep(df=None, metric='best_mae', ax=None):
    """Error vs local steps per round, one panel per method, one line per alpha.
    Default metric is best scaled MAE (training-aligned); caveats: single seed,
    fixed 300-round cap (le conflated with total compute), fedgen-le5 early-stops."""
    import matplotlib.pyplot as plt
    df = le_sweep_dataframe() if df is None else df
    methods = sorted(df.method.unique())
    if ax is None:
        fig, axes = plt.subplots(1, len(methods), figsize=(6.2 * len(methods), 4.8),
                                 sharey=True)
    else:
        axes = ax
    if len(methods) == 1:
        axes = [axes]
    for a_, m in zip(axes, methods):
        sub = df[df.method == m]
        for alpha in sorted(sub.alpha.unique()):
            s = sub[sub.alpha == alpha].sort_values('le')
            # NB: s['le'], not s.le — pandas reserves .le for the <= operator
            a_.plot(s['le'], s[metric], 'o-', lw=1.8, ms=5, label=f'α={alpha}')
        a_.set_xticks(sorted(df['le'].unique()))
        a_.set_xlabel('local steps per round (batch=32, K=1)')
        a_.set_title(m); a_.grid(alpha=.3)
    axes[0].set_ylabel(f'{metric} (scaled, lower = better)')
    axes[-1].legend(fontsize=8)
    import matplotlib.pyplot as _plt
    _plt.suptitle('UC2 — error vs local steps (seed 0; MAE = trained objective)',
                  fontweight='bold')
    _plt.tight_layout()
    return _plt.gcf()


# ───────────────────────── plotting ─────────────────────────
def plot_mse_vs_alpha(df, protocol='client_local', ax=None):
    import matplotlib.pyplot as plt
    d = df[df.protocol == protocol]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    for m in METHODS:
        sub = d[d.method == m].sort_values('alpha')
        if sub.empty:
            continue
        ax.errorbar(sub.alpha, sub.best_mse, yerr=sub.best_mse_std, marker='o', ms=6,
                    lw=1.9, capsize=3, color=COLORS.get(m), label=m)
    ax.set_xscale('log'); ax.grid(alpha=.3)   # ascending: small α left, large α right
    ax.set_xlabel('Dirichlet α (log, ←more heterogeneous)')
    ax.set_ylabel('best scaled MSE (lower = better)')
    ax.set_title(f'UC2 — accuracy vs heterogeneity  [{protocol}]', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_convergence(protocol='client_local', alpha='1.0', ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    for m in ['fedavg', 'fedgen', 'fedgen-partial', 'centralized']:
        t = trajectory(protocol, m, alpha)
        if t:
            ax.plot(range(1, len(t) + 1), t, lw=1.8, color=COLORS.get(m), label=m)
    ax.set_xlabel('communication round'); ax.set_ylabel('scaled MSE (test)')
    ax.set_title(f'UC2 — convergence  [{protocol}, α={alpha}]', fontweight='bold')
    ax.grid(alpha=.3); ax.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_local_vs_global(df, ax=None):
    """FedGen full vs partial under the two evaluation protocols."""
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.2
    alphas = sorted(df.alpha.unique())
    x = np.arange(len(alphas))
    series = [('fedgen', 'client_local', 'tab:green', '-'),
              ('fedgen', 'global', 'tab:green', '//'),
              ('fedgen-partial', 'client_local', 'tab:olive', '-'),
              ('fedgen-partial', 'global', 'tab:olive', '//')]
    for i, (m, proto, col, hatch) in enumerate(series):
        sub = df[(df.method == m) & (df.protocol == proto)].set_index('alpha')
        vals = [sub.best_mse.get(a, np.nan) for a in alphas]
        ax.bar(x + (i - 1.5) * width, vals, width, color=col,
               hatch=(None if hatch == '-' else hatch),
               edgecolor='k', lw=.4, label=f'{m} [{proto}]')
    ax.set_xticks(x); ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel('Dirichlet α'); ax.set_ylabel('best scaled MSE')
    ax.set_title('UC2 — partial vs full under client-local vs global test', fontweight='bold')
    ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=8)
    plt.tight_layout()
    return plt.gcf()


def main():
    df = build_dataframe()
    p = write_csv(df=df)
    print('Wrote', p, '|', len(df), 'rows')
    print('\n[client_local] best scaled MSE (lower=better):')
    d = df[df.protocol == 'client_local']
    for a in sorted(d.alpha.unique()):
        s = d[d.alpha == a]
        def g(m):
            r = s[s.method == m].best_mse
            return f'{r.iloc[0]:.4f}' if len(r) else '  --  '
        print(f'  α={a:<5} central={g("centralized")} fedavg={g("fedavg")} '
              f'fedgen={g("fedgen")} fedgen-part={g("fedgen-partial")}')


if __name__ == '__main__':
    main()
