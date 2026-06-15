import sys, os, glob
sys.path.insert(0, 'lib')
import numpy as np, torch

NP = 'data/partitions/new_partitions/lookback_60/steps_1/clean_test'
DEV = 'cpu'

def clean_cl(a):
    return torch.load(f'{NP}/clientlocal_alpha_{a}.pt', weights_only=False)

@torch.no_grad()
def mae(model, X, y):
    out = model(X)['output'].numpy().reshape(-1)
    return float(np.mean(np.abs(out - y.numpy().reshape(-1))))

def perclient(method, a):
    """Seed-averaged per-client MAE vector (length 20) on the clean client-local set."""
    repdirs = sorted(glob.glob(f'results/newpart/{method}/alpha_{a}/lstm/rep_*'))
    if not repdirs:
        return None
    ds = clean_cl(a); users = ds['users']
    per_seed = []
    for rd in repdirs:
        mdirs = glob.glob(os.path.join(rd, 'models', '*'))
        if not mdirs:
            continue
        vec = []
        ok = True
        for u in users:
            f = os.path.join(mdirs[0], f'user_{u}.pt')
            if not os.path.exists(f):
                ok = False; break
            m = torch.load(f, map_location=DEV, weights_only=False).to(DEV).eval()
            X, y = ds['user_data'][u]['x'], ds['user_data'][u]['y']
            vec.append(mae(m, X, y))
            del m
        if ok:
            per_seed.append(np.array(vec))
    return np.mean(per_seed, axis=0) if per_seed else None

# Wilcoxon signed-rank, normal approximation with continuity correction (n=20 is ample)
def wilcoxon_sr(d):
    d = d[d != 0]; n = len(d)
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    Wp = r[d > 0].sum(); Wm = r[d < 0].sum()
    W = min(Wp, Wm)
    mu = n*(n+1)/4.0; sig = np.sqrt(n*(n+1)*(2*n+1)/24.0)
    z = (W - mu + 0.5)/sig
    from math import erf, sqrt
    p = 2*0.5*(1+erf(-abs(z)/sqrt(2)))
    return p, n

ALPH = ['0.01','0.1','0.5','1.0','5.0','10.0']
cache = {}
def get(m,a):
    if (m,a) not in cache: cache[(m,a)] = perclient(m,a)
    return cache[(m,a)]

def compare(mA, mB, label):
    print(f'== {label} (A<B => A better) ==', flush=True)
    rows=[]
    for a in ALPH:
        A=get(mA,a); B=get(mB,a)
        if A is None or B is None: print(f'  a={a}: missing'); continue
        p,n=wilcoxon_sr(A-B)
        rows.append((a,float(np.median(A-B)),p,int(np.sum(A<B)),len(A)))
    ps=[r[2] for r in rows]; order=sorted(range(len(ps)),key=lambda i:ps[i]); m=len(ps); adj=[0]*len(ps)
    for rank,i in enumerate(order): adj[i]=min(1.0, ps[i]*(m-rank))
    for r,ph in zip(rows,adj):
        a,md,p,nw,n=r
        print(f'  a={a:<5} median(A-B)={md:+.4f}  A-better={nw}/{n}  p={p:.4f}  p_holm={ph:.4f}', flush=True)

compare('fedgen','fedavg','FedGen-full vs FedAvg (SGD)')
compare('fedgen-adam','fedavg-adam','FedGen-Adam vs FedAvg-Adam')
print('[done]', flush=True)
