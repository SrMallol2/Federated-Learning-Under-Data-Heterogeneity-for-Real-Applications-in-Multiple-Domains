"""export_perround.py  —  per-round trajectory data for the UC1 communication frontier.

Run once from the same folder as reframe_results.py:  python export_perround.py
Writes uc1_perround.csv next to it.

NOTE: per-round PER-CLIENT AUC was never logged (see reframe_results.py docstring),
so the only per-round signal available is the POOLED val/test from history[].
cum_mb is the logged cumulative communication (pre-correction; differs <1% from the
corrected final_mb used in the endpoint figures).
"""
import glob, os, json, csv
import reframe_results as R

out = []
for root in R.ROOTS.values():
    for path in glob.glob(f'{root}/**/*.json', recursive=True):
        parts = path.replace(os.sep, '/').split('/')
        dc, alpha = parts[-4], float(parts[-3].split('_')[1])
        variant = os.path.splitext(parts[-1])[0]
        seed = parts[-2]                      # seed sub-dir (label only; we average over it)
        d = json.load(open(path))
        hist = d.get('history', []); cmb = d.get('cumul_mb', [])
        for t, h in enumerate(hist):
            if t >= len(cmb) or 'val' not in h or 'test' not in h:
                continue                       # keep every column strictly numeric
            out.append(dict(data_case=dc, variant=variant, sharing=R.kind(variant),
                            alpha=alpha, seed=seed, round=t,
                            cum_mb=cmb[t], pooled_val=h['val'], pooled_test=h['test']))

dst = os.path.join(R.BASE, 'uc1_perround.csv')
with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print('wrote', dst, '|', len(out), 'rows')