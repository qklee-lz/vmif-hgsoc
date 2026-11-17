import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmif import cohorts as C
from vmif import platinum as PL
from vmif.channels import GIGA
from vmif.config import COHORTS, N_JOBS, work
from vmif.features import extract_path
from vmif.spatial import grid_from_patches, shuffle
from vmif.stats import within_centre_auc

ap = argparse.ArgumentParser()
ap.add_argument('--n-null', type=int, default=1000)
a = ap.parse_args()
OUT = work('results', 'spatial')

for cohort in COHORTS:
    sl = C.slides(cohort)
    sl = sl[[extract_path(cohort, s).exists() for s in sl.index]]
    grids = {}
    for s in sl.index:
        z = np.load(extract_path(cohort, s))
        grids[s] = grid_from_patches(z['patches'], z['giga512'], GIGA)
    pats = sorted(sl.patient.unique())
    meta = sl.groupby('patient')[['label', 'site']].first().loc[pats]
    y, sites = meta.label.values.astype(int), meta.site.values.astype(str)

    def matrix(seed):
        F = np.stack([g.descriptors(None if seed is None else shuffle(len(g.ck), seed)) for g in grids.values()])
        return pd.DataFrame(F, index=list(grids)).groupby(sl.loc[list(grids), 'patient'].values).mean().loc[pats].values

    def evaluate(seed):
        X = matrix(seed)
        rows = []
        for grid in PL.GRIDS:
            R = PL.probe_random(X, y, grid)
            o = PL.probe_centre(X, y, sites, grid)
            ok = ~np.isnan(o)
            rows.append(dict(cohort=cohort, version='original' if seed is None else f'null{seed}', grid=grid,
                             partition_auc=np.mean([roc_auc_score(y, R[p]) for p in PL.PARTITIONS]),
                             centre_auc=roc_auc_score(y[ok], o[ok]), within_centre=within_centre_auc(o, y, sites)[1]))
        return rows

    rows = evaluate(None) + [r for rr in Parallel(n_jobs=N_JOBS)(delayed(evaluate)(s) for s in range(a.n_null)) for r in rr]
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f'rows_{cohort}.csv', index=False)
    summ = []
    for grid, g in d.groupby('grid'):
        o, null = g[g.version == 'original'].iloc[0], g[g.version != 'original']
        for m in ['partition_auc', 'centre_auc', 'within_centre']:
            summ.append(dict(cohort=cohort, grid=grid, metric=m, original=o[m], null_median=null[m].median(),
                             p=(1 + (null[m] >= o[m]).sum()) / (1 + len(null))))
    pd.DataFrame(summ).to_csv(OUT / f'summary_{cohort}.csv', index=False)
    print(pd.DataFrame(summ).round(3).to_string(), flush=True)
