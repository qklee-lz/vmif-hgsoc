import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmif import cohorts as C
from vmif import supervised as S
from vmif.channels import MARKERS
from vmif.config import COHORTS, N_JOBS, work
from vmif.features import Features, patient_mean

OUT = work('results', 'supervised')
PRIMARY = {'PTRC-HGSOC': ['prot'], 'TCGA-OV': ['cptac', 'arr']}
PANEL_MARKERS = ['CK', 'CD8', 'CD3', 'CD4', 'CD14', 'CD16', 'CD11c', 'CD138', 'CD34', 'Ki67', 'Tryptase', 'aSMA', 'Transgelin', 'Caspase3-D', 'CD68']


def patient_blocks(cohort, config, ts_only, uni_key='uni_mean'):
    F = Features(cohort, config)
    sl = C.slides(cohort, ts_only=ts_only)
    ids = [s for s in sl.index if s in F.ix]
    pat = sl.loc[ids, 'patient']
    raw = F.raw(ids)
    uni = F.frame(uni_key, ids)
    return F, sl.loc[ids], dict(uni=patient_mean(uni, pat), vmif=patient_mean(F.frame('feat', ids), pat),
                                raw=patient_mean(raw.drop(columns='n'), pat, raw.n),
                                density=patient_mean(raw[['density']], pat), logn=patient_mean(np.log(raw[['n']]), pat))


def learning_curves(cohort, config='all', uni_key='uni_mean', reps=200):
    T = C.truth(cohort)
    _, _, B = patient_blocks(cohort, config, ts_only=True, uni_key=uni_key)
    targets = [(src, m) for src in PRIMARY[cohort] for m in MARKERS if m in B['raw'].columns and f'{src}_{m}' in T]
    if cohort == 'TCGA-OV':
        targets.append(('cibersort', 'CD8'))
    rows = []
    for src, m in targets:
        y = T[('cibersort_CD8_abs' if src == 'cibersort' else f'{src}_{m}')].reindex(B['uni'].index)
        ids = y.index[y.notna()]
        if len(ids) < 30:
            continue
        X = {'uni': B['uni'].loc[ids].values.astype(float), 'vmif': B['vmif'].loc[ids].values.astype(float)}
        R = Parallel(n_jobs=N_JOBS)(delayed(S.learning_curve_rep)(r, X, y.loc[ids].values.astype(float), B['raw'].loc[ids, m].values)
                                    for r in range(reps))
        rows += [dict(cohort=cohort, config=config, reference=uni_key, target=f'{src}_{m}', source=src, marker=m, **x) for rr in R for x in rr]
    curves = pd.DataFrame(rows)
    ns = []
    for tgt, d in curves.groupby('target'):
        n, cens = S.nstar(d)
        lo, hi = S.nstar_ci(d)
        ns.append(dict(cohort=cohort, config=config, reference=uni_key, target=tgt, nstar=n, censor=cens, lo=lo, hi=hi,
                       zero_shot=d.drop_duplicates('rep').zero_shot.median(), n_full=int(d.n.max())))
    return curves, pd.DataFrame(ns)


def tasks(cohort, T):
    if cohort == 'PTRC-HGSOC':
        out = [(f'prot_{m}', 'cont', f'prot_{m}') for m in PANEL_MARKERS if T[f'prot_{m}'].notna().sum() > 50]
        return out + [('prot_PTPRC', 'cont', 'prot_PTPRC'), ('platinum', 'bin', 'label')]
    out = [(f'cptac_{m}', 'cont', f'cptac_{m}') for m in PANEL_MARKERS if T[f'cptac_{m}'].notna().sum() > 50]
    out += [(c, 'cont', c) for c in ['cibersort_CD8_abs', 'arr_Ki67', 'arr_Transgelin', 'rppa_CD20', 'LF', 'purity', 'cptac_PTPRC', 'HRD']]
    out += [(c, 'bin', c) for c in ['immunoreactive', 'mesenchymal', 'BRCA']] + [('platinum', 'bin', 'label')]
    return out + [('OS', 'surv', ('OS_MONTHS', 'OS_EVENT')), ('PFS', 'surv', ('PFS_MONTHS', 'PFS_EVENT'))]


def target(T, ids, kind, col):
    if kind == 'surv':
        y = T.loc[ids, list(col)]
        ok = y.notna().all(1) & (y.iloc[:, 0] > 0)
    else:
        y = T.loc[ids, col]
        ok = y.notna()
    return ids[ok.values], y[ok].values.astype(float)


def panel(cohort, seeds=range(20)):
    T = C.truth(cohort)
    F, sl, B = patient_blocks(cohort, 'all', ts_only=False)
    T = T.loc[T.index.intersection(B['uni'].index)]  # select tasks among eligible feature patients, as reported
    slides_of = sl.groupby('patient').apply(lambda d: list(d.index)).to_dict()
    rp = Features(cohort).frame('rp', list(sl.index)).values.reshape(len(sl), 200, 46)
    rpP = pd.DataFrame(rp[:, 0], index=sl.index).groupby(sl.patient.values).mean()
    sets = ['uni', 'uni+vmif', 'uni+rp0', 'uni+pca', 'vmif', 'rp0', 'pca', 'density']
    rows = []
    for name, kind, col in tasks(cohort, T):
        ids, y = target(T, B['uni'].index.intersection(T.index), kind, col)
        blocks = {'uni': B['uni'].loc[ids].values, 'vmif': B['vmif'].loc[ids].values, 'rp0': rpP.loc[ids].values,
                  'density': B['density'].loc[ids].values}

        def pca_fn(tr, te, ids=ids):
            train = [s for p in ids[tr] for s in slides_of[p]]
            need = [s for p in ids for s in slides_of[p]]
            Pm = F.pca(train, need).groupby([sl.loc[s, 'patient'] for s in need]).mean()
            return Pm.loc[ids[tr]].values, Pm.loc[ids[te]].values

        R = Parallel(n_jobs=N_JOBS)(delayed(S.panel_seed)(s, kind, y, blocks, sets, pca_fn) for s in seeds)
        rows += [dict(cohort=cohort, task=name, kind=kind, **x) for r in R for x in r]
        print(cohort, name, flush=True)
    return pd.DataFrame(rows)


def projections(cohort):
    T = C.truth(cohort)
    F, sl, B = patient_blocks(cohort, 'all', ts_only=False)
    T = T.loc[T.index.intersection(B['uni'].index)]
    rp = pd.DataFrame(F.frame('rp', list(sl.index)).values, index=sl.index).groupby(sl.patient.values).mean()
    rows = []
    for name, kind, col in [t for t in tasks(cohort, T) if t[1] != 'surv']:
        ids, y = target(T, B['uni'].index.intersection(T.index), kind, col)
        sets = {'vmif': B['vmif'].loc[ids].values}
        sets.update({f'rp{k:03d}': rp.loc[ids].values.reshape(len(ids), 200, 46)[:, k] for k in range(200)})
        R = Parallel(n_jobs=N_JOBS)(delayed(S.projection_score)(X, kind, y) for X in sets.values())
        rows += [dict(cohort=cohort, task=name, features=f, **x) for f, r in zip(sets, R) for x in r]
    d = pd.DataFrame(rows)
    m = d.groupby(['task', 'ntrain', 'features']).score.mean().unstack('features')
    rpc = [c for c in m.columns if c.startswith('rp')]
    return d, pd.DataFrame({'vmif': m.vmif, 'rp_median': m[rpc].median(1), 'vmif_percentile': m[rpc].lt(m.vmif, axis=0).mean(1) * 100})


def provenance():
    rows = []
    for cohort in COHORTS:
        F = Features(cohort)
        sl = C.slides(cohort, tumour_only=False)
        sl = sl.loc[[s for s in sl.index if s in F.ix]]
        X = {'vmif': F.frame('feat', sl.index).values, 'uni': F.frame('uni_mean', sl.index).values,
             'density': F.frame('density', sl.index).values, 'logn': np.log(F.frame('n', sl.index).values)}
        if cohort == 'PTRC-HGSOC':
            jobs = [('centre', pd.Categorical(sl.site).codes, sl.index, True), ('metastatic', (sl.tissue == 'Metastatic').astype(int).values, sl.index, False)]
        else:
            t = sl.stype == '01'
            n = sl.stype.isin(['01', '11'])
            jobs = [('ffpe_vs_frozen', (sl[t].prep == 'DX').astype(int).values, sl.index[t], False),
                    ('normal_vs_tumour', (sl[n].stype == '11').astype(int).values, sl.index[n], False)]
        for name, y, ids, multi in jobs:
            pos = [list(sl.index).index(s) for s in ids]
            for f, M in X.items():
                P = Parallel(n_jobs=N_JOBS)(delayed(S.grouped_logistic)(M[pos], y, sl.loc[ids, 'patient'].values, s, multi) for s in range(20))
                rows += [dict(cohort=cohort, task=name, features=f, seed=s, auc=S.auc(y, p, multi)) for s, p in enumerate(P)]
    return pd.DataFrame(rows)


if __name__ == '__main__':
    allc, alln = [], []
    for cohort in COHORTS:
        for config, ref in [('all', 'uni_mean'), ('gigatime512', 'uni_mean'), ('gigatime256', 'uni_mean'), ('flash', 'uni_mean'),
                            ('hex', 'uni_mean'), ('hex', 'musk_mean')]:
            c, n = learning_curves(cohort, config, ref)
            allc.append(c)
            alln.append(n)
        panel(cohort).to_csv(OUT / f'panel_{cohort}.csv', index=False)
        d, summ = projections(cohort)
        d.to_csv(OUT / f'projections_{cohort}.csv', index=False)
        summ.to_csv(OUT / f'projections_summary_{cohort}.csv')
    pd.concat(allc).to_csv(OUT / 'learning_curves.csv', index=False)
    pd.concat(alln).to_csv(OUT / 'nstar.csv', index=False)
    provenance().to_csv(OUT / 'provenance.csv', index=False)
