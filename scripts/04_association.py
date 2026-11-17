import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmif import cohorts as C
from vmif.channels import GIGA_PROTEIN, IMMUNE, MARKERS, STROMAL_CONTROLS
from vmif.config import COHORTS, work
from vmif.features import Features, patient_mean
from vmif.stats import boot_spearman, participation_ratio, rank, specificity_test

SOURCES = {'PTRC-HGSOC': ['prot'], 'TCGA-OV': ['cptac', 'arr', 'rna']}
OUT = work('results', 'association')
rows = {k: [] for k in ['specificity', 'markers', 'offtarget', 'decoders', 'compartment', 'adjusted', 'subsets', 'pr']}


def patient_table(cohort, config='all', weight=True, with_dx=False, slide_ids=None):
    F = Features(cohort, config)
    sl = C.slides(cohort, ts_only=not with_dx)
    ids = [s for s in sl.index if s in F.ix and (slide_ids is None or s in slide_ids)]
    df = F.raw(ids)
    P = patient_mean(df.drop(columns='n'), sl.loc[ids, 'patient'], df.n if weight else None)
    P['immune_sum'] = P[[c for c in IMMUNE if c in P.columns]].sum(1)
    return P


def markers_for(P, T, src, min_n=20):
    return [m for m in MARKERS if m in P.columns and f'{src}_{m}' in T and T[f'{src}_{m}'].notna().sum() >= min_n]


def S_row(tag, P, T, src, cov=None, mk=None, **kw):
    pts = P.index.intersection(T.index)
    mk = mk or markers_for(P.loc[pts], T.loc[pts], src)
    X, Y = P.loc[pts, mk].values, T.loc[pts, [f'{src}_{m}' for m in mk]].values.astype(float)
    c = None if cov is None else (cov.loc[pts].values if hasattr(cov, 'loc') else cov)
    r = specificity_test(X, Y, c, **kw)
    return dict(tag, source=src, n=r['n'], k=r['k'], S=r['S'], S_lo=r['S_lo'], S_hi=r['S_hi'], p_perm=r['p_perm']), r, mk


for cohort in COHORTS:
    T = C.truth(cohort)
    for with_dx in ([False, True] if cohort == 'TCGA-OV' else [False]):
        P = patient_table(cohort, with_dx=with_dx)
        for src in SOURCES[cohort]:
            for cov in ['none', 'density', 'dapi']:
                c = None if cov == 'none' else P[cov]
                row, r, mk = S_row(dict(cohort=cohort, with_dx=with_dx, covariate=cov), P, T, src, c)
                rows['specificity'].append(row)
                rows['markers'] += [dict(cohort=cohort, with_dx=with_dx, source=src, covariate=cov, marker=m, rho=r['rho'][i],
                                         lo=r['rho_lo'][i], hi=r['rho_hi'][i], d=r['d'][i]) for i, m in enumerate(mk)]
                if cov != 'dapi' and not with_dx:
                    for agg in ['mean', 'max']:
                        rows['offtarget'].append(S_row(dict(cohort=cohort, covariate=cov, agg=agg), P, T, src, c, agg=agg)[0])

    decs = ['gigatime512', 'gigatime256', 'flash', 'hex']
    for src in SOURCES[cohort][:2]:
        Pd = {d: patient_table(cohort, d, weight=False) for d in decs}
        pts = sorted(set.intersection(*[set(p.index) for p in Pd.values()]) & set(T.index))
        common = [m for m in MARKERS if all(m in Pd[d].columns for d in decs) and f'{src}_{m}' in T and T.loc[pts, f'{src}_{m}'].notna().sum() >= 20]
        for d in decs:
            for cov in ['none', 'density']:
                c = None if cov == 'none' else Pd[d].loc[pts, 'density']
                rows['decoders'].append(S_row(dict(cohort=cohort, decoder=d, covariate=cov), Pd[d].loc[pts], T, src, c, mk=common)[0])

    if cohort == 'PTRC-HGSOC':
        G = C.ptrc_patient_genes('before', genes=STROMAL_CONTROLS + ['ACTA2', 'DES', 'TAGLN'])
        G['ACTA2/DES'] = C.zmean(G, ['ACTA2', 'DES'])
        G['stromal_score'] = C.zmean(G, STROMAL_CONTROLS)
        for d in ['all', 'gigatime256', 'hex']:
            Pc = patient_table(cohort, d)  # patch-count weights, as in the reported stromal-control analysis
            pts = Pc.index.intersection(G.index)
            for v, target in [('aSMA', 'ACTA2/DES'), ('Transgelin', 'TAGLN')]:
                if v not in Pc.columns:
                    continue
                x = Pc.loc[pts, v].values
                for g in ['ACTA2', 'DES', 'TAGLN', 'ACTA2/DES'] + STROMAL_CONTROLS + ['stromal_score']:
                    r, lo, hi = boot_spearman(x, G.loc[pts, g].values)
                    members = {'ACTA2/DES': ('ACTA2', 'DES'), 'TAGLN': ()}[target]
                    role = ('matched' if g == target else 'target_member' if g in members
                            else 'other_target' if g in ('ACTA2', 'DES', 'TAGLN', 'ACTA2/DES')
                            else 'stromal_score' if g == 'stromal_score' else 'stromal_control')
                    rows['compartment'].append(dict(decoder=d, channel=v, protein=g, role=role, rho=r, lo=lo, hi=hi))
                r, lo, hi = boot_spearman(x, G.loc[pts, target].values, G.loc[pts, 'stromal_score'].values)
                rows['compartment'].append(dict(decoder=d, channel=v, protein=target, role='partial_stromal', rho=r, lo=lo, hi=hi))

        P = patient_table(cohort)
        site = T.loc[P.index.intersection(T.index), 'site']
        dummies = pd.get_dummies(site, drop_first=True).astype(float)
        rows['adjusted'].append(S_row(dict(cohort=cohort, adjustment='centre'), P, T, 'prot', dummies)[0])
        rows['adjusted'].append(S_row(dict(cohort=cohort, adjustment='centre+density'), P, T, 'prot',
                                      pd.DataFrame(np.c_[dummies.values, rank(P.loc[dummies.index, 'density'].values)], index=dummies.index))[0])
        for s in sorted(site.unique()):
            ids = site.index[site == s]
            rows['adjusted'].append(S_row(dict(cohort=cohort, adjustment=f'within_{s}'), P.loc[ids], T.loc[ids], 'prot')[0])

        sl = C.slides(cohort)
        blocks = {C.proteome_block(c) for c in C.ptrc_proteome().index}
        same = sl[(sl.patient + '|' + sl.sample_id.astype(str)).isin(blocks)]
        samples = [c for c in C.ptrc_proteome().index if C.proteome_block(c) in set(same.patient + '|' + same.sample_id.astype(str))]
        Gb = C.composites(C.ptrc_patient_genes('before', samples=samples, genes=C.ALL_GENES), 'prot')
        Pb = patient_table(cohort, slide_ids=set(same.index))
        rows['subsets'].append(S_row(dict(cohort=cohort, subset='same_block'), Pb, Gb, 'prot')[0])
        Gi = C.composites(C.ptrc_patient_genes('imputed', genes=C.ALL_GENES), 'prot')
        rows['subsets'].append(S_row(dict(cohort=cohort, subset='imputed'), P, Gi, 'prot')[0])
        mk = markers_for(P, T, 'prot')
        cc = T[[f'prot_{m}' for m in mk]].dropna().index.intersection(P.index)
        rows['subsets'].append(S_row(dict(cohort=cohort, subset='complete_case'), P.loc[cc], T.loc[cc], 'prot')[0])

        Tc = T.loc[cc, [f'prot_{m}' for m in mk]]
        Ti = Gi.reindex(P.index.intersection(T.index))[[f'prot_{m}' for m in mk]].dropna()
        for name, X in [('predicted', P.loc[cc, mk]), ('measured', Tc), ('measured_imputed', Ti), ('predicted_imputed_set', P.loc[Ti.index, mk])]:
            pr, pc1 = participation_ratio(X.values)
            rows['pr'].append(dict(cohort=cohort, features=name, n=len(X), k=X.shape[1], pr=pr, pc1=pc1))
    else:
        P = patient_table(cohort)
        pts = P.index.intersection(T.index[T.purity.notna()])
        for src in SOURCES[cohort]:
            rows['adjusted'].append(S_row(dict(cohort=cohort, adjustment='purity'), P.loc[pts], T.loc[pts], src, T.loc[pts, 'purity'])[0])
        mk = markers_for(P, T, 'cptac')
        cc = T[[f'cptac_{m}' for m in mk]].dropna().index.intersection(P.index)
        for name, X in [('predicted', P.loc[cc, mk]), ('measured', T.loc[cc, [f'cptac_{m}' for m in mk]])]:
            pr, pc1 = participation_ratio(X.values)
            rows['pr'].append(dict(cohort=cohort, features=name, n=len(X), k=X.shape[1], pr=pr, pc1=pc1))

    for weight in [True, False]:
        Pw = patient_table(cohort, weight=weight)
        for src in SOURCES[cohort]:
            rows['subsets'].append(S_row(dict(cohort=cohort, subset='patch_weighted' if weight else 'equal_slides'), Pw, T, src)[0])

    F = Features(cohort)
    sl = C.slides(cohort, ts_only=True)
    ids = [s for s in sl.index if s in F.ix]
    for name, X in [('gigatime_20', patient_mean(F.raw(ids)[GIGA_PROTEIN], sl.loc[ids, 'patient'], F.raw(ids).n)),
                    ('uni_1024', patient_mean(F.frame('uni_mean', ids), sl.loc[ids, 'patient']))]:
        pr, pc1 = participation_ratio(X.values)
        rows['pr'].append(dict(cohort=cohort, features=name, n=len(X), k=X.shape[1], pr=pr, pc1=pc1))
    print(cohort, 'done', flush=True)

for k, v in rows.items():
    pd.DataFrame(v).to_csv(OUT / f'{k}.csv', index=False)
print(pd.DataFrame(rows['specificity']).round(3).to_string())
