import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmif import cohorts as C
from vmif import platinum as PL
from vmif.channels import GIGA, GIGA_MIL
from vmif.config import COHORTS, work
from vmif.features import decoder_tiles, extract_path, logit
from vmif.stats import boot_auc, within_centre_auc

ARMS = ['uni', 'giga', 'hex', 'flash', 'giga256', 'uni+giga', 'uni+hex', 'uni+cov', 'uni+giga+cov', 'uni+hex+cov']
SEED_ARMS = ['uni', 'giga', 'uni+giga', 'uni+hex']
CONTRASTS = [('uni+giga', 'uni'), ('uni+hex', 'uni'), ('uni+giga+cov', 'uni+cov'), ('uni+hex+cov', 'uni+cov'), ('uni+cov', 'uni'),
             ('giga', 'uni'), ('hex', 'uni'), ('flash', 'uni'), ('giga256', 'uni'), ('hex', 'giga')]
OUT = work('results', 'platinum')


def bags(cohort):
    sl = C.slides(cohort).sort_values(['patient']).copy()
    sl = sl[[extract_path(cohort, s).exists() for s in sl.index]]
    pats = sorted(sl.patient.unique())
    pid = {p: i for i, p in enumerate(pats)}
    ci = [GIGA.index(c) for c in GIGA_MIL]
    B, tp = {k: [] for k in ['uni', 'giga', 'hex', 'flash', 'giga256']}, []
    for s in sl.index:
        z = np.load(extract_path(cohort, s))
        B['uni'].append(z['uni'][z['index']])
        B['giga'].append(logit(decoder_tiles(z, 'gigatime512')[:, ci]).astype(np.float32))
        B['giga256'].append(logit(z['giga256'][:, ci]).astype(np.float32))
        B['flash'].append(logit(z['flash'][:, ci]).astype(np.float32))
        B['hex'].append(z['hex'].astype(np.float32))
        tp.append(np.full(len(z['index']), pid[sl.loc[s, 'patient']]))
    pl = sl.groupby('patient').agg(label=('label', 'first'), site=('site', 'first')).loc[pats]
    cov = pd.get_dummies(pl.site).astype(np.float32).values
    nact = None
    if cohort == 'PTRC-HGSOC':
        nact = sl.groupby('patient').neoadj.apply(lambda x: float((x == 'yes').any())).loc[pats].values.astype(np.float32)
        cov = np.c_[cov, nact]
    return {k: np.concatenate(v) for k, v in B.items()}, np.concatenate(tp), np.array(pats), pl.label.values.astype(int), pl.site.values.astype(str), cov, nact


def train(cohort, arms, inits, fixed_epochs, tag):
    X, tp, pats, y, sites, cov, _ = bags(cohort)
    need = sorted({b for a in arms for b in a.replace('+cov', '').split('+')})
    bg = PL.Bags({b: X[b] for b in need}, tp, len(pats), y, cov)
    rows = []
    for scheme, part, fold, te in PL.folds(y, sites):
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(np.unique(y[tr])) < 2:
            continue
        tri, va = PL.inner_split(tr, y)
        for init in inits:
            for arm in arms:
                for pool in ['attn', 'mean']:
                    out, _ = PL.fit_predict(bg, arm, pool, init, tri, va, te, fixed_epochs)
                    rows += [dict(scheme=scheme, partition=part, fold=fold, arm=arm, pool=pool, init=init, patient=pats[i],
                                  label=y[i], site=sites[i], logit=o) for i, o in zip(te, out)]
        print(cohort, scheme, part, fold, flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'oof_{cohort}_{tag}.csv', index=False)


def averaged(O, arm, pool, init=2026):
    r = O[(O.scheme == 'random') & (O.arm == arm) & (O.pool == pool) & (O.init == init)]
    piv = r.pivot_table(index='patient', columns='partition', values='logit')
    meta = r.groupby('patient')[['label', 'site']].first().loc[piv.index]
    return meta.label.values, meta.site.values.astype(str), piv, piv[PL.PARTITIONS].mean(1).values


def centre(O, arm, pool, init=2026):
    r = O[(O.scheme == 'centre') & (O.arm == arm) & (O.pool == pool) & (O.init == init)].sort_values('patient')
    return r.label.values, r.site.values.astype(str), r.logit.values


def summarise(cohort, tag):
    O = pd.read_csv(OUT / f'oof_{cohort}_{tag}.csv')
    summ, con, late, var = [], [], [], []
    for arm in O.arm.unique():
        for pool in ['attn', 'mean']:
            y, s, piv, avg = averaged(O, arm, pool)
            aucs = [roc_auc_score(y, piv[p].values) for p in PL.PARTITIONS]
            pt, lo, hi = boot_auc(y, avg)
            summ.append(dict(cohort=cohort, scheme='random', arm=arm, pool=pool, auc=pt, lo=lo, hi=hi, partition_mean=np.mean(aucs),
                             partition_sd=np.std(aucs, ddof=1), within_centre=within_centre_auc(avg, y, s)[1]))
            y, s, o = centre(O, arm, pool)
            pt, lo, hi = boot_auc(y, o)
            summ.append(dict(cohort=cohort, scheme='centre', arm=arm, pool=pool, auc=pt, lo=lo, hi=hi, within_centre=within_centre_auc(o, y, s)[1]))
            M = np.array([[roc_auc_score(*O[(O.scheme == 'random') & (O.arm == arm) & (O.pool == pool) & (O.init == i) & (O.partition == p)][['label', 'logit']].values.T)
                           for p in PL.PARTITIONS] for i in sorted(O[O.arm == arm].init.unique())])
            if len(M) > 1:
                gm, (I, P) = M.mean(), M.shape
                ms_p = I * ((M.mean(0) - gm) ** 2).sum() / (P - 1)
                ms_i = P * ((M.mean(1) - gm) ** 2).sum() / (I - 1)
                ms_e = (((M - gm) ** 2).sum() - ms_p * (P - 1) - ms_i * (I - 1)) / ((P - 1) * (I - 1))
                var.append(dict(cohort=cohort, arm=arm, pool=pool, auc_mean=gm, auc_min=M.min(), auc_max=M.max(),
                                var_partition=max((ms_p - ms_e) / I, 0), var_init=max((ms_i - ms_e) / P, 0), var_resid=ms_e))
    for a, b in [c for c in CONTRASTS if c[0] in O.arm.unique() and c[1] in O.arm.unique()]:
        for pool in ['attn', 'mean']:
            y, _, _, s1 = averaged(O, a, pool)
            _, _, _, s2 = averaged(O, b, pool)
            con.append(dict(cohort=cohort, scheme='random', contrast=f'{a} - {b}', pool=pool, **dict(zip(['diff', 'lo', 'hi'], boot_auc(y, s1, s2)))))
            y, _, s1 = centre(O, a, pool)
            _, _, s2 = centre(O, b, pool)
            con.append(dict(cohort=cohort, scheme='centre', contrast=f'{a} - {b}', pool=pool, **dict(zip(['diff', 'lo', 'hi'], boot_auc(y, s1, s2)))))
    zs = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
    if (O.init == 0).any():
        for v in ['giga', 'hex', 'flash', 'giga256']:
            for pool in ['attn', 'mean']:
                y, _, _, u = averaged(O, 'uni', pool)
                u0, g = averaged(O, 'uni', pool, 0)[3], averaged(O, v, pool)[3]
                pt, lo, hi = boot_auc(y, (zs(u) + zs(g)) / 2, (zs(u) + zs(u0)) / 2)
                late.append(dict(cohort=cohort, pool=pool, contrast=f'late(uni,{v}) - late(uni,uni)', diff=pt, lo=lo, hi=hi))
    return summ, con, late, var


def probes(cohort):
    X, tp, pats, y, sites, cov, nact = bags(cohort)
    n = np.bincount(tp, minlength=len(pats))[:, None]

    def tile_mean(M):
        Z = np.zeros((len(pats), M.shape[1]))
        np.add.at(Z, tp, M.astype(np.float64))
        return Z / n

    sets = {'uni_mean': tile_mean(X['uni']), 'giga_mean': tile_mean(X['giga'])}
    site_d = pd.get_dummies(pd.Series(sites)).astype(float).values
    if cohort == 'PTRC-HGSOC':
        sets.update({'site': site_d, 'nact': nact[:, None], 'site+nact': np.c_[site_d, nact]})
    else:
        sets['site'] = site_d
    rows = []
    subsets = [('all', np.ones(len(y), bool))] + ([('no_nact', nact == 0)] if nact is not None else [])
    for sub, k in subsets:
        for name, M in sets.items():
            if sub != 'all' and name not in ('uni_mean', 'giga_mean'):
                continue
            for grid in PL.GRIDS:
                R = PL.probe_random(M[k], y[k], grid)
                pt, lo, hi = boot_auc(y[k], np.mean([R[p] for p in PL.PARTITIONS], 0))
                rows.append(dict(cohort=cohort, subset=sub, features=name, grid=grid, scheme='random', auc=pt, lo=lo, hi=hi,
                                 partition_mean=np.mean([roc_auc_score(y[k], R[p]) for p in PL.PARTITIONS])))
                if not name.startswith('site'):
                    o = PL.probe_centre(M[k], y[k], sites[k], grid)
                    pt, lo, hi = boot_auc(y[k], o)
                    rows.append(dict(cohort=cohort, subset=sub, features=name, grid=grid, scheme='centre', auc=pt, lo=lo, hi=hi,
                                     within_centre=within_centre_auc(o, y[k], sites[k])[1]))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['train', 'summarise'], default='train')
    ap.add_argument('--fixed-epochs', type=int, default=0)
    a = ap.parse_args()
    tag = f'fixed{a.fixed_epochs}' if a.fixed_epochs else 'primary'
    if a.stage == 'train':
        for cohort in COHORTS:
            train(cohort, ARMS, [2026], a.fixed_epochs, tag)
            if not a.fixed_epochs:  # extra initialisations only for the primary protocol, as reported
                train(cohort, SEED_ARMS, [0, 1, 2, 3], a.fixed_epochs, tag + '_seeds')
    else:
        res = [[], [], [], []]
        prb = []
        for cohort in COHORTS:
            pd.concat([pd.read_csv(OUT / f'oof_{cohort}_{t}.csv') for t in [tag, tag + '_seeds'] if (OUT / f'oof_{cohort}_{t}.csv').exists()]).to_csv(OUT / f'oof_{cohort}_{tag}_all.csv', index=False)
            for acc, r in zip(res, summarise(cohort, tag + '_all')):
                acc += r
            prb += probes(cohort)
        for name, r in zip(['summary', 'contrasts', 'late_fusion', 'variance'], res):
            pd.DataFrame(r).to_csv(OUT / f'{name}_{tag}.csv', index=False)
        pd.DataFrame(prb).to_csv(OUT / 'probes.csv', index=False)
