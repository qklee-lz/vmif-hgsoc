import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
ALPHAS = 10.0 ** np.arange(-2, 6)
CS = 10.0 ** np.arange(-4, 3)


class InFold:
    def __init__(self, dims):
        self.dims = list(dims)

    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        if len(self.dims) > 1:
            Z = Z * np.concatenate([np.full(d, np.sqrt(1024 / d)) for d in self.dims])
        return Z


def ridge(Xtr, ytr, Xte, seed=0, k=5):
    cv = KFold(min(k, len(ytr)), shuffle=True, random_state=seed)
    return RidgeCV(alphas=ALPHAS, cv=cv).fit(Xtr, ytr).predict(Xte)


def logistic(Xtr, ytr, Xte, seed=0):
    gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                      {'logisticregression__C': CS}, cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring='roc_auc')
    return gs.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def cox(Xtr, ytr, Xte, seed=0):
    t, e = ytr[:, 0], ytr[:, 1]
    pca = PCA(min(32, Xtr.shape[1], len(t) - 5), random_state=seed).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    sd = Ztr.std(0) + 1e-8
    Ztr, Zte = Ztr / sd, Zte / sd

    def fit(Z, tt, ee, pen):
        return CoxPHFitter(penalizer=pen).fit(pd.DataFrame(Z).assign(T=tt, E=ee), 'T', 'E')

    def cv_c(pen):
        cs = []
        for a, b in KFold(3, shuffle=True, random_state=seed).split(Ztr):
            try:
                m = fit(Ztr[a], t[a], e[a], pen)
                cs.append(concordance_index(t[b], -m.predict_partial_hazard(pd.DataFrame(Ztr[b])).values, e[b]))
            except Exception:
                cs.append(0.5)
        return np.mean(cs)

    best = max([0.1, 1.0, 10.0], key=cv_c)
    return -fit(Ztr, t, e, best).predict_partial_hazard(pd.DataFrame(Zte)).values


FIT = {'cont': ridge, 'bin': logistic, 'surv': cox}


def score(kind, y, p):
    if kind == 'cont':
        return spearmanr(p, y)[0]
    if kind == 'bin':
        return roc_auc_score(y, p)
    return concordance_index(y[:, 0], p, y[:, 1])


def learning_curve_rep(rep, X, y, zero, grid=(10, 20, 40, 80, 'full')):
    r = np.random.default_rng(rep)
    perm = r.permutation(len(y))
    nte = int(round(0.3 * len(y)))
    te, pool = perm[:nte], perm[nte:]
    z0 = spearmanr(zero[te], y[te])[0]
    du = X['uni'].shape[1]
    sets = {'uni': X['uni'], 'vmif': X['vmif'], 'both': np.c_[X['uni'], X['vmif']]}
    out = []
    for n in grid:
        nn = len(pool) if n == 'full' else n
        if nn > len(pool):
            continue
        tr = r.choice(pool, nn, replace=False)
        row = dict(rep=rep, n=nn, n_test=nte, zero_shot=z0)
        for name, M in sets.items():
            T = InFold([du, M.shape[1] - du] if name == 'both' else [M.shape[1]]).fit(M[tr])
            row[name] = spearmanr(ridge(T.transform(M[tr]), y[tr], T.transform(M[te])), y[te])[0]
        out.append(row)
    return out


def nstar(curve):
    g = curve.groupby('n').uni.median().sort_index()
    z = curve.drop_duplicates('rep').zero_shot.median()
    ns, us = g.index.values.astype(float), g.values
    if us[0] >= z:
        return ns[0], 'left'
    for k in range(1, len(ns)):
        if us[k] >= z:
            f = (z - us[k - 1]) / (us[k] - us[k - 1])
            return float(np.exp(np.log(ns[k - 1]) + f * (np.log(ns[k]) - np.log(ns[k - 1])))), ''
    return ns[-1], 'right'


def nstar_ci(curve, n_boot=1000, seed=0):
    # each resampled repeat occurrence gets a new id, so it counts once per draw in BOTH medians
    rng = np.random.default_rng(seed)
    groups = [g for _, g in curve.groupby('rep')]
    bs = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(groups), len(groups))
        bs.append(nstar(pd.concat([groups[i].assign(rep=k) for k, i in enumerate(draw)]))[0])
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def panel_seed(seed, kind, y, blocks, sets, pca_fn=None, ntrains=('full', 40)):
    n = len(y)
    strat = (y[:, 1] if kind == 'surv' else (y if kind == 'bin' else y > np.median(y))).astype(int)
    rng = np.random.default_rng(1000 + seed)
    out = []
    for ntr in ntrains:
        oof = {s: np.full(n, np.nan) for s in sets}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), strat):
            if ntr != 'full':
                tr = rng.choice(tr, min(ntr, len(tr)), replace=False)
            for s in sets:
                parts = s.split('+')
                mats = [pca_fn(tr, te) if p == 'pca' and pca_fn else (blocks[p][tr], blocks[p][te]) for p in parts]
                Xtr, Xte = np.concatenate([m[0] for m in mats], 1), np.concatenate([m[1] for m in mats], 1)
                T = InFold([m[0].shape[1] for m in mats]).fit(Xtr)
                try:
                    oof[s][te] = FIT[kind](T.transform(Xtr), y[tr], T.transform(Xte), seed)
                except Exception:
                    pass
        for s in sets:
            ok = ~np.isnan(oof[s])
            out.append(dict(ntrain=str(ntr), features=s, seed=seed, n=n,
                            score=score(kind, y[ok], oof[s][ok]) if ok.sum() > 10 else np.nan))
    return out


def projection_score(X, kind, y, seeds=range(5), ntrains=('full', 40)):
    strat = (y if kind == 'bin' else y > np.median(y)).astype(int)
    out = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        for ntr in ntrains:
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
                if ntr != 'full':
                    tr = rng.choice(tr, min(40, len(tr)), replace=False)
                T = InFold([X.shape[1]]).fit(X[tr])
                if kind == 'cont':
                    oof[te] = RidgeCV(alphas=ALPHAS).fit(T.transform(X[tr]), y[tr]).predict(T.transform(X[te]))
                else:
                    m = LogisticRegressionCV(Cs=CS, cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring='roc_auc',
                                             max_iter=3000).fit(T.transform(X[tr]), y[tr])
                    oof[te] = m.predict_proba(T.transform(X[te]))[:, 1]
            out.append(dict(seed=seed, ntrain=str(ntr), score=score(kind, y, oof)))
    return out


def grouped_logistic(X, y, groups, seed, multiclass=False):
    oof = np.zeros((len(y), len(np.unique(y))) if multiclass else len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), {'logisticregression__C': CS},
                          cv=GroupKFold(3), scoring='roc_auc_ovr' if multiclass else 'roc_auc')
        p = gs.fit(X[tr], y[tr], groups=groups[tr]).predict_proba(X[te])
        oof[te] = p if multiclass else p[:, 1]
    return oof


def auc(y, p, multiclass=False):
    return roc_auc_score(y, p, multi_class='ovr', average='macro') if multiclass else roc_auc_score(y, p)
