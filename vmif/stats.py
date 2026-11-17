import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def rank(x):
    return rankdata(x) / len(x)


def _resid(r, design):
    return r - design @ np.linalg.lstsq(design, r, rcond=None)[0]


def spearman(a, b, cov=None, min_n=10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if cov is not None:
        cov = np.asarray(cov, float)
        ok &= ~np.isnan(cov.reshape(len(cov), -1)).any(1)
    if ok.sum() < min_n:
        return np.nan
    ra, rb = rank(a[ok]), rank(b[ok])
    if cov is not None:
        c = cov[ok]
        design = np.c_[np.ones(ok.sum()), rank(c) if c.ndim == 1 else c]
        ra, rb = _resid(ra, design), _resid(rb, design)
    return np.corrcoef(ra, rb)[0, 1]


def rho_matrix(X, Y, cov=None):
    return np.array([[spearman(X[:, i], Y[:, j], cov) for j in range(Y.shape[1])] for i in range(X.shape[1])])


OFFTARGET = {'median': np.nanmedian, 'mean': np.nanmean, 'max': np.nanmax}


def specificity(R, agg='median'):
    d = np.array([R[i, i] - OFFTARGET[agg](np.delete(R[i], i)) for i in range(R.shape[0])])
    return np.nanmean(d), d


def specificity_test(X, Y, cov=None, n_perm=10000, n_boot=1000, seed=2026, agg='median'):
    rng = np.random.default_rng(seed)
    R = rho_matrix(X, Y, cov)
    S, d = specificity(R, agg)
    k, n = R.shape[0], len(X)
    null = np.array([specificity(R[:, rng.permutation(k)], agg)[0] for _ in range(n_perm)])
    boots, diag = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        Rb = rho_matrix(X[ix], Y[ix], None if cov is None else np.asarray(cov)[ix])
        boots.append(specificity(Rb, agg)[0])
        diag.append(np.diag(Rb))
    diag = np.array(diag)
    return dict(S=S, S_lo=np.nanpercentile(boots, 2.5), S_hi=np.nanpercentile(boots, 97.5),
                p_perm=(1 + np.sum(null >= S)) / (1 + n_perm), n=n, k=k,
                rho=np.diag(R), rho_lo=np.nanpercentile(diag, 2.5, 0), rho_hi=np.nanpercentile(diag, 97.5, 0), d=d, R=R)


def boot_spearman(x, y, cov=None, n_boot=1000, seed=2026):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    bs = [spearman(x[ix], y[ix], None if cov is None else np.asarray(cov)[ix])
          for ix in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    return spearman(x, y, cov), np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def participation_ratio(X):
    X = np.asarray(X, float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    lam = np.linalg.svd(X, full_matrices=False)[1] ** 2
    lam = lam / lam.sum()
    return 1 / np.sum(lam ** 2), lam[0]


def boot_auc(y, s, s2=None, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(s) if s2 is None else ~(np.isnan(s) | np.isnan(s2))
    y, s = y[ok], s[ok]
    s2 = None if s2 is None else s2[ok]
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        a = roc_auc_score(y[i], s[i])
        vals.append(a if s2 is None else a - roc_auc_score(y[i], s2[i]))
    point = roc_auc_score(y, s) - (0 if s2 is None else roc_auc_score(y, s2))
    return point, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def within_centre_auc(score, y, sites):
    res = {}
    for s in sorted(set(sites)):
        m = (sites == s) & ~np.isnan(score)
        if len(np.unique(y[m])) == 2:
            res[s] = (roc_auc_score(y[m], score[m]), int(m.sum()))
    if not res:
        return res, np.nan
    n = np.array([v[1] for v in res.values()])
    return res, float(np.sum([a * k for a, k in res.values()]) / n.sum())


def boot_median(v, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    b = np.median(v[rng.integers(0, len(v), (n_boot, len(v)))], 1)
    return np.median(v), np.percentile(b, 2.5), np.percentile(b, 97.5)
