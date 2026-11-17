import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

from .channels import SPATIAL_MARKERS

STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class SlideGrid:
    def __init__(self, rc, values, ck):
        idx = {tuple(x): i for i, x in enumerate(rc)}
        nb = [[idx[(r + dr, c + dc)] for dr, dc in STEPS if (r + dr, c + dc) in idx] for r, c in rc]
        rows = np.repeat(np.arange(len(nb)), [len(n) for n in nb])
        cols = np.concatenate([np.array(n, int) for n in nb]) if len(rows) else np.array([], int)
        self.A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nb), len(nb)))
        self.deg = np.asarray(self.A.sum(1)).ravel()
        self.V, self.ck = values.astype(np.float64), ck.astype(np.float64)
        lo, hi = np.quantile(self.ck, [1 / 3, 2 / 3])
        self.invariant = [(spearmanr(v, self.ck)[0], np.log((v[self.ck >= hi].mean() + 1e-4) / (v[self.ck <= lo].mean() + 1e-4)))
                          for v in self.V.T]

    def moran(self, v):
        z = v - v.mean()
        zz, w = z @ z, self.deg.sum()
        return (len(z) / w) * (z @ (self.A @ z)) / zz if w and zz > 0 else np.nan

    def descriptors(self, perm=None):
        V, ck = self.V, self.ck
        if perm is not None:
            V, ck = np.empty_like(V), np.empty_like(ck)
            V[perm], ck[perm] = self.V, self.ck
        has = self.deg > 0
        lag = np.full(len(ck), np.nan)
        lag[has] = (self.A @ ck)[has] / self.deg[has]
        out = []
        for j in range(V.shape[1]):
            v = V[:, j]
            out += [*self.invariant[j], np.corrcoef(v[has], lag[has])[0, 1] if has.sum() > 5 else np.nan, self.moran(v)]
        return np.array(out + [self.moran(ck)], np.float32)


def grid_from_patches(patches, giga, channels):
    ci = [channels.index(m) for m in SPATIAL_MARKERS]
    return SlideGrid(patches, giga[:, ci], giga[:, channels.index('CK')])


def shuffle(n, seed):
    return np.random.default_rng(1000 + seed).permutation(n)
