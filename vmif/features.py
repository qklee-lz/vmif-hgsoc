import numpy as np
import pandas as pd

from .channels import DECODERS
from .config import SEED, work

N_RP, D_RP = 200, 23


def logit(v):
    v = np.clip(v, 1e-4, 1 - 1e-4)
    return np.log(v / (1 - v))


def extract_path(cohort, slide):
    return work('extract', cohort, f'{slide}.npz')


def decoder_tiles(z, decoder):
    if decoder == 'gigatime512':
        key = {tuple(p): i for i, p in enumerate(z['patches'])}
        rc = z['tiles'][z['index']] // 2
        return z['giga512'][[key[tuple(p)] for p in rc]]
    return z[{'gigatime256': 'giga256', 'flash': 'flash', 'hex': 'hex'}[decoder]]


def summarise(v, decoder):
    hex_ = decoder == 'hex'
    lv = v if hex_ else logit(v)
    return dict(raw=v.mean(0), feat=np.r_[lv.mean(0), lv.std(0)],
                density=v[:, 1:].sum(1).mean() if hex_ else v[:, 3:].sum(1).mean(), dapi=v[:, 0].mean(), n=len(v))


def random_projections():
    return np.random.default_rng(SEED).standard_normal((1024, N_RP * D_RP)) / np.sqrt(1024)


def build(cohort, slides, config):
    R = random_projections()
    rng = np.random.default_rng(SEED)
    rows, uni_mean, musk_mean, rp, sub, off = [], [], [], [], [], [0]
    M2 = np.lib.format.open_memmap(work('features', f'{cohort}_{config}_M2.npy'), mode='w+', dtype=np.float32,
                                   shape=(len(slides), 1024, 1024))
    decoder = 'gigatime512' if config == 'all' else config
    for i, s in enumerate(slides):
        z = np.load(extract_path(cohort, s))
        if config == 'all':
            v, U = z['giga512'], z['uni'].astype(np.float64)
        else:
            v, U = decoder_tiles(z, decoder), z['uni'][z['index']].astype(np.float64)
        rows.append(summarise(v, decoder))
        uni_mean.append(U.mean(0))
        musk_mean.append(np.zeros(1024) if config == 'all' else z['musk'].astype(np.float64).mean(0))
        Z = U @ R
        rp.append(np.c_[Z.mean(0).reshape(N_RP, D_RP), Z.std(0).reshape(N_RP, D_RP)])
        sub.append(U[rng.choice(len(U), min(256, len(U)), replace=False)].astype(np.float32))
        off.append(off[-1] + len(sub[-1]))
        M2[i] = U.T @ U / len(U)
    M2.flush()
    out = work('features', f'{cohort}_{config}.npz')
    np.savez(out, slide_id=np.array(slides), channels=np.array(DECODERS[decoder]),
             raw=np.array([r['raw'] for r in rows], np.float32), feat=np.array([r['feat'] for r in rows], np.float32),
             density=np.array([r['density'] for r in rows]), dapi=np.array([r['dapi'] for r in rows]),
             n=np.array([r['n'] for r in rows]), uni_mean=np.array(uni_mean, np.float32),
             musk_mean=np.array(musk_mean, np.float32), rp=np.array(rp, np.float32),
             sub=np.concatenate(sub), off=np.array(off))
    return out


class Features:
    def __init__(self, cohort, config='all'):
        self.z = dict(np.load(work('features', f'{cohort}_{config}.npz')))
        self.M2 = np.load(work('features', f'{cohort}_{config}_M2.npy'), mmap_mode='r')
        self.slides = list(self.z['slide_id'])
        self.ix = {s: i for i, s in enumerate(self.slides)}
        self.channels = list(self.z['channels'])
        self._basis = {}

    def frame(self, key, slides=None):
        slides = self.slides if slides is None else list(slides)
        X = np.asarray(self.z[key])[[self.ix[s] for s in slides]]
        X = X.reshape(len(slides), -1)
        cols = self.channels if key == 'raw' else None
        return pd.DataFrame(X, index=slides, columns=cols)

    def raw(self, slides=None):
        df = self.frame('raw', slides)
        for k in ('density', 'dapi', 'n'):
            df[k] = self.frame(k, slides).values[:, 0]
        return df

    def basis(self, train_slides, d=23):
        key = (frozenset(train_slides), d)
        if key not in self._basis:
            rows = np.concatenate([self.z['sub'][self.z['off'][self.ix[s]]:self.z['off'][self.ix[s] + 1]] for s in train_slides]).astype(np.float64)
            mu = rows.mean(0)
            P = np.linalg.eigh((rows - mu).T @ (rows - mu))[1][:, ::-1][:, :d]
            proj = (rows - mu) @ P
            self._basis = {} if len(self._basis) > 64 else self._basis
            self._basis[key] = (mu, P, proj.mean(0), proj.std(0))
        return self._basis[key]

    def pca(self, train_slides, slides, d=23):
        mu, P, pm, psd = self.basis(train_slides, d)
        out = np.zeros((len(slides), 2 * d))
        for k, s in enumerate(slides):
            i = self.ix[s]
            m = np.asarray(self.z['uni_mean'][i], np.float64) @ P
            var = np.einsum('ij,ij->j', P, np.asarray(self.M2[i], np.float64) @ P) - m ** 2
            out[k, :d] = (m - mu @ P - pm) / psd
            out[k, d:] = np.sqrt(np.clip(var, 0, None)) / psd
        return pd.DataFrame(out, index=list(slides))


def patient_mean(df, patients, weights=None):
    if weights is None:
        return df.groupby(np.asarray(patients)).mean()
    w = pd.Series(np.asarray(weights, float), index=df.index)
    return df.mul(w, axis=0).groupby(np.asarray(patients)).sum().div(w.groupby(np.asarray(patients)).sum(), axis=0)
